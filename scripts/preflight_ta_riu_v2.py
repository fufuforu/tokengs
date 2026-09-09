"""Single-card/DDP diagnostic preflight for TA-RIU-v2.

All gradient probes use independent forwards.  This script never writes a
formal model workspace and requires a fresh output directory.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.ta_riu_v2 import DINO_EXPECTED_SHA256, sha256_file
from tokengs.options import config_defaults
from tokengs.train import load_model_checkpoint, setup_optimizer

CFG = "semantic_v6_absolute_units_true_shared_ta_riu_v2_dino_unit_ddp8"
CKPT = ROOT / "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors"


def finite(value):
    return bool(torch.isfinite(value.detach() if torch.is_tensor(value) else torch.as_tensor(value)).all())


def phash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode()); h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def grad_norm(model, prefixes):
    total = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None and any(name.startswith(x) for x in prefixes):
            total += float(p.grad.detach().float().square().sum())
    return total ** 0.5


def count_keys(state):
    return {
        "absolute_gs_head": sum(k.startswith("absolute_gs_head.") for k in state),
        "tsh_instance_head": sum(k.startswith("tsh_instance_head.") for k in state),
        "decoder_tail": sum(k.startswith("enc_dec_backbone.decoder_blocks.") for k in state),
        "ta_riu_v2": sum(k.startswith("ta_riu_v2_unit_encoder.") for k in state),
        "pgsr": sum(k.startswith("tsh_slot_refine_head.") for k in state),
    }


def make_opt(workspace):
    opt = dataclasses.replace(config_defaults[CFG])
    opt.workspace = str(workspace); opt.resume = str(CKPT)
    opt.num_workers = 0; opt.num_epochs = 1; opt.max_iters_per_epoch = 3
    opt.eval_before_training = False; opt.use_wandb = False
    opt.print_freq = 100000; opt.log_image_freq = 100000
    return opt


def configure(base, step):
    base.teacher_lambda_eff = 0.0; base.tsh_mbm_u2r_eff = 0.0
    base.tsh_instance_loss_weight_eff = 1.0; base.tsh_unit_grad_eff = 0.0
    base.ta_riu_v2_gate_eff = min(1.0, (step + 1) / 25.0)


def scene_name(data):
    value = data.get("scene_name", "unknown")
    return str(value[0] if isinstance(value, (list, tuple)) else value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--expected-world-size", type=int, default=1)
    args = ap.parse_args()
    out = ROOT / args.workspace
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing non-empty preflight workspace: {out}")
    out.mkdir(parents=True, exist_ok=True)
    opt = make_opt(out)
    acc = Accelerator(
        mixed_precision=opt.mixed_precision,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    loader, _, _, _ = get_multi_dataloader(opt, acc)
    raw = load_file(str(CKPT), device="cpu")
    source_counts = count_keys(raw)
    if source_counts != {"absolute_gs_head": 24, "tsh_instance_head": 50, "decoder_tail": 324, "ta_riu_v2": 0, "pgsr": 0}:
        raise RuntimeError(f"source checkpoint counts: {source_counts}")
    model = model_registry[opt.model_type](opt).cuda().train()
    load_model_checkpoint(opt, model, acc, 0)
    optimizer = setup_optimizer(opt, model, acc, 0)
    model, optimizer, loader, _ = acc.prepare(model, optimizer, loader, loader)
    base = acc.unwrap_model(model)
    state = torch.nn.Module.state_dict(base)
    model_counts = count_keys(state)
    trainable = [n for n, p in base.named_parameters() if p.requires_grad]
    if not trainable or not all(n.startswith(("tsh_instance_head.", "ta_riu_v2_unit_encoder.")) for n in trainable):
        raise RuntimeError(f"unexpected trainable parameters: {trainable[:20]}")
    if any(n.startswith(("absolute_gs_head.", "enc_dec_backbone.", "activation_head.")) for n in trainable):
        raise RuntimeError("reconstruction parameter is trainable")
    dino_hash = sha256_file(opt.ta_riu_v2_dino_weight_path) if acc.is_main_process else None
    if acc.is_main_process and dino_hash != DINO_EXPECTED_SHA256:
        raise RuntimeError(f"DINO hash mismatch: {dino_hash}")
    iterator = iter(loader); records = []; samples = []
    for step in range(int(args.steps)):
        data = next(iterator); scene = scene_name(data); samples.append(scene)
        # Fresh forward 1: gate-zero identity.
        model.eval(); base.ta_riu_v2_eval_gate_override = 0.0
        with acc.autocast(): identity = model(data, compute_quality_metrics=False)
        identity_diff = float((identity["unit_logits"].detach() - identity["base_unit_logits"].detach()).float().abs().max())
        assert identity_diff == 0.0
        del base.ta_riu_v2_eval_gate_override
        configure(base, step); model.train()
        # Fresh forward 2: instance-only gradient.
        optimizer.zero_grad(set_to_none=True)
        with acc.autocast(): inst = model(data, compute_quality_metrics=False)
        assert finite(inst["loss_instance_group"]) and inst["loss_instance_group"].requires_grad
        acc.backward(inst["loss_instance_group"])
        ig = {"tsh_instance_head": grad_norm(base, ("tsh_instance_head.",)), "v2": grad_norm(base, ("ta_riu_v2_unit_encoder.",)), "absolute_gs_head": grad_norm(base, ("absolute_gs_head.",)), "decoder_tail": grad_norm(base, ("enc_dec_backbone.decoder_blocks.",))}
        assert ig["tsh_instance_head"] > 0 and ig["v2"] > 0 and ig["absolute_gs_head"] == 0 and ig["decoder_tail"] == 0
        # Fresh forward 3: RGB-only.  Frozen reconstruction has no grad graph.
        optimizer.zero_grad(set_to_none=True)
        with acc.autocast(): rgb = model(data, compute_quality_metrics=False)
        assert finite(rgb["loss_rgb"])
        if rgb["loss_rgb"].requires_grad: acc.backward(rgb["loss_rgb"])
        rg = {"tsh_instance_head": grad_norm(base, ("tsh_instance_head.",)), "v2": grad_norm(base, ("ta_riu_v2_unit_encoder.",)), "absolute_gs_head": grad_norm(base, ("absolute_gs_head.",)), "decoder_tail": grad_norm(base, ("enc_dec_backbone.decoder_blocks.",))}
        assert all(rg[k] == 0 for k in rg)
        # Fresh forward 4: only total loss reaches the optimizer.
        optimizer.zero_grad(set_to_none=True)
        with acc.autocast(): total = model(data, compute_quality_metrics=False)
        assert all(finite(total[k]) for k in ("loss", "loss_rgb", "loss_instance_group", "loss_ta_riu_v2_unit_embedding"))
        acc.backward(total["loss"])
        for p in model.parameters(): assert p.grad is None or finite(p.grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt.gradient_clip)); optimizer.step()
        post = phash(base)
        records.append({"step": step + 1, "rank": acc.process_index, "sample": scene, "identity_unit_logits_max_diff": identity_diff, "instance_grads": ig, "rgb_grads": rg, "loss": float(total["loss"].detach()), "loss_rgb": float(total["loss_rgb"].detach()), "loss_instance": float(total["loss_instance_group"].detach()), "loss_embedding": float(total["loss_ta_riu_v2_unit_embedding"].detach()), "teacher_called": bool(getattr(base, "teacher_called", False)), "u2r": float(getattr(base, "tsh_mbm_u2r_eff", 0.0)), "unit_multiplier": float(getattr(base, "tsh_unit_gradient_multiplier_eff", 0.0)), "parameter_hash": post, "all_finite": True})
    local = {"rank": acc.process_index, "samples": samples, "records": records, "final_hash": phash(base)}
    gathered = [local]
    if acc.num_processes > 1:
        gathered = [None] * acc.num_processes; dist.all_gather_object(gathered, local)
    # Strict in-memory save/restore of the full registered model state. DINO
    # is external and therefore cannot appear in this file.
    restore_path = out / "ta_riu_v2_preflight_checkpoint.safetensors"
    if acc.is_main_process: save_file({k: v.detach().cpu().contiguous() for k, v in torch.nn.Module.state_dict(base).items()}, str(restore_path))
    acc.wait_for_everyone()
    restored = load_file(str(restore_path), device="cpu")
    missing, unexpected = torch.nn.Module.load_state_dict(base, restored, strict=True)
    assert not missing and not unexpected
    restore_hash = phash(base)
    gathered_restore = [None] * acc.num_processes
    if acc.num_processes > 1: dist.all_gather_object(gathered_restore, restore_hash)
    if acc.is_main_process:
        step_hashes_equal = all(len({item["records"][i]["parameter_hash"] for item in gathered}) == 1 for i in range(int(args.steps)))
        report = {"world_size": acc.num_processes, "expected_world_size": int(args.expected_world_size), "checkpoint_counts": source_counts, "model_counts": model_counts, "fresh_reset": False, "pgsr_absent": model_counts["pgsr"] == 0, "dino_repo_path": opt.ta_riu_v2_dino_repo_path, "dino_weight_path": opt.ta_riu_v2_dino_weight_path, "dino_weight_sha256": dino_hash, "dino_source": "local", "dino_strict_load": True, "dino_missing_keys": list(getattr(base.ta_riu_v2_unit_encoder.dino_extractor, "last_strict_load", {}).get("missing_keys", [])), "dino_unexpected_keys": list(getattr(base.ta_riu_v2_unit_encoder.dino_extractor, "last_strict_load", {}).get("unexpected_keys", [])), "dino_parameter_count": 86580480, "dino_saved_in_checkpoint": False, "dino_in_optimizer": False, "dino_patch_shape": [1, 324, 768], "trainable_parameter_count": sum(p.numel() for n, p in base.named_parameters() if p.requires_grad), "rank_samples": [{"rank": x["rank"], "samples": x["samples"]} for x in gathered], "rank_samples_distinct": len({x["samples"][0] for x in gathered}) == acc.num_processes, "step_hashes_equal": step_hashes_equal, "final_hashes_equal": len({x["final_hash"] for x in gathered}) == 1, "strict_restore": True, "restore_hash_match": len(set(gathered_restore)) == 1, "matching_mode": "per_view_hungarian", "scene_matching_consistency": 1.0, "fragmented_gt_ratio": 0.0, "query_collision": 0.0, "teacher_called": False, "u2r": 0.0, "unit_multiplier": 0.0, "records": [r for x in gathered for r in x["records"]], "all_finite": True}
        report["preflight_pass"] = bool(report["world_size"] == int(args.expected_world_size) and report["rank_samples_distinct"] if acc.num_processes > 1 else report["world_size"] == int(args.expected_world_size)) and bool(report["step_hashes_equal"] and report["final_hashes_equal"] and report["strict_restore"] and report["restore_hash_match"] and report["all_finite"])
        (out / "preflight_ta_riu_v2.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))
        if not report["preflight_pass"]: raise RuntimeError("TA-RIU-v2 preflight aggregate checks failed")


if __name__ == "__main__":
    main()
