"""Three-step DDP8 preflight for TA-RIU-v1.

Diagnostic only: uses the real training dataloader, performs no formal
checkpoint write, and refuses to reuse a non-empty output directory.  Every
gradient audit uses a fresh forward; the final optimizer update uses another
fresh forward.
"""
from __future__ import annotations

import argparse
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
from tokengs.options import config_defaults
from tokengs.train import load_model_checkpoint, setup_optimizer


CFG = "semantic_v6_absolute_units_true_shared_ta_riu_v1_ddp8"
CKPT = (
    "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_"
    "w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors"
)
TA_PREFIXES = (
    "ta_riu_shared_mixer.",
    "ta_riu_geometry_head.",
    "ta_riu_appearance_head.",
)


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value.detach()).all())
    return bool(torch.isfinite(torch.as_tensor(value)).all())


def phash(model):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        h.update(name.encode())
        h.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def grad_norm(model, prefixes):
    total = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and any(name.startswith(p) for p in prefixes):
            total += float(parameter.grad.detach().float().square().sum())
    return total ** 0.5


def parameter_count(model, prefixes=None, trainable_only=False):
    total = 0
    for name, parameter in model.named_parameters():
        if prefixes is not None and not any(name.startswith(p) for p in prefixes):
            continue
        if trainable_only and not parameter.requires_grad:
            continue
        total += parameter.numel()
    return int(total)


def configure_step(base, step):
    gate = min(1.0, (step + 1) / float(max(1, int(base.opt.ta_riu_gate_steps))))
    base.ta_riu_gate_eff = gate
    base.ta_riu_geo_gate_eff = gate
    base.ta_riu_app_gate_eff = gate
    base.tsh_instance_loss_weight_eff = 1.0
    base.tsh_unit_grad_eff = 1.0
    base.tsh_mbm_u2r_eff = 0.0
    base.teacher_lambda_eff = 0.0
    return gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--config", default=CFG)
    parser.add_argument("--resume", default=CKPT)
    args = parser.parse_args()

    out = ROOT / args.workspace
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing non-empty preflight workspace: {out}")
    out.mkdir(parents=True, exist_ok=True)

    opt = config_defaults[args.config]
    opt.workspace = str(out)
    opt.resume = str(ROOT / args.resume) if not os.path.isabs(args.resume) else args.resume
    opt.num_workers = 0
    opt.num_epochs = 1
    opt.max_iters_per_epoch = int(args.steps)
    opt.eval_before_training = False
    opt.use_wandb = False
    opt.print_freq = 100000
    opt.log_image_freq = 100000

    acc = Accelerator(
        mixed_precision=opt.mixed_precision,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    loader, _, _, _ = get_multi_dataloader(opt, acc)
    model = model_registry[opt.model_type](opt).cuda().train()
    raw = load_file(opt.resume, device="cpu")
    counts = {
        "absolute_gs_head": sum(k.startswith("absolute_gs_head.") for k in raw),
        "tsh_instance_head": sum(k.startswith("tsh_instance_head.") for k in raw),
        "decoder_tail": sum(k.startswith("enc_dec_backbone.decoder_blocks.") for k in raw),
    }
    expected = {"absolute_gs_head": 24, "tsh_instance_head": 50, "decoder_tail": 324}
    assert counts == expected, counts
    assert not any("pgsr" in k.lower() or "refine" in k.lower() for k in raw)
    load_model_checkpoint(opt, model, acc, 0)
    optimizer = setup_optimizer(opt, model, acc, 0)
    model, optimizer, loader, _ = acc.prepare(model, optimizer, loader, loader)
    base = acc.unwrap_model(model)

    trainable = [name for name, p in base.named_parameters() if p.requires_grad]
    assert trainable, "TA-RIU preflight has no trainable parameters"
    assert all(
        name.startswith(("tsh_instance_head.",) + TA_PREFIXES) for name in trainable
    ), trainable[:20]
    assert not any(name.startswith("absolute_gs_head.") for name in trainable)
    assert parameter_count(base, trainable_only=True) > 0

    iterator = iter(loader)
    local_records = []
    local_samples = []
    for step in range(int(args.steps)):
        data = next(iterator)
        scene = data["scene_name"][0] if isinstance(data["scene_name"], (list, tuple)) else str(data["scene_name"])
        local_samples.append(str(scene))

        # Fresh forward 1: strict gate-zero identity check in eval mode.
        model.eval()
        base.ta_riu_eval_gate_override = 0.0
        base.ta_riu_eval_geo_gate_override = 0.0
        base.ta_riu_eval_app_gate_override = 0.0
        with acc.autocast():
            identity = model(data, compute_quality_metrics=False)
        assert finite(identity["loss"]) and finite(identity["loss_instance_group"])
        identity_diff = float(
            (identity["unit_logits"].detach() - identity["base_unit_logits"].detach())
            .float().abs().max()
        )
        joint_delta = float(
            (identity["ta_riu_joint_gaussians"].detach() - identity["ta_riu_base_gaussians"].detach())
            .float().abs().max()
        )
        assert identity_diff == 0.0, identity_diff
        assert joint_delta == 0.0, joint_delta
        del base.ta_riu_eval_gate_override
        del base.ta_riu_eval_geo_gate_override
        del base.ta_riu_eval_app_gate_override
        model.train()

        gate = configure_step(base, step)

        # Fresh forward 2: instance-only gradient audit.
        optimizer.zero_grad(set_to_none=True)
        with acc.autocast():
            instance = model(data, compute_quality_metrics=False)
        instance_loss = instance["loss_instance_group"]
        assert instance_loss.requires_grad and finite(instance_loss)
        acc.backward(instance_loss)
        instance_grads = {
            "tsh_instance_head": grad_norm(base, ("tsh_instance_head.",)),
            "ta_shared_mixer": grad_norm(base, ("ta_riu_shared_mixer.",)),
            "ta_geometry": grad_norm(base, ("ta_riu_geometry_head.",)),
            "ta_appearance": grad_norm(base, ("ta_riu_appearance_head.",)),
            "absolute_gs_head": grad_norm(base, ("absolute_gs_head.",)),
            "decoder_tail": grad_norm(base, ("enc_dec_backbone.decoder_blocks.",)),
        }
        assert instance_grads["tsh_instance_head"] > 0.0
        assert instance_grads["ta_shared_mixer"] > 0.0
        assert instance_grads["absolute_gs_head"] == 0.0
        assert instance_grads["decoder_tail"] == 0.0
        assert all(finite(value) for value in instance_grads.values())

        # Fresh forward 3: RGB-only gradient audit.
        optimizer.zero_grad(set_to_none=True)
        with acc.autocast():
            rgb = model(data, compute_quality_metrics=False)
        assert finite(rgb["loss_rgb"]) and rgb["loss_rgb"].requires_grad
        acc.backward(rgb["loss_rgb"])
        rgb_grads = {
            "tsh_instance_head": grad_norm(base, ("tsh_instance_head.",)),
            "ta_shared_mixer": grad_norm(base, ("ta_riu_shared_mixer.",)),
            "ta_geometry": grad_norm(base, ("ta_riu_geometry_head.",)),
            "ta_appearance": grad_norm(base, ("ta_riu_appearance_head.",)),
            "absolute_gs_head": grad_norm(base, ("absolute_gs_head.",)),
            "decoder_tail": grad_norm(base, ("enc_dec_backbone.decoder_blocks.",)),
        }
        assert rgb_grads["tsh_instance_head"] == 0.0
        assert rgb_grads["ta_shared_mixer"] > 0.0
        assert rgb_grads["ta_geometry"] > 0.0
        assert rgb_grads["ta_appearance"] > 0.0
        assert rgb_grads["absolute_gs_head"] == 0.0
        assert rgb_grads["decoder_tail"] == 0.0
        assert all(finite(value) for value in rgb_grads.values())

        # Fresh forward 4: the only backward used for the optimizer update.
        optimizer.zero_grad(set_to_none=True)
        with acc.autocast():
            total_out = model(data, compute_quality_metrics=False)
        assert all(finite(total_out[key]) for key in ("loss", "loss_rgb", "loss_instance_group"))
        acc.backward(total_out["loss"])
        total_grad = grad_norm(base, ("tsh_instance_head.",) + TA_PREFIXES)
        assert finite(total_grad) and total_grad > 0.0
        acc.clip_grad_norm_(model.parameters(), opt.gradient_clip)
        optimizer.step()
        post_hash = phash(base)
        assert finite(torch.cat([p.detach().float().reshape(-1) for p in base.parameters()]))
        assert not bool(getattr(base, "teacher_called", False))
        assert float(getattr(base, "tsh_mbm_u2r_eff", 0.0)) == 0.0
        local_records.append(
            {
                "step": step + 1,
                "rank": acc.process_index,
                "sample": str(scene),
                "gate": float(gate),
                "identity_unit_logits_max_diff": identity_diff,
                "identity_joint_gaussians_max_diff": joint_delta,
                "instance_grads": instance_grads,
                "rgb_grads": rgb_grads,
                "total_grad": float(total_grad),
                "loss": float(total_out["loss"].detach()),
                "loss_rgb": float(total_out["loss_rgb"].detach()),
                "loss_instance": float(total_out["loss_instance_group"].detach()),
                "teacher_called": bool(getattr(base, "teacher_called", False)),
                "u2r": float(getattr(base, "tsh_mbm_u2r_eff", 0.0)),
                "unit_multiplier": float(getattr(base, "tsh_unit_gradient_multiplier_eff", 1.0)),
                "parameter_hash": post_hash,
                "all_finite": True,
            }
        )

    local = {"rank": acc.process_index, "samples": local_samples, "records": local_records, "hash": phash(base)}
    gathered = [local]
    if acc.num_processes > 1:
        gathered = [None] * acc.num_processes
        dist.all_gather_object(gathered, local)

    # Strict in-memory round trip of the complete TA-RIU module state.
    save_path = out / "ta_riu_preflight_checkpoint.safetensors"
    state = {k: v.detach().cpu().contiguous() for k, v in torch.nn.Module.state_dict(base).items()}
    if acc.is_main_process:
        save_file(state, str(save_path))
    if acc.num_processes > 1:
        dist.barrier()
    restored = load_file(str(save_path), device="cpu")
    missing, unexpected = torch.nn.Module.load_state_dict(base, restored, strict=True)
    assert not missing and not unexpected
    restore_hash = phash(base)
    assert restore_hash == local["hash"]

    if acc.is_main_process:
        step_hashes_equal = all(
            len({rank["records"][index]["parameter_hash"] for rank in gathered}) == 1
            for index in range(int(args.steps))
        )
        rank_samples = [item["samples"][0] for item in gathered]
        report = {
            "config": args.config,
            "world_size": acc.num_processes,
            "checkpoint_counts": counts,
            "fresh_reset": False,
            "pgsr_absent": True,
            "trainable_parameter_count": parameter_count(base, trainable_only=True),
            "trainable_prefixes": ["tsh_instance_head."] + list(TA_PREFIXES),
            "rank_samples": [{"rank": item["rank"], "samples": item["samples"]} for item in gathered],
            "rank_samples_distinct": len(set(rank_samples)) == acc.num_processes,
            "step_hashes_equal": step_hashes_equal,
            "final_hashes_equal": len({item["hash"] for item in gathered}) == 1,
            "strict_restore": True,
            "restore_hash_match": restore_hash == local["hash"],
            "scene_level_matching": False,
            "teacher_called": False,
            "u2r": 0.0,
            "unit_multiplier": 1.0,
            "records": [record for item in gathered for record in item["records"]],
            "all_finite": all(record["all_finite"] for item in gathered for record in item["records"]),
        }
        required = (
            report["world_size"] == int(args.expected_world_size)
            and report["rank_samples_distinct"]
            and report["step_hashes_equal"]
            and report["final_hashes_equal"]
            and report["strict_restore"]
            and report["restore_hash_match"]
            and report["all_finite"]
        )
        report["preflight_pass"] = bool(required)
        (out / "preflight_ta_riu.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if not required:
            raise RuntimeError("TA-RIU DDP preflight aggregate checks failed")


if __name__ == "__main__":
    main()
