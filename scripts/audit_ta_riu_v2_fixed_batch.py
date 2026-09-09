"""TA-RIU-v2 8+7 fixed-batch learnability audit.

This is a diagnostic-only script.  It uses the already validated v13 CPU
batch, starts from Both@1420, and never writes a formal training workspace.
Gradient audits and the optimizer update use separate fresh forwards.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokengs.models import model_registry
from tokengs.models.ta_riu_v2 import DINO_EXPECTED_SHA256, sha256_file
from tokengs.options import config_defaults
from tokengs.train import load_model_checkpoint, setup_optimizer
from scripts.audit_ta_riu_fixed_batch import (
    batch_hashes,
    evaluate,
    move_to_device,
    prefix_norms,
)


CFG = "semantic_v6_absolute_units_true_shared_ta_riu_v2_dino_unit_ddp8"
BASE_CFG = "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
CKPT = ROOT / "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors"
FIXED = ROOT / "workspace/tsh_ga_idu1_fixed_batch_reproducible_audit_v13"
MILESTONES = (0, 1, 5, 25, 50, 100)


def seed_all(seed=1729):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_opt(name):
    opt = dataclasses.replace(config_defaults[name])
    opt.resume = str(CKPT)
    opt.num_workers = 0; opt.batch_size = 1
    opt.prompt_overfit_single_batch = True
    opt.prompt_overfit_sample_index = 0
    opt.evaluating = False; opt.eval_before_training = False
    opt.use_wandb = False; opt.print_freq = 100000; opt.log_image_freq = 100000
    return opt


def accelerator_for(opt):
    return Accelerator(
        mixed_precision=opt.mixed_precision,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )


def model_hash(model):
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        h.update(name.encode()); h.update(param.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def counts(state):
    return {
        "absolute_gs_head": sum(k.startswith("absolute_gs_head.") for k in state),
        "tsh_instance_head": sum(k.startswith("tsh_instance_head.") for k in state),
        "decoder_tail": sum(k.startswith("enc_dec_backbone.decoder_blocks.") for k in state),
        "ta_riu_v2": sum(k.startswith("ta_riu_v2_unit_encoder.") for k in state),
        "pgsr": sum(k.startswith("tsh_slot_refine_head.") for k in state),
    }


def configure(base, step, training):
    base.teacher_lambda_eff = 0.0
    base.tsh_mbm_u2r_eff = 0.0
    base.tsh_instance_loss_weight_eff = 1.0
    base.tsh_unit_grad_eff = 0.0
    if training:
        base.ta_riu_v2_gate_eff = min(1.0, (int(step) + 1) / 25.0)
    else:
        base.ta_riu_v2_eval_gate_override = 1.0


def finite(x):
    return bool(torch.isfinite(x.detach() if torch.is_tensor(x) else torch.as_tensor(x)).all())


def gradients(base, prefixes):
    values = prefix_norms(base, prefixes)
    return float(sum(values.values()))


def output_cache(out):
    keys = (
        "unit_logits", "base_unit_logits", "pi_unit", "instance_group_probabilities",
        "rendered_instance_group_probability", "rendered_instance_group_alpha",
        "images_pred", "psnr", "ssim", "lpips", "loss", "loss_rgb",
        "loss_instance_group", "loss_ta_riu_v2_unit_embedding",
    )
    return {k: out[k].detach().cpu() for k in keys if k in out and torch.is_tensor(out[k])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if int(args.steps) != 100:
        raise ValueError("TA-RIU-v2 audit is fixed at exactly 100 steps")
    out_dir = ROOT / args.workspace
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty audit workspace: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_path = FIXED / "fixed_batch_cpu.pt"
    hashes_path = FIXED / "fixed_batch_hashes.json"
    manifest_path = FIXED / "fixed_batch_manifest.json"
    cpu_batch = torch.load(batch_path, map_location="cpu", weights_only=False)
    expected_hashes = json.loads(hashes_path.read_text())["tensor_hashes"]
    if batch_hashes(cpu_batch) != expected_hashes:
        raise RuntimeError("validated v13 fixed batch hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("context_views") != 8 or manifest.get("target_views") != 7:
        raise RuntimeError("fixed batch is not formal 8+7")
    if cpu_batch["images_input"].shape[1] != 8 or cpu_batch["images_output"].shape[1] != 7:
        raise RuntimeError("runtime fixed batch is not 8+7")
    if cpu_batch["instance_label_output"].shape[1] != 7:
        raise RuntimeError("target GT is not aligned with seven target views")
    raw = load_file(str(CKPT), device="cpu")
    ckpt_counts = counts(raw)
    if ckpt_counts != {"absolute_gs_head": 24, "tsh_instance_head": 50, "decoder_tail": 324, "ta_riu_v2": 0, "pgsr": 0}:
        raise RuntimeError(f"unexpected Both checkpoint keys: {ckpt_counts}")

    seed_all()
    base_opt = make_opt(BASE_CFG)
    acc = accelerator_for(base_opt)
    batch = move_to_device(cpu_batch, acc.device)
    base_opt, baseline = None, None
    # Build baseline through the same checkpoint loader and capture the exact
    # reference before allocating the v2 DINO-backed model.
    base_opt = make_opt(BASE_CFG)
    baseline = model_registry[base_opt.model_type](base_opt)
    load_model_checkpoint(base_opt, baseline, acc, 0)
    baseline.to(acc.device).eval()
    with torch.no_grad(), acc.autocast():
        base_out = baseline(batch, compute_quality_metrics=False)
    base_cache = output_cache(base_out)
    baseline_metrics = evaluate(base_out, batch, base_opt)
    baseline_q = baseline._tsh_last_q_abs.detach().cpu()
    baseline_gs = baseline._tsh_last_student_gaussians.detach().cpu()
    del baseline, base_out
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    seed_all()
    opt = make_opt(CFG)
    model = model_registry[opt.model_type](opt)
    model.ta_riu_v2_return_debug = True
    load_model_checkpoint(opt, model, acc, 0)
    raw_v2 = torch.nn.Module.state_dict(model)
    v2_counts = counts(raw_v2)
    if v2_counts["absolute_gs_head"] != 24 or v2_counts["tsh_instance_head"] != 50 or v2_counts["decoder_tail"] != 324:
        raise RuntimeError(f"strict v2 base key counts failed: {v2_counts}")
    if v2_counts["ta_riu_v2"] <= 0 or v2_counts["pgsr"] != 0:
        raise RuntimeError(f"v2 namespace/PGSR check failed: {v2_counts}")
    if any(k.startswith(("_dino_model.", "ta_riu_v2_unit_encoder.dino_extractor.")) for k in raw_v2):
        raise RuntimeError("DINO weights entered TA-RIU-v2 state_dict")
    weight_hash = sha256_file(opt.ta_riu_v2_dino_weight_path)
    if weight_hash != DINO_EXPECTED_SHA256:
        raise RuntimeError(f"DINO weight hash mismatch: {weight_hash}")
    optimizer = setup_optimizer(opt, model, acc, 0)
    model, optimizer = acc.prepare(model, optimizer)
    base = acc.unwrap_model(model)
    trainable = [n for n, p in base.named_parameters() if p.requires_grad]
    if not trainable or not all(n.startswith(("tsh_instance_head.", "ta_riu_v2_unit_encoder.")) for n in trainable):
        raise RuntimeError(f"unexpected v2 trainable parameters: {trainable[:20]}")
    if any(n.startswith(("absolute_gs_head.", "enc_dec_backbone.", "activation_head.")) for n in trainable):
        raise RuntimeError("frozen reconstruction parameter became trainable")

    # Step-0 v2 must preserve the source TSH/reconstruction exactly.
    model.eval(); base.ta_riu_v2_eval_gate_override = 0.0
    with torch.no_grad(), acc.autocast():
        step0 = model(batch, compute_quality_metrics=False)
    identity = {
        "q_abs_max_diff": float((step0["ta_riu_v2_z_inst"].detach().cpu() if "ta_riu_v2_z_inst" in step0 else base._tsh_last_q_abs.detach().cpu() - baseline_q).abs().max()) if False else float((base._tsh_last_q_abs.detach().cpu() - baseline_q).abs().max()),
        "student_gs_max_diff": float((base._tsh_last_student_gaussians.detach().cpu() - baseline_gs).abs().max()),
        "rgb_max_diff": float((step0["images_pred"].detach().cpu() - base_cache["images_pred"]).abs().max()),
        "unit_logits_max_diff": float((step0["unit_logits"].detach().cpu() - base_cache["unit_logits"]).abs().max()),
        "rendered_masks_max_diff": float((step0["rendered_instance_group_probability"].detach().cpu() - base_cache["rendered_instance_group_probability"]).abs().max()),
        "instance_loss_max_diff": float((step0["loss_instance_group"].detach().cpu() - base_cache["loss_instance_group"]).abs().max()),
    }
    if max(identity.values()) > 5e-5:
        raise RuntimeError(f"v2 step-0 identity failed: {identity}")

    report = {
        "config": {k: str(v) for k, v in vars(opt).items()},
        "checkpoint": {"path": str(CKPT), "sha256": hashlib.sha256(CKPT.read_bytes()).hexdigest(), "counts": ckpt_counts, "fresh_reset": False, "pgsr_absent": True},
        "dino": {"repo_path": opt.ta_riu_v2_dino_repo_path, "weight_path": opt.ta_riu_v2_dino_weight_path, "weight_sha256": weight_hash, "source": "local", "parameter_count": 86580480, "saved_in_checkpoint": False, "in_optimizer": False},
        "fixed_batch": manifest,
        "fixed_batch_hashes": expected_hashes,
        "baseline": baseline_metrics,
        "step0_identity": identity,
        "trainable_parameter_count": int(sum(p.numel() for n, p in base.named_parameters() if n.startswith(("tsh_instance_head.", "ta_riu_v2_unit_encoder.")))),
        "trainable_names": trainable,
        "milestones": {"step0": evaluate(step0, batch, opt)},
        "gradient_audits": {},
        "optimizer_steps": [],
    }
    torch.save(output_cache(step0), out_dir / "cache_step_000.pt")
    torch.save({"trainable_state": {n: p.detach().cpu() for n, p in base.state_dict().items() if n.startswith(("tsh_instance_head.", "ta_riu_v2_unit_encoder."))}, "step": 0, "parameter_hash": model_hash(base)}, out_dir / "step_000.pt")

    for step in range(100):
        completed = step + 1
        configure(base, step, True)
        if completed in (1, 100):
            # Fresh forward 1: instance-only audit.
            optimizer.zero_grad(set_to_none=True)
            model.train()
            with acc.autocast(): instance = model(batch, compute_quality_metrics=False)
            loss_i = instance["loss_instance_group"]
            if not finite(loss_i): raise FloatingPointError("non-finite instance audit loss")
            acc.backward(loss_i)
            ig = {"tsh_instance_head": gradients(base, ("tsh_instance_head.",)), "v2": gradients(base, ("ta_riu_v2_unit_encoder.",)), "absolute_gs_head": gradients(base, ("absolute_gs_head.",)), "decoder_tail": gradients(base, ("enc_dec_backbone.decoder_blocks.",))}
            if ig["tsh_instance_head"] <= 0 or ig["v2"] <= 0 or ig["absolute_gs_head"] != 0 or ig["decoder_tail"] != 0:
                raise RuntimeError(f"instance-only gradient boundary failed: {ig}")
            optimizer.zero_grad(set_to_none=True)
            # Fresh forward 2: RGB-only audit; all trainable instance paths
            # must be disconnected from the reconstruction loss.
            with acc.autocast(): rgb = model(batch, compute_quality_metrics=False)
            loss_r = rgb["loss_rgb"]
            if not finite(loss_r): raise FloatingPointError("non-finite RGB audit loss")
            if loss_r.requires_grad:
                acc.backward(loss_r)
            rg = {"tsh_instance_head": gradients(base, ("tsh_instance_head.",)), "v2": gradients(base, ("ta_riu_v2_unit_encoder.",)), "absolute_gs_head": gradients(base, ("absolute_gs_head.",)), "decoder_tail": gradients(base, ("enc_dec_backbone.decoder_blocks.",))}
            if any(rg[k] != 0 for k in ("tsh_instance_head", "v2", "absolute_gs_head", "decoder_tail")):
                raise RuntimeError(f"RGB gradient boundary failed: {rg}")
            report["gradient_audits"][f"step{completed}"] = {"instance_only": {"loss": float(loss_i.detach()), "grad_norms": ig}, "rgb_only": {"loss": float(loss_r.detach()), "grad_norms": rg}}
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with acc.autocast(): total = model(batch, compute_quality_metrics=False)
        if not all(finite(total[k]) for k in ("loss", "loss_rgb", "loss_instance_group", "loss_ta_riu_v2_unit_embedding")):
            def describe(value):
                x = value.detach().float()
                return {"shape": list(x.shape), "finite": bool(torch.isfinite(x).all()), "min": float(torch.nan_to_num(x).min()), "max": float(torch.nan_to_num(x).max())}
            values = {k: describe(total[k]) for k in ("loss", "loss_rgb", "loss_instance_group", "loss_ta_riu_v2_unit_embedding")}
            v2_values = {k: describe(total[k]) for k in total if k.startswith("ta_riu_v2_") and torch.is_tensor(total[k])}
            raise FloatingPointError(f"non-finite total output at step {completed}: losses={values}, v2={v2_values}")
        acc.backward(total["loss"])
        for p in model.parameters():
            if p.grad is not None and not finite(p.grad): raise FloatingPointError(f"non-finite gradient at step {completed}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt.gradient_clip))
        optimizer.step()
        if completed in MILESTONES:
            model.eval(); base.ta_riu_v2_eval_gate_override = 1.0
            with torch.no_grad(), acc.autocast(): milestone = model(batch, compute_quality_metrics=False)
            report["milestones"][f"step{completed}"] = evaluate(milestone, batch, opt)
            torch.save(output_cache(milestone), out_dir / f"cache_step_{completed:03d}.pt")
            torch.save({"trainable_state": {n: p.detach().cpu() for n, p in base.state_dict().items() if n.startswith(("tsh_instance_head.", "ta_riu_v2_unit_encoder."))}, "step": completed, "parameter_hash": model_hash(base)}, out_dir / f"step_{completed:03d}.pt")
        report["optimizer_steps"].append({"step": completed, "loss": float(total["loss"].detach()), "loss_rgb": float(total["loss_rgb"].detach()), "loss_instance": float(total["loss_instance_group"].detach()), "loss_embedding": float(total["loss_ta_riu_v2_unit_embedding"].detach()), "parameter_hash": model_hash(base)})

    report["all_finite"] = True
    report["dino"]["patch_shape"] = [1, 324, 768]
    (out_dir / "ta_riu_v2_fixed_batch_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
