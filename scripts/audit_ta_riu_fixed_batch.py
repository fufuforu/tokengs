"""TA-RIU v1 8+7 fixed-batch causal learnability audit.

This is a diagnostic-only probe.  It reuses the already validated v13
runtime-collated CPU batch and never writes a formal training workspace.
Every gradient audit uses a fresh forward; the optimizer update uses a third
fresh forward in the same iteration.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

torch._dynamo.config.disable = True
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokengs.models import model_registry
from tokengs.options import config_defaults
from tokengs.train import load_model_checkpoint, setup_optimizer
from scripts.eval_instance_lsm_protocol import (
    _mask_diagnostics,
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)


CFG = "semantic_v6_absolute_units_true_shared_ta_riu_v1_ddp8"
BASE_CFG = "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
CKPT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_"
    "w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors"
)
V13 = ROOT / "workspace/tsh_ga_idu1_fixed_batch_reproducible_audit_v13"
MILESTONES = (0, 1, 5, 25, 50, 100, 200)
TA_PREFIXES = (
    "ta_riu_shared_mixer.",
    "ta_riu_geometry_head.",
    "ta_riu_appearance_head.",
)
TRAIN_PREFIXES = ("tsh_instance_head.",) + TA_PREFIXES


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def model_hash(model) -> str:
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        h.update(name.encode())
        h.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prefix_hash(model, prefixes):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            h.update(name.encode())
            h.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def detach_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: detach_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_cpu(item) for item in value)
    return value


def batch_hashes(value, prefix=""):
    result = {}
    if torch.is_tensor(value):
        result[prefix] = {
            "sha256": tensor_hash(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    elif isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(batch_hashes(item, name))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(batch_hashes(item, f"{prefix}[{index}]"))
    return result


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def code_provenance():
    script = Path(__file__).resolve()
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "--", str(script)], cwd=ROOT
    )
    return {
        "script": str(script),
        "script_sha256": sha256_file(script),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def make_opt(name: str):
    opt = dataclasses.replace(config_defaults[name])
    opt.resume = str(CKPT)
    opt.num_workers = 0
    opt.batch_size = 1
    opt.prompt_overfit_single_batch = True
    opt.prompt_overfit_sample_index = 0
    opt.evaluating = False
    opt.eval_before_training = False
    opt.use_wandb = False
    opt.print_freq = 100000
    opt.log_image_freq = 100000
    return opt


def make_accelerator(opt):
    return Accelerator(
        mixed_precision=opt.mixed_precision,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )


def group_counts(raw_state):
    return {
        "absolute_gs_head": sum(k.startswith("absolute_gs_head.") for k in raw_state),
        "tsh_instance_head": sum(k.startswith("tsh_instance_head.") for k in raw_state),
        "decoder_tail": sum(k.startswith("enc_dec_backbone.decoder_blocks.") for k in raw_state),
        "pgsr": sum(k.startswith("tsh_slot_refine_head.") for k in raw_state),
        "query_memory_refiner": sum(k.startswith("tsh_instance_head.query_memory_refiner.") for k in raw_state),
    }


def set_ta_gates(base, gate: float):
    base.ta_riu_eval_gate_override = float(gate)
    base.ta_riu_eval_geo_gate_override = float(gate)
    base.ta_riu_eval_app_gate_override = float(gate)
    base.ta_riu_gate_eff = float(gate)
    base.ta_riu_geo_gate_eff = float(gate)
    base.ta_riu_app_gate_eff = float(gate)


def set_training_state(base, step: int):
    gate = min(1.0, (int(step) + 1) / float(max(1, int(base.opt.ta_riu_gate_steps))))
    base.ta_riu_gate_eff = gate
    base.ta_riu_geo_gate_eff = gate
    base.ta_riu_app_gate_eff = gate
    base.tsh_instance_loss_weight_eff = 1.0
    base.tsh_unit_grad_eff = 1.0
    base.tsh_mbm_u2r_eff = 0.0
    base.teacher_lambda_eff = 0.0
    return gate


def output_cache(output):
    keys = (
        "base_unit_logits", "unit_logits", "pi_unit", "pi_gs_refined",
        "rendered_instance_group_probability", "gaussians", "images_pred",
        "ta_riu_q_abs_base", "ta_riu_z_shared", "ta_riu_memory",
        "ta_riu_delta", "ta_riu_base_gaussians", "ta_riu_joint_gaussians",
    )
    return {key: output[key].detach().cpu() for key in keys if key in output}


def _gt_best_values(preds, pids, gts, gids):
    values = []
    for gt, gid in zip(gts, gids):
        gt = np.asarray(gt, dtype=np.float32)
        area = float(gt.sum())
        best = 0.0
        for pred, pid in zip(preds, pids):
            if pid != gid:
                continue
            pred = np.asarray(pred, dtype=np.float32)
            inter = float((pred * gt).sum())
            union = float(pred.sum()) + area - inter
            best = max(best, inter / max(union, 1e-8))
        values.append(best)
    return np.asarray(values, dtype=np.float32)


def evaluate(output, batch, opt, reference=None):
    probability = output["rendered_instance_group_probability"][0].detach().float().cpu().numpy()
    labels = batch["instance_label_output"][0].long().cpu().numpy()
    preds, scores, pred_ids, gts, gt_ids = [], [], [], [], []
    active_counts = []
    entropy = []
    for view in range(probability.shape[1]):
        probs = probability[:, view, 0]
        group_ids = np.argmax(probs, axis=0)
        active_counts.append(int(sum((group_ids == q).any() for q in range(probability.shape[0] - 1))))
        nonvoid = probs[:-1]
        mass = nonvoid / np.maximum(nonvoid.sum(0, keepdims=True), 1e-8)
        entropy.append(float(-(mass * np.log(np.maximum(mass, 1e-8))).sum(0).mean()))
        masks, view_scores = masks_from_group_probs(
            probs, void_channel=probability.shape[0] - 1, min_mask_area=1
        )
        image_id = f"{batch['scene_name'][0]}:{view}"
        preds.extend(masks)
        scores.extend(view_scores)
        pred_ids.extend([image_id] * len(masks))
        view_gts = gt_masks_from_instance_map(labels[view], min_mask_area=1)
        gts.extend(view_gts)
        gt_ids.extend([image_id] * len(view_gts))
    ap = instance_ap(
        preds, scores, gts, thresholds=(0.25, 0.5, 0.75), vectorized=True,
        pred_image_ids=pred_ids, gt_image_ids=gt_ids,
    )
    diagnostics = _mask_diagnostics(preds, gts, pred_ids, gt_ids)
    best = _gt_best_values(preds, pred_ids, gts, gt_ids)
    bins = [0.0, 0.25, 0.4, 0.5, 0.75, float("inf")]
    hist = {
        "lt_0.25": int((best < 0.25).sum()),
        "0.25_0.40": int(((best >= 0.25) & (best < 0.4)).sum()),
        "0.40_0.50": int(((best >= 0.4) & (best < 0.5)).sum()),
        "0.50_0.75": int(((best >= 0.5) & (best < 0.75)).sum()),
        "ge_0.75": int((best >= 0.75).sum()),
    }
    # Saved milestone caches retain ``pi_unit``; live model outputs also
    # expose the historical alias ``instance_group_probabilities``.
    pi_unit = output.get("instance_group_probabilities", output["pi_unit"])[0].detach().float()
    # Query usage is over the unit axes; keep the group/query axis intact.
    usage = pi_unit[..., :-1].mean(dim=(0, 1))
    logits = output["unit_logits"].detach().float()
    base_logits = output.get("base_unit_logits", output["unit_logits"]).detach().float()
    final_pi = torch.softmax(logits, dim=-1)
    base_pi = torch.softmax(base_logits, dim=-1)
    assignment_changed = logits[..., :-1].argmax(-1) != base_logits[..., :-1].argmax(-1)
    result = {
        "ap25": float(ap["ap_25"]),
        "ap50": float(ap["ap_50"]),
        "ap75": float(ap["ap_75"]),
        "mean_best_gt_iou": diagnostics["mean_best_gt_iou"],
        "recall_iou25": diagnostics["recall_iou25"],
        "recall_iou50": diagnostics["recall_iou50"],
        "recall_iou75": diagnostics["recall_iou75"],
        "pred_gt": len(preds) / max(1, len(gts)),
        "prediction_count": len(preds),
        "gt_count": len(gts),
        "void_ratio": float(probability[-1].mean()),
        "active_queries_mean": float(np.mean(active_counts)),
        "active_queries_max": int(max(active_counts, default=0)),
        "mass_queries_gt_001": int((usage > 0.001).sum()),
        "effective_query_count": float(np.exp(np.mean(entropy))),
        "assignment_entropy": float(np.mean(entropy)),
        "assignment_changed_vs_base": float(assignment_changed.float().mean()),
        "best_iou_histogram": hist,
        "best_iou_values": best.tolist(),
        "psnr": float(output["psnr"].detach()),
        "ssim": float(output["ssim"].detach()) if torch.isfinite(output["ssim"]).item() else None,
        "lpips": float(output["lpips"].detach()) if torch.isfinite(output["lpips"]).item() else None,
    }
    if "ta_riu_base_gaussians" in output:
        base_gs = output["ta_riu_base_gaussians"].detach().float()
        joint_gs = output["ta_riu_joint_gaussians"].detach().float()
        delta = (joint_gs - base_gs).abs().flatten().cpu().numpy()
        result["joint_gaussian_delta"] = {
            "mean": float(delta.mean()), "std": float(delta.std()),
            "p50": float(np.percentile(delta, 50)), "p90": float(np.percentile(delta, 90)),
            "p99": float(np.percentile(delta, 99)), "max": float(delta.max()),
            "gt_1e-6": float((delta > 1e-6).mean()),
        }
        result["z_shared_delta_l2"] = float(
            (output["ta_riu_z_shared"].detach().float() - output["ta_riu_q_abs_base"].detach().float()).square().mean().sqrt()
        )
    if reference is not None:
        # For a milestone, "assignment_changed_vs_base" must compare the
        # trained output with the frozen Both@1420 reference, not with the
        # output's own base logits (which are identical when TA-RIU has no
        # post-logit residual).  The latter made this diagnostic silently
        # report zero even when the trained TSH readout changed.
        reference_logits = reference.get("unit_logits", reference.get("base_unit_logits"))
        if reference_logits is not None:
            reference_logits = reference_logits.detach().float()
            result["assignment_changed_vs_base"] = float(
                (logits[..., :-1].argmax(-1) != reference_logits[..., :-1].argmax(-1))
                .float().mean()
            )
        ref_prob = reference["rendered_instance_group_probability"][0].detach().float()
        cur_prob = output["rendered_instance_group_probability"][0].detach().float()
        ref_binary = ref_prob[:-1].amax(0) > ref_prob[-1]
        cur_binary = cur_prob[:-1].amax(0) > cur_prob[-1]
        result["binary_mask_pixel_change_vs_step0"] = float((ref_binary != cur_binary).float().mean())
        result["soft_mask_max_diff_vs_step0"] = float((cur_prob - ref_prob).abs().max())
        result["rgb_max_diff_vs_step0"] = float((output["images_pred"].detach().float() - reference["images_pred"].detach().float()).abs().max())
    return result


def prefix_norms(model, prefixes, gradients=True):
    result = {}
    for prefix in prefixes:
        value = 0.0
        for name, parameter in model.named_parameters():
            if name.startswith(prefix):
                tensor = parameter.grad if gradients else parameter
                if tensor is not None:
                    value += float(tensor.detach().float().square().sum())
        result[prefix] = value ** 0.5
    return result


def gradient_audit(model, accelerator, batch, step, kind):
    model.train()
    base = accelerator.unwrap_model(model)
    set_training_state(base, step)
    for parameter in model.parameters():
        parameter.grad = None
    with accelerator.autocast():
        output = model(batch, compute_quality_metrics=False)
    loss = output["loss_instance_group"] if kind == "instance" else output["loss_rgb"]
    if not torch.isfinite(loss).all():
        raise FloatingPointError(f"non-finite {kind} audit loss")
    accelerator.backward(loss)
    norms = prefix_norms(base, TRAIN_PREFIXES)
    norms["absolute_gs_head."] = prefix_norms(base, ("absolute_gs_head.",))["absolute_gs_head."]
    norms["enc_dec_backbone.decoder_blocks."] = prefix_norms(base, ("enc_dec_backbone.decoder_blocks.",))["enc_dec_backbone.decoder_blocks."]
    for parameter in model.parameters():
        parameter.grad = None
    return {"loss": float(loss.detach()), "grad_norms": norms}


def trainable_state(base):
    return {
        name: parameter.detach().cpu()
        for name, parameter in base.state_dict().items()
        if name.startswith(TRAIN_PREFIXES)
    }


def load_model(name, accelerator, train=False):
    opt = make_opt(name)
    model = model_registry[opt.model_type](opt)
    if train:
        model.train()
    else:
        model.eval()
    load_model_checkpoint(opt, model, accelerator, 0)
    return opt, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    out = ROOT / args.workspace
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing non-empty audit workspace: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if args.steps != 200:
        raise ValueError("TA-RIU audit is fixed at exactly 200 optimizer steps")
    if not CKPT.is_file():
        raise FileNotFoundError(CKPT)
    batch_path = V13 / "fixed_batch_cpu.pt"
    manifest_path = V13 / "fixed_batch_manifest.json"
    hashes_path = V13 / "fixed_batch_hashes.json"
    if not batch_path.is_file() or not manifest_path.is_file() or not hashes_path.is_file():
        raise FileNotFoundError("validated v13 fixed-batch artifact is incomplete")
    cpu_batch = torch.load(batch_path, map_location="cpu", weights_only=False)
    saved_hashes = json.loads(hashes_path.read_text())['tensor_hashes']
    if batch_hashes(cpu_batch) != saved_hashes:
        raise RuntimeError("v13 fixed batch hash mismatch")
    fixed = json.loads(manifest_path.read_text())
    if fixed.get("scene_id") != "scene0016_02":
        raise RuntimeError(f"unexpected fixed scene: {fixed.get('scene_id')}")
    if fixed.get("context_views") != 8 or fixed.get("target_views") != 7:
        raise RuntimeError("v13 artifact is not 8+7")
    if cpu_batch["images_input"].shape[1] != 8 or cpu_batch["images_output"].shape[1] != 7:
        raise RuntimeError("runtime batch is not 8+7")
    if cpu_batch["instance_label_output"].shape[1] != 7:
        raise RuntimeError("target GT is not aligned with seven target views")

    raw = torch.load if False else None
    from safetensors.torch import load_file
    raw_state = load_file(str(CKPT), device="cpu")
    counts = group_counts(raw_state)
    expected = {"absolute_gs_head": 24, "tsh_instance_head": 50, "decoder_tail": 324, "pgsr": 0, "query_memory_refiner": 0}
    if counts != expected:
        raise RuntimeError(f"checkpoint namespace mismatch: {counts} != {expected}")
    if any("ga_idu" in key.lower() or "refine" in key.lower() for key in raw_state):
        raise RuntimeError("TA-RIU source checkpoint contains forbidden refiner/GA keys")

    set_seed(1729)
    acc = make_accelerator(make_opt(CFG))
    device = acc.device
    # Keep independent tensor containers for the baseline and TA forwards so
    # an evaluator/model-side cache mutation cannot contaminate step-0
    # equivalence.
    baseline_batch = move_to_device(cpu_batch, device)
    batch = move_to_device(cpu_batch, device)
    provenance = code_provenance()

    # Baseline reference is evaluated before the TA model is allocated.
    base_opt, baseline = load_model(BASE_CFG, acc, train=False)
    baseline = baseline.to(device)
    base_model = baseline
    baseline.eval()
    with torch.no_grad(), acc.autocast():
        base_output = baseline(baseline_batch, compute_quality_metrics=False)
        baseline_first_encoder_values = base_model._last_encoder_values.detach().cpu()
        baseline_repeat = baseline(baseline_batch, compute_quality_metrics=False)
    base_cache = output_cache(base_output)
    baseline_q_abs = base_model._tsh_last_q_abs.detach().cpu()
    baseline_gs = base_model._tsh_last_student_gaussians.detach().cpu()
    baseline_hidden = base_output["gs_token_hidden"].detach().cpu()
    baseline_encoder_values = baseline_first_encoder_values
    baseline_repeat_encoder_values = base_model._last_encoder_values.detach().cpu()
    baseline_prefix_hashes = {
        "absolute": prefix_hash(base_model, ("absolute_gs_head.",)),
        "tsh": prefix_hash(base_model, ("tsh_instance_head.",)),
        "tail": prefix_hash(base_model, ("enc_dec_backbone.decoder_blocks.",)),
        "encoder_decoder": prefix_hash(base_model, ("enc_dec_backbone.",)),
        "gs_tokens": prefix_hash(base_model, ("gs_tokens",)),
        "patch_embed": prefix_hash(base_model, ("patch_embed.",)),
        "patch_plucker": prefix_hash(base_model, ("patch_plucker_embed.",)),
        "activation": prefix_hash(base_model, ("activation_head.",)),
        "anchor": prefix_hash(base_model, ("anchor_pos_encoder.",)),
    }
    baseline_metrics = evaluate(base_output, baseline_batch, base_opt)
    del baseline, base_model, base_output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # The TA constructor adds fresh modules after the shared v6 modules.  A
    # fresh seed before its construction keeps all uncheckpointed base
    # parameters/buffers identical to the Both reference instance.
    set_seed(1729)
    opt, model = load_model(CFG, acc, train=True)
    optimizer = setup_optimizer(opt, model, acc, 0)
    model, optimizer = acc.prepare(model, optimizer)
    base = acc.unwrap_model(model)
    trainable = [name for name, parameter in base.named_parameters() if parameter.requires_grad]
    if not trainable or not all(name.startswith(TRAIN_PREFIXES) for name in trainable):
        raise RuntimeError(f"unexpected TA trainable set: {trainable[:12]}")
    if any(name.startswith("absolute_gs_head.") for name in trainable):
        raise RuntimeError("absolute_gs_head is trainable in TA-RIU")
    initial = {name: parameter.detach().float().clone() for name, parameter in base.named_parameters() if name.startswith(TRAIN_PREFIXES)}
    ta_prefix_hashes = {
        "absolute": prefix_hash(base, ("absolute_gs_head.",)),
        "tsh": prefix_hash(base, ("tsh_instance_head.",)),
        "tail": prefix_hash(base, ("enc_dec_backbone.decoder_blocks.",)),
        "encoder_decoder": prefix_hash(base, ("enc_dec_backbone.",)),
        "gs_tokens": prefix_hash(base, ("gs_tokens",)),
        "patch_embed": prefix_hash(base, ("patch_embed.",)),
        "patch_plucker": prefix_hash(base, ("patch_plucker_embed.",)),
        "activation": prefix_hash(base, ("activation_head.",)),
        "anchor": prefix_hash(base, ("anchor_pos_encoder.",)),
    }

    set_ta_gates(base, 0.0)
    model.eval()
    with torch.no_grad():
        step0 = model(batch, compute_quality_metrics=False)
    step0_cache = output_cache(step0)
    identity = {
        "q_abs_max_diff": float((step0["ta_riu_q_abs_base"].detach().cpu() - baseline_q_abs).abs().max()),
        "gs_token_hidden_max_diff": float((step0["gs_token_hidden"].detach().cpu() - baseline_hidden).abs().max()),
        "encoder_values_max_diff": float((base._last_encoder_values.detach().cpu() - baseline_encoder_values).abs().max()),
        "baseline_repeat_encoder_max_diff": float((baseline_repeat["gs_token_hidden"].detach().cpu() - baseline_hidden).abs().max()),
        "baseline_repeat_rgb_max_diff": float((baseline_repeat["images_pred"].detach().cpu() - base_cache["images_pred"]).abs().max()),
        "student_gs_max_diff": float((step0["ta_riu_base_gaussians"].detach().cpu() - baseline_gs).abs().max()),
        "rgb_max_diff": float((step0["images_pred"].detach().cpu() - base_cache["images_pred"]).abs().max()),
        "unit_logits_max_diff": float((step0["unit_logits"].detach().cpu() - base_cache["unit_logits"]).abs().max()),
        "rendered_masks_max_diff": float((step0["rendered_instance_group_probability"].detach().cpu() - base_cache["rendered_instance_group_probability"]).abs().max()),
        "joint_base_max_diff": float((step0["ta_riu_joint_gaussians"].detach() - step0["ta_riu_base_gaussians"].detach()).abs().max()),
        "baseline_prefix_hashes": baseline_prefix_hashes,
        "ta_prefix_hashes": ta_prefix_hashes,
    }
    numeric_identity = [value for value in identity.values() if isinstance(value, (int, float))]
    if max(numeric_identity) > 5e-5 or baseline_prefix_hashes != ta_prefix_hashes:
        raise RuntimeError(f"TA-RIU step0 identity failed: {identity}")

    metrics = {
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "device_count": torch.cuda.device_count(),
        },
        "checkpoint": {
            "path": str(CKPT), "sha256": sha256_file(CKPT),
            "counts": counts, "fresh_reset": False,
            "pgsr_absent": counts["pgsr"] == 0,
            "query_memory_refiner_absent": counts["query_memory_refiner"] == 0,
        },
        "config": {key: str(value) for key, value in vars(opt).items()},
        "code_provenance": provenance,
        "fixed_batch": fixed,
        "fixed_batch_hashes": saved_hashes,
        "trainable": trainable,
        "trainable_parameter_count": int(sum(p.numel() for n, p in base.named_parameters() if n.startswith(TRAIN_PREFIXES))),
        "step0_identity": identity,
        "baseline": baseline_metrics,
        "milestones": {"step0": {**evaluate(step0, batch, opt), "gate": 0.0, "model_hash": model_hash(base)}},
        "gradient_audits": {},
        "gate_ablations": {},
    }
    (out / "step_000_cache.pt").__class__
    torch.save(step0_cache, out / "cache_step_000.pt")
    torch.save({"trainable_state": trainable_state(base), "step": 0, "parameter_hash": model_hash(base)}, out / "step_000.pt")

    for step in range(args.steps):
        completed = step + 1
        model.train()
        gate = set_training_state(base, step)
        if completed in (1, 100):
            metrics["gradient_audits"][f"step{completed}"] = {
                "instance_only": gradient_audit(model, acc, batch, step, "instance"),
                "rgb_only": gradient_audit(model, acc, batch, step, "rgb"),
            }
        optimizer.zero_grad(set_to_none=True)
        # Third independent forward: only this graph reaches the formal
        # total-loss backward and optimizer step.
        with acc.autocast():
            output = model(batch, compute_quality_metrics=False)
        total_loss = output["loss"]
        if not torch.isfinite(total_loss).all():
            raise FloatingPointError(f"non-finite total loss at step {completed}")
        acc.backward(total_loss)
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"non-finite gradient at step {completed}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt.gradient_clip))
        optimizer.step()
        for parameter in model.parameters():
            if not torch.isfinite(parameter.detach()).all():
                raise FloatingPointError(f"non-finite parameter at step {completed}")

        if completed in MILESTONES[1:]:
            model.eval()
            set_ta_gates(base, 1.0)
            with torch.no_grad():
                snapshot = model(batch, compute_quality_metrics=False)
            rec = evaluate(snapshot, batch, opt, reference=step0)
            rec.update({
                "gate": gate,
                "model_hash": model_hash(base),
                "total_train_loss": float(total_loss.detach()),
                "module_grad_norms_after_total": prefix_norms(base, TRAIN_PREFIXES),
                "module_update_norms_from_step0": {
                    prefix: float(sum((parameter.detach().float() - initial[name]).square().sum() for name, parameter in base.named_parameters() if name == name and name.startswith(prefix)).sqrt())
                    for prefix in TRAIN_PREFIXES
                },
            })
            metrics["milestones"][f"step{completed}"] = rec
            torch.save(output_cache(snapshot), out / f"cache_step_{completed:03d}.pt")
            torch.save({
                "trainable_state": trainable_state(base),
                "optimizer": optimizer.state_dict(),
                "scheduler": None,
                "scheduler_reason": "fixed-batch smoke uses constant LR",
                "step": completed,
                "gate": 1.0 if completed >= 25 else gate,
                "parameter_hash": model_hash(base),
                "rng": torch.get_rng_state(),
            }, out / f"step_{completed:03d}.pt")
            ablations = {}
            for name, mode in (("full", "normal"), ("memory_zero", "zero"), ("memory_shuffle", "shuffle")):
                base.ta_riu_memory_mode = mode
                with torch.no_grad():
                    ablation_output = model(batch, compute_quality_metrics=False)
                ablations[name] = evaluate(ablation_output, batch, opt)
            base.ta_riu_memory_mode = "normal"
            metrics["gate_ablations"][f"step{completed}"] = ablations

    # Independent restore from the final diagnostic checkpoint.
    final_state = torch.load(out / "step_200.pt", map_location="cpu", weights_only=False)
    set_seed(1729)
    restore_opt, restored = load_model(CFG, acc, train=False)
    restored = restored.to(device)
    restored_base = restored
    restored_base.load_state_dict(final_state["trainable_state"], strict=False)
    set_ta_gates(restored_base, 1.0)
    restored.eval()
    with torch.no_grad():
        restored_output = restored(batch, compute_quality_metrics=False)
    current_cache = torch.load(out / "cache_step_200.pt", map_location="cpu", weights_only=False)
    restore_diffs = {
        key: float((restored_output[key].detach().cpu().float() - value.float()).abs().max())
        for key, value in current_cache.items() if key in restored_output
    }
    metrics["independent_restore"] = {
        "parameter_hash_match": model_hash(restored_base) == final_state["parameter_hash"],
        "cache_max_diffs": restore_diffs,
        "match": all(value <= 1e-5 for value in restore_diffs.values()),
    }
    metrics["all_finite"] = True
    metrics["ready_for_short_multiscene_probe"] = False
    (out / "per_step_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
