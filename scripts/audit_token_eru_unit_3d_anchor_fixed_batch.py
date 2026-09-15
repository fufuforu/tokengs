"""Matched 100-step A0/A1 fixed-batch audit for 3D unit anchors."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.token_eru.unit_3d_anchor import Unit3DAnchorOutput  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import (  # noqa: E402
    configure_joint_formation_trainability,
    load_model_checkpoint,
    setup_optimizer,
    setup_scheduler,
)
from tokengs.utils.instance_ap import (  # noqa: E402
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)
from scripts.eval_token_eru_short200 import _mask_diagnostics  # noqa: E402


def _load_cached_gsplat_extension() -> None:
    so_path = os.environ.get("GSPLAT_PRECOMPILED_SO")
    if not so_path or not os.path.isfile(so_path) or "gsplat_cuda" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location("gsplat_cuda", so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load cached gsplat extension: {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["gsplat_cuda"] = module
    import gsplat
    sys.modules.setdefault("gsplat.csrc", module)
    setattr(gsplat, "csrc", module)


_load_cached_gsplat_extension()


PARENT = Path(
    "/space/mawb/tokengs/workspace/"
    "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_ddp8/checkpoints/model_step_000250.safetensors"
)
PARENT_SHA = "61debbf5f55b76eea16012aea00fe19291309177bfdc6dd0420f6d1cb58d8cba"
A0 = "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_local250_control_short200_ddp8"
A1 = "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_local250_3d_anchor_v1_short200_ddp8"
MILESTONES = (0, 1, 2, 5, 25, 50, 100)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    return value


def set_paired_seed(seed: int, rank: int = 0) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def batch_hash(batch) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(batch.items()):
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_parameter_hash(model) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        count += 1
    if count == 0:
        raise RuntimeError("empty parameter hash")
    return digest.hexdigest()


def trainable_parameter_hash(model) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(str(tuple(parameter.shape)).encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
            count += 1
    if count == 0:
        raise RuntimeError("empty trainable parameter hash")
    return digest.hexdigest()


def group_grad_norms(optimizer) -> dict[str, float]:
    values = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        squared = 0.0
        for parameter in group["params"]:
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().float().square().sum())
        values[name] = float(squared ** 0.5)
    return values


def parameter_update_norms(model, before: dict[str, torch.Tensor]) -> dict[str, float]:
    values = {}
    for name, parameter in model.named_parameters():
        if name in before:
            values[name] = float((parameter.detach() - before[name]).float().norm())
    return {
        "total": float(sum(value * value for value in values.values()) ** 0.5),
    }


def evaluate_native(output: dict, batch: dict) -> dict:
    probability = output["rendered_instance_group_probability"].detach().float().cpu().numpy()[0]
    labels = batch["instance_label_output"].detach().cpu().numpy()[0]
    predictions, scores, pred_ids, ground_truth, gt_ids = [], [], [], [], []
    for view in range(probability.shape[1]):
        masks, view_scores = masks_from_group_probs(
            probability[:, view, 0], void_channel=probability.shape[0] - 1, min_mask_area=1
        )
        image_id = f"fixed_target_{view}"
        predictions.extend(masks)
        scores.extend(view_scores)
        pred_ids.extend([image_id] * len(masks))
        gt = gt_masks_from_instance_map(labels[view], min_mask_area=1)
        ground_truth.extend(gt)
        gt_ids.extend([image_id] * len(gt))
    ap = instance_ap(
        predictions, scores, ground_truth,
        thresholds=(0.25, 0.5, 0.75), vectorized=True,
        pred_image_ids=pred_ids, gt_image_ids=gt_ids,
    )
    diag = _mask_diagnostics(predictions, ground_truth, pred_ids, gt_ids)
    return {
        "ap25": float(ap["ap_25"]),
        "ap50": float(ap["ap_50"]),
        "ap75": float(ap["ap_75"]),
        "best_iou": float(diag["mean_best_gt_iou"]),
        "recall25": float(diag["recall_iou25"]),
        "recall50": float(diag["recall_iou50"]),
        "recall75": float(diag["recall_iou75"]),
        "pred_gt": float(len(predictions) / max(1, len(ground_truth))),
        "prediction_count": len(predictions),
        "gt_count": len(ground_truth),
    }


def output_snapshot(output: dict, batch: dict) -> dict:
    metrics = evaluate_native(output, batch)
    values = {
        key: float(output[key].detach().float().mean())
        for key in ("loss", "loss_rgb", "loss_instance_group", "loss_instance_metric", "psnr", "ssim", "lpips")
        if key in output and torch.is_tensor(output[key]) and bool(torch.isfinite(output[key]).all())
    }
    anchor = getattr(output, "unit_3d_anchor_delta", None)
    if anchor is None:
        anchor = output.get("unit_3d_anchor_delta")
    if torch.is_tensor(anchor):
        values.update({
            "anchor_delta_mean": float(anchor.detach().float().abs().mean()),
            "anchor_delta_max": float(anchor.detach().float().abs().max()),
        })
    for key in ("unit_3d_anchor_centers_world", "unit_3d_anchor_centers_normalized"):
        if key in output:
            tensor = output[key].detach().float()
            values[f"{key}_mean"] = float(tensor.mean())
            values[f"{key}_std"] = float(tensor.std())
            values[f"{key}_min"] = float(tensor.min())
            values[f"{key}_max"] = float(tensor.max())
    if "unit_3d_anchor_fallback_mask" in output:
        values["opacity_fallback_ratio"] = float(output["unit_3d_anchor_fallback_mask"].float().mean())
    values.update(metrics)
    return values


def build_runtime(config_name: str, root: Path):
    opt = dataclasses.replace(config_defaults[config_name])
    opt.resume = str(PARENT)
    opt.workspace = str(root)
    opt.num_workers = 0
    opt.tsh_ddp8 = False
    opt.use_wandb = False
    opt.eval_before_training = False
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(int(accelerator.local_process_index))
    model = model_registry[opt.model_type](opt).to(accelerator.device)
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        raise RuntimeError(f"strict parent restore failed for {config_name}")
    configure_joint_formation_trainability(model, opt)
    optimizer = setup_optimizer(opt, model, accelerator, 0)
    scheduler = setup_scheduler(opt, optimizer, 200, accelerator, 0)
    return opt, accelerator, model, optimizer, scheduler


def save_trainable_checkpoint(model, output: Path, step: int, metadata: dict) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"model_step_{step:06d}_trainable.safetensors"
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    save_file(state, str(path))
    (output / f"metadata_step_{step:06d}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def set_training_schedule(model, opt, effective_step: int) -> None:
    """Mirror the formal trainer's per-step schedule assignments exactly."""
    model.set_token_eru_step(effective_step)
    model.set_token_eru_dino_metric_step(effective_step)
    if hasattr(model, "compute_instance_group_lambda_eff"):
        model.instance_group_lambda_eff = model.compute_instance_group_lambda_eff(
            effective_step, opt
        )
    if hasattr(model, "compute_teacher_lambda_eff"):
        model.teacher_lambda_eff = model.compute_teacher_lambda_eff(
            effective_step, opt
        )
    if hasattr(model, "compute_instance_stage_eff"):
        model.instance_stage_eff = model.compute_instance_stage_eff(
            effective_step, opt
        )
    if bool(getattr(opt, "abs_true_shared_units", False)) and hasattr(
        model, "compute_tsh_effs"
    ):
        model.tsh_instance_loss_weight_eff = 1.0
        model.tsh_unit_grad_eff = 1.0
    if bool(getattr(opt, "abs_true_shared_units", False)) and hasattr(
        model, "compute_tsh_mbm_u2r_eff"
    ):
        model.tsh_mbm_u2r_eff = model.compute_tsh_mbm_u2r_eff(
            effective_step, opt
        )


def instance_gradient_probe(model, opt, batch) -> dict[str, float | bool]:
    """Fresh instance-only backward used solely for the gradient audit."""
    model.zero_grad(set_to_none=True)
    output = model(batch, compute_quality_metrics=False)
    render_gs = getattr(model, "_tsh_last_render_gs", None)
    anchor_delta = output.get("unit_3d_anchor_delta")
    if render_gs is None or not torch.is_tensor(render_gs):
        raise RuntimeError("instance gradient probe did not expose live render GS")
    render_gs.retain_grad()
    if torch.is_tensor(anchor_delta):
        anchor_delta.retain_grad()
    instance_loss = output["loss_instance_group"] * float(
        getattr(opt, "tsh_lambda_instance", 0.05)
    ) * float(getattr(model, "tsh_instance_loss_weight_eff", 1.0))
    if not bool(torch.isfinite(instance_loss)):
        raise RuntimeError("non-finite instance-only probe loss")
    instance_loss.backward()
    def norm_or_zero(value):
        if value is None:
            return 0.0
        return float(value.detach().float().norm())
    anchor_module = getattr(model, "token_eru_3d_anchor", None)
    first_grad = None
    last_grad = None
    if anchor_module is not None:
        first_grad = anchor_module.position_mlp[0].weight.grad
        last_grad = anchor_module.position_mlp[2].weight.grad
    gs_grad = render_gs.grad
    result = {
        "instance_loss": float(instance_loss.detach()),
        "anchor_first_linear_grad_norm": norm_or_zero(first_grad),
        "anchor_last_linear_grad_norm": norm_or_zero(last_grad),
        "gs_xyz_grad_norm": norm_or_zero(None if gs_grad is None else gs_grad[..., :3]),
        "gs_opacity_grad_norm": norm_or_zero(None if gs_grad is None else gs_grad[..., 3:4]),
        "gs_sh_grad_norm": norm_or_zero(None if gs_grad is None else gs_grad[..., 11:]),
        "anchor_delta_grad_norm": norm_or_zero(None if not torch.is_tensor(anchor_delta) else anchor_delta.grad),
        "anchor_last_linear_grad_nonzero": bool(last_grad is not None and torch.any(last_grad != 0)),
        "gs_xyz_grad_nonzero": bool(gs_grad is not None and torch.any(gs_grad[..., :3] != 0)),
        "gs_opacity_grad_nonzero": bool(gs_grad is not None and torch.any(gs_grad[..., 3:4] != 0)),
        "gs_sh_grad_nonzero": bool(gs_grad is not None and torch.any(gs_grad[..., 11:] != 0)),
    }
    model.zero_grad(set_to_none=True)
    return result


def run_one(config_name: str, root: Path, batch: dict, steps: int) -> tuple[dict, object, object, object, dict[str, torch.Tensor]]:
    opt, accelerator, model, optimizer, scheduler = build_runtime(config_name, root)
    set_paired_seed(42)
    model.train()
    set_training_schedule(model, opt, 960)
    initial_output = model(batch, compute_quality_metrics=False)
    milestones = {0: output_snapshot(initial_output, batch)}
    identity = {
        key: initial_output[key].detach().cpu().clone()
        for key in ("gaussians", "images_pred", "unit_logits", "rendered_instance_group_probability")
        if key in initial_output and torch.is_tensor(initial_output[key])
    }
    before_params = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    reports = {0: milestones[0]}
    saved = {}
    for local_step in range(1, steps + 1):
        effective = 960 + local_step
        set_training_schedule(model, opt, effective)
        optimizer.zero_grad(set_to_none=True)
        gradient_probe = None
        if local_step in (1, 2, 100) and getattr(model, "token_eru_3d_anchor", None) is not None:
            gradient_probe = instance_gradient_probe(model, opt, batch)
        output = model(batch, compute_quality_metrics=False)
        loss = output["loss"]
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss at {config_name} step {local_step}")
        loss.backward()
        group_grads = group_grad_norms(optimizer)
        pre_clip = float(torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        ))
        post_clip = group_grad_norms(optimizer)
        optimizer.step()
        scheduler.step()
        update = parameter_update_norms(model, before_params)
        before_params = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if local_step in MILESTONES:
            model.eval()
            with torch.no_grad():
                eval_output = model(batch, compute_quality_metrics=False)
            item = output_snapshot(eval_output, batch)
            for key in (
                "loss", "loss_rgb", "loss_instance_group", "loss_instance_metric",
                "loss_instance_metric_weighted", "loss_instance_group_weighted",
            ):
                if key in output and torch.is_tensor(output[key]):
                    item[f"train_{key}"] = float(output[key].detach().float())
            if gradient_probe is not None:
                item["instance_gradient_probe"] = gradient_probe
            item.update({
                "local_step": local_step,
                "effective_step": effective,
                "group_grad_norm": group_grads,
                "post_clip_group_grad_norm": post_clip,
                "pre_clip_grad_norm": pre_clip,
                "post_clip_grad_norm": float(sum(v * v for v in post_clip.values()) ** 0.5),
                "update_norm": update,
                "trainable_parameter_hash": trainable_parameter_hash(model),
            })
            reports[local_step] = item
            if local_step in (25, 50, 100):
                saved[local_step] = save_trainable_checkpoint(
                    model, root / "checkpoints", local_step,
                    {"config": config_name, "parent_path": str(PARENT), "parent_sha256": PARENT_SHA,
                     "stage_local_step": local_step, "effective_optimizer_step": effective,
                     "batch_hash": batch_hash(batch), "trainable_parameter_hash": trainable_parameter_hash(model)},
                )
            model.train()
    model.eval()
    with torch.no_grad():
        final_output = model(batch, compute_quality_metrics=False)
    final_eval = {
        key: final_output[key].detach().cpu().clone()
        for key in ("gaussians", "images_pred", "unit_logits", "rendered_instance_group_probability", "unit_3d_anchor_delta")
        if key in final_output and torch.is_tensor(final_output[key])
    }
    return reports, model, optimizer, saved, identity, final_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps != 100:
        raise ValueError("this audit is fixed at 100 optimizer steps")
    if not PARENT.is_file() or sha256_file(PARENT) != PARENT_SHA:
        raise RuntimeError("J2 local250 parent is missing or SHA256 mismatched")
    root = args.output_dir.resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing non-empty fixed-batch workspace: {root}")
    root.mkdir(parents=True, exist_ok=True)
    base_opt = dataclasses.replace(config_defaults[A0])
    base_opt.resume = str(PARENT)
    base_opt.workspace = str(root / "batch_loader")
    base_opt.num_workers = 0
    base_opt.tsh_ddp8 = False
    base_opt.use_wandb = False
    base_opt.eval_before_training = False
    accelerator = Accelerator(mixed_precision="no", dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True))
    if torch.cuda.is_available():
        torch.cuda.set_device(int(accelerator.local_process_index))
    loader, _, _, _ = get_multi_dataloader(base_opt, accelerator)
    batch = move(next(iter(loader)), accelerator.device)
    if batch["input"].shape[1] != 15:
        raise RuntimeError("fixed batch is not 8 context + 7 target")
    fingerprint = {"batch_hash": batch_hash(batch), "scene": str(batch.get("scene_name", "unknown")), "frame_ids": batch.get("frame_ids").detach().cpu().tolist() if torch.is_tensor(batch.get("frame_ids")) else None}
    (root / "batch_fingerprint.json").write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    reports = {}
    model_a0 = None
    model_a1 = None
    trainable_counts = {}
    identity_outputs = {}
    for label, config_name in (("control", A0), ("anchor", A1)):
        item, model, optimizer, saved, identity, final_eval = run_one(config_name, root / label, batch, args.steps)
        reports[label] = item
        identity_outputs[label] = identity
        trainable_counts[label] = {
            "tensor_count": sum(p.requires_grad for p in model.parameters()),
            "numel": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        if label == "anchor":
            anchor_final_eval = final_eval
            anchor_saved = saved
        if label == "control":
            model_a0 = model
            del model_a0
        else:
            model_a1 = model
            anchor = getattr(model, "token_eru_3d_anchor", None)
            if anchor is None:
                raise RuntimeError("A1 did not construct Unit3DAnchor")
            del model_a1
        del model, optimizer
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    identity_diffs = {}
    for key in sorted(set(identity_outputs["control"]) & set(identity_outputs["anchor"])):
        identity_diffs[key] = float((identity_outputs["control"][key].float() - identity_outputs["anchor"][key].float()).abs().max())
    # Independent model-only restore of A1 step100.  The frozen parent is
    # loaded first, then the trainable-only state is overlaid strictly on its
    # trainable namespace; no optimizer or scheduler is restored.
    restore_opt, restore_accelerator, restore_model, _, _ = build_runtime(
        A1, root / "independent_restore"
    )
    trainable_state = load_file(str(anchor_saved[100]), device="cpu")
    expected_trainable = {
        name for name, parameter in restore_model.named_parameters() if parameter.requires_grad
    }
    if set(trainable_state) != expected_trainable:
        raise RuntimeError("step100 trainable-only checkpoint key set mismatch")
    restore_model.load_state_dict(trainable_state, strict=False)
    restore_model.set_token_eru_step(1060)
    restore_model.set_token_eru_dino_metric_step(1060)
    restore_model.eval()
    with torch.no_grad():
        restored_output = restore_model(batch, compute_quality_metrics=False)
    restore_diffs = {
        key: float((restored_output[key].detach().cpu().float() - value.float()).abs().max())
        for key, value in anchor_final_eval.items()
        if key in restored_output and torch.is_tensor(restored_output[key])
    }
    independent_restore_max_diff = max(restore_diffs.values(), default=0.0)
    if independent_restore_max_diff > 1e-6:
        raise RuntimeError(f"A1 step100 independent restore mismatch: {restore_diffs}")

    payload = {
        "parent_path": str(PARENT),
        "parent_sha256": PARENT_SHA,
        "config_control": A0,
        "config_anchor": A1,
        "batch_fingerprint": fingerprint,
        "milestones": MILESTONES,
        "reports": reports,
        "control_trainable_tensor_count": trainable_counts["control"]["tensor_count"],
        "anchor_trainable_tensor_count": trainable_counts["anchor"]["tensor_count"],
        "control_trainable_numel": trainable_counts["control"]["numel"],
        "anchor_trainable_numel": trainable_counts["anchor"]["numel"],
        "step0_control_treatment_identity_max_diff": max(identity_diffs.values(), default=0.0),
        "step0_control_treatment_identity_diffs": identity_diffs,
        "step100_independent_restore_max_diff": independent_restore_max_diff,
        "step100_independent_restore_diffs": restore_diffs,
        "formal_training_started": False,
        "formal_evaluation_started": False,
    }
    (root / "fixed_batch_report.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
