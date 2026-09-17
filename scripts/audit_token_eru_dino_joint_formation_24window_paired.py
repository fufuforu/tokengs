"""Strict held-out validation audit for TokenGS-ERU JointFormation.

This is an isolated evaluator.  It uses the existing validation DataLoader and
native-query AP implementation, and never performs an optimizer step.  The
formal mode refuses to run until the stage-local step 710 checkpoint and its
completion marker are present and no matching training job is active.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_gsi_v2_short355 import _input_fingerprint, _scene_name  # noqa: E402
from scripts.eval_instance_lsm_protocol import _mask_diagnostics  # noqa: E402
from scripts.eval_token_eru_dino_joint_formation import (  # noqa: E402
    _candidate_bundle,
    load_joint_formation_checkpoint,
    _move,
    _reconstruction_metrics,
    _sha256,
    ap101_upper_bound,
)
from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import (  # noqa: E402
    _load_cached_gsplat_extension,
    configure_joint_formation_trainability,
    load_model_checkpoint,
)
from tokengs.utils.instance_ap import (  # noqa: E402
    instance_ap,
    make_target_view_image_id,
    masks_from_group_probs,
)
from tokengs.utils.metrics import MetricsCalculator  # noqa: E402


JOINT_CONFIG = (
    "semantic_v6_absolute_units_true_shared_token_eru1_"
    "dino_metric_joint_formation_v1_ddp8"
)
PARENT_CONFIG = (
    "semantic_v6_absolute_units_true_shared_token_eru1_"
    "scene_hungarian_short200_ddp8"
)
SOURCE = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_"
    "scene_hungarian_resume200_to500_persist_ddp8/checkpoints/"
    "model_step_000500.safetensors"
)
SOURCE_SHA256 = "cca23e2d24af7a374be6a62f2cf4026645afdf0df49686722cb5277c9e479209"
MANIFEST_SHA256 = "90a7a66ccbb10b7ca6334b262b70f94e9c3d8f33d2f60609dbd54be7bc7fde7d"
EXPECTED_SCENES = [
    "scene0059_00", "scene0072_02", "scene0132_01", "scene0472_01",
    "scene0559_01", "scene0568_02", "scene0615_00", "scene0695_00",
]
EXPECTED_WINDOW_COUNTS = {
    "scene0072_02": 7, "scene0059_00": 7, "scene0132_01": 5,
    "scene0615_00": 1, "scene0568_02": 1, "scene0559_01": 1,
    "scene0472_01": 1, "scene0695_00": 1,
}
JOINT_ROOT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_"
    "dino_metric_joint_formation_v1_ddp8"
)
FORMAL_OUTPUT = ROOT / (
    "workspace/token_eru_dino_joint_formation_v1_24window_paired_eval"
)


def _write_json(path: Path, payload: Any) -> None:
    def default(value: Any):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        raise TypeError(type(value).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=default), encoding="utf-8")


def _tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _field_hash(batch: dict[str, Any], key: str) -> str:
    value = batch[key]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return _tensor_hash(value)


def _frame_values(batch: dict[str, Any]) -> list[int]:
    values = batch["frame_ids"]
    if torch.is_tensor(values):
        return [int(value) for value in values[0].detach().cpu().tolist()]
    return [int(value) for value in values[0]]


def _window_fingerprint(batch: dict[str, Any], index: int) -> dict[str, Any]:
    frames = _frame_values(batch)
    if len(frames) != 15:
        raise RuntimeError(f"expected 15 frame ids, got {len(frames)}")
    scene = _scene_name(batch)
    sample = batch["sample_id"][0] if isinstance(batch["sample_id"], (list, tuple)) else str(batch["sample_id"])
    return {
        "order": int(index),
        "scene_id": str(scene),
        "sample_id": str(sample),
        "context_frame_ids": frames[:8],
        "target_frame_ids": frames[8:],
        "context_rgb_sha256": _field_hash(batch, "images_input"),
        "target_rgb_sha256": _field_hash(batch, "images_output"),
        "target_gt_instance_sha256": _field_hash(batch, "instance_label_output"),
        "target_camera_sha256": hashlib.sha256(
            bytes.fromhex(_field_hash(batch, "cam_view"))
            + bytes.fromhex(_field_hash(batch, "intrinsics"))
        ).hexdigest(),
        "context_shape": list(batch["images_input"].shape),
        "target_shape": list(batch["images_output"].shape),
    }


def _manifest_path(opt) -> Path:
    path = Path(str(opt.dataset_kwargs["small_manifest_path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validation_runtime(output_dir: Path):
    base = config_defaults[JOINT_CONFIG]
    opt = dataclasses.replace(
        base,
        resume=str(SOURCE),
        workspace=str(output_dir / "runtime"),
        evaluating=True,
        num_workers=0,
        batch_size=1,
        max_eval_iters=24,
        tsh_ddp8=False,
        eval_before_training=False,
        use_wandb=False,
        token_eru_dino_eval_mode="query",
    )
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )
    _, test_loader, _, test_dataset = get_multi_dataloader(opt, accelerator)
    if len(test_dataset) != 24 or len(test_loader) != 24:
        raise RuntimeError(
            f"held-out validation protocol requires 24 windows, got dataset={len(test_dataset)} loader={len(test_loader)}"
        )
    return opt, accelerator, test_loader, test_dataset


def _audit_manifest(opt, test_dataset) -> dict[str, Any]:
    manifest = _manifest_path(opt)
    observed_sha = _sha256(manifest)
    if observed_sha != MANIFEST_SHA256:
        raise RuntimeError(f"validation manifest SHA mismatch: {observed_sha}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    train_scenes = sorted(str(value) for value in payload["train_scenes"])
    validation_scenes = sorted(str(value) for value in payload["validation_scenes"])
    samples = payload["validation_samples"]
    counts = defaultdict(int)
    for sample in samples:
        counts[str(sample["scene"])] += 1
    if validation_scenes != sorted(EXPECTED_SCENES):
        raise RuntimeError(f"validation scene set changed: {validation_scenes}")
    if len(samples) != 24 or dict(sorted(counts.items())) != dict(sorted(EXPECTED_WINDOW_COUNTS.items())):
        raise RuntimeError(f"validation window distribution changed: {dict(counts)}")
    if set(train_scenes) & set(validation_scenes):
        raise RuntimeError("train/validation scene intersection is non-empty")
    underlying = test_dataset.datasets[0].dataset
    if getattr(underlying, "split", None) != "validation":
        raise RuntimeError(f"test dataset is not validation split: {getattr(underlying, 'split', None)}")
    return {
        "path": str(manifest),
        "sha256": observed_sha,
        "train_scene_count": len(train_scenes),
        "validation_scene_count": len(validation_scenes),
        "validation_window_count": len(samples),
        "validation_scenes": validation_scenes,
        "validation_window_counts": dict(sorted(counts.items())),
        "train_validation_intersection": [],
        "context_views": 8,
        "target_views": 7,
        "split": "validation",
        "sampling": "manifest order, no resampling",
    }


def _strict_full_model_restore(model: torch.nn.Module, checkpoint: Path) -> dict[str, Any]:
    state = load_file(str(checkpoint), device="cpu")
    expected = model.state_dict()
    actual_keys = set(state)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    shape_mismatch = {
        key: {"expected": list(expected[key].shape), "actual": list(state[key].shape)}
        for key in sorted(expected_keys & actual_keys)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    }
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            f"strict JointFormation restore failed: missing={missing[:8]} unexpected={unexpected[:8]} shape={shape_mismatch}"
        )
    with torch.no_grad():
        for key, target in expected.items():
            target.copy_(state[key].to(device=target.device, dtype=target.dtype))
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError(f"non-finite parameter after restore: {checkpoint}")
    return {
        "path": str(checkpoint),
        "sha256": _sha256(checkpoint),
        "state_dict_key_count": len(state),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch": shape_mismatch,
        "strict": True,
        "external_dino_in_checkpoint": any("_dino_model" in key.lower() for key in state),
    }


def _strict_joint_model_only_restore(
    model: torch.nn.Module,
    checkpoint: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    """Strictly overlay a formal JointFormation model-only checkpoint.

    Formal JointFormation model-only saves contain every parameter that was
    trainable in the trainer, including valid legacy compatibility modules
    that are frozen by the isolated evaluator.  They are not unexpected model
    keys: the authoritative invariant is that every current JointFormation
    trainable key is present, and that every saved key maps to this model with
    the exact shape.  Frozen parent keys remain supplied by the protected
    ERU@500 restore.
    """
    state = load_file(str(checkpoint), device="cpu")
    runtime = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    raw_unexpected = set(state) - set(runtime)
    legacy_prefixes = (
        "gaussian_feature_head.",
        "prompt_semantic_adapter.",
        "semantic_token_adapter.",
        "log_temperature",
    )
    ignored_legacy = sorted(
        key for key in raw_unexpected
        if key.startswith(legacy_prefixes)
    )
    unexpected = sorted(raw_unexpected - set(ignored_legacy))
    shape_mismatch = {
        key: {"expected": list(runtime[key].shape), "actual": list(state[key].shape)}
        for key in sorted(set(state) & set(runtime))
        if tuple(runtime[key].shape) != tuple(state[key].shape)
    }
    required_trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    missing_trainable = sorted(required_trainable - set(state))
    if unexpected or shape_mismatch or missing_trainable:
        raise RuntimeError(
            "strict JointFormation model-only restore failed: "
            f"missing_trainable={missing_trainable[:8]} unexpected={unexpected[:8]} "
            f"shape_mismatch={shape_mismatch}"
        )
    with torch.no_grad():
        for key, value in state.items():
            if key in ignored_legacy:
                continue
            runtime[key].copy_(value.to(device=runtime[key].device, dtype=runtime[key].dtype))
    observed_sha = _sha256(checkpoint)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    declared_sha = metadata.get("checkpoint_sha256") or metadata.get("sha256")
    if declared_sha is not None and str(declared_sha) != observed_sha:
        raise RuntimeError(f"checkpoint SHA mismatch: metadata={declared_sha} observed={observed_sha}")
    if any("_dino_model" in key.lower() for key in state):
        raise RuntimeError("frozen DINO parameters must not be present in Joint checkpoint")
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("non-finite parameter after JointFormation restore")
    return {
        "path": str(checkpoint),
        "sha256": observed_sha,
        "state_dict_key_count": len(state),
        "required_trainable_key_count": len(required_trainable),
        "missing_keys": missing_trainable,
        "unexpected_keys": unexpected,
        "shape_mismatch": shape_mismatch,
        "ignored_legacy_compatibility_keys": ignored_legacy,
        "valid_compatibility_keys": sorted(set(state) - required_trainable),
        "strict": True,
        "external_dino_in_checkpoint": False,
    }


def _metadata_for(checkpoint: Path, step: int) -> tuple[Path, dict[str, Any]]:
    metadata_path = checkpoint.parent / f"metadata_step_{step:06d}.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(payload.get("optimizer_step", -1)) != int(step):
        raise RuntimeError(f"metadata optimizer_step mismatch: {metadata_path}")
    marker = checkpoint.parent / f"step_{step:06d}.complete"
    if not marker.is_file():
        raise RuntimeError(f"checkpoint completion marker missing: {marker}")
    return metadata_path, payload


def _active_joint_job() -> list[str]:
    try:
        jobs = subprocess.check_output(
            ["squeue", "-h", "-u", os.environ.get("USER", "mawb"), "-o", "%A"],
            text=True,
        ).split()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    matches = []
    for job in jobs:
        try:
            detail = subprocess.check_output(["scontrol", "show", "job", job], text=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        if "joint_formation" in detail or "train_joint_formation" in detail:
            matches.append(job)
    return matches


def _build_model(config_name: str, output_dir: Path, checkpoint: Path | None = None):
    base = config_defaults[config_name]
    resume = str(SOURCE if checkpoint is None else checkpoint)
    opt = dataclasses.replace(
        base,
        resume=resume,
        workspace=str(output_dir / "runtime"),
        evaluating=True,
        num_workers=0,
        batch_size=1,
        tsh_ddp8=False,
        eval_before_training=False,
        use_wandb=False,
    )
    accelerator = Accelerator(mixed_precision="no")
    model = model_registry[opt.model_type](opt)
    load_model_checkpoint(opt, model, accelerator, 0)
    if config_name == JOINT_CONFIG:
        configure_joint_formation_trainability(model, opt)
    model.to(accelerator.device).eval()
    return opt, accelerator, model


def _native_window_metrics(
    out: dict[str, Any],
    batch: dict[str, Any],
    scene_id: str,
    context_frame_ids: list[int],
    target_frame_ids: list[int],
) -> dict[str, Any]:
    probability = out["rendered_instance_group_probability"][0].detach().float().cpu().numpy()[:, :, 0]
    labels = batch["instance_label_output"][0].detach().long().cpu().numpy()
    all_predictions: list[np.ndarray] = []
    all_scores: list[float] = []
    all_pred_ids: list[str] = []
    all_ground_truth: list[np.ndarray] = []
    all_gt_ids: list[str] = []
    per_view = []
    for view in range(probability.shape[1]):
        image_id = make_target_view_image_id(
            scene_id=scene_id,
            context_frame_ids=context_frame_ids,
            target_frame_ids=target_frame_ids,
            target_view_index=view,
        )
        # The shared candidate diagnostic stores a numeric target_view_id by
        # parsing its image-id suffix.  Keep that diagnostic field view-local
        # while using the full per-target-view ID for all AP bookkeeping IDs
        # below.
        bundle = _candidate_bundle(
            probability[:, view], labels[view], f"target_view:{view}"
        )
        bundle["image_id"] = image_id
        per_view.append({key: value for key, value in bundle.items() if key not in {"masks", "scores", "gt_masks"}})
        all_predictions.extend(bundle["masks"])
        all_scores.extend(bundle["scores"])
        all_pred_ids.extend([image_id] * len(bundle["masks"]))
        all_ground_truth.extend(bundle["gt_masks"])
        all_gt_ids.extend([image_id] * len(bundle["gt_masks"]))
    pooled_ap = instance_ap(
        all_predictions, all_scores, all_ground_truth,
        thresholds=(0.25, 0.5, 0.75), vectorized=True,
        pred_image_ids=all_pred_ids, gt_image_ids=all_gt_ids,
    )
    pooled_diag = _mask_diagnostics(
        all_predictions, all_ground_truth, all_pred_ids, all_gt_ids,
        thresholds=(0.25, 0.5, 0.75),
    )
    assignment = out.get("gsi_v2_instance_assignment_probabilities")
    if assignment is None:
        assignment = out["instance_group_probabilities"]
    assignment = assignment.detach().float()
    usage = assignment[..., :-1].mean(dim=(0, 1, 2))
    entropy = (-(assignment * assignment.clamp_min(1e-8).log()).sum(-1)).mean()
    nonempty_query_ids = {
        int(record["query_id"])
        for record in [record for row in per_view for record in row["records"]]
    }
    void_ratio = float(probability[-1].mean())
    invariant = all(
        -1e-9 <= row["ap50"] <= 1.0 + 1e-9
        and -1e-9 <= row["max_achievable_recall50"] <= 1.0 + 1e-9
        and row["tp_iou50_count"] <= row["gt_count"]
        and row["ap50"] <= ap101_upper_bound(row["max_achievable_recall50"]) + 1e-9
        for row in per_view
    )
    return {
        "per_view": per_view,
        "macro": {
            "ap": float(np.mean([(row["ap25"] + row["ap50"] + row["ap75"]) / 3.0 for row in per_view])),
            "ap25": float(np.mean([row["ap25"] for row in per_view])),
            "ap50": float(np.mean([row["ap50"] for row in per_view])),
            "ap75": float(np.mean([row["ap75"] for row in per_view])),
            "recall25": float(np.mean([row["recall25"] for row in per_view])),
            "recall50": float(np.mean([row["recall50"] for row in per_view])),
            "recall75": float(np.mean([row["recall75"] for row in per_view])),
            "best_gt_iou": float(np.mean([row["best_gt_iou"] for row in per_view])),
        },
        "pooled": {
            "ap": float(pooled_ap["ap_mean"]),
            "ap25": float(pooled_ap["ap_25"]),
            "ap50": float(pooled_ap["ap_50"]),
            "ap75": float(pooled_ap["ap_75"]),
            "recall25": float(pooled_diag["recall_iou25"]),
            "recall50": float(pooled_diag["recall_iou50"]),
            "recall75": float(pooled_diag["recall_iou75"]),
            "best_gt_iou": float(pooled_diag["mean_best_gt_iou"]),
        },
        "prediction_count": len(all_predictions),
        "gt_count": len(all_ground_truth),
        "pred_gt": float(len(all_predictions) / max(1, len(all_ground_truth))),
        "nonempty_query_count": len(nonempty_query_ids),
        "active_query_count_mass_gt_0.001": int((usage > 0.001).sum()),
        "effective_query_count": float(torch.exp(entropy).item()),
        "void_ratio": void_ratio,
        "duplicate_prediction_count": int(sum(row["duplicate_count"] for row in per_view)),
        "invariant_valid": bool(invariant),
        "_predictions": all_predictions,
        "_scores": all_scores,
        "_prediction_ids": all_pred_ids,
        "_ground_truth": all_ground_truth,
        "_gt_ids": all_gt_ids,
    }


def _compact_window(
    out: dict[str, Any],
    batch: dict[str, Any],
    fp: dict[str, Any],
    metrics: MetricsCalculator,
    native: dict[str, Any],
) -> dict[str, Any]:
    reconstruction = _reconstruction_metrics(out, batch, metrics)
    numeric_finite = bool(
        reconstruction["finite"]
        and torch.isfinite(out["rendered_instance_group_probability"]).all()
        and torch.isfinite(out["images_pred"]).all()
    )
    return {
        "fingerprint": fp,
        "native": native,
        "reconstruction": reconstruction,
        "numeric_finite": numeric_finite,
        "ap101_invariant_valid": bool(native["invariant_valid"]),
        # Compatibility alias for older display consumers.  Selection and
        # summary state use the two explicit fields above.
        "finite": numeric_finite,
    }


def _parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        value = parameter.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _mean_std(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {"mean": float(values.mean()) if values.size else 0.0, "std": float(values.std()) if values.size else 0.0}


def _aggregate(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    pooled: dict[str, Any],
) -> dict[str, Any]:
    window_rows = [
        {
            "scene_id": record["fingerprint"]["scene_id"],
            **record["native"]["macro"],
            "pred_gt": record["native"]["pred_gt"],
            "nonempty_queries": record["native"]["nonempty_query_count"],
            "active_queries": record["native"]["active_query_count_mass_gt_0.001"],
            "effective_queries": record["native"]["effective_query_count"],
            "void_ratio": record["native"]["void_ratio"],
            "psnr": record["reconstruction"]["mean_psnr"],
            "ssim": record["reconstruction"]["mean_ssim"],
            "lpips": record["reconstruction"]["mean_lpips"],
        }
        for record in records
    ]
    scene_rows = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        grouped[row["scene_id"]].append(row)
    scenes_to_aggregate = (
        manifest["validation_scenes"]
        if len(records) == int(manifest["validation_window_count"])
        else sorted(grouped)
    )
    for scene in scenes_to_aggregate:
        rows = grouped.get(scene, [])
        if not rows:
            raise RuntimeError(f"missing validation scene in evaluated records: {scene}")
        scene_rows.append({"scene_id": scene, **{
            key: float(np.mean([row[key] for row in rows]))
            for key in window_rows[0]
            if key != "scene_id"
        }})
    return {
        "window_macro": {key: _mean_std(window_rows, key) for key in (
            "ap", "ap25", "ap50", "ap75", "best_gt_iou", "recall25", "recall50", "recall75",
            "pred_gt", "nonempty_queries", "active_queries", "effective_queries", "void_ratio",
            "psnr", "ssim", "lpips",
        )},
        "scene_macro": {key: _mean_std(scene_rows, key) for key in (
            "ap", "ap25", "ap50", "ap75", "best_gt_iou", "recall25", "recall50", "recall75",
            "pred_gt", "nonempty_queries", "active_queries", "effective_queries", "void_ratio",
            "psnr", "ssim", "lpips",
        )},
        "pooled": pooled,
        "window_count": len(window_rows),
        "scene_count": len(scene_rows),
        "window_rows": window_rows,
        "scene_rows": scene_rows,
    }


def _evaluate_model(
    config_name: str,
    checkpoint: Path,
    step: int,
    output_dir: Path,
    reference_fingerprints: list[dict[str, Any]],
    max_windows: int,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    metadata_path, metadata = _metadata_for(checkpoint, step)
    if config_name == JOINT_CONFIG:
        opt, accelerator, model = _build_model(config_name, output_dir / f"joint_{step:06d}")
        # Formal model-only checkpoints intentionally contain the complete
        # trainable JointFormation namespace.  Frozen parent namespaces are
        # restored first by load_model_checkpoint; this helper then performs
        # the strict trainable-only overlay without accepting missing or
        # unexpected JointFormation keys.
        restore = _strict_joint_model_only_restore(model, checkpoint, metadata_path)
        schedule = model.set_token_eru_step(step)
        dino_schedule = model.set_token_eru_dino_metric_step(step)
    else:
        opt, accelerator, model = _build_model(config_name, output_dir / "eru_000500", checkpoint)
        restore = {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "strict_parent_loader": bool(getattr(model, "_token_eru_loaded_from_checkpoint", False)),
            "missing_keys": [], "unexpected_keys": [], "shape_mismatch": {}, "strict": True,
        }
        schedule = model.set_token_eru_step(500)
        dino_schedule = {"dino_gate": 0.0, "metric_loss_weight": 0.0}
    if restore["sha256"] != (_sha256(checkpoint)):
        raise RuntimeError("checkpoint hash changed during load")
    if config_name == JOINT_CONFIG and restore["external_dino_in_checkpoint"]:
        raise RuntimeError("DINO backbone is unexpectedly present in Joint checkpoint")
    loader_opt, loader_accelerator, loader, test_dataset = _validation_runtime(output_dir / f"loader_{step:06d}")
    del loader_opt, loader_accelerator
    records = []
    pooled_predictions: list[np.ndarray] = []
    pooled_scores: list[float] = []
    pooled_prediction_ids: list[str] = []
    pooled_ground_truth: list[np.ndarray] = []
    pooled_gt_ids: list[str] = []
    metrics = MetricsCalculator(device=accelerator.device)
    parameter_hash_before = _parameter_hash(model)
    for index, raw_batch in enumerate(loader):
        if index >= max_windows:
            break
        fp = _window_fingerprint(raw_batch, index)
        if fp != reference_fingerprints[index]:
            raise RuntimeError(f"validation fingerprint mismatch at window {index}")
        batch = _move(raw_batch, accelerator.device)
        with torch.inference_mode():
            with torch.autocast(device_type=accelerator.device.type, enabled=False):
                out = model(batch, compute_quality_metrics=False)
        fp["input_fingerprint"] = _input_fingerprint(batch, fp["scene_id"])
        native = _native_window_metrics(
            out,
            batch,
            fp["scene_id"],
            fp["context_frame_ids"],
            fp["target_frame_ids"],
        )
        pooled_predictions.extend(native.pop("_predictions"))
        pooled_scores.extend(native.pop("_scores"))
        pooled_prediction_ids.extend(native.pop("_prediction_ids"))
        pooled_ground_truth.extend(native.pop("_ground_truth"))
        pooled_gt_ids.extend(native.pop("_gt_ids"))
        record = _compact_window(out, batch, fp, metrics, native)
        record["schedule"] = {**schedule, **dino_schedule}
        record["checkpoint_step"] = int(step)
        record["checkpoint_metadata_path"] = str(metadata_path)
        records.append(record)
        del out, batch
    parameter_hash_after = _parameter_hash(model)
    if parameter_hash_before != parameter_hash_after:
        raise RuntimeError("model parameters changed during evaluation")
    if len(records) != max_windows:
        raise RuntimeError(f"evaluated {len(records)} windows, expected {max_windows}")
    pooled_ap = instance_ap(
        pooled_predictions,
        pooled_scores,
        pooled_ground_truth,
        thresholds=(0.25, 0.5, 0.75),
        vectorized=True,
        pred_image_ids=pooled_prediction_ids,
        gt_image_ids=pooled_gt_ids,
    )
    pooled_diag = _mask_diagnostics(
        pooled_predictions,
        pooled_ground_truth,
        pooled_prediction_ids,
        pooled_gt_ids,
        thresholds=(0.25, 0.5, 0.75),
    )
    pooled_summary = {
        "ap": float(pooled_ap["ap_mean"]),
        "ap25": float(pooled_ap["ap_25"]),
        "ap50": float(pooled_ap["ap_50"]),
        "ap75": float(pooled_ap["ap_75"]),
        "recall25": float(pooled_diag["recall_iou25"]),
        "recall50": float(pooled_diag["recall_iou50"]),
        "recall75": float(pooled_diag["recall_iou75"]),
        "best_gt_iou": float(pooled_diag["mean_best_gt_iou"]),
        "prediction_count": len(pooled_predictions),
        "gt_count": len(pooled_ground_truth),
        "pred_gt": float(len(pooled_predictions) / max(1, len(pooled_ground_truth))),
    }
    aggregate = _aggregate(records, manifest, pooled_summary)
    result = {
        "model": "eru_500" if config_name == PARENT_CONFIG else f"joint_{step:06d}",
        "checkpoint": restore,
        "metadata": metadata,
        "schedule": {**schedule, **dino_schedule},
        "parameter_hash_before": parameter_hash_before,
        "parameter_hash_after": parameter_hash_after,
        "records": records,
        "aggregate": aggregate,
        "numeric_finite": all(record["numeric_finite"] for record in records),
        "ap101_invariant_valid": all(record["ap101_invariant_valid"] for record in records),
        # Compatibility alias; numeric finiteness is intentionally independent
        # of the AP protocol invariant.
        "all_finite": all(record["numeric_finite"] for record in records),
        "native_query_only": True,
        "p_u_used": False,
        "oracle_used": False,
        "ttt_used": False,
    }
    del model, accelerator, metrics, loader, test_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _checkpoint_table() -> dict[str, Any]:
    rows = {}
    for step in (50, 100, 250, 500, 710):
        checkpoint = JOINT_ROOT / "checkpoints" / f"model_step_{step:06d}.safetensors"
        metadata = JOINT_ROOT / "checkpoints" / f"metadata_step_{step:06d}.json"
        marker = JOINT_ROOT / "checkpoints" / f"step_{step:06d}.complete"
        rows[str(step)] = {
            "checkpoint": str(checkpoint), "metadata": str(metadata), "complete_marker": str(marker),
            "checkpoint_exists": checkpoint.is_file(), "metadata_exists": metadata.is_file(),
            "complete_marker_exists": marker.is_file(),
            "checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
        }
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(FORMAL_OUTPUT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--joint-steps", nargs="+", type=int, default=[50, 100, 250, 500, 710])
    return parser


def main() -> None:
    args = _parser().parse_args()
    # The training entry point preloads the validated local extension.  This
    # isolated evaluator must do the same before the first renderer call so a
    # missing import cannot trigger an unsupported JIT build on the node.
    _load_cached_gsplat_extension()
    output = Path(args.output_dir).resolve()
    if output.exists():
        # Slurm must have the log directory before launching the node task.
        # Those launcher-created logs are not evaluation results and are safe
        # to reuse; every other pre-existing entry still makes the run refuse
        # to avoid overwriting a partial or completed evaluation.
        unexpected = [entry for entry in output.iterdir() if entry.name != "logs"]
        if unexpected:
            raise RuntimeError(
                f"refusing output directory with existing evaluation entries: {unexpected}"
            )
    output.mkdir(parents=True, exist_ok=True)
    if not SOURCE.is_file() or _sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("protected ERU@500 source checkpoint SHA mismatch")
    if args.smoke:
        max_windows = 1 if args.max_windows is None else int(args.max_windows)
        if max_windows != 1:
            raise ValueError("smoke is fixed to exactly one held-out window")
        steps = [int(args.joint_steps[0])]
    else:
        max_windows = 24 if args.max_windows is None else int(args.max_windows)
        if max_windows != 24:
            raise ValueError("formal evaluation is fixed to all 24 validation windows")
        if _active_joint_job():
            raise RuntimeError("formal evaluation refused while JointFormation training job is active")
        table = _checkpoint_table()
        for step in (50, 100, 250, 500, 710):
            row = table[str(step)]
            if not (row["checkpoint_exists"] and row["metadata_exists"] and row["complete_marker_exists"]):
                raise RuntimeError(f"formal evaluation refused: incomplete Joint checkpoint step {step}: {row}")
        if not (JOINT_ROOT / "status" / "COMPLETE").is_file():
            raise RuntimeError("formal evaluation refused: JointFormation status/COMPLETE is missing")
        steps = [50, 100, 250, 500, 710]
    # Build the validation runtime once for the authoritative manifest and
    # reference order; every model gets a fresh identical validation loader.
    opt, accelerator, _, test_dataset = _validation_runtime(output / "protocol")
    manifest_info = _audit_manifest(opt, test_dataset)
    _, _, reference_loader, _ = _validation_runtime(output / "reference")
    reference_fingerprints = []
    for index, batch in enumerate(reference_loader):
        if index >= max_windows:
            break
        reference_fingerprints.append(_window_fingerprint(batch, index))
    if len(reference_fingerprints) != max_windows:
        raise RuntimeError("reference validation fingerprint count mismatch")
    protocol = {
        "manifest": manifest_info,
        "window_fingerprints": reference_fingerprints,
        "window_distribution_fixed": EXPECTED_WINDOW_COUNTS,
        "window_macro_equal_weight": True,
        "scene_macro_equal_weight_after_window_mean": True,
        "pooled_all_window_predictions_and_gt": True,
        "native_query_only": True,
        "metric_cluster_diagnostic_only": True,
        "p_u_used": False,
        "oracle_used": False,
        "ttt_used": False,
        "precision": "fp32",
        "max_predictions_per_image": 100,
        "min_mask_area": 1,
        "ap_interpolation": "101-point recall interpolation",
        "instance_ap_image_identity": "per_target_view_v1",
        "cross_target_view_matching": False,
    }
    protocol_path = output / "protocol_fingerprints.json"
    if protocol_path.is_file():
        existing_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing_protocol != protocol:
            raise RuntimeError(
                f"existing protocol fingerprint differs; refusing overwrite: {protocol_path}"
            )
    else:
        _write_json(protocol_path, protocol)
    del opt, accelerator, test_dataset, reference_loader
    results = {}
    parent_meta = SOURCE.parent / "metadata_step_000500.json"
    if not parent_meta.is_file():
        raise FileNotFoundError(parent_meta)
    results["eru_000500"] = _evaluate_model(
        PARENT_CONFIG, SOURCE, 500, output, reference_fingerprints, max_windows, manifest_info
    )
    for step in steps:
        checkpoint = JOINT_ROOT / "checkpoints" / f"model_step_{step:06d}.safetensors"
        results[f"joint_{step:06d}"] = _evaluate_model(
            JOINT_CONFIG, checkpoint, step, output, reference_fingerprints, max_windows, manifest_info
        )
    _write_json(output / "checkpoint_metadata.json", {
        key: {
            "checkpoint": value["checkpoint"], "metadata": value["metadata"], "schedule": value["schedule"],
            "strict": value["checkpoint"].get("strict", False),
        }
        for key, value in results.items()
    })
    _write_json(output / "per_window_metrics.json", {
        key: {"records": value["records"], "aggregate": value["aggregate"]}
        for key, value in results.items()
    })
    _write_json(output / "per_scene_metrics.json", {
        key: {
            "summary": value["aggregate"]["scene_macro"],
            "scenes": value["aggregate"]["scene_rows"],
        }
        for key, value in results.items()
    })
    baseline = results["eru_000500"]["aggregate"]["scene_macro"]["ap50"]["mean"]
    comparisons = {}
    baseline_scene_rows = {
        row["scene_id"]: row for row in results["eru_000500"]["aggregate"]["scene_rows"]
    }
    for key, value in results.items():
        scene = value["aggregate"]["scene_macro"]
        scene_deltas = {}
        for row in value["aggregate"]["scene_rows"]:
            base_row = baseline_scene_rows[row["scene_id"]]
            scene_deltas[row["scene_id"]] = {
                metric: float(row[metric] - base_row[metric])
                for metric in ("ap25", "ap50", "ap75", "best_gt_iou", "recall50", "pred_gt", "psnr", "nonempty_queries", "effective_queries")
            }
        comparisons[key] = {
            "delta_scene_macro_ap50": scene["ap50"]["mean"] - baseline,
            "scene_macro": scene,
            "scene_deltas_vs_eru_500": scene_deltas,
            "ap50_not_lower_scene_count": int(sum(
                row["ap50"] >= baseline_scene_rows[row["scene_id"]]["ap50"]
                for row in value["aggregate"]["scene_rows"]
            )),
            "query_collapse": (
                scene["nonempty_queries"]["mean"] < 0.70 * results["eru_000500"]["aggregate"]["scene_macro"]["nonempty_queries"]["mean"]
                or scene["effective_queries"]["mean"] < 0.60 * results["eru_000500"]["aggregate"]["scene_macro"]["effective_queries"]["mean"]
                or scene["void_ratio"]["mean"] > results["eru_000500"]["aggregate"]["scene_macro"]["void_ratio"]["mean"] + 0.10
                or scene["pred_gt"]["mean"] < 0.60 or scene["pred_gt"]["mean"] > 2.50
            ),
        }
    _write_json(output / "comparison_vs_eru500.json", comparisons)
    candidates = []
    baseline_aggregate = results["eru_000500"]["aggregate"]
    for step in steps:
        key = f"joint_{step:06d}"
        value = results[key]
        aggregate = value["aggregate"]
        scene = aggregate["scene_macro"]
        comparison = comparisons[key]
        eligible = bool(
            value["numeric_finite"]
            and value["ap101_invariant_valid"]
            and comparison["delta_scene_macro_ap50"] >= 0.02
            and aggregate["pooled"]["ap50"] >= baseline_aggregate["pooled"]["ap50"]
            and scene["best_gt_iou"]["mean"] >= baseline_aggregate["scene_macro"]["best_gt_iou"]["mean"]
            and scene["recall50"]["mean"] >= baseline_aggregate["scene_macro"]["recall50"]["mean"]
            and comparison["ap50_not_lower_scene_count"] >= 5
            and scene["psnr"]["mean"] >= baseline_aggregate["scene_macro"]["psnr"]["mean"] - 0.20
            and not comparison["query_collapse"]
        )
        candidates.append({"step": step, "eligible": eligible, "scene_macro_ap50": scene["ap50"]["mean"], "pooled_ap50": aggregate["pooled"]["ap50"], "best_gt_iou": scene["best_gt_iou"]["mean"], "recall50": scene["recall50"]["mean"], "psnr": scene["psnr"]["mean"], "ap50_not_lower_scene_count": comparison["ap50_not_lower_scene_count"], "query_collapse": comparison["query_collapse"]})
    eligible = [row for row in candidates if row["eligible"]]
    eligible.sort(key=lambda row: (-row["scene_macro_ap50"], -row["pooled_ap50"], -row["best_gt_iou"], -row["recall50"], row["step"]))
    selection = {"candidates": candidates, "best_checkpoint": eligible[0] if eligible else None, "ready_for_lsm40": bool(eligible)}
    _write_json(output / "checkpoint_selection.json", selection)
    summary = {
        "training_completed_at_eval_start": bool((JOINT_ROOT / "status" / "COMPLETE").is_file()),
        "checkpoint_table": _checkpoint_table(),
        "results": {key: {
            "aggregate": value["aggregate"],
            "numeric_finite": value["numeric_finite"],
            "ap101_invariant_valid": value["ap101_invariant_valid"],
            "all_finite": value["all_finite"],
            "parameter_hash_before": value["parameter_hash_before"],
            "parameter_hash_after": value["parameter_hash_after"],
            "native_query_only": value["native_query_only"], "p_u_used": value["p_u_used"],
            "oracle_used": value["oracle_used"], "ttt_used": value["ttt_used"],
        } for key, value in results.items()},
        "checkpoint_selection": selection,
        "formal_24_window_evaluation": not args.smoke,
    }
    _write_json(output / "summary.json", summary)
    (output / "status").mkdir(exist_ok=True)
    (output / "status" / ("SMOKE_COMPLETE" if args.smoke else "COMPLETE")).write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
