"""Read-only JointFormation fixed-batch/one-scene evaluator.

This adapter intentionally does not replace either historical evaluator.  It
loads the persisted ERU parent, overlays a JointFormation trainable-only
snapshot when requested, and applies the same native mask conversion and AP
helpers used by the paired evaluator.  The command is diagnostic-only; it
never performs an optimizer step and never consumes GT in model inference.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_gsi_v2_short355 import _input_fingerprint, _scene_name  # noqa: E402
from scripts.eval_instance_lsm_protocol import _mask_diagnostics  # noqa: E402
from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import (  # noqa: E402
    configure_joint_formation_trainability,
    load_model_checkpoint,
    setup_optimizer,
)
from tokengs.utils.instance_ap import (  # noqa: E402
    gt_masks_from_instance_map,
    instance_ap,
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
FROZEN_CONFIG = (
    "semantic_v6_absolute_units_true_shared_token_eru1_"
    "dino_metric_treatment_resume500_to700_ddp8"
)
SOURCE = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_"
    "scene_hungarian_resume200_to500_persist_ddp8/checkpoints/"
    "model_step_000500.safetensors"
)
SOURCE_SHA256 = "cca23e2d24af7a374be6a62f2cf4026645afdf0df49686722cb5277c9e479209"
FIXED_BATCH_HASH = "ce2227280c967941a53e39b819d5370d47f3cfab09325a285c1767634dff32ec"


def ap101_upper_bound(max_recall: float, eps: float = 1e-9) -> float:
    """Upper bound induced by the existing 101-point AP interpolation."""
    if not math.isfinite(max_recall):
        raise ValueError("max_recall must be finite")
    if max_recall < -eps or max_recall > 1.0 + eps:
        raise ValueError("max_recall must be in [0, 1]")
    recall = min(1.0, max(0.0, max_recall))
    valid_points = math.floor(100.0 * recall + eps) + 1
    return valid_points / 101.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(*named: tuple[str, torch.Tensor | None]) -> str:
    digest = hashlib.sha256()
    for name, value in named:
        digest.update(name.encode())
        if value is None:
            digest.update(b"<none>")
            continue
        value = value.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _move(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _batch_hash(batch: dict[str, Any]) -> str:
    # This is intentionally byte-for-byte the hash used by the saved v5/v1
    # fixed-batch audits (key name plus contiguous tensor bytes; no shape or
    # dtype prefix).
    digest = hashlib.sha256()
    for key, value in sorted(batch.items()):
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _build_runtime(config_name: str, output_dir: Path):
    base = config_defaults[config_name]
    opt = dataclasses.replace(
        base,
        resume=str(SOURCE),
        workspace=str(output_dir),
        num_workers=0,
        batch_size=1,
        tsh_ddp8=False,
        eval_before_training=False,
        use_wandb=False,
        max_eval_iters=1,
        token_eru_dino_eval_mode="metric_cluster",
    )
    # Match the saved fixed-batch audit's loader construction exactly.  The
    # seed must be set before get_multi_dataloader creates its sampler.
    torch.manual_seed(int(opt.seed))
    random.seed(int(opt.seed))
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    loader, _, _, _ = get_multi_dataloader(opt, accelerator)
    model = model_registry[opt.model_type](opt)
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        raise RuntimeError(f"{config_name} did not restore the ERU parent namespaces")
    # Keep the exact sampler construction/Accelerate preparation used by the
    # saved fixed-batch audit.  In particular, preparing the loader after
    # setup_optimizer consumes the same seedable-sampler state.
    optimizer = setup_optimizer(opt, model, accelerator, 0)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model = accelerator.unwrap_model(model)
    model.eval()
    return opt, accelerator, model, loader


def _build_model_only(config_name: str, output_dir: Path):
    """Build a checkpoint model without creating a second sampler/worker set."""
    base = config_defaults[config_name]
    opt = dataclasses.replace(
        base,
        resume=str(SOURCE),
        workspace=str(output_dir),
        num_workers=0,
        batch_size=1,
        tsh_ddp8=False,
        eval_before_training=False,
        use_wandb=False,
        max_eval_iters=1,
    )
    torch.manual_seed(int(opt.seed))
    random.seed(int(opt.seed))
    accelerator = Accelerator(mixed_precision="no")
    model = model_registry[opt.model_type](opt)
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        raise RuntimeError(f"{config_name} did not restore the ERU parent namespaces")
    if getattr(opt, "token_eru_dino_metric_joint_formation", False):
        configure_joint_formation_trainability(model, opt)
    model.to(accelerator.device).eval()
    return opt, accelerator, model


def _overlay_trainable_snapshot(
    model: torch.nn.Module,
    snapshot_path: Path,
    *,
    expected_trainable: set[str] | None = None,
) -> dict[str, Any]:
    if not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)
    state = load_file(str(snapshot_path), device="cpu")
    model_state = model.state_dict()
    named_parameters = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    runtime_state = {**named_parameters, **named_buffers}
    missing_model_keys = sorted(set(state) - set(runtime_state))
    shape_mismatch = {
        key: {"expected": list(runtime_state[key].shape), "actual": list(value.shape)}
        for key, value in state.items()
        if key in runtime_state and tuple(value.shape) != tuple(runtime_state[key].shape)
    }
    if missing_model_keys or shape_mismatch:
        raise RuntimeError(
            f"snapshot is not compatible: unexpected={missing_model_keys}, "
            f"shape_mismatch={shape_mismatch}"
        )
    if expected_trainable is not None:
        unexpected = sorted(set(state) - expected_trainable)
        missing = sorted(expected_trainable - set(state))
        if unexpected or missing:
            raise RuntimeError(
                f"trainable-only snapshot mismatch: missing={missing[:8]}, "
                f"unexpected={unexpected[:8]}"
            )
    with torch.no_grad():
        for key, value in state.items():
            target = runtime_state[key]
            target.copy_(value.to(device=target.device, dtype=target.dtype))
            if not torch.equal(target.detach().cpu(), value):
                raise RuntimeError(f"snapshot overlay verification failed for {key}")
    return {
        "path": str(snapshot_path.resolve()),
        "sha256": _sha256(snapshot_path),
        "key_count": len(state),
        "keys": sorted(state),
        "strict_overlay": True,
    }


def load_joint_formation_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    metadata_path: str,
) -> dict:
    """Strictly restore a JointFormation snapshot and return verified metadata.

    The fixed-batch artifact is deliberately trainable-only.  Its omitted
    frozen parameters must already have been restored from the protected
    ERU@500 parent before this function is called.
    """
    checkpoint = Path(checkpoint_path).resolve()
    metadata = Path(metadata_path).resolve()
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    restored = _overlay_trainable_snapshot(
        model, checkpoint, expected_trainable=trainable
    )
    observed_sha = _sha256(checkpoint)
    declared_sha = payload.get("checkpoint_sha256") or payload.get("sha256")
    if declared_sha is not None and str(declared_sha) != observed_sha:
        raise RuntimeError(
            f"checkpoint SHA mismatch: metadata={declared_sha}, observed={observed_sha}"
        )
    if any("_dino_model" in key.lower() for key in restored["keys"]):
        raise RuntimeError("external frozen DINO parameters must not be in checkpoint")
    if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise FloatingPointError("non-finite parameter after JointFormation restore")
    payload = dict(payload)
    payload["checkpoint_sha256_verified"] = observed_sha
    payload["strict_restore"] = restored
    payload["external_dino_in_checkpoint"] = False
    return payload


def _candidate_bundle(probability: np.ndarray, labels: np.ndarray, image_id: str):
    masks, scores = masks_from_group_probs(
        probability, void_channel=probability.shape[0] - 1, min_mask_area=1
    )
    group_ids = np.argmax(probability, axis=0)
    candidates = []
    for query_id in range(probability.shape[0] - 1):
        mask = group_ids == query_id
        if int(mask.sum()) < 1:
            continue
        confidence = float(np.max(probability, axis=0)[mask].mean())
        candidates.append((confidence, query_id, mask))
    candidates.sort(key=lambda item: -item[0])
    gt_ids = [int(value) for value in np.unique(labels) if int(value) not in (0, 255, -1)]
    gt_masks = [labels == gt_id for gt_id in gt_ids]
    unmatched = set(range(len(gt_masks)))
    records = []
    tp_count = 0
    duplicate_count = 0
    for confidence, query_id, mask in candidates:
        ious = np.asarray(
            [
                float(np.logical_and(mask, gt).sum())
                / max(1.0, float(np.logical_or(mask, gt).sum()))
                for gt in gt_masks
            ],
            dtype=np.float32,
        )
        best_any = int(ious.argmax()) if len(ious) else None
        best_any_iou = float(ious[best_any]) if best_any is not None else 0.0
        eligible = sorted(unmatched)
        best_unmatched = max(eligible, key=lambda index: float(ious[index])) if eligible else None
        if best_unmatched is not None and float(ious[best_unmatched]) >= 0.5:
            matched = best_unmatched
            unmatched.remove(matched)
            status = "TP"
            tp_count += 1
        else:
            matched = None
            status = "DUPLICATE" if best_any_iou >= 0.5 else "FP"
            if status == "DUPLICATE":
                duplicate_count += 1
        records.append({
            "query_id": int(query_id),
            "target_view_id": int(image_id.rsplit(":", 1)[-1]),
            "confidence": confidence,
            "matched_gt_id": None if matched is None else int(gt_ids[matched]),
            "matched_iou": 0.0 if matched is None else float(ious[matched]),
            "best_any_gt_id": None if best_any is None else int(gt_ids[best_any]),
            "best_any_iou": best_any_iou,
            "status_at_iou50": status,
            "mask_area": int(mask.sum()),
        })
    ap = instance_ap(
        masks, scores, gt_masks, thresholds=(0.25, 0.5, 0.75), vectorized=True
    )
    best = []
    for gt in gt_masks:
        best.append(
            max(
                [
                    float(np.logical_and(mask, gt).sum())
                    / max(1.0, float(np.logical_or(mask, gt).sum()))
                    for mask in masks
                ]
                or [0.0]
            )
        )
    return {
        "image_id": image_id,
        "gt_count": len(gt_masks),
        "prediction_count": len(masks),
        "tp_iou50_count": tp_count,
        "fp_count": sum(record["status_at_iou50"] == "FP" for record in records),
        "duplicate_count": duplicate_count,
        "max_achievable_recall50": float(tp_count / max(1, len(gt_masks))),
        "ap25": float(ap["ap_25"]), "ap50": float(ap["ap_50"]), "ap75": float(ap["ap_75"]),
        "best_gt_iou": float(np.mean(best) if best else 0.0),
        "recall25": float(np.mean(np.asarray(best) >= 0.25) if best else 0.0),
        "recall50": float(np.mean(np.asarray(best) >= 0.50) if best else 0.0),
        "recall75": float(np.mean(np.asarray(best) >= 0.75) if best else 0.0),
        "records": records,
        "masks": masks,
        "scores": scores,
        "gt_masks": gt_masks,
        "gt_ids": gt_ids,
        "ap101_upper_bound": ap101_upper_bound(
            float(tp_count / max(1, len(gt_masks)))
        ),
    }


def _native_metrics(probability: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    views = []
    predictions, scores, prediction_ids = [], [], []
    ground_truth, gt_ids = [], []
    for view in range(probability.shape[1]):
        bundle = _candidate_bundle(probability[:, view], labels[view], f"fixed:target:{view}")
        views.append({key: value for key, value in bundle.items() if key not in {"masks", "scores", "gt_masks"}})
        predictions.extend(bundle["masks"]); scores.extend(bundle["scores"])
        prediction_ids.extend([bundle["image_id"]] * len(bundle["masks"]))
        ground_truth.extend(bundle["gt_masks"]); gt_ids.extend([bundle["image_id"]] * len(bundle["gt_masks"]))
    pooled_ap = instance_ap(
        predictions, scores, ground_truth, thresholds=(0.25, 0.5, 0.75),
        vectorized=True, pred_image_ids=prediction_ids, gt_image_ids=gt_ids,
    )
    pooled_diag = _mask_diagnostics(predictions, ground_truth, prediction_ids, gt_ids)
    return {
        "per_view": views,
        "macro": {
            "ap25": float(np.mean([row["ap25"] for row in views])),
            "ap50": float(np.mean([row["ap50"] for row in views])),
            "ap75": float(np.mean([row["ap75"] for row in views])),
            "recall50": float(np.mean([row["recall50"] for row in views])),
            "best_gt_iou": float(np.mean([row["best_gt_iou"] for row in views])),
        },
        "pooled": {
            "ap25": float(pooled_ap["ap_25"]), "ap50": float(pooled_ap["ap_50"]),
            "ap75": float(pooled_ap["ap_75"]),
            "recall50": float(pooled_diag["recall_iou50"]),
            "best_gt_iou": float(pooled_diag["mean_best_gt_iou"]),
        },
        "prediction_count": len(predictions),
        "gt_count": len(ground_truth),
        "pred_gt": float(len(predictions) / max(1, len(ground_truth))),
        "records": [record for view in views for record in view["records"]],
        "invariant_valid": all(
            row["ap50"]
            <= row["ap101_upper_bound"] + 1e-8
            and row["ap50"] >= 0.0
            and row["ap50"] <= 1.0
            and row["max_achievable_recall50"] >= 0.0
            and row["max_achievable_recall50"] <= 1.0
            and row["tp_iou50_count"] <= row["gt_count"]
            for row in views
        ),
        "raw_masks": predictions,
    }


def _reconstruction_metrics(out: dict[str, Any], batch: dict[str, Any], metrics: MetricsCalculator) -> dict[str, Any]:
    pred = out["images_pred"].float().clamp(0, 1)
    target = batch["images_output"].float().clamp(0, 1)
    mse = (pred - target).square().mean(dim=(2, 3, 4))[0]
    psnr = metrics.calculate_psnr(pred, target, reduction="none")[0]
    ssim = metrics.calculate_ssim(pred, target, reduction="none")[0]
    lpips = metrics.calculate_lpips(pred, target, reduction="none")[0]
    result = {
        "mse_per_view": [float(x) for x in mse],
        "psnr_per_view": [float(x) for x in psnr],
        "ssim_per_view": [float(x) for x in ssim],
        "lpips_per_view": [float(x) for x in lpips],
        "mean_psnr": float(psnr.mean()), "pooled_psnr": float(-10.0 * np.log10(max(float(mse.mean()), 1e-8))),
        "mean_ssim": float(ssim.mean()), "pooled_ssim": float(ssim.mean()),
        "mean_lpips": float(lpips.mean()), "pooled_lpips": float(lpips.mean()),
        "finite": bool(torch.isfinite(pred).all() and torch.isfinite(target).all()),
    }
    for key in ("alphas_pred", "depths_pred"):
        value = out.get(key)
        result[f"{key}_finite"] = value is None or bool(torch.isfinite(value).all())
    return result


def _forward_record(
    model: torch.nn.Module,
    batch: dict[str, Any],
    step: int,
    metrics: MetricsCalculator,
    *,
    schedule_step: int | None = None,
) -> dict[str, Any]:
    model.eval()
    native_schedule_step = int(step if schedule_step is None else schedule_step)
    if hasattr(model, "set_token_eru_step"):
        model.set_token_eru_step(native_schedule_step)
    if hasattr(model, "set_token_eru_dino_metric_step"):
        model.set_token_eru_dino_metric_step(native_schedule_step)
    with torch.inference_mode():
        with torch.autocast(device_type=batch["input"].device.type, enabled=False):
            out = model(batch, compute_quality_metrics=False)
    probability = out["rendered_instance_group_probability"][0].float().cpu().numpy()[:, :, 0]
    labels = batch["instance_label_output"][0].long().cpu().numpy()
    native = _native_metrics(probability, labels)
    result = {
        "step": int(step), "native": native,
        "schedule_step": native_schedule_step,
        "reconstruction": _reconstruction_metrics(out, batch, metrics),
        "native_probability": out["rendered_instance_group_probability"].detach().cpu(),
        "native_unit_logits": out["unit_logits"].detach().cpu(),
        "gaussians": out["gaussians"].detach().cpu(),
        "images_pred": out["images_pred"].detach().cpu(),
        "records": native["records"],
        "dino_gate": float(out.get("dino_gate", torch.tensor(0.0)).detach().cpu()),
        "loss_instance_group": float(out.get("loss_instance_group", torch.tensor(0.0)).detach().cpu()),
    }
    cluster = out.get("metric_cluster_output")
    if cluster is not None:
        result["metric_cluster_diagnostic"] = {
            "cluster_count": [int(x) for x in cluster.cluster_count],
            "rendered_masks_shape": list(cluster.rendered_masks.shape),
        }
    return result


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    delta = (a.float() - b.float()).abs()
    return {"max": float(delta.max()), "mean": float(delta.mean())}


def evaluate_joint_formation(
    model: torch.nn.Module,
    dataloader,
    *,
    checkpoint_step: int,
    output_dir: str,
    max_predictions_per_image: int = 100,
) -> dict:
    """Run the unchanged native-query evaluation protocol on supplied data."""
    if int(max_predictions_per_image) != 100:
        raise ValueError("JointFormation protocol fixes max_predictions_per_image=100")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    metrics = MetricsCalculator(device=device)
    batch = _move(next(iter(dataloader)), device)
    return _forward_record(model, batch, int(checkpoint_step), metrics)


def _write_json_safe(path: Path, payload: dict[str, Any]) -> None:
    def default(value):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(payload, indent=2, default=default), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty diagnostic directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if _sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("protected ERU@500 SHA256 mismatch")

    # One DataLoader and one batch are shared by all model variants.
    _, accelerator, joint_model, loader = _build_runtime(JOINT_CONFIG, output / "joint_runtime")
    batch = _move(next(iter(loader)), accelerator.device)
    if _batch_hash(batch) != FIXED_BATCH_HASH:
        raise RuntimeError(f"fixed batch mismatch: {_batch_hash(batch)}")
    if batch["input"].shape[1] != 15 or batch["instance_label_output"].shape[1] != 7:
        raise RuntimeError("diagnostic batch is not 8 context + 7 target views")
    metrics = MetricsCalculator(device=accelerator.device)

    # Parent native ERU and Joint step0 use the same loaded parent state.
    _, parent_acc, parent_model = _build_model_only(PARENT_CONFIG, output / "parent_runtime")
    parent_model.set_token_eru_step(500)
    with torch.inference_mode():
        parent_out = parent_model(batch, compute_quality_metrics=False)
    parent_record = _forward_record(parent_model, batch, 500, metrics)
    joint_model.set_token_eru_step(0)
    joint_model.set_token_eru_dino_metric_step(0)
    step0 = _forward_record(joint_model, batch, 0, metrics)

    # Create a temporary metadata sidecar for the fixed-batch trainable-only
    # snapshot.  The historical workspace is not modified.
    sidecar = output / "joint_step100_metadata.json"
    step100_path = ROOT / "workspace/token_eru_dino_joint_formation_v1_fixed_batch/step_100_trainable.safetensors"
    sidecar.write_text(json.dumps({"optimizer_step": 100, "checkpoint_sha256": _sha256(step100_path)}), encoding="utf-8")
    trainable = {name for name, p in joint_model.named_parameters() if p.requires_grad}
    restore = load_joint_formation_checkpoint(joint_model, str(step100_path), str(sidecar))
    step100 = _forward_record(joint_model, batch, 100, metrics)

    # Frozen Stage-M v5 is a read-only comparison using its saved trainable
    # snapshot over the same ERU@500 parent and the same batch.
    frozen_model = None
    frozen_record = None
    frozen_snapshot = ROOT / "workspace/token_eru_dino_metric_learning_audit_v5/step_100_trainable.safetensors"
    if frozen_snapshot.is_file():
        _, _, frozen_model = _build_model_only(FROZEN_CONFIG, output / "frozen_runtime")
        frozen_trainable = {name for name, p in frozen_model.named_parameters() if p.requires_grad}
        _overlay_trainable_snapshot(frozen_model, frozen_snapshot, expected_trainable=frozen_trainable)
        # Stage-M checkpoints use a parent-relative local step, while the
        # native gate schedule is expressed in the original global step.
        frozen_record = _forward_record(
            frozen_model, batch, 100, metrics, schedule_step=600
        )

    identity = {
        key: _diff(parent_record[key], step0[key])
        for key in ("gaussians", "images_pred", "native_probability", "native_unit_logits")
    }
    reconst0, reconst100 = step0["reconstruction"], step100["reconstruction"]
    gs0, gs100 = step0["gaussians"], step100["gaussians"]
    geometry_changes = {
        "xyz": _diff(gs0[..., 0:3], gs100[..., 0:3]),
        "opacity": _diff(gs0[..., 3:4], gs100[..., 3:4]),
        "scale": _diff(gs0[..., 4:7], gs100[..., 4:7]),
        "rotation": _diff(gs0[..., 7:11], gs100[..., 7:11]),
        "sh": _diff(gs0[..., 11:14], gs100[..., 11:14]),
    }
    final = {
        "protocol": {
            "fixed_batch_hash": FIXED_BATCH_HASH,
            "scene": str(batch["scene_name"][0] if isinstance(batch["scene_name"], (list, tuple)) else batch["scene_name"]),
            "context_views": 8, "target_views": 7, "eval_precision": "fp32",
            "native_output": True, "metric_cluster_diagnostic_only": True,
            "p_u_used": False, "target_rgb_to_dino": False,
            "input_fingerprint": _input_fingerprint(batch, _scene_name(batch)),
        },
        "source": {"path": str(SOURCE.resolve()), "sha256": _sha256(SOURCE), "strict": True},
        "parent": parent_record,
        "joint_step0": step0,
        "joint_step100": step100,
        "frozen_stage_m_v5_step100": frozen_record,
        "joint_step100_restore": restore,
        "step0_parent_identity": identity,
        "step100_vs_step0": {
            "rgb": _diff(step0["images_pred"], step100["images_pred"]),
            "native_masks": _diff(step0["native_probability"], step100["native_probability"]),
            "native_logits": _diff(step0["native_unit_logits"], step100["native_unit_logits"]),
            "geometry_changes": geometry_changes,
            "psnr_delta": reconst100["mean_psnr"] - reconst0["mean_psnr"],
            "ssim_delta": reconst100["mean_ssim"] - reconst0["mean_ssim"],
            "lpips_delta": reconst100["mean_lpips"] - reconst0["mean_lpips"],
        },
        "checks": {
            "ap_field_is_pooled": True,
            "per_view_ap_recall_invariant_valid": bool(step0["native"]["invariant_valid"] and step100["native"]["invariant_valid"]),
            "native_cluster_output_mixed": False,
            "all_finite": bool(
                step0["reconstruction"]["finite"] and step100["reconstruction"]["finite"]
            ),
            "one_scene_smoke_requested": bool(args.smoke),
        },
    }
    # Raw native masks and per-prediction matching records are intentionally
    # separate from JSON so the JSON remains reviewable.
    torch.save(
        {
            "step0_native_probability": step0["native_probability"],
            "step100_native_probability": step100["native_probability"],
            "gt_instance_maps": batch["instance_label_output"].detach().cpu(),
            "step0_records": step0["records"],
            "step100_records": step100["records"],
        },
        output / "native_masks_and_matches.pt",
    )
    for name in ("parent", "joint_step0", "joint_step100"):
        record = final[name]
        record["native"].pop("raw_masks", None)
        record.pop("native_probability", None)
        record.pop("native_unit_logits", None)
        record.pop("gaussians", None)
        record.pop("images_pred", None)
        record.pop("records", None)
        record.pop("raw_masks", None)
    if final["frozen_stage_m_v5_step100"] is not None:
        final["frozen_stage_m_v5_step100"]["native"].pop("raw_masks", None)
        for key in ("native_probability", "native_unit_logits", "gaussians", "images_pred", "records"):
            final["frozen_stage_m_v5_step100"].pop(key, None)
    _write_json_safe(output / "fixed_batch_reevaluation.json", final)
    print(json.dumps({
        "output": str(output),
        "step0_pooled_ap50": step0["native"]["pooled"]["ap50"],
        "step100_pooled_ap50": step100["native"]["pooled"]["ap50"],
        "step100_macro_ap50": step100["native"]["macro"]["ap50"],
        "step100_pooled_recall50": step100["native"]["pooled"]["recall50"],
        "step100_best_iou": step100["native"]["pooled"]["best_gt_iou"],
        "per_view_invariant": final["checks"]["per_view_ap_recall_invariant_valid"],
    }, indent=2))


if __name__ == "__main__":
    main()
