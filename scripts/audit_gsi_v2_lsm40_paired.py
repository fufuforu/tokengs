"""Strict paired LSM-40 evaluation for GSI-v2 Joint@200.

This adapter deliberately reuses the LSM evaluator's manifest validation,
mask conversion, pruning, diagnostics, and AP implementation.  It adds only
the model-loading paths needed for GSI-v2 and records input fingerprints so
the historical Both@1420 result can be compared on exactly the same data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _240_path_remap import remap_opt, remap_path  # noqa: E402
from scripts.eval_instance_lsm_protocol import (  # noqa: E402
    _audit_lsm_manifest,
    _load_checkpoint_arch,
    _mask_diagnostics,
    _prune_inactive_groups,
)
from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.utils.instance_ap import (  # noqa: E402
    gt_masks_from_instance_map,
    instance_ap,
    make_target_view_image_id,
    masks_from_group_probs,
)
from tokengs.utils.metrics import MetricsCalculator  # noqa: E402


EXPECTED_COMMIT = "b975eb7ea236ef891721072dfb40c104c4c71f91"
MANIFEST = ROOT / "data/scannet_prompt/lsm_instance_eval_manifest.json"
JOINT = ROOT / (
    "workspace/gsi_v2_joint_scannet_short355_ddp8/checkpoints/"
    "model_step_000200.safetensors"
)
R1 = ROOT / (
    "workspace/gsi_v2_recon_scannet_adapt_ddp8/checkpoints/"
    "model_step_000250.safetensors"
)
BOTH = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8/"
    "checkpoints/model_step_001420.safetensors"
)


class _LocalAccelerator:
    is_main_process = True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(*named_tensors: tuple[str, torch.Tensor | None]) -> str:
    digest = hashlib.sha256()
    for name, value in named_tensors:
        digest.update(name.encode("utf-8"))
        if value is None:
            digest.update(b"<none>")
            continue
        value = value.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _metadata(checkpoint: Path, expected_step: int) -> dict[str, object]:
    path = checkpoint.parent / f"metadata_step_{expected_step:06d}.json"
    if not path.is_file():
        raise FileNotFoundError(f"required checkpoint metadata is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = payload.get("optimizer_step", payload.get("step"))
    if observed != expected_step:
        raise RuntimeError(
            f"metadata step mismatch for {checkpoint}: expected {expected_step}, "
            f"observed {observed} in {path}"
        )
    return {
        "path": str(path.resolve()),
        "optimizer_step": int(observed),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "payload": payload,
    }


def _manifest_fingerprints(data, scene: str, scene_entry: dict[str, object]) -> dict[str, object]:
    return {
        "scene_id": scene,
        "context_frame_ids": [int(x) for x in scene_entry["context_raw_frame_ids"]],
        "target_frame_ids": [int(x) for x in scene_entry["test_raw_frame_ids"]],
        "context_rgb_sha256": _tensor_hash(("context_rgb", data["images_input"])),
        "target_rgb_sha256": _tensor_hash(("target_rgb", data["images_output"])),
        "context_camera_sha256": _tensor_hash(
            ("context_intrinsics", data["intrinsics_input"]),
            ("context_cam_view", data["cam_view_input"]),
        ),
        "target_camera_sha256": _tensor_hash(
            ("target_intrinsics", data["intrinsics"]),
            ("target_cam_view", data["cam_view"]),
        ),
        "gt_instance_mask_sha256": _tensor_hash(
            ("gt_instance_mask", data["instance_label_output"])
        ),
    }


def _move_data(data: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in data.items()
    }


def _build_lsm_data(manifest: Path):
    opt = config_defaults["eval_scannet_lsm_instance"].evolve(
        num_workers=0,
        evaluating=True,
        max_eval_iters=0,
        num_input_views=8,
        num_views=15,
    )
    opt.dataset_kwargs = {
        **dict(opt.dataset_kwargs or {}),
        "lsm_manifest_path": str(manifest),
    }
    _, test_loader, _, test_dataset = get_multi_dataloader(opt, _LocalAccelerator())
    return opt, test_loader, test_dataset


def _build_gsi_model(checkpoint: Path, *, phase: str, step: int, workspace: Path):
    config_name = (
        "gsi_v2_joint_scannet_short355_ddp8"
        if phase == "joint"
        else "gsi_v2_recon_scannet_eval"
    )
    opt = config_defaults[config_name].evolve(
        resume=str(checkpoint),
        workspace=str(workspace),
        evaluating=True,
        num_workers=0,
        max_eval_iters=0,
        num_input_views=8,
        num_views=15,
        gsi_v2_return_debug_tensors=False,
    )
    model = model_registry[opt.model_type](opt).cuda()
    state = load_file(str(checkpoint), device="cpu")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict GSI restore failed: {incompatible}")
    schedule = None
    if phase == "joint":
        schedule = model.set_eval_schedule(step)
        if schedule["gate"] != 1.0:
            raise RuntimeError(f"Joint@200 native gate is not 1: {schedule}")
    model.set_eval_stage()
    model.eval()
    return model, schedule, {
        "strict": True,
        "state_key_count": len(state),
        "fresh_reset": False,
        "pgsr_absent": not any("pgsr" in key.lower() for key in state),
        "checkpoint_keys_sha256": _sha256_file(checkpoint),
    }


def _build_both_model(checkpoint: Path, workspace: Path):
    opt = config_defaults["eval_scannet_lsm_instance"].evolve(
        model_type="semantic_tokengs_v6",
        resume=str(checkpoint),
        workspace=str(workspace),
        experiment_name=workspace.name,
        evaluating=True,
        num_workers=0,
        max_eval_iters=0,
        num_input_views=8,
        num_views=15,
        instance_group_num_groups=100,
    )
    args = SimpleNamespace(resume=str(checkpoint), num_groups=100)
    _load_checkpoint_arch(args, opt)
    remap_opt(opt)
    model = model_registry[opt.model_type](opt)
    state = load_file(str(checkpoint), device="cpu")
    torch.nn.Module.load_state_dict(model, state, strict=False)
    if not any(key.startswith("enc_dec_backbone.") for key in state):
        backbone_path = Path(str(getattr(opt, "backbone_resume", "") or ""))
        if not backbone_path.is_file():
            raise RuntimeError(f"Both checkpoint requires missing backbone: {backbone_path}")
        backbone_state = load_file(str(backbone_path), device="cpu")
        native = torch.nn.Module.state_dict(model)
        prefixes = (
            "enc_dec_backbone.",
            "patch_embed.",
            "patch_plucker_embed.",
            "activation_head.",
            "anchor_pos_encoder.",
        )
        loadable = {
            key: value
            for key, value in backbone_state.items()
            if (key.startswith(prefixes) or key == "gs_tokens")
            and key in native
            and native[key].shape == value.shape
        }
        torch.nn.Module.load_state_dict(model, loadable, strict=False)
    model.eval()
    model.cuda()
    old_calls = {"count": 0}
    if bool(getattr(opt, "instance_branch_abs_units", False)) and hasattr(model, "absolute_gs_head"):
        original = model.activation_head.forward

        def _counting_activation(*args, **kwargs):
            old_calls["count"] += 1
            return original(*args, **kwargs)

        model.activation_head.forward = _counting_activation
        source = "absolute_student"
    else:
        source = "old_gs_head"
    return model, None, {
        "strict": False,
        "state_key_count": len(state),
        "fresh_reset": False,
        "pgsr_absent": not any("pgsr" in key.lower() for key in state),
        "gaussians_source": source,
        "old_gs_head_calls": old_calls,
        "checkpoint_keys_sha256": _sha256_file(checkpoint),
    }


def _assignment_stats(out: dict[str, torch.Tensor], probability: torch.Tensor) -> dict[str, float]:
    assignment = out.get("gsi_v2_instance_assignment_probabilities")
    if assignment is None:
        assignment = out.get("gaussian_group_probabilities")
    if assignment is None:
        assignment = probability.permute(0, 2, 3, 4, 5, 1).reshape(-1, probability.shape[1])
    assignment = assignment.detach().float()
    if assignment.shape[-1] < 2:
        raise RuntimeError(f"assignment tensor has no void channel: {tuple(assignment.shape)}")
    entropy = (-(assignment * assignment.clamp_min(1e-8).log()).sum(-1)).mean()
    usage = assignment[..., :-1].mean(dim=tuple(range(assignment.ndim - 1)))
    return {
        "effective_query_count": float(torch.exp(entropy).item()),
        "assignment_entropy": float(entropy.item()),
        "active_query_count_mass_gt_0.001": int((usage > 0.001).sum().item()),
    }


def _reconstruction_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    metrics: MetricsCalculator,
    reconstruction_loss: torch.Tensor,
) -> dict[str, object]:
    pred = pred.float().clamp(0, 1)
    gt = gt.float().clamp(0, 1)
    if pred.ndim != 5 or pred.shape[1] != 7:
        raise RuntimeError(f"expected seven target predictions, got {tuple(pred.shape)}")
    mse = (pred - gt).square().mean(dim=(2, 3, 4))[0]
    psnr = metrics.calculate_psnr(pred, gt, reduction="none")[0]
    ssim = metrics.calculate_ssim(pred, gt, reduction="none")[0]
    lpips = metrics.calculate_lpips(pred, gt, reduction="none")[0]
    return {
        "mse_per_target_view": [float(x) for x in mse],
        "psnr_per_target_view": [float(x) for x in psnr],
        "ssim_per_target_view": [float(x) for x in ssim],
        "lpips_per_target_view": [float(x) for x in lpips],
        "psnr": float(psnr.mean()),
        "ssim": float(ssim.mean()),
        "lpips": float(lpips.mean()),
        "reconstruction_loss": float(reconstruction_loss.detach().float().item()),
        "pooled_mse": float(mse.mean()),
    }


def _instance_metrics(
    out: dict[str, torch.Tensor], data: dict[str, object], scene: str,
    scene_entry: dict[str, object],
    *, max_predictions: int = 100, min_pixels: int = 1,
) -> dict[str, object]:
    probability = out["rendered_instance_group_probability"].detach().float().cpu()
    labels = data["instance_label_output"].detach().long().cpu()
    if probability.ndim != 6 or probability.shape[2] != 7:
        raise RuntimeError(f"expected [B,G+1,7,1,H,W] assignment output, got {tuple(probability.shape)}")
    void_channel = probability.shape[1] - 1
    pred_masks, pred_scores, pred_ids = [], [], []
    gt_masks, gt_ids = [], []
    per_target_view = []
    nonempty_query_ids = set()
    filtered_by_max = 0
    for batch_index in range(probability.shape[0]):
        for view in range(probability.shape[2]):
            image_id = make_target_view_image_id(
                scene_id=scene,
                context_frame_ids=scene_entry["context_raw_frame_ids"],
                target_frame_ids=scene_entry["test_raw_frame_ids"],
                target_view_index=view,
            )
            probs = probability[batch_index, :, view, 0].numpy()
            group_ids = np.argmax(probs, axis=0)
            nonempty_query_ids.update(int(x) for x in np.unique(group_ids) if int(x) != void_channel)
            masks, scores = masks_from_group_probs(
                probs, void_channel=void_channel, min_mask_area=min_pixels
            )
            filtered_by_max += max(0, len(masks) - max_predictions)
            pred_masks.extend(masks[:max_predictions])
            pred_scores.extend(scores[:max_predictions])
            pred_ids.extend([image_id] * min(len(masks), max_predictions))
            gt = gt_masks_from_instance_map(labels[batch_index, view].numpy(), min_mask_area=min_pixels)
            gt_masks.extend(gt)
            gt_ids.extend([image_id] * len(gt))
            view_ap = instance_ap(
                masks,
                scores,
                gt,
                thresholds=(0.25, 0.5, 0.75),
                vectorized=True,
                pred_image_ids=[image_id] * len(masks),
                gt_image_ids=[image_id] * len(gt),
            )
            view_diag = _mask_diagnostics(
                masks,
                gt,
                [image_id] * len(masks),
                [image_id] * len(gt),
            )
            per_target_view.append({
                "batch_index": int(batch_index),
                "target_view_index": int(view),
                "image_id": image_id,
                "ap25": float(view_ap["ap_25"]),
                "ap50": float(view_ap["ap_50"]),
                "ap75": float(view_ap["ap_75"]),
                "best_gt_iou": float(view_diag["mean_best_gt_iou"]),
                "recall50": float(view_diag["recall_iou50"]),
                "prediction_count": int(len(masks)),
                "gt_count": int(len(gt)),
            })
    results = instance_ap(
        pred_masks, pred_scores, gt_masks, thresholds=(0.25, 0.5, 0.75),
        vectorized=True, pred_image_ids=pred_ids, gt_image_ids=gt_ids,
    )
    coco = instance_ap(
        pred_masks, pred_scores, gt_masks,
        thresholds=tuple(t / 100 for t in range(50, 100, 5)),
        vectorized=True, pred_image_ids=pred_ids, gt_image_ids=gt_ids,
    )
    diag = _mask_diagnostics(pred_masks, gt_masks, pred_ids, gt_ids)
    query = _assignment_stats(out, probability.to(probability.device))
    return {
        "ap": float(coco["ap_mean"]),
        "ap25": float(results["ap_25"]),
        "ap50": float(results["ap_50"]),
        "ap75": float(results["ap_75"]),
        "pooled_inputs": {
            "pred_masks": pred_masks, "pred_scores": pred_scores,
            "pred_ids": pred_ids, "gt_masks": gt_masks, "gt_ids": gt_ids,
        },
        "per_target_view": per_target_view,
        "mean_best_gt_iou": float(diag["mean_best_gt_iou"]),
        "recall_iou25": float(diag["recall_iou25"]),
        "recall_iou50": float(diag["recall_iou50"]),
        "recall_iou75": float(diag["recall_iou75"]),
        "pred_gt": float(len(pred_masks) / max(1, len(gt_masks))),
        "prediction_count_nonempty": int(len(pred_masks)),
        "gt_count": int(len(gt_masks)),
        "nonempty_query_count": int(len(nonempty_query_ids)),
        "filtered_by_max_predictions": int(filtered_by_max),
        "score_threshold_filtered": 0,
        "active_queries": int(query["active_query_count_mass_gt_0.001"]),
        "effective_query_count": float(query["effective_query_count"]),
        "assignment_entropy": float(query["assignment_entropy"]),
        "void_ratio": float(probability[:, -1].mean().item()),
    }


def _evaluate_model(
    label: str, model, schedule, test_loader, test_dataset, manifest_entries,
    *, is_instance: bool, metrics: MetricsCalculator, metadata: dict[str, object],
    max_scenes: int = 0,
) -> dict[str, object]:
    per_scene = {}
    fingerprints = []
    pooled_pred_masks, pooled_pred_scores, pooled_pred_ids = [], [], []
    pooled_gt_masks, pooled_gt_ids = [], []
    all_finite = True
    seen = []
    for data_cpu in test_loader:
        if max_scenes > 0 and len(seen) >= max_scenes:
            break
        scene = str(data_cpu["scene_name"][0])
        if scene not in manifest_entries:
            raise RuntimeError(f"scene not in LSM manifest: {scene}")
        seen.append(scene)
        fingerprint = _manifest_fingerprints(data_cpu, scene, manifest_entries[scene])
        fingerprints.append(fingerprint)
        data = _move_data(data_cpu, torch.device("cuda"))
        with torch.inference_mode():
            if label != "both_1420":
                # Match eval_gsi_v2_short355.py: this computes the model's
                # reconstruction loss without enabling gradients or updates.
                out = model(data)
            else:
                # Match eval_instance_lsm_protocol.py for the historical
                # semantic model path.
                out = model(data, compute_quality_metrics=True)
        if "loss_reconstruction" in out:
            reconstruction_loss = out["loss_reconstruction"]
            reconstruction_loss_source = "loss_reconstruction"
        elif "loss_rgb" in out:
            # The historical semantic LSM model exposes its reconstruction
            # term under loss_rgb rather than loss_reconstruction.
            reconstruction_loss = out["loss_rgb"]
            reconstruction_loss_source = "loss_rgb"
        else:
            raise RuntimeError(
                f"{label} forward returned neither loss_reconstruction nor loss_rgb"
            )
        recon = _reconstruction_metrics(
            out["images_pred"], data["images_output"], metrics,
            reconstruction_loss,
        )
        row = {
            "scene_id": scene,
            "context_frame_ids": fingerprint["context_frame_ids"],
            "target_frame_ids": fingerprint["target_frame_ids"],
            "context_views": 8,
            "target_views": 7,
            **recon,
            "input_fingerprint": fingerprint,
        }
        if is_instance:
            instance = _instance_metrics(
                out, data, scene, manifest_entries[scene]
            )
            pooled = instance.pop("pooled_inputs")
            pooled_pred_masks.extend(pooled["pred_masks"])
            pooled_pred_scores.extend(pooled["pred_scores"])
            pooled_pred_ids.extend(pooled["pred_ids"])
            pooled_gt_masks.extend(pooled["gt_masks"])
            pooled_gt_ids.extend(pooled["gt_ids"])
            row.update(instance)
        if not all(
            np.isfinite(float(value))
            for value in (
                row["reconstruction_loss"],
                *row["mse_per_target_view"],
                *row["psnr_per_target_view"],
                *row["ssim_per_target_view"],
                *row["lpips_per_target_view"],
            )
        ):
            all_finite = False
        per_scene[scene] = row
    expected_scenes = list(manifest_entries)
    if max_scenes > 0:
        if len(seen) != max_scenes:
            raise RuntimeError(
                f"LSM smoke scene count mismatch: expected {max_scenes}, got {seen}"
            )
    elif seen != expected_scenes:
        raise RuntimeError(f"LSM scene order/count mismatch: expected {expected_scenes}, got {seen}")
    result = {
        "label": label,
        "scene_count": len(per_scene),
        "per_scene": per_scene,
        "input_fingerprints": fingerprints,
        "checkpoint_metadata": metadata,
        "schedule": schedule,
        "reconstruction_loss_source": reconstruction_loss_source,
        "all_finite": bool(all_finite),
        "instance_ap_image_identity": "per_target_view_v1",
        "cross_target_view_matching": False,
        "target_view_image_ids": sorted(
            set(pooled_pred_ids).union(pooled_gt_ids)
        ),
    }
    keys = ("psnr", "ssim", "lpips", "reconstruction_loss")
    result["mean"] = {key: float(np.mean([row[key] for row in per_scene.values()])) for key in keys}
    pooled_mse = float(np.mean([mse for row in per_scene.values() for mse in row["mse_per_target_view"]]))
    result["pooled_mse"] = pooled_mse
    result["pooled_psnr"] = float(-10.0 * math.log10(max(pooled_mse, 1e-8)))
    result["pooled_ssim"] = float(
        np.mean([value for row in per_scene.values() for value in row["ssim_per_target_view"]])
    )
    result["pooled_lpips"] = float(
        np.mean([value for row in per_scene.values() for value in row["lpips_per_target_view"]])
    )
    result["mean"]["psnr"] = float(np.mean([row["psnr"] for row in per_scene.values()]))
    result["mean"]["ssim"] = float(np.mean([row["ssim"] for row in per_scene.values()]))
    result["mean"]["lpips"] = float(np.mean([row["lpips"] for row in per_scene.values()]))
    if is_instance:
        for key in ("ap", "ap25", "ap50", "ap75", "mean_best_gt_iou", "recall_iou25", "recall_iou50", "recall_iou75", "pred_gt", "void_ratio", "effective_query_count"):
            result["mean"][key] = float(np.mean([row[key] for row in per_scene.values()]))
        result["mean"]["nonempty_query_count"] = float(np.mean([row["nonempty_query_count"] for row in per_scene.values()]))
        result["mean"]["active_queries"] = float(np.mean([row["active_queries"] for row in per_scene.values()]))
        result["mean"]["score_threshold_filtered"] = int(sum(row["score_threshold_filtered"] for row in per_scene.values()))
        result["mean"]["filtered_by_max_predictions"] = int(sum(row["filtered_by_max_predictions"] for row in per_scene.values()))
        pooled = instance_ap(
            pooled_pred_masks, pooled_pred_scores, pooled_gt_masks,
            thresholds=(0.25, 0.5, 0.75), vectorized=True,
            pred_image_ids=pooled_pred_ids, gt_image_ids=pooled_gt_ids,
        )
        pooled_coco = instance_ap(
            pooled_pred_masks, pooled_pred_scores, pooled_gt_masks,
            thresholds=tuple(t / 100 for t in range(50, 100, 5)),
            vectorized=True, pred_image_ids=pooled_pred_ids, gt_image_ids=pooled_gt_ids,
        )
        pooled_diag = _mask_diagnostics(pooled_pred_masks, pooled_gt_masks, pooled_pred_ids, pooled_gt_ids)
        result["pooled"] = {
            "ap": float(pooled_coco["ap_mean"]),
            "ap25": float(pooled["ap_25"]), "ap50": float(pooled["ap_50"]), "ap75": float(pooled["ap_75"]),
            "mean_best_gt_iou": float(pooled_diag["mean_best_gt_iou"]),
            "recall_iou25": float(pooled_diag["recall_iou25"]),
            "recall_iou50": float(pooled_diag["recall_iou50"]),
            "recall_iou75": float(pooled_diag["recall_iou75"]),
            "pred_count": len(pooled_pred_masks), "gt_count": len(pooled_gt_masks),
        }
    return result


def _fingerprints_equal(results: list[dict[str, object]]) -> bool:
    reference = results[0]["input_fingerprints"]
    return all(result["input_fingerprints"] == reference for result in results[1:])


def _diff_table(joint: dict[str, object], both: dict[str, object]) -> dict[str, object]:
    scenes = {}
    for scene in both["per_scene"]:
        left, right = joint["per_scene"][scene], both["per_scene"][scene]
        scenes[scene] = {
            key: float(left[key] - right[key])
            for key in ("ap25", "ap50", "ap75", "mean_best_gt_iou", "recall_iou50", "pred_gt", "psnr")
        }
    def count(key: str, sign: str) -> int:
        values = [row[key] for row in scenes.values()]
        return sum(value > 0 for value in values) if sign == ">" else sum(value == 0 for value in values) if sign == "=" else sum(value < 0 for value in values)
    ranked = sorted(scenes.items(), key=lambda item: item[1]["ap50"], reverse=True)
    return {
        "per_scene": scenes,
        "ap50_improved_scenes": count("ap50", ">"),
        "ap50_tied_scenes": count("ap50", "="),
        "ap50_declined_scenes": count("ap50", "<"),
        "best_iou_improved_scenes": count("mean_best_gt_iou", ">"),
        "recall50_improved_scenes": count("recall_iou50", ">"),
        "psnr_declined_over_0.2db_scenes": sum(row["psnr"] < -0.2 for row in scenes.values()),
        "largest_ap50_losses": [{"scene": scene, "delta_ap50": row["ap50"]} for scene, row in ranked[-5:][::-1]],
        "largest_ap50_gains": [{"scene": scene, "delta_ap50": row["ap50"]} for scene, row in ranked[:5]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--max_scenes", type=int, default=0)
    args = parser.parse_args()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != EXPECTED_COMMIT:
        raise RuntimeError("unexpected git HEAD")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "eval.log"
    log_path.write_text("LSM-40 GSI-v2 paired audit\n", encoding="utf-8")
    manifest_audit = _audit_lsm_manifest(str(args.manifest))
    if manifest_audit["scene_count"] != 40:
        raise RuntimeError("LSM-40 manifest audit failed")
    manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_manifest_entries = manifest_payload["scenes"]
    if args.max_scenes < 0:
        raise ValueError("--max_scenes must be non-negative")
    manifest_entries = dict(
        list(all_manifest_entries.items())[: args.max_scenes]
        if args.max_scenes
        else all_manifest_entries.items()
    )
    metadata = {
        "joint_step200": _metadata(JOINT, 200),
        "r1_step250": _metadata(R1, 250),
        "both_step1420": _metadata(BOTH, 1420),
    }
    data_opt, test_loader, test_dataset = _build_lsm_data(args.manifest)
    del data_opt, test_dataset
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; LSM evaluation must run on a compute node")
    device = torch.device("cuda")
    metrics = MetricsCalculator(device=device)
    smoke = {}
    outputs = {}
    for label, checkpoint, phase, step, instance in (
        ("both_1420", BOTH, "both", 1420, True),
        ("joint_200", JOINT, "joint", 200, True),
        ("r1_250", R1, "reconstruction", 250, False),
    ):
        workspace = args.output_dir / label
        workspace.mkdir(parents=True, exist_ok=True)
        if phase == "both":
            model, schedule, restore = _build_both_model(checkpoint, workspace)
        else:
            model, schedule, restore = _build_gsi_model(checkpoint, phase=phase, step=step, workspace=workspace)
        result = _evaluate_model(
            label, model, schedule, test_loader, test_dataset=None,
            manifest_entries=manifest_entries, is_instance=instance,
            metrics=metrics, metadata=metadata["joint_step200" if label == "joint_200" else "r1_step250" if label == "r1_250" else "both_step1420"],
            max_scenes=args.max_scenes,
        )
        result["checkpoint"] = str(checkpoint.resolve())
        result["checkpoint_sha256"] = _sha256_file(checkpoint)
        result["optimizer_step"] = step
        result["eval_precision"] = "fp32"
        result["native_gate"] = 1.0 if label == "joint_200" else None
        result["restore"] = restore
        if label == "both_1420":
            result["gaussians_source"] = restore["gaussians_source"]
            result["old_gs_head_calls"] = restore["old_gs_head_calls"]["count"]
        else:
            result["gaussians_source"] = "absolute_student"
            result["old_gs_head_calls"] = 0
        outputs[label] = result
        smoke[label] = {
            "scene_id": next(iter(result["per_scene"])),
            "psnr": result["per_scene"][next(iter(result["per_scene"]))]["psnr"],
            "instance": instance,
        }
        json.dump(result, (workspace / "result.json").open("w", encoding="utf-8"), indent=2)
        del model
        torch.cuda.empty_cache()
    fingerprints_valid = _fingerprints_equal([outputs["both_1420"], outputs["joint_200"], outputs["r1_250"]])
    if not fingerprints_valid:
        raise RuntimeError("LSM-40 protocol fingerprints differ; refusing comparison")
    comparison = _diff_table(outputs["joint_200"], outputs["both_1420"])
    reconstruction = {
        "r1_step250": {key: outputs["r1_250"]["mean"][key] for key in ("psnr", "ssim", "lpips")},
        "joint_step200": {key: outputs["joint_200"]["mean"][key] for key in ("psnr", "ssim", "lpips")},
        "delta_joint_minus_r1": {
            key: outputs["joint_200"]["mean"][key] - outputs["r1_250"]["mean"][key]
            for key in ("psnr", "ssim", "lpips")
        },
        "pooled_psnr": {
            "r1_step250": outputs["r1_250"]["pooled_psnr"],
            "joint_step200": outputs["joint_200"]["pooled_psnr"],
            "delta_joint_minus_r1": outputs["joint_200"]["pooled_psnr"] - outputs["r1_250"]["pooled_psnr"],
        },
    }
    summary = {
        "git_commit": EXPECTED_COMMIT,
        "protocol": manifest_audit,
        "lsm40_protocol_valid": fingerprints_valid,
        "scenes_evaluated": len(manifest_entries),
        "joint_step": 200,
        "joint_native_gate": 1.0,
        "joint_mean_ap50": outputs["joint_200"]["mean"]["ap50"],
        "joint_pooled_ap50": outputs["joint_200"]["pooled"]["ap50"],
        "joint_best_iou": outputs["joint_200"]["mean"]["mean_best_gt_iou"],
        "joint_recall50": outputs["joint_200"]["mean"]["recall_iou50"],
        "both_mean_ap50": outputs["both_1420"]["mean"]["ap50"],
        "both_pooled_ap50": outputs["both_1420"]["pooled"]["ap50"],
        "delta_mean_ap50": outputs["joint_200"]["mean"]["ap50"] - outputs["both_1420"]["mean"]["ap50"],
        "delta_pooled_ap50": outputs["joint_200"]["pooled"]["ap50"] - outputs["both_1420"]["pooled"]["ap50"],
        "ap50_improved_scenes": comparison["ap50_improved_scenes"],
        "best_iou_improved_scenes": comparison["best_iou_improved_scenes"],
        "recall50_improved_scenes": comparison["recall50_improved_scenes"],
        "r1_mean_psnr": outputs["r1_250"]["mean"]["psnr"],
        "joint_mean_psnr": outputs["joint_200"]["mean"]["psnr"],
        "delta_mean_psnr": reconstruction["delta_joint_minus_r1"]["psnr"],
        "reconstruction_preserved": bool(
            reconstruction["delta_joint_minus_r1"]["psnr"] >= -0.20
            and reconstruction["pooled_psnr"]["delta_joint_minus_r1"] >= -0.20
        ),
        "outputs": {
            "joint_200": "joint_200/result.json",
            "both_1420": "both_1420/result.json",
            "r1_250": "r1_250/result.json",
        },
        "training_started": False,
        "ttt_started": False,
    }
    for filename, payload in (
        ("summary.json", summary),
        ("per_scene_metrics.json", {label: result["per_scene"] for label, result in outputs.items()}),
        ("protocol_fingerprints.json", {label: result["input_fingerprints"] for label, result in outputs.items()}),
        ("checkpoint_metadata.json", metadata),
        ("comparison_vs_both.json", comparison),
        ("reconstruction_comparison.json", reconstruction),
        ("smoke.json", smoke),
    ):
        (args.output_dir / filename).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
