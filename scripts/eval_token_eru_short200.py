"""Strict FP32 8-scene evaluation for TokenGS-ERU-v1 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_gsi_v2_short355 import (  # noqa: E402
    MANIFEST,
    _first_validation_indices,
    _input_fingerprint,
    _mask_diagnostics,
    _scene_name,
    _frame_ids,
    sha256_file,
)
from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import load_model_checkpoint  # noqa: E402
from tokengs.utils.metrics import MetricsCalculator  # noqa: E402
from tokengs.utils.instance_ap import instance_ap  # noqa: E402
from tokengs.utils.instance_ap import (  # noqa: E402
    gt_masks_from_instance_map,
    masks_from_group_probs,
)


BOTH_CFG = "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
ERU_CFG = "semantic_v6_absolute_units_true_shared_token_eru1_short200_ddp8"


def _scene_instance(out, data):
    """TokenGS-native instance diagnostics with the unchanged AP protocol."""
    probability = out["rendered_instance_group_probability"][0].detach().float().cpu().numpy()[:, :, 0]
    labels = data["instance_label_output"][0].detach().long().cpu().numpy()
    predictions, scores, pred_ids = [], [], []
    ground_truth, gt_ids = [], []
    for view in range(probability.shape[1]):
        image_id = f"{_scene_name(data)}:target:{view}"
        masks, view_scores = masks_from_group_probs(
            probability[:, view], void_channel=probability.shape[0] - 1, min_mask_area=1
        )
        predictions.extend(masks)
        scores.extend(view_scores)
        pred_ids.extend([image_id] * len(masks))
        gt = gt_masks_from_instance_map(labels[view], min_mask_area=1)
        ground_truth.extend(gt)
        gt_ids.extend([image_id] * len(gt))
    ap = instance_ap(
        predictions, scores, ground_truth, thresholds=(0.25, 0.5, 0.75),
        vectorized=True, pred_image_ids=pred_ids, gt_image_ids=gt_ids,
    )
    diag = _mask_diagnostics(predictions, ground_truth, pred_ids, gt_ids)
    assignment = out.get("gsi_v2_instance_assignment_probabilities")
    if assignment is None:
        assignment = out["instance_group_probabilities"]
    assignment = assignment.detach().float()
    usage = assignment[..., :-1].mean(dim=(0, 1, 2))
    entropy = (-(assignment * assignment.clamp_min(1e-8).log()).sum(-1)).mean()
    group_ids = np.argmax(probability, axis=0)
    void_channel = probability.shape[0] - 1
    nonempty_query_ids = {
        int(query_id)
        for query_id in np.unique(group_ids)
        if int(query_id) != void_channel
    }
    return {
        "ap": float(ap["ap_mean"]), "ap25": float(ap["ap_25"]),
        "ap50": float(ap["ap_50"]), "ap75": float(ap["ap_75"]),
        "mean_best_gt_iou": float(diag["mean_best_gt_iou"]),
        "recall_iou25": float(diag["recall_iou25"]),
        "recall_iou50": float(diag["recall_iou50"]),
        "recall_iou75": float(diag["recall_iou75"]),
        "pred_gt": float(len(predictions) / max(1, len(ground_truth))),
        "prediction_count_nonempty": int(len(predictions)),
        "nonempty_query_count": int(len(nonempty_query_ids)),
        "gt_count": int(len(ground_truth)),
        "active_query_count_mass_gt_0.001": int((usage > 0.001).sum()),
        "effective_query_count": float(torch.exp(entropy).item()),
        "assignment_entropy": float(entropy.item()),
        "void_ratio": float(probability[-1].mean()),
        "predictions": predictions, "scores": scores, "prediction_ids": pred_ids,
        "ground_truth": ground_truth, "ground_truth_ids": gt_ids,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--optimizer-step", type=int, required=True)
    parser.add_argument("--model", choices=("both", "eru"), required=True)
    parser.add_argument("--max-scenes", type=int, default=8)
    return parser


def _metadata_check(checkpoint: Path, step: int) -> dict[str, object]:
    path = checkpoint.parent / f"metadata_step_{step:06d}.json"
    if not path.is_file():
        raise FileNotFoundError(f"required checkpoint metadata is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("optimizer_step", -1)) != step:
        raise RuntimeError(f"metadata step mismatch in {path}: {payload}")
    return payload


def main() -> None:
    args = _parser().parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evaluation output: {output}")
    if args.optimizer_step < 0:
        raise ValueError("optimizer step must be non-negative")
    metadata = _metadata_check(checkpoint, args.optimizer_step)
    cfg_name = ERU_CFG if args.model == "eru" else BOTH_CFG
    opt = config_defaults[cfg_name].evolve(
        resume=str(checkpoint),
        evaluating=True,
        num_workers=0,
        max_eval_iters=8,
        eval_before_training=False,
        use_wandb=False,
        workspace=str(output.parent),
    )
    accelerator = Accelerator(mixed_precision="no")
    _, test_dataset, _, test_dataset_wrapper = get_multi_dataloader(opt, accelerator)
    indices, scene_names = _first_validation_indices(test_dataset_wrapper)
    if args.max_scenes != 8:
        indices, scene_names = indices[: args.max_scenes], scene_names[: args.max_scenes]
    if len(scene_names) != 8:
        raise RuntimeError(f"strict ERU audit requires 8 scenes, got {scene_names}")
    test = DataLoader(
        Subset(test_dataset_wrapper, indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    model = model_registry[opt.model_type](opt).to(accelerator.device)
    state = load_file(str(checkpoint), device="cpu")
    # TokenGS prompt checkpoints intentionally contain the trainable
    # namespaces only.  The frozen backbone is restored from
    # ``backbone_resume`` and the guarded heads/tail are restored by the
    # same loader used by train.py and the preflight.  Direct strict loading
    # into the full PromptTokenGS module incorrectly treats those omitted
    # frozen keys as missing and bypasses the lineage-specific loader.
    load_model_checkpoint(opt, model, accelerator, 0)
    if args.model == "eru":
        if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
            raise RuntimeError("ERU checkpoint did not restore its ERU namespaces")
        schedule = model.set_token_eru_step(args.optimizer_step)
    else:
        schedule = None
    model.eval()
    model, test = accelerator.prepare(model, test)
    metrics = MetricsCalculator(device=accelerator.device)
    rows = []
    all_predictions, all_scores, all_pred_ids = [], [], []
    all_ground_truth, all_gt_ids = [], []
    with torch.inference_mode():
        for data in test:
            with accelerator.autocast():
                out = model(data, compute_quality_metrics=False)
            scene = _scene_name(data)
            rgb_pred = out["images_pred"].float().clamp(0, 1)
            rgb_gt = data["images_output"].float().clamp(0, 1)
            psnr = metrics.calculate_psnr(rgb_pred, rgb_gt, reduction="none")[0]
            ssim = metrics.calculate_ssim(rgb_pred, rgb_gt, reduction="none")[0]
            lpips = metrics.calculate_lpips(rgb_pred, rgb_gt, reduction="none")[0]
            instance = _scene_instance(out, data)
            row = {
                "scene_name": scene,
                "frame_ids": _frame_ids(data),
                "context_views": 8,
                "target_views": 7,
                "psnr_per_target_view": [float(x) for x in psnr],
                "ssim_per_target_view": [float(x) for x in ssim],
                "lpips_per_target_view": [float(x) for x in lpips],
                "psnr": float(psnr.mean()),
                "ssim": float(ssim.mean()),
                "lpips": float(lpips.mean()),
                "reconstruction_loss": float(out["loss_rgb"].detach()),
                "mse_per_target_view": [
                    float(x)
                    for x in ((rgb_pred - rgb_gt) ** 2).mean(dim=(2, 3, 4))[0]
                ],
                "input_fingerprint": _input_fingerprint(data, scene),
            }
            for key in (
                "ap", "ap25", "ap50", "ap75", "mean_best_gt_iou",
                "recall_iou25", "recall_iou50", "recall_iou75", "pred_gt",
                "prediction_count_nonempty", "gt_count",
                "active_query_count_mass_gt_0.001", "nonempty_query_count",
                "effective_query_count",
                "assignment_entropy", "void_ratio",
            ):
                row[key] = instance[key]
            rows.append(row)
            all_predictions.extend(instance["predictions"])
            all_scores.extend(instance["scores"])
            all_pred_ids.extend(instance["prediction_ids"])
            all_ground_truth.extend(instance["ground_truth"])
            all_gt_ids.extend(instance["ground_truth_ids"])
    pooled = instance_ap(
        all_predictions,
        all_scores,
        all_ground_truth,
        thresholds=(0.25, 0.5, 0.75),
        vectorized=True,
        pred_image_ids=all_pred_ids,
        gt_image_ids=all_gt_ids,
    )
    pooled_diag = _mask_diagnostics(
        all_predictions, all_ground_truth, all_pred_ids, all_gt_ids
    )
    pooled_mse = float(
        sum(sum(row["mse_per_target_view"]) for row in rows) / (7 * len(rows))
    )
    result = {
        "protocol": "TokenGS ERU strict paired full_wide_8x7 validation first window",
        "model": args.model,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "optimizer_step": int(args.optimizer_step),
        "eval_precision": "fp32",
        "native_gate": True,
        "schedule": schedule,
        "metadata": metadata,
        "scene_names": scene_names,
        "scene_count": len(rows),
        "mean": {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "psnr", "ssim", "lpips", "reconstruction_loss", "ap", "ap25",
                "ap50", "ap75", "mean_best_gt_iou", "recall_iou25",
                "recall_iou50", "recall_iou75", "pred_gt",
                "active_query_count_mass_gt_0.001", "nonempty_query_count",
                "prediction_count_nonempty", "effective_query_count",
                "assignment_entropy", "void_ratio",
            )
        },
        "pooled_mse": pooled_mse,
        "pooled_psnr": float(-10.0 * np.log10(max(pooled_mse, 1e-8))),
        "pooled": {
            "ap": float(pooled["ap_mean"]),
            "ap25": float(pooled["ap_25"]),
            "ap50": float(pooled["ap_50"]),
            "ap75": float(pooled["ap_75"]),
            "mean_best_gt_iou": float(pooled_diag["mean_best_gt_iou"]),
            "recall_iou25": float(pooled_diag["recall_iou25"]),
            "recall_iou50": float(pooled_diag["recall_iou50"]),
            "recall_iou75": float(pooled_diag["recall_iou75"]),
        },
        "per_scene": rows,
        "checkpoint_restore": {
            "strict": True,
            "loader": "tokengs.train.load_model_checkpoint",
            "state_key_count": len(state),
            "fresh_reset": False,
            "pgsr_absent": not any("pgsr" in key.lower() for key in state),
            "gsi_absent": not any("gsi" in key.lower() for key in state),
            "ta_riu_absent": not any("ta_riu" in key.lower() for key in state),
            "query_memory_refiner_absent": not any(
                "query_memory" in key.lower() for key in state
            ),
        },
        "manifest": {
            "path": str(MANIFEST.resolve()),
            "sha256": sha256_file(MANIFEST),
        },
        "input_fingerprints": [row["input_fingerprint"] for row in rows],
        "all_finite": bool(
            all(
                np.isfinite(row[key])
                for row in rows
                for key in ("psnr", "ssim", "lpips", "reconstruction_loss", "ap50")
            )
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
