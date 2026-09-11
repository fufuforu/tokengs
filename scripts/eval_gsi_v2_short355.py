"""Fixed eight-scene evaluation for GSI-v2 short355 checkpoints.

One invocation evaluates one strict Phase-J checkpoint using the canonical
validation first-window samples.  The launcher can run independent
checkpoints on separate GPUs; this script never changes model parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

torch._dynamo.config.disable = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_instance_lsm_protocol import _mask_diagnostics  # noqa: E402
from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.utils.instance_ap import (  # noqa: E402
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)
from tokengs.utils.metrics import MetricsCalculator  # noqa: E402


MANIFEST = ROOT / "data/scannet_prompt/scannet_prompt_full_wide_8x7.json"
CFG = "gsi_v2_joint_scannet_eval"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _first_validation_indices(test_dataset):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    scenes = [str(value) for value in manifest["validation_scenes"]]
    if len(test_dataset.datasets) != 1:
        raise RuntimeError("expected one canonical GSI dataset")
    samples = test_dataset.datasets[0].dataset.sample_list
    first = {}
    for index, sample in enumerate(samples):
        first.setdefault(str(getattr(sample, "scene", "")), index)
    missing = [scene for scene in scenes if scene not in first]
    if missing:
        raise RuntimeError(f"validation scenes missing: {missing}")
    return [first[scene] for scene in scenes], scenes


def _scene_name(data) -> str:
    value = data["scene_name"]
    return str(value[0] if isinstance(value, (tuple, list)) else value)


def _frame_ids(data):
    value = data["frame_ids"]
    if torch.is_tensor(value):
        return [int(item) for item in value[0].reshape(-1).tolist()]
    return [int(item) for item in value[0]]


def _scene_instance(out, data):
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
        predictions, scores, ground_truth, thresholds=(.25, .5, .75), vectorized=True,
        pred_image_ids=pred_ids, gt_image_ids=gt_ids,
    )
    diag = _mask_diagnostics(predictions, ground_truth, pred_ids, gt_ids)
    assignment = out["gsi_v2_instance_assignment_probabilities"].detach().float()
    usage = assignment[..., :-1].mean(dim=(0, 1))
    entropy = (-(assignment * assignment.clamp_min(1e-8).log()).sum(-1)).mean()
    return {
        "ap": float(ap["ap_mean"]), "ap25": float(ap["ap_25"]),
        "ap50": float(ap["ap_50"]), "ap75": float(ap["ap_75"]),
        "mean_best_gt_iou": float(diag["mean_best_gt_iou"]),
        "recall_iou25": float(diag["recall_iou25"]),
        "recall_iou50": float(diag["recall_iou50"]),
        "recall_iou75": float(diag["recall_iou75"]),
        "pred_gt": float(len(predictions) / max(1, len(ground_truth))),
        "prediction_count_nonempty": int(len(predictions)),
        "gt_count": int(len(ground_truth)),
        "active_query_count_mass_gt_0.001": int((usage > .001).sum()),
        "effective_query_count": float(torch.exp(entropy).item()),
        "assignment_entropy": float(entropy.item()),
        "void_ratio": float(probability[-1].mean()),
        "predictions": predictions, "scores": scores, "prediction_ids": pred_ids,
        "ground_truth": ground_truth, "ground_truth_ids": gt_ids,
    }


def main():
    cli = args()
    checkpoint = Path(cli.checkpoint).resolve()
    output = Path(cli.output).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evaluation output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    opt = config_defaults[CFG].evolve(
        resume=str(checkpoint), gsi_v2_resume_mode="strict",
        workspace=str(output.parent), gsi_v2_return_debug_tensors=True,
        num_workers=0, max_eval_iters=8,
    )
    accelerator = Accelerator(mixed_precision=opt.mixed_precision)
    _train, test, _train_ds, test_dataset = get_multi_dataloader(opt, accelerator)
    indices, scene_names = _first_validation_indices(test_dataset)
    test = DataLoader(
        Subset(test_dataset, indices), batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    model = model_registry[opt.model_type](opt).to(accelerator.device)
    state = load_file(str(checkpoint), device="cpu")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict restore failed: {incompatible}")
    model.set_eval_stage()
    model.eval()
    model, test = accelerator.prepare(model, test)
    metrics = MetricsCalculator(device=accelerator.device)
    rows = []
    all_predictions, all_scores, all_pred_ids = [], [], []
    all_ground_truth, all_gt_ids = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for data in test:
            out = model(data)
            scene = _scene_name(data)
            rgb_pred = out["images_pred"].float().clamp(0, 1)
            rgb_gt = data["images_output"].float().clamp(0, 1)
            row = {
                "scene_name": scene,
                "frame_ids": _frame_ids(data),
                "context_views": 8, "target_views": 7,
                "psnr": float(metrics.calculate_psnr(rgb_pred, rgb_gt).mean()),
                "ssim": float(metrics.calculate_ssim(rgb_pred, rgb_gt).mean()),
                "lpips": float(metrics.calculate_lpips(rgb_pred, rgb_gt).mean()),
                "reconstruction_loss": float(out["loss_reconstruction"].detach()),
            }
            instance = _scene_instance(out, data)
            for key in (
                "ap", "ap25", "ap50", "ap75", "mean_best_gt_iou",
                "recall_iou25", "recall_iou50", "recall_iou75", "pred_gt",
                "prediction_count_nonempty", "gt_count",
                "active_query_count_mass_gt_0.001", "effective_query_count",
                "assignment_entropy", "void_ratio",
            ):
                row[key] = instance[key]
            rows.append(row)
            all_predictions.extend(instance["predictions"])
            all_scores.extend(instance["scores"])
            all_pred_ids.extend(instance["prediction_ids"])
            all_ground_truth.extend(instance["ground_truth"])
            all_gt_ids.extend(instance["ground_truth_ids"])
    pooled_ap = instance_ap(
        all_predictions, all_scores, all_ground_truth, thresholds=(.25, .5, .75), vectorized=True,
        pred_image_ids=all_pred_ids, gt_image_ids=all_gt_ids,
    )
    pooled_diag = _mask_diagnostics(all_predictions, all_ground_truth, all_pred_ids, all_gt_ids)
    result = {
        "protocol": "GSI-v2 short355 canonical full_wide_8x7 validation first window",
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "scene_names": scene_names, "scene_count": len(rows),
        "mean": {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("psnr", "ssim", "lpips", "reconstruction_loss", "ap", "ap25", "ap50", "ap75", "mean_best_gt_iou", "recall_iou25", "recall_iou50", "recall_iou75", "pred_gt", "effective_query_count", "assignment_entropy", "void_ratio")
        },
        "pooled": {
            "ap": float(pooled_ap["ap_mean"]), "ap25": float(pooled_ap["ap_25"]),
            "ap50": float(pooled_ap["ap_50"]), "ap75": float(pooled_ap["ap_75"]),
            "mean_best_gt_iou": float(pooled_diag["mean_best_gt_iou"]),
            "recall_iou25": float(pooled_diag["recall_iou25"]),
            "recall_iou50": float(pooled_diag["recall_iou50"]),
            "recall_iou75": float(pooled_diag["recall_iou75"]),
        },
        "per_scene": rows,
        "checkpoint_restore": {
            "strict": True, "state_key_count": len(state),
            "reconstruction_keys": sum(key.startswith("reconstruction.") for key in state),
            "fresh_reset": False, "pgsr_absent": not any("pgsr" in key.lower() for key in state),
        },
        "manifest": {"path": str(MANIFEST.resolve()), "sha256": sha256_file(MANIFEST)},
        "all_finite": bool(all(np.isfinite(row[key]) for row in rows for key in ("psnr", "ssim", "lpips", "reconstruction_loss", "ap50"))),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    output.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
