"""Evaluator for the isolated ``siu3r_global_multiview_v1`` contract.

The model runner is deliberately separate.  It must emit one ``.npz`` per
official pair, with the schema documented in ``--help``.  This evaluator never
calls ``per_target_view_v1`` and never performs matching, clustering or
test-time adaptation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokengs.siu3r_protocol import (  # noqa: E402
    CONTEXT_VIEWS,
    IMAGE_SIZE,
    PAIR_RECORDS,
    TARGET_VIEWS,
    build_instance_masks,
    concat_views_height,
    global_semantic_iou,
    validate_val_pairs,
    TorchMetricBackend,
)


SCHEMA = "SIU3R pair npz v1: context/target semantic+instance pred/gt, target RGB/depth pred/gt; explicit pred labels/scores"


def _npz_array(bundle: Any, key: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if key not in bundle:
        raise KeyError(f"pair bundle missing {key}")
    value = np.asarray(bundle[key])
    if shape is not None and value.shape != shape:
        raise ValueError(f"{key} expected {shape}, got {value.shape}")
    if not np.isfinite(value.astype(np.float64, copy=False)).all():
        raise ValueError(f"{key} contains non-finite values")
    return value


def _ids_to_mapping(ids: np.ndarray, values: np.ndarray) -> dict[int, Any]:
    unique = sorted(int(value) for value in np.unique(ids) if int(value) != 0)
    if len(unique) != len(values):
        raise ValueError(f"explicit prediction metadata length {len(values)} != IDs {len(unique)}")
    return dict(zip(unique, values.tolist()))


def _torch_map_input(backend: TorchMetricBackend, sem: np.ndarray, ins: np.ndarray, *, pred: bool, labels=None, scores=None) -> dict[str, Any]:
    packed = build_instance_masks(sem, ins, prediction=pred, labels=labels, scores=scores)
    torch = backend.torch
    masks = np.asarray(packed["masks"], dtype=np.bool_)
    if masks.size == 0:
        masks = np.zeros((0, *sem.shape), dtype=np.bool_)
    output: dict[str, Any] = {
        "masks": torch.from_numpy(masks).to(backend.device),
        "labels": torch.tensor(packed["labels"], dtype=torch.long, device=backend.device),
    }
    if pred:
        output["scores"] = torch.tensor(packed["scores"], dtype=torch.float32, device=backend.device)
    return output


def _read_pair(bundle_path: Path) -> dict[str, Any]:
    with np.load(bundle_path, allow_pickle=False) as bundle:
        context_instance_pred = _npz_array(bundle, "context_instance_pred", (2, *IMAGE_SIZE))
        target_instance_pred = _npz_array(bundle, "target_instance_pred", (6, *IMAGE_SIZE))
        return {
            "context_sem_pred": _npz_array(bundle, "context_semantic_pred", (2, *IMAGE_SIZE)),
            "context_sem_gt": _npz_array(bundle, "context_semantic_gt", (2, *IMAGE_SIZE)),
            "context_ins_pred": context_instance_pred,
            "context_ins_gt": _npz_array(bundle, "context_instance_gt", (2, *IMAGE_SIZE)),
            "target_sem_pred": _npz_array(bundle, "target_semantic_pred", (6, *IMAGE_SIZE)),
            "target_sem_gt": _npz_array(bundle, "target_semantic_gt", (6, *IMAGE_SIZE)),
            "target_ins_pred": target_instance_pred,
            "target_ins_gt": _npz_array(bundle, "target_instance_gt", (6, *IMAGE_SIZE)),
            "rgb_pred": _npz_array(bundle, "target_rgb_pred", (6, 3, *IMAGE_SIZE)),
            "rgb_gt": _npz_array(bundle, "target_rgb_gt", (6, 3, *IMAGE_SIZE)),
            "depth_pred": _npz_array(bundle, "target_depth_pred", (6, *IMAGE_SIZE)),
            "depth_gt": _npz_array(bundle, "target_depth_gt", (6, *IMAGE_SIZE)),
            # Visualizer/pred.json stores 1-based ScanNet label_id values;
            # official evaluator.py subtracts one before TorchMetrics.
            "context_pred_ids": {
                key: int(value) - 1
                for key, value in _ids_to_mapping(context_instance_pred, _npz_array(bundle, "context_pred_labels")).items()
            },
            "context_pred_scores": _ids_to_mapping(context_instance_pred, _npz_array(bundle, "context_pred_scores")),
            "target_pred_ids": {
                key: int(value) - 1
                for key, value in _ids_to_mapping(target_instance_pred, _npz_array(bundle, "target_pred_labels")).items()
            },
            "target_pred_scores": _ids_to_mapping(target_instance_pred, _npz_array(bundle, "target_pred_scores")),
        }


def _semantic_counts(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intersection = np.zeros(21, dtype=np.int64)
    union = np.zeros(21, dtype=np.int64)
    for semantic_id in range(1, 21):
        pred = prediction == semantic_id
        truth = target == semantic_id
        intersection[semantic_id] = np.logical_and(pred, truth).sum()
        union[semantic_id] = np.logical_or(pred, truth).sum()
    return intersection, union


class DatasetAccumulator:
    """Dataset-global state matching the official evaluator's final compute."""

    def __init__(self, device: str) -> None:
        # Official evaluator accumulates segmentation metrics across all scenes,
        # while reconstruction/depth are arithmetic means over target images.
        self.context_backend = TorchMetricBackend(device=device)
        self.target_backend = TorchMetricBackend(device=device)
        self.context_intersection = np.zeros(21, dtype=np.int64)
        self.context_union = np.zeros(21, dtype=np.int64)
        self.target_intersection = np.zeros(21, dtype=np.int64)
        self.target_union = np.zeros(21, dtype=np.int64)
        self.reconstruction: list[dict[str, float]] = []
        self.depth: list[dict[str, float]] = []

    def add(self, bundle_path: Path) -> None:
        arrays = _read_pair(bundle_path)
        context_sem_pred = concat_views_height(list(arrays["context_sem_pred"]))
        context_sem_gt = concat_views_height(list(arrays["context_sem_gt"]))
        context_ins_pred = concat_views_height(list(arrays["context_ins_pred"]))
        context_ins_gt = concat_views_height(list(arrays["context_ins_gt"]))
        target_sem_pred = concat_views_height(list(arrays["target_sem_pred"]))
        target_sem_gt = concat_views_height(list(arrays["target_sem_gt"]))
        target_ins_pred = concat_views_height(list(arrays["target_ins_pred"]))
        target_ins_gt = concat_views_height(list(arrays["target_ins_gt"]))

        c_intersection, c_union = _semantic_counts(context_sem_pred, context_sem_gt)
        t_intersection, t_union = _semantic_counts(target_sem_pred, target_sem_gt)
        self.context_intersection += c_intersection
        self.context_union += c_union
        self.target_intersection += t_intersection
        self.target_union += t_union

        for index in range(TARGET_VIEWS):
            prediction = self.context_backend.torch.from_numpy(arrays["rgb_pred"][index]).unsqueeze(0).to(self.context_backend.device)
            target = self.context_backend.torch.from_numpy(arrays["rgb_gt"][index]).unsqueeze(0).to(self.context_backend.device)
            self.reconstruction.append(self.context_backend.reconstruction_one(prediction, target))
            from tokengs.siu3r_protocol import depth_metrics
            # Bundle depths are already in metres.  The official PNG path
            # divides uint16 millimetres by 1000 exactly once.
            self.depth.append(depth_metrics(arrays["depth_pred"][index], arrays["depth_gt"][index]))

        context_map_pred = _torch_map_input(self.context_backend, context_sem_pred, context_ins_pred, pred=True, labels=arrays["context_pred_ids"], scores=arrays["context_pred_scores"])
        context_map_gt = _torch_map_input(self.context_backend, context_sem_gt, context_ins_gt, pred=False)
        target_map_pred = _torch_map_input(self.target_backend, target_sem_pred, target_ins_pred, pred=True, labels=arrays["target_pred_ids"], scores=arrays["target_pred_scores"])
        target_map_gt = _torch_map_input(self.target_backend, target_sem_gt, target_ins_gt, pred=False)
        self.context_backend.update_map(context_map_pred, context_map_gt)
        self.target_backend.update_map(target_map_pred, target_map_gt)
        self.context_backend.update_pq(np.stack([context_sem_pred, context_ins_pred], axis=-1), np.stack([context_sem_gt, context_ins_gt], axis=-1))
        self.target_backend.update_pq(np.stack([target_sem_pred, target_ins_pred], axis=-1), np.stack([target_sem_gt, target_ins_gt], axis=-1))

    @staticmethod
    def _miou(intersection: np.ndarray, union: np.ndarray) -> tuple[list[float], float]:
        ious = np.divide(intersection[1:], union[1:], out=np.zeros(20, dtype=np.float64), where=union[1:] > 0)
        # Official evaluator.py averages the complete per-class vector,
        # including zero-valued classes with empty union.
        return ious.tolist(), float(ious.mean())

    def compute(self) -> dict[str, Any]:
        context_ious, context_miou = self._miou(self.context_intersection, self.context_union)
        target_ious, target_miou = self._miou(self.target_intersection, self.target_union)
        context_pqs = self.context_backend.compute_pq()
        target_pqs = self.target_backend.compute_pq()
        return {
            "reconstruction": {key: float(np.mean([row[key] for row in self.reconstruction])) for key in ("psnr", "ssim", "lpips")},
            "depth": {key: float(np.mean([row[key] for row in self.depth])) for key in ("absrel", "rmse")},
            "context_ious_per_class": context_ious,
            "context_miou": context_miou,
            "target_ious_per_class": target_ious,
            "target_miou": target_miou,
            "context_pqs_per_class": context_pqs,
            "context_pq": float(np.mean(np.asarray(context_pqs))),
            "target_pqs_per_class": target_pqs,
            "target_pq": float(np.mean(np.asarray(target_pqs))),
            "context_map": self.context_backend.compute_map(),
            "target_map": self.target_backend.compute_map(),
        }


def evaluate_pair(bundle_path: Path, backend: TorchMetricBackend | None) -> dict[str, Any]:
    arrays = _read_pair(bundle_path)
    csp, csg = arrays["context_sem_pred"], arrays["context_sem_gt"]
    cip, cig = arrays["context_ins_pred"], arrays["context_ins_gt"]
    tsp, tsg = arrays["target_sem_pred"], arrays["target_sem_gt"]
    tip, tig = arrays["target_ins_pred"], arrays["target_ins_gt"]
    rgb_pred, rgb_gt = arrays["rgb_pred"], arrays["rgb_gt"]
    depth_pred, depth_gt = arrays["depth_pred"], arrays["depth_gt"]
    context_pred_ids, context_pred_scores = arrays["context_pred_ids"], arrays["context_pred_scores"]
    target_pred_ids, target_pred_scores = arrays["target_pred_ids"], arrays["target_pred_scores"]

    # The official evaluator forms one large H*6-by-W image per pair.  No
    # view-level image identity is passed to the metric.
    context_sem_pred = concat_views_height(list(csp))
    context_sem_gt = concat_views_height(list(csg))
    context_ins_pred = concat_views_height(list(cip))
    context_ins_gt = concat_views_height(list(cig))
    target_sem_pred = concat_views_height(list(tsp))
    target_sem_gt = concat_views_height(list(tsg))
    target_ins_pred = concat_views_height(list(tip))
    target_ins_gt = concat_views_height(list(tig))

    result: dict[str, Any] = {
        "bundle": str(bundle_path),
        "protocol": "siu3r_global_multiview_v1",
        "context_concat_shape": list(context_sem_pred.shape),
        "target_concat_shape": list(target_sem_pred.shape),
        "target_depth": [],
    }
    _, result["context_miou"] = global_semantic_iou([context_sem_pred], [context_sem_gt])
    _, result["target_miou"] = global_semantic_iou([target_sem_pred], [target_sem_gt])
    if backend is None:
        result["metric_backend"] = "unavailable (torchmetrics is required for PSNR/SSIM/LPIPS/mAP/PQ)"
        return result

    for index in range(TARGET_VIEWS):
        prediction = backend.torch.from_numpy(rgb_pred[index]).unsqueeze(0).to(backend.device)
        target = backend.torch.from_numpy(rgb_gt[index]).unsqueeze(0).to(backend.device)
        result.setdefault("reconstruction_per_target", []).append(backend.reconstruction_one(prediction, target))

        from tokengs.siu3r_protocol import depth_metrics
        result["target_depth"].append(depth_metrics(depth_pred[index], depth_gt[index]))

    context_map_pred = _torch_map_input(backend, context_sem_pred, context_ins_pred, pred=True, labels=context_pred_ids, scores=context_pred_scores)
    context_map_gt = _torch_map_input(backend, context_sem_gt, context_ins_gt, pred=False)
    target_map_pred = _torch_map_input(backend, target_sem_pred, target_ins_pred, pred=True, labels=target_pred_ids, scores=target_pred_scores)
    target_map_gt = _torch_map_input(backend, target_sem_gt, target_ins_gt, pred=False)
    backend.update_map(context_map_pred, context_map_gt)
    target_backend = TorchMetricBackend(device=str(backend.device))
    target_backend.update_map(target_map_pred, target_map_gt)
    backend.update_pq(np.stack([context_sem_pred, context_ins_pred], axis=-1), np.stack([context_sem_gt, context_ins_gt], axis=-1))
    target_backend.update_pq(np.stack([target_sem_pred, target_ins_pred], axis=-1), np.stack([target_sem_gt, target_ins_gt], axis=-1))
    result["context_map"] = backend.compute_map()
    result["target_map"] = target_backend.compute_map()
    result["context_pq"] = backend.compute_pq()
    result["target_pq"] = target_backend.compute_pq()
    result["reconstruction"] = {
        key: float(np.mean([row[key] for row in result["reconstruction_per_target"]]))
        for key in ("psnr", "ssim", "lpips")
    }
    result["depth"] = {
        key: float(np.mean([row[key] for row in result["target_depth"]]))
        for key in ("absrel", "rmse")
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=SCHEMA)
    parser.add_argument("--pairs", type=Path, default=ROOT / "workspace/siu3r_protocol_alignment_v1/val_pair.json")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-index", type=int, default=None, help="single official pair for smoke")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-formal-1860", action="store_true", help="required only for a deliberate full run")
    args = parser.parse_args()
    manifest = validate_val_pairs(args.pairs)
    if args.pair_index is None and not args.allow_formal_1860:
        raise SystemExit("refusing dataset-wide evaluation without --allow-formal-1860")
    if args.pair_index is not None and not (0 <= args.pair_index < PAIR_RECORDS):
        raise SystemExit("pair index must be in [0,1859]")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing result: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    indices = [args.pair_index] if args.pair_index is not None else list(range(PAIR_RECORDS))
    results = []
    if args.pair_index is None:
        # The official evaluator updates mIoU/PQ state for every pair and only
        # computes once after all scenes; mAP is likewise updated with the full
        # list of pair dictionaries.  Keep this path separate from pair smoke.
        accumulator = DatasetAccumulator(device=args.device)
        for index in indices:
            bundle = args.predictions_dir / f"pair_{index:04d}.npz"
            if not bundle.is_file():
                raise SystemExit(f"missing prediction bundle for official pair index {index}: {bundle}")
            accumulator.add(bundle)
        results.append(accumulator.compute())
    else:
        backend = None
        try:
            backend = TorchMetricBackend(device=args.device)
        except Exception as exc:
            print(f"warning: exact TorchMetrics backend unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        bundle = args.predictions_dir / f"pair_{args.pair_index:04d}.npz"
        if not bundle.is_file():
            raise SystemExit(f"missing prediction bundle for official pair index {args.pair_index}: {bundle}")
        results.append(evaluate_pair(bundle, backend))
    output = {
        "protocol": "siu3r_global_multiview_v1",
        "manifest": manifest,
        "pair_indices": indices,
        "formal_1860_evaluation_started": args.pair_index is None,
        "results": results,
    }
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
