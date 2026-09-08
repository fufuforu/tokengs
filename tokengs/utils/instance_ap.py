"""Class-agnostic 2D instance AP utilities for the LSM ScanNet protocol.

The protocol (shared by LSM, Gaussian Grouping, ObjectGS, IGGT+LUDVIG and
InstOk3D) evaluates rendered novel-view instance masks against ScanNet 2D
instance labels. Predictions are processed by descending confidence and
matched to the highest-IoU unmatched GT from the *same image*. Image ids are
supported explicitly so multi-view/multi-scene evaluation cannot match masks
from unrelated pixel coordinate systems. This module only contains the
matching/metrics machinery, so it can be unit-tested independently of the
model.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Sequence

import numpy as np


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU between two boolean 2D masks."""
    intersection = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    if union <= 0.0:
        return 0.0
    return intersection / union


def iou_matrix(
    pred_masks: Sequence[np.ndarray], gt_masks: Sequence[np.ndarray]
) -> np.ndarray:
    """IoU matrix of shape [num_pred, num_gt]."""
    if not pred_masks or not gt_masks:
        return np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    matrix = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for i, pred in enumerate(pred_masks):
        for j, gt in enumerate(gt_masks):
            matrix[i, j] = mask_iou(pred, gt)
    return matrix


def iou_matrix_vectorized(
    pred_masks: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    chunk: int = 512,
) -> np.ndarray:
    """IoU matrix computed with a chunked GPU matmul (fast for large sets).

    Masks are flattened boolean vectors; IoU = intersection / union with
    union = |pred| + |gt| - intersection. Falls back to the CPU loop when
    CUDA is unavailable.
    """
    if not pred_masks or not gt_masks:
        return np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no cuda")
        pred_np = np.asarray(pred_masks, dtype=np.bool_)
        gt_np = np.asarray(gt_masks, dtype=np.bool_)
        pred = torch.from_numpy(pred_np.reshape(len(pred_np), -1)).cuda()
        gt = torch.from_numpy(gt_np.reshape(len(gt_np), -1)).cuda()
        pred_f = pred.float()
        gt_f = gt.float()
        pred_sizes = pred_f.sum(dim=1)
        gt_sizes = gt_f.sum(dim=1)
        rows = []
        with torch.inference_mode():
            for start in range(0, pred_f.shape[0], chunk):
                block = pred_f[start : start + chunk]
                inter = block @ gt_f.t()
                union = (
                    pred_sizes[start : start + chunk, None]
                    + gt_sizes[None, :]
                    - inter
                )
                iou = torch.where(
                    union > 0.0,
                    inter / union.clamp_min(1.0),
                    torch.zeros_like(inter),
                )
                rows.append(iou.cpu().numpy().astype(np.float32))
        return np.concatenate(rows, axis=0)
    except Exception:
        return iou_matrix(pred_masks, gt_masks)


def _interp_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """COCO-style AP: 101-point recall interpolation."""
    if precision.size == 0:
        return 0.0
    # Sort by recall ascending and take max precision to the right.
    order = np.argsort(recall)
    recall_sorted = recall[order]
    precision_sorted = precision[order]
    max_precision = np.maximum.accumulate(precision_sorted[::-1])[::-1]
    recall_grid = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for r in recall_grid:
        idx = np.searchsorted(recall_sorted, r, side="left")
        if idx >= len(max_precision):
            p = 0.0
        else:
            p = float(max_precision[idx])
        ap += p / 101.0
    return ap


def precision_recall_curve(
    pred_masks: Sequence[np.ndarray],
    pred_scores: Sequence[float],
    gt_masks: Sequence[np.ndarray],
    iou_threshold: float,
    vectorized: bool = False,
    pred_image_ids: Sequence[Hashable] | None = None,
    gt_image_ids: Sequence[Hashable] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a PR curve for one threshold.

    Predictions are sorted by confidence descending; each prediction is
    matched greedily to the highest-IoU unmatched GT from the same image,
    producing one TP/FP decision per prediction. ``pred_image_ids`` and
    ``gt_image_ids`` must be supplied together for multi-image evaluation.
    """
    if len(pred_masks) != len(pred_scores):
        raise ValueError("pred_masks and pred_scores must have equal length")
    if (pred_image_ids is None) != (gt_image_ids is None):
        raise ValueError(
            "pred_image_ids and gt_image_ids must be provided together"
        )
    pred_ids = (
        [0] * len(pred_masks)
        if pred_image_ids is None
        else list(pred_image_ids)
    )
    gt_ids = (
        [0] * len(gt_masks)
        if gt_image_ids is None
        else list(gt_image_ids)
    )
    if len(pred_ids) != len(pred_masks) or len(gt_ids) != len(gt_masks):
        raise ValueError("image-id arrays must match their mask arrays")
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return np.array([0.0]), np.array([1.0])

    order = np.argsort(-np.asarray(pred_scores, dtype=np.float32), kind="stable")
    gt_by_image: dict[Hashable, list[int]] = {}
    for gt_idx, image_id in enumerate(gt_ids):
        gt_by_image.setdefault(image_id, []).append(gt_idx)
    unmatched_gt = {
        image_id: set(indices) for image_id, indices in gt_by_image.items()
    }
    pred_by_image: dict[Hashable, list[int]] = {}
    for pred_idx, image_id in enumerate(pred_ids):
        pred_by_image.setdefault(image_id, []).append(pred_idx)
    matrices: dict[Hashable, tuple[list[int], list[int], np.ndarray]] = {}
    for image_id, pred_indices in pred_by_image.items():
        gt_indices = gt_by_image.get(image_id, [])
        if not gt_indices:
            continue
        pred_subset = [pred_masks[i] for i in pred_indices]
        gt_subset = [gt_masks[i] for i in gt_indices]
        matrix = (
            iou_matrix_vectorized(pred_subset, gt_subset)
            if vectorized
            else iou_matrix(pred_subset, gt_subset)
        )
        matrices[image_id] = (pred_indices, gt_indices, matrix)

    tp_cum = 0
    fp_cum = 0
    total_gt = len(gt_masks)
    precision: list[float] = []
    recall: list[float] = []
    for pred_idx in order:
        image_id = pred_ids[int(pred_idx)]
        entry = matrices.get(image_id)
        matched_gt_idx = None
        if entry is not None and unmatched_gt.get(image_id):
            pred_indices, gt_indices, matrix = entry
            local_pred = pred_indices.index(int(pred_idx))
            candidates = sorted(unmatched_gt[image_id])
            local_candidates = [gt_indices.index(gt_idx) for gt_idx in candidates]
            best_local = max(
                local_candidates,
                key=lambda local_gt: float(matrix[local_pred, local_gt]),
            )
            if float(matrix[local_pred, best_local]) >= float(iou_threshold):
                matched_gt_idx = gt_indices[best_local]
        if matched_gt_idx is not None:
            tp_cum += 1
            unmatched_gt[image_id].discard(matched_gt_idx)
        else:
            fp_cum += 1
        precision.append(tp_cum / max(1, tp_cum + fp_cum))
        recall.append(tp_cum / max(1, total_gt))
    return np.asarray(precision, dtype=np.float32), np.asarray(
        recall, dtype=np.float32
    )


def instance_ap(
    pred_masks: Sequence[np.ndarray],
    pred_scores: Sequence[float],
    gt_masks: Sequence[np.ndarray],
    thresholds: Iterable[float] = (0.25, 0.5, 0.75),
    vectorized: bool = False,
    pred_image_ids: Sequence[Hashable] | None = None,
    gt_image_ids: Sequence[Hashable] | None = None,
) -> dict[str, float]:
    """AP at several IoU thresholds, plus the mean over thresholds."""
    results: dict[str, float] = {}
    ap_values = []
    for threshold in thresholds:
        precision, recall = precision_recall_curve(
            pred_masks,
            pred_scores,
            gt_masks,
            float(threshold),
            vectorized=vectorized,
            pred_image_ids=pred_image_ids,
            gt_image_ids=gt_image_ids,
        )
        ap = _interp_pr(precision, recall)
        results[f"ap_{int(round(threshold * 100)):02d}"] = ap
        ap_values.append(ap)
    results["ap_mean"] = float(np.mean(ap_values))
    return results


def masks_from_group_probs(
    group_probs: np.ndarray,
    void_channel: int | None = None,
    min_mask_area: int = 0,
) -> tuple[list[np.ndarray], list[float]]:
    """Turn a per-pixel group probability map into predicted instance masks.

    ``group_probs`` is [G(+1), H, W] in [0, 1] (already alpha-composited and
    normalized over groups, the last channel being void when provided).
    Each non-void group's pixels are collected by argmax; the confidence of
    an instance is the mean probability of that group over its pixels.
    Returns (masks, scores) sorted by score descending.
    """
    num_groups, height, width = group_probs.shape
    if num_groups == 0:
        return [], []
    group_ids = np.argmax(group_probs, axis=0)
    scores = np.max(group_probs, axis=0)
    masks = []
    confidences = []
    for group_id in range(num_groups):
        if void_channel is not None and group_id == void_channel:
            continue
        mask = group_ids == group_id
        if int(mask.sum()) < max(1, int(min_mask_area)):
            continue
        confidence = float(scores[mask].mean())
        masks.append(mask)
        confidences.append(confidence)
    order = np.argsort(-np.asarray(confidences, dtype=np.float32), kind="stable")
    return [masks[i] for i in order], [confidences[i] for i in order]


def gt_masks_from_instance_map(
    instance_map: np.ndarray,
    ignore_ids: Sequence[int] = (0, 255, -1),
    min_mask_area: int = 0,
) -> list[np.ndarray]:
    """Split a 2D instance-id label map into per-instance boolean masks."""
    ignore = set(int(value) for value in ignore_ids)
    masks = []
    for instance_id in np.unique(instance_map):
        if int(instance_id) in ignore:
            continue
        mask = instance_map == instance_id
        if int(mask.sum()) >= max(1, int(min_mask_area)):
            masks.append(mask)
    return masks
