"""Isolated SIU3R Official Protocol Alignment v1 primitives.

This module intentionally has no dependency on the legacy LSM evaluator.  The
array-level functions are usable by CPU audits and tests; the exact official
TorchMetrics calls are imported lazily by ``TorchMetricBackend`` so a missing
GPU/ML environment cannot silently change protocol semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json

import numpy as np


PROTOCOL = "siu3r_global_multiview_v1"
PAIR_SHA256 = "59cf5594ec2f3223a41d27d7af4da49d348ce9b00c1d151020378c9a8f4b4b3b"
PAIR_RECORDS = 1860
PAIR_SCENES = 312
CONTEXT_VIEWS = 2
TARGET_VIEWS = 6
IMAGE_SIZE = (256, 256)

PANOPTIC_CLASSES = {
    0: "unlabeled",
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "desk",
    14: "curtain",
    15: "refrigerator",
    16: "shower curtain",
    17: "toilet",
    18: "sink",
    19: "bathtub",
    20: "otherfurniture",
}
VALID_SEMANTIC_IDS = tuple(range(1, 21))
STUFF_IDS = (1, 2)  # wall, floor
THING_IDS = tuple(range(3, 21))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_val_pairs(path: str | Path) -> dict[str, Any]:
    """Validate the official manifest without changing record order or IDs."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("val_pair.json must be a list")
    scenes: set[str] = set()
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} is not an object")
        if set(("scan", "context_ids", "target_ids")) - set(record):
            raise ValueError(f"record {index} has an invalid schema")
        context = record["context_ids"]
        target = record["target_ids"]
        if len(context) != CONTEXT_VIEWS or len(target) != TARGET_VIEWS:
            raise ValueError(f"record {index} does not have 2 context / 6 target IDs")
        if not all(isinstance(value, int) for value in context + target):
            raise ValueError(f"record {index} contains non-integer view IDs")
        if len(set(context)) != CONTEXT_VIEWS or len(set(target)) != TARGET_VIEWS:
            raise ValueError(f"record {index} contains duplicate view IDs")
        if not set(context).issubset(set(target)) or len(set(target) - set(context)) != TARGET_VIEWS - CONTEXT_VIEWS:
            raise ValueError(f"record {index} does not contain both context IDs in targets")
        scenes.add(str(record["scan"]))
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "sha256_matches": sha256_file(path) == PAIR_SHA256,
        "records": len(payload),
        "records_matches": len(payload) == PAIR_RECORDS,
        "unique_scenes": len(scenes),
        "scenes_matches": len(scenes) == PAIR_SCENES,
        "context_views": CONTEXT_VIEWS,
        "target_views": TARGET_VIEWS,
        "resolution": list(IMAGE_SIZE),
        "cardinality_matches": len(payload) == PAIR_RECORDS and len(scenes) == PAIR_SCENES,
    }


def normalize_intrinsics(intrinsics: np.ndarray, width: int = 256, height: int = 256) -> np.ndarray:
    """Official SIU3R normalization: row 0 / width, row 1 / height."""
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"expected [...,3,3] intrinsics, got {matrix.shape}")
    output = np.zeros_like(matrix)
    output[..., 0, 0] = matrix[..., 0, 0] / width
    output[..., 0, 2] = matrix[..., 0, 2] / width
    output[..., 1, 1] = matrix[..., 1, 1] / height
    output[..., 1, 2] = matrix[..., 1, 2] / height
    output[..., 2, 2] = 1.0
    return output


def relative_opencv_c2w(
    context_extrinsics: Sequence[np.ndarray], target_extrinsics: Sequence[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Make the first OpenCV camera-to-world matrix the canonical frame."""
    if len(context_extrinsics) != CONTEXT_VIEWS:
        raise ValueError("SIU3R requires exactly two context extrinsics")
    canonical_inv = np.linalg.inv(np.asarray(context_extrinsics[0]))
    return (
        [canonical_inv @ np.asarray(extrinsic) for extrinsic in context_extrinsics],
        [canonical_inv @ np.asarray(extrinsic) for extrinsic in target_extrinsics],
    )


def decode_panoptic_rgb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode SIU3R RGB-encoded ``1000 * semantic + instance`` maps."""
    encoded = np.asarray(rgb)
    if encoded.shape[-1] != 3:
        raise ValueError(f"expected [...,3] RGB panoptic map, got {encoded.shape}")
    encoded = encoded.astype(np.int64)
    segment_id = encoded[..., 0] + 256 * encoded[..., 1] + 256 * 256 * encoded[..., 2]
    return segment_id // 1000, segment_id % 1000


def concat_views_height(views: Sequence[np.ndarray]) -> np.ndarray:
    """Official multi-view spatial concatenation; view order is preserved."""
    if not views:
        raise ValueError("at least one view is required")
    arrays = [np.asarray(view) for view in views]
    if any(array.ndim < 2 for array in arrays):
        raise ValueError("views must have height and width dimensions")
    return np.concatenate(arrays, axis=-2)


def global_semantic_iou(
    predictions: Iterable[np.ndarray], targets: Iterable[np.ndarray]
) -> tuple[np.ndarray, float]:
    """Exact SIU3R ``MeanIoU`` state accumulation, excluding background."""
    intersection = np.zeros(21, dtype=np.int64)
    union = np.zeros(21, dtype=np.int64)
    for prediction, target in zip(predictions, targets):
        prediction = np.asarray(prediction)
        target = np.asarray(target)
        if prediction.shape != target.shape:
            raise ValueError("semantic prediction and target shapes differ")
        for semantic_id in range(21):
            if semantic_id == 0:
                continue
            pred = prediction == semantic_id
            truth = target == semantic_id
            intersection[semantic_id] += np.logical_and(pred, truth).sum()
            union[semantic_id] += np.logical_or(pred, truth).sum()
    ious = np.divide(
        intersection[1:], union[1:], out=np.zeros(20, dtype=np.float64), where=union[1:] > 0
    )
    # SIU3R's official evaluator requests per_class=True and then applies
    # np.mean to the complete 20-class vector. Empty-union classes therefore
    # contribute zero to the reported mean.
    return ious, float(ious.mean())


def fit_scale_and_shift(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Exact SIU3R depth alignment: least squares on target > 0 only."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = target > 0
    if not np.any(valid):
        raise ValueError("depth target has no positive pixels")
    values = prediction[valid]
    truth = target[valid]
    design = np.stack([values, np.ones_like(values)], axis=1)
    scale, shift = np.linalg.lstsq(design, truth, rcond=None)[0]
    return float(scale), float(shift)


def depth_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    scale, shift = fit_scale_and_shift(prediction, target)
    prediction = np.asarray(prediction, dtype=np.float64) * scale + shift
    target = np.asarray(target, dtype=np.float64)
    valid = target > 0
    residual = prediction[valid] - target[valid]
    return {
        "scale": scale,
        "shift": shift,
        "absrel": float(np.mean(np.abs(residual) / target[valid])),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


@dataclass(frozen=True)
class PanopticRecord:
    semantic: np.ndarray
    instance: np.ndarray

    def tensor(self) -> np.ndarray:
        return np.stack([self.semantic, self.instance], axis=-1)


def build_instance_masks(
    semantic: np.ndarray,
    instance: np.ndarray,
    *,
    prediction: bool,
    labels: Mapping[int, int] | None = None,
    scores: Mapping[int, float] | None = None,
) -> dict[str, list[Any]]:
    """Build class-aware TorchMetrics inputs from global-ID panoptic maps."""
    semantic = np.asarray(semantic)
    instance = np.asarray(instance)
    ids = []
    masks = []
    out_labels = []
    out_scores = []
    for raw_id in np.unique(instance):
        instance_id = int(raw_id)
        if instance_id == 0:
            continue
        mask = instance == instance_id
        if not np.any(mask):
            continue
        semantic_id = int(np.unique(semantic[mask])[0])
        # Official SIU3R excludes wall/floor only from instance GT.  Predicted
        # stuff masks remain predictions (and can therefore be penalized as
        # false positives), exactly as process_segmentation does.
        if not prediction and (semantic_id in STUFF_IDS or semantic_id == 0):
            continue
        if prediction and labels is None:
            raise ValueError("SIU3R mAP requires explicit class-aware prediction labels")
        if prediction and scores is None:
            raise ValueError("SIU3R mAP requires explicit prediction confidence scores")
        ids.append(instance_id)
        masks.append(mask)
        out_labels.append(int(labels[instance_id] if labels is not None else semantic_id - 1))
        if prediction:
            out_scores.append(float(scores[instance_id]))
    output = {"masks": masks, "labels": out_labels}
    if prediction:
        output["scores"] = out_scores
    return output


class TorchMetricBackend:
    """Thin exact wrapper around the SIU3R evaluator's TorchMetrics calls."""

    def __init__(
        self,
        device: str = "cpu",
        *,
        image_metrics: bool = True,
        segmentation_metrics: bool = True,
    ) -> None:
        import torch
        from torchmetrics.detection import MeanAveragePrecision, PanopticQuality
        from torchmetrics.image import (
            LearnedPerceptualImagePatchSimilarity,
            PeakSignalNoiseRatio,
            StructuralSimilarityIndexMeasure,
        )

        self.torch = torch
        self.device = torch.device(device)
        self.psnr = self.ssim = self.lpips = None
        self.map = self.pq = None
        if image_metrics:
            self.psnr = PeakSignalNoiseRatio(sync_on_compute=False).to(device)
            self.ssim = StructuralSimilarityIndexMeasure(sync_on_compute=False).to(device)
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                "vgg", normalize=True, sync_on_compute=False
            ).to(device)
        if segmentation_metrics:
            self.map = MeanAveragePrecision(
                iou_type="segm", class_metrics=True, sync_on_compute=False
            ).to(device)
            self.pq = PanopticQuality(
                things=list(THING_IDS),
                stuffs=list(STUFF_IDS),
                return_per_class=True,
                allow_unknown_preds_category=True,
                sync_on_compute=False,
            ).to(device)

    def reconstruction_one(self, prediction: Any, target: Any) -> dict[str, float]:
        # Keep one metric call per target image, matching official evaluator.py:251-268.
        if self.psnr is None or self.ssim is None or self.lpips is None:
            raise RuntimeError("image metrics were disabled for this backend")
        return {
            "psnr": float(self.psnr(prediction, target).item()),
            "ssim": float(self.ssim(prediction, target).item()),
            "lpips": float(self.lpips(prediction, target).item()),
        }

    def update_map(self, prediction: dict[str, Any], target: dict[str, Any]) -> None:
        if self.map is None:
            raise RuntimeError("segmentation metrics were disabled for this backend")
        self.map.update([prediction], [target])

    def update_pq(self, prediction: np.ndarray, target: np.ndarray) -> None:
        if self.pq is None:
            raise RuntimeError("segmentation metrics were disabled for this backend")
        pred = self.torch.from_numpy(np.asarray(prediction)).unsqueeze(0).to(self.device)
        truth = self.torch.from_numpy(np.asarray(target)).unsqueeze(0).to(self.device)
        self.pq.update(pred, truth)

    def compute_map(self) -> dict[str, Any]:
        if self.map is None:
            raise RuntimeError("segmentation metrics were disabled for this backend")
        output = {}
        for key, value in self.map.compute().items():
            output[key] = value.detach().cpu().tolist() if hasattr(value, "detach") else value
        return output

    def compute_pq(self) -> Any:
        if self.pq is None:
            raise RuntimeError("segmentation metrics were disabled for this backend")
        result = self.pq.compute()
        return result.detach().cpu().tolist() if hasattr(result, "detach") else result


def require_official_prediction_contract(record: Mapping[str, Any]) -> None:
    """Reject prediction bundles that could hide semantic/oracle substitutions."""
    required = {"scene", "context_ids", "target_ids", "target_rgb", "target_depth"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"prediction bundle missing required fields: {missing}")
    if len(record["context_ids"]) != CONTEXT_VIEWS or len(record["target_ids"]) != TARGET_VIEWS:
        raise ValueError("prediction bundle must use the official 2+6 view lists")
    if "oracle" in record or "ttt" in record or "p_u" in record:
        raise ValueError("oracle/TTT/p_u fields are forbidden in SIU3R evaluation")
