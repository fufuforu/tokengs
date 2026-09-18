"""Build a parity bundle from the exact files consumed by SIU3R Evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _panoptic(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(path))
    encoded = rgb[..., 0].astype(np.int64) + 256 * rgb[..., 1].astype(np.int64) + 65536 * rgb[..., 2].astype(np.int64)
    return encoded // 1000, encoded % 1000


def _stack_panoptic(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [_panoptic(path) for path in sorted(directory.glob("*.png"))]
    if not rows:
        raise ValueError(f"no PNG files in {directory}")
    return np.stack([row[0] for row in rows]), np.stack([row[1] for row in rows])


def _stack_depth(directory: Path) -> np.ndarray:
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise ValueError(f"no PNG files in {directory}")
    return np.stack([np.asarray(Image.open(path), dtype=np.float32) / 1000.0 for path in paths])


def _stack_rgb(directory: Path) -> np.ndarray:
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise ValueError(f"no PNG files in {directory}")
    return np.stack([np.asarray(Image.open(path), dtype=np.float32).transpose(2, 0, 1) / 255.0 for path in paths])


def _metadata(directory: Path, instance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = json.loads((directory / "pred.json").read_text(encoding="utf-8"))
    by_id = {int(row["id"]): row for row in rows}
    ids = sorted(int(value) for value in np.unique(instance) if int(value) != 0)
    if set(ids) != set(by_id):
        raise ValueError(f"{directory}: pred.json IDs do not match PNG IDs")
    return (
        np.asarray([int(by_id[value]["label_id"]) for value in ids], dtype=np.int64),
        np.asarray([float(by_id[value]["score"]) for value in ids], dtype=np.float32),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    context_sem_pred, context_instance_pred = _stack_panoptic(args.scene_dir / "context_seg_pred")
    context_sem_gt, context_instance_gt = _stack_panoptic(args.scene_dir / "context_seg_gt")
    target_sem_pred, target_instance_pred = _stack_panoptic(args.scene_dir / "target_seg_pred")
    target_sem_gt, target_instance_gt = _stack_panoptic(args.scene_dir / "target_seg_gt")
    context_labels, context_scores = _metadata(args.scene_dir / "context_seg_pred", context_instance_pred)
    target_labels, target_scores = _metadata(args.scene_dir / "target_seg_pred", target_instance_pred)
    payload = {
        "context_semantic_pred": context_sem_pred,
        "context_instance_pred": context_instance_pred,
        "context_semantic_gt": context_sem_gt,
        "context_instance_gt": context_instance_gt,
        "target_semantic_pred": target_sem_pred,
        "target_instance_pred": target_instance_pred,
        "target_semantic_gt": target_sem_gt,
        "target_instance_gt": target_instance_gt,
        "target_rgb_pred": _stack_rgb(args.scene_dir / "rgb"),
        "target_rgb_gt": _stack_rgb(args.scene_dir / "rgb_gt"),
        "target_depth_pred": _stack_depth(args.scene_dir / "depth"),
        "target_depth_gt": _stack_depth(args.scene_dir / "depth_gt"),
        "context_pred_labels": context_labels,
        "context_pred_scores": context_scores,
        "target_pred_labels": target_labels,
        "target_pred_scores": target_scores,
    }
    shapes = {key: list(value.shape) for key, value in payload.items()}
    if any(value.shape[-2:] != (256, 256) for value in payload.values() if value.ndim >= 2):
        raise ValueError(f"saved SIU3R output is not 256x256: {shapes}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "shapes": shapes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
