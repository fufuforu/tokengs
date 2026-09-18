#!/usr/bin/env python3
"""Validate the complete extracted SIU3R ScanNet val tree without mutation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


REQUIRED_DIRS = ("color", "depth", "extrinsic", "instance", "panoptic", "semantic")
FRAME_DIRS = ("color", "depth", "extrinsic", "instance", "panoptic", "semantic")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image(path: Path) -> np.ndarray:
    with Image.open(path) as obj:
        obj.load()
        return np.asarray(obj).copy()


def encoded_segment_id(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected RGB encoded segment map, got {array.shape}")
    values = array.astype(np.int64)
    return values[..., 0] + 256 * values[..., 1] + 256 * 256 * values[..., 2]


def finite_matrix(path: Path, shape: tuple[int, int]) -> None:
    value = np.loadtxt(path)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"invalid matrix {path}: shape={value.shape}")


def check_scene(scene_dir: Path, frame_ids: set[int]) -> dict:
    errors: list[str] = []
    for name in REQUIRED_DIRS:
        if not (scene_dir / name).is_dir():
            errors.append(f"missing directory {name}")
    if not (scene_dir / "intrinsic.txt").is_file():
        errors.append("missing intrinsic.txt")
    else:
        try:
            finite_matrix(scene_dir / "intrinsic.txt", (4, 4))
        except Exception as exc:
            errors.append(str(exc))
    if (scene_dir / "iou.pt").is_file():
        try:
            import torch
            value = torch.load(scene_dir / "iou.pt", map_location="cpu", weights_only=True)
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            if not np.isfinite(np.asarray(value, dtype=np.float64)).all():
                errors.append("iou.pt contains NaN/Inf")
        except Exception as exc:
            errors.append(f"iou.pt unreadable: {exc}")
    else:
        errors.append("missing iou.pt")
    if not (scene_dir / "iou.png").is_file():
        errors.append("missing iou.png")

    checked_frames = 0
    for frame_id in sorted(frame_ids):
        stem = str(frame_id)
        paths = {name: scene_dir / name / (stem + (".jpg" if name == "color" else ".txt" if name == "extrinsic" else ".png")) for name in FRAME_DIRS}
        if any(not path.is_file() for path in paths.values()):
            errors.extend(f"missing frame {frame_id} {name}" for name, path in paths.items() if not path.is_file())
            continue
        try:
            color = image(paths["color"])
            depth = image(paths["depth"])
            semantic = image(paths["semantic"])
            instance = image(paths["instance"])
            panoptic = image(paths["panoptic"])
            if color.shape != (256, 256, 3):
                errors.append(f"frame {frame_id} color shape {color.shape}")
            if depth.shape != (256, 256):
                errors.append(f"frame {frame_id} depth shape {depth.shape}")
            if semantic.shape != (256, 256):
                errors.append(f"frame {frame_id} semantic not single-channel 256x256: {semantic.shape}")
            if semantic.size == 0 or not np.isfinite(semantic.astype(np.float64)).all() or semantic.min() < 0 or semantic.max() > 20:
                errors.append(f"frame {frame_id} semantic invalid value range")
            if instance.shape[:2] != (256, 256) or panoptic.shape[:2] != (256, 256):
                errors.append(f"frame {frame_id} encoded label spatial mismatch")
            if encoded_segment_id(instance).size == 0 or encoded_segment_id(panoptic).size == 0:
                errors.append(f"frame {frame_id} empty encoded labels")
            finite_matrix(paths["extrinsic"], (4, 4))
            checked_frames += 1
        except Exception as exc:
            errors.append(f"frame {frame_id}: {exc}")
    return {"scene": scene_dir.name, "referenced_frames": len(frame_ids), "checked_frames": checked_frames, "errors": errors, "integrity_status": "valid" if not errors and checked_frames == len(frame_ids) else "invalid"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.pairs.read_text())
    scenes: dict[str, set[int]] = {}
    for index, record in enumerate(records):
        if len(record.get("context_ids", [])) != 2 or len(record.get("target_ids", [])) != 6:
            raise SystemExit(f"invalid pair cardinality at index {index}")
        scene = str(record["scan"])
        scenes.setdefault(scene, set()).update(int(x) for x in record["context_ids"] + record["target_ids"])
    reports = []
    for scene, frame_ids in sorted(scenes.items()):
        reports.append(check_scene(args.data_root / "val" / scene, frame_ids))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root),
        "pair_file": str(args.pairs),
        "pair_records": len(records),
        "scene_count": len(reports),
        "all_referenced_frames_found": all(item["checked_frames"] == item["referenced_frames"] for item in reports),
        "data_integrity_valid": len(reports) == 312 and all(item["integrity_status"] == "valid" for item in reports),
        "scenes": reports,
    }
    target = args.result / "data_integrity_report.json"
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(target)
    md = args.result / "data_integrity_report.md"
    lines = ["# SIU3R processed ScanNet validation integrity", "", f"- pair records: {len(records)}", f"- scenes: {len(reports)}", f"- all referenced frames found: {payload['all_referenced_frames_found']}", f"- data integrity valid: {payload['data_integrity_valid']}", ""]
    bad = [item for item in reports if item["integrity_status"] != "valid"]
    if bad:
        lines.append("## Failures")
        for item in bad:
            lines.append(f"- `{item['scene']}`: " + "; ".join(item["errors"]))
    else:
        lines.append("All 312 referenced scenes and frames passed the read-only checks.")
    md.write_text("\n".join(lines) + "\n")
    print(json.dumps({key: payload[key] for key in ("pair_records", "scene_count", "all_referenced_frames_found", "data_integrity_valid")}, indent=2))
    return 0 if payload["data_integrity_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
