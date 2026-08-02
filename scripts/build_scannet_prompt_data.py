#!/usr/bin/env python3
"""Build a provisional non-eval ScanNet split and metadata-only image-query bank."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from tokengs.data.static.scannet import ScanNet


def _numeric_pngs(path: Path) -> dict[int, Path]:
    return {
        int(item.stem): item
        for item in path.glob("*.png")
        if item.stem.isdigit()
    }


def _select_evenly(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, num=count, dtype=int)
    return [values[index] for index in sorted(set(indices.tolist()))]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _candidate_metadata(
    scene: str,
    raw_frame_id: int,
    class_id: int,
    mask: np.ndarray,
    instance_id: int | None,
    source_type: str,
    filters: dict,
) -> dict | None:
    mask_area = int(mask.sum())
    if mask_area < int(filters["min_mask_area"]):
        return None
    x0, y0, x1, y1 = _bbox(mask)
    bbox_w, bbox_h = x1 - x0, y1 - y0
    if min(bbox_w, bbox_h) < int(filters["min_bbox_side"]):
        return None
    if max(bbox_w / bbox_h, bbox_h / bbox_w) > float(filters["max_aspect_ratio"]):
        return None
    foreground_ratio = mask_area / float(bbox_w * bbox_h)
    if foreground_ratio < float(filters["min_foreground_ratio"]):
        return None
    height, width = mask.shape
    border_edges = sum((x0 == 0, y0 == 0, x1 == width, y1 == height))
    if source_type == "instance" and border_edges > int(filters["max_object_border_edges"]):
        return None
    return {
        "scene": scene,
        "raw_frame_id": raw_frame_id,
        "class_id": class_id,
        "bbox": [x0, y0, x1, y1],
        "mask_area": mask_area,
        "image_size": [width, height],
        "instance_id": instance_id,
        "source_type": source_type,
        "foreground_ratio": foreground_ratio,
        "border_edges": border_edges,
    }


def extract_frame_entries(
    scene: str,
    raw_frame_id: int,
    semantic_raw: np.ndarray,
    instances: np.ndarray,
    label_lut: np.ndarray,
    unknown_label_id: int,
    filters: dict,
) -> list[dict]:
    mapped = np.full(semantic_raw.shape, unknown_label_id, dtype=np.int64)
    valid = (semantic_raw >= 0) & (semantic_raw < len(label_lut))
    mapped[valid] = label_lut[semantic_raw[valid]]
    entries = []

    for class_id in (1, 2, 3):
        mask = mapped == class_id
        if mask.any():
            entry = _candidate_metadata(
                scene, raw_frame_id, class_id, mask, None, "semantic", filters
            )
            if entry is not None:
                entries.append(entry)

    for instance_id in np.unique(instances):
        instance_id = int(instance_id)
        if instance_id == 0:
            continue
        instance_mask = instances == instance_id
        instance_area = int(instance_mask.sum())
        if instance_area < int(filters["min_mask_area"]):
            continue
        class_counts = np.bincount(mapped[instance_mask], minlength=9)
        class_id = int(np.argmax(class_counts[1:])) + 1
        if class_id not in (4, 5, 6, 7):
            continue
        semantic_purity = int(class_counts[class_id]) / float(instance_area)
        if semantic_purity < float(filters["min_semantic_purity"]):
            continue
        mask = instance_mask & (mapped == class_id)
        entry = _candidate_metadata(
            scene,
            raw_frame_id,
            class_id,
            mask,
            instance_id,
            "instance",
            filters,
        )
        if entry is not None:
            entry["semantic_purity"] = semantic_purity
            entries.append(entry)
    return entries


def _load_eval_scenes(path: Path) -> list[str]:
    return list(json.loads(path.read_text(encoding="utf-8")))


def build(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol)
    with protocol_path.open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    training_cfg = protocol["training"]
    filters = protocol["query_bank_filters"]
    if args.max_frames_per_scene is not None:
        filters["max_frames_per_scene"] = args.max_frames_per_scene

    scan_root = Path(args.scan_root)
    label_root = Path(args.label_root)
    eval_manifest = Path(training_cfg["eval_manifest"])
    eval_scenes = set(_load_eval_scenes(eval_manifest))
    scan_scenes = sorted(
        path.name
        for path in scan_root.glob("scene*")
        if path.is_dir() and (path / f"{path.name}.sens").is_file()
    )

    official_train_candidates = [
        scan_root.parent / "scannetv2_train.txt",
        scan_root.parent / "Tasks" / "Benchmark" / "scannetv2_train.txt",
        label_root.parent / "scannetv2_train.txt",
    ]
    official_train_path = next(
        (path for path in official_train_candidates if path.is_file()), None
    )
    if official_train_path is None:
        source = "all available scans minus C3G8 eval scenes"
        train_scenes = [scene for scene in scan_scenes if scene not in eval_scenes]
        provisional = True
    else:
        source = str(official_train_path)
        official_scenes = {
            line.strip()
            for line in official_train_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        train_scenes = sorted(official_scenes & set(scan_scenes) - eval_scenes)
        provisional = False

    manifest = {
        "version": 1,
        "split": "train",
        "provisional": provisional,
        "source": source,
        "scan_root": str(scan_root.resolve()),
        "label_root": str(label_root.resolve()),
        "excluded_eval_manifest": str(eval_manifest.resolve()),
        "excluded_eval_scenes": sorted(eval_scenes),
        "scenes": train_scenes,
    }
    manifest_path = Path(training_cfg["train_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {'provisional' if provisional else 'official'} train manifest: "
        f"{manifest_path} ({len(train_scenes)} scenes)",
        flush=True,
    )

    label_lut, unknown_label_id, _ = ScanNet._load_c3g8_protocol(protocol_path)
    all_entries = []
    skipped_missing = []
    per_scene_class_counts: dict[str, Counter] = defaultdict(Counter)
    max_per_class = int(filters["max_entries_per_class_per_scene"])
    for scene_index, scene in enumerate(train_scenes, start=1):
        semantic_paths = _numeric_pngs(label_root / scene / "label-filt")
        instance_paths = _numeric_pngs(label_root / scene / "instance-filt")
        common_frames = sorted(set(semantic_paths) & set(instance_paths))
        if not common_frames:
            skipped_missing.append(scene)
            continue
        selected_frames = _select_evenly(
            common_frames, int(filters["max_frames_per_scene"])
        )
        for raw_frame_id in selected_frames:
            semantic_raw = np.asarray(Image.open(semantic_paths[raw_frame_id]))
            instances = np.asarray(Image.open(instance_paths[raw_frame_id]))
            if semantic_raw.shape != instances.shape:
                continue
            frame_entries = extract_frame_entries(
                scene,
                raw_frame_id,
                semantic_raw,
                instances,
                label_lut,
                unknown_label_id,
                filters,
            )
            for entry in frame_entries:
                class_id = entry["class_id"]
                if per_scene_class_counts[scene][class_id] >= max_per_class:
                    continue
                per_scene_class_counts[scene][class_id] += 1
                all_entries.append(entry)
        if scene_index % 100 == 0 or scene_index == len(train_scenes):
            print(
                f"Processed {scene_index}/{len(train_scenes)} scenes, "
                f"bank entries={len(all_entries)}",
                flush=True,
            )

    distribution = Counter(entry["class_id"] for entry in all_entries)
    bank = {
        "version": 1,
        "metadata_only": True,
        "train_manifest": str(manifest_path.resolve()),
        "excluded_eval_scenes": sorted(eval_scenes),
        "filters": filters,
        "skipped_missing_scenes": skipped_missing,
        "class_distribution": {str(key): distribution[key] for key in range(1, 8)},
        "entries": all_entries,
    }
    bank_path = Path(training_cfg["query_bank"])
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_text(json.dumps(bank) + "\n", encoding="utf-8")
    print(f"Wrote query bank: {bank_path} ({len(all_entries)} entries)")
    print("Class distribution:", bank["class_distribution"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan-root", default="/space0/mawb/tokengs/data/ScanNet/scans"
    )
    parser.add_argument(
        "--label-root", default="/space0/mawb/tokengs/data/scannet2d_labels"
    )
    parser.add_argument(
        "--protocol",
        default="/space0/mawb/tokengs/configs/semantic/scannet_c3g8.yaml",
    )
    parser.add_argument("--max-frames-per-scene", type=int)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
