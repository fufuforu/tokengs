#!/usr/bin/env python3
"""Build a reproducible balanced 64/8-scene prompt-training manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from tokengs.data.static.scannet import ScanNet, ScanNetSensReader


CLASS_IDS = tuple(range(1, 9))
IMAGE_CLASS_IDS = tuple(range(1, 8))


def _numeric_frame_ids(path: Path) -> list[int]:
    """Return all numeric semantic-label frame ids in deterministic order."""
    return sorted(
        int(item.stem)
        for item in path.glob("*.png")
        if item.stem.isdigit()
    )


def _scene_classes(entries: list[dict]) -> dict[str, set[int]]:
    output: dict[str, set[int]] = defaultdict(set)
    for entry in entries:
        output[str(entry["scene"])].add(int(entry["class_id"]))
    return output


def _select_diverse_scenes(
    candidates: list[str],
    scene_classes: dict[str, set[int]],
    count: int,
    rng: np.random.Generator,
) -> list[str]:
    order = list(candidates)
    rng.shuffle(order)
    rank = {scene: index for index, scene in enumerate(order)}
    class_frequency = Counter(
        class_id for scene in candidates for class_id in scene_classes[scene]
    )
    selected: list[str] = []
    selected_counts = Counter()
    remaining = set(candidates)
    while len(selected) < count:
        best = max(
            remaining,
            key=lambda scene: (
                sum(
                    1.0 / ((selected_counts[class_id] + 1) * class_frequency[class_id])
                    for class_id in scene_classes[scene]
                ),
                -rank[scene],
            ),
        )
        selected.append(best)
        remaining.remove(best)
        selected_counts.update(scene_classes[best])
    return sorted(selected)


def _context_frames(valid_ids: list[int], target: int) -> tuple[int, int] | None:
    before = [frame_id for frame_id in valid_ids if frame_id < target]
    after = [frame_id for frame_id in valid_ids if frame_id > target]
    if before and after:
        return before[-1], after[0]
    nearest = sorted(
        (frame_id for frame_id in valid_ids if frame_id != target),
        key=lambda frame_id: (abs(frame_id - target), frame_id),
    )
    if len(nearest) < 2:
        return None
    return tuple(sorted(nearest[:2]))


def _map_label(path: Path, lut: np.ndarray, fallback: int) -> np.ndarray:
    raw = np.asarray(Image.open(path))
    if raw.ndim == 3:
        raw = raw[..., 0]
    mapped = np.full(raw.shape, fallback, dtype=np.int64)
    valid = (raw >= 0) & (raw < len(lut))
    mapped[valid] = lut[raw[valid]]
    return mapped


def _center_square(label: np.ndarray) -> np.ndarray:
    height, width = label.shape
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return label[top : top + side, left : left + side]


def _build_target_candidates(
    scenes: list[str],
    entries_by_scene: dict[str, list[dict]],
    scan_root: Path,
    label_root: Path,
    lut: np.ndarray,
    fallback: int,
    min_pixels: int,
    target_frames_per_scene: int = 0,
    min_foreground_ratio: float = 0.0,
) -> tuple[dict[str, dict[int, list[dict]]], dict[str, dict]]:
    candidates: dict[str, dict[int, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    reader_reports = {}
    for scene_index, scene in enumerate(scenes, start=1):
        reader = ScanNetSensReader(scan_root / scene / f"{scene}.sens", frame_stride=1)
        valid_ids = list(reader.frame_ids)
        valid_set = set(valid_ids)
        reader_reports[scene] = reader.stats()
        if target_frames_per_scene > 0:
            # Target supervision must be independent from the query bank. The
            # bank is intentionally sparse and must not silently cap target
            # frame diversity.
            label_frames = [
                frame_id
                for frame_id in _numeric_frame_ids(label_root / scene / "label-filt")
                if frame_id in valid_set
            ]
            if len(label_frames) > target_frames_per_scene:
                indices = np.linspace(
                    0,
                    len(label_frames) - 1,
                    num=target_frames_per_scene,
                    dtype=int,
                )
                target_frames = [label_frames[index] for index in sorted(set(indices.tolist()))]
            else:
                target_frames = label_frames
            frame_class_pairs = [
                (frame_id, class_id)
                for frame_id in target_frames
                for class_id in CLASS_IDS
            ]
        else:
            # Backward-compatible path for the original small manifests.
            frame_class_pairs = sorted(
                {
                    (int(entry["raw_frame_id"]), int(entry["class_id"]))
                    for entry in entries_by_scene[scene]
                    if int(entry["raw_frame_id"]) in valid_set
                }
            )
            target_frames = sorted({frame_id for frame_id, _ in frame_class_pairs})
        label_cache = {}

        def mapped_label(frame_id: int) -> np.ndarray | None:
            if frame_id not in label_cache:
                path = label_root / scene / "label-filt" / f"{frame_id}.png"
                label_cache[frame_id] = (
                    _map_label(path, lut, fallback) if path.is_file() else None
                )
            return label_cache[frame_id]

        for frame_id, class_id in frame_class_pairs:
            context = _context_frames(valid_ids, frame_id)
            label = mapped_label(frame_id)
            if context is None or label is None:
                continue
            transformed_label = _center_square(label)
            area = int((transformed_label == class_id).sum())
            foreground_ratio = area / float(transformed_label.size)
            if area < min_pixels or foreground_ratio < min_foreground_ratio:
                continue
            candidates[scene][class_id].append(
                {
                    "scene": scene,
                    "input_frame_ids": list(context),
                    "target_frame_id": frame_id,
                    "class_id": class_id,
                    "foreground_pixels": area,
                    "image_pixels": int(transformed_label.size),
                    "foreground_ratio": foreground_ratio,
                }
            )

        other_search_frames = target_frames
        if not other_search_frames and target_frames_per_scene <= 0:
            indices = np.linspace(0, len(valid_ids) - 1, num=min(16, len(valid_ids)), dtype=int)
            other_search_frames = [valid_ids[index] for index in sorted(set(indices.tolist()))]
        for frame_id in other_search_frames:
            context = _context_frames(valid_ids, frame_id)
            label = mapped_label(frame_id)
            if context is None or label is None:
                continue
            if target_frames_per_scene <= 0:
                transformed_label = _center_square(label)
                area = int((transformed_label == 8).sum())
                foreground_ratio = area / float(transformed_label.size)
                if area < min_pixels or foreground_ratio < min_foreground_ratio:
                    continue
                candidates[scene][8].append(
                    {
                        "scene": scene,
                        "input_frame_ids": list(context),
                        "target_frame_id": frame_id,
                        "class_id": 8,
                        "foreground_pixels": area,
                        "image_pixels": int(transformed_label.size),
                        "foreground_ratio": foreground_ratio,
                    }
                )
        if scene_index % 16 == 0 or scene_index == len(scenes):
            print(f"Indexed target candidates for {scene_index}/{len(scenes)} scenes")
    return candidates, reader_reports


def _assign_balanced_samples(
    scenes: list[str],
    target_candidates: dict[str, dict[int, list[dict]]],
    samples_per_class: int,
    split: str,
    rng: np.random.Generator,
) -> list[dict]:
    quotas = {class_id: samples_per_class for class_id in CLASS_IDS}
    assigned: dict[int, list[dict]] = defaultdict(list)
    class_scene_frequency = Counter(
        class_id
        for scene in scenes
        for class_id, items in target_candidates[scene].items()
        if items
    )
    scene_cursor = Counter()

    for scene in scenes:
        available = [
            class_id
            for class_id in CLASS_IDS
            if target_candidates[scene].get(class_id) and quotas[class_id] > 0
        ]
        if not available:
            raise RuntimeError(f"Selected {split} scene has no target candidates: {scene}")
        class_id = min(
            available,
            key=lambda value: (class_scene_frequency[value], -quotas[value], value),
        )
        items = target_candidates[scene][class_id]
        assigned[class_id].append(items[scene_cursor[(scene, class_id)] % len(items)].copy())
        scene_cursor[(scene, class_id)] += 1
        quotas[class_id] -= 1

    for class_id in CLASS_IDS:
        class_scenes = [
            scene for scene in scenes if target_candidates[scene].get(class_id)
        ]
        if not class_scenes:
            raise RuntimeError(f"No {split} target candidates for class {class_id}")
        unique_available = sum(
            len(target_candidates[scene][class_id]) for scene in class_scenes
        )
        if unique_available < samples_per_class:
            raise RuntimeError(
                f"{split} class {class_id} has only {unique_available} unique "
                f"targets for quota {samples_per_class}"
            )
        cursor = 0
        while quotas[class_id] > 0:
            available_scenes = [
                scene
                for scene in class_scenes
                if scene_cursor[(scene, class_id)]
                < len(target_candidates[scene][class_id])
            ]
            if not available_scenes:
                raise RuntimeError(
                    f"{split} class {class_id} exhausted unique target frames"
                )
            scene = available_scenes[cursor % len(available_scenes)]
            items = target_candidates[scene][class_id]
            item_index = scene_cursor[(scene, class_id)]
            assigned[class_id].append(items[item_index].copy())
            scene_cursor[(scene, class_id)] += 1
            quotas[class_id] -= 1
            cursor += 1

    for class_id in CLASS_IDS:
        keys = [
            (item["scene"], int(item["target_frame_id"]))
            for item in assigned[class_id]
        ]
        if len(keys) != len(set(keys)):
            raise RuntimeError(
                f"{split} class {class_id} target quota exceeded unique frames"
            )

    samples = []
    total_samples = samples_per_class * len(CLASS_IDS)
    desired_text = round(total_samples * 0.4)
    desired_image = round(total_samples * 0.3)
    eligible_text = desired_text - samples_per_class
    text_per_class = [eligible_text // len(IMAGE_CLASS_IDS)] * len(IMAGE_CLASS_IDS)
    for index in range(eligible_text % len(IMAGE_CLASS_IDS)):
        text_per_class[index] += 1
    image_per_class = [desired_image // len(IMAGE_CLASS_IDS)] * len(IMAGE_CLASS_IDS)
    for index in range(desired_image % len(IMAGE_CLASS_IDS)):
        image_per_class[index] += 1
    for class_id in CLASS_IDS:
        class_samples = assigned[class_id]
        if class_id == 8:
            modes = ["text_only"] * samples_per_class
        elif split == "validation" and samples_per_class == 3:
            modes = ["text_only", "image_only", "text_image_mixed"]
        else:
            text_count = text_per_class[class_id - 1]
            image_count = image_per_class[class_id - 1]
            mixed_count = samples_per_class - text_count - image_count
            modes = (
                ["text_only"] * text_count
                + ["image_only"] * image_count
                + ["text_image_mixed"] * mixed_count
            )
        rng.shuffle(class_samples)
        rng.shuffle(modes)
        for item, mode in zip(class_samples, modes, strict=True):
            item["prompt_mode"] = mode
            samples.append(item)
    rng.shuffle(samples)
    return samples


def _attach_queries(
    samples: list[dict],
    bank_entries: list[dict],
    allowed_query_scenes: set[str],
) -> None:
    entries_by_class_scene: dict[int, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in bank_entries:
        if entry["scene"] in allowed_query_scenes:
            entries_by_class_scene[int(entry["class_id"])][entry["scene"]].append(
                entry
            )

    # Interleave scenes before walking entries so image prompts see distinct
    # cross-scene appearances instead of repeatedly taking the first bank item.
    entries_by_class: dict[int, list[dict]] = {}
    for class_id, entries_by_scene in entries_by_class_scene.items():
        ordered_scenes = sorted(entries_by_scene)
        for entries in entries_by_scene.values():
            entries.sort(
                key=lambda entry: (
                    int(entry["raw_frame_id"]),
                    -1
                    if entry.get("instance_id") is None
                    else int(entry["instance_id"]),
                )
            )
        max_scene_entries = max(len(entries) for entries in entries_by_scene.values())
        entries_by_class[class_id] = [
            entries_by_scene[scene][entry_index]
            for entry_index in range(max_scene_entries)
            for scene in ordered_scenes
            if entry_index < len(entries_by_scene[scene])
        ]

    cursors = Counter()
    for sample in samples:
        mode = sample["prompt_mode"]
        class_id = int(sample["class_id"])
        if mode == "text_only":
            sample["query"] = None
            continue
        class_entries = entries_by_class[class_id]
        selected_index = next(
            (
                (cursors[class_id] + offset) % len(class_entries)
                for offset in range(len(class_entries))
                if class_entries[(cursors[class_id] + offset) % len(class_entries)][
                    "scene"
                ]
                != sample["scene"]
            ),
            None,
        )
        if selected_index is None:
            raise RuntimeError(
                f"No cross-scene query for class {class_id}, target {sample['scene']}"
            )
        entry = class_entries[selected_index]
        cursors[class_id] = selected_index + 1
        sample["query"] = {
            key: entry.get(key)
            for key in (
                "scene",
                "raw_frame_id",
                "class_id",
                "bbox",
                "mask_area",
                "image_size",
                "instance_id",
                "source_type",
            )
        }


def _distribution(samples: list[dict]) -> dict:
    return {
        "classes": dict(sorted(Counter(str(item["class_id"]) for item in samples).items())),
        "prompt_modes": dict(sorted(Counter(item["prompt_mode"] for item in samples).items())),
        "scenes_represented": len({item["scene"] for item in samples}),
    }


def build(args: argparse.Namespace) -> None:
    provisional = json.loads(Path(args.train_manifest).read_text(encoding="utf-8"))
    bank = json.loads(Path(args.query_bank).read_text(encoding="utf-8"))
    protocol_path = Path(args.protocol)
    lut, fallback, class_names = ScanNet._load_c3g8_protocol(protocol_path)
    excluded = set(provisional["excluded_eval_scenes"])
    provisional_scenes = set(provisional["scenes"])
    bank_entries = [
        entry
        for entry in bank["entries"]
        if entry["scene"] in provisional_scenes and entry["scene"] not in excluded
    ]
    scene_classes = _scene_classes(bank_entries)
    eligible_scenes = sorted(scene_classes)
    rng = np.random.default_rng(args.seed)

    if args.scene_split_from is not None:
        scene_split = json.loads(
            Path(args.scene_split_from).read_text(encoding="utf-8")
        )
        train_scenes = [str(scene) for scene in scene_split["train_scenes"]]
        validation_scenes = [
            str(scene) for scene in scene_split["validation_scenes"]
        ]
        if len(train_scenes) != args.train_scenes:
            raise ValueError("scene_split_from has an unexpected train-scene count")
        if len(validation_scenes) != args.validation_scenes:
            raise ValueError("scene_split_from has an unexpected validation-scene count")
        if (set(train_scenes) | set(validation_scenes)) - set(eligible_scenes):
            raise ValueError("scene_split_from contains unavailable scenes")
    else:
        validation_scenes = _select_diverse_scenes(
            eligible_scenes, scene_classes, args.validation_scenes, rng
        )
        train_candidates = sorted(set(eligible_scenes) - set(validation_scenes))
        train_scenes = _select_diverse_scenes(
            train_candidates, scene_classes, args.train_scenes, rng
        )
    selected = train_scenes + validation_scenes
    entries_by_scene: dict[str, list[dict]] = defaultdict(list)
    for entry in bank_entries:
        if entry["scene"] in selected:
            entries_by_scene[entry["scene"]].append(entry)

    target_candidates, reader_reports = _build_target_candidates(
        selected,
        entries_by_scene,
        Path(args.scan_root),
        Path(args.label_root),
        lut,
        fallback,
        args.min_target_pixels,
        target_frames_per_scene=args.target_frames_per_scene,
        min_foreground_ratio=args.min_target_foreground_ratio,
    )
    train_samples = _assign_balanced_samples(
        train_scenes,
        target_candidates,
        args.train_samples_per_class,
        "train",
        rng,
    )
    validation_samples = _assign_balanced_samples(
        validation_scenes,
        target_candidates,
        args.validation_samples_per_class,
        "validation",
        rng,
    )
    _attach_queries(train_samples, bank_entries, set(train_scenes))
    _attach_queries(validation_samples, bank_entries, set(train_scenes))
    if args.validation_samples_from is not None:
        validation_source = json.loads(
            Path(args.validation_samples_from).read_text(encoding="utf-8")
        )
        validation_samples = json.loads(
            json.dumps(validation_source["validation_samples"])
        )
        if len(validation_samples) != args.validation_samples_per_class * len(CLASS_IDS):
            raise ValueError(
                "validation_samples_from has an unexpected sample count"
            )
        if {str(item["scene"]) for item in validation_samples} - set(
            validation_scenes
        ):
            raise ValueError(
                "validation_samples_from contains scenes outside the validation split"
            )
    for split, samples in (("train", train_samples), ("validation", validation_samples)):
        for index, sample in enumerate(samples):
            sample["sample_id"] = f"{split}_{index:04d}"

    if args.preserve_validation_queries_from is not None:
        baseline_path = Path(args.preserve_validation_queries_from)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_samples = baseline["validation_samples"]
        if len(baseline_samples) != len(validation_samples):
            raise ValueError("Validation sample count changed while preserving queries")
        for sample, baseline_sample in zip(
            validation_samples, baseline_samples, strict=True
        ):
            sample_without_query = {
                key: value for key, value in sample.items() if key != "query"
            }
            baseline_without_query = {
                key: value for key, value in baseline_sample.items() if key != "query"
            }
            if sample_without_query != baseline_without_query:
                raise ValueError(
                    "Validation target changed while preserving queries: "
                    f"{sample['sample_id']}"
                )
            sample["query"] = baseline_sample["query"]

    output = {
        "version": 1,
        "seed": args.seed,
        "provisional": True,
        "source_manifest": str(Path(args.train_manifest).resolve()),
        "query_bank": str(Path(args.query_bank).resolve()),
        "excluded_eval_scenes": sorted(excluded),
        "class_names": list(class_names),
        "prompt_mode_ratio_for_image_classes": {
            "text_only": 0.4,
            "image_only": 0.3,
            "text_image_mixed": 0.3,
        },
        "train_scenes": train_scenes,
        "validation_scenes": validation_scenes,
        "target_frames_per_scene": args.target_frames_per_scene,
        "min_target_foreground_ratio": args.min_target_foreground_ratio,
        "scene_split_from": (
            str(Path(args.scene_split_from).resolve())
            if args.scene_split_from is not None
            else None
        ),
        "validation_samples_from": (
            str(Path(args.validation_samples_from).resolve())
            if args.validation_samples_from is not None
            else None
        ),
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "distribution": {
            "train": _distribution(train_samples),
            "validation": _distribution(validation_samples),
        },
        "reader_reports": reader_reports,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(json.dumps(output["distribution"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-manifest",
        default="/space0/mawb/tokengs/data/scannet_prompt/scannet_c3g8_train_provisional.json",
    )
    parser.add_argument(
        "--query-bank",
        default="/space0/mawb/tokengs/data/scannet_prompt/scannet_c3g8_query_bank.json",
    )
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
    parser.add_argument(
        "--output",
        default="/space0/mawb/tokengs/data/scannet_prompt/scannet_prompt_small_64_8.json",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--train-scenes", type=int, default=64)
    parser.add_argument("--validation-scenes", type=int, default=8)
    parser.add_argument("--train-samples-per-class", type=int, default=40)
    parser.add_argument("--validation-samples-per-class", type=int, default=3)
    parser.add_argument("--min-target-pixels", type=int, default=64)
    parser.add_argument(
        "--min-target-foreground-ratio",
        type=float,
        default=0.0,
        help="Minimum target area fraction after the deterministic center crop.",
    )
    parser.add_argument(
        "--target-frames-per-scene",
        type=int,
        default=0,
        help="Independent label-frame budget per scene; 0 preserves the old query-bank behavior.",
    )
    parser.add_argument(
        "--scene-split-from",
        help="Reuse train/validation scenes from an existing manifest.",
    )
    parser.add_argument("--preserve-validation-queries-from")
    parser.add_argument(
        "--validation-samples-from",
        help="Reuse the complete fixed validation rows from an existing manifest.",
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
