"""Build a wide-baseline prompt training manifest aligned with the LSM eval.

For every training scene, 15 frames are selected (stride-10 first 15, with an
evenly-spaced fallback for short scenes) and split interleaved into 8
context views + 7 test views, exactly matching the LSM ScanNet instance
evaluation protocol. Each existing prompt sample keeps its class / prompt
mode / query but is re-anchored to the scene's wide-baseline frame structure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _scene_frames(scene_dir: Path, num_views: int = 15) -> list[int]:
    from tokengs.data.static.scannet import ScanNetSensReader

    reader = ScanNetSensReader(scene_dir / f"{scene_dir.name}.sens", frame_stride=1)
    valid = list(reader.frame_ids)
    stride10 = [value for value in valid if value % 10 == 0]
    if len(stride10) >= num_views:
        return stride10[:num_views]
    indices = np.linspace(0, len(valid) - 1, num_views).round().astype(int)
    return [valid[int(i)] for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=(
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "scannet_prompt_full_ratio02_16f.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "scannet_prompt_full_wide_8x7.json"
        ),
    )
    parser.add_argument("--num_context", type=int, default=8)
    parser.add_argument("--num_test", type=int, default=7)
    args = parser.parse_args()

    source = json.loads(Path(args.source).read_text(encoding="utf-8"))
    num_views = args.num_context + args.num_test
    train_scenes = sorted(
        {str(sample["scene"]) for sample in source["train_samples"]}
    )
    print(f"train scenes: {len(train_scenes)}")

    scan_root = Path(source["scan_root"]) if "scan_root" in source else None
    if scan_root is None or not scan_root.is_dir():
        scan_root = Path("/datasets/ScanNet/scans")

    def _rebuild(samples, cache):
        wide = []
        for sample in samples:
            scene = str(sample["scene"])
            if scene not in cache:
                cache[scene] = _scene_frames(scan_root / scene, num_views)
                if len(cache[scene]) != num_views:
                    raise RuntimeError(
                        f"{scene}: need {num_views} frames, "
                        f"got {len(cache[scene])}"
                    )
                print(f"scanned {len(cache)} scenes")
            frames = cache[scene]
            context = [frames[j] for j in range(0, num_views, 2)]
            test = [frames[j] for j in range(1, num_views, 2)]
            assert len(context) == args.num_context
            assert len(test) == args.num_test
            wide.append(
                {
                    **sample,
                    "sample_id": f"{sample['sample_id']}_wide",
                    "input_frame_ids": context,
                    "target_frame_ids": test,
                }
            )
        return wide

    frame_cache: dict[str, list[int]] = {}
    wide_samples = _rebuild(source["train_samples"], frame_cache)
    wide_validation = _rebuild(source.get("validation_samples", []), frame_cache)

    output = dict(source)
    output["train_samples"] = wide_samples
    output["validation_samples"] = wide_validation
    output["version"] = int(source.get("version", 1)) + 1
    output["source_manifest"] = str(Path(args.source).resolve())
    output["wide_baseline"] = {
        "num_context": args.num_context,
        "num_test": args.num_test,
        "frame_stride": 10,
    }
    # Wide-baseline targets span frames 0..140, so the near-baseline
    # same-scene query gap (90) no longer applies. Queries only pool CLIP
    # features (no pixel copy), so only exact frame overlap is forbidden.
    output["query_same_scene_min_frame_gap"] = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Wrote wide-baseline manifest: {output_path} "
        f"({len(wide_samples)} train samples, {len(frame_cache)} scenes)"
    )
    example = wide_samples[0]
    print("example context:", example["input_frame_ids"])
    print("example test:   ", example["target_frame_ids"])


if __name__ == "__main__":
    main()
