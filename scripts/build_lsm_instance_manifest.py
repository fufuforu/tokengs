"""Build the LSM ScanNet instance-evaluation manifest.

Follows the protocol shared by LSM / Gaussian Grouping / ObjectGS / IGGT /
InstOk3D: for each of the 40 C3G test scenes, sample frames from the full
sequence with stride 10 (the stride-10 color frames already shipped in
``/space0/mawb/C3G/datasets/scannet_test``), take the first 15 sampled
frames, and split them interleaved into 8 context views and 7 test views.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _frame_id_from_name(name: str) -> int:
    return int(re.sub(r"\D", "", Path(name).stem))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_root",
        default="/space0/mawb/C3G/datasets/scannet_test",
    )
    parser.add_argument(
        "--manifest",
        default=(
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "lsm_instance_eval_manifest.json"
        ),
    )
    args = parser.parse_args()

    test_root = Path(args.test_root)
    seq_file = test_root / "selected_seqs_test.json"
    scenes = json.loads(seq_file.read_text(encoding="utf-8"))
    scene_names = list(scenes) if isinstance(scenes, dict) else scenes

    output = {"version": 1, "scenes": {}}
    for scene in scene_names:
        image_dir = test_root / scene / "images"
        raw_ids = sorted(
            {
                _frame_id_from_name(path.name)
                for path in image_dir.glob("*.jpg")
            }
        )
        sampled = [value for value in raw_ids if value % 10 == 0][:15]
        if len(sampled) < 15:
            raise RuntimeError(
                f"{scene}: need 15 stride-10 frames, got {len(sampled)}"
            )
        context = [sampled[i] for i in range(0, 15, 2)]
        test = [sampled[i] for i in range(1, 15, 2)]
        output["scenes"][scene] = {
            "context_raw_frame_ids": context,
            "test_raw_frame_ids": test,
        }

    output_path = Path(args.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote LSM instance eval manifest: {output_path}")
    print(f"scenes={len(output['scenes'])} context_views=8 test_views=7")
    example = next(iter(output["scenes"].values()))
    print("example:", example)


if __name__ == "__main__":
    main()
