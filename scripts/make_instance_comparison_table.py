"""Build the InstOk3D-style class-agnostic instance segmentation table.

Reads ``instance_ap.json`` outputs written by
``eval_instance_lsm_protocol.py`` for our checkpoints and merges them with
the published InstOk3D Table 2 numbers (LSM 40-scene protocol, 8 context
views, evaluated on 7 test views per scene). Columns follow the paper:
AP (COCO-style mean over IoU 0.5:0.05:0.95), AP50, AP25, PSNR, SSIM, LPIPS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# Method -> [AP, AP50, AP25, PSNR, SSIM, LPIPS] from InstOk3D Table 2
# (40 LSM ScanNet scenes, 8 context views; per-scene optimized baselines
# re-run by InstOk3D under the shared protocol).
PUBLISHED = {
    "Gaussian Grouping (InstOk3D T2)": [0.139, 0.288, 0.440, 23.20, 0.715, 0.325],
    "ObjectGS (InstOk3D T2)": [0.178, 0.337, 0.489, 24.34, 0.733, 0.310],
    "IGGT + LUDVIG (InstOk3D T2)": [0.122, 0.265, 0.442, 22.75, 0.712, 0.323],
    "InstOk3D (Table 2)": [0.235, 0.438, 0.564, 22.41, 0.709, 0.355],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        nargs="+",
        metavar="LABEL=PATH",
        help="Our eval results as label=/path/to/instance_ap.json",
    )
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="Report pooled AP columns instead of per-scene mean.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output path for the markdown table.",
    )
    args = parser.parse_args()

    rows: list[tuple[str, list[float | str]]] = []
    for item in args.results or []:
        label, _, path = item.partition("=")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        prefix = "pooled" if args.pooled else "mean"
        row = [
            label,
            payload[f"{prefix}_ap"],
            payload[f"{prefix}_ap50"],
            payload[f"{prefix}_ap25"],
            payload["mean_psnr"],
            payload["mean_ssim"],
            payload["mean_lpips"],
        ]
        rows.append((label, row))

    rows.sort(key=lambda item: -float(item[1][1]))
    lines = [
        "| Method | AP | AP50 | AP25 | PSNR | SSIM | LPIPS |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, row in rows:
        cells = " | ".join(
            f"{value:.3f}" if isinstance(value, float) else str(value)
            for value in row[1:]
        )
        lines.append(f"| {label} | {cells} |")
    lines.append("")
    lines.append(
        "Published baselines are from InstOk3D Table 2 (same 40-scene LSM "
        "protocol, 8 context views, 7 test views; per-scene mean AP)."
    )
    table = "\n".join(lines)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
