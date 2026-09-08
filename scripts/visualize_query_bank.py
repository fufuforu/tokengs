#!/usr/bin/env python3
"""Randomly visualize small-object query crops from the query bank.

Saves one contact sheet per C3G8 class (chair/table/sofa/bed by default) with
the original crop, the binary mask, and the mask overlay, so low-quality or
border-truncated queries can be spotted before training.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from tokengs.data.static.scannet import ScanNetSensReader


C3G8_CLASS_NAMES = ("wall", "floor", "ceiling", "chair", "table", "sofa", "bed", "other")
RAW_IDS = {
    4: [2, 10, 23, 74, 885, 1184, 1291, 1338],
    5: [4, 24, 44, 45, 108, 222, 1193, 1355],
    6: [6, 1313],
    7: [11, 494, 786, 1191, 1349],
}


def _load_crop_mask(
    scan_root: Path, label_root: Path, entry: dict
) -> tuple[np.ndarray, np.ndarray] | None:
    scene = entry["scene"]
    frame_id = entry["raw_frame_id"]
    x0, y0, x1, y1 = entry["bbox"]
    reader = ScanNetSensReader(
        scan_root / scene / f"{scene}.sens", frame_stride=1
    )
    rgb = np.asarray(reader.read_color(frame_id), dtype=np.uint8).copy()
    semantic_raw = np.asarray(
        Image.open(label_root / scene / "label-filt" / f"{frame_id}.png")
    )
    class_id = int(entry["class_id"])
    raw_ids = RAW_IDS[class_id]
    if entry.get("instance_id") is None:
        mask = np.isin(semantic_raw, raw_ids)
    else:
        instances = np.asarray(
            Image.open(label_root / scene / "instance-filt" / f"{frame_id}.png")
        )
        mask = (instances == int(entry["instance_id"])) & np.isin(
            semantic_raw, raw_ids
        )
    if not mask[y0:y1, x0:x1].any():
        return None
    return rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank",
        default="/space0/mawb/tokengs/data/scannet_prompt/scannet_c3g8_query_bank.json",
    )
    parser.add_argument(
        "--scan-root", default="/space0/mawb/tokengs/data/ScanNet/scans"
    )
    parser.add_argument(
        "--label-root", default="/space0/mawb/tokengs/data/scannet2d_labels"
    )
    parser.add_argument(
        "--classes", nargs="+", type=int, default=[4, 5, 6, 7]
    )
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", default="workspace/query_bank_inspection"
    )
    args = parser.parse_args()

    bank = json.load(open(args.bank, encoding="utf-8"))
    entries = [
        entry
        for entry in bank["entries"]
        if int(entry["class_id"]) in args.classes
    ]
    print("Bank filters:", json.dumps(bank.get("filters", {}), indent=1))
    by_class: dict[int, list[dict]] = defaultdict(list)
    for entry in entries:
        by_class[int(entry["class_id"])].append(entry)
    for class_id in args.classes:
        class_entries = by_class[class_id]
        widths = [e["bbox"][2] - e["bbox"][0] for e in class_entries]
        heights = [e["bbox"][3] - e["bbox"][1] for e in class_entries]
        borders = sum(1 for e in class_entries if e["border_edges"] > 0)
        print(
            f"class {class_id} ({C3G8_CLASS_NAMES[class_id - 1]}): "
            f"n={len(class_entries)} width med={int(statistics.median(widths)) if widths else 0} "
            f"border_edges>0={borders}"
        )

    rng = np.random.default_rng(args.seed)
    scan_root = Path(args.scan_root)
    label_root = Path(args.label_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_id in args.classes:
        class_entries = by_class[class_id]
        if not class_entries:
            continue
        selected = list(
            rng.choice(len(class_entries), size=min(args.per_class, len(class_entries)), replace=False)
        )
        rows = []
        for index in selected:
            loaded = _load_crop_mask(scan_root, label_root, class_entries[index])
            if loaded is None:
                continue
            crop, mask = loaded
            overlay = crop.copy()
            overlay_rgb = overlay[..., 0]
            overlay_rgb[mask] = np.clip(overlay_rgb[mask].astype(int) + 120, 0, 255)
            rows.append((crop, (mask * 255).astype(np.uint8), overlay))
        if not rows:
            continue
        n = len(rows)
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1:
            axes = axes[None, :]
        for row_index, (crop, mask, overlay) in enumerate(rows):
            for col_index, panel in enumerate((crop, mask, overlay)):
                axes[row_index, col_index].imshow(
                    panel, cmap="gray" if col_index == 1 else None
                )
                axes[row_index, col_index].axis("off")
            axes[row_index, 0].set_ylabel(f"#{row_index}", fontsize=8)
        axes[0, 0].set_title("crop")
        axes[0, 1].set_title("mask")
        axes[0, 2].set_title("overlay")
        fig.suptitle(
            f"class {class_id} ({C3G8_CLASS_NAMES[class_id - 1]}) query samples",
            fontsize=12,
        )
        fig.tight_layout()
        path = output_dir / f"class{class_id}_{C3G8_CLASS_NAMES[class_id - 1]}_queries.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
