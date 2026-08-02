#!/usr/bin/env python3
"""Load the ScanNet debug batch and save RGB/label alignment images."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tokengs.data import get_multi_dataloader
from tokengs.options import config_defaults


class _LocalAccelerator:
    is_main_process = True


def _colorize(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int64)
    colors = np.stack(
        [
            (labels * 37 + 17) % 255,
            (labels * 67 + 29) % 255,
            (labels * 97 + 43) % 255,
        ],
        axis=-1,
    ).astype(np.uint8)
    colors[labels == 0] = 0
    return colors


def main() -> None:
    opt = config_defaults["debug_scannet_dataset"]
    _, test_loader, _, test_dataset = get_multi_dataloader(
        opt, _LocalAccelerator()
    )
    batch = next(iter(test_loader))

    print("Batch fields:")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"  {key}: {value!r}")

    raw_dataset = test_dataset.datasets[0].dataset
    print("Scene names:", batch["scene_name"])
    print("Raw frame IDs:", batch["frame_ids"].tolist())
    print("Pose filtering:", raw_dataset.get_reader_stats(0))

    rgb = (
        batch["images_all"][0, 0]
        .permute(1, 2, 0)
        .clamp(0, 1)
        .mul(255)
        .byte()
        .cpu()
        .numpy()
    )
    labels = batch["semantic_label_all"][0, 0].cpu().numpy()
    label_rgb = _colorize(labels)
    overlay = rgb.copy()
    foreground = labels != 0
    overlay[foreground] = (
        0.55 * rgb[foreground] + 0.45 * label_rgb[foreground]
    ).astype(np.uint8)

    output_dir = Path(opt.workspace)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rgb": output_dir / "rgb.png",
        "label": output_dir / "label.png",
        "overlay": output_dir / "overlay.png",
    }
    Image.fromarray(rgb).save(paths["rgb"])
    Image.fromarray(label_rgb).save(paths["label"])
    Image.fromarray(overlay).save(paths["overlay"])
    print("Label unique IDs:", np.unique(labels).tolist())
    print("Visualizations:", {key: str(path) for key, path in paths.items()})


if __name__ == "__main__":
    main()
