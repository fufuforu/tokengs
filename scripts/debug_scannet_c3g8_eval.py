#!/usr/bin/env python3
"""Validate C3G8 ScanNet eval data, inspect samples, and save alignment images."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tokengs.data import get_multi_dataloader
from tokengs.options import config_defaults


C3G8_COLORS = np.asarray(
    [
        [0, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [67, 99, 216],
        [245, 130, 49],
        [145, 30, 180],
        [66, 212, 244],
        [128, 128, 0],
    ],
    dtype=np.uint8,
)


class _LocalAccelerator:
    is_main_process = True


def _rgb_uint8(image: torch.Tensor) -> np.ndarray:
    return (
        image.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
    )


def _save_visualizations(batch: dict, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = _rgb_uint8(batch["images_output"][0, 0])
    labels = batch["semantic_label_output"][0, 0].cpu().numpy()
    label_rgb = C3G8_COLORS[labels]
    overlay = rgb.copy()
    valid = labels != 0
    overlay[valid] = (
        0.55 * rgb[valid].astype(np.float32)
        + 0.45 * label_rgb[valid].astype(np.float32)
    ).astype(np.uint8)
    paths = {
        "rgb": output_dir / "target_rgb.png",
        "label": output_dir / "target_label.png",
        "overlay": output_dir / "target_overlay.png",
    }
    Image.fromarray(rgb).save(paths["rgb"])
    Image.fromarray(label_rgb).save(paths["label"])
    Image.fromarray(overlay).save(paths["overlay"])
    return {name: str(path.resolve()) for name, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()

    opt = config_defaults["eval_scannet_c3g8"]
    _, test_loader, _, test_dataset = get_multi_dataloader(opt, _LocalAccelerator())
    provider = test_dataset.datasets[0]
    dataset = provider.dataset
    print("Manifest report:", dataset.get_manifest_report(validate_frames=True))

    batch = next(iter(test_loader))
    print("Batch scene:", batch["scene_name"])
    print("Batch frame IDs [input0, input1, target]:", batch["frame_ids"].tolist())
    print("Batch fields:")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"  {key}: {value!r}")

    sample_indices = np.linspace(
        0, len(provider) - 1, num=min(args.num_samples, len(provider)), dtype=int
    )
    pixel_counts = Counter()
    sampled_scenes = []
    for sample_index in sample_indices:
        item = provider.get_item(int(sample_index))
        sampled_scenes.append(item["scene_name"])
        values, counts = torch.unique(
            item["semantic_label_output"], return_counts=True
        )
        pixel_counts.update(
            {int(value): int(count) for value, count in zip(values, counts)}
        )
    class_names = ("background", *dataset.semantic_class_names)
    print("Sampled scenes:", sampled_scenes)
    print(
        "Target pixel counts:",
        {class_names[class_id]: pixel_counts[class_id] for class_id in range(9)},
    )
    print(
        "Visualizations:",
        _save_visualizations(batch, Path(opt.workspace) / "visualizations"),
    )


if __name__ == "__main__":
    main()
