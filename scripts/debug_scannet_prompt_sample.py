#!/usr/bin/env python3
"""Inspect prompt-training batches and save target/query alignment artifacts."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tokengs.data import get_multi_dataloader
from tokengs.options import config_defaults


class _LocalAccelerator:
    is_main_process = True


def _rgb_uint8(image: torch.Tensor) -> np.ndarray:
    return image.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()


def _mask_uint8(mask: torch.Tensor) -> np.ndarray:
    return mask.squeeze().gt(0.5).byte().mul(255).cpu().numpy()


def _save_debug(batch: dict, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_rgb = _rgb_uint8(batch["images_output"][0, 0])
    target_mask = _mask_uint8(batch["binary_mask_output"][0, 0])
    query_rgb = _rgb_uint8(batch["query_image"][0])
    query_mask = _mask_uint8(batch["query_mask"][0])
    target_overlay = target_rgb.copy()
    target_overlay[target_mask > 0] = (
        0.55 * target_rgb[target_mask > 0] + np.asarray([230, 25, 75]) * 0.45
    ).astype(np.uint8)
    query_overlay = query_rgb.copy()
    query_overlay[query_mask > 0] = (
        0.55 * query_rgb[query_mask > 0] + np.asarray([60, 180, 75]) * 0.45
    ).astype(np.uint8)
    artifacts = {
        "target_rgb": output_dir / "target_rgb.png",
        "target_mask": output_dir / "target_mask.png",
        "target_overlay": output_dir / "target_overlay.png",
        "query_rgb": output_dir / "query_rgb.png",
        "query_mask": output_dir / "query_mask.png",
        "query_overlay": output_dir / "query_overlay.png",
        "prompts": output_dir / "prompts.txt",
    }
    Image.fromarray(target_rgb).save(artifacts["target_rgb"])
    Image.fromarray(target_mask).save(artifacts["target_mask"])
    Image.fromarray(target_overlay).save(artifacts["target_overlay"])
    Image.fromarray(query_rgb).save(artifacts["query_rgb"])
    Image.fromarray(query_mask).save(artifacts["query_mask"])
    Image.fromarray(query_overlay).save(artifacts["query_overlay"])
    artifacts["prompts"].write_text(
        f"positive: {batch['positive_text_prompt'][0]}\n"
        f"negative: {batch['negative_text_prompt'][0]}\n",
        encoding="utf-8",
    )
    return {name: str(path.resolve()) for name, path in artifacts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=32)
    args = parser.parse_args()
    opt = config_defaults["debug_scannet_prompt"]
    _, test_loader, _, test_dataset = get_multi_dataloader(opt, _LocalAccelerator())
    provider = test_dataset.datasets[0]
    dataset = provider.dataset
    batch = next(iter(test_loader))

    print(
        "Train manifest:",
        {
            "path": str(dataset.train_manifest_path),
            "source": dataset.train_split_source,
            "provisional": dataset.train_split_is_provisional,
            "scenes": len(dataset.train_scene_names),
            "excluded_eval_scenes": len(dataset.eval_scene_names),
        },
    )
    distribution = {
        class_id: sum(len(entries) for entries in by_scene.values())
        for class_id, by_scene in dataset.query_entries_by_class_scene.items()
    }
    print(
        "Query bank:",
        {
            "path": str(dataset.query_bank_path),
            "entries": len(dataset.query_entries),
            "class_distribution": distribution,
        },
    )
    print("Batch target/query:")
    print("  target scene:", batch["scene_name"])
    print("  target frames:", batch["frame_ids"].tolist())
    print("  prompt class:", batch["prompt_class_id"].tolist())
    print("  positive text:", batch["positive_text_prompt"])
    print("  negative text:", batch["negative_text_prompt"])
    print("  query scene:", batch["query_scene_name"])
    print("  query frame:", batch["query_frame_id"].tolist())
    print("  query class:", batch["query_class_id"].tolist())
    print("New batch fields:")
    prompt_keys = (
        "prompt_class_id",
        "positive_text_prompt",
        "negative_text_prompt",
        "binary_mask_output",
        "query_image",
        "query_mask",
        "query_scene_name",
        "query_frame_id",
        "query_class_id",
        "has_image_query",
    )
    for key in prompt_keys:
        value = batch[key]
        if torch.is_tensor(value):
            print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"  {key}: {value!r}")

    sample_indices = np.linspace(
        0, len(provider) - 1, num=min(args.num_samples, len(provider)), dtype=int
    )
    attempts = successes = failures = 0
    for index in sample_indices:
        attempts += 1
        try:
            item = provider.get_item(int(index))
            success = (
                bool(item["has_image_query"])
                and item["query_scene_name"] != item["scene_name"]
                and int(item["query_class_id"]) == int(item["prompt_class_id"])
                and item["query_scene_name"] not in dataset.eval_scene_names
            )
            successes += int(success)
        except Exception as error:
            failures += 1
            print(f"Sampling failure at index {index}: {error}")
    print(
        "Cross-scene sampling:",
        {
            "attempts": attempts,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / attempts if attempts else 0.0,
        },
    )
    print(
        "Visualizations:",
        _save_debug(batch, Path(opt.workspace) / "visualizations"),
    )


if __name__ == "__main__":
    main()
