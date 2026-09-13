#!/usr/bin/env python3
"""Read-only DDP8 audit of the first batch after a step-200 fork."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from torch.utils.data._utils.collate import default_collate

from tokengs.data import get_multi_dataloader
from tokengs.options import config_defaults


def _hash_value(digest: "hashlib._Hash", value: Any, path: str) -> None:
    digest.update(path.encode("utf-8"))
    digest.update(b"\0")
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value):
            _hash_value(digest, value[key], f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for index, item in enumerate(value):
            _hash_value(digest, item, f"{path}[{index}]")
    elif isinstance(value, (str, int, float, bool)) or value is None:
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(value).encode("utf-8"))
    else:
        raise TypeError(f"unsupported batch value at {path}: {type(value)!r}")


def _batch_hash(batch: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, batch, "batch")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="semantic_v6_absolute_units_true_shared_token_eru1_scene_hungarian_short200_ddp8",
    )
    parser.add_argument("--skip", type=int, default=200)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.config not in config_defaults:
        raise ValueError(f"unknown config: {args.config}")
    if args.skip < 0:
        raise ValueError("skip must be non-negative")

    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )
    if accelerator.num_processes != 8:
        raise RuntimeError(f"expected DDP8, got world_size={accelerator.num_processes}")

    opt = config_defaults[args.config].evolve(
        evaluating=False,
        num_workers=2,
        batch_size=1,
        max_iters_per_epoch=710,
        prompt_save_validation_checkpoints=False,
        eval_before_training=False,
        use_wandb=False,
    )
    train_loader, test_loader, _train_dataset, _test_dataset = get_multi_dataloader(
        opt, accelerator
    )
    train_loader, test_loader = accelerator.prepare(train_loader, test_loader)
    del test_loader
    if len(train_loader) != 710:
        raise RuntimeError(
            f"expected prepared local loader length 710, got {len(train_loader)}"
        )

    # Enumerate the prepared batch sampler rather than decoding discarded
    # samples.  This is the same index stream consumed by the production
    # iterator; only the one candidate batch is materialized for fingerprinting.
    batch_sampler = train_loader.batch_sampler
    selected_indices = None
    for batch_index, indices in enumerate(batch_sampler):
        if batch_index == args.skip:
            selected_indices = list(indices)
            break
    if selected_indices is None:
        raise RuntimeError(f"prepared sampler ended before local batch {args.skip}")
    if len(selected_indices) != 1:
        raise RuntimeError(f"expected per-rank batch size 1, got indices={selected_indices}")
    batch = default_collate([train_loader.dataset[selected_indices[0]]])
    scene = str(batch["scene_name"][0])
    frame_ids = [
        int(value) for value in batch["frame_ids"][0].detach().cpu().tolist()
    ]
    record = {
        "rank": int(accelerator.process_index),
        "world_size": int(accelerator.num_processes),
        "skip_local_batches": int(args.skip),
        "selected_dataset_index": int(selected_indices[0]),
        "prepared_loader_len": int(len(train_loader)),
        "scene": scene,
        "frame_ids": frame_ids,
        "context_frame_ids": frame_ids[:8],
        "target_frame_ids": frame_ids[8:],
        "batch_hash": _batch_hash(batch),
        "batch_keys": sorted(batch),
        "tensor_shapes": {
            key: list(value.shape)
            for key, value in sorted(batch.items())
            if torch.is_tensor(value)
        },
    }
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    (output_dir / f"rank{accelerator.process_index:02d}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        records = [
            json.loads((output_dir / f"rank{rank:02d}.json").read_text())
            for rank in range(8)
        ]
        if len({item["scene"] for item in records}) != 8:
            raise RuntimeError("the eight ranks did not receive distinct scenes")
        if any(
            len(item["context_frame_ids"]) != 8
            or len(item["target_frame_ids"]) != 7
            for item in records
        ):
            raise RuntimeError("the resumed batch is not 8+7")
        summary = {
            "config": args.config,
            "skip_local_batches": int(args.skip),
            "world_size": 8,
            "all_rank_scenes_distinct": True,
            "records": records,
            "formal_dataloader_path": "tokengs.data.get_multi_dataloader + Accelerator.prepare",
            "optimizer_steps": 0,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
