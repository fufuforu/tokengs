#!/usr/bin/env python3
"""Create a non-destructive GSI-v2 continuation workspace from step1000."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def _set_yaml_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(key)}:.*$")
    replacement = f"{key}: {value}"
    if pattern.search(text):
        return pattern.sub(replacement, text, count=1)
    return text.rstrip() + "\n" + replacement + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    source_ckpt = source / "checkpoints"
    step = 1000
    required = {
        "model": source_ckpt / f"model_step_{step:06d}.safetensors",
        "optimizer": source_ckpt / f"optimizer_step_{step:06d}.pth",
        "scheduler": source_ckpt / f"scheduler_step_{step:06d}.pth",
        "metadata": source_ckpt / f"metadata_step_{step:06d}.json",
        "config": source_ckpt / f"config_step_{step:06d}.yaml",
    }
    required.update({
        f"rng_rank{rank}": source_ckpt / f"rng_step_{step:06d}_rank{rank:02d}.pth"
        for rank in range(8)
    })
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError("cannot prepare step1000 resume:\n" + "\n".join(missing))
    if destination.exists():
        raise FileExistsError(f"refusing to create or overwrite: {destination}")

    destination.mkdir(parents=True)
    (destination / "checkpoints").mkdir()
    (destination / ".fork_state").mkdir()
    shutil.copy2(required["model"], destination / "model.safetensors")
    shutil.copy2(required["optimizer"], destination / "optimizer.pth")
    shutil.copy2(required["scheduler"], destination / "scheduler.pth")

    config_text = required["config"].read_text(encoding="utf-8")
    config_values = {
        "workspace": "workspace/gsi_v2_recon_scannet_adapt_ddp8_resume1000",
        "experiment_name": "gsi_v2_recon_scannet_adapt_ddp8_resume1000",
        "num_workers": "2",
        "num_epochs": "3",
        "max_iters_per_epoch": "710",
        "log_image_freq": "0",
        "eval_before_training": "false",
        "tsh_fork_continue_step": "1000",
    }
    for key, value in config_values.items():
        config_text = _set_yaml_scalar(config_text, key, value)
    config_text = _set_yaml_scalar(config_text, "gsi_v2_disable_training_eval", "true")
    (destination / "config.yaml").write_text(config_text, encoding="utf-8")

    source_metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
    metadata = dict(source_metadata)
    metadata.update({
        "epoch": 1,
        "step": step,
        "optimizer_step": step,
        "equivalent_global_samples": step * 8,
        "resume_source_checkpoint": str(required["model"]),
        "resume_source_optimizer": str(required["optimizer"]),
        "resume_source_scheduler": str(required["scheduler"]),
        "resume_source_metadata": str(required["metadata"]),
        "resume_first_step": 1001,
        "resume_total_target_step": 2130,
        "resume_epoch": 1,
        "resume_batches_consumed_in_epoch": 290,
        "resume_batches_remaining_in_epoch": 420,
        "resume_num_workers_per_rank": 2,
        "resume_preparation": "non-destructive step1000 fork",
        "visualization_disabled": True,
        "training_eval_disabled": True,
    })
    metadata.update({
        "config_path": str((destination / "config.yaml").resolve()),
        "optimizer_state": str((destination / "optimizer.pth").resolve()),
        "scheduler_state": str((destination / "scheduler.pth").resolve()),
        "rng_state_pattern": str((destination / ".fork_state" / "rank*.pt").resolve()),
    })
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for rank in range(8):
        shutil.copy2(required[f"rng_rank{rank}"], destination / ".fork_state" / f"rank{rank}.pt")

    print(json.dumps({
        "source": str(source),
        "destination": str(destination),
        "source_step": step,
        "resume_epoch": 1,
        "batches_consumed_in_epoch": 290,
        "batches_remaining_in_epoch": 420,
        "first_new_step": 1001,
        "total_target_step": 2130,
        "rng_sidecars": 8,
    }, indent=2))


if __name__ == "__main__":
    main()
