#!/usr/bin/env python3
"""Materialize a non-destructive GSI-v2 step-500 fork workspace.

The source checkpoint directory is never written.  The destination receives
workspace-level names understood by tokengs.train plus per-rank fork RNG
sidecars, so the normal trainer can skip the first 500 batches of epoch 0 and
start the next optimizer update at step 501.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    source_ckpt = source / "checkpoints"
    required = {
        "model": source_ckpt / "model_step_000500.safetensors",
        "optimizer": source_ckpt / "optimizer_step_000500.pth",
        "scheduler": source_ckpt / "scheduler_step_000500.pth",
        "metadata": source_ckpt / "metadata_step_000500.json",
        "config": source_ckpt / "config_step_000500.yaml",
    }
    required.update(
        {
            f"rng_rank{rank}": source_ckpt / f"rng_step_000500_rank{rank:02d}.pth"
            for rank in range(8)
        }
    )
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError("cannot prepare resume fork; missing files:\n" + "\n".join(missing))
    if destination.exists():
        raise FileExistsError(
            f"refusing to create or overwrite existing resume workspace: {destination}"
        )

    destination.mkdir(parents=True)
    (destination / "checkpoints").mkdir()
    (destination / ".fork_state").mkdir()

    shutil.copy2(required["model"], destination / "model.safetensors")
    shutil.copy2(required["optimizer"], destination / "optimizer.pth")
    shutil.copy2(required["scheduler"], destination / "scheduler.pth")
    shutil.copy2(required["config"], destination / "config.yaml")

    with required["metadata"].open("r", encoding="utf-8") as handle:
        source_metadata = json.load(handle)
    metadata = dict(source_metadata)
    metadata.update(
        {
            "epoch": 0,
            "step": 500,
            "optimizer_step": 500,
            "equivalent_global_samples": 4000,
            "resume_source_checkpoint": str(required["model"]),
            "resume_source_optimizer": str(required["optimizer"]),
            "resume_source_scheduler": str(required["scheduler"]),
            "resume_first_step": 501,
            "resume_total_target_step": 2130,
            "resume_num_workers_per_rank": 2,
            "resume_preparation": "non-destructive step500 fork",
        }
    )
    metadata["config_path"] = str((destination / "config.yaml").resolve())
    metadata["optimizer_state"] = str((destination / "optimizer.pth").resolve())
    metadata["scheduler_state"] = str((destination / "scheduler.pth").resolve())
    metadata["rng_state_pattern"] = str((destination / ".fork_state" / "rank*.pt").resolve())
    with (destination / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    for rank in range(8):
        shutil.copy2(
            required[f"rng_rank{rank}"],
            destination / ".fork_state" / f"rank{rank}.pt",
        )

    print(json.dumps({
        "source": str(source),
        "destination": str(destination),
        "source_step": 500,
        "first_new_step": 501,
        "total_target_step": 2130,
        "rng_sidecars": 8,
    }, indent=2))


if __name__ == "__main__":
    main()
