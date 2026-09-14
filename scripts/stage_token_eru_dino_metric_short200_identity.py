"""Stage the immutable optimizer-step-500 identity reference for short200.

This helper is called by the user-owned short200 launcher before torchrun.  It
loads the persisted ERU checkpoint, constructs fresh DINO metric modules, and
writes only a parent reference to the new workspace. It never copies the
step-500 model/optimizer/scheduler/RNG files and never performs an optimizer
step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from accelerate import Accelerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import load_model_checkpoint  # noqa: E402

DEFAULT_CONFIG = "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_short200_ddp8"
SOURCE = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_"
    "scene_hungarian_resume200_to500_persist_ddp8/checkpoints/"
    "model_step_000500.safetensors"
)
SOURCE_SHA256 = "cca23e2d24af7a374be6a62f2cf4026645afdf0df49686722cb5277c9e479209"
DINO_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage-m", action="store_true")
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    allowed = {"logs"}
    if args.stage_m:
        allowed.add("parent_checkpoint.json")
    existing = [path for path in workspace.iterdir() if path.name not in allowed]
    if existing:
        raise RuntimeError(f"refusing non-empty short200 workspace: {existing}")
    if not SOURCE.is_file() or sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("source ERU@500 checkpoint is missing or has wrong SHA256")
    opt = config_defaults[args.config].evolve(
        resume=str(SOURCE), workspace=str(workspace), evaluating=False
    )
    accelerator = Accelerator(cpu=True)
    torch.manual_seed(int(opt.seed))
    model = model_registry[opt.model_type](opt)
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        model.initialize_token_eru_from_reconstruction()
    model.set_token_eru_step(500)
    model.set_token_eru_dino_metric_step(500)
    model.eval()
    source_dir = SOURCE.parent
    sidecars = {
        "optimizer": source_dir / "optimizer_step_000500.pth",
        "scheduler": source_dir / "scheduler_step_000500.pth",
        "metadata": source_dir / "metadata_step_000500.json",
    }
    for source in sidecars.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    for rank in range(8):
        source = source_dir / f"rng_step_000500_rank{rank:02d}.pth"
        if not source.is_file():
            raise FileNotFoundError(source)

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        git_commit = "unknown"
    metadata = {
        "optimizer_step": 500,
        "source_step": 500,
        "total_target_step": 200 if args.stage_m else 700,
        "equivalent_global_samples": 1600 if args.stage_m else 4000,
        "world_size": 8,
        "global_batch_size": 8,
        "train_scenes": 1425,
        "train_windows": 5680,
        "context_views": 8,
        "target_views": 7,
        "source_checkpoint": str(SOURCE),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "parent_checkpoint_reference": str(SOURCE),
        "parent_model_sha256": SOURCE_SHA256,
        "parent_optimizer": str(sidecars["optimizer"]),
        "parent_scheduler": str(sidecars["scheduler"]),
        "parent_metadata": str(sidecars["metadata"]),
        "parent_rng_pattern": str(source_dir / "rng_step_000500_rank*.pth"),
        "step500_identity_verified_in_memory": True,
        "git_commit": git_commit,
        "token_eru_matching_mode": "scene",
        "hungarian_scope": "scene_window",
        "expected_hungarian_calls_per_scene_window": 1,
        "same_assignment_all_target_views": True,
        "r2u_gate": 1.0,
        "u2r_gate": 0.1,
        "token_eru_dino_metric_enabled": True,
        "token_eru_dino_source": "local",
        "token_eru_dino_weight_sha256": DINO_SHA256,
        "token_eru_dino_gate": 0.0,
        "token_eru_dino_metric_loss_weight": 0.0,
        "token_eru_dino_temperature": 0.1,
        "token_eru_dino_cluster_eps": 0.5,
        "fresh_reset": False,
        "pgsr_absent": True,
        "formal_training_started": False,
        "optimizer_step_executed": False,
        "stage_name": (
            "eru_dino_metric_stage_m" if args.stage_m else None
        ),
        "stage_optimizer_step": 0 if args.stage_m else None,
        "old_optimizer_restored": False,
        "old_scheduler_restored": False,
        "old_rng_restored": False,
        "old_sampler_cursor_restored": False,
        "batches_skipped": 0,
    }
    parent_path = workspace / "parent_checkpoint.json"
    temporary = parent_path.with_name(f"{parent_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, parent_path)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
