"""One-batch, no-optimizer-step Stage-J2 parent identity audit."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from accelerate import Accelerator, DataLoaderConfiguration  # noqa: E402
from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import (  # noqa: E402
    _load_cached_gsplat_extension,
    configure_joint_formation_trainability,
    load_model_checkpoint,
    write_token_eru_joint_parent_reference,
)

_load_cached_gsplat_extension()

CONFIG = "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_ddp8"
PARENT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_v1_ddp8/checkpoints/model_step_000710.safetensors"
)
EXPECTED_SHA = "e848f733fe7f3812db15f143787c5e663475fdd3a0bd01b302c2a047f1a33c2d"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move(v, device) for v in value)
    return value


def tensor_summary(value):
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "finite": bool(torch.isfinite(value).all())}
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not PARENT.is_file() or sha256_file(PARENT) != EXPECTED_SHA:
        raise RuntimeError("JointFormation@710 parent missing or SHA mismatch")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty identity workspace: {output}")
    output.mkdir(parents=True, exist_ok=True)
    opt = dataclasses.replace(config_defaults[CONFIG])
    opt.resume = str(PARENT)
    opt.workspace = str(output)
    opt.num_workers = 0
    opt.tsh_ddp8 = False
    opt.use_wandb = False
    opt.eval_before_training = False
    torch.manual_seed(42)
    random.seed(42)
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(int(accelerator.local_process_index))
    model = model_registry[opt.model_type](opt)
    model.to(accelerator.device)
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        raise RuntimeError("parent was not strictly restored")
    loader, _, _, _ = get_multi_dataloader(opt, accelerator)
    batch = move(next(iter(loader)), accelerator.device)
    if batch["input"].shape[1] != 15:
        raise RuntimeError("identity batch is not 8 context + 7 target")
    model.eval()
    with torch.no_grad():
        model.set_token_eru_step(711)
        model.set_token_eru_dino_metric_step(711)
        before = model(batch, compute_quality_metrics=False)
        before_internal = {
            name: getattr(model, name).detach().clone()
            for name in (
                "_token_eru_last_reconstruction_tokens",
                "_token_eru_last_understanding_tokens",
                "_token_eru_understanding_units",
                "_tsh_last_q_abs_live",
                "_tsh_last_student_gaussians_live",
            )
        }
    configure_joint_formation_trainability(model, opt)
    write_token_eru_joint_parent_reference(opt, accelerator)
    model.eval()
    with torch.no_grad():
        model.set_token_eru_step(711)
        model.set_token_eru_dino_metric_step(711)
        after = model(batch, compute_quality_metrics=False)
        after_internal = {
            name: getattr(model, name).detach().clone()
            for name in before_internal
        }
    compared = {}
    max_diff = 0.0
    for key in sorted(set(before) & set(after)):
        if torch.is_tensor(before[key]) and torch.is_tensor(after[key]):
            # gsplat's projected 2D auxiliary is produced by a nondeterministic
            # rasterizer reduction and is not a model output used by the
            # checkpoint/evaluator identity contract.
            if key == "means2d_pred":
                continue
            diff = float((before[key].float() - after[key].float()).abs().max())
            compared[key] = diff
            max_diff = max(max_diff, diff)
    for key in sorted(before_internal):
        diff = float((before_internal[key].float() - after_internal[key].float()).abs().max())
        compared[key] = diff
        max_diff = max(max_diff, diff)
    top_diffs = sorted(
        ((key, value) for key, value in compared.items() if torch.isfinite(torch.tensor(value))),
        key=lambda item: item[1], reverse=True,
    )[:20]
    if max_diff != 0.0:
        raise RuntimeError(
            "parent identity changed by trainability setup: "
            f"max_diff={max_diff} top_diffs={top_diffs}"
        )
    report = {
        "config": CONFIG,
        "parent_checkpoint": str(PARENT),
        "parent_sha256": EXPECTED_SHA,
        "batch_context_views": 8,
        "batch_target_views": 7,
        "stage_local_step": 0,
        "effective_optimizer_step": 710,
        "comparison_max_abs_diff": max_diff,
        "tensor_diffs": compared,
        "top_diffs": top_diffs,
        "finite": all(
            bool(torch.isfinite(value).all())
            for key, value in after.items()
            if torch.is_tensor(value) and key not in ("ssim", "lpips")
        ),
        "optimizer_step_executed": 0,
    }
    (output / "identity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
