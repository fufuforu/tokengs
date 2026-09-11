#!/usr/bin/env python3
"""Read-only strict restore audit for the completed GSI-v2 step-1000 save."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from safetensors.torch import load_file

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.options import config_defaults
from tokengs.train import _setup_gsi_v2_optimizer, setup_scheduler


def finite(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (float, int)):
        return math.isfinite(float(value))
    return True


def parameter_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zero-forward", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    step = int(args.step)
    ckpt = source / "checkpoints"
    files = {
        "model": ckpt / f"model_step_{step:06d}.safetensors",
        "optimizer": ckpt / f"optimizer_step_{step:06d}.pth",
        "scheduler": ckpt / f"scheduler_step_{step:06d}.pth",
        "metadata": ckpt / f"metadata_step_{step:06d}.json",
        "config": ckpt / f"config_step_{step:06d}.yaml",
    }
    files.update({
        f"rng_rank{rank}": ckpt / f"rng_step_{step:06d}_rank{rank:02d}.pth"
        for rank in range(8)
    })
    missing = [f"{name}: {path}" for name, path in files.items() if not path.is_file()]
    if missing:
        raise RuntimeError("step1000 full-state files missing:\n" + "\n".join(missing))

    metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
    if int(metadata.get("optimizer_step", -1)) != step:
        raise RuntimeError(f"metadata optimizer_step mismatch: {metadata}")
    if int(metadata.get("world_size", -1)) != 8:
        raise RuntimeError(f"metadata world_size mismatch: {metadata}")
    state = load_file(str(files["model"]), device="cpu")
    if not finite(state):
        raise RuntimeError("model checkpoint is non-finite")
    optimizer_state = torch.load(files["optimizer"], map_location="cpu", weights_only=False)
    scheduler_state = torch.load(files["scheduler"], map_location="cpu", weights_only=False)
    if not finite(optimizer_state) or not finite(scheduler_state):
        raise RuntimeError("optimizer/scheduler is non-finite")
    rng_payloads = [
        torch.load(files[f"rng_rank{rank}"], map_location="cpu", weights_only=False)
        for rank in range(8)
    ]
    if any(int(payload.get("optimizer_step", -1)) != step for payload in rng_payloads):
        raise RuntimeError("one or more RNG sidecars is not at step1000")
    if any(not finite(payload) for payload in rng_payloads):
        raise RuntimeError("one or more RNG sidecars is non-finite")

    opt = copy.deepcopy(config_defaults["gsi_v2_recon_scannet_adapt"])
    opt.workspace = str(source)
    opt.resume = str(files["model"])
    opt.gsi_v2_resume_mode = "strict"
    opt.num_workers = 0
    opt.num_epochs = 3
    opt.max_iters_per_epoch = 710
    opt.eval_before_training = False
    opt.use_wandb = False
    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )
    model = model_registry[opt.model_type](opt)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict model restore failed: {incompatible}")
    model_hash = parameter_hash(model)
    optimizer = _setup_gsi_v2_optimizer(opt, model, accelerator, epoch_start=0)
    optimizer.load_state_dict(optimizer_state)
    scheduler = setup_scheduler(opt, optimizer, 710, accelerator, epoch_start=0)
    scheduler.load_state_dict(scheduler_state)
    for group, saved_lr in zip(optimizer.param_groups, scheduler_state["_last_lr"]):
        group["lr"] = float(saved_lr)
    if parameter_hash(model) != model_hash:
        raise RuntimeError("optimizer/scheduler restore changed model parameters")
    if int(scheduler_state.get("last_epoch", -1)) != step:
        raise RuntimeError(f"scheduler last_epoch mismatch: {scheduler_state}")
    lr = [float(group["lr"]) for group in optimizer.param_groups]
    if any(not math.isfinite(value) for value in lr):
        raise RuntimeError("optimizer LR is non-finite")
    if len(optimizer_state.get("state", {})) != 453:
        raise RuntimeError(
            f"unexpected optimizer state entry count: {len(optimizer_state.get('state', {}))}"
        )
    lineage = model.lineage_metadata()
    if len(state) != 454:
        raise RuntimeError(f"GSI checkpoint key count is not 454: {len(state)}")

    report = {
        "source": str(source),
        "step": step,
        "files": {key: str(value) for key, value in files.items()},
        "metadata": metadata,
        "model_keys": len(state),
        "model_numel": int(sum(value.numel() for value in state.values())),
        "model_finite": finite(state),
        "strict_model_restore": True,
        "optimizer_state_entries": len(optimizer_state.get("state", {})),
        "optimizer_finite": finite(optimizer_state),
        "optimizer_lr": lr,
        "scheduler_last_epoch": int(scheduler_state["last_epoch"]),
        "scheduler_last_lr": [float(value) for value in scheduler_state["_last_lr"]],
        "scheduler_finite": finite(scheduler_state),
        "rng_rank_count": len(rng_payloads),
        "rng_steps": [int(payload["optimizer_step"]) for payload in rng_payloads],
        "parameter_hash_before_restore": model_hash,
        "parameter_hash_after_restore": parameter_hash(model),
        "zero_optimizer_step": True,
        "official_lineage": lineage,
        "official_checkpoint_keys": 454,
    }

    if args.zero_forward:
        train_loader, _, _, _ = get_multi_dataloader(opt, accelerator)
        model, train_loader = accelerator.prepare(model, train_loader)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.set_train_step(step)
        before = parameter_hash(unwrapped)
        data = next(iter(train_loader))
        with torch.no_grad():
            output = model(data)
        scalars = {
            key: float(value.detach().float().cpu().item())
            for key, value in output.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        }
        if not finite(scalars):
            raise RuntimeError("zero-step forward produced non-finite scalars")
        after = parameter_hash(unwrapped)
        report.update({
            "zero_step_forward_finite": True,
            "zero_step_forward_scalars": scalars,
            "zero_step_forward_parameter_hash_unchanged": before == after,
        })
        if before != after:
            raise RuntimeError("zero-step forward changed model parameters")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
