#!/usr/bin/env python3
"""Read-only validation of a GSI-v2 intra-epoch full-state checkpoint.

This intentionally performs no optimizer step.  It restores the model,
optimizer, scheduler, and all per-rank RNG sidecars from step 500 and then
does one ordinary forward on a single, non-distributed batch to prove that
the restored model is usable.
"""

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


def _finite(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return math.isfinite(float(value))
    return True


def _parameter_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_rng(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid RNG payload: {path}")
    if not _finite(payload):
        raise RuntimeError(f"non-finite RNG payload: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--zero-forward", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    ckpt = source / "checkpoints"
    model_path = ckpt / "model_step_000500.safetensors"
    optimizer_path = ckpt / "optimizer_step_000500.pth"
    scheduler_path = ckpt / "scheduler_step_000500.pth"
    metadata_path = ckpt / "metadata_step_000500.json"
    required = [model_path, optimizer_path, scheduler_path, metadata_path]
    required += [ckpt / f"rng_step_000500_rank{rank:02d}.pth" for rank in range(8)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("step500 full-state files missing: " + ", ".join(missing))

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if int(metadata.get("optimizer_step", -1)) != 500:
        raise RuntimeError(f"metadata optimizer_step is not 500: {metadata}")
    if int(metadata.get("world_size", -1)) != 8:
        raise RuntimeError(f"metadata world_size is not 8: {metadata}")

    state = load_file(str(model_path), device="cpu")
    model_finite = _finite(state)
    if not model_finite:
        raise RuntimeError("model checkpoint contains non-finite tensors")

    opt_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
    scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=False)
    if not _finite(opt_state) or not _finite(scheduler_state):
        raise RuntimeError("optimizer or scheduler contains non-finite values")
    rng_payloads = [_load_rng(ckpt / f"rng_step_000500_rank{rank:02d}.pth") for rank in range(8)]
    if any(int(payload.get("optimizer_step", -1)) != 500 for payload in rng_payloads):
        raise RuntimeError("one or more RNG sidecars is not at optimizer step 500")

    opt = copy.deepcopy(config_defaults["gsi_v2_recon_scannet_adapt"])
    opt.workspace = str(source)
    opt.resume = str(model_path)
    opt.gsi_v2_resume_mode = "strict"
    opt.num_workers = 0
    opt.num_epochs = 1
    opt.max_iters_per_epoch = 1
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
    model_hash_before = _parameter_hash(model)

    optimizer = _setup_gsi_v2_optimizer(opt, model, accelerator, epoch_start=0)
    optimizer.load_state_dict(opt_state)
    scheduler = setup_scheduler(opt, optimizer, 710, accelerator, epoch_start=0)
    scheduler.load_state_dict(scheduler_state)
    model_hash_after_restore = _parameter_hash(model)
    if model_hash_before != model_hash_after_restore:
        raise RuntimeError("optimizer/scheduler restore changed model parameters")

    report = {
        "source": str(source),
        "model_path": str(model_path),
        "optimizer_step": 500,
        "metadata": metadata,
        "model_keys": len(state),
        "model_numel": int(sum(value.numel() for value in state.values())),
        "model_finite": model_finite,
        "strict_model_restore": True,
        "optimizer_state_entries": len(opt_state.get("state", {})),
        "optimizer_finite": _finite(opt_state),
        "scheduler_last_epoch": int(scheduler_state.get("last_epoch", -1)),
        "scheduler_finite": _finite(scheduler_state),
        "rng_rank_count": len(rng_payloads),
        "rng_steps": [int(payload["optimizer_step"]) for payload in rng_payloads],
        "parameter_hash_before": model_hash_before,
        "parameter_hash_after_restore": model_hash_after_restore,
        "zero_optimizer_step": True,
        "zero_optimizer_step_parameter_hash_unchanged": True,
    }

    if args.zero_forward:
        train_loader, _test_loader, _train_dataset, _test_dataset = get_multi_dataloader(
            opt, accelerator
        )
        model, train_loader = accelerator.prepare(model, train_loader)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.set_train_step(500)
        model.train()
        data = next(iter(train_loader))
        with torch.no_grad():
            output = model(data)
        forward_values = {
            key: value.detach().float().cpu()
            for key, value in output.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        }
        if not _finite(forward_values):
            raise RuntimeError("zero-step restore forward produced non-finite output")
        report["zero_step_forward_finite"] = True
        report["zero_step_forward_scalars"] = {
            key: float(value.item()) for key, value in forward_values.items()
        }
        report["parameter_hash_after_zero_step_forward"] = _parameter_hash(unwrapped)
        report["zero_step_forward_parameter_hash_unchanged"] = (
            report["parameter_hash_after_zero_step_forward"] == model_hash_before
        )
        if not report["zero_step_forward_parameter_hash_unchanged"]:
            raise RuntimeError("zero-step forward changed model parameters")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
