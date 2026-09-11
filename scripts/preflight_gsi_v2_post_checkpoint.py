#!/usr/bin/env python3
"""DDP8 checkpoint-sentinel preflight from the completed GSI step1000 state.

This test publishes a simulated step1000 milestone, then performs exactly two
real optimizer updates (1001 and 1002) in a fresh temporary workspace.  It
does not write the formal resume workspace or modify the parent checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator, DataLoaderConfiguration
from safetensors.torch import load_file

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.options import config_defaults
from tokengs.train import (
    _setup_gsi_v2_optimizer,
    save_gsi_v2_intra_epoch_checkpoint_synchronized,
    setup_scheduler,
)


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


def full_gsi_state_hash(model) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    # GlobalSplatInstanceV2.state_dict() intentionally excludes the frozen
    # perceptual metric module; the remaining 454-key state is the complete
    # GSI checkpoint/model state and is the same namespace used for strict
    # save/restore.  This avoids hashing an unrelated metric child while
    # still proving that no GSI namespace is silently empty.
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        tensor = value.detach().float().cpu().contiguous()
        digest.update(tensor.numpy().tobytes())
        count += tensor.numel()
    return digest.hexdigest(), count


def gather_equal(value: str) -> list[str]:
    values = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(values, value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    workspace = args.workspace.resolve()
    source_ckpt = source / "checkpoints"
    model_path = source_ckpt / "model_step_001000.safetensors"
    optimizer_path = source_ckpt / "optimizer_step_001000.pth"
    scheduler_path = source_ckpt / "scheduler_step_001000.pth"
    if not all(path.is_file() for path in (model_path, optimizer_path, scheduler_path)):
        raise RuntimeError("step1000 source checkpoint is incomplete")

    opt = copy.deepcopy(config_defaults["gsi_v2_recon_scannet_adapt_resume1000"])
    opt.workspace = str(workspace)
    opt.resume = str(model_path)
    opt.gsi_v2_resume_mode = "strict"
    opt.num_workers = 0
    opt.num_epochs = 1
    opt.max_iters_per_epoch = 2
    opt.eval_before_training = False
    opt.gsi_v2_disable_training_eval = True
    opt.abs_ckpt_steps_extra = ()
    opt.abs_ckpt_every = 0
    opt.use_wandb = False

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )
    rank = int(accelerator.process_index)
    world = int(accelerator.num_processes)
    if world != 8:
        raise RuntimeError(f"expected DDP8, got world_size={world}")
    if rank == 0:
        if workspace.exists():
            raise FileExistsError(f"preflight workspace already exists: {workspace}")
        workspace.mkdir(parents=True)
        (workspace / "checkpoints").mkdir()
        (workspace / "config.yaml").write_text(
            "checkpoint_protocol: shared_fs_sentinel_v1\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()

    state = load_file(str(model_path), device="cpu")
    model = model_registry[opt.model_type](opt)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict source restore failed: {incompatible}")
    optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
    scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=False)
    optimizer = _setup_gsi_v2_optimizer(opt, model, accelerator, epoch_start=0)
    optimizer.load_state_dict(optimizer_state)
    train_loader, _, _, _ = get_multi_dataloader(opt, accelerator)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    scheduler = setup_scheduler(opt, optimizer, 710, accelerator, epoch_start=0)
    scheduler.load_state_dict(scheduler_state)

    events = {"rank": rank, "world_size": world, "milestone_entry": time.time()}
    save_gsi_v2_intra_epoch_checkpoint_synchronized(
        opt, accelerator, model, optimizer, scheduler, epoch=1, completed_step=1000
    )
    events["milestone_complete"] = time.time()
    sentinel = workspace / "checkpoints" / "step_001000.complete"
    if not sentinel.is_file():
        raise RuntimeError(f"missing preflight sentinel: {sentinel}")

    step_records = []
    iterator = iter(train_loader)
    for step in (1001, 1002):
        events[f"step_{step}_start"] = time.time()
        optimizer.zero_grad(set_to_none=True)
        data = next(iterator)
        output = model(data)
        loss = output["loss"]
        if not finite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        accelerator.backward(loss)
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        ):
            raise RuntimeError(f"non-finite gradient at step {step}")
        accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)
        optimizer.step()
        scheduler.step()
        digest, parameter_count = full_gsi_state_hash(accelerator.unwrap_model(model))
        hashes = gather_equal(digest)
        if len(set(hashes)) != 1:
            raise RuntimeError(f"full GSI parameter hash mismatch at step {step}: {hashes}")
        scene = str(data["scene_name"][0]) if "scene_name" in data else "?"
        scenes = [None for _ in range(world)]
        dist.all_gather_object(scenes, scene)
        step_records.append({
            "step": step,
            "loss": float(loss.detach().float().cpu().item()),
            "scene_by_rank": scenes,
            "full_parameter_hash": digest,
            "parameter_count": parameter_count,
            "finite": True,
        })
        events[f"step_{step}_done"] = time.time()

    reloaded = model_registry[opt.model_type](opt)
    reload_incompatible = reloaded.load_state_dict(
        load_file(str(sentinel.parent / "model_step_001000.safetensors"), device="cpu"),
        strict=True,
    )
    if reload_incompatible.missing_keys or reload_incompatible.unexpected_keys:
        raise RuntimeError(f"strict sentinel reload failed: {reload_incompatible}")
    reload_hash, reload_count = full_gsi_state_hash(reloaded)
    reload_hashes = gather_equal(reload_hash)
    if len(set(reload_hashes)) != 1:
        raise RuntimeError(f"full GSI reload hash mismatch: {reload_hashes}")
    accelerator.wait_for_everyone()
    if rank == 0:
        report = {
            "world_size": world,
            "workspace": str(workspace),
            "source_step": 1000,
            "simulated_milestone_sentinel": str(sentinel),
            "sentinel_exists": sentinel.is_file(),
            "checkpoint_protocol": "shared_fs_sentinel_v1",
            "no_long_io_nccl_barrier": True,
            "post_sentinel_accelerator_barrier": True,
            "step_records": step_records,
            "full_gsi_parameter_hash_sync": True,
            "strict_sentinel_reload": True,
            "reload_parameter_count": reload_count,
            "events_by_rank": events,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
