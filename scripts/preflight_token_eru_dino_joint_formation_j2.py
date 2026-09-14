"""Bounded Stage-J2 preflight; never runs the formal 710-step stage."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import (  # noqa: E402
    _joint_formation_prepare_audit,
    _load_cached_gsplat_extension,
    configure_joint_formation_trainability,
    load_model_checkpoint,
    setup_optimizer,
    setup_scheduler,
    write_token_eru_joint_parent_reference,
)

_load_cached_gsplat_extension()

CONFIG = "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_ddp8"
PARENT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_v1_ddp8/checkpoints/model_step_000710.safetensors"
)
PARENT_SHA256 = "e848f733fe7f3812db15f143787c5e663475fdd3a0bd01b302c2a047f1a33c2d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    return value


def gather(value):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        values = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(values, value)
        return values
    return [value]


def manifest_hash(model, trainable_only=False):
    digest = hashlib.sha256()
    names = []
    numel = 0
    for name, parameter in sorted(model.named_parameters()):
        if trainable_only and not parameter.requires_grad:
            continue
        names.append((name, tuple(parameter.shape)))
        numel += int(parameter.numel())
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
    if not names:
        raise RuntimeError("refusing to hash an empty parameter namespace")
    return digest.hexdigest(), len(names), numel


def value_hash(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build(args):
    opt = dataclasses.replace(config_defaults[CONFIG])
    opt.resume = str(PARENT)
    opt.workspace = str(args.output_dir)
    opt.num_workers = 0
    opt.eval_before_training = False
    opt.use_wandb = False
    opt.tsh_ddp8 = bool(args.ddp)
    opt.joint_formation_ddp_manifest_audit = True
    torch.manual_seed(42)
    random.seed(42)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[ddp_kwargs],
    )
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", accelerator.local_process_index))
        torch.cuda.set_device(local_rank)
        accelerator.print(
            f"[j2-preflight] rank={accelerator.process_index} local_rank={local_rank} "
            f"device={torch.cuda.current_device()}"
        )
    model = model_registry[opt.model_type](opt)
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        raise RuntimeError("Stage-J2 parent was not marked as strictly restored")
    categories = configure_joint_formation_trainability(model, opt)
    train_hash, train_count, train_numel = manifest_hash(model, True)
    if (train_count, train_numel) != (903, 364_349_232):
        raise RuntimeError(
            f"Stage-J2 trainable audit mismatch: {(train_count, train_numel)}"
        )
    write_token_eru_joint_parent_reference(opt, accelerator)
    loader, test_loader, _, _ = get_multi_dataloader(opt, accelerator)
    optimizer = setup_optimizer(opt, model, accelerator, 0)
    _joint_formation_prepare_audit(opt, accelerator, model, optimizer, loader, test_loader)
    model, optimizer, loader, test_loader = accelerator.prepare(
        model, optimizer, loader, test_loader
    )
    iters_per_epoch = min(len(loader), int(opt.max_iters_per_epoch))
    scheduler = setup_scheduler(opt, optimizer, iters_per_epoch, accelerator, 0)
    return opt, accelerator, model, optimizer, scheduler, loader, categories, train_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    if not 1 <= args.steps <= 3:
        raise ValueError("J2 preflight is limited to 1..3 optimizer steps")
    if not PARENT.is_file() or sha256_file(PARENT) != PARENT_SHA256:
        raise RuntimeError("JointFormation@710 parent is missing or SHA256 mismatched")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty preflight workspace: {output}")
    output.mkdir(parents=True, exist_ok=True)
    opt, accelerator, model, optimizer, scheduler, loader, categories, train_hash = build(args)
    unwrapped = accelerator.unwrap_model(model)
    batch = move(next(iter(loader)), accelerator.device)
    if batch["input"].shape[1] != 15 or batch["instance_label_output"].shape[1] != 7:
        raise RuntimeError("J2 preflight did not receive formal 8+7 data")
    first_samples = gather({
        "rank": int(accelerator.process_index),
        "scene": str(batch["scene_name"][0]),
        "frames": [int(x) for x in batch["frame_ids"][0].tolist()],
    })
    if accelerator.num_processes == 8 and len({json.dumps(x, sort_keys=True) for x in first_samples}) != 8:
        raise RuntimeError(f"DDP8 first samples are duplicated: {first_samples}")

    records = []
    for local_step in range(1, args.steps + 1):
        effective_step = 710 + local_step
        unwrapped.set_token_eru_step(effective_step)
        unwrapped.set_token_eru_dino_metric_step(effective_step)
        unwrapped.tsh_instance_loss_weight_eff = 1.0
        unwrapped.tsh_unit_grad_eff = 1.0
        unwrapped.teacher_lambda_eff = 0.0
        optimizer.zero_grad(set_to_none=True)
        result = model(batch, compute_quality_metrics=False)
        loss = result["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at local step {local_step}")
        accelerator.backward(loss)
        pre_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not torch.isfinite(torch.tensor(pre_clip, device=accelerator.device)):
            raise FloatingPointError("non-finite pre-clip gradient norm")
        optimizer.step()
        scheduler.step()
        if not all(torch.isfinite(parameter).all() for parameter in unwrapped.parameters()):
            raise FloatingPointError(f"non-finite parameter at local step {local_step}")
        local_value_hash = value_hash(unwrapped)
        all_value_hashes = gather(local_value_hash)
        if len(set(all_value_hashes)) != 1:
            raise RuntimeError(f"post-step parameter hash mismatch: {all_value_hashes}")
        records.append({
            "stage_local_step": local_step,
            "effective_optimizer_step": effective_step,
            "loss": float(loss.detach()),
            "pre_clip_grad_norm": pre_clip,
            "lr": [float(group["lr"]) for group in optimizer.param_groups],
            "parameter_hash": local_value_hash,
        })

    if accelerator.is_main_process:
        state_path = output / "preflight_model.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in unwrapped.state_dict().items()}, str(state_path))
        torch.save(optimizer.state_dict(), output / "preflight_optimizer.pth")
        torch.save(scheduler.state_dict(), output / "preflight_scheduler.pth")
    accelerator.wait_for_everyone()
    state_path = output / "preflight_model.safetensors"
    restored = load_file(str(state_path), device="cpu")
    current = {k: v.detach().cpu() for k, v in unwrapped.state_dict().items()}
    if set(restored) != set(current):
        raise RuntimeError("strict preflight save/reload key mismatch")
    max_diff = max(float((current[k] - restored[k]).abs().max()) for k in current)
    if max_diff != 0.0:
        raise RuntimeError(f"strict preflight save/reload diff={max_diff}")
    dino = unwrapped.token_eru_dino_encoder.dino_extractor.__dict__.get("_dino_model")
    dino_optimizer_entries = 0 if dino is None else sum(
        id(p) in {id(q) for g in optimizer.param_groups for q in g["params"]}
        for p in dino.parameters()
    )
    if dino_optimizer_entries:
        raise RuntimeError("DINO parameter entered Stage-J2 optimizer")
    report = {
        "config": CONFIG,
        "world_size": int(accelerator.num_processes),
        "steps": args.steps,
        "first_new_optimizer_step": 1,
        "final_stage_local_step": args.steps,
        "first_effective_step": 711,
        "final_effective_step": 710 + args.steps,
        "batches_skipped": 0,
        "first_samples": first_samples,
        "trainable_tensor_count": 903,
        "trainable_numel": 364_349_232,
        "trainable_manifest_hash": train_hash,
        "trainability": categories,
        "records": records,
        "strict_save_reload": True,
        "strict_save_reload_max_diff": max_diff,
        "dino_in_checkpoint": any("_dino_model" in key for key in restored),
        "dino_optimizer_entries": dino_optimizer_entries,
        "target_image_to_dino": False,
        "p_u_used_in_formal_eval": False,
        "formal_training_started": False,
        "formal_evaluation_started": False,
    }
    if accelerator.is_main_process:
        (output / "preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

