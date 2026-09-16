"""Bounded EQC-v1 preflight using the production construction path.

This script is deliberately limited to three optimizer steps.  It restores
only the J2 model parent, never restores an optimizer/cursor, and writes only
to the caller-provided preflight directory.
"""

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
)

_load_cached_gsplat_extension()

CONTROL = "semantic_v6_j2_local250_eqc_v1_control_short200_ddp8"
TREATMENT = "semantic_v6_j2_local250_eqc_v1_treatment_short200_ddp8"
PARENT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_"
    "dino_metric_joint_formation_j2_ddp8/checkpoints/"
    "model_step_000250.safetensors"
)
PARENT_SHA256 = "61debbf5f55b76eea16012aea00fe19291309177bfdc6dd0420f6d1cb58d8cba"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def parameter_manifest(model: torch.nn.Module, trainable_only: bool = False):
    digest = hashlib.sha256()
    names = []
    numel = 0
    for name, parameter in sorted(model.named_parameters()):
        if trainable_only and not parameter.requires_grad:
            continue
        names.append({"name": name, "shape": list(parameter.shape)})
        numel += int(parameter.numel())
        digest.update(name.encode())
        digest.update(repr(tuple(parameter.shape)).encode())
    return digest.hexdigest(), len(names), numel, names


def value_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        value = parameter.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
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


def build(args):
    config_name = TREATMENT if args.treatment else CONTROL
    opt = dataclasses.replace(config_defaults[config_name])
    opt.resume = str(PARENT)
    opt.workspace = str(args.output_dir)
    opt.num_workers = 0
    opt.eval_before_training = False
    opt.use_wandb = False
    opt.tsh_ddp8 = bool(args.ddp)
    opt.joint_formation_ddp_manifest_audit = True
    torch.manual_seed(42)
    random.seed(42)
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", accelerator.local_process_index))
        torch.cuda.set_device(local_rank)
    model = model_registry[opt.model_type](opt)
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        raise RuntimeError("J2 parent strict restore was not recorded")
    categories = configure_joint_formation_trainability(model, opt)
    train_manifest = parameter_manifest(model, True)
    loader, test_loader, _, _ = get_multi_dataloader(opt, accelerator)
    optimizer = setup_optimizer(opt, model, accelerator, 0)
    _joint_formation_prepare_audit(opt, accelerator, model, optimizer, loader, test_loader)
    model, optimizer, loader, test_loader = accelerator.prepare(
        model, optimizer, loader, test_loader
    )
    scheduler = setup_scheduler(opt, optimizer, min(len(loader), int(opt.max_iters_per_epoch)), accelerator, 0)
    return config_name, opt, accelerator, model, optimizer, scheduler, loader, categories, train_manifest


def optimizer_audit(model, optimizer):
    named = dict(model.named_parameters())
    ids = []
    names = []
    missing = []
    frozen = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            ids.append(id(parameter))
            found = [name for name, value in named.items() if value is parameter]
            if not found:
                missing.append(id(parameter))
            else:
                names.append(found[0])
                if not parameter.requires_grad:
                    frozen.append(found[0])
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    return {
        "group_count": len(optimizer.param_groups),
        "parameter_count": len(ids),
        "duplicate_count": len(ids) - len(set(ids)),
        "missing_from_model": missing,
        "frozen_in_optimizer": frozen,
        "trainable_not_in_optimizer": sum(1 for p in trainable if p not in set(ids)),
        "parameter_names_sha256": hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest(),
        "groups": [
            {"name": g.get("name"), "lr": float(g["lr"]), "weight_decay": float(g["weight_decay"]), "count": len(g["params"])}
            for g in optimizer.param_groups
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--treatment", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 3:
        raise ValueError("preflight is limited to 1..3 optimizer steps")
    if not PARENT.is_file() or sha256_file(PARENT) != PARENT_SHA256:
        raise RuntimeError("J2 local250 parent is missing or SHA mismatched")
    output = args.output_dir.resolve()
    if output.exists() and any(entry.name != "logs" for entry in output.iterdir()):
        raise RuntimeError(f"refusing non-empty preflight output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config_name, opt, accelerator, model, optimizer, scheduler, loader, categories, train_manifest = build(args)
    unwrapped = accelerator.unwrap_model(model)
    first_batch = move(next(iter(loader)), accelerator.device)
    if int(first_batch["input"].shape[1]) != 15:
        raise RuntimeError("preflight batch is not 8+7")
    first_samples = gather({
        "rank": int(accelerator.process_index),
        "scene": str(first_batch["scene_name"][0]),
        "frames": [int(x) for x in first_batch["frame_ids"][0].tolist()],
    })
    if accelerator.num_processes == 8 and len({json.dumps(x, sort_keys=True) for x in first_samples}) != 8:
        raise RuntimeError(f"DDP8 first samples duplicated: {first_samples}")
    records = []
    for local_step in range(1, args.steps + 1):
        effective_step = 960 + local_step
        unwrapped.set_token_eru_step(effective_step)
        unwrapped.set_token_eru_dino_metric_step(effective_step)
        unwrapped.set_token_eru_early_query_step(local_step)
        unwrapped.tsh_instance_loss_weight_eff = 1.0
        unwrapped.tsh_unit_grad_eff = 1.0
        unwrapped.teacher_lambda_eff = 0.0
        optimizer.zero_grad(set_to_none=True)
        result = model(first_batch, compute_quality_metrics=False)
        loss = result["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {local_step}")
        accelerator.backward(loss)
        pre_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not torch.isfinite(torch.tensor(pre_clip, device=accelerator.device)):
            raise FloatingPointError(f"non-finite gradient norm at step {local_step}")
        optimizer.step()
        scheduler.step()
        if not all(bool(torch.isfinite(p).all()) for p in unwrapped.parameters()):
            raise FloatingPointError(f"non-finite parameter at step {local_step}")
        post_hash = value_hash(unwrapped)
        hashes = gather(post_hash)
        if len(set(hashes)) != 1:
            raise RuntimeError(f"post-step parameter hash mismatch: {hashes}")
        records.append({
            "stage_local_step": local_step,
            "effective_optimizer_step": effective_step,
            "loss": float(loss.detach().cpu()),
            "pre_clip_grad_norm": pre_clip,
            "post_step_parameter_hash": post_hash,
            "lr": [float(g["lr"]) for g in optimizer.param_groups],
        })
    if accelerator.is_main_process:
        save_file({k: v.detach().cpu().contiguous() for k, v in unwrapped.state_dict().items()}, str(output / "preflight_model.safetensors"))
        torch.save(optimizer.state_dict(), output / "preflight_optimizer.pth")
        torch.save(scheduler.state_dict(), output / "preflight_scheduler.pth")
    accelerator.wait_for_everyone()
    restored = load_file(str(output / "preflight_model.safetensors"), device="cpu")
    current = {k: v.detach().cpu() for k, v in unwrapped.state_dict().items()}
    if set(restored) != set(current):
        raise RuntimeError("strict preflight model key mismatch")
    max_diff = max(float((current[k] - restored[k]).abs().max()) for k in current)
    if max_diff != 0.0:
        raise RuntimeError(f"strict save/reload max diff={max_diff}")
    dino = getattr(getattr(unwrapped, "token_eru_dino_encoder", None), "dino_extractor", None)
    dino_model = None if dino is None else dino.__dict__.get("_dino_model")
    optimizer_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    dino_optimizer_entries = 0 if dino_model is None else sum(id(p) in optimizer_ids for p in dino_model.parameters())
    report = {
        "config": config_name,
        "world_size": int(accelerator.num_processes),
        "steps": args.steps,
        "parent": {"path": str(PARENT), "sha256": PARENT_SHA256},
        "first_samples": first_samples,
        "trainable_tensor_count": train_manifest[1],
        "trainable_numel": train_manifest[2],
        "trainable_manifest_hash": train_manifest[0],
        "trainability": categories,
        "optimizer_audit": optimizer_audit(unwrapped, optimizer),
        "records": records,
        "strict_save_reload": True,
        "strict_save_reload_max_diff": max_diff,
        "dino_optimizer_entries": dino_optimizer_entries,
        "dino_in_checkpoint": any("_dino_model" in key.lower() for key in restored),
        "target_image_to_dino": False,
        "p_u_used_in_formal_eval": False,
        "metric_cluster_formal": False,
        "formal_training_started": False,
        "formal_evaluation_started": False,
    }
    if accelerator.is_main_process:
        (output / "preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
