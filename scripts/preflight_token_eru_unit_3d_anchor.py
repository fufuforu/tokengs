"""Bounded single-card/DDP8 preflight for TokenGS-ERU-3DAnchor-v1."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
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
    configure_joint_formation_trainability,
    load_model_checkpoint,
    setup_optimizer,
    setup_scheduler,
)


def _load_cached_gsplat_extension() -> None:
    so_path = os.environ.get("GSPLAT_PRECOMPILED_SO")
    if not so_path or not os.path.isfile(so_path) or "gsplat_cuda" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location("gsplat_cuda", so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load cached gsplat extension: {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["gsplat_cuda"] = module
    import gsplat
    sys.modules.setdefault("gsplat.csrc", module)
    setattr(gsplat, "csrc", module)


_load_cached_gsplat_extension()

PARENT = Path(
    "/space/mawb/tokengs/workspace/"
    "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_ddp8/checkpoints/model_step_000250.safetensors"
)
PARENT_SHA = "61debbf5f55b76eea16012aea00fe19291309177bfdc6dd0420f6d1cb58d8cba"
A0 = "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_local250_control_short200_ddp8"
A1 = "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_local250_3d_anchor_v1_short200_ddp8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
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


def hash_names(model, trainable_only=False):
    digest = hashlib.sha256()
    names = []
    count = 0
    numel = 0
    for name, parameter in sorted(model.named_parameters()):
        if trainable_only and not parameter.requires_grad:
            continue
        names.append([name, list(parameter.shape)])
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        count += 1
        numel += parameter.numel()
    return digest.hexdigest(), names, count, numel


def value_hash(model, trainable_only=False):
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if trainable_only and not parameter.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        count += 1
    if count == 0:
        raise RuntimeError("empty parameter value hash")
    return digest.hexdigest()


def gradient_hash(model, trainable_only=False):
    """Hash gradient values for diagnosing pre-optimizer DDP divergence."""
    digest = hashlib.sha256()
    count = 0
    missing = 0
    for name, parameter in sorted(model.named_parameters()):
        if trainable_only and not parameter.requires_grad:
            continue
        digest.update(name.encode())
        if parameter.grad is None:
            digest.update(b"<none>")
            missing += 1
        else:
            digest.update(parameter.grad.detach().cpu().contiguous().numpy().tobytes())
        count += 1
    if count == 0:
        raise RuntimeError("empty gradient hash")
    return digest.hexdigest(), missing


def missing_gradient_names(model, trainable_only=True):
    return [
        name
        for name, parameter in sorted(model.named_parameters())
        if (not trainable_only or parameter.requires_grad) and parameter.grad is None
    ]


def distributed_max_range(model, *, gradients=False, trainable_only=True):
    """Return the largest rank-to-rank scalar range and its parameter name."""
    entries = []
    for name, parameter in sorted(model.named_parameters()):
        if trainable_only and not parameter.requires_grad:
            continue
        value = parameter.grad if gradients else parameter.detach()
        if value is None:
            entries.append((name, None, None))
            continue
        value = value.detach()
        entries.append((name, value.amin().float(), value.amax().float()))
    if not entries:
        raise RuntimeError("empty distributed range audit")
    device = next(parameter for parameter in model.parameters()).device
    local_min = torch.tensor(
        [float(item[1]) if item[1] is not None else 0.0 for item in entries],
        dtype=torch.float32,
        device=device,
    )
    local_max = torch.tensor(
        [float(item[2]) if item[2] is not None else 0.0 for item in entries],
        dtype=torch.float32,
        device=device,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        global_min = local_min.clone()
        global_max = local_max.clone()
        torch.distributed.all_reduce(global_min, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX)
    else:
        global_min, global_max = local_min, local_max
    ranges = (global_max - global_min).detach().cpu()
    index = int(torch.argmax(ranges))
    return float(ranges[index]), entries[index][0]


def object_hash(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous()
        return hashlib.sha256(value.numpy().tobytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def batch_hash(batch):
    payload = []
    for key, value in sorted(batch.items()):
        if torch.is_tensor(value):
            payload.append((key, str(value.dtype), tuple(value.shape), object_hash(value)))
    return hashlib.sha256(repr(payload).encode()).hexdigest()


def gather_objects(value):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        values = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(values, value)
        return values
    return [value]


def set_training_schedule(model, opt, effective_step: int) -> None:
    """Mirror formal train_step schedule assignments for the preflight."""
    model.set_token_eru_step(effective_step)
    model.set_token_eru_dino_metric_step(effective_step)
    if hasattr(model, "compute_instance_group_lambda_eff"):
        model.instance_group_lambda_eff = model.compute_instance_group_lambda_eff(
            effective_step, opt
        )
    if hasattr(model, "compute_teacher_lambda_eff"):
        model.teacher_lambda_eff = model.compute_teacher_lambda_eff(
            effective_step, opt
        )
    if hasattr(model, "compute_instance_stage_eff"):
        model.instance_stage_eff = model.compute_instance_stage_eff(
            effective_step, opt
        )
    if bool(getattr(opt, "abs_true_shared_units", False)) and hasattr(
        model, "compute_tsh_effs"
    ):
        model.tsh_instance_loss_weight_eff = 1.0
        model.tsh_unit_grad_eff = 1.0
    if bool(getattr(opt, "abs_true_shared_units", False)) and hasattr(
        model, "compute_tsh_mbm_u2r_eff"
    ):
        model.tsh_mbm_u2r_eff = model.compute_tsh_mbm_u2r_eff(
            effective_step, opt
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=(A0, A1), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()
    if args.steps not in (1, 3):
        raise ValueError("preflight permits only 1 or 3 steps")
    if not PARENT.is_file() or sha256_file(PARENT) != PARENT_SHA:
        raise RuntimeError("J2 local250 parent missing or SHA mismatch")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and int(os.environ.get("RANK", "0")) == 0:
        raise RuntimeError(f"refusing non-empty preflight workspace: {output}")
    output.mkdir(parents=True, exist_ok=True)
    opt = dataclasses.replace(config_defaults[args.config])
    opt.resume = str(PARENT)
    opt.workspace = str(output)
    opt.num_workers = 0
    opt.tsh_ddp8 = bool(args.ddp)
    opt.use_wandb = False
    opt.eval_before_training = False
    rank = int(os.environ.get("RANK", "0"))
    random.seed(42 + rank)
    np.random.seed(42 + rank)
    torch.manual_seed(42 + rank)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        torch.cuda.manual_seed_all(42 + rank)
    accelerator = Accelerator(
        mixed_precision="no",
        # The active JointFormation graph contains a small set of trainable
        # compatibility parameters that are legitimately unused on a given
        # batch.  Let DDP discover those parameters so the remaining buckets
        # are reduced; this does not change the forward or optimizer graph.
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )
    model = model_registry[opt.model_type](opt).to(accelerator.device)
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        raise RuntimeError("strict parent restore marker missing")
    set_training_schedule(model, opt, 960)
    categories = configure_joint_formation_trainability(model, opt)
    optimizer = setup_optimizer(opt, model, accelerator, 0)
    scheduler = setup_scheduler(opt, optimizer, 200, accelerator, 0)
    loader, _, _, _ = get_multi_dataloader(opt, accelerator)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    base_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        print(
            "[ddp-preflight] prepared_model_type="
            f"{type(model).__module__}.{type(model).__name__} "
            f"num_processes={accelerator.num_processes} "
            f"sync_gradients={accelerator.sync_gradients}"
        )
    batch = move(next(iter(loader)), accelerator.device)
    if batch["input"].shape[1] != 15:
        raise RuntimeError("preflight requires 8 context + 7 target")
    local_info = {
        "rank": rank,
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "first_scene": str(batch.get("scene_name", "unknown")),
        "first_batch_hash": batch_hash(batch),
        "trainable_manifest": hash_names(base_model, True)[0],
        "trainable_value_hash_before": value_hash(base_model, True),
        "trainable_tensor_count": hash_names(base_model, True)[2],
        "trainable_numel": hash_names(base_model, True)[3],
    }
    gathered = gather_objects(local_info)
    if len({item["trainable_manifest"] for item in gathered}) != 1:
        raise RuntimeError("trainable manifest differs across ranks")
    initial_value_hashes = gather_objects(value_hash(base_model, True))
    if len(set(initial_value_hashes)) != 1:
        raise RuntimeError(f"initial trainable parameter values differ across ranks: {initial_value_hashes}")
    if len({item["first_batch_hash"] for item in gathered}) != len(gathered):
        raise RuntimeError("DDP first samples are not different")
    records = []
    for local_step in range(1, args.steps + 1):
        effective_step = 960 + local_step
        set_training_schedule(base_model, opt, effective_step)
        optimizer.zero_grad(set_to_none=True)
        output_value = model(batch, compute_quality_metrics=False)
        loss = output_value["loss"]
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite loss")
        accelerator.backward(loss)
        if torch.cuda.is_available():
            # Complete DDP's asynchronous reducer work before diagnostics.
            torch.cuda.synchronize()
        raw_gradient_range, raw_gradient_range_name = distributed_max_range(
            base_model, gradients=True
        )
        pre_clip = float(accelerator.clip_grad_norm_(model.parameters(), 1.0))
        if not np.isfinite(pre_clip):
            raise RuntimeError("nonfinite gradient norm")
        local_gradient_hash, missing_gradients = gradient_hash(base_model, True)
        local_missing_gradient_names = missing_gradient_names(base_model, True)
        gradient_hashes = gather_objects(local_gradient_hash)
        missing_gradient_counts = gather_objects(missing_gradients)
        if len(set(gradient_hashes)) != 1:
            gradient_range, gradient_range_name = distributed_max_range(
                base_model, gradients=True
            )
            raise RuntimeError(
                "post-backward trainable gradient hash mismatch at local step "
                f"{local_step}: {gradient_hashes}; missing={missing_gradient_counts}; "
                f"raw_max_range={raw_gradient_range} raw_name={raw_gradient_range_name}; "
                f"clipped_max_range={gradient_range} clipped_name={gradient_range_name}; "
                f"missing_names={local_missing_gradient_names}"
            )
        optimizer.step()
        scheduler.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in base_model.parameters()):
            raise RuntimeError("nonfinite parameter")
        current_hash = value_hash(base_model, True)
        hashes = gather_objects(current_hash)
        if len(set(hashes)) != 1:
            parameter_range, parameter_range_name = distributed_max_range(base_model)
            raise RuntimeError(
                f"post-step trainable parameter hash mismatch at local step {local_step}: "
                f"{hashes}; max_range={parameter_range} name={parameter_range_name}"
            )
        records.append({
            "stage_local_step": local_step,
            "effective_optimizer_step": effective_step,
            "loss": float(loss.detach()),
            "pre_clip_grad_norm": pre_clip,
            "gradient_hash": local_gradient_hash,
            "missing_gradients": missing_gradients,
            "missing_gradient_names": local_missing_gradient_names,
            "parameter_hash": current_hash,
        })
    checkpoint_path = output / "preflight_trainable.safetensors"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_file({
            name: parameter.detach().cpu()
            for name, parameter in base_model.named_parameters()
            if parameter.requires_grad
        }, str(checkpoint_path))
    accelerator.wait_for_everyone()
    saved = load_file(str(checkpoint_path), device="cpu")
    expected = {name for name, parameter in base_model.named_parameters() if parameter.requires_grad}
    if set(saved) != expected:
        raise RuntimeError("strict trainable save/reload key mismatch")
    before = {name: parameter.detach().cpu().clone() for name, parameter in base_model.named_parameters() if parameter.requires_grad}
    base_model.load_state_dict(saved, strict=False)
    restore_diff = max(
        float((parameter.detach().cpu() - before[name]).abs().max())
        for name, parameter in base_model.named_parameters()
        if name in before
    )
    report = {
        "config": args.config,
        "world_size": accelerator.num_processes,
        "steps": args.steps,
        "first_new_optimizer_step": 1,
        "final_effective_step": 960 + args.steps,
        "rank_info": gathered,
        "initial_trainable_value_hashes": initial_value_hashes,
        "trainability": categories,
        "optimizer_groups": [
            {"name": group.get("name"), "lr": group["lr"], "weight_decay": group["weight_decay"], "count": len(group["params"])}
            for group in optimizer.param_groups
        ],
        "dino_optimizer_entries": sum(
            1
            for group in optimizer.param_groups
            for parameter in group["params"]
            if id(parameter) in {
                id(p)
                for p in getattr(
                    base_model.token_eru_dino_encoder.dino_extractor,
                    "_dino_model",
                    torch.nn.Module(),
                ).parameters()
            }
        ),
        "dino_in_checkpoint": any(key.startswith("_dino_model") for key in saved),
        "target_image_to_dino": False,
        "p_u_used_in_formal_eval": False,
        "metric_cluster_used_in_formal_eval": False,
        "strict_save_reload": restore_diff == 0.0,
        "strict_save_reload_max_diff": restore_diff,
        "records": records,
        "formal_training_started": False,
        "formal_evaluation_started": False,
    }
    if accelerator.is_main_process:
        (output / "preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
