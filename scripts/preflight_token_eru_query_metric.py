"""Single-card/DDP8 bounded preflight for QueryMetric-Coupling-v1."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from accelerate.utils import DistributedDataParallelKwargs

from scripts import preflight_token_eru_unit_3d_anchor as base
from tokengs.options import config_defaults


CONFIG = (
    "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_local250_query_metric_v1_short200_ddp8"
)
PARENT = Path(
    "/space/mawb/tokengs/workspace/semantic_v6_absolute_units_true_shared_"
    "token_eru1_dino_metric_joint_formation_j2_ddp8/checkpoints/"
    "model_step_000250.safetensors"
)
PARENT_SHA = "61debbf5f55b76eea16012aea00fe19291309177bfdc6dd0420f6d1cb58d8cba"


def set_schedule(model, opt, local_step: int) -> None:
    effective = 960 + int(local_step)
    base.set_training_schedule(model, opt, effective)
    model.set_token_eru_query_metric_step(int(local_step))


def optimizer_manifest_hash(model, optimizer) -> str:
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    digest = hashlib.sha256()
    for index, group in enumerate(optimizer.param_groups):
        names = sorted(names_by_id[id(parameter)] for parameter in group["params"])
        digest.update(str(index).encode())
        digest.update(str(group.get("name", "")).encode())
        digest.update(str(float(group["lr"])).encode())
        digest.update(str(float(group["weight_decay"])).encode())
        for name in names:
            digest.update(name.encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()
    if args.steps not in (1, 3):
        raise ValueError("QMC preflight permits only 1 or 3 steps")
    if not PARENT.is_file() or base.sha256_file(PARENT) != PARENT_SHA:
        raise RuntimeError("QMC parent checkpoint is missing or SHA mismatched")
    output = args.output_dir.resolve()
    rank = int(os.environ.get("RANK", "0"))
    if output.exists() and any(output.iterdir()) and rank == 0:
        raise RuntimeError(f"refusing non-empty preflight workspace: {output}")
    output.mkdir(parents=True, exist_ok=True)

    opt = dataclasses.replace(config_defaults[CONFIG])
    opt.resume = str(PARENT)
    opt.workspace = str(output)
    opt.num_workers = 0
    opt.tsh_ddp8 = bool(args.ddp)
    opt.use_wandb = False
    opt.eval_before_training = False
    seed = 42 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        torch.cuda.manual_seed_all(seed)
    accelerator = base.Accelerator(
        mixed_precision="no",
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ] if args.ddp else None,
        dataloader_config=base.DataLoaderConfiguration(use_seedable_sampler=True),
    )
    model = base.model_registry[opt.model_type](opt).to(accelerator.device)
    base.load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        raise RuntimeError("QMC parent strict restore marker missing")
    base.configure_joint_formation_trainability(model, opt)
    optimizer = base.setup_optimizer(opt, model, accelerator, 0)
    loader, _, _, _ = base.get_multi_dataloader(opt, accelerator)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    base_model = accelerator.unwrap_model(model)
    if args.ddp and int(accelerator.num_processes) != 8:
        raise RuntimeError(
            f"QMC DDP preflight requires world_size=8, got {accelerator.num_processes}"
        )
    set_schedule(base_model, opt, 0)
    batch = base.move(next(iter(loader)), accelerator.device)
    if batch["input"].shape[1] != 15:
        raise RuntimeError("QMC preflight requires 8 context + 7 target")
    trainable_hash, _, trainable_count, trainable_numel = base.hash_names(
        base_model, True
    )
    expected_hash = base.gather_objects(trainable_hash)
    optimizer_hash = optimizer_manifest_hash(base_model, optimizer)
    optimizer_hashes = base.gather_objects(optimizer_hash)
    if len(set(optimizer_hashes)) != 1:
        raise RuntimeError(
            "QMC optimizer manifest differs across ranks: "
            f"{optimizer_hashes}"
        )
    initial_value_hashes = base.gather_objects(base.value_hash(base_model, True))
    if len(set(initial_value_hashes)) != 1:
        raise RuntimeError(
            "QMC initial trainable parameter values differ across ranks: "
            f"{initial_value_hashes}"
        )
    first_batch_hashes = base.gather_objects(base.batch_hash(batch))
    if len(set(expected_hash)) != 1:
        raise RuntimeError("QMC trainable manifest differs across ranks")
    if args.ddp and len(set(first_batch_hashes)) != len(first_batch_hashes):
        raise RuntimeError("QMC DDP first batches are not distinct")
    if trainable_count != 907 or trainable_numel != 364_382_513:
        raise RuntimeError(
            f"QMC trainability mismatch: {trainable_count}/{trainable_numel} "
            "expected 907/364382513"
        )

    records = []
    post_step_parameter_hashes = []
    for local_step in range(1, args.steps + 1):
        set_schedule(base_model, opt, local_step)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch, compute_quality_metrics=False)
        loss = out["loss"]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("QMC preflight loss is non-finite")
        accelerator.backward(loss)
        pre_clip = float(accelerator.clip_grad_norm_(model.parameters(), 1.0))
        if not np.isfinite(pre_clip):
            raise FloatingPointError("QMC preflight gradient is non-finite")
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in base_model.parameters()):
            raise FloatingPointError("QMC preflight parameter is non-finite")
        value_hash = base.value_hash(base_model, True)
        hashes = base.gather_objects(value_hash)
        if len(set(hashes)) != 1:
            raise RuntimeError(f"QMC post-step hash mismatch at step {local_step}")
        post_step_parameter_hashes.append(hashes)
        records.append(
            {
                "local_step": local_step,
                "effective_step": 960 + local_step,
                "loss": float(loss.detach()),
                "pre_clip_grad_norm": pre_clip,
                "parameter_hash": value_hash,
                "qmc_gate": float(
                    base_model.token_eru_query_metric_gate(local_step, opt)
                ),
            }
        )

    checkpoint_path = output / "preflight_model.safetensors"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        from safetensors.torch import save_file

        save_file(
            {
                name: parameter.detach().cpu()
                for name, parameter in base_model.named_parameters()
                if parameter.requires_grad
            },
            str(checkpoint_path),
        )
    accelerator.wait_for_everyone()
    from safetensors.torch import load_file

    restored = load_file(str(checkpoint_path), device="cpu")
    expected = {
        name for name, parameter in base_model.named_parameters() if parameter.requires_grad
    }
    if set(restored) != expected:
        raise RuntimeError("QMC strict trainable save/reload key mismatch")
    report = {
        "config": CONFIG,
        "parent": str(PARENT),
        "parent_sha256": PARENT_SHA,
        "world_size": int(accelerator.num_processes),
        "steps": args.steps,
        "first_new_optimizer_step": 1,
        "first_effective_step": 961,
        "trainable_tensor_count": trainable_count,
        "trainable_numel": trainable_numel,
        "trainable_manifest_hashes": expected_hash,
        "optimizer_manifest_hashes": optimizer_hashes,
        "initial_trainable_value_hashes": initial_value_hashes,
        "first_batch_hashes": first_batch_hashes,
        "post_step_parameter_hashes": post_step_parameter_hashes,
        "optimizer_groups": [
            {
                "name": group.get("name"),
                "lr": group["lr"],
                "weight_decay": group["weight_decay"],
                "tensor_count": len(group["params"]),
            }
            for group in optimizer.param_groups
        ],
        "dino_optimizer_entries": 0,
        "dino_checkpoint_keys": [
            key
            for key in restored
            if "_dino_model" in key.lower()
            or "dino_extractor" in key.lower()
        ],
        "target_image_to_dino": False,
        "p_u_used_in_formal_eval": False,
        "metric_cluster_used_in_formal_eval": False,
        "records": records,
        "strict_save_reload": True,
        "formal_training_started": False,
        "formal_evaluation_started": False,
    }
    if accelerator.is_main_process:
        (output / "preflight_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
