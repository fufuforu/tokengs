"""Bounded single-card/DDP8 preflight for JointFormation.

This file intentionally stops after three optimizer updates.  It is not a
formal-training launcher.
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
    configure_joint_formation_trainability,
    load_model_checkpoint,
    write_token_eru_joint_parent_reference,
    _joint_formation_prepare_audit,
    _load_cached_gsplat_extension,
    setup_optimizer,
)

_load_cached_gsplat_extension()

SOURCE = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_"
    "scene_hungarian_resume200_to500_persist_ddp8/checkpoints/"
    "model_step_000500.safetensors"
)
SOURCE_SHA256 = "cca23e2d24af7a374be6a62f2cf4026645afdf0df49686722cb5277c9e479209"
CONFIG = "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_v1_ddp8"


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


def parameter_hash(model, trainable_only=False):
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if trainable_only and not parameter.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
        count += 1
    if count == 0:
        raise RuntimeError("empty parameter hash")
    return digest.hexdigest(), count


def gradient_norms(model):
    prefixes = {
        "reconstruction_query": ("gs_tokens",),
        "reconstruction_decoder": ("enc_dec_backbone.decoder_blocks.",),
        "reconstruction_unit": ("absolute_gs_head.tok_", "absolute_gs_head.unit_"),
        "absolute_gs_head": ("absolute_gs_head.slot_emb", "absolute_gs_head.gs_decoder.", "absolute_gs_head.center_mlp."),
        "understanding_decoder": ("token_eru_decoder.understanding_decoder_blocks.",),
        "understanding_unit": ("token_eru_unit_formation.",),
        "pair_adapters": ("token_eru_decoder.reconstruction_to_understanding.", "token_eru_decoder.understanding_to_reconstruction."),
        "instance_head": ("tsh_instance_head.",),
        "metric_head": ("token_eru_dino_encoder.unit_projector.", "token_eru_dino_fusion.", "token_eru_metric_head."),
    }
    output = {}
    for group, group_prefixes in prefixes.items():
        values = [
            parameter.grad.detach().float().square().sum()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and name.startswith(group_prefixes)
            and parameter.grad is not None
        ]
        output[group] = float(torch.stack(values).sum().sqrt()) if values else 0.0
    return output


def build_runtime(output: Path, ddp: bool):
    opt = dataclasses.replace(config_defaults[CONFIG])
    opt.resume = str(SOURCE)
    opt.workspace = str(output)
    opt.num_workers = 0
    opt.tsh_ddp8 = bool(ddp)
    opt.eval_before_training = False
    opt.use_wandb = False
    torch.manual_seed(int(opt.seed))
    random.seed(int(opt.seed))
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if torch.cuda.is_available():
        local_rank = int(
            os.environ.get(
                "LOCAL_RANK", getattr(accelerator, "local_process_index", 0)
            )
        )
        torch.cuda.set_device(local_rank)
        accelerator.print(
            f"[ddp-device] rank={getattr(accelerator, 'process_index', -1)} "
            f"local_rank={local_rank} current_device={torch.cuda.current_device()}"
        )
    model = model_registry[opt.model_type](opt)
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        model.initialize_token_eru_from_reconstruction()
    categories = configure_joint_formation_trainability(model, opt)
    write_token_eru_joint_parent_reference(opt, accelerator)
    loader, test_loader, _, _ = get_multi_dataloader(opt, accelerator)
    optimizer = setup_optimizer(opt, model, accelerator, 0)
    opt.joint_formation_ddp_manifest_audit = True
    _joint_formation_prepare_audit(
        opt, accelerator, model, optimizer, loader, test_loader
    )
    model, optimizer, loader, test_loader = accelerator.prepare(
        model, optimizer, loader, test_loader
    )
    return opt, accelerator, model, optimizer, loader, categories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output-dir", default="workspace/token_eru_dino_joint_formation_v1_preflight")
    args = parser.parse_args()
    if not 1 <= args.steps <= 3:
        raise ValueError("preflight is limited to 1..3 steps")
    if not SOURCE.is_file() or sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("ERU@500 source checkpoint is missing or SHA256 mismatched")
    output = (ROOT / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty preflight workspace: {output}")
    output.mkdir(parents=True, exist_ok=True)
    opt, accelerator, model, optimizer, loader, categories = build_runtime(output, args.ddp)
    unwrapped = accelerator.unwrap_model(model)
    batch = move(next(iter(loader)), accelerator.device)
    if batch["input"].shape[1] != 15 or batch["instance_label_output"].shape[1] != 7:
        raise RuntimeError("JointFormation preflight did not receive formal 8+7 data")
    first_samples = gather({
        "rank": int(accelerator.process_index),
        "scene": str(batch["scene_name"][0]),
    })
    if accelerator.num_processes == 8 and len({json.dumps(value, sort_keys=True) for value in first_samples}) != 8:
        raise RuntimeError(f"DDP8 first samples are duplicated: {first_samples}")
    records = []
    for step in range(1, args.steps + 1):
        unwrapped.set_token_eru_step(step)
        unwrapped.set_token_eru_dino_metric_step(step)
        unwrapped.tsh_instance_loss_weight_eff = 1.0
        unwrapped.tsh_unit_grad_eff = 1.0
        unwrapped.teacher_lambda_eff = 0.0
        optimizer.zero_grad(set_to_none=True)
        result = model(batch, compute_quality_metrics=False)
        loss = result["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        accelerator.backward(loss)
        norms = gradient_norms(unwrapped)
        if any(not torch.isfinite(torch.tensor(value)) for value in norms.values()):
            raise FloatingPointError(f"non-finite gradient at step {step}: {norms}")
        pre = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in unwrapped.parameters()):
            raise FloatingPointError(f"non-finite parameter at step {step}")
        local_hash, trainable_numel = parameter_hash(unwrapped, True)
        if len(set(gather(local_hash))) != 1:
            raise RuntimeError(f"trainable parameter hash mismatch at step {step}")
        dino_model = unwrapped.token_eru_dino_encoder.dino_extractor.__dict__.get("_dino_model")
        if dino_model is not None and any(parameter.grad is not None for parameter in dino_model.parameters()):
            raise RuntimeError("DINO backbone received a gradient")
        optimizer_dino = sum(
            1 for group in optimizer.param_groups for parameter in group["params"]
            if dino_model is not None and id(parameter) in {id(item) for item in dino_model.parameters()}
        )
        if optimizer_dino:
            raise RuntimeError("DINO backbone entered optimizer")
        records.append({"step": step, "loss": float(loss.detach()), "pre_clip_grad_norm": pre, "gradient_norms": norms, "trainable_hash": local_hash, "trainable_numel": trainable_numel})
    checkpoint = output / "preflight_model.safetensors"
    if accelerator.is_main_process:
        save_file({key: value.detach().cpu().contiguous() for key, value in unwrapped.state_dict().items()}, str(checkpoint))
    accelerator.wait_for_everyone()
    reloaded = load_file(str(checkpoint), device="cpu")
    current_state = {
        key: value.detach().cpu()
        for key, value in unwrapped.state_dict().items()
    }
    if set(current_state) != set(reloaded):
        raise RuntimeError(
            "strict save/reload key mismatch: "
            f"missing={sorted(set(current_state) - set(reloaded))} "
            f"unexpected={sorted(set(reloaded) - set(current_state))}"
        )
    max_diff = 0.0
    for key, value in reloaded.items():
        max_diff = max(max_diff, float((current_state[key] - value).abs().max()))
    if max_diff != 0.0:
        raise RuntimeError(f"strict save/reload mismatch: {max_diff}")
    report = {
        "config": CONFIG,
        "ddp": bool(args.ddp),
        "world_size": int(accelerator.num_processes),
        "first_new_optimizer_step": 1,
        "final_preflight_step": args.steps,
        "source_checkpoint": str(SOURCE),
        "source_sha256": SOURCE_SHA256,
        "batches_skipped": 0,
        "first_samples": first_samples,
        "trainability": categories,
        "records": records,
        "strict_save_reload": True,
        "strict_save_reload_max_diff": max_diff,
        "dino_in_checkpoint": any("_dino_model" in key for key in reloaded),
        "dino_optimizer_entries": 0,
        "target_image_to_dino": False,
        "p_u_used_in_eval": False,
        "formal_training_started": False,
        "formal_evaluation_started": False,
    }
    if accelerator.is_main_process:
        (output / "preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
