"""Single-card/DDP8 TokenGS-ERU preflight (no formal training)."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
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
from tokengs.train import load_model_checkpoint, setup_optimizer  # noqa: E402


CKPT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_"
    "w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors"
)
CFG = "semantic_v6_absolute_units_true_shared_token_eru1_ddp8"


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _full_trainable_hash(model) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(repr(tuple(parameter.shape)).encode())
        tensor = parameter.detach().cpu().contiguous()
        digest.update(tensor.numpy().tobytes())
        count += parameter.numel()
    digest.update(str(count).encode())
    return digest.hexdigest()


def _gather_objects(value):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        values = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(values, value)
        return values
    return [value]


def _assert_finite(model, output):
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter.detach()).all():
            raise FloatingPointError(f"non-finite parameter: {name}")
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite gradient: {name}")
    for name, value in output.items():
        if (
            name not in {"ssim", "lpips"}
            and torch.is_tensor(value)
            and not torch.isfinite(value).all()
        ):
            raise FloatingPointError(f"non-finite output: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()
    output_dir = ROOT / args.workspace
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty preflight directory: {output_dir}")
    if int(args.steps) < 1 or int(args.steps) > 3:
        raise ValueError("preflight is limited to 1-3 optimizer steps")
    output_dir.mkdir(parents=True, exist_ok=True)

    opt = dataclasses.replace(config_defaults[CFG])
    opt.resume = str(CKPT)
    opt.workspace = str(output_dir)
    opt.num_workers = 0
    opt.batch_size = 1
    opt.eval_before_training = False
    opt.use_wandb = False
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    loader, _, _, _ = get_multi_dataloader(opt, accelerator)
    model = model_registry[opt.model_type](opt)
    load_model_checkpoint(opt, model, accelerator, 0)
    copy_report = {"decoder_blocks": 0, "unit_keys": 0}
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        copy_report = model.initialize_token_eru_from_reconstruction()
    optimizer = setup_optimizer(opt, model, accelerator, 0)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    unwrapped = accelerator.unwrap_model(model)
    batch = _move(next(iter(loader)), accelerator.device)
    if batch["input"].shape[1] != 15 or batch["instance_label_output"].shape[1] != 7:
        raise RuntimeError("ERU preflight did not receive the formal 8+7 batch")

    samples = _gather_objects(
        {
            "rank": accelerator.process_index,
            "scene": str(batch["scene_name"][0]),
            "frame_ids": batch["frame_ids"][0].detach().cpu().tolist(),
        }
    )
    if accelerator.num_processes == 8 and len(
        {json.dumps(item, sort_keys=True) for item in samples}
    ) != 8:
        raise RuntimeError(f"DDP8 sample duplication: {samples}")

    step_records = []
    for step in range(int(args.steps)):
        completed = step + 1
        unwrapped.set_token_eru_step(completed)
        unwrapped.tsh_instance_loss_weight_eff = 1.0
        unwrapped.tsh_unit_grad_eff = 1.0
        unwrapped.tsh_mbm_u2r_eff = 0.0
        unwrapped.teacher_lambda_eff = 0.0
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            output = model(batch, compute_quality_metrics=False)
        if not torch.isfinite(output["loss_instance_group"]):
            raise FloatingPointError("non-finite instance loss")
        accelerator.backward(output["loss"])
        _assert_finite(unwrapped, output)
        accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)
        optimizer.step()
        digest = _full_trainable_hash(unwrapped)
        hashes = _gather_objects(digest)
        if len(set(hashes)) != 1:
            raise RuntimeError(f"parameter hash mismatch at step {completed}: {hashes}")
        step_records.append(
            {
                "step": completed,
                "loss": float(output["loss"].detach()),
                "instance_loss": float(output["loss_instance_group"].detach()),
                "parameter_hash": digest,
                "gates": unwrapped.set_token_eru_step(completed),
                "finite": True,
            }
        )

    if accelerator.is_main_process:
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in unwrapped.state_dict().items()
        }
        save_file(state, str(output_dir / "preflight_model.safetensors"))
    accelerator.wait_for_everyone()
    strict_state = load_file(str(output_dir / "preflight_model.safetensors"), device="cpu")
    restore_opt = dataclasses.replace(
        opt,
        resume=str(output_dir / "preflight_model.safetensors"),
    )
    fresh = model_registry[restore_opt.model_type](restore_opt)
    load_model_checkpoint(restore_opt, fresh, accelerator, 0)
    if not getattr(fresh, "_token_eru_loaded_from_checkpoint", False):
        fresh.initialize_token_eru_from_reconstruction()
    restored_state = fresh.state_dict()
    required_prefixes = (
        "token_eru_decoder.",
        "token_eru_unit_formation.",
        "absolute_gs_head.",
        "tsh_instance_head.",
        "enc_dec_backbone.decoder_blocks.",
    )
    required_saved = {
        key: value
        for key, value in strict_state.items()
        if key.startswith(required_prefixes)
    }
    missing_required = [
        key for key in required_saved if key not in restored_state
    ]
    reload_diffs = [
        (
            restored_state[key].detach().cpu()
            - value.detach().cpu()
        ).float().abs().max().item()
        for key, value in required_saved.items()
        if key in restored_state
    ]
    reload_max_diff = max(reload_diffs) if reload_diffs else 0.0
    if missing_required or reload_max_diff != 0.0:
        raise RuntimeError(
            "preflight ERU strict reload failed: "
            f"missing={missing_required[:5]} max_diff={reload_max_diff}"
        )
    report = {
        "hostname": os.uname().nodename,
        "world_size": accelerator.num_processes,
        "rank_samples": samples,
        "copy_report": copy_report,
        "strict_24_50_324_source": {
            "absolute_gs_head": sum(k.startswith("absolute_gs_head.") for k in strict_state),
            "tsh_instance_head": sum(k.startswith("tsh_instance_head.") for k in strict_state),
            "decoder_tail": sum(k.startswith("enc_dec_backbone.decoder_blocks.") for k in strict_state),
            "pgsr": sum(k.startswith("tsh_slot_refine_head.") for k in strict_state),
        },
        "instance_group_scene_level_matching": bool(
            getattr(opt, "instance_group_scene_level_matching", False)
        ),
        "matching_mode": (
            "scene_level"
            if bool(getattr(opt, "instance_group_scene_level_matching", False))
            else "per_view"
        ),
        "hungarian_calls_per_scene": (
            1
            if bool(getattr(opt, "instance_group_scene_level_matching", False))
            else 7
        ),
        "assignment_reused_target_views": (
            7
            if bool(getattr(opt, "instance_group_scene_level_matching", False))
            else 0
        ),
        "teacher_called": False,
        "u_to_r_old_path": 0.0,
        "unit_multiplier_old_path": 0.0,
        "step_records": step_records,
        "strict_reload": True,
        "strict_reload_checked_prefixes": list(required_prefixes),
        "strict_reload_max_diff": reload_max_diff,
    }
    if accelerator.is_main_process:
        (output_dir / "preflight.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
