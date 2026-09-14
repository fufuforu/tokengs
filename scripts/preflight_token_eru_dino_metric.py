"""Bounded single-card/DDP8 preflight for the DINO metric branch."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
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


def _load_cached_gsplat_extension() -> None:
    """Use the already-built cluster extension when JIT compilation races.

    The preflight is allowed to reuse the existing CUDA extension cache.  On
    this cluster eight ranks can concurrently regenerate gsplat's ninja build
    graph; loading the validated shared object first keeps this diagnostic
    independent of that race and does not alter model semantics.
    """
    so_path = os.environ.get("GSPLAT_PRECOMPILED_SO")
    if not so_path or not os.path.isfile(so_path):
        return
    if "gsplat_cuda" in sys.modules:
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

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import (  # noqa: E402
    load_model_checkpoint,
    save_token_eru_intra_epoch_checkpoint_synchronized,
    save_token_eru_model_only_checkpoint_synchronized,
    setup_optimizer,
)

SOURCE = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_"
    "scene_hungarian_resume200_to500_persist_ddp8/checkpoints/"
    "model_step_000500.safetensors"
)
EXPECTED_SOURCE_SHA256 = (
    "cca23e2d24af7a374be6a62f2cf4026645afdf0df49686722cb5277c9e479209"
)


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value


def digest(model):
    h = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            t = parameter.detach().cpu().contiguous()
            h.update(name.encode()); h.update(t.numpy().tobytes()); count += t.numel()
    h.update(str(count).encode())
    return h.hexdigest(), count


def gradient_norms(model):
    """Return finite L2 norms for the required trainable metric groups."""
    groups = {
        "tsh_instance_head": ("tsh_instance_head.",),
        "understanding_decoder": (
            "token_eru_decoder.understanding_decoder_blocks.",
        ),
        "understanding_unit_formation": ("token_eru_unit_formation.",),
        "pair_adapters": (
            "token_eru_decoder.reconstruction_to_understanding.",
            "token_eru_decoder.understanding_to_reconstruction.",
        ),
        "dino_metric": (
            "token_eru_dino_encoder.unit_projector.",
            "token_eru_dino_fusion.",
            "token_eru_metric_head.",
        ),
    }
    norms = {}
    for group, prefixes in groups.items():
        squared = 0.0
        found = False
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or not name.startswith(prefixes):
                continue
            found = True
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"non-finite gradient in {name}")
            squared += float(parameter.grad.detach().float().pow(2).sum())
        if not found:
            raise RuntimeError(f"required preflight gradient group is absent: {group}")
        norms[group] = squared ** 0.5
        if norms[group] <= 0.0:
            raise RuntimeError(
                f"required preflight gradient group is zero at this step: {group}"
            )
    return norms


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gather(value):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        out = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(out, value)
        return out
    return [value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--checkpoint-smoke",
        action="store_true",
        help="exercise an isolated checkpoint write in the preflight workspace",
    )
    parser.add_argument("--output-dir", default="workspace/token_eru_dino_metric_preflight_v1")
    parser.add_argument(
        "--config",
        default="semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_short200_ddp8",
    )
    parser.add_argument(
        "--stage-m",
        action="store_true",
        help="run the independent parent-weight-only Stage-M preflight",
    )
    args = parser.parse_args()
    if not 1 <= args.steps <= 3:
        raise ValueError("preflight is limited to 1..3 steps")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source_sha256 = sha256_file(SOURCE)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"source checkpoint SHA256 mismatch: {source_sha256}")
    output = ROOT / args.output_dir
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty preflight directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cfg = args.config
    opt = dataclasses.replace(config_defaults[cfg])
    opt.resume = str(SOURCE); opt.workspace = str(output); opt.num_workers = 0
    if args.stage_m:
        opt.token_eru_dino_metric_stage_m = True
        opt.token_eru_dino_metric_stage_m_parent_step = 500
        opt.tsh_fork_continue_step = 0
        opt.seed = 42
    opt.tsh_ddp8 = bool(args.ddp); opt.eval_before_training = False; opt.use_wandb = False
    # Match tokengs.train's pre-model seed; all ranks must start fresh metric
    # modules from identical values before DDP synchronizes updates.
    torch.manual_seed(opt.seed)
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        # Match train.py for the True-Shared decoder-tail/ERU graph.  The
        # metric path contains intentionally gated branches at fork step 500.
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    loader, _, _, _ = get_multi_dataloader(opt, accelerator)
    model = model_registry[opt.model_type](opt)
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        model.initialize_token_eru_from_reconstruction()
    optimizer = setup_optimizer(opt, model, accelerator, 0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    unwrapped = accelerator.unwrap_model(model)
    batch = move(next(iter(loader)), accelerator.device)
    if batch["input"].shape[1] != 15 or batch["instance_label_output"].shape[1] != 7:
        raise RuntimeError("DINO metric preflight did not receive 8+7 input")
    ranks = gather({"rank": accelerator.process_index, "scene": str(batch["scene_name"][0])})
    if accelerator.num_processes == 8 and len({json.dumps(x, sort_keys=True) for x in ranks}) != 8:
        raise RuntimeError(f"DDP8 first samples are duplicated: {ranks}")
    records = []
    for index in range(args.steps):
        stage_step = 1 + index if args.stage_m else 501 + index
        schedule_step = (
            int(getattr(opt, "token_eru_dino_metric_stage_m_parent_step", 500))
            + stage_step
            if args.stage_m
            else stage_step
        )
        unwrapped.set_token_eru_step(schedule_step)
        unwrapped.set_token_eru_dino_metric_step(schedule_step)
        unwrapped.tsh_instance_loss_weight_eff = 1.0; unwrapped.tsh_unit_grad_eff = 1.0
        unwrapped.teacher_lambda_eff = 0.0
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            result = model(batch, compute_quality_metrics=False)
        if not torch.isfinite(result["loss"]):
            raise FloatingPointError(f"non-finite preflight loss at {stage_step}")
        accelerator.backward(result["loss"])
        grad_norms = gradient_norms(unwrapped)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if args.checkpoint_smoke:
            if args.stage_m:
                save_token_eru_intra_epoch_checkpoint_synchronized(
                    opt, accelerator, model, optimizer, scheduler,
                    epoch=0, completed_step=stage_step
                )
            elif stage_step == 502:
                save_token_eru_model_only_checkpoint_synchronized(
                    opt, accelerator, model, epoch=1, completed_step=stage_step
                )
            elif stage_step == 503:
                save_token_eru_intra_epoch_checkpoint_synchronized(
                    opt, accelerator, model, optimizer, scheduler,
                    epoch=1, completed_step=stage_step
                )
        local_hash, trainable_numel = digest(unwrapped)
        hashes = gather(local_hash)
        if len(set(hashes)) != 1:
            raise RuntimeError(f"trainable hash mismatch at {stage_step}: {hashes}")
        dino_model = unwrapped.token_eru_dino_encoder.dino_extractor.__dict__.get("_dino_model")
        if dino_model is not None and any(parameter.requires_grad for parameter in dino_model.parameters()):
            raise RuntimeError("DINO backbone became trainable")
        dino_param_ids = set()
        if dino_model is not None:
            dino_param_ids = {id(parameter) for parameter in dino_model.parameters()}
        optimizer_dino_params = sum(
            1
            for group in optimizer.param_groups
            for parameter in group["params"]
            if id(parameter) in dino_param_ids
        )
        if optimizer_dino_params != 0:
            raise RuntimeError(f"DINO backbone entered optimizer: {optimizer_dino_params}")
        records.append({
            "step": stage_step,
            "stage_optimizer_step": stage_step if args.stage_m else None,
            "effective_schedule_step": schedule_step,
            "loss": float(result["loss"].detach()),
            "hash": local_hash,
            "trainable_numel": trainable_numel,
            "gradient_norms": grad_norms,
        })
    # Strict runtime save/reload check.  The external DINO backbone is not a
    # registered state-dict member, so this file contains only ERU plus the
    # trainable metric modules.
    checkpoint_path = output / "preflight_model.safetensors"
    if accelerator.is_main_process:
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in unwrapped.state_dict().items()
        }
        save_file(state, str(checkpoint_path))
    accelerator.wait_for_everyone()
    reloaded_state = load_file(str(checkpoint_path), device="cpu")
    fresh_opt = dataclasses.replace(opt)
    fresh_opt.workspace = str(output / "strict_reload")
    fresh_opt.resume = str(checkpoint_path)
    torch.manual_seed(fresh_opt.seed)
    fresh = model_registry[fresh_opt.model_type](fresh_opt)

    # PromptTokenGS checkpoints intentionally omit frozen backbone tensors;
    # use the same strict fork loader as train.py so those tensors are
    # restored from backbone_resume while every saved ERU/metric key is
    # checked strictly.  A raw Module.load_state_dict would incorrectly
    # classify the omitted frozen namespace as a checkpoint failure.
    class _ReloadPrinter:
        @staticmethod
        def print(*values, **kwargs):
            del kwargs
            print(*values)

    load_model_checkpoint(fresh_opt, fresh, _ReloadPrinter(), 0)
    reload_max_diff = 0.0
    reload_max_key = None
    for key, value in reloaded_state.items():
        fresh_state = fresh.state_dict()
        if key not in fresh_state or fresh_state[key].shape != value.shape:
            raise RuntimeError(f"metric preflight strict reload missing/shape mismatch: {key}")
        key_diff = float((fresh_state[key].cpu() - value).abs().max())
        if key_diff > reload_max_diff:
            reload_max_diff = key_diff
            reload_max_key = key
    if reload_max_diff != 0.0:
        raise RuntimeError(f"metric preflight strict reload max diff={reload_max_diff} key={reload_max_key}")
    report = {
        "config": cfg,
        "source_checkpoint": str(SOURCE),
        "source_sha256": source_sha256,
        "ddp": args.ddp,
        "world_size": accelerator.num_processes,
        "first_new_optimizer_step": 1 if args.stage_m else 501,
        "stage_name": "eru_dino_metric_stage_m" if args.stage_m else None,
        "batches_skipped": 0,
        "records": records,
        "formal_training_started": False,
        "formal_evaluation_started": False,
        "strict_reload": True,
        "strict_reload_max_diff": reload_max_diff,
        "preflight_checkpoint": str(checkpoint_path),
        "dino_in_state_dict": any("_dino_model" in key for key in reloaded_state),
        "dino_optimizer_parameter_count": 0,
        "target_image_to_dino": False,
        "p_u_used_in_eval": False,
    }
    (output / "preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
