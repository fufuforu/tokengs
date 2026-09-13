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
from tokengs.train import load_model_checkpoint, setup_optimizer  # noqa: E402

SOURCE = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_"
    "scene_hungarian_resume200_to500_persist_ddp8/checkpoints/"
    "model_step_000500.safetensors"
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
    parser.add_argument("--output-dir", default="workspace/token_eru_dino_metric_preflight_v1")
    args = parser.parse_args()
    if not 1 <= args.steps <= 3:
        raise ValueError("preflight is limited to 1..3 steps")
    output = ROOT / args.output_dir
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty preflight directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cfg = "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_treatment_resume500_to700_ddp8"
    opt = dataclasses.replace(config_defaults[cfg])
    opt.resume = str(SOURCE); opt.workspace = str(output); opt.num_workers = 0
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
        step = 501 + index
        unwrapped.set_token_eru_step(step); unwrapped.set_token_eru_dino_metric_step(step)
        unwrapped.tsh_instance_loss_weight_eff = 1.0; unwrapped.tsh_unit_grad_eff = 1.0
        unwrapped.teacher_lambda_eff = 0.0
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            result = model(batch, compute_quality_metrics=False)
        if not torch.isfinite(result["loss"]):
            raise FloatingPointError(f"non-finite preflight loss at {step}")
        accelerator.backward(result["loss"])
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        local_hash, trainable_numel = digest(unwrapped)
        hashes = gather(local_hash)
        if len(set(hashes)) != 1:
            raise RuntimeError(f"trainable hash mismatch at {step}: {hashes}")
        dino_model = unwrapped.token_eru_dino_encoder.dino_extractor.__dict__.get("_dino_model")
        if dino_model is not None and any(parameter.requires_grad for parameter in dino_model.parameters()):
            raise RuntimeError("DINO backbone became trainable")
        records.append({"step": step, "loss": float(result["loss"].detach()), "hash": local_hash, "trainable_numel": trainable_numel})
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
    report = {"ddp": args.ddp, "world_size": accelerator.num_processes, "records": records, "formal_training_started": False, "formal_evaluation_started": False, "strict_reload": True, "strict_reload_max_diff": reload_max_diff, "preflight_checkpoint": str(checkpoint_path), "dino_in_state_dict": any("_dino_model" in key for key in reloaded_state)}
    (output / "preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
