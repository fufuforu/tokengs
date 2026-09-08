"""Read-only fresh-forward gradient audit for TA-RIU short-250 checkpoints.

This uses the production DataLoader construction and performs independent
forward/backward passes for instance and RGB losses.  It never calls an
optimizer and writes only to the explicitly supplied diagnostic directory.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.options import config_defaults
from tokengs.train import load_model_checkpoint


CFG = "semantic_v6_absolute_units_true_shared_ta_riu_v1_short250_retry_v2_ddp8"
MILESTONES = (1, 25, 100, 250)
SOURCE_CKPT = ROOT / "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors"
CKPT_ROOT = ROOT / "workspace/semantic_v6_absolute_units_true_shared_ta_riu_v1_short250_retry_v2_ddp8/checkpoints"


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def prefix_norms(model, prefixes):
    result = {}
    for prefix in prefixes:
        sq = 0.0
        for name, parameter in model.named_parameters():
            if name.startswith(prefix) and parameter.grad is not None:
                sq += float(parameter.grad.detach().float().square().sum())
        result[prefix] = sq ** 0.5
    return result


def prefix_vector(model, prefix):
    parts = []
    for name, parameter in model.named_parameters():
        if name.startswith(prefix):
            if parameter.grad is None:
                parts.append(torch.zeros(parameter.numel(), device=parameter.device, dtype=torch.float32))
            else:
                parts.append(parameter.grad.detach().float().reshape(-1).clone())
    return torch.cat(parts) if parts else torch.empty(0, device=next(model.parameters()).device)


def set_effective_state(model, optimizer_step: int):
    gate = min(1.0, float(optimizer_step) / 25.0)
    model.ta_riu_gate_eff = gate
    model.ta_riu_geo_gate_eff = gate
    model.ta_riu_app_gate_eff = gate
    model.tsh_instance_loss_weight_eff = 1.0
    model.tsh_unit_grad_eff = 1.0
    model.tsh_mbm_u2r_eff = 0.0
    model.teacher_lambda_eff = 0.0
    return gate


def batch_signature(batch):
    result = {}
    for key in ("images_input", "images_output", "instance_label_output"):
        if key in batch and torch.is_tensor(batch[key]):
            result[key] = {"shape": list(batch[key].shape), "sha256": tensor_hash(batch[key])}
    for key in ("scene_name", "frame_ids", "target_frame_ids", "input_frame_ids"):
        if key in batch:
            value = batch[key]
            result[key] = value.tolist() if torch.is_tensor(value) else value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = ROOT / args.out
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    opt = dataclasses.replace(config_defaults[CFG])
    opt.resume = str(CKPT_ROOT / "model_step_000250.safetensors")
    opt.num_workers = 0
    opt.batch_size = 1
    opt.evaluating = False
    opt.eval_before_training = False
    opt.use_wandb = False
    opt.prompt_overfit_single_batch = False

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    # Exactly the same dataset/dataloader factory used by train.py.
    train_loader, _, _, _ = get_multi_dataloader(opt, accelerator)
    batch_cpu = next(iter(train_loader))
    batch_sig = batch_signature(batch_cpu)
    batch = move_to_device(batch_cpu, accelerator.device)

    prefixes = (
        "tsh_instance_head.",
        "ta_riu_shared_mixer.",
        "ta_riu_geometry_head.",
        "ta_riu_appearance_head.",
        "absolute_gs_head.",
        "enc_dec_backbone.decoder_blocks.",
    )
    results = {
        "config": CFG,
        "production_dataloader": "tokengs.train.get_multi_dataloader -> tokengs.data.get_multi_dataloader",
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(accelerator.device),
            "device_count": torch.cuda.device_count(),
        },
        "batch": batch_sig,
        "milestones": {},
    }

    for step in MILESTONES:
        # Step 1 has no persisted post-step model in the formal probe.  The
        # exact pre-update state is the Both@1420 source plus freshly created
        # TA-RIU modules, with the production step-1 gate.
        ckpt = SOURCE_CKPT if step == 1 else CKPT_ROOT / f"model_step_{step:06d}.safetensors"
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        opt.resume = str(ckpt)
        model = model_registry[opt.model_type](opt)
        load_model_checkpoint(opt, model, accelerator, 0)
        model = model.to(accelerator.device)
        model.train()
        gate = set_effective_state(model, step)
        grads = {}
        losses = {}
        shared_vectors = {}
        for kind in ("instance", "rgb"):
            for parameter in model.parameters():
                parameter.grad = None
            with accelerator.autocast():
                output = model(batch, compute_quality_metrics=False)
            loss = output["loss_instance_group"] if kind == "instance" else output["loss_rgb"]
            if not loss.requires_grad or not torch.isfinite(loss).all():
                raise FloatingPointError(f"invalid {kind} loss at step {step}")
            accelerator.backward(loss)
            grads[kind] = prefix_norms(model, prefixes)
            shared_vectors[kind] = prefix_vector(model, "ta_riu_shared_mixer.")
            losses[kind] = float(loss.detach().cpu())
            if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
                raise FloatingPointError(f"non-finite {kind} gradients at step {step}")
        # The vectors above were detached immediately after their own fresh
        # backward pass; no graph or live loss/output is reused here.
        vi = shared_vectors["instance"]
        vr = shared_vectors["rgb"]
        ni, nr = float(vi.norm()), float(vr.norm())
        dot = float(torch.dot(vi, vr))
        results["milestones"][f"step{step}"] = {
            "gate": gate,
            "losses": losses,
            "instance_grad_norms": grads["instance"],
            "rgb_grad_norms": grads["rgb"],
            "shared_mixer_instance_grad_norm": ni,
            "shared_mixer_rgb_grad_norm": nr,
            "shared_mixer_gradient_cosine": dot / max(ni * nr, 1e-30),
            "all_finite": True,
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (out / "formal_gradient_audit.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
