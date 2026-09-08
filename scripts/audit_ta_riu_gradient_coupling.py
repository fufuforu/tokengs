"""Read-only SharedUnitMixer gradient cosine audit for saved TA-RIU states."""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_ta_riu_fixed_batch import (
    CFG,
    V13,
    load_model,
    make_accelerator,
    make_opt,
    move_to_device,
    set_seed,
    set_training_state,
)


def vector(model, prefix):
    parts = []
    for name, parameter in model.named_parameters():
        if name.startswith(prefix):
            if parameter.grad is None:
                parts.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1))
            else:
                parts.append(parameter.grad.detach().float().reshape(-1).clone())
    return torch.cat(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    out = ROOT / args.workspace
    batch = torch.load(V13 / "fixed_batch_cpu.pt", map_location="cpu", weights_only=False)
    acc = make_accelerator(make_opt(CFG))
    device = acc.device
    batch = move_to_device(batch, device)
    results = {}
    for step in (1, 100, 200):
        state = torch.load(out / f"step_{step:03d}.pt", map_location="cpu", weights_only=False)
        set_seed(1729)
        opt, model = load_model(CFG, acc, train=True)
        model = model.to(device)
        base = model
        base.load_state_dict(state["trainable_state"], strict=False)
        set_training_state(base, step - 1)
        model.train()
        grads = {}
        losses = {}
        for kind in ("instance", "rgb"):
            for parameter in model.parameters():
                parameter.grad = None
            with acc.autocast():
                output = model(batch, compute_quality_metrics=False)
            loss = output["loss_instance_group"] if kind == "instance" else output["loss_rgb"]
            assert torch.isfinite(loss).all() and loss.requires_grad
            acc.backward(loss)
            grads[kind] = vector(base, "ta_riu_shared_mixer.")
            losses[kind] = float(loss.detach())
        dot = float(torch.dot(grads["instance"], grads["rgb"]))
        ni = float(grads["instance"].norm())
        nr = float(grads["rgb"].norm())
        results[f"step{step}"] = {
            "instance_loss": losses["instance"],
            "rgb_loss": losses["rgb"],
            "instance_shared_mixer_grad_norm": ni,
            "rgb_shared_mixer_grad_norm": nr,
            "gradient_dot": dot,
            "gradient_cosine": dot / max(ni * nr, 1e-30),
            "all_finite": bool(torch.isfinite(grads["instance"]).all() and torch.isfinite(grads["rgb"]).all()),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (out / "gradient_coupling_audit.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
