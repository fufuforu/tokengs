"""Read-only residual and loss-subcomponent audit for saved TA-RIU states."""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

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
    set_ta_gates,
    set_training_state,
    evaluate,
)


STEPS = (0, 1, 5, 25, 50, 100, 200)


def stats(value):
    x = value.detach().float().abs().cpu().numpy().reshape(-1)
    return {
        "mean_abs": float(x.mean()),
        "std_abs": float(x.std()),
        "p50_abs": float(np.percentile(x, 50)),
        "p90_abs": float(np.percentile(x, 90)),
        "p99_abs": float(np.percentile(x, 99)),
        "max_abs": float(x.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    out = ROOT / args.workspace
    batch = torch.load(V13 / "fixed_batch_cpu.pt", map_location="cpu", weights_only=False)
    acc = make_accelerator(make_opt(CFG))
    batch = move_to_device(batch, acc.device)
    eval_batch = torch.load(V13 / "fixed_batch_cpu.pt", map_location="cpu", weights_only=False)
    set_seed(1729)
    opt, model = load_model(CFG, acc, train=True)
    model = model.to(acc.device)
    base = model
    report = {}
    reference_eval = None
    for step in STEPS:
        state = torch.load(out / f"step_{step:03d}.pt", map_location="cpu", weights_only=False)
        base.load_state_dict(state["trainable_state"], strict=False)
        set_training_state(base, max(0, step - 1))
        if step == 0:
            set_ta_gates(base, 0.0)
        else:
            gate = min(1.0, float(step) / 25.0)
            set_ta_gates(base, gate)
            base.ta_riu_gate_eff = gate
            base.ta_riu_geo_gate_eff = gate
            base.ta_riu_app_gate_eff = gate
        model.train()
        with torch.no_grad(), acc.autocast():
            output = model(batch, compute_quality_metrics=False)
        eval_output = {
            key: value.detach().cpu()
            for key, value in output.items()
            if torch.is_tensor(value)
        }
        actual_gate_metrics = evaluate(
            eval_output, eval_batch, opt, reference=reference_eval
        )
        if step == 0:
            reference_eval = eval_output
        geo = output["ta_riu_geo_raw"]
        app = output["ta_riu_app_raw"]
        delta = output["ta_riu_delta"]
        base_gs = output["ta_riu_base_gaussians"].float()
        joint_gs = output["ta_riu_joint_gaussians"].float()
        gs_delta = joint_gs - base_gs
        loss_keys = (
            "loss_instance_group", "tsh_loss_instance_group_mask", "tsh_loss_instance_group_dice",
            "tsh_loss_instance_group_void", "tsh_loss_instance_group_unmatched",
            "tsh_loss_instance_group_ce", "tsh_loss_instance_group_entropy",
        )
        losses = {
            key: float(output[key].detach())
            for key in loss_keys if key in output and torch.isfinite(output[key]).all()
        }
        report[f"step{step}"] = {
            "gate": float(step == 0 and 0.0 or min(1.0, step / 25.0)),
            "loss_components": losses,
            "metrics_at_actual_gate": actual_gate_metrics,
            "z_delta": stats(delta),
            "geo_raw": stats(geo),
            "appearance_raw": stats(app),
            "gs_delta": stats(gs_delta),
            "geo_field_delta": {
                "xyz": stats(gs_delta[..., 0:3]),
                "scale": stats(gs_delta[..., 4:7]),
                "rotation_quaternion": stats(gs_delta[..., 7:11]),
                "opacity": stats(gs_delta[..., 3:4]),
            },
            "appearance_field_delta": {"rgb": stats(gs_delta[..., 11:14])},
            # SSIM/LPIPS are intentionally NaN when compute_quality_metrics
            # is false; finite-ness here covers the audited losses and TA
            # tensors rather than those disabled metrics.
            "all_finite": all(
                torch.isfinite(output[key]).all()
                for key in (
                    "loss_instance_group", "ta_riu_q_abs_base", "ta_riu_z_shared",
                    "ta_riu_delta", "ta_riu_geo_raw", "ta_riu_app_raw",
                    "ta_riu_base_gaussians", "ta_riu_joint_gaussians",
                )
                if key in output
            ),
        }
    (out / "saved_state_diagnostics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
