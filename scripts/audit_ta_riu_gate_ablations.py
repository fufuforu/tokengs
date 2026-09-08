"""Read-only TA-RIU gate and memory ablations over saved fixed-batch states."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_ta_riu_fixed_batch import (
    CFG,
    MILESTONES,
    V13,
    evaluate,
    load_model,
    make_accelerator,
    make_opt,
    move_to_device,
    set_seed,
)


def set_gates(base, shared, geo, app, memory="normal"):
    base.ta_riu_eval_gate_override = float(shared)
    base.ta_riu_eval_geo_gate_override = float(geo)
    base.ta_riu_eval_app_gate_override = float(app)
    base.ta_riu_memory_mode = memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-workspace", required=True)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    source = ROOT / args.source_workspace
    out = ROOT / args.workspace
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing non-empty workspace: {out}")
    out.mkdir(parents=True, exist_ok=True)
    cpu_batch = torch.load(V13 / "fixed_batch_cpu.pt", map_location="cpu", weights_only=False)
    opt = make_opt(CFG)
    acc = make_accelerator(opt)
    set_seed(1729)
    model_opt, model = load_model(CFG, acc, train=False)
    model = model.to(acc.device)
    base = model
    batch = move_to_device(cpu_batch, acc.device)
    definitions = {
        "full": (1.0, 1.0, 1.0, "normal"),
        "g_shared_0": (0.0, 1.0, 1.0, "normal"),
        "g_geo_0": (1.0, 0.0, 1.0, "normal"),
        "g_app_0": (1.0, 1.0, 0.0, "normal"),
        "memory_zero": (1.0, 1.0, 1.0, "zero"),
        "memory_shuffle": (1.0, 1.0, 1.0, "shuffle"),
    }
    report = {"source_workspace": str(source), "milestones": {}}
    for step in MILESTONES:
        state = torch.load(source / f"step_{step:03d}.pt", map_location="cpu", weights_only=False)
        base.load_state_dict(state["trainable_state"], strict=False)
        report["milestones"][f"step{step}"] = {}
        for name, (shared, geo, app, memory) in definitions.items():
            set_gates(base, shared, geo, app, memory)
            model.eval()
            with torch.no_grad(), acc.autocast():
                output = model(batch, compute_quality_metrics=False)
            report["milestones"][f"step{step}"][name] = evaluate(output, batch, opt)
    report["all_finite"] = True
    (out / "gate_ablation_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
