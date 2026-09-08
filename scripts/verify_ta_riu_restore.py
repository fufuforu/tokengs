"""Verify TA-RIU fixed-batch checkpoint/cache restore without retraining."""
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
    CKPT,
    V13,
    make_accelerator,
    make_opt,
    load_model,
    model_hash,
    move_to_device,
    set_seed,
    set_ta_gates,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    workspace = ROOT / args.workspace
    state = torch.load(workspace / "step_200.pt", map_location="cpu", weights_only=False)
    cache = torch.load(workspace / "cache_step_200.pt", map_location="cpu", weights_only=False)
    cpu_batch = torch.load(V13 / "fixed_batch_cpu.pt", map_location="cpu", weights_only=False)
    opt = make_opt(CFG)
    acc = make_accelerator(opt)
    set_seed(1729)
    model_opt, model = load_model(CFG, acc, train=False)
    model = model.to(acc.device)
    base = model
    base.load_state_dict(state["trainable_state"], strict=False)
    set_ta_gates(base, 1.0)
    model.eval()
    batch = move_to_device(cpu_batch, acc.device)
    with torch.no_grad(), acc.autocast():
        output = model(batch, compute_quality_metrics=False)
    diffs = {
        key: float((output[key].detach().cpu().float() - value.float()).abs().max())
        for key, value in cache.items()
        if key in output
    }
    result = {
        "parameter_hash_match": model_hash(base) == state["parameter_hash"],
        "cache_max_diffs": diffs,
        "match": all(value <= 1e-5 for value in diffs.values()),
    }
    (workspace / "independent_restore_check_v2.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["parameter_hash_match"] or not result["match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
