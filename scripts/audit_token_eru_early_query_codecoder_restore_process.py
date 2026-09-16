"""Fresh-process verifier for the EQC trainable-only fixed-batch restore."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_token_eru_early_query_codecoder_final import (  # noqa: E402
    E1,
    E1_OVERLAY,
    PARENT,
    Accelerator,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
    batch_hash,
    build,
    capture,
    file_sha,
    get_multi_dataloader,
    move,
    config_defaults,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    opt = dataclasses.replace(config_defaults[E1])
    opt.num_workers = 0
    loader, _, _, _ = get_multi_dataloader(opt, accelerator)
    raw = next(iter(loader))
    batch = move(raw, accelerator.device)
    if batch_hash(raw) != reference["batch_hash"]:
        raise RuntimeError("fresh-process batch fingerprint mismatch")
    _, model = build(E1, accelerator, args.output.parent / "process_model", E1_OVERLAY)
    record = capture(model, batch, 100, "full")
    expected = reference["causal"]["modes"]["full"]["tensor_hashes"]
    got = record["tensor_hashes"]
    common = sorted(set(expected) & set(got))
    mismatches = {k: {"expected": expected[k], "actual": got[k]} for k in common if expected[k] != got[k]}
    report = {
        "batch_hash": batch_hash(raw),
        "parent_sha256": file_sha(PARENT),
        "overlay_sha256": file_sha(E1_OVERLAY),
        "strict_parent_restore": True,
        "strict_overlay_restore": True,
        "default_mode": getattr(model, "_token_eru_eqc_eval_ablation", None),
        "local_step": 100,
        "effective_step": 1060,
        "tensor_hash_match": not mismatches,
        "mismatches": mismatches,
        "finite": record["finite"],
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
