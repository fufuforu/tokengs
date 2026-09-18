"""Summarize the completed EQC diagnostic40 rank traces without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ws = args.workspace.resolve()
    samples = (ws / ".ddp_trace" / "rank_samples.txt").read_text(encoding="utf-8")
    hashes = (ws / ".ddp_trace" / "rank_hashes.txt").read_text(encoding="utf-8")
    sample_rows = [
        {
            "rank": int(rank),
            "step": int(step),
            "scene": scene,
        }
        for rank, step, scene in re.findall(
            r"rank=(\d+) step=(\d+) epoch=\d+ scene=([^\s]+)", samples
        )
    ]
    hash_rows = [
        {"rank": int(rank), "step": int(step), "sha256": digest}
        for rank, step, digest in re.findall(
            r"rank=(\d+) step=(\d+) sha=([0-9a-f]+)", hashes
        )
    ]
    by_step = {}
    hash_by_step = {}
    for row in sample_rows:
        by_step.setdefault(row["step"], []).append(row)
    for row in hash_rows:
        hash_by_step.setdefault(row["step"], []).append(row)
    ranks = sorted({row["rank"] for row in sample_rows})
    steps = sorted(by_step)
    all_rank_steps = all(
        sorted(row["rank"] for row in by_step.get(step, [])) == ranks
        for step in steps
    )
    hash_sync = all(
        len({row["sha256"] for row in hash_by_step.get(step, [])}) == 1
        and sorted(row["rank"] for row in hash_by_step.get(step, [])) == ranks
        for step in steps
    )
    first_step_rows = by_step.get(steps[0], []) if steps else []
    first_step_rank_scene_distinct = (
        len({row["scene"] for row in first_step_rows}) == len(ranks)
    )
    stdout = ws / "logs" / "slurm_53649.out"
    stdout_text = stdout.read_text(encoding="utf-8", errors="replace")
    losses = re.findall(r"\[INFO\] step=(\d+).*?loss: ([0-9.eE+-]+)", stdout_text)
    report = {
        "workspace": str(ws),
        "job_id": "53649",
        "node": "3dimage-11",
        "world_size": len(ranks),
        "ranks": ranks,
        "completed_steps": steps,
        "all_ranks_reached_each_step": all_rank_steps,
        "all_rank_parameter_hashes_match_each_step": hash_sync,
        "first_step_rank_scene_labels_distinct": first_step_rank_scene_distinct,
        "rank_trace_note": (
            "The existing trainer trace records scene labels, not full frame/window "
            "fingerprints; repeated scene labels at later steps can still be distinct "
            "windows."
        ),
        "step40_completed": 40 in steps and all_rank_steps,
        "step40_complete_marker": (ws / "checkpoints" / "step_000040.complete").is_file(),
        "loss_records_finite": all(
            __import__("math").isfinite(float(value)) for _, value in losses
        ),
        "loss_record_count": len(losses),
        "sample_records": sample_rows,
        "hash_records": hash_rows,
        "stderr_has_cuda_or_watchdog_error": bool(
            re.search(
                r"watchdog|SIGABRT|illegal memory|Traceback|CUDA error",
                (ws / "logs" / "slurm_53649.err").read_text(
                    encoding="utf-8", errors="replace"
                ),
                re.I,
            )
        ),
        "training_started": True,
        "formal_eqc_retry_started": False,
        "optimizer_steps_in_formal_retry": 0,
        "checkpoint_step40_sha256": hashlib.sha256(
            (ws / "checkpoints" / "model_step_000040.safetensors").read_bytes()
        ).hexdigest(),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
