#!/usr/bin/env python3
"""Closure wrapper for the official SIU3R pair-0 validation smoke."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


EXPECTED_CKPT_SIZE = 5464307091
EXPECTED_CKPT_SHA = "0c6b3e6eac8a44a864ec98c07b417aabcce4a12e510bed5f46c77afed74e46a0"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--partition", default="3090")
    parser.add_argument("--prediction-bundle", type=Path, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    started = time.time()
    ckpt = {"path": str(args.checkpoint), "exists": args.checkpoint.is_file(), "size": None, "sha256": None}
    if ckpt["exists"]:
        ckpt["size"] = args.checkpoint.stat().st_size
        if ckpt["size"] == EXPECTED_CKPT_SIZE:
            ckpt["sha256"] = sha256(args.checkpoint)
    pair = {"path": str(args.pairs), "exists": args.pairs.is_file(), "sha256": sha256(args.pairs) if args.pairs.is_file() else None}
    payload = {
        "protocol": "siu3r_official_reference_pair0_smoke",
        "pair_index": 0,
        "partition": args.partition,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": ckpt,
        "val_pair": pair,
        "optimizer_step_executed": False,
        "ttt_started": False,
        "oracle_used": False,
    }
    if not ckpt["exists"] or ckpt["size"] != EXPECTED_CKPT_SIZE or ckpt["sha256"] != EXPECTED_CKPT_SHA:
        payload.update({"status": "BLOCKED_CHECKPOINT_SHA_MISMATCH", "official_forward_valid": False})
        write_atomic(args.output, payload)
        return 3
    if not pair["exists"]:
        payload.update({"status": "BLOCKED_VAL_PAIR_MISSING", "official_forward_valid": False})
        write_atomic(args.output, payload)
        return 3
    probe = subprocess.run([sys.executable, "-c", "import torch; print('cuda_available='+str(torch.cuda.is_available()))"], capture_output=True, text=True, check=False)
    payload["cuda_probe"] = {"returncode": probe.returncode, "stdout": probe.stdout.strip(), "stderr": probe.stderr.strip()}
    if probe.returncode != 0 or "cuda_available=True" not in probe.stdout:
        payload.update({"status": "OFFICIAL_REFERENCE_RUNTIME_FAILED", "reason": "CUDA device unavailable; pair-0 not started", "official_forward_valid": False, "elapsed_seconds": time.time() - started})
        write_atomic(args.output, payload)
        return 4
    runner = Path(__file__).with_name("run_siu3r_official_pair0.py")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    runner_command = [sys.executable, str(runner), "--data-dir", str(args.data_dir), "--pairs", str(args.pairs), "--checkpoint", str(args.checkpoint), "--output", str(args.output), "--partition", args.partition]
    if args.prediction_bundle is not None:
        runner_command.extend(["--prediction-bundle", str(args.prediction_bundle)])
    with args.log.open("w", encoding="utf-8") as log:
        process = subprocess.run(runner_command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0 and not args.output.exists():
        payload.update({"status": "OFFICIAL_REFERENCE_RUNTIME_FAILED", "reason": f"official runner returncode={process.returncode}", "official_forward_valid": False, "runner_returncode": process.returncode, "elapsed_seconds": time.time() - started})
        write_atomic(args.output, payload)
        return process.returncode
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
