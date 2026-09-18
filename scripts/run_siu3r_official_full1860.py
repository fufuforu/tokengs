#!/usr/bin/env python3
"""Run the official SIU3R validation entry point for all 1860 pairs.

This wrapper is intentionally gated by a successful pair-0 parity artifact and
CUDA availability.  It never invokes the training branch.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--partition", default="3090")
    parser.add_argument("--pair0-gate", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, default=Path("/space/mawb/SIU3R"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size != EXPECTED_CKPT_SIZE or sha256(args.checkpoint) != EXPECTED_CKPT_SHA:
        raise SystemExit("official checkpoint size/SHA256 gate failed")
    if not args.pairs.is_file() or not args.data_dir.is_dir():
        raise SystemExit("official data/pair gate failed")
    gate = json.loads(args.pair0_gate.read_text()) if args.pair0_gate.is_file() else {}
    if gate.get("status") not in {"PASS", "NUMERICAL_PARITY_PASS"}:
        raise SystemExit("pair-0 official/adapter parity gate is not proven; full1860 not started")
    probe = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"], capture_output=True, text=True, check=False)
    if probe.returncode != 0 or probe.stdout.strip() != "True":
        raise SystemExit("official full1860 requires CUDA; full1860 not started")
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    official_repo = args.official_repo.resolve()
    if not (official_repo / "src" / "run.py").is_file():
        raise SystemExit(f"official SIU3R repository is missing src/run.py: {official_repo}")
    command = [sys.executable, str(official_repo / "src/run.py"), "mode=val", f"ckpt_path={args.checkpoint}", "trainer.accelerator=gpu", "trainer.devices=1", "trainer.strategy=ddp_find_unused_parameters_true", "datamodule.val_loader_cfg.batch_size=8", "datamodule.val_loader_cfg.num_workers=0", f"datamodule.dataset_cfg.data_dir={args.data_dir}", f"datamodule.dataset_cfg.val_pair_json={args.pairs}", f"hydra.run.dir={output_dir}", "hydra.job.chdir=false"]
    started = time.time()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=official_repo, stdout=log, stderr=subprocess.STDOUT, check=False)
    payload = {"protocol": "siu3r_global_multiview_v1", "started_at": datetime.now(timezone.utc).isoformat(), "partition": args.partition, "command": command, "returncode": process.returncode, "elapsed_seconds": time.time() - started, "pairs_expected": 1860, "training_started": False, "optimizer_step_executed": False, "ttt_started": False, "oracle_used": False}
    metrics = sorted(output_dir.rglob("results.json"))
    if process.returncode == 0 and metrics:
        source = metrics[-1]
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        shutil.copyfile(source, tmp)
        tmp.replace(args.output)
    else:
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(args.output)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
