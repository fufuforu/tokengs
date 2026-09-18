#!/usr/bin/env python3
"""Capture CUDA toolkit visibility from inside the requested Slurm allocation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time


def run(command: list[str]) -> dict:
    p = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {"command": command, "returncode": p.returncode, "output": p.stdout}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    started = time.time()
    paths = sorted(str(p) for p in Path("/usr/local").glob("cuda*") if p.is_dir())
    data = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cuda_home": os.environ.get("CUDA_HOME", ""),
        "cuda_path": os.environ.get("CUDA_PATH", ""),
        "path": os.environ.get("PATH", ""),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "nvcc": shutil.which("nvcc"),
        "module": shutil.which("module"),
        "cuda_directories": paths,
        "commands": [run(["hostname"]), run(["nvidia-smi"]), run(["bash", "-lc", "command -v nvcc || true"]), run(["bash", "-lc", "nvcc --version || true"]), run(["bash", "-lc", "command -v module || true"]), run(["bash", "-lc", "module avail 2>&1 || true"]), run(["bash", "-lc", "ls -ld /usr/local/cuda* 2>/dev/null || true"])],
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(args.output)
    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
