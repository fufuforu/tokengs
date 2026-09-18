#!/usr/bin/env python3
"""Capture immutable GPU/official-venv preflight evidence inside Slurm."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout.strip()


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    started = time.time()
    nvidia_rc, nvidia = run(["nvidia-smi"])
    import torch
    import torchvision
    import torchmetrics

    cuda_test = None
    device_name = None
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        x = torch.randn((16, 16), device="cuda")
        cuda_test = bool(torch.isfinite(x).all().item())
        torch.cuda.synchronize()
    payload = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi_returncode": nvidia_rc,
        "nvidia_smi": nvidia,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torchmetrics": torchmetrics.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device_name": device_name,
        "cuda_tensor_finite": cuda_test,
        "elapsed_seconds": time.time() - started,
        "preflight_valid": bool(
            nvidia_rc == 0
            and os.environ.get("CUDA_VISIBLE_DEVICES", "")
            and torch.cuda.is_available()
            and torch.cuda.device_count() >= 1
            and cuda_test is True
        ),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["preflight_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
