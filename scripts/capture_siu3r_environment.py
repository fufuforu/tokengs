#!/usr/bin/env python3
"""Capture official SIU3R environment facts without importing model code."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True).stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = command_output([sys.executable, "-m", "pip", "freeze"])
    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "torch": version("torch"),
        "torchvision": version("torchvision"),
        "torchmetrics": version("torchmetrics"),
        "cuda_runtime": command_output([sys.executable, "-c", "import torch; print(torch.version.cuda); print('cuda_available='+str(torch.cuda.is_available()))"]),
        "gpu_model": command_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        "gpu_driver": command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "pip_freeze": freeze.splitlines(),
        "official_expected": {"python": "3.10", "torch": "2.4.1", "torchvision": "0.19.1", "torchmetrics": "1.7.3", "cuda": "11.8 compatible"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(args.output)
    print(json.dumps({key: payload[key] for key in ("python", "torch", "torchvision", "torchmetrics", "gpu_model")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
