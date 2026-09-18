#!/usr/bin/env python3
"""GPU-side import and rasterizer smoke; never loads a checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback


def write_new(path: Path, data: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-repo", type=Path, required=True)
    ap.add_argument("--gsplat-output", type=Path, required=True)
    ap.add_argument("--official-output", type=Path, required=True)
    args = ap.parse_args()
    started = time.time()
    sys.path.insert(0, str(args.official_repo.resolve()))
    import torch

    base = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    gs = dict(base)
    try:
        import gsplat
        from gsplat import rasterization
        gs.update({
            "gsplat_import_valid": True,
            "gsplat_extension_path": str(getattr(gsplat, "__file__", None)),
            "gsplat_version": getattr(gsplat, "__version__", None),
        })
        # This branch is intentionally only reachable after the real extension imports.
        # Use a minimal legal 3D Gaussian and the exact public API used by SIU3R.
        means = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
        quats = torch.zeros((1, 4), device="cuda", dtype=torch.float32); quats[:, 0] = 1
        scales = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
        opacities = torch.ones((1,), device="cuda", dtype=torch.float32)
        colors = torch.ones((1, 3), device="cuda", dtype=torch.float32)
        viewmats = torch.eye(4, device="cuda", dtype=torch.float32)[None]
        Ks = torch.eye(3, device="cuda", dtype=torch.float32)[None]
        Ks[:, 0, 0] = Ks[:, 1, 1] = 32
        Ks[:, 0, 2] = Ks[:, 1, 2] = 16
        out = rasterization(means, quats, scales, opacities, colors, viewmats, Ks, 32, 32, sh_degree=None)
        tensors = out if isinstance(out, (tuple, list)) else (out,)
        gs["gsplat_cuda_kernel_valid"] = True
        gs["gsplat_output_shapes"] = [list(x.shape) for x in tensors if hasattr(x, "shape")]
        gs["gsplat_output_finite"] = all(bool(torch.isfinite(x).all().item()) for x in tensors if torch.is_tensor(x))
        gs["gsplat_backward_valid"] = False
    except Exception as exc:
        gs.update({
            "gsplat_import_valid": False,
            "gsplat_extension_path": None,
            "gsplat_version": None,
            "gsplat_cuda_kernel_valid": False,
            "gsplat_output_finite": False,
            "gsplat_backward_valid": False,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        })
    gs["elapsed_seconds"] = time.time() - started
    write_new(args.gsplat_output, gs)

    official = dict(base)
    imports = ["src.pipeline", "src.models.model", "src.evaluator"]
    official["imports"] = {}
    for name in imports:
        try:
            __import__(name)
            official["imports"][name] = {"valid": True}
        except Exception as exc:
            official["imports"][name] = {
                "valid": False,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
    official["official_pipeline_import_valid"] = all(v["valid"] for v in official["imports"].values())
    official["checkpoint_loaded"] = False
    official["model_forward_started"] = False
    official["elapsed_seconds"] = time.time() - started
    write_new(args.official_output, official)
    print(json.dumps({"gsplat_import_valid": gs["gsplat_import_valid"], "official_pipeline_import_valid": official["official_pipeline_import_valid"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
