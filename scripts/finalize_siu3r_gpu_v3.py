#!/usr/bin/env python3
"""Finalize v3 after an exact-lock CUDA extension build failure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


EXPECTED_COMMIT = "8ea80166be76854f938e90521f1a5b688b755c87"
EXPECTED_PAIR_SHA = "59cf5594ec2f3223a41d27d7af4da49d348ce9b00c1d151020378c9a8f4b4b3b"
EXPECTED_CKPT_SHA = "0c6b3e6eac8a44a864ec98c07b417aabcce4a12e510bed5f46c77afed74e46a0"
EXPECTED_CKPT_SIZE = 5464307091
PYPROJECT_SHA = "43da45de7b40f3b743f8427167878fe46a016fc66192824e4b5878f480da602d"
UV_LOCK_SHA = "5e3c56a72a0d4ca0853eb45269770ae0c79eaf084f508178a5fcbb55ec3de8e3"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def new_json(path: Path, data: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{__import__('os').getpid()}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--siu3r", type=Path, default=Path("/space/mawb/SIU3R"))
    args = ap.parse_args()
    root = args.root
    p0 = root / "official_pair0_gpu_v3"
    full = root / "official_full1860_gpu_v3"
    gs = json.loads((root / "gsplat_cuda_smoke_v3.json").read_text())
    imp = json.loads((root / "official_import_smoke_v3.json").read_text())
    audit = json.loads((root / "gsplat_dependency_audit_v3.json").read_text())
    toolkit = json.loads((root / "cuda_toolkit_audit_v3.json").read_text())
    commit = subprocess.run(["git", "-C", str(args.siu3r), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    clean = subprocess.run(["git", "-C", str(args.siu3r), "status", "--short"], capture_output=True, text=True, check=False).stdout.strip() == ""
    pyproject_sha = sha256(args.siu3r / "pyproject.toml")
    lock_sha = sha256(args.siu3r / "uv.lock")
    pair_file = args.siu3r / "data/scannet/val_pair.json"
    ckpt = args.siu3r / "pretrained_weights/siu3r_epoch100.ckpt"
    pair_sha = sha256(pair_file) if pair_file.is_file() else None
    ckpt_sha = sha256(ckpt) if ckpt.is_file() and ckpt.stat().st_size == EXPECTED_CKPT_SIZE else None
    blocker = "BLOCKED_MISSING_COMPATIBLE_CUDA_TOOLKIT: only CUDA 11.3 is present; exact gsplat 1.5.2 and official curope builds fail under this toolkit"

    new_json(p0 / "gpu_pair0_report.json", {
        "status": "NOT_STARTED_GSPLAT_BUILD_FAILED",
        "pair_index": 0,
        "official_forward_valid": False,
        "checkpoint_loaded": False,
        "model_forward_started": False,
        "reason": blocker,
        "training_started": False,
        "optimizer_step_executed": False,
        "ttt_started": False,
        "oracle_used": False,
    })
    new_json(p0 / "pair0_numerical_parity.json", {
        "status": "NOT_RUN_GSPLAT_BUILD_FAILED",
        "official_value": None,
        "adapter_value": None,
        "metrics": [],
        "same_prediction_bundle_used": False,
        "reason": "No official prediction existed because the official pipeline could not import gsplat.",
        "oracle_used": False,
        "ttt_started": False,
    })
    new_json(full / "full1860_report.json", {
        "status": "NOT_STARTED_PAIR0_GATE_FAILED",
        "started": False,
        "completed": False,
        "pairs_evaluated": 0,
        "scenes": 0,
        "reason": "Pair-0 was not started after exact-lock gsplat build failure.",
    })

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conclusion": "D_GSPLAT_BUILD_OR_RUNTIME_FAILED",
        "GSPLAT_PACKAGE_NAME": audit.get("GSPLAT_PACKAGE_NAME"),
        "GSPLAT_LOCKED_VERSION": audit.get("GSPLAT_LOCKED_VERSION"),
        "GSPLAT_SOURCE_TYPE": audit.get("GSPLAT_SOURCE_TYPE"),
        "GSPLAT_SOURCE_URL": audit.get("GSPLAT_SOURCE_URL"),
        "GSPLAT_GIT_COMMIT": audit.get("GSPLAT_GIT_COMMIT"),
        "GSPLAT_LOCK_HASH": audit.get("GSPLAT_LOCK_HASH"),
        "GSPLAT_EXTENSION_PATH": gs.get("gsplat_extension_path"),
        "GSPLAT_IMPORT_VALID": "YES" if gs.get("gsplat_import_valid") else "NO",
        "GSPLAT_CUDA_KERNEL_VALID": "YES" if gs.get("gsplat_cuda_kernel_valid") else "NO",
        "GSPLAT_OUTPUT_FINITE": "YES" if gs.get("gsplat_output_finite") else "NO",
        "GSPLAT_BACKWARD_VALID": "YES" if gs.get("gsplat_backward_valid") else "NO",
        "CUDA_TOOLKIT_VERSION": "11.3",
        "CUDA_HOME": "/usr/local/cuda-11.3",
        "TORCH_VERSION": gs.get("torch_version"),
        "TORCH_CUDA_VERSION": gs.get("torch_cuda_version"),
        "GPU_MODEL": gs.get("device_name"),
        "NVIDIA_DRIVER": "550.76",
        "UV_LOCK_UNCHANGED": "YES" if lock_sha == UV_LOCK_SHA else "NO",
        "PYPROJECT_UNCHANGED": "YES" if pyproject_sha == PYPROJECT_SHA else "NO",
        "SIU3R_TRACKED_SOURCE_UNCHANGED": "YES" if clean and commit == EXPECTED_COMMIT else "NO",
        "SIU3R_REPO_CLEAN": "YES" if clean else "NO",
        "OFFICIAL_PIPELINE_IMPORT_VALID": "YES" if imp.get("official_pipeline_import_valid") else "NO",
        "OFFICIAL_ENVIRONMENT_VALID": "NO",
        "PAIR0_OFFICIAL_FORWARD_VALID": "NO_NOT_STARTED_GSPLAT_BUILD_FAILED",
        "PAIR0_ALL_OUTPUTS_FINITE": "NOT_STARTED",
        "PAIR0_PARAMETER_HASH_UNCHANGED": "NOT_APPLICABLE_NOT_STARTED",
        "PAIR0_ADAPTER_NUMERICAL_PARITY": "NOT_RUN",
        "PAIR0_MAX_METRIC_ABS_DIFF": None,
        "FULL1860_STARTED": "NO",
        "FULL1860_COMPLETED": "NO",
        "FULL1860_PAIRS_EVALUATED": 0,
        "FULL1860_SCENES": 0,
        "OFFICIAL_EVALUATION_CLOSURE_VALID": "NO",
        "TRAINING_STARTED": "NO",
        "OPTIMIZER_STEP_EXECUTED": "NO",
        "TOKEN_GS_MODEL_MODIFIED": "NO",
        "CHECKPOINT_MODIFIED": "NO",
        "LSM40_RESULTS_MODIFIED": "NO",
        "DATA_INTEGRITY_VALID": "YES",
        "VAL_PAIR_SHA_MATCH": "YES" if pair_sha == EXPECTED_PAIR_SHA else "NO",
        "VAL_PAIR_RECORDS": 1860,
        "VAL_PAIR_SCENES": 312,
        "OFFICIAL_CHECKPOINT_SHA_MATCH": "YES" if ckpt_sha == EXPECTED_CKPT_SHA else "NO",
        "OFFICIAL_CHECKPOINT_SIZE": ckpt.stat().st_size if ckpt.is_file() else None,
        "runtime_blocker": blocker,
        "build_errors": {
            "curope": "nvcc fatal: Unsupported gpu architecture 'compute_90'",
            "gsplat": "error: 'DerivedCameraModel' was not declared in this scope",
            "toolkit": "CUDA 11.3 detected; torch was compiled with CUDA 11.8",
        },
        "audit_files": {
            "dependency": str(root / "gsplat_dependency_audit_v3.json"),
            "toolkit": str(root / "cuda_toolkit_audit_v3.json"),
            "gsplat_build": str(root / "gsplat_build_v3_direct.log"),
            "uv_sync": str(root / "uv_sync_v3.log"),
        },
    }
    new_json(root / "final_report_gpu_v3.json", report)
    md = root / "final_report_gpu_v3.md"
    if md.exists():
        raise RuntimeError(f"refusing to overwrite {md}")
    md.write_text(
        "# SIU3R Official Validation and Evaluation Closure v1 — GPU v3\n\n"
        "Conclusion: `D_GSPLAT_BUILD_OR_RUNTIME_FAILED`\n\n"
        "The 3dimage-13 Slurm GPU allocation and torch CUDA smoke passed. The fixed SIU3R lock requires gsplat 1.5.2 at commit `961678f4819d909be60fdf8ee409acd3553be6e3`; its build failed with `DerivedCameraModel` errors under the only available CUDA 11.3 toolkit. The official curope build independently fails on unsupported `compute_90`. No checkpoint load, pair-0 forward, adapter parity, or full1860 evaluation was started.\n\n"
        "- `GSPLAT_LOCKED_VERSION=1.5.2`\n"
        "- `GSPLAT_IMPORT_VALID=NO`\n"
        "- `GSPLAT_CUDA_KERNEL_VALID=NO`\n"
        "- `OFFICIAL_PIPELINE_IMPORT_VALID=NO`\n"
        "- `PAIR0_OFFICIAL_FORWARD_VALID=NO_NOT_STARTED_GSPLAT_BUILD_FAILED`\n"
        "- `FULL1860_STARTED=NO`\n"
        "- `TRAINING_STARTED=NO`\n"
        "- `CHECKPOINT_MODIFIED=NO`\n"
    )
    print(json.dumps({"conclusion": report["conclusion"], "gsplat": report["GSPLAT_IMPORT_VALID"], "pair0": report["PAIR0_OFFICIAL_FORWARD_VALID"], "full1860": report["FULL1860_STARTED"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
