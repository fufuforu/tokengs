#!/usr/bin/env python3
"""Finalize the GPU-v2 closure without inventing metrics after a blocked smoke."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


PAIR_SHA = "59cf5594ec2f3223a41d27d7af4da49d348ce9b00c1d151020378c9a8f4b4b3b"
CKPT_SHA = "0c6b3e6eac8a44a864ec98c07b417aabcce4a12e510bed5f46c77afed74e46a0"
CKPT_SIZE = 5464307091
REPO_COMMIT = "8ea80166be76854f938e90521f1a5b688b755c87"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{__import__('os').getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--siu3r", type=Path, default=Path("/space/mawb/SIU3R"))
    args = ap.parse_args()
    root = args.root
    pair0 = root / "official_pair0_gpu_v2"
    full = root / "official_full1860_gpu_v2"
    pair_report = json.loads((pair0 / "gpu_pair0_report.json").read_text())
    gpu = json.loads((pair0 / "gpu_preflight.json").read_text())
    pair_file = args.siu3r / "data/scannet/val_pair.json"
    ckpt = args.siu3r / "pretrained_weights/siu3r_epoch100.ckpt"
    commit = subprocess.run(["git", "-C", str(args.siu3r), "rev-parse", "HEAD"], text=True, capture_output=True, check=False).stdout.strip()
    status = subprocess.run(["git", "-C", str(args.siu3r), "status", "--short"], text=True, capture_output=True, check=False).stdout.strip()
    pair_sha = sha256(pair_file) if pair_file.is_file() else None
    ckpt_sha = sha256(ckpt) if ckpt.is_file() and ckpt.stat().st_size == CKPT_SIZE else None

    parity = {
        "protocol": "siu3r_global_multiview_v1",
        "status": "NOT_RUN_OFFICIAL_FORWARD_FAILED",
        "official_value": None,
        "adapter_value": None,
        "metrics": [],
        "first_intermediate_divergence": "official pipeline import: ModuleNotFoundError: No module named 'gsplat'",
        "tolerance": {"fp32": 1e-6, "lpips_or_gpu_reduction": 1e-5},
        "same_prediction_bundle_used": False,
        "reason": "No official prediction tensor was produced; adapter parity is not applicable and no substitute prediction was generated.",
        "oracle_used": False,
        "ttt_started": False,
    }
    write_new(pair0 / "pair0_numerical_parity.json", parity)
    (pair0 / "pair0_numerical_parity.md").write_text(
        "# Pair-0 numerical parity (GPU v2)\n\n"
        "Status: `NOT_RUN_OFFICIAL_FORWARD_FAILED`. The valid GPU allocation passed CUDA preflight, "
        "but the official pipeline import failed because the locked `gsplat` extension was unavailable. "
        "No prediction tensor existed, so no adapter comparison or substitute metric was run.\n"
    )
    write_new(pair0 / "pair0_prediction_manifest.json", {
        "status": "NOT_CREATED_OFFICIAL_FORWARD_FAILED",
        "prediction_hashes": {},
        "gt_hashes": {},
        "reason": "official forward did not start",
    })

    full_payloads = {
        "official_metrics_raw.json": {"status": "NOT_STARTED_PAIR0_GATE_FAILED"},
        "official_metrics_normalized.json": {"status": "NOT_STARTED_PAIR0_GATE_FAILED"},
        "per_pair_metrics.json": {"status": "NOT_STARTED_PAIR0_GATE_FAILED", "pairs": []},
        "paper_comparison.json": {"status": "NOT_EVALUATED_PAIR0_GATE_FAILED", "metrics": []},
        "evaluation_manifest.json": {
            "started": False, "completed": False, "pairs_evaluated": 0,
            "reason": "Pair-0 official forward and same-prediction parity gates were not satisfied",
        },
    }
    for name, value in full_payloads.items():
        write_new(full / name, value)
    (full / "paper_comparison.md").write_text(
        "# Paper comparison (GPU v2)\n\nStatus: `NOT_EVALUATED_PAIR0_GATE_FAILED`; no official full evaluation was started.\n"
    )
    write_new(root / "full1860_report.json", {
        "status": "NOT_STARTED_PAIR0_GATE_FAILED",
        "started": False,
        "completed": False,
        "pairs_evaluated": 0,
        "scenes": 0,
        "reason": "The pair-0 official forward/import gate failed inside a valid Slurm GPU allocation.",
    })

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conclusion": "D_GPU_RUNTIME_FAILED_INSIDE_VALID_SLURM_ALLOCATION",
        "GPU_NODE": gpu.get("hostname"),
        "GPU_PARTITION": gpu.get("slurm_partition"),
        "GPU_MODEL": gpu.get("device_name"),
        "NVIDIA_DRIVER": "550.76",
        "CUDA_VISIBLE_DEVICES": gpu.get("cuda_visible_devices"),
        "TORCH_CUDA_AVAILABLE": "YES" if gpu.get("cuda_available") else "NO",
        "TORCH_CUDA_VERSION": gpu.get("torch_cuda"),
        "CUDA_TENSOR_TEST": "YES" if gpu.get("cuda_tensor_finite") else "NO",
        "SIU3R_REPO_COMMIT": commit,
        "SIU3R_REPO_CLEAN": "YES" if not status else "NO",
        "HF_DATASET_REVISION": "65ad169493bfd26081f99ba26a5dc964aae40139",
        "VAL_DOWNLOAD_COMPLETE": "YES",
        "VAL_ARCHIVE_COUNT": 312,
        "VAL_EXTRACTED_SCENE_COUNT": 312,
        "VAL_PAIR_SHA_MATCH": "YES" if pair_sha == PAIR_SHA else "NO",
        "VAL_PAIR_RECORDS": 1860,
        "VAL_PAIR_SCENES": 312,
        "ALL_REFERENCED_FRAMES_FOUND": "YES",
        "DATA_INTEGRITY_VALID": "YES",
        "OFFICIAL_CHECKPOINT_FOUND": "YES" if ckpt.is_file() else "NO",
        "OFFICIAL_CHECKPOINT_SIZE": ckpt.stat().st_size if ckpt.is_file() else None,
        "OFFICIAL_CHECKPOINT_SHA_MATCH": "YES" if ckpt_sha == CKPT_SHA else "NO",
        "OFFICIAL_ENVIRONMENT_VALID": "NO",
        "OFFICIAL_TORCH_VERSION": gpu.get("torch"),
        "OFFICIAL_TORCHVISION_VERSION": gpu.get("torchvision"),
        "OFFICIAL_TORCHMETRICS_VERSION": gpu.get("torchmetrics"),
        "PAIR0_OFFICIAL_FORWARD_VALID": "YES" if pair_report.get("official_forward_valid") else "NO",
        "PAIR0_ALL_OUTPUTS_FINITE": "NOT_STARTED_FORWARD_FAILED",
        "PAIR0_PARAMETER_HASH_UNCHANGED": "NOT_APPLICABLE_FORWARD_NOT_STARTED",
        "PAIR0_ADAPTER_NUMERICAL_PARITY": "NOT_RUN_OFFICIAL_FORWARD_FAILED",
        "PAIR0_MAX_METRIC_ABS_DIFF": None,
        "FULL1860_STARTED": "NO",
        "FULL1860_COMPLETED": "NO",
        "FULL1860_PAIRS_EVALUATED": 0,
        "FULL1860_UNIQUE_PAIRS": 0,
        "FULL1860_SCENES": 0,
        "FULL1860_ALL_OUTPUTS_FINITE": "NOT_STARTED",
        "FULL1860_PARAMETER_HASH_UNCHANGED": "NOT_STARTED",
        "PAPER_RECONSTRUCTION_METRICS_MATCH": "NOT_EVALUATED",
        "PAPER_CONTEXT_UNDERSTANDING_METRICS_MATCH": "NOT_EVALUATED",
        "PAPER_NOVEL_UNDERSTANDING_METRICS_MATCH": "NOT_EVALUATED",
        "OFFICIAL_EVALUATION_CLOSURE_VALID": "NO",
        "TRAINING_STARTED": "NO",
        "OPTIMIZER_STEP_EXECUTED": "NO",
        "TOKEN_GS_MODEL_MODIFIED": "NO",
        "CHECKPOINT_MODIFIED": "NO",
        "LSM40_RESULTS_MODIFIED": "NO",
        "runtime_blocker": pair_report.get("reason"),
        "gpu_preflight": gpu,
        "official_repo_status": status,
        "files_modified": [str(pair0), str(full), "/space/mawb/tokengs/scripts"],
    }
    write_new(root / "final_report_gpu_v2.json", report)
    lines = [
        "# SIU3R Official Validation and Evaluation Closure v1 — GPU v2",
        "",
        "Conclusion: `D_GPU_RUNTIME_FAILED_INSIDE_VALID_SLURM_ALLOCATION`",
        "",
        "The GPU allocation was valid (`3dimage-13`, RTX 3090, CUDA tensor test passed). The official pair-0 attempt then failed before model forward because the locked `gsplat` extension was unavailable after its CUDA build failed. Full 1860-pair evaluation was not started.",
        "",
        "Key state:",
    ]
    for key in ("GPU_NODE", "GPU_PARTITION", "GPU_MODEL", "CUDA_VISIBLE_DEVICES", "TORCH_CUDA_AVAILABLE", "TORCH_CUDA_VERSION", "CUDA_TENSOR_TEST", "SIU3R_REPO_COMMIT", "SIU3R_REPO_CLEAN", "VAL_PAIR_SHA_MATCH", "VAL_PAIR_RECORDS", "VAL_PAIR_SCENES", "DATA_INTEGRITY_VALID", "OFFICIAL_CHECKPOINT_SHA_MATCH", "OFFICIAL_ENVIRONMENT_VALID", "PAIR0_OFFICIAL_FORWARD_VALID", "PAIR0_ADAPTER_NUMERICAL_PARITY", "FULL1860_STARTED", "FULL1860_COMPLETED", "TRAINING_STARTED", "OPTIMIZER_STEP_EXECUTED", "TOKEN_GS_MODEL_MODIFIED", "CHECKPOINT_MODIFIED", "LSM40_RESULTS_MODIFIED"):
        lines.append(f"- `{key}={report[key]}`")
    (root / "final_report_gpu_v2.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"conclusion": report["conclusion"], "pair0": report["PAIR0_OFFICIAL_FORWARD_VALID"], "full1860": report["FULL1860_STARTED"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
