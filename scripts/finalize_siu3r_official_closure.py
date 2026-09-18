#!/usr/bin/env python3
"""Write explicit no-GPU closure artifacts without inventing metrics."""

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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--siu3r", type=Path, required=True)
    args = parser.parse_args()
    root, siu = args.root, args.siu3r
    pair = siu / "data/scannet/val_pair.json"
    ckpt = siu / "pretrained_weights/siu3r_epoch100.ckpt"
    env = json.loads((root / "environment.json").read_text())
    integrity = json.loads((root / "data_integrity_report.json").read_text())
    extraction = json.loads((root / "extraction_manifest.json").read_text())
    download = json.loads((root / "download_manifest.json").read_text())
    commit = subprocess.run(["git", "-C", str(siu), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    status = subprocess.run(["git", "-C", str(siu), "status", "--short"], capture_output=True, text=True, check=False).stdout.strip()
    pair0 = json.loads((root / "pair0/pair0_official_raw.json").read_text())
    pair_dir = root / "pair0"
    atomic(pair_dir / "pair0_adapter_raw.json", {"status": "NOT_RUN_OFFICIAL_SMOKE_FAILED", "reason": "No common prediction tensor exists because official pair-0 forward was blocked by unavailable CUDA", "adapter": "tokengs_siu3r_isolated", "oracle_used": False, "ttt_started": False})
    parity = {"status": "NOT_RUN_OFFICIAL_SMOKE_FAILED", "tolerance": {"fp32": 1e-6, "lpips_or_gpu_reduction": 1e-5}, "metrics": [], "first_intermediate_divergence": None, "reason": "Official pair-0 did not produce predictions; same-prediction comparison is therefore not applicable."}
    atomic(pair_dir / "pair0_numerical_parity.json", parity)
    (pair_dir / "pair0_numerical_parity.md").write_text("# Pair-0 numerical parity\n\nStatus: `NOT_RUN_OFFICIAL_SMOKE_FAILED`. No official prediction tensor was produced because the CUDA preflight reported `cuda_available=False`; no adapter rerun or substitute prediction was used.\n")
    atomic(pair_dir / "pair0_prediction_manifest.json", {"status": "NOT_CREATED_OFFICIAL_SMOKE_FAILED", "prediction_hashes": {}, "gt_hashes": {}, "reason": "No prediction bundle without a successful official forward"})
    full = root / "full1860"
    for name, payload in {
        "official_metrics_raw.json": {"status": "NOT_STARTED_PAIR0_GATE_FAILED"},
        "official_metrics_normalized.json": {"status": "NOT_STARTED_PAIR0_GATE_FAILED"},
        "per_pair_metrics.json": {"status": "NOT_STARTED_PAIR0_GATE_FAILED", "pairs": []},
        "paper_comparison.json": {"status": "NOT_EVALUATED_PAIR0_GATE_FAILED", "metrics": []},
        "evaluation_manifest.json": {"started": False, "completed": False, "pairs_evaluated": 0, "reason": "Pair-0 official forward and parity gates were not satisfied"},
    }.items():
        atomic(full / name, payload)
    (full / "paper_comparison.md").write_text("# Paper comparison\n\nStatus: `NOT_EVALUATED_PAIR0_GATE_FAILED`; no paper comparison is reported without an official complete evaluation.\n")
    ckpt_sha = sha256(ckpt) if ckpt.is_file() and ckpt.stat().st_size == EXPECTED_CKPT_SIZE else None
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conclusion": "D_OFFICIAL_REFERENCE_RUNTIME_FAILED",
        "SIU3R_REPO_COMMIT": commit,
        "SIU3R_REPO_CLEAN": "YES" if not status else "NO",
        "HF_DATASET_REVISION": download.get("hf_revision"),
        "VAL_DOWNLOAD_COMPLETE": "YES" if download.get("download_complete") else "NO",
        "VAL_ARCHIVE_COUNT": download.get("archive_count"),
        "VAL_EXTRACTED_SCENE_COUNT": extraction.get("scene_count"),
        "VAL_PAIR_SHA_MATCH": "YES" if pair.is_file() and sha256(pair) == EXPECTED_PAIR_SHA else "NO",
        "VAL_PAIR_RECORDS": integrity.get("pair_records"),
        "VAL_PAIR_SCENES": integrity.get("scene_count"),
        "ALL_REFERENCED_FRAMES_FOUND": "YES" if integrity.get("all_referenced_frames_found") else "NO",
        "DATA_INTEGRITY_VALID": "YES" if integrity.get("data_integrity_valid") else "NO",
        "OFFICIAL_CHECKPOINT_FOUND": "YES" if ckpt.is_file() else "NO",
        "OFFICIAL_CHECKPOINT_SIZE": ckpt.stat().st_size if ckpt.is_file() else None,
        "OFFICIAL_CHECKPOINT_SHA_MATCH": "YES" if ckpt_sha == EXPECTED_CKPT_SHA else "NO",
        "OFFICIAL_ENVIRONMENT_VALID": "NO",
        "OFFICIAL_TORCH_VERSION": env.get("torch"),
        "OFFICIAL_TORCHVISION_VERSION": env.get("torchvision"),
        "OFFICIAL_TORCHMETRICS_VERSION": env.get("torchmetrics"),
        "PAIR0_OFFICIAL_FORWARD_VALID": "YES" if pair0.get("official_forward_valid") else "NO",
        "PAIR0_ALL_OUTPUTS_FINITE": "NO",
        "PAIR0_PARAMETER_HASH_UNCHANGED": "NOT_APPLICABLE_FORWARD_NOT_STARTED",
        "PAIR0_ADAPTER_NUMERICAL_PARITY": "NO_NOT_RUN",
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
        "FILES_MODIFIED": ["isolated TokenGS SIU3R scripts/tests", str(root), str(siu / "data/scannet"), str(siu / "pretrained_weights/siu3r_epoch100.ckpt"), str(siu / ".venv")],
        "TRAINING_STARTED": "NO",
        "OPTIMIZER_STEP_EXECUTED": "NO",
        "TOKEN_GS_MODEL_MODIFIED": "NO",
        "CHECKPOINT_MODIFIED": "NO",
        "LSM40_RESULTS_MODIFIED": "NO",
        "runtime_blocker": pair0.get("reason"),
        "official_environment": env,
    }
    atomic(root / "final_report.json", report)
    lines = ["# SIU3R Official Validation and Evaluation Closure v1", "", "Conclusion: `D_OFFICIAL_REFERENCE_RUNTIME_FAILED`", "", "The official val data and checkpoint are complete and validated. Pair-0 was attempted but CUDA was unavailable (`cuda_available=False`), so no checkpoint forward, prediction bundle, adapter parity comparison, or full 1860-pair evaluation was started. No training, optimizer step, TTT, oracle, checkpoint modification, or LSM-40 modification occurred.", "", "Key state:"]
    for key in ("SIU3R_REPO_COMMIT", "SIU3R_REPO_CLEAN", "HF_DATASET_REVISION", "VAL_DOWNLOAD_COMPLETE", "VAL_ARCHIVE_COUNT", "VAL_EXTRACTED_SCENE_COUNT", "VAL_PAIR_SHA_MATCH", "VAL_PAIR_RECORDS", "VAL_PAIR_SCENES", "ALL_REFERENCED_FRAMES_FOUND", "DATA_INTEGRITY_VALID", "OFFICIAL_CHECKPOINT_SHA_MATCH", "OFFICIAL_ENVIRONMENT_VALID", "PAIR0_OFFICIAL_FORWARD_VALID", "PAIR0_ADAPTER_NUMERICAL_PARITY", "FULL1860_STARTED", "FULL1860_COMPLETED", "TRAINING_STARTED", "OPTIMIZER_STEP_EXECUTED", "TOKEN_GS_MODEL_MODIFIED", "CHECKPOINT_MODIFIED", "LSM40_RESULTS_MODIFIED"):
        lines.append(f"- `{key}={report[key]}`")
    (root / "final_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"conclusion": report["conclusion"], "data_integrity": report["DATA_INTEGRITY_VALID"], "pair0": report["PAIR0_OFFICIAL_FORWARD_VALID"], "full1860": report["FULL1860_STARTED"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
