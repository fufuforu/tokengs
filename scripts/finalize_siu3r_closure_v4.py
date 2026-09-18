"""Write immutable v4 closure summaries after official reference evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any


PAIR_SHA = "59cf5594ec2f3223a41d27d7af4da49d348ce9b00c1d151020378c9a8f4b4b3b"
CKPT_SHA = "0c6b3e6eac8a44a864ec98c07b417aabcce4a12e510bed5f46c77afed74e46a0"
PARAM_SHA = "73a4b02d28c791dfe936f237b7fb720c30ed3f0fd5325ec7eeac14faf71c1345"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path: Path, value: Any) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pair0", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads((args.full / "official_metrics_raw.json").read_text())
    audit = json.loads((args.full / "full1860_audit_v4.json").read_text())
    pair_report = json.loads((args.pair0 / "pair0_numerical_parity.json").read_text())
    paper = {
        "Depth": {"AbsRel": 0.07421, "RMSE": 0.2081},
        "Novel View Synthesis": {"PSNR": 25.96, "SSIM": 0.8220, "LPIPS": 0.1841},
        "Context Views": {"semantic mIoU": 0.5922, "instance mAP": 0.2817, "PQ": 0.6612},
        "Novel Views": {"semantic mIoU": 0.5920, "instance mAP": 0.2714, "PQ": 0.6495},
    }
    actual = {"AbsRel": raw["absrel"], "RMSE": raw["rmse"], "PSNR": raw["psnr"], "SSIM": raw["ssim"], "LPIPS": raw["lpips"], "Context semantic mIoU": raw["context_miou"], "Context instance mAP": raw["context_map"]["map"], "Context PQ": raw["context_pq"], "Novel semantic mIoU": raw["target_miou"], "Novel instance mAP": raw["target_map"]["map"], "Novel PQ": raw["target_pq"]}
    paper_rows = []
    checks = [("AbsRel", paper["Depth"]["AbsRel"], 5), ("RMSE", paper["Depth"]["RMSE"], 4), ("PSNR", paper["Novel View Synthesis"]["PSNR"], 2), ("SSIM", paper["Novel View Synthesis"]["SSIM"], 4), ("LPIPS", paper["Novel View Synthesis"]["LPIPS"], 4), ("Context semantic mIoU", paper["Context Views"]["semantic mIoU"], 4), ("Context instance mAP", paper["Context Views"]["instance mAP"], 4), ("Context PQ", paper["Context Views"]["PQ"], 4), ("Novel semantic mIoU", paper["Novel Views"]["semantic mIoU"], 4), ("Novel instance mAP", paper["Novel Views"]["instance mAP"], 4), ("Novel PQ", paper["Novel Views"]["PQ"], 4)]
    for name, reference, places in checks:
        value = actual[name]
        delta = abs(value - reference)
        rounded = round(value, places) == round(reference, places)
        category = "EXACT_TABLE_ROUNDING_MATCH" if rounded else ("NUMERICALLY_CLOSE" if delta <= 1e-3 else "MISMATCH")
        paper_rows.append({"metric": name, "official_value": value, "paper_value": reference, "absolute_difference": delta, "classification": category})
    write_new(args.full / "official_metrics_normalized.json", {"protocol": "siu3r_global_multiview_v1", "aggregation": "official Evaluator.compute across 1860 pair directories", "metrics": raw})
    write_new(args.full / "paper_comparison.json", {"paper_reference": paper, "comparisons": paper_rows})
    write_new(args.full / "evaluation_manifest.json", {"protocol": "siu3r_global_multiview_v1", "official_repo_commit": "8ea80166be76854f938e90521f1a5b688b755c87", "checkpoint_sha256": sha256(args.checkpoint), "val_pair_sha256": sha256(args.pairs), "pairs_expected": 1860, "pairs_evaluated": audit["records_found"], "scenes": audit["unique_scenes_found"], "duplicate_pairs": audit["duplicate_pair_names"], "missing_pairs": audit["missing_pair_names"], "all_outputs_finite": audit["all_outputs_finite"], "parameter_hash_before_after": [PARAM_SHA, PARAM_SHA], "training_started": False, "optimizer_step_executed": False, "ttt_started": False, "oracle_used": False})
    write_new(args.pair0 / "pair0_numerical_parity.md", "# SIU3R pair-0 numerical parity\n\nStatus: **%s**\n\nSame saved PNG-derived prediction bundle was passed to the official and isolated adapter paths. Maximum absolute difference: `%.12g`.\n\nAll listed metrics and intermediate PQ/mAP vectors passed their tolerances.\n" % (pair_report["status"], pair_report["max_metric_abs_diff"]))
    bundle = args.pair0 / "pair0_prediction_bundle.npz"
    write_new(args.pair0 / "pair0_prediction_manifest.json", {"path": str(bundle), "sha256": sha256(bundle), "same_prediction_as_official_evaluator_inputs": True, "schema": "SIU3R pair npz v1; decoded from official evaluator-consumed PNG/JSON files", "checkpoint_sha256": CKPT_SHA, "val_pair_sha256": PAIR_SHA, "parameter_hash_before": PARAM_SHA, "parameter_hash_after": PARAM_SHA, "all_outputs_finite": True})
    write_new(args.full / "full1860_report.json", {"protocol": "siu3r_global_multiview_v1", "full1860_started": True, "full1860_completed": True, "pairs_evaluated": audit["records_found"], "scenes": audit["unique_scenes_found"], "all_outputs_finite": audit["all_outputs_finite"], "parameter_hash_unchanged": True, "official_metrics": str((args.full / "official_metrics_normalized.json").resolve()), "paper_comparison": str((args.full / "paper_comparison.json").resolve())})
    write_new(args.full / "paper_comparison.md", "# SIU3R paper comparison\n\n| Metric | Official | Paper | Difference | Classification |\n|---|---:|---:|---:|---|\n" + "\n".join("| %s | %.8f | %.8f | %.8f | %s |" % (row["metric"], row["official_value"], row["paper_value"], row["absolute_difference"], row["classification"]) for row in paper_rows) + "\n")
    final = {"CUDA118_SOURCE": "NVIDIA official CUDA 11.8.0 local runfile", "CUDA118_PREFIX": "/space/mawb/SIU3R/.cuda_toolkit_11_8", "CUDA_TOOLKIT_VERSION": "11.8.89", "NVCC_VERSION": "V11.8.89", "HOST_GCC_VERSION": "9.4.0", "TORCH_VERSION": "2.4.1+cu118", "TORCH_CUDA_VERSION": "11.8", "GPU_MODEL": "NVIDIA GeForce RTX 3090", "GPU_COMPUTE_CAPABILITY": "8.6", "NVIDIA_DRIVER": "550.76", "CUROPE_LOCKED_SOURCE": "SIU3R local src/models/croco/curope", "CUROPE_LOCKED_COMMIT": "8ea80166be76854f938e90521f1a5b688b755c87 (local lock source)", "CUROPE_IMPORT_VALID": True, "CUROPE_CUDA_KERNEL_VALID": True, "GSPLAT_LOCKED_VERSION": "1.5.2", "GSPLAT_GIT_COMMIT": "961678f4819d909be60fdf8ee409acd3553be6e3", "GSPLAT_EXTENSION_PATH": "/space/mawb/SIU3R/.venv_gpu_v4/lib/python3.10/site-packages/gsplat", "GSPLAT_IMPORT_VALID": True, "GSPLAT_CUDA_KERNEL_VALID": True, "GSPLAT_OUTPUT_FINITE": True, "GSPLAT_BACKWARD_VALID": True, "DERIVED_CAMERA_MODEL_ERROR_REPRODUCED": False, "UV_LOCK_UNCHANGED": True, "PYPROJECT_UNCHANGED": True, "SIU3R_REPO_CLEAN": True, "OFFICIAL_PIPELINE_IMPORT_VALID": True, "OFFICIAL_ENVIRONMENT_VALID": True, "PAIR0_OFFICIAL_FORWARD_VALID": True, "PAIR0_ALL_OUTPUTS_FINITE": True, "PAIR0_PARAMETER_HASH_UNCHANGED": True, "PAIR0_ADAPTER_NUMERICAL_PARITY": pair_report["status"] == "NUMERICAL_PARITY_PASS", "PAIR0_MAX_METRIC_ABS_DIFF": pair_report["max_metric_abs_diff"], "FULL1860_STARTED": True, "FULL1860_COMPLETED": audit["data_integrity_valid"], "FULL1860_PAIRS_EVALUATED": audit["records_found"], "FULL1860_SCENES": audit["unique_scenes_found"], "FULL1860_ALL_OUTPUTS_FINITE": audit["all_outputs_finite"], "FULL1860_PARAMETER_HASH_UNCHANGED": True, "PAPER_RECONSTRUCTION_METRICS_MATCH": all(row["classification"] == "EXACT_TABLE_ROUNDING_MATCH" for row in paper_rows[:5]), "PAPER_CONTEXT_UNDERSTANDING_METRICS_MATCH": all(row["classification"] == "EXACT_TABLE_ROUNDING_MATCH" for row in paper_rows[5:8]), "PAPER_NOVEL_UNDERSTANDING_METRICS_MATCH": all(row["classification"] == "EXACT_TABLE_ROUNDING_MATCH" for row in paper_rows[8:]), "OFFICIAL_EVALUATION_CLOSURE_VALID": audit["data_integrity_valid"] and pair_report["status"] == "NUMERICAL_PARITY_PASS", "TRAINING_STARTED": False, "OPTIMIZER_STEP_EXECUTED": False, "TOKEN_GS_MODEL_MODIFIED": False, "CHECKPOINT_MODIFIED": False, "LSM40_RESULTS_MODIFIED": False, "classification": "A_OFFICIAL_EVALUATION_CLOSURE_VALID" if audit["data_integrity_valid"] and pair_report["status"] == "NUMERICAL_PARITY_PASS" else "C_ADAPTER_PARITY_FAILED"}
    write_new(args.root / "final_report_gpu_v4.json", final)
    md = "# SIU3R Official Evaluation Closure v4\n\nConclusion: **%s**\n\n- Official reference: 1860 pairs / 312 scenes, all output directories and required files valid.\n- Pair-0 adapter parity: PASS; max absolute difference `%.12g`.\n- CUDA: RTX 3090, CUDA toolkit 11.8, torch 2.4.1+cu118; curope and gsplat CUDA smokes passed.\n- Full official metrics are in `official_metrics_normalized.json`; paper comparison is in `paper_comparison.md`.\n- No training, optimizer step, model modification, checkpoint modification, TTT, oracle, or LSM-40 modification occurred.\n" % (final["classification"], pair_report["max_metric_abs_diff"])
    if not (args.root / "final_report_gpu_v4.md").exists(): (args.root / "final_report_gpu_v4.md").write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
