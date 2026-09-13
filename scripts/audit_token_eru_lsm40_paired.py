"""Strict paired LSM-40 evaluation for Both@1420 and TokenGS-ERU@500.

This adapter deliberately reuses the existing LSM evaluator's manifest,
DataLoader, mask conversion, AP implementation, and reconstruction metrics.
It adds only the ERU checkpoint-loading path and the requested paired
summary.  No raw prediction cache is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_gsi_v2_lsm40_paired import (  # noqa: E402
    _LocalAccelerator,
    _build_both_model,
    _evaluate_model,
    _manifest_fingerprints,
    _metadata,
    _sha256_file,
    _build_lsm_data,
)
from scripts.eval_instance_lsm_protocol import _audit_lsm_manifest  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import load_model_checkpoint  # noqa: E402
from tokengs.utils.metrics import MetricsCalculator  # noqa: E402


BOTH = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8/"
    "checkpoints/model_step_001420.safetensors"
)
ERU = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_scene_hungarian_"
    "resume200_to500_persist_ddp8/checkpoints/model_step_000500.safetensors"
)
ERU_SHA256 = "cca23e2d24af7a374be6a62f2cf4026645afdf0df49686722cb5277c9e479209"
MANIFEST = ROOT / "data/scannet_prompt/lsm_instance_eval_manifest.json"
ERU_CONFIG = "semantic_v6_absolute_units_true_shared_token_eru1_scene_hungarian_short200_ddp8"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--max_scenes", type=int, default=0)
    return parser


def _build_eru_model(checkpoint: Path, workspace: Path, accelerator: Accelerator):
    opt = config_defaults[ERU_CONFIG].evolve(
        resume=str(checkpoint),
        workspace=str(workspace),
        evaluating=True,
        num_workers=0,
        max_eval_iters=0,
        num_input_views=8,
        num_views=15,
        gsi_v2_return_debug_tensors=False,
    )
    model = model_registry[opt.model_type](opt).to(accelerator.device)
    state = load_file(str(checkpoint), device="cpu")
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        raise RuntimeError("ERU checkpoint did not restore its ERU namespaces")
    schedule = model.set_token_eru_step(500)
    if abs(float(schedule["reconstruction_to_understanding_gate"]) - 1.0) > 1e-8:
        raise RuntimeError(f"unexpected ERU r2u gate: {schedule}")
    if abs(float(schedule["understanding_to_reconstruction_gate"]) - 0.1) > 1e-8:
        raise RuntimeError(f"unexpected ERU u2r gate: {schedule}")
    model.eval()
    return model, schedule, {
        "strict": True,
        "loader": "tokengs.train.load_model_checkpoint",
        "state_key_count": len(state),
        "fresh_reset": False,
        "pgsr_absent": not any("pgsr" in key.lower() for key in state),
        "gsi_absent": not any("gsi" in key.lower() for key in state),
        "ta_riu_absent": not any("ta_riu" in key.lower() for key in state),
        "query_memory_refiner_absent": not any(
            "query_memory" in key.lower() for key in state
        ),
        "token_eru_loaded": True,
        "checkpoint_sha256": _sha256_file(checkpoint),
    }


def _fingerprints_equal(results: list[dict[str, object]]) -> bool:
    reference = results[0]["input_fingerprints"]
    return all(result["input_fingerprints"] == reference for result in results[1:])


def _metadata_for(checkpoint: Path, step: int) -> dict[str, object]:
    return _metadata(checkpoint, step)


def _finite_result(result: dict[str, object]) -> bool:
    for value in result.get("mean", {}).values():
        if isinstance(value, (int, float)) and not np.isfinite(float(value)):
            return False
    for value in result.get("pooled", {}).values():
        if isinstance(value, (int, float)) and not np.isfinite(float(value)):
            return False
    return bool(result.get("all_finite", False))


def _comparison(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    left_rows = left["per_scene"]
    right_rows = right["per_scene"]
    deltas = {}
    for scene in right_rows:
        a, b = left_rows[scene], right_rows[scene]
        deltas[scene] = {
            "delta_ap25": float(a["ap25"] - b["ap25"]),
            "delta_ap50": float(a["ap50"] - b["ap50"]),
            "delta_ap75": float(a["ap75"] - b["ap75"]),
            "delta_best_iou": float(a["mean_best_gt_iou"] - b["mean_best_gt_iou"]),
            "delta_recall50": float(a["recall_iou50"] - b["recall_iou50"]),
            "delta_pred_gt": float(a["pred_gt"] - b["pred_gt"]),
            "delta_psnr": float(a["psnr"] - b["psnr"]),
            "delta_ssim": float(a["ssim"] - b["ssim"]),
            "delta_lpips": float(a["lpips"] - b["lpips"]),
            "eru": a,
            "both": b,
        }
    values = list(deltas.values())
    def count(field: str, predicate) -> int:
        return sum(predicate(float(row[field])) for row in values)
    ranked = sorted(values, key=lambda row: row["delta_ap50"])
    return {
        "per_scene": deltas,
        "ap50_not_lower_scenes": count("delta_ap50", lambda x: x >= 0),
        "ap50_improved_scenes": count("delta_ap50", lambda x: x > 0),
        "ap50_tied_scenes": count("delta_ap50", lambda x: x == 0),
        "ap50_declined_scenes": count("delta_ap50", lambda x: x < 0),
        "best_iou_improved_scenes": count("delta_best_iou", lambda x: x > 0),
        "recall50_improved_scenes": count("delta_recall50", lambda x: x > 0),
        "psnr_improved_scenes": count("delta_psnr", lambda x: x > 0),
        "psnr_declined_over_0_2db_scenes": count("delta_psnr", lambda x: x < -0.2),
        "largest_ap50_losses": [
            {"scene": scene, "delta_ap50": row["delta_ap50"]}
            for scene, row in sorted(deltas.items(), key=lambda item: item[1]["delta_ap50"])[:5]
        ],
        "largest_ap50_gains": [
            {"scene": scene, "delta_ap50": row["delta_ap50"]}
            for scene, row in sorted(deltas.items(), key=lambda item: item[1]["delta_ap50"], reverse=True)[:5]
        ],
    }


def main() -> None:
    args = _parser().parse_args()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != "cdb8bd5640de5928557a4d740e90675046adaa71":
        raise RuntimeError("unexpected git HEAD")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_audit = _audit_lsm_manifest(str(args.manifest))
    if manifest_audit["scene_count"] != 40:
        raise RuntimeError(f"LSM-40 manifest audit failed: {manifest_audit}")
    manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_entries = manifest_payload["scenes"]
    if args.max_scenes < 0:
        raise ValueError("--max_scenes must be non-negative")
    entries = dict(list(all_entries.items())[:args.max_scenes] if args.max_scenes else all_entries.items())
    if len(entries) != (args.max_scenes if args.max_scenes else 40):
        raise RuntimeError("invalid requested scene count")
    for checkpoint, step in ((BOTH, 1420), (ERU, 500)):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not _metadata_for(checkpoint, step):
            raise RuntimeError(f"metadata validation failed: {checkpoint}")
    eru_hash = _sha256_file(ERU)
    if eru_hash != ERU_SHA256:
        raise RuntimeError(f"ERU SHA256 mismatch: {eru_hash}")
    data_opt, test_loader, test_dataset = _build_lsm_data(args.manifest)
    del data_opt, test_dataset
    accelerator = Accelerator(mixed_precision="no")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; LSM evaluation must run on a compute node")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, got {torch.cuda.device_count()}")
    metrics = MetricsCalculator(device=accelerator.device)
    outputs = {}
    metadata = {
        "both_1420": _metadata_for(BOTH, 1420),
        "eru_step500": _metadata_for(ERU, 500),
    }
    for label, checkpoint, kind, step in (
        ("both_1420", BOTH, "both", 1420),
        ("eru_step500", ERU, "eru", 500),
    ):
        model_workspace = args.output_dir / label
        model_workspace.mkdir(parents=True, exist_ok=True)
        if kind == "both":
            model, schedule, restore = _build_both_model(checkpoint, model_workspace)
        else:
            model, schedule, restore = _build_eru_model(checkpoint, model_workspace, accelerator)
        result = _evaluate_model(
            label, model, schedule, test_loader, test_dataset=None,
            manifest_entries=entries, is_instance=True, metrics=metrics,
            metadata=metadata[label], max_scenes=args.max_scenes,
        )
        result.update({
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "optimizer_step": step,
            "eval_precision": "fp32",
            "native_gate": None if kind == "both" else 1.0,
            "schedule": schedule,
            "restore": restore,
            "gaussians_source": "absolute_student" if kind == "eru" else restore.get("gaussians_source"),
            "old_gs_head_calls": 0 if kind == "eru" else restore.get("old_gs_head_calls", {}).get("count", 0),
        })
        if not _finite_result(result):
            raise RuntimeError(f"non-finite result for {label}")
        outputs[label] = result
        (model_workspace / "result.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        del model
        torch.cuda.empty_cache()
    protocol_valid = _fingerprints_equal([outputs["both_1420"], outputs["eru_step500"]])
    if not protocol_valid:
        raise RuntimeError("LSM-40 protocol fingerprints differ; refusing comparison")
    comparison = _comparison(outputs["eru_step500"], outputs["both_1420"])
    recon = {
        "both_1420": {key: outputs["both_1420"]["mean"][key] for key in ("psnr", "ssim", "lpips")},
        "eru_step500": {key: outputs["eru_step500"]["mean"][key] for key in ("psnr", "ssim", "lpips")},
        "delta_eru_minus_both": {
            key: outputs["eru_step500"]["mean"][key] - outputs["both_1420"]["mean"][key]
            for key in ("psnr", "ssim", "lpips")
        },
        "pooled_psnr": {
            "both_1420": outputs["both_1420"]["pooled_psnr"],
            "eru_step500": outputs["eru_step500"]["pooled_psnr"],
            "delta_eru_minus_both": outputs["eru_step500"]["pooled_psnr"] - outputs["both_1420"]["pooled_psnr"],
        },
        "reconstruction_preserved": bool(
            outputs["eru_step500"]["mean"]["psnr"] - outputs["both_1420"]["mean"]["psnr"] >= -0.10
            and outputs["eru_step500"]["pooled_psnr"] - outputs["both_1420"]["pooled_psnr"] >= -0.10
        ),
    }
    eru_mean = outputs["eru_step500"]["mean"]
    both_mean = outputs["both_1420"]["mean"]
    eru_pooled = outputs["eru_step500"]["pooled"]
    both_pooled = outputs["both_1420"]["pooled"]
    query_collapse = bool(
        eru_mean.get("effective_query_count", 0.0) < 2.5
        or eru_mean.get("pred_gt", 0.0) < 0.75
        or eru_mean.get("void_ratio", 0.0) - both_mean.get("void_ratio", 0.0) > 0.10
        or eru_mean.get("nonempty_query_count", 0.0) < 0.70 * both_mean.get("nonempty_query_count", 0.0)
    )
    delta_ap50 = eru_mean["ap50"] - both_mean["ap50"]
    delta_pooled = eru_pooled["ap50"] - both_pooled["ap50"]
    if query_collapse:
        # Protocol-valid evaluations with a disqualifying query-occupancy
        # collapse are not A/B successes; keep them in the conservative C
        # bucket rather than labeling a valid run as an invalid experiment.
        classification = "C"
    elif delta_ap50 >= 0.04 and delta_pooled >= 0.04 and comparison["ap50_not_lower_scenes"] >= 24 and recon["reconstruction_preserved"]:
        classification = "A"
    elif delta_ap50 >= 0.02 and delta_pooled > 0 and comparison["ap50_not_lower_scenes"] >= 20 and not query_collapse and recon["reconstruction_preserved"]:
        classification = "B"
    elif delta_ap50 < 0.02 or delta_pooled < 0 or comparison["ap50_not_lower_scenes"] < 20 or eru_mean["mean_best_gt_iou"] < both_mean["mean_best_gt_iou"] and eru_mean["recall_iou50"] < both_mean["recall_iou50"]:
        classification = "C"
    else:
        classification = "D"
    summary = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "manifest": manifest_audit,
        "manifest_sha256": _sha256_file(args.manifest),
        "protocol_fingerprints_valid": protocol_valid,
        "scenes_evaluated": len(entries),
        "both_1420": {"mean": outputs["both_1420"]["mean"], "pooled": outputs["both_1420"]["pooled"]},
        "eru_step500": {"mean": outputs["eru_step500"]["mean"], "pooled": outputs["eru_step500"]["pooled"]},
        "delta_mean_ap50": delta_ap50,
        "delta_pooled_ap50": delta_pooled,
        "delta_best_iou": eru_mean["mean_best_gt_iou"] - both_mean["mean_best_gt_iou"],
        "delta_recall50": eru_mean["recall_iou50"] - both_mean["recall_iou50"],
        "delta_psnr": eru_mean["psnr"] - both_mean["psnr"],
        "comparison": {key: value for key, value in comparison.items() if key != "per_scene"},
        "query_collapse": query_collapse,
        "reconstruction": recon,
        "classification": classification,
        "training_started": False,
        "ttt_started": False,
    }
    files = {
        "summary.json": summary,
        "per_scene_metrics.json": {label: result["per_scene"] for label, result in outputs.items()},
        "comparison_vs_both.json": comparison,
        "protocol_fingerprints.json": {
            "manifest_audit": manifest_audit,
            "manifest_sha256": _sha256_file(args.manifest),
            "both_1420": outputs["both_1420"]["input_fingerprints"],
            "eru_step500": outputs["eru_step500"]["input_fingerprints"],
            "identical": protocol_valid,
        },
        "checkpoint_metadata.json": metadata,
        "query_usage.json": {
            label: {key: result["mean"].get(key) for key in ("active_queries", "active_query_count_mass_gt_0.001", "nonempty_query_count", "effective_query_count", "void_ratio", "pred_gt", "prediction_count_nonempty")}
            for label, result in outputs.items()
        },
        "reconstruction_comparison.json": recon,
    }
    for name, payload in files.items():
        (args.output_dir / name).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (args.output_dir / "eval.log").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
