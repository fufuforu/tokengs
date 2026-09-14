"""Strict paired LSM-40 evaluator for Both, ERU@500, and JointFormation@710.

This isolated entry point reuses the repository's LSM manifest, dataloader,
mask conversion, confidence-ordered AP, and reconstruction metrics.  It has a
one-scene smoke mode for validation and refuses any partial formal run.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_gsi_v2_lsm40_paired import (  # noqa: E402
    _build_both_model,
    _build_lsm_data,
    _evaluate_model,
    _fingerprints_equal,
    _metadata,
    _sha256_file,
)
from scripts.audit_token_eru_dino_joint_formation_24window_paired import (  # noqa: E402
    JOINT_CONFIG,
    _build_model,
    _metadata_for,
    _strict_joint_model_only_restore,
)
from scripts.eval_instance_lsm_protocol import _audit_lsm_manifest  # noqa: E402
from tokengs.utils.metrics import MetricsCalculator  # noqa: E402


BOTH = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8/"
    "checkpoints/model_step_001420.safetensors"
)
ERU = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_scene_hungarian_"
    "resume200_to500_persist_ddp8/checkpoints/model_step_000500.safetensors"
)
JOINT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_v1_ddp8/checkpoints/model_step_000710.safetensors"
)
JOINT_ROOT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_v1_ddp8"
)
MANIFEST = ROOT / "data/scannet_prompt/lsm_instance_eval_manifest.json"
ERU_SHA256 = "cca23e2d24af7a374be6a62f2cf4026645afdf0df49686722cb5277c9e479209"
JOINT_SHA256 = "e848f733fe7f3812db15f143787c5e663475fdd3a0bd01b302c2a047f1a33c2d"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float) + "\n", encoding="utf-8")


def _build_eru(checkpoint: Path, workspace: Path, accelerator: Accelerator):
    from scripts.audit_token_eru_lsm40_paired import _build_eru_model
    return _build_eru_model(checkpoint, workspace, accelerator)


def _build_joint(checkpoint: Path, workspace: Path, accelerator: Accelerator):
    opt, _, model = _build_model(JOINT_CONFIG, workspace, checkpoint)
    metadata_path, _ = _metadata_for(checkpoint, 710)
    restore = _strict_joint_model_only_restore(model, checkpoint, metadata_path)
    schedule = model.set_token_eru_step(710)
    dino_schedule = model.set_token_eru_dino_metric_step(710)
    schedule = {**schedule, **dino_schedule}
    model.to(accelerator.device).eval()
    del opt
    return model, schedule, restore


def _finite_result(result: dict[str, object]) -> bool:
    for section in (result.get("mean", {}), result.get("pooled", {})):
        for value in section.values():
            if isinstance(value, (int, float)) and not np.isfinite(float(value)):
                return False
    return bool(result.get("all_finite", False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_dir.exists():
        unexpected = [entry for entry in args.output_dir.iterdir() if entry.name != "logs"]
        if unexpected:
            raise RuntimeError(f"refusing to overwrite non-empty output: {unexpected}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_audit = _audit_lsm_manifest(str(args.manifest))
    if manifest_audit["scene_count"] != 40:
        raise RuntimeError(f"LSM manifest is not 40 scenes: {manifest_audit}")
    manifest_sha = _sha256_file(args.manifest)
    manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_entries = manifest_payload["scenes"]
    if args.smoke:
        entries = dict(list(all_entries.items())[:1])
    else:
        entries = dict(all_entries)
        if len(entries) != 40:
            raise RuntimeError("formal LSM-40 requires all 40 manifest scenes")
    for checkpoint in (BOTH, ERU, JOINT):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    eru_sha = _sha256_file(ERU)
    joint_sha = _sha256_file(JOINT)
    if eru_sha != ERU_SHA256:
        raise RuntimeError(f"ERU SHA256 mismatch: {eru_sha}")
    if joint_sha != JOINT_SHA256:
        raise RuntimeError(f"Joint710 SHA256 mismatch: {joint_sha}")
    if not (JOINT_ROOT / "status" / "COMPLETE").is_file():
        raise RuntimeError("JointFormation status/COMPLETE is missing")
    metadata = {
        "both_1420": _metadata(BOTH, 1420),
        "eru_500": _metadata(ERU, 500),
        "joint_710": _metadata_for(JOINT, 710)[1],
    }
    _, loader, dataset = _build_lsm_data(args.manifest)
    del dataset
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("LSM smoke/formal requires exactly one visible CUDA device")
    accelerator = Accelerator(mixed_precision="no")
    metrics = MetricsCalculator(device=accelerator.device)
    outputs = {}
    model_specs = (
        ("both_1420", BOTH, "both", 1420),
        ("eru_000500", ERU, "eru", 500),
        ("joint_000710", JOINT, "joint", 710),
    )
    for label, checkpoint, kind, step in model_specs:
        model_workspace = args.output_dir / label
        model_workspace.mkdir(parents=True, exist_ok=True)
        if kind == "both":
            model, schedule, restore = _build_both_model(checkpoint, model_workspace)
        elif kind == "eru":
            model, schedule, restore = _build_eru(checkpoint, model_workspace, accelerator)
        else:
            model, schedule, restore = _build_joint(checkpoint, model_workspace, accelerator)
        result = _evaluate_model(
            label, model, schedule, loader, test_dataset=None,
            manifest_entries=entries, is_instance=True, metrics=metrics,
            metadata=metadata[
                "eru_500" if label == "eru_000500"
                else "joint_710" if label == "joint_000710"
                else label
            ],
            max_scenes=1 if args.smoke else 0,
        )
        result.update({
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "optimizer_step": step,
            "eval_precision": "fp32",
            "p_u_used": False,
            "oracle_used": False,
            "ttt_used": False,
            "native_query_only": True,
            "metric_cluster_diagnostic_only": True,
            "restore": restore,
            "schedule": schedule,
        })
        if not _finite_result(result):
            raise RuntimeError(f"non-finite LSM result for {label}")
        outputs[label] = result
        _write(model_workspace / "result.json", result)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    protocol_valid = _fingerprints_equal(list(outputs.values()))
    if not protocol_valid:
        raise RuntimeError("LSM model input fingerprints differ")
    baseline = outputs["both_1420"]
    comparison = {}
    for label, result in outputs.items():
        if label == "both_1420":
            continue
        per_scene = {}
        for scene, row in result["per_scene"].items():
            base = baseline["per_scene"][scene]
            per_scene[scene] = {
                "delta_ap25": float(row["ap25"] - base["ap25"]),
                "delta_ap50": float(row["ap50"] - base["ap50"]),
                "delta_ap75": float(row["ap75"] - base["ap75"]),
                "delta_best_iou": float(row["mean_best_gt_iou"] - base["mean_best_gt_iou"]),
                "delta_recall50": float(row["recall_iou50"] - base["recall_iou50"]),
                "delta_psnr": float(row["psnr"] - base["psnr"]),
                "delta_ssim": float(row["ssim"] - base["ssim"]),
                "delta_lpips": float(row["lpips"] - base["lpips"]),
            }
        comparison[label] = {
            "per_scene": per_scene,
            "ap50_not_lower_scenes": sum(v["delta_ap50"] >= 0 for v in per_scene.values()),
            "ap50_improved_scenes": sum(v["delta_ap50"] > 0 for v in per_scene.values()),
            "best_iou_improved_scenes": sum(v["delta_best_iou"] > 0 for v in per_scene.values()),
            "recall50_improved_scenes": sum(v["delta_recall50"] > 0 for v in per_scene.values()),
            "query_collapse": bool(
                result["mean"].get("nonempty_query_count", 0.0) < 0.70 * baseline["mean"].get("nonempty_query_count", 0.0)
                or result["mean"].get("effective_query_count", 0.0) < 0.60 * baseline["mean"].get("effective_query_count", 0.0)
                or result["mean"].get("void_ratio", 0.0) > baseline["mean"].get("void_ratio", 0.0) + 0.10
                or not 0.60 <= result["mean"].get("pred_gt", 0.0) <= 2.50
            ),
        }
    reconstruction = {
        label: {
            "mean_psnr": result["mean"]["psnr"],
            "pooled_psnr": result["pooled_psnr"],
            "delta_mean_psnr_vs_both": result["mean"]["psnr"] - baseline["mean"]["psnr"],
            "delta_pooled_psnr_vs_both": result["pooled_psnr"] - baseline["pooled_psnr"],
        }
        for label, result in outputs.items()
    }
    summary = {
        "manifest": manifest_audit,
        "manifest_sha256": manifest_sha,
        "scene_count": len(entries),
        "protocol_fingerprints_valid": protocol_valid,
        "native_query_only": True,
        "p_u_used": False,
        "oracle_used": False,
        "ttt_used": False,
        "eval_precision": "fp32",
        "max_predictions_per_image": 100,
        "min_mask_area": 1,
        "outputs": {key: f"{key}/result.json" for key in outputs},
        "training_started": False,
        "comparison_vs_both": comparison,
        "reconstruction_comparison": reconstruction,
    }
    _write(args.output_dir / "summary.json", summary)
    _write(args.output_dir / "protocol_fingerprints.json", {
        "manifest": manifest_audit,
        "manifest_sha256": manifest_sha,
        "models": {key: value["input_fingerprints"] for key, value in outputs.items()},
        "identical": protocol_valid,
    })
    _write(args.output_dir / "checkpoint_metadata.json", metadata)
    _write(args.output_dir / "per_scene_metrics.json", {
        label: result["per_scene"] for label, result in outputs.items()
    })
    _write(args.output_dir / "comparison_vs_both.json", comparison)
    _write(args.output_dir / "reconstruction_comparison.json", reconstruction)
    _write(args.output_dir / "query_usage.json", {
        label: {
            key: result["mean"].get(key)
            for key in ("active_queries", "nonempty_query_count", "effective_query_count", "void_ratio", "pred_gt")
        }
        for label, result in outputs.items()
    })
    (args.output_dir / "eval.log").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "status").mkdir(exist_ok=True)
    (args.output_dir / "status" / ("SMOKE_COMPLETE" if args.smoke else "COMPLETE")).write_text("ok\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
