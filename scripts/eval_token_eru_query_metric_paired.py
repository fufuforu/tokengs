"""Paired 24-window evaluator for the isolated QMC treatment.

This adapter reuses the existing JointFormation evaluator implementation and
changes only its configuration/checkpoint roster.  It never performs an
optimizer step and never invokes metric clustering or p_u-derived logic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


def _preload_gsplat_before_evaluator_import() -> None:
    so_path = os.environ.get("GSPLAT_PRECOMPILED_SO")
    if not so_path or not os.path.isfile(so_path):
        return
    if "gsplat_cuda" not in sys.modules:
        spec = importlib.util.spec_from_file_location("gsplat_cuda", so_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load cached gsplat extension: {so_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["gsplat_cuda"] = module
    else:
        module = sys.modules["gsplat_cuda"]
    import gsplat

    sys.modules.setdefault("gsplat.csrc", module)
    setattr(gsplat, "csrc", module)


_preload_gsplat_before_evaluator_import()

from scripts import audit_token_eru_dino_joint_formation_24window_paired as base


QMC_CONFIG = (
    "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_local250_query_metric_v1_short200_ddp8"
)
A0_CONFIG = (
    "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_local250_control_short200_ddp8"
)
QMC_ROOT = Path(
    "/space/mawb/tokengs/workspace/"
    "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_local250_query_metric_v1_short200_ddp8"
)
A0_ROOT = Path(
    "/space/mawb/tokengs/workspace/"
    "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_local250_control_short200_ddp8"
)
PARENT = Path(
    "/space/mawb/tokengs/workspace/semantic_v6_absolute_units_true_shared_"
    "token_eru1_dino_metric_joint_formation_j2_ddp8/checkpoints/"
    "model_step_000250.safetensors"
)
OUTPUT_DEFAULT = Path(
    "/space/mawb/tokengs/workspace/token_eru_query_metric_v1_24window_paired_eval"
)


def _checkpoint(root: Path, step: int) -> Path:
    return root / "checkpoints" / f"model_step_{step:06d}.safetensors"


def _scene_rows(result: dict) -> dict[str, dict]:
    return {
        row["scene_id"]: row
        for row in result["aggregate"]["scene_rows"]
    }


def _scene_mean(result: dict, field: str) -> float:
    return float(result["aggregate"]["scene_macro"][field]["mean"])


def _pooled(result: dict, field: str) -> float:
    return float(result["aggregate"]["pooled"][field])


def _query_collapse(result: dict, parent: dict) -> bool:
    scene = result["aggregate"]["scene_macro"]
    base = parent["aggregate"]["scene_macro"]
    return bool(
        _scene_mean(result, "nonempty_queries") < 0.70 * _scene_mean(parent, "nonempty_queries")
        or _scene_mean(result, "effective_queries") < 0.60 * _scene_mean(parent, "effective_queries")
        or _scene_mean(result, "void_ratio") > _scene_mean(parent, "void_ratio") + 0.10
        or _scene_mean(result, "pred_gt") < 0.60
        or _scene_mean(result, "pred_gt") > 2.50
    )


def _scene_deltas(result: dict, reference: dict) -> dict[str, dict[str, float]]:
    ref = _scene_rows(reference)
    out = {}
    fields = (
        "ap", "ap25", "ap50", "ap75", "best_gt_iou", "recall25", "recall50",
        "recall75", "pred_gt", "nonempty_queries", "active_queries", "effective_queries",
        "void_ratio", "psnr", "ssim", "lpips",
    )
    for scene_id, row in _scene_rows(result).items():
        out[scene_id] = {field: float(row[field] - ref[scene_id][field]) for field in fields}
    return out


def _comparison(result: dict, reference: dict) -> dict:
    return {
        "scene_macro_deltas": _scene_deltas(result, reference),
        "delta_scene_macro": {
            field: _scene_mean(result, field) - _scene_mean(reference, field)
            for field in ("ap", "ap25", "ap50", "ap75", "best_gt_iou", "recall25", "recall50", "recall75", "psnr", "ssim", "lpips")
        },
        "delta_pooled": {
            field: _pooled(result, field) - _pooled(reference, field)
            for field in ("ap", "ap25", "ap50", "ap75", "recall25", "recall50", "recall75", "best_gt_iou", "pred_gt")
        },
        "ap50_not_lower_scene_count": int(sum(
            row["ap50"] >= _scene_rows(reference)[scene_id]["ap50"]
            for scene_id, row in _scene_rows(result).items()
        )),
    }


def _bootstrap(values: list[float], *, seed: int = 42, iterations: int = 10000) -> dict:
    samples = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(samples), size=(iterations, len(samples)))
    means = samples[draws].mean(axis=1)
    observed = float(samples.mean())
    return {
        "iterations": iterations,
        "seed": seed,
        "observed_delta": observed,
        "bootstrap_mean": float(means.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "p_delta_gt_zero": float(np.mean(means > 0.0)),
        "scene_deltas": [float(x) for x in samples],
    }


def _sign_test(values: list[float]) -> dict:
    positives = sum(value > 0 for value in values)
    negatives = sum(value < 0 for value in values)
    ties = len(values) - positives - negatives
    n = positives + negatives
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(positives, negatives) + 1)) / (2 ** n)
        p_value = min(1.0, 2.0 * tail)
    return {
        "positive": positives,
        "negative": negatives,
        "tie": ties,
        "non_tie_count": n,
        "exact_two_sided_p_value": float(p_value),
    }


def _write_derived_outputs(output: Path, results: dict[str, dict]) -> dict:
    parent = results["j2_parent_000250"]
    qmc_keys = [f"qmc_{step:06d}" for step in (50, 100, 150, 200)]
    a0_keys = [f"a0_{step:06d}" for step in (50, 100, 150, 200)]
    qmc_vs_parent = {key: _comparison(results[key], parent) for key in qmc_keys}
    qmc_vs_control = {}
    for qmc_key, a0_key in zip(qmc_keys, a0_keys):
        qmc_vs_control[qmc_key] = {
            "control_key": a0_key,
            **_comparison(results[qmc_key], results[a0_key]),
        }
    base._write_json(output / "comparison_vs_parent.json", qmc_vs_parent)
    base._write_json(output / "comparison_vs_control.json", qmc_vs_control)

    eligible = []
    for key in qmc_keys:
        result = results[key]
        eligible.append({
            "key": key,
            "step": int(key.rsplit("_", 1)[1]),
            "numeric_finite": bool(result["numeric_finite"]),
            "ap101_invariant_valid": bool(result["ap101_invariant_valid"]),
            "query_collapse": _query_collapse(result, parent),
            "scene_macro_ap50": _scene_mean(result, "ap50"),
            "pooled_ap50": _pooled(result, "ap50"),
            "best_iou": _scene_mean(result, "best_gt_iou"),
            "recall50": _scene_mean(result, "recall50"),
            "psnr": _scene_mean(result, "psnr"),
        })
    for row in eligible:
        row["eligible"] = bool(
            row["numeric_finite"]
            and row["ap101_invariant_valid"]
            and not row["query_collapse"]
            and row["psnr"] >= _scene_mean(parent, "psnr") - 0.20
        )
    eligible_rows = [row for row in eligible if row["eligible"]]
    eligible_rows.sort(key=lambda row: (-row["scene_macro_ap50"], -row["pooled_ap50"], -row["best_iou"], -row["recall50"], row["step"]))
    control_rows = [
        {"key": key, "step": int(key.rsplit("_", 1)[1]), "scene_macro_ap50": _scene_mean(results[key], "ap50")}
        for key in a0_keys
    ]
    control_rows.sort(key=lambda row: (-row["scene_macro_ap50"], row["step"]))
    selection = {
        "qmc_candidates": eligible,
        "qmc_best": eligible_rows[0] if eligible_rows else None,
        "control_best": control_rows[0] if control_rows else None,
        "ready_for_lsm40": False,
    }
    if eligible_rows:
        best = eligible_rows[0]
        selection["ready_for_lsm40"] = bool(
            best["scene_macro_ap50"] - _scene_mean(parent, "ap50") >= 0.005
            and best["pooled_ap50"] >= _pooled(parent, "ap50")
        )
    base._write_json(output / "checkpoint_selection.json", selection)

    best_key = selection["qmc_best"]["key"] if selection["qmc_best"] else qmc_keys[-1]
    best_step = int(best_key.rsplit("_", 1)[1])
    same_control_key = f"a0_{best_step:06d}"
    best_control_key = selection["control_best"]["key"] if selection["control_best"] else same_control_key
    paired = {
        "qmc_best": best_key,
        "same_step_control": same_control_key,
        "control_best": best_control_key,
        "best_vs_same_step_control": _comparison(results[best_key], results[same_control_key]),
        "best_vs_control_best": _comparison(results[best_key], results[best_control_key]),
    }
    base._write_json(output / "comparison_vs_control_best.json", paired)
    scene_ref = _scene_rows(results[same_control_key])
    scene_qmc = _scene_rows(results[best_key])
    deltas = [float(scene_qmc[s]["ap50"] - scene_ref[s]["ap50"]) for s in sorted(scene_ref)]
    base._write_json(output / "paired_bootstrap.json", {
        "full_vs_same_step_control_ap50": _bootstrap(deltas),
        "full_vs_control_best_ap50": _bootstrap([
            float(scene_qmc[s]["ap50"] - _scene_rows(results[best_control_key])[s]["ap50"])
            for s in sorted(scene_ref)
        ]),
    })
    base._write_json(output / "paired_sign_test.json", {
        "full_vs_same_step_control_ap50": _sign_test(deltas),
        "full_vs_control_best_ap50": _sign_test([
            float(scene_qmc[s]["ap50"] - _scene_rows(results[best_control_key])[s]["ap50"])
            for s in sorted(scene_ref)
        ]),
    })
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--qmc-steps", nargs="+", type=int, default=[50, 100, 150, 200])
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    # Register the validated sm_86 gsplat shared object before importing any
    # model path that could otherwise enter the incompatible JIT fallback.
    base._load_cached_gsplat_extension()
    if output.exists() and any(entry.name != "logs" for entry in output.iterdir()):
        raise RuntimeError(f"refusing non-empty QMC evaluation output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not base.SOURCE.is_file() or base._sha256(base.SOURCE) != base.SOURCE_SHA256:
        raise RuntimeError("protected ERU@500 parent SHA mismatch")
    if args.smoke:
        max_windows = 1 if args.max_windows is None else int(args.max_windows)
        if max_windows != 1:
            raise ValueError("QMC smoke is fixed to one validation window")
        qmc_steps = [int(args.qmc_steps[0])]
        a0_steps = [int(args.qmc_steps[0])]
    else:
        max_windows = 24 if args.max_windows is None else int(args.max_windows)
        if max_windows != 24:
            raise ValueError("formal QMC evaluation requires all 24 records")
        if base._active_joint_job():
            raise RuntimeError("formal QMC evaluation refused while training is active")
        qmc_steps = [50, 100, 150, 200]
        a0_steps = [50, 100, 150, 200]
        if not (QMC_ROOT / "status" / "COMPLETE").is_file():
            raise RuntimeError("QMC formal evaluation requires status/COMPLETE")
        for root, steps in ((QMC_ROOT, qmc_steps), (A0_ROOT, a0_steps)):
            for step in steps:
                checkpoint = _checkpoint(root, step)
                if not checkpoint.is_file():
                    raise RuntimeError(f"missing checkpoint: {checkpoint}")
                if not (checkpoint.parent / f"metadata_step_{step:06d}.json").is_file():
                    raise RuntimeError(f"missing metadata for: {checkpoint}")
                if not (checkpoint.parent / f"step_{step:06d}.complete").is_file():
                    raise RuntimeError(f"missing completion marker for: {checkpoint}")

    # Rebind only the evaluator's config/root constants.  The underlying
    # manifest, dataloader, GT conversion, native AP and aggregation routines
    # remain the existing implementation.
    base.JOINT_CONFIG = QMC_CONFIG
    base.JOINT_ROOT = QMC_ROOT
    # QMC model-only checkpoints are strict overlays on the protected J2
    # local250 parent.  The generic JointFormation evaluator's SOURCE points
    # at the older ERU@500 parent, which is insufficient to construct the
    # QMC/DINO metric namespace before the overlay is applied.
    base.SOURCE = PARENT
    base.FORMAL_OUTPUT = output
    opt, accelerator, _, test_dataset = base._validation_runtime(output / "protocol")
    manifest_info = base._audit_manifest(opt, test_dataset)
    _, _, reference_loader, _ = base._validation_runtime(output / "reference")
    reference_fingerprints = []
    for index, batch in enumerate(reference_loader):
        if index >= max_windows:
            break
        reference_fingerprints.append(base._window_fingerprint(batch, index))
    if len(reference_fingerprints) != max_windows:
        raise RuntimeError("QMC validation fingerprint count mismatch")
    protocol = {
        "manifest": manifest_info,
        "window_fingerprints": reference_fingerprints,
        "window_count": max_windows,
        "native_query_only": True,
        "metric_cluster_diagnostic_only": True,
        "p_u_used": False,
        "oracle_used": False,
        "ttt_used": False,
        "precision": "fp32",
        "max_predictions_per_image": 100,
        "min_mask_area": 1,
        "ap_interpolation": "101-point recall interpolation",
    }
    base._write_json(output / "protocol_fingerprints.json", protocol)
    del opt, accelerator, test_dataset, reference_loader

    results = {}
    results["j2_parent_000250"] = base._evaluate_model(
        base.PARENT_CONFIG,
        PARENT,
        250,
        output,
        reference_fingerprints,
        max_windows,
        manifest_info,
    )
    if A0_ROOT.exists():
        for step in a0_steps:
            results[f"a0_{step:06d}"] = base._evaluate_model(
                A0_CONFIG,
                _checkpoint(A0_ROOT, step),
                step,
                output,
                reference_fingerprints,
                max_windows,
                manifest_info,
            )
    for step in qmc_steps:
        results[f"qmc_{step:06d}"] = base._evaluate_model(
            QMC_CONFIG,
            _checkpoint(QMC_ROOT, step),
            step,
            output,
            reference_fingerprints,
            max_windows,
            manifest_info,
        )
    base._write_json(output / "checkpoint_metadata.json", results)
    base._write_json(
        output / "per_window_metrics.json",
        {key: value["records"] for key, value in results.items()},
    )
    base._write_json(
        output / "per_scene_metrics.json",
        {key: value["aggregate"] for key, value in results.items()},
    )
    selection = _write_derived_outputs(output, results)
    summary = {
        "records": max_windows,
        "unique_scene_windows": 8,
        "held_out_scenes": 8,
        "results": {
            key: value["aggregate"] for key, value in results.items()
        },
        "checkpoint_selection": selection,
        "formal_training_started": False,
        "formal_evaluation_started": True,
    }
    base._write_json(output / "summary.json", summary)
    (output / "status").mkdir(exist_ok=True)
    (output / "status" / ("SMOKE_COMPLETE" if args.smoke else "COMPLETE")).write_text(
        "ok\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
