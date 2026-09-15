"""Formal 24-window paired evaluator for the A0/A1 3D-anchor experiment.

This is a preparation-only entry point in the current task.  The launcher
refuses to run until both matched short200 workspaces are complete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts import audit_token_eru_dino_joint_formation_24window_paired as audit

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_ddp8/checkpoints/model_step_000250.safetensors"
)
PARENT_META_STEP = 250
PARENT_CONFIG = "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_local250_3d_anchor_v1_short200_ddp8"
CONFIGS = {
    "control": "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_local250_control_short200_ddp8",
    "anchor": "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_local250_3d_anchor_v1_short200_ddp8",
}
ROOTS = {
    "control": ROOT / "workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_local250_control_short200_ddp8",
    "anchor": ROOT / "workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_local250_3d_anchor_v1_short200_ddp8",
}
STEPS = (50, 100, 150, 200)


def _configure_module(config_name: str, root: Path, parent_config: str) -> None:
    audit.JOINT_CONFIG = config_name
    audit.JOINT_ROOT = root
    audit.SOURCE = PARENT
    audit.PARENT_CONFIG = parent_config


def _check_workspace(root: Path) -> None:
    for step in STEPS:
        checkpoint = root / "checkpoints" / f"model_step_{step:06d}.safetensors"
        metadata = root / "checkpoints" / f"metadata_step_{step:06d}.json"
        marker = root / "checkpoints" / f"step_{step:06d}.complete"
        if not (checkpoint.is_file() and metadata.is_file() and marker.is_file()):
            raise RuntimeError(f"incomplete A0/A1 checkpoint step {step}: {root}")
    if not (root / "status" / "COMPLETE").is_file():
        raise RuntimeError(f"missing status/COMPLETE: {root}")


def _run_one(label: str, output: Path) -> dict:
    config = CONFIGS[label]
    root = ROOTS[label]
    other = CONFIGS["anchor" if label == "control" else "control"]
    _configure_module(config, root, other)
    output.mkdir(parents=True, exist_ok=True)
    opt, accelerator, _, test_dataset = audit._validation_runtime(output / "protocol")
    manifest = audit._audit_manifest(opt, test_dataset)
    _, _, reference_loader, _ = audit._validation_runtime(output / "reference")
    reference = [audit._window_fingerprint(batch, index) for index, batch in enumerate(reference_loader)]
    if len(reference) != 24:
        raise RuntimeError(f"expected 24 validation records, got {len(reference)}")
    results = {
        "j2_parent_local250": audit._evaluate_model(
            other, PARENT, PARENT_META_STEP, output, reference, 24, manifest
        )
    }
    for step in STEPS:
        results[f"{label}_step{step:03d}"] = audit._evaluate_model(
            config,
            root / "checkpoints" / f"model_step_{step:06d}.safetensors",
            step,
            output,
            reference,
            24,
            manifest,
        )
    audit._write_json(output / "protocol_fingerprints.json", {
        "manifest": manifest,
        "records": 24,
        "held_out_scenes": 8,
        "native_query_only": True,
        "metric_cluster_diagnostic_only": True,
        "p_u_used": False,
        "oracle_used": False,
        "ttt_used": False,
        "precision": "fp32",
        "window_fingerprints": reference,
    })
    audit._write_json(output / "per_window_metrics.json", {
        key: {"records": value["records"], "aggregate": value["aggregate"]}
        for key, value in results.items()
    })
    audit._write_json(output / "per_scene_metrics.json", {
        key: {"summary": value["aggregate"]["scene_macro"], "scenes": value["aggregate"]["scene_rows"]}
        for key, value in results.items()
    })
    parent_scene = results["j2_parent_local250"]["aggregate"]["scene_macro"]
    comparisons = {}
    for key, value in results.items():
        scene = value["aggregate"]["scene_macro"]
        comparisons[key] = {
            "delta_scene_macro_ap50_vs_parent": scene["ap50"]["mean"] - parent_scene["ap50"]["mean"],
            "delta_pooled_ap50_vs_parent": value["aggregate"]["pooled"]["ap50"] - results["j2_parent_local250"]["aggregate"]["pooled"]["ap50"],
            "scene_rows": value["aggregate"]["scene_rows"],
        }
    audit._write_json(output / "comparison_vs_j2_parent.json", comparisons)
    audit._write_json(output / "summary.json", {
        "label": label,
        "records_description": "24 records / 8 unique scene-windows / 8 held-out scenes",
        "results": results,
        "formal_training_started": False,
        "formal_evaluation_started": True,
    })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty paired evaluation output: {output}")
    if not PARENT.is_file():
        raise RuntimeError(f"missing J2 local250 parent: {PARENT}")
    for root in ROOTS.values():
        _check_workspace(root)
    results = {
        "control": _run_one("control", output / "control"),
        "anchor": _run_one("anchor", output / "anchor"),
    }
    (output / "status").mkdir(parents=True, exist_ok=True)
    (output / "status" / "COMPLETE").write_text("ok\n", encoding="utf-8")
    (output / "paired_summary.json").write_text(json.dumps({"results": results}, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
