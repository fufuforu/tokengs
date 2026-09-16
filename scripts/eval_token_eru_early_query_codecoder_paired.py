"""User-run paired 24-window evaluator for EQC-v1.

This is an isolated orchestration layer around the already validated native
JointFormation evaluator.  It does not alter AP, mask conversion, or model
forward semantics.  It is intentionally not called by any preflight script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import audit_token_eru_dino_joint_formation_24window_paired as audit

ROOT = audit.ROOT
PARENT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_"
    "dino_metric_joint_formation_j2_ddp8/checkpoints/model_step_000250.safetensors"
)
PARENT_CONFIG = "semantic_v6_j2_local250_eqc_v1_control_short200_ddp8"
CONFIGS = {
    "control": "semantic_v6_j2_local250_eqc_v1_control_short200_ddp8",
    "eqc": "semantic_v6_j2_local250_eqc_v1_treatment_short200_ddp8",
}
ROOTS = {
    "control": ROOT / "workspace/semantic_v6_j2_local250_eqc_v1_control_short200_ddp8",
    "eqc": ROOT / "workspace/semantic_v6_j2_local250_eqc_v1_treatment_short200_ddp8",
}
STEPS = (50, 100, 150, 200)


def _checkpoint(root: Path, step: int) -> Path:
    return root / "checkpoints" / f"model_step_{step:06d}.safetensors"


def _required_checkpoint(root: Path, step: int) -> tuple[Path, Path]:
    checkpoint = _checkpoint(root, step)
    metadata = checkpoint.parent / f"metadata_step_{step:06d}.json"
    marker = checkpoint.parent / f"step_{step:06d}.complete"
    if not checkpoint.is_file() or not metadata.is_file() or not marker.is_file():
        raise RuntimeError(f"incomplete EQC checkpoint step={step}: {checkpoint}")
    return checkpoint, metadata


def _run_model(config: str, checkpoint: Path, step: int, output: Path, refs, manifest):
    audit.JOINT_CONFIG = config
    return audit._evaluate_model(
        config, checkpoint, step, output, refs, len(refs), manifest
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, default=24)
    args = parser.parse_args()
    if args.max_windows not in (1, 24):
        raise ValueError("only one-window smoke or all 24 records are supported")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty evaluation output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    audit._load_cached_gsplat_extension()
    audit.JOINT_CONFIG = CONFIGS["control"]
    audit.SOURCE = PARENT
    opt, accelerator, _, test_dataset = audit._validation_runtime(output / "protocol")
    manifest = audit._audit_manifest(opt, test_dataset)
    _, _, reference_loader, _ = audit._validation_runtime(output / "reference")
    refs = [audit._window_fingerprint(batch, i) for i, batch in enumerate(reference_loader) if i < args.max_windows]
    if len(refs) != args.max_windows:
        raise RuntimeError(f"validation reference count={len(refs)} expected={args.max_windows}")
    results = {}
    results["j2_parent_local250"] = _run_model(PARENT_CONFIG, PARENT, 250, output / "parent", refs, manifest)
    if args.max_windows == 24:
        for label in ("control", "eqc"):
            for step in STEPS:
                checkpoint, _ = _required_checkpoint(ROOTS[label], step)
                results[f"{label}_step{step:03d}"] = _run_model(CONFIGS[label], checkpoint, step, output / f"{label}_{step:03d}", refs, manifest)
    audit._write_json(output / "protocol_fingerprints.json", {
        "manifest": manifest,
        "records": args.max_windows,
        "held_out_scenes": 8,
        "window_fingerprints": refs,
        "native_query_only": True,
        "p_u_used": False,
        "metric_cluster_formal": False,
        "oracle_used": False,
        "ttt_used": False,
        "precision": "fp32",
    })
    audit._write_json(output / "per_window_metrics.json", {
        key: {"records": value["records"], "aggregate": value["aggregate"]}
        for key, value in results.items()
    })
    audit._write_json(output / "per_scene_metrics.json", {
        key: {"summary": value["aggregate"]["scene_macro"], "scenes": value["aggregate"]["scene_rows"]}
        for key, value in results.items()
    })
    audit._write_json(output / "checkpoint_metadata.json", {
        key: {"checkpoint": value["checkpoint"], "metadata": value["metadata"], "schedule": value["schedule"]}
        for key, value in results.items()
    })
    audit._write_json(output / "summary.json", {
        "records_description": "24 records / 8 unique scene-windows / 8 held-out scenes",
        "results": {
            key: {
                "aggregate": value["aggregate"],
                "numeric_finite": value["numeric_finite"],
                "ap101_invariant_valid": value["ap101_invariant_valid"],
                "native_query_only": value["native_query_only"],
                "p_u_used": value["p_u_used"],
                "metric_cluster_formal": False,
            }
            for key, value in results.items()
        },
        "formal_evaluation_started": args.max_windows == 24,
        "training_started": False,
    })
    if args.max_windows == 24:
        (output / "status").mkdir(exist_ok=True)
        (output / "status" / "COMPLETE").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
