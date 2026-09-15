"""Paired 24-window evaluator for the isolated QMC treatment.

This adapter reuses the existing JointFormation evaluator implementation and
changes only its configuration/checkpoint roster.  It never performs an
optimizer step and never invokes metric clustering or p_u-derived logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--qmc-steps", nargs="+", type=int, default=[50, 100, 150, 200])
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
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
    summary = {
        "records": max_windows,
        "unique_scene_windows": 8,
        "held_out_scenes": 8,
        "results": {
            key: value["aggregate"] for key, value in results.items()
        },
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
