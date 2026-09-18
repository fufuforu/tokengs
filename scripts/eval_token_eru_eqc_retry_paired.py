"""Prepared per-target-view paired evaluator for the EQC retry workspace.

It is intentionally guarded by the v2 COMPLETE marker and is not invoked by
any preflight or diagnostic command.
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
CONTROL_CONFIG = PARENT_CONFIG
EQC_CONFIG = "semantic_v6_j2_local250_eqc_v1_treatment_short200_ddp8"
CONTROL_ROOT = ROOT / "workspace/semantic_v6_j2_local250_eqc_v1_control_short200_ddp8"
EQC_ROOT = ROOT / "workspace/semantic_v6_j2_local250_eqc_v1_treatment_short200_ddp8_v2"
STEPS = (50, 100, 150, 200)


def checkpoint(root: Path, step: int) -> tuple[Path, Path]:
    model = root / "checkpoints" / f"model_step_{step:06d}.safetensors"
    metadata = root / "checkpoints" / f"metadata_step_{step:06d}.json"
    marker = root / "checkpoints" / f"step_{step:06d}.complete"
    if not (model.is_file() and metadata.is_file() and marker.is_file()):
        raise RuntimeError(f"incomplete checkpoint step={step}: {model}")
    return model, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty evaluation output: {output}")
    if not (EQC_ROOT / "status" / "COMPLETE").is_file():
        raise RuntimeError("EQC retry is incomplete; refusing formal evaluation")
    output.mkdir(parents=True, exist_ok=True)
    audit._load_cached_gsplat_extension()
    audit.JOINT_CONFIG = CONTROL_CONFIG
    audit.SOURCE = PARENT
    opt, accelerator, _, test_dataset = audit._validation_runtime(output / "protocol")
    manifest = audit._audit_manifest(opt, test_dataset)
    _, _, reference_loader, _ = audit._validation_runtime(output / "reference")
    refs = [audit._window_fingerprint(batch, i) for i, batch in enumerate(reference_loader)]
    if len(refs) != 24:
        raise RuntimeError(f"expected 24 validation records, got {len(refs)}")
    results = {}

    def run(label: str, config: str, model_path: Path, step: int, folder: str) -> None:
        audit.JOINT_CONFIG = config
        results[label] = audit._evaluate_model(
            config, model_path, step, output / folder, refs, len(refs), manifest
        )

    run("j2_parent_local250", PARENT_CONFIG, PARENT, 250, "j2_parent_local250")
    for step in STEPS:
        model, _ = checkpoint(CONTROL_ROOT, step)
        run(f"eqc_e0_control_step{step:03d}", CONTROL_CONFIG, model, step, f"eqc_e0_control_{step:03d}")
    for step in STEPS:
        model, _ = checkpoint(EQC_ROOT, step)
        run(f"eqc_e1_treatment_v2_step{step:03d}", EQC_CONFIG, model, step, f"eqc_e1_retry_{step:03d}")

    audit._write_json(output / "protocol_fingerprints.json", {
        "instance_ap_image_identity": "per_target_view_v1",
        "cross_target_view_matching": False,
        "records_description": "24 records / 8 unique scene-windows / 8 held-out scenes",
        "records": 24, "held_out_scenes": 8, "window_fingerprints": refs,
        "native_query_only": True, "p_u_used": False, "metric_cluster_formal": False,
        "oracle_used": False, "ttt_used": False, "precision": "fp32",
    })
    audit._write_json(output / "per_window_metrics.json", {k: {"records": v["records"], "aggregate": v["aggregate"]} for k, v in results.items()})
    audit._write_json(output / "per_scene_metrics.json", {k: {"summary": v["aggregate"]["scene_macro"], "scenes": v["aggregate"]["scene_rows"]} for k, v in results.items()})
    audit._write_json(output / "checkpoint_metadata.json", {k: {"checkpoint": v["checkpoint"], "metadata": v["metadata"], "schedule": v["schedule"]} for k, v in results.items()})
    audit._write_json(output / "summary.json", {
        "records_description": "24 records / 8 unique scene-windows / 8 held-out scenes",
        "results": {k: {"aggregate": v["aggregate"], "numeric_finite": v["numeric_finite"], "ap101_invariant_valid": v["ap101_invariant_valid"], "native_query_only": True, "p_u_used": False, "metric_cluster_formal": False} for k, v in results.items()},
        "formal_evaluation_started": True, "training_started": False,
    })
    (output / "status").mkdir(exist_ok=True)
    (output / "status" / "COMPLETE").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
