"""Read-only paired re-evaluation with image-local target-view AP IDs.

This adapter selects the already existing TokenGS checkpoints and delegates
model construction, strict restore, rendering, post-processing, and AP to the
validated isolated evaluators.  It only changes the image identity used by
the AP bookkeeping through the shared helper in ``tokengs.utils.instance_ap``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# The repository's validated 240 compatibility shim supplies the local
# fused_ssim_cuda module.  Add it before importing model/evaluator modules;
# those imports load rendering dependencies eagerly.
sys.path.insert(0, str(ROOT / "240_shims"))
sys.path.insert(0, str(ROOT))

from scripts import audit_gsi_v2_lsm40_paired as lsm_base  # noqa: E402
from scripts import audit_token_eru_dino_joint_formation_24window_paired as heldout  # noqa: E402
from tokengs.utils.metrics import MetricsCalculator  # noqa: E402


LSM_MANIFEST = ROOT / "data/scannet_prompt/lsm_instance_eval_manifest.json"
J2_CONFIG = "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_ddp8"
JOINT_CONFIG = "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_v1_ddp8"
EQC_CONTROL_CONFIG = "semantic_v6_j2_local250_eqc_v1_control_short200_ddp8"
J2_250 = ROOT / "workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_ddp8/checkpoints/model_step_000250.safetensors"
J2_710 = ROOT / "workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_ddp8/checkpoints/model_step_000710.safetensors"
JOINT_710 = ROOT / "workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_v1_ddp8/checkpoints/model_step_000710.safetensors"
EQC_200 = ROOT / "workspace/semantic_v6_j2_local250_eqc_v1_control_short200_ddp8/checkpoints/model_step_000200.safetensors"
J2_250_SHA = "61debbf5f55b76eea16012aea00fe19291309177bfdc6dd0420f6d1cb58d8cba"
J2_710_SHA = "5d846c745b96a43230d8488e9b394eea72630e6f9c3769e37fde22e82532b476"
JOINT_710_SHA = "e848f733fe7f3812db15f143787c5e663475fdd3a0bd01b302c2a047f1a33c2d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: object) -> None:
    def default(item: object):
        if isinstance(item, (np.floating, np.integer)):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if torch.is_tensor(item):
            return item.detach().cpu().tolist()
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=default) + "\n", encoding="utf-8")


def _check_checkpoint(label: str, checkpoint: Path, step: int, expected_sha: str | None) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{label}: {checkpoint}")
    metadata = checkpoint.parent / f"metadata_step_{step:06d}.json"
    marker = checkpoint.parent / f"step_{step:06d}.complete"
    if not metadata.is_file() or not marker.is_file():
        raise RuntimeError(f"{label}: incomplete checkpoint {checkpoint}")
    observed_sha = _sha256(checkpoint)
    if expected_sha is not None and observed_sha != expected_sha:
        raise RuntimeError(f"{label}: SHA mismatch {observed_sha} != {expected_sha}")
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    observed_step = payload.get("optimizer_step", payload.get("step"))
    if int(observed_step) != step:
        raise RuntimeError(f"{label}: metadata step {observed_step} != {step}")
    return {
        "label": label,
        "path": str(checkpoint.resolve()),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": observed_sha,
        "metadata_path": str(metadata.resolve()),
        "complete_marker": str(marker.resolve()),
        "metadata": payload,
    }


def _configure_audit(config: str, checkpoint: Path) -> None:
    heldout.JOINT_CONFIG = config
    heldout.SOURCE = checkpoint


def _build_token_model(config: str, checkpoint: Path, step: int, output: Path):
    _configure_audit(config, checkpoint)
    opt, accelerator, model = heldout._build_model(config, output, checkpoint)
    metadata_path, metadata = heldout._metadata_for(checkpoint, step)
    restore = heldout._strict_joint_model_only_restore(model, checkpoint, metadata_path)
    schedule = model.set_token_eru_step(step)
    dino_schedule = model.set_token_eru_dino_metric_step(step)
    model.to(accelerator.device).eval()
    return model, accelerator, {**schedule, **dino_schedule}, restore, metadata


def _run_lsm_model(spec: dict, output: Path, loader, entries, metrics, *, max_scenes: int = 0):
    model, accelerator, schedule, restore, metadata = _build_token_model(
        spec["config"], spec["checkpoint"], spec["step"], output / spec["label"]
    )
    result = lsm_base._evaluate_model(
        spec["label"],
        model,
        schedule,
        loader,
        test_dataset=None,
        manifest_entries=entries,
        is_instance=True,
        metrics=metrics,
        max_scenes=max_scenes,
        metadata={
            "path": str((spec["checkpoint"].parent / f"metadata_step_{spec['step']:06d}.json").resolve()),
            "optimizer_step": spec["step"],
            "checkpoint_sha256": restore["sha256"],
            "payload": metadata,
        },
    )
    result.update({
        "checkpoint": restore,
        "schedule": schedule,
        "native_query_only": True,
        "p_u_used": False,
        "metric_cluster_formal": False,
        "oracle_used": False,
        "ttt_used": False,
        "eval_precision": "fp32",
    })
    _write(output / spec["label"] / "result.json", result)
    del model, accelerator
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _run_24_model(spec: dict, output: Path, refs: list[dict], manifest: dict):
    _configure_audit(spec["config"], spec["checkpoint"])
    result = heldout._evaluate_model(
        spec["config"],
        spec["checkpoint"],
        spec["step"],
        output,
        refs,
        len(refs),
        manifest,
    )
    return result


def _flatten_target_views(results: dict[str, dict], *, lsm: bool) -> dict[str, list[dict]]:
    flattened = {}
    for label, result in results.items():
        rows = []
        if lsm:
            for scene_id, scene_row in result["per_scene"].items():
                for row in scene_row.get("per_target_view", []):
                    rows.append({"model": label, "scene_id": scene_id, **row})
        else:
            for record in result["records"]:
                for row in record.get("native", {}).get("per_view", []):
                    rows.append({
                        "model": label,
                        "scene_id": record["fingerprint"]["scene_id"],
                        **row,
                    })
        flattened[label] = rows
    return flattened


def _run_lsm(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    heldout._load_cached_gsplat_extension()
    manifest_audit = lsm_base._audit_lsm_manifest(str(LSM_MANIFEST))
    payload = json.loads(LSM_MANIFEST.read_text(encoding="utf-8"))
    entries = payload["scenes"]
    if len(entries) != 40 or _sha256(LSM_MANIFEST) != "823744228c249e0fd713813e7eb707e798e7c4e112603444e1dc3aac9417035d":
        raise RuntimeError("LSM manifest audit failed")
    specs = [
        {"label": "joint_formation_000710", "config": JOINT_CONFIG, "checkpoint": JOINT_710, "step": 710},
        {"label": "j2_local250", "config": J2_CONFIG, "checkpoint": J2_250, "step": 250},
        {"label": "j2_local710", "config": J2_CONFIG, "checkpoint": J2_710, "step": 710},
        {"label": "eqc_control_000200", "config": EQC_CONTROL_CONFIG, "checkpoint": EQC_200, "step": 200},
    ]
    expected = {"j2_local250": J2_250_SHA, "j2_local710": J2_710_SHA, "joint_formation_000710": JOINT_710_SHA}
    checkpoint_info = [_check_checkpoint(s["label"], s["checkpoint"], s["step"], expected.get(s["label"])) for s in specs]
    _, loader, _ = lsm_base._build_lsm_data(LSM_MANIFEST)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("LSM re-evaluation requires one visible CUDA device")
    metrics = MetricsCalculator(device=torch.device("cuda"))
    # Keep the complete manifest available to the per-scene loader.  The
    # shared evaluator limits iteration by max_scenes for smoke; slicing the
    # manifest itself would reject the loader's deterministic dataset order.
    results = {}
    for spec in specs:
        result = _run_lsm_model(
            spec, output, loader, entries, metrics,
            max_scenes=1 if args.smoke else 0,
        )
        results[spec["label"]] = result
    if not args.smoke:
        _write(output / "per_scene_metrics.json", {k: v["per_scene"] for k, v in results.items()})
        _write(output / "per_target_view_metrics.json", _flatten_target_views(results, lsm=True))
        _write(output / "results_table.json", {
            k: {"mean": v["mean"], "pooled": v["pooled"], "all_finite": v["all_finite"]}
            for k, v in results.items()
        })
    _write(output / "protocol_fingerprints.json", {
        "protocol_name": "TokenGS posed ScanNet-LSM40 per-target-view AP protocol v1",
        "manifest": manifest_audit,
        "instance_ap_image_identity": "per_target_view_v1",
        "cross_target_view_matching": False,
        "camera_source": "ScanNet .sens intrinsics/poses",
        "shared_colmap_cameras": False,
        "native_query_only": True,
        "p_u_used": False,
        "metric_cluster_formal": False,
        "oracle_used": False,
        "ttt_used": False,
        "fp32": True,
        "models": {k: v["input_fingerprints"] for k, v in results.items()},
        "fingerprints_match": lsm_base._fingerprints_equal(list(results.values())),
    })
    _write(output / "checkpoint_metadata.json", {k: v["checkpoint_metadata"] for k, v in results.items()})
    _write(output / "summary.json", {
        "protocol_name": "TokenGS posed ScanNet-LSM40 per-target-view AP protocol v1",
        "scene_count": len(entries),
        "models": {k: {"mean": v["mean"], "pooled": v["pooled"], "numeric_finite": v["all_finite"], "image_identity": v["instance_ap_image_identity"], "cross_target_view_matching": v["cross_target_view_matching"]} for k, v in results.items()},
        "checkpoint_info": checkpoint_info,
        "smoke": bool(args.smoke),
        "training_started": False,
        "optimizer_step_executed": False,
    })
    if not args.smoke:
        (output / "status").mkdir(exist_ok=True)
        (output / "status" / "COMPLETE").write_text("ok\n", encoding="utf-8")


def _run_24(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    heldout._load_cached_gsplat_extension()
    heldout.JOINT_CONFIG = J2_CONFIG
    heldout.SOURCE = J2_250
    opt, accelerator, _, test_dataset = heldout._validation_runtime(output / "protocol")
    manifest = heldout._audit_manifest(opt, test_dataset)
    _, _, reference_loader, _ = heldout._validation_runtime(output / "reference")
    refs = [heldout._window_fingerprint(batch, i) for i, batch in enumerate(reference_loader)]
    if len(refs) != 24:
        raise RuntimeError(f"expected 24 validation records, got {len(refs)}")
    specs = [{"label": "j2_local250", "config": J2_CONFIG, "checkpoint": J2_250, "step": 250}]
    for step in (50, 100, 150, 200):
        checkpoint = ROOT / f"workspace/semantic_v6_j2_local250_eqc_v1_control_short200_ddp8/checkpoints/model_step_{step:06d}.safetensors"
        _check_checkpoint(f"eqc_control_{step}", checkpoint, step, None)
        specs.append({"label": f"eqc_control_{step:03d}", "config": EQC_CONTROL_CONFIG, "checkpoint": checkpoint, "step": step})
    checkpoint_info = [_check_checkpoint(s["label"], s["checkpoint"], s["step"], J2_250_SHA if s["label"] == "j2_local250" else None) for s in specs]
    del opt, accelerator, test_dataset, reference_loader
    results = {}
    for spec in specs:
        _configure_audit(spec["config"], spec["checkpoint"])
        results[spec["label"]] = heldout._evaluate_model(spec["config"], spec["checkpoint"], spec["step"], output, refs, 24, manifest)
    _write(output / "protocol_fingerprints.json", {
        "protocol_name": "TokenGS posed ScanNet validation per-target-view AP protocol v1",
        "manifest": manifest,
        "records": 24,
        "instance_ap_image_identity": "per_target_view_v1",
        "cross_target_view_matching": False,
        "native_query_only": True,
        "p_u_used": False,
        "metric_cluster_formal": False,
        "oracle_used": False,
        "ttt_used": False,
        "precision": "fp32",
        "models": {k: v["records"][0]["fingerprint"] if v["records"] else None for k, v in results.items()},
    })
    _write(output / "checkpoint_metadata.json", {k: v["checkpoint"] for k, v in results.items()})
    _write(output / "per_window_metrics.json", {k: v["records"] for k, v in results.items()})
    _write(output / "per_scene_metrics.json", {k: v["aggregate"] for k, v in results.items()})
    _write(output / "per_target_view_metrics.json", _flatten_target_views(results, lsm=False))
    _write(output / "results_table.json", {k: {"aggregate": v["aggregate"], "numeric_finite": v["numeric_finite"], "ap101_invariant_valid": v["ap101_invariant_valid"]} for k, v in results.items()})
    _write(output / "summary.json", {
        "protocol_name": "TokenGS posed ScanNet validation per-target-view AP protocol v1",
        "records": 24,
        "models": {k: {"aggregate": v["aggregate"], "numeric_finite": v["numeric_finite"], "ap101_invariant_valid": v["ap101_invariant_valid"]} for k, v in results.items()},
        "checkpoint_info": checkpoint_info,
        "training_started": False,
        "optimizer_step_executed": False,
    })
    (output / "status").mkdir(exist_ok=True)
    (output / "status" / "COMPLETE").write_text("ok\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("lsm40", "validation24"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.protocol == "lsm40":
        _run_lsm(args)
    else:
        if args.smoke:
            raise ValueError("validation24 smoke is not supported by this formal re-evaluator")
        _run_24(args)


if __name__ == "__main__":
    main()
