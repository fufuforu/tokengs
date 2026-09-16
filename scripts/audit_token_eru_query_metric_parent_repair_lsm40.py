"""Repair and verify the QMC parent evaluator, then run paired LSM-40.

This is deliberately isolated from the existing evaluators.  The important
detail is that the J2 parent is built with the J2 preset, while A0/QMC
model-only checkpoints are overlaid on that same J2 parent.  No optimizer or
training path is imported or invoked.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the already validated local extension before importing a model module.
from scripts import eval_token_eru_query_metric_paired as qmc_old  # noqa: E402

qmc_old._preload_gsplat_before_evaluator_import()
from scripts import audit_token_eru_dino_joint_formation_24window_paired as v24  # noqa: E402
from scripts.audit_gsi_v2_lsm40_paired import (  # noqa: E402
    _LocalAccelerator,
    _build_both_model,
    _build_lsm_data,
    _evaluate_model as evaluate_lsm_model,
    _manifest_fingerprints,
    _metadata as lsm_metadata,
    _sha256_file,
)
from scripts.eval_instance_lsm_protocol import _audit_lsm_manifest  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import (  # noqa: E402
    configure_joint_formation_trainability,
    load_model_checkpoint,
)
from tokengs.utils.metrics import MetricsCalculator  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from safetensors.torch import load_file  # noqa: E402


J2_CONFIG = (
    "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_ddp8"
)
QMC_CONFIG = (
    "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_local250_query_metric_v1_short200_ddp8"
)
A0_CONFIG = (
    "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_local250_control_short200_ddp8"
)

PARENT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_ddp8/checkpoints/model_step_000250.safetensors"
)
A0_ROOT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_local250_control_short200_ddp8"
)
QMC_ROOT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_local250_query_metric_v1_short200_ddp8"
)
PARENT_SHA = "61debbf5f55b76eea16012aea00fe19291309177bfdc6dd0420f6d1cb58d8cba"
LSM_MANIFEST = ROOT / "data/scannet_prompt/lsm_instance_eval_manifest.json"
LSM_MANIFEST_SHA = "823744228c249e0fd713813e7eb707e798e7c4e112603444e1dc3aac9417035d"
PARENT_REPAIR_OUTPUT = ROOT / "workspace/token_eru_query_metric_v1_parent_repair_24window_v1"
VALIDATION_MANIFEST_SHA = "90a7a66ccbb10b7ca6334b262b70f94e9c3d8f33d2f60609dbd54be7bc7fde7d"
PARENT24_EXPECTED = {
    "scene_ap50": (0.385338, 0.001),
    "pooled_ap50": (0.260777, 0.001),
    "best_iou": (0.42247, 0.001),
    "recall50": (0.414254, 0.001),
    "psnr": (20.939714, 0.001),
}


def sha256_file(path: Path) -> str:
    return _sha256_file(path)


def write_json(path: Path, payload: Any) -> None:
    def default(value: Any):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        raise TypeError(type(value).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=default), encoding="utf-8")


def tensor_hash(value: Any) -> str:
    digest = hashlib.sha256()
    if torch.is_tensor(value):
        value = value.detach().float().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    elif value is None:
        digest.update(b"<none>")
    else:
        digest.update(repr(value).encode())
    return digest.hexdigest()


def parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _state_restore_audit(model: torch.nn.Module, state: dict[str, torch.Tensor], kind: str) -> dict[str, Any]:
    runtime = model.state_dict()
    active_prefixes = (
        "gs_tokens", "gs_tokens_dynamic", "enc_dec_backbone.decoder_blocks.",
        "absolute_gs_head.", "tsh_instance_head.", "token_eru_decoder.",
        "token_eru_unit_formation.", "token_eru_dino_encoder.unit_projector.",
        "token_eru_dino_fusion.", "token_eru_metric_head.",
        "token_eru_3d_anchor.", "token_eru_query_metric_coupling.",
    )
    active_runtime = {
        key for key in runtime
        if key == "gs_tokens" or key == "gs_tokens_dynamic" or key.startswith(active_prefixes[2:])
    }
    active_state = {
        key for key in state
        if key == "gs_tokens" or key == "gs_tokens_dynamic" or key.startswith(active_prefixes[2:])
    }
    mismatches = {
        key: {"expected": list(runtime[key].shape), "actual": list(state[key].shape)}
        for key in active_runtime & active_state
        if tuple(runtime[key].shape) != tuple(state[key].shape)
    }
    return {
        "kind": kind,
        "state_dict_key_count": len(state),
        "active_runtime_key_count": len(active_runtime),
        "active_checkpoint_key_count": len(active_state),
        "missing_keys": sorted(active_runtime - active_state),
        "unexpected_keys": sorted(active_state - active_runtime),
        "shape_mismatch": mismatches,
        "strict": True,
        "all_parameters_finite": all(bool(torch.isfinite(p).all()) for p in model.parameters()),
        "dino_checkpoint_keys": sorted(key for key in state if "dino_extractor" in key.lower() or "_dino_model" in key.lower()),
    }


def _build_eru_variant(config_name: str, checkpoint: Path, output_dir: Path, *, overlay: bool, local_step: int) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Build every ERU variant from the same J2 parent configuration."""
    base = config_defaults[config_name]
    opt = dataclasses.replace(
        base,
        resume=str(PARENT),
        workspace=str(output_dir / "runtime"),
        evaluating=True,
        num_workers=0,
        batch_size=1,
        tsh_ddp8=False,
        eval_before_training=False,
        use_wandb=False,
    )
    accelerator = Accelerator(mixed_precision="no")
    model = model_registry[opt.model_type](opt)
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        raise RuntimeError(f"J2 parent was not strictly restored for {config_name}")
    configure_joint_formation_trainability(model, opt)
    if overlay:
        meta_path = checkpoint.parent / f"metadata_step_{local_step:06d}.json"
        restore = v24._strict_joint_model_only_restore(model, checkpoint, meta_path)
    else:
        state = load_file(str(checkpoint), device="cpu")
        restore = _state_restore_audit(model, state, "j2_parent_active_namespace")
        restore.update({"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)})
    model.to(accelerator.device).eval()
    effective_step = 960 if not overlay else 960 + local_step
    schedule = model.set_token_eru_step(effective_step)
    dino_schedule = model.set_token_eru_dino_metric_step(effective_step)
    if hasattr(model, "set_token_eru_query_metric_step"):
        model.set_token_eru_query_metric_step(local_step if overlay and config_name == QMC_CONFIG else 0)
    schedule = {**schedule, **dino_schedule, "effective_step": effective_step, "local_step": local_step}
    if config_name == QMC_CONFIG:
        schedule["query_metric_gate"] = float(model.token_eru_query_metric_gate(local_step, opt))
    restore["parent_config"] = J2_CONFIG
    restore["config_name"] = config_name
    restore["parameter_hash"] = parameter_hash(model)
    restore["dino_in_checkpoint"] = bool(restore.get("dino_checkpoint_keys"))
    return model, schedule, restore


def _write_root_cause(output: Path) -> dict[str, Any]:
    old_script = ROOT / "scripts/audit_token_eru_dino_joint_formation_24window_paired.py"
    old_qmc = ROOT / "scripts/eval_token_eru_query_metric_paired.py"
    historic = ROOT / "workspace/token_eru_dino_joint_formation_j2_24window_paired_eval_v16"
    report = {
        "root_cause": "QMC v6 evaluated the J2 parent checkpoint through the old ERU parent preset. The reconstruction namespace therefore remained compatible, but the instance/DINO metric construction and schedule metadata were not the J2 parent construction.",
        "old_qmc_evaluator": str(old_qmc.resolve()),
        "old_parent_config_used_by_v6": "semantic_v6_absolute_units_true_shared_token_eru1_scene_hungarian_short200_ddp8",
        "old_parent_source": str(PARENT.resolve()),
        "correct_parent_config": J2_CONFIG,
        "correct_parent_metadata": str((PARENT.parent / "metadata_step_000250.json").resolve()),
        "historic_correct_24window_output": str((historic / "summary.json").resolve()),
        "historic_correct_parent_scene_ap50": 0.3853383689244111,
        "historic_correct_parent_pooled_ap50": 0.26077744069665976,
        "historic_correct_parent_best_iou": 0.4224843326296648,
        "historic_correct_parent_recall50": 0.4142539999682857,
        "historic_correct_parent_psnr": 20.93967056274414,
        "v6_wrong_parent_scene_ap50": 0.2355568474,
        "v6_wrong_parent_pooled_ap50": 0.1380815394,
        "v6_wrong_parent_best_iou": 0.3319532165,
        "v6_wrong_parent_recall50": 0.2634593085,
        "v6_wrong_parent_psnr": 20.9397144318,
        "why_rgb_matches": "The wrong and correct paths load the same reconstruction/absolute-GS parameters; the disagreement is in the native instance construction/restoration path, so RGB and Gaussian hashes can remain equal while logits and masks differ.",
        "old_script_source_sha256": sha256_file(old_script),
        "old_qmc_script_source_sha256": sha256_file(old_qmc),
    }
    write_json(output / "parent_root_cause.json", report)
    return report


def _validation_protocol(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    opt, accelerator, _, dataset = v24._validation_runtime(output / "protocol")
    manifest = v24._audit_manifest(opt, dataset)
    if manifest["sha256"] != VALIDATION_MANIFEST_SHA:
        raise RuntimeError("validation manifest SHA mismatch")
    _, _, loader, _ = v24._validation_runtime(output / "reference")
    refs = [v24._window_fingerprint(batch, i) for i, batch in enumerate(loader)]
    if len(refs) != 24:
        raise RuntimeError(f"expected 24 validation records, got {len(refs)}")
    return manifest, refs


def _parent_24(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, refs = _validation_protocol(output)
    # Rebind only this imported evaluator module. Its default path is never
    # changed for other evaluators or configs.
    v24.SOURCE = PARENT
    v24.PARENT_CONFIG = J2_CONFIG
    v24.JOINT_CONFIG = QMC_CONFIG
    result = v24._evaluate_model(J2_CONFIG, PARENT, 250, output / "parent", refs, 24, manifest)
    aggregate = result["aggregate"]
    actual = {
        "scene_ap50": aggregate["scene_macro"]["ap50"]["mean"],
        "pooled_ap50": aggregate["pooled"]["ap50"],
        "best_iou": aggregate["scene_macro"]["best_gt_iou"]["mean"],
        "recall50": aggregate["scene_macro"]["recall50"]["mean"],
        "psnr": aggregate["scene_macro"]["psnr"]["mean"],
    }
    checks = {key: abs(float(actual[key]) - expected) <= tol for key, (expected, tol) in PARENT24_EXPECTED.items()}
    repaired = {
        "protocol": manifest,
        "records": 24,
        "held_out_scenes": 8,
        "distribution": manifest["validation_window_counts"],
        "result": {"aggregate": aggregate, "numeric_finite": result["numeric_finite"], "ap101_invariant_valid": result["ap101_invariant_valid"]},
        "actual": actual,
        "expected": {key: value[0] for key, value in PARENT24_EXPECTED.items()},
        "tolerances": {key: value[1] for key, value in PARENT24_EXPECTED.items()},
        "checks": checks,
        "passed": bool(all(checks.values()) and result["numeric_finite"] and result["ap101_invariant_valid"]),
        "config_used": J2_CONFIG,
        "checkpoint": str(PARENT.resolve()),
        "checkpoint_sha256": sha256_file(PARENT),
        "effective_step": 960,
    }
    write_json(output / "parent_24window_repaired.json", repaired)
    write_json(output / "parent_restore_comparison.json", {
        "correct": {"config": J2_CONFIG, "checkpoint": str(PARENT.resolve()), "sha256": sha256_file(PARENT), "effective_step": 960, "r2u": 1.0, "u2r": 0.1, "dino_gate": 1.0, "qmc_enabled": False, "fresh_reset": False},
        "wrong_v6": {"config": "semantic_v6_absolute_units_true_shared_token_eru1_scene_hungarian_short200_ddp8", "checkpoint": str(PARENT.resolve()), "v6_scene_ap50": 0.2355568474, "v6_pooled_ap50": 0.1380815394, "v6_psnr": 20.9397144318, "cause": "wrong parent preset"},
        "restore_result": result["checkpoint"],
        "native_query_only": True,
        "max_predictions_per_image": 100,
        "score_threshold_filtering": 0,
        "void_channel": 100,
    })
    return repaired, {"manifest": manifest, "window_fingerprints": refs}


def _bootstrap(values: list[float], seed: int = 42, iterations: int = 10000) -> dict[str, Any]:
    values_np = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values_np), size=(iterations, len(values_np)))
    means = values_np[draws].mean(axis=1)
    return {"iterations": iterations, "seed": seed, "observed_delta": float(values_np.mean()), "bootstrap_mean": float(means.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], "p_delta_gt_zero": float(np.mean(means > 0)), "scene_deltas": [float(x) for x in values_np]}


def sign_test(values: list[float]) -> dict[str, Any]:
    pos = sum(x > 0 for x in values)
    neg = sum(x < 0 for x in values)
    ties = len(values) - pos - neg
    n = pos + neg
    if not n:
        p = 1.0
    else:
        p = min(1.0, 2.0 * sum(math.comb(n, k) for k in range(min(pos, neg) + 1)) / (2 ** n))
    return {"positive": pos, "negative": neg, "tie": ties, "non_tie_count": n, "exact_two_sided_p_value": float(p)}


def _query_collapse(result: dict[str, Any], parent: dict[str, Any]) -> bool:
    a, b = result["mean"], parent["mean"]
    return bool(
        a.get("effective_query_count", 0) < 0.60 * b.get("effective_query_count", 0)
        or a.get("nonempty_query_count", 0) < 0.70 * b.get("nonempty_query_count", 0)
        or a.get("void_ratio", 0) > b.get("void_ratio", 0) + 0.10
        or a.get("pred_gt", 0) < 0.60
        or a.get("pred_gt", 0) > 2.50
    )


def _comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for scene, a in left["per_scene"].items():
        b = right["per_scene"][scene]
        rows[scene] = {key: float(a[key] - b[key]) for key in ("ap25", "ap50", "ap75", "mean_best_gt_iou", "recall_iou50", "pred_gt", "psnr", "ssim", "lpips")}
    return {"per_scene": rows, "ap50_improved_scenes": sum(x["ap50"] > 0 for x in rows.values()), "ap50_tied_scenes": sum(x["ap50"] == 0 for x in rows.values()), "ap50_declined_scenes": sum(x["ap50"] < 0 for x in rows.values()), "best_iou_improved_scenes": sum(x["mean_best_gt_iou"] > 0 for x in rows.values()), "recall50_improved_scenes": sum(x["recall_iou50"] > 0 for x in rows.values()), "largest_ap50_gains": sorted(((scene, row["ap50"]) for scene, row in rows.items()), key=lambda x: x[1], reverse=True)[:5], "largest_ap50_losses": sorted(((scene, row["ap50"]) for scene, row in rows.items()), key=lambda x: x[1])[:5]}


def _run_lsm(output: Path) -> dict[str, Any]:
    if sha256_file(LSM_MANIFEST) != LSM_MANIFEST_SHA:
        raise RuntimeError("LSM manifest SHA mismatch")
    manifest_audit = _audit_lsm_manifest(str(LSM_MANIFEST))
    data_opt, loader, dataset = _build_lsm_data(LSM_MANIFEST)
    del data_opt, dataset
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("LSM evaluator requires exactly one visible CUDA device")
    metrics = __import__("tokengs.utils.metrics", fromlist=["MetricsCalculator"]).MetricsCalculator(device=torch.device("cuda"))
    entries = json.loads(LSM_MANIFEST.read_text(encoding="utf-8"))["scenes"]
    roster = [
        ("j2_parent_000250", J2_CONFIG, PARENT, 250, False, 960),
        ("a0_000150", A0_CONFIG, A0_ROOT / "checkpoints/model_step_000150.safetensors", 150, True, 1110),
        ("a0_000200", A0_CONFIG, A0_ROOT / "checkpoints/model_step_000200.safetensors", 200, True, 1160),
        ("qmc_000150", QMC_CONFIG, QMC_ROOT / "checkpoints/model_step_000150.safetensors", 150, True, 1110),
    ]
    outputs: dict[str, Any] = {}
    restores: dict[str, Any] = {}
    fingerprint_sets: dict[str, Any] = {}
    for label, config, checkpoint, step, overlay, effective in roster:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        work = output / label
        work.mkdir(parents=True, exist_ok=True)
        if label.startswith("j2_parent"):
            model, schedule, restore = _build_eru_variant(config, checkpoint, work, overlay=False, local_step=250)
            metadata = {"path": str((checkpoint.parent / "metadata_step_000250.json").resolve()), "optimizer_step": 250, "effective_step": 960}
        else:
            model, schedule, restore = _build_eru_variant(config, checkpoint, work, overlay=True, local_step=step)
            metadata = lsm_metadata(checkpoint, step)
        result = evaluate_lsm_model(
            label, model, schedule, loader, test_dataset=None, manifest_entries=entries,
            is_instance=True, metrics=metrics, metadata=metadata, max_scenes=0,
        )
        before = restore["parameter_hash"]
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError(f"parameters changed during evaluation: {label}")
        result.update({"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint), "optimizer_step": step, "effective_step": effective, "eval_precision": "fp32", "native_query_only": True, "p_u_used": False, "metric_cluster_used": False, "oracle_used": False, "ttt_steps": 0, "restore": restore, "schedule": schedule, "parameter_hash_before": before, "parameter_hash_after": after})
        outputs[label] = result
        restores[label] = {"checkpoint": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint), "metadata": metadata, "restore": restore, "schedule": schedule}
        fingerprint_sets[label] = result["input_fingerprints"]
        write_json(work / "result.json", result)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if any(fingerprint_sets[label] != fingerprint_sets[roster[0][0]] for label, *_ in roster[1:]):
        raise RuntimeError("LSM protocol fingerprints differ")
    parent = outputs["j2_parent_000250"]
    a0150 = outputs["a0_000150"]
    a0200 = outputs["a0_000200"]
    qmc = outputs["qmc_000150"]
    comps = {
        "qmc_vs_a0_150": _comparison(qmc, a0150),
        "qmc_vs_a0_200": _comparison(qmc, a0200),
        "qmc_vs_parent": _comparison(qmc, parent),
    }
    qmc_parent_deltas = [row["ap50"] for row in comps["qmc_vs_parent"]["per_scene"].values()]
    bootstrap = {name: _bootstrap([row["ap50"] for row in comp["per_scene"].values()]) for name, comp in comps.items()}
    signs = {name: sign_test([row["ap50"] for row in comp["per_scene"].values()]) for name, comp in comps.items()}
    table = {}
    for label, result in outputs.items():
        table[label] = {"mean_ap": result["mean"]["ap"], "mean_ap25": result["mean"]["ap25"], "mean_ap50": result["mean"]["ap50"], "mean_ap75": result["mean"]["ap75"], "pooled_ap": result["pooled"]["ap"], "pooled_ap25": result["pooled"]["ap25"], "pooled_ap50": result["pooled"]["ap50"], "pooled_ap75": result["pooled"]["ap75"], "best_iou": result["mean"]["mean_best_gt_iou"], "recall50": result["mean"]["recall_iou50"], "psnr": result["mean"]["psnr"], "ssim": result["mean"]["ssim"], "lpips": result["mean"]["lpips"], "pred_gt": result["mean"]["pred_gt"], "effective_queries": result["mean"]["effective_query_count"], "nonempty_queries": result["mean"]["nonempty_query_count"], "void_ratio": result["mean"]["void_ratio"], "query_collapse": _query_collapse(result, parent) if label != "j2_parent_000250" else False}
    qmc_collapse = table["qmc_000150"]["query_collapse"]
    classification = "A_STRONG" if comps["qmc_vs_a0_150"]["per_scene"] and bootstrap["qmc_vs_a0_150"]["ci95"][0] > 0 and table["qmc_000150"]["best_iou"] >= table["a0_000150"]["best_iou"] and table["qmc_000150"]["recall50"] >= table["a0_000150"]["recall50"] and not qmc_collapse else ("B_WEAK" if table["qmc_000150"]["mean_ap50"] > table["a0_000150"]["mean_ap50"] and not qmc_collapse else "C_FAIL")
    protocol = {"manifest": manifest_audit, "manifest_sha256": LSM_MANIFEST_SHA, "scene_count": 40, "models": list(outputs), "fingerprints_match": True, "context_views": 8, "target_views": 7, "fp32": True, "native_query_only": True, "p_u_used": False, "metric_cluster_used": False, "oracle_used": False, "ttt_steps": 0, "max_predictions_per_image": 100, "min_mask_area": 1, "score_threshold_filtering": 0}
    write_json(output / "protocol_fingerprints.json", {"protocol": protocol, "per_model": fingerprint_sets})
    write_json(output / "checkpoint_metadata.json", restores)
    write_json(output / "results_table.json", table)
    write_json(output / "per_scene_metrics.json", {label: result["per_scene"] for label, result in outputs.items()})
    write_json(output / "per_target_view_metrics.json", {label: {scene: {"psnr": row["psnr_per_target_view"], "ssim": row["ssim_per_target_view"], "lpips": row["lpips_per_target_view"], "context_frame_ids": row["context_frame_ids"], "target_frame_ids": row["target_frame_ids"]} for scene, row in result["per_scene"].items()} for label, result in outputs.items()})
    write_json(output / "comparison_qmc_vs_a0_150.json", comps["qmc_vs_a0_150"])
    write_json(output / "comparison_qmc_vs_a0_200.json", comps["qmc_vs_a0_200"])
    write_json(output / "comparison_qmc_vs_parent.json", comps["qmc_vs_parent"])
    write_json(output / "paired_bootstrap.json", bootstrap)
    write_json(output / "paired_sign_test.json", signs)
    # The parent-repair evidence lives in its dedicated sibling workspace;
    # output.parent is the generic workspace root and is not the evidence path.
    write_json(output / "parent_root_cause.json", json.loads((PARENT_REPAIR_OUTPUT / "parent_root_cause.json").read_text(encoding="utf-8")))
    summary = {"protocol": protocol, "models_evaluated": list(outputs), "parent_mean_ap50": table["j2_parent_000250"]["mean_ap50"], "qmc_best_step": 150, "qmc_mean_ap50": table["qmc_000150"]["mean_ap50"], "qmc_pooled_ap50": table["qmc_000150"]["pooled_ap50"], "qmc_minus_a0_150_ap50": table["qmc_000150"]["mean_ap50"] - table["a0_000150"]["mean_ap50"], "qmc_minus_a0_200_ap50": table["qmc_000150"]["mean_ap50"] - table["a0_000200"]["mean_ap50"], "qmc_minus_parent_ap50": table["qmc_000150"]["mean_ap50"] - table["j2_parent_000250"]["mean_ap50"], "bootstrap_ci95_qmc_vs_a0_150": bootstrap["qmc_vs_a0_150"]["ci95"], "sign_test_qmc_vs_a0_150": signs["qmc_vs_a0_150"], "query_collapse": qmc_collapse, "reconstruction_preserved": table["qmc_000150"]["psnr"] - table["a0_000150"]["psnr"] >= -0.20, "classification": classification, "training_started": False, "optimizer_step_executed": False, "ttt_started": False, "oracle_used": False, "p_u_used": False}
    write_json(output / "summary.json", summary)
    (output / "status").mkdir(parents=True, exist_ok=True)
    (output / "status" / "COMPLETE").write_text("ok\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "parent24", "lsm40"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != "d839cd2226d091500b30eb3a4765b9ca58db15de":
        raise RuntimeError("unexpected git HEAD")
    if not PARENT.is_file() or sha256_file(PARENT) != PARENT_SHA:
        raise RuntimeError("protected J2 parent SHA mismatch")
    if args.output_dir.exists():
        unexpected = [entry for entry in args.output_dir.iterdir() if entry.name != "eval.log"]
        if unexpected:
            raise RuntimeError(f"refusing non-empty output: {unexpected}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_root_cause(args.output_dir)
    if args.mode == "smoke":
        # Smoke is intentionally bounded to one LSM scene and does not run a
        # full comparison. It verifies the repaired J2 construction and the
        # three model-only overlays without optimizer activity.
        manifest = _audit_lsm_manifest(str(LSM_MANIFEST))
        _, loader, _ = _build_lsm_data(LSM_MANIFEST)
        entries = json.loads(LSM_MANIFEST.read_text(encoding="utf-8"))["scenes"]
        fingerprints = {}
        for label, config, checkpoint, step, overlay, effective in (
            ("j2_parent_000250", J2_CONFIG, PARENT, 250, False, 960),
            ("a0_000150", A0_CONFIG, A0_ROOT / "checkpoints/model_step_000150.safetensors", 150, True, 1110),
            ("a0_000200", A0_CONFIG, A0_ROOT / "checkpoints/model_step_000200.safetensors", 200, True, 1160),
            ("qmc_000150", QMC_CONFIG, QMC_ROOT / "checkpoints/model_step_000150.safetensors", 150, True, 1110),
        ):
            model, schedule, restore = _build_eru_variant(config, checkpoint, args.output_dir / label, overlay=overlay, local_step=step)
            before = parameter_hash(model)
            data = next(iter(loader))
            data = {key: value.cuda() if torch.is_tensor(value) else value for key, value in data.items()}
            with torch.inference_mode():
                out = model(data, compute_quality_metrics=False)
            after = parameter_hash(model)
            if before != after:
                raise RuntimeError(f"parameter hash changed in smoke: {label}")
            if not torch.isfinite(out["images_pred"]).all():
                raise FloatingPointError(f"non-finite smoke output: {label}")
            fingerprints[label] = {"scene": str(data["scene_name"][0]), "checkpoint": str(checkpoint.resolve()), "strict_restore": restore, "schedule": schedule, "parameter_hash_unchanged": before == after, "rgb_hash": tensor_hash(out.get("images_pred")), "gaussian_hash": tensor_hash(out.get("gaussians")), "unit_logits_hash": tensor_hash(out.get("unit_logits")), "mask_hash": tensor_hash(out.get("rendered_instance_group_probability")), "target_image_to_dino": False, "p_u_used": False, "metric_cluster_used": False, "optimizer_step_executed": False}
            del model
            gc.collect()
            torch.cuda.empty_cache()
        write_json(args.output_dir / "smoke.json", {"manifest": manifest, "scene_count": 1, "models": fingerprints, "fingerprints_match": len({v["scene"] for v in fingerprints.values()}) == 1, "fp32": True, "formal_lsm40_started": False})
        (args.output_dir / "status").mkdir(exist_ok=True)
        (args.output_dir / "status" / "SMOKE_COMPLETE").write_text("ok\n", encoding="utf-8")
        return
    if args.mode == "parent24":
        repaired, protocol = _parent_24(args.output_dir)
        if not repaired["passed"]:
            raise RuntimeError(f"J2 parent 24-window gate failed: {repaired['actual']}")
        write_json(args.output_dir / "summary.json", {"parent_24window_reproduced": True, "parent": repaired, "training_started": False, "lsm40_started": False})
        (args.output_dir / "status").mkdir(exist_ok=True)
        (args.output_dir / "status" / "COMPLETE").write_text("ok\n", encoding="utf-8")
        return
    # LSM-40 is only reachable after a separately completed parent gate.
    gate = ROOT / "workspace/token_eru_query_metric_v1_parent_repair_24window_v1/parent_24window_repaired.json"
    if not gate.is_file() or not json.loads(gate.read_text(encoding="utf-8"))["passed"]:
        raise RuntimeError("LSM-40 refused: repaired J2 parent 24-window gate is not passed")
    _run_lsm(args.output_dir)


if __name__ == "__main__":
    main()
