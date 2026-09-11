"""Strict paired FP32/BF16/gate audit for GSI-v2 Phase-J short355.

This script only evaluates existing checkpoints.  It never constructs an
optimizer and never calls backward or step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_gsi_v2_short355 import (  # noqa: E402
    _first_validation_indices,
    _input_fingerprint,
)
from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


EXPECTED_COMMIT = "b975eb7ea236ef891721072dfb40c104c4c71f91"
DEFAULT_R1 = ROOT / "workspace/gsi_v2_recon_scannet_adapt_ddp8/checkpoints/model_step_000250.safetensors"
DEFAULT_JOINT = ROOT / "workspace/gsi_v2_joint_scannet_short355_ddp8"
DEFAULT_MANIFEST = ROOT / "data/scannet_prompt/scannet_prompt_full_wide_8x7.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r1_checkpoint", type=Path, required=True)
    parser.add_argument("--joint_workspace", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval_steps", type=int, nargs="+", required=True)
    parser.add_argument("--device_partition", required=True)
    parser.add_argument("--run_bf16_control", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _state_hash(state: dict[str, torch.Tensor], *, exclude_prefixes=()) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _reconstruction_state(model) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    tri_prefixes = (
        "reconstruction.slot_encoder.slot_to_ins.",
        "reconstruction.slot_encoder.ins_rounds.",
        "reconstruction.slot_encoder.tri_adapters.",
        "instance_head.",
    )
    return {key: value for key, value in state.items() if not any(key.startswith(p) for p in tri_prefixes)}


def _max_mean_diff(left: torch.Tensor, right: torch.Tensor) -> dict[str, object]:
    diff = (left.detach().float() - right.detach().float()).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "allclose_atol_1e-6_rtol_1e-5": bool(torch.allclose(left, right, atol=1e-6, rtol=1e-5)),
    }


def identity_check(manifest: Path, r1_checkpoint: Path, step0_checkpoint: Path) -> dict[str, object]:
    """Compare standalone R1 and joint step0 on the exact same eight batches."""
    accelerator = Accelerator(mixed_precision="no")
    data_opt = config_defaults["gsi_v2_joint_scannet_short355_ddp8"].evolve(num_workers=0)
    _train, _test, _train_ds, test_dataset = get_multi_dataloader(data_opt, accelerator)
    indices, scenes = _first_validation_indices(test_dataset)
    loader = DataLoader(Subset(test_dataset, indices), batch_size=1, shuffle=False, num_workers=0)
    r1_opt = config_defaults["gsi_v2_recon_scannet_eval"].evolve(num_workers=0)
    joint_opt = config_defaults["gsi_v2_joint_scannet_short355_ddp8"].evolve(num_workers=0, gsi_v2_return_debug_tensors=True)
    r1 = model_registry[r1_opt.model_type](r1_opt).to(accelerator.device)
    r1.load_state_dict(load_file(str(r1_checkpoint), device="cpu"), strict=True)
    r1.set_eval_stage()
    r1.eval()
    joint = model_registry[joint_opt.model_type](joint_opt).to(accelerator.device)
    joint.load_state_dict(load_file(str(step0_checkpoint), device="cpu"), strict=True)
    joint.set_eval_schedule(0)
    joint.set_eval_stage()
    joint.eval()
    field_names = {
        "rgb": "images_pred",
        "alpha": "alphas_pred",
        "depth": "depths_pred",
        "means": "means",
        "scales": "scales",
        "rotations": "rotations",
        "opacity": "opacities",
        "sh": "sh",
    }
    aggregate = {name: {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "allclose_atol_1e-6_rtol_1e-5": True} for name in field_names}
    fingerprints = []
    with torch.inference_mode():
        for data in loader:
            data = _move(data, accelerator.device)
            scene = str(data["scene_name"][0])
            fingerprints.append(_input_fingerprint(data, scene))
            left = r1(data)
            right = joint(data)
            for name, key in field_names.items():
                current = _max_mean_diff(left[key], right[key])
                aggregate[name]["max_abs_diff"] = max(aggregate[name]["max_abs_diff"], current["max_abs_diff"])
                aggregate[name]["mean_abs_diff"] += current["mean_abs_diff"]
                aggregate[name]["allclose_atol_1e-6_rtol_1e-5"] &= current["allclose_atol_1e-6_rtol_1e-5"]
    for value in aggregate.values():
        value["mean_abs_diff"] /= len(scenes)
    r1_hash = _state_hash(_reconstruction_state(r1))
    joint_hash = _state_hash(_reconstruction_state(joint))
    return {
        "scene_count": len(scenes),
        "scene_names": scenes,
        "input_fingerprints": fingerprints,
        "field_differences": aggregate,
        "r1_reconstruction_state_hash": r1_hash,
        "joint_step0_reconstruction_state_hash": joint_hash,
        "reconstruction_state_hash_match": r1_hash == joint_hash,
        "identity_pass": bool(all(item["allclose_atol_1e-6_rtol_1e-5"] for item in aggregate.values()) and r1_hash == joint_hash),
        "manifest": str(manifest.resolve()),
    }


def run_eval(
    checkpoint: Path,
    output: Path,
    optimizer_step: int,
    precision: str,
    *,
    gate_override: float | None = None,
    preset: str = "gsi_v2_joint_scannet_eval",
    max_scenes: int = 0,
    log,
) -> dict:
    command = [
        sys.executable, str(ROOT / "scripts/eval_gsi_v2_short355.py"),
        "--checkpoint", str(checkpoint), "--output", str(output),
        "--optimizer_step", str(optimizer_step), "--eval_precision", precision,
        "--preset", preset,
    ]
    if gate_override is not None:
        command.extend(["--gate_override", str(gate_override)])
    if max_scenes:
        command.extend(["--max_scenes", str(max_scenes)])
    log.write(f"\n$ {' '.join(command)}\n")
    log.flush()
    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def _protocol_equal(results: list[dict]) -> tuple[bool, dict]:
    if not results:
        return False, {}
    reference = results[0].get("input_fingerprints", [])
    detail = {"reference": reference, "comparisons": []}
    valid = True
    for result in results[1:]:
        current = result.get("input_fingerprints", [])
        same = current == reference
        detail["comparisons"].append({"checkpoint": result.get("checkpoint_path"), "match": same})
        valid &= same
    return valid, detail


def _mean(result: dict, key: str) -> float:
    return float(result["mean"][key])


def main() -> None:
    cli = parse_args()
    commit = git_commit()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"unexpected git HEAD: {commit} != {EXPECTED_COMMIT}")
    if not cli.manifest.is_file() or sha256_file(cli.manifest) != sha256_file(DEFAULT_MANIFEST):
        raise RuntimeError("audit manifest does not match the repository canonical manifest")
    if cli.output_dir.exists() and any(cli.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty audit directory: {cli.output_dir}")
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = cli.output_dir / "audit.log"
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("gsi_phase_j_audit")
    logger.info("git_commit=%s device_partition=%s", commit, cli.device_partition)
    log = log_path.open("a", encoding="utf-8")
    try:
        joint_ckpt_dir = cli.joint_workspace / "checkpoints"
        smoke = {}
        smoke["r1_step0250_fp32"] = run_eval(
            cli.r1_checkpoint, cli.output_dir / "single_scene" / "r1_step0250_fp32" / "result.json", 250, "fp32",
            preset="gsi_v2_recon_scannet_eval", max_scenes=1, log=log,
        )
        smoke["joint_step000000_fp32_native"] = run_eval(
            joint_ckpt_dir / "model_step_000000.safetensors",
            cli.output_dir / "single_scene" / "joint_step000000_fp32_native" / "result.json", 0, "fp32",
            max_scenes=1, log=log,
        )
        smoke["joint_step000200_fp32_native"] = run_eval(
            joint_ckpt_dir / "model_step_000200.safetensors",
            cli.output_dir / "single_scene" / "joint_step000200_fp32_native" / "result.json", 200, "fp32",
            max_scenes=1, log=log,
        )
        smoke["joint_step000200_fp32_gate0"] = run_eval(
            joint_ckpt_dir / "model_step_000200.safetensors",
            cli.output_dir / "single_scene" / "joint_step000200_fp32_gate0" / "result.json", 200, "fp32",
            gate_override=0.0, max_scenes=1, log=log,
        )
        r1_result = run_eval(
            cli.r1_checkpoint, cli.output_dir / "r1_step0250_fp32" / "result.json", 250, "fp32",
            preset="gsi_v2_recon_scannet_eval", log=log,
        )
        native = {}
        for step in cli.eval_steps:
            checkpoint = joint_ckpt_dir / f"model_step_{step:06d}.safetensors"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            native[str(step)] = run_eval(
                checkpoint, cli.output_dir / f"joint_step{step:06d}_fp32_native" / "result.json",
                step, "fp32", log=log,
            )
        precision = {}
        if cli.run_bf16_control:
            precision["r1_step250"] = run_eval(
                cli.r1_checkpoint, cli.output_dir / "r1_step0250_bf16" / "result.json", 250, "bf16",
                preset="gsi_v2_recon_scannet_eval", log=log,
            )
            precision["joint_step0"] = run_eval(
                joint_ckpt_dir / "model_step_000000.safetensors",
                cli.output_dir / "joint_step000000_bf16_native" / "result.json", 0, "bf16", log=log,
            )
        gate_zero = {}
        for step in (100, 200, 355):
            if str(step) not in native:
                raise RuntimeError(f"gate-zero step {step} must be in --eval_steps")
            checkpoint = joint_ckpt_dir / f"model_step_{step:06d}.safetensors"
            gate_zero[str(step)] = run_eval(
                checkpoint, cli.output_dir / f"joint_step{step:06d}_fp32_gate0" / "result.json",
                step, "fp32", gate_override=0.0, log=log,
            )
    finally:
        log.close()

    all_protocol_results = [r1_result, *native.values(), *gate_zero.values()]
    paired_valid, protocol_detail = _protocol_equal(all_protocol_results)
    identity = identity_check(cli.manifest, cli.r1_checkpoint, joint_ckpt_dir / "model_step_000000.safetensors")
    (cli.output_dir / "identity_check.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    (cli.output_dir / "protocol_fingerprints.json").write_text(json.dumps(protocol_detail, indent=2), encoding="utf-8")
    (cli.output_dir / "precision_control.json").write_text(json.dumps(precision, indent=2), encoding="utf-8")
    (cli.output_dir / "gate_ablation.json").write_text(json.dumps(gate_zero, indent=2), encoding="utf-8")
    (cli.output_dir / "single_scene_smoke.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    per_scene = {
        "r1_step0250_fp32": r1_result.get("per_scene", []),
        **{f"joint_step{step}_fp32_native": result.get("per_scene", []) for step, result in native.items()},
        **{f"joint_step{step}_fp32_gate0": result.get("per_scene", []) for step, result in gate_zero.items()},
    }
    (cli.output_dir / "per_scene_metrics.json").write_text(json.dumps(per_scene, indent=2), encoding="utf-8")

    step0 = native["0"]
    best_step = max(native, key=lambda step: (_mean(native[step], "ap50"), _mean(native[step], "ap25"), -int(step)))
    threshold = _mean(step0, "psnr") - 0.20
    eligible = [step for step, result in native.items() if _mean(result, "psnr") >= threshold]
    if eligible:
        best_step = max(eligible, key=lambda step: (_mean(native[step], "ap50"), _mean(native[step], "ap25"), -int(step)))
    short_step = native[best_step]
    scene_iou_improved = sum(
        float(current["mean_best_gt_iou"]) > float(base["mean_best_gt_iou"])
        for base, current in zip(step0["per_scene"], short_step["per_scene"])
    )
    short_valid = bool(
        _mean(short_step, "ap50") - _mean(step0, "ap50") >= 0.05
        and _mean(short_step, "mean_best_gt_iou") - _mean(step0, "mean_best_gt_iou") >= 0.05
        and _mean(short_step, "recall_iou50") - _mean(step0, "recall_iou50") >= 0.05
        and scene_iou_improved >= 5
        and _mean(short_step, "psnr") >= threshold
        and _mean(short_step, "void_ratio") < 0.5
        and _mean(short_step, "pred_gt") > 0.1
    )
    summary = {
        "git_commit": commit,
        "manifest_sha256": sha256_file(cli.manifest),
        "paired_protocol_valid": paired_valid,
        "protocol_detail": protocol_detail,
        "r1_step250_fp32": r1_result["mean"],
        "joint_native_fp32": {step: result["mean"] | {"pooled": result["pooled"], "schedule": result["schedule"]} for step, result in native.items()},
        "precision_control": precision,
        "gate_zero_ablation": gate_zero,
        "step0_identity": identity,
        "selected_native_checkpoint": str((joint_ckpt_dir / f"model_step_{int(best_step):06d}.safetensors").resolve()),
        "selected_native_step": int(best_step),
        "short_pipeline_valid": short_valid,
        "ready_for_long_phase_j": False,
        "training_started": False,
        "lsm40_started": False,
        "identity_failure_stop": not identity["identity_pass"],
        "formal_selection": {
            "psnr_floor": threshold,
            "eligible_steps": [int(step) for step in eligible],
            "best_step_ap50": _mean(short_step, "ap50"),
            "best_step_pooled_ap50": float(short_step["pooled"]["ap50"]),
            "best_step_delta_psnr_vs_step0": _mean(short_step, "psnr") - _mean(step0, "psnr"),
            "gate_zero_delta_ap50_step200": _mean(native["200"], "ap50") - _mean(gate_zero["200"], "ap50"),
            "gate_zero_delta_psnr_step200": _mean(native["200"], "psnr") - _mean(gate_zero["200"], "psnr"),
            "scene_iou_improved": int(scene_iou_improved),
        },
    }
    (cli.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
