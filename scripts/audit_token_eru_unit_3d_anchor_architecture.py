"""One-batch architecture audit for TokenGS-ERU-3DAnchor-v1."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import load_model_checkpoint  # noqa: E402


def _load_cached_gsplat_extension() -> None:
    so_path = os.environ.get("GSPLAT_PRECOMPILED_SO")
    if not so_path or not os.path.isfile(so_path) or "gsplat_cuda" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location("gsplat_cuda", so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load cached gsplat extension: {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["gsplat_cuda"] = module
    import gsplat
    sys.modules.setdefault("gsplat.csrc", module)
    setattr(gsplat, "csrc", module)


_load_cached_gsplat_extension()


CONFIG = "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_local250_3d_anchor_v1_short200_ddp8"
PARENT = Path(
    "/space/mawb/tokengs/workspace/"
    "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_ddp8/checkpoints/model_step_000250.safetensors"
)
PARENT_SHA = "61debbf5f55b76eea16012aea00fe19291309177bfdc6dd0420f6d1cb58d8cba"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    return value


def tensor_stats(value):
    if not torch.is_tensor(value):
        return None
    return {
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(value).all()),
        "mean": float(value.float().mean()),
        "std": float(value.float().std()),
        "min": float(value.float().min()),
        "max": float(value.float().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not PARENT.is_file() or sha256_file(PARENT) != PARENT_SHA:
        raise RuntimeError("J2 local250 parent is missing or SHA256 mismatched")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty audit directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    opt = dataclasses.replace(config_defaults[CONFIG])
    opt.resume = str(PARENT)
    opt.workspace = str(output)
    opt.num_workers = 0
    opt.tsh_ddp8 = False
    opt.use_wandb = False
    opt.eval_before_training = False
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(int(accelerator.local_process_index))
    model = model_registry[opt.model_type](opt).to(accelerator.device)
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        raise RuntimeError("parent strict restore marker was not set")
    model.set_token_eru_step(960)
    model.set_token_eru_dino_metric_step(960)
    loader, _, _, _ = get_multi_dataloader(opt, accelerator)
    batch = move(next(iter(loader)), accelerator.device)
    if batch["input"].shape[1] != 15:
        raise RuntimeError("architecture audit requires 8 context + 7 target views")
    model.eval()
    with torch.no_grad():
        result = model(batch, compute_quality_metrics=False)
    gaussians = result["gaussians"]
    unit_prob = result["instance_group_probabilities"]
    gs_prob = result["gaussian_group_probabilities"]
    anchor = result["unit_3d_anchor_centers_world"]
    normalized = result["unit_3d_anchor_centers_normalized"]
    delta = result["unit_3d_anchor_delta"]
    if tuple(gaussians.shape) != (gaussians.shape[0], 65536, 14):
        raise RuntimeError(f"unexpected Gaussian shape: {tuple(gaussians.shape)}")
    if tuple(anchor.shape[1:]) != (1024, 8, 3):
        raise RuntimeError(f"unexpected unit center shape: {tuple(anchor.shape)}")
    if not bool(torch.isfinite(anchor).all() and torch.isfinite(normalized).all()):
        raise RuntimeError("unit centers are not finite")
    if float(anchor.std()) <= 0.0 or float(normalized.std()) <= 0.0:
        raise RuntimeError("unit centers are degenerate")
    expected_gs_prob = unit_prob.unsqueeze(-2).expand(
        *unit_prob.shape[:3], 8, unit_prob.shape[-1]
    ).reshape_as(gs_prob)
    mapping_diff = float((gs_prob - expected_gs_prob).abs().max())
    if mapping_diff != 0.0:
        raise RuntimeError(f"unit-to-child assignment mapping changed: {mapping_diff}")
    payload = {
        "config": CONFIG,
        "parent_path": str(PARENT),
        "parent_sha256": PARENT_SHA,
        "context_views": 8,
        "target_views": 7,
        "gaussian_shape": list(gaussians.shape),
        "unit_center_shape": list(anchor.shape),
        "normalized_center_shape": list(normalized.shape),
        "position_feature_shape": list(result["unit_3d_anchor_position_features"].shape),
        "anchored_unit_shape": list(result["unit_3d_anchor_anchored_units"].shape),
        "child_layout": "[B,1024,8,8,14] from contiguous [B,65536,14]",
        "assignment_inheritance_max_diff": mapping_diff,
        "activated_opacity_slice": "gaussians[...,3:4]",
        "unit_centers": tensor_stats(anchor),
        "normalized_centers": tensor_stats(normalized),
        "opacity_mass": tensor_stats(result["unit_3d_anchor_opacity_mass"]),
        "fallback_ratio": float(result["unit_3d_anchor_fallback_mask"].float().mean()),
        "anchor_delta": tensor_stats(delta),
        "unit_logits_shape_canonical": list(unit_prob.shape),
        "native_query_channels": int(unit_prob.shape[-1]),
        "target_image_to_anchor": False,
        "target_gt_to_anchor": False,
        "p_u_used": False,
        "metric_cluster_used": False,
        "native_query_formal_output": True,
        "gaussian_count_changed": False,
        "query_count_changed": False,
    }
    (output / "architecture_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
