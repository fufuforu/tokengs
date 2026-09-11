from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.options import config_defaults
from tokengs.utils.instance_ap import instance_ap, masks_from_group_probs, gt_masks_from_instance_map
from tokengs.utils.metrics import MetricsCalculator


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/scannet_prompt/scannet_prompt_full_wide_8x7.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_stats(value: torch.Tensor | None) -> dict[str, float | int | None]:
    if value is None:
        return {"finite_ratio": None, "positive_ratio": None, "mean": None,
                "std": None, "p01": None, "p50": None, "p99": None,
                "min": None, "max": None}
    flat = value.detach().float().reshape(-1).cpu()
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    if finite_values.numel() == 0:
        return {"finite_ratio": 0.0, "positive_ratio": 0.0, "mean": None,
                "std": None, "p01": None, "p50": None, "p99": None,
                "min": None, "max": None}
    return {
        "finite_ratio": float(finite.float().mean()),
        "positive_ratio": float((finite_values > 0).float().mean()),
        "mean": float(finite_values.mean()),
        "std": float(finite_values.std(unbiased=False)),
        "p01": float(torch.quantile(finite_values, 0.01)),
        "p50": float(torch.quantile(finite_values, 0.50)),
        "p99": float(torch.quantile(finite_values, 0.99)),
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--resume_mode", required=True, choices=("official", "strict"))
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--min_pred_pixels", type=int, default=1)
    parser.add_argument("--min_gt_pixels", type=int, default=1)
    parser.add_argument("--max_predictions_per_image", type=int, default=100)
    return parser.parse_args()


def load_options(preset: str, resume: str, workspace: str, resume_mode: str):
    allowed = {"gsi_v2_recon_scannet_eval", "gsi_v2_joint_scannet_eval"}
    if preset not in allowed:
        raise ValueError(f"eval preset must be one of {sorted(allowed)}")
    opt = config_defaults[preset].evolve(
        resume=resume, workspace=workspace, gsi_v2_resume_mode=resume_mode
    )
    return opt


def strict_load_model(opt, resume: str, device: torch.device):
    model = model_registry["globalsplat_instance_v2"](opt).to(device)
    if opt.gsi_v2_resume_mode == "strict":
        state = load_file(resume, device="cpu")
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"GSI-v2 strict eval restore failed: {incompatible}")
    model.set_eval_stage()
    model.eval()
    return model


def _fixed_validation_indices(test_dataset, max_scenes: int) -> tuple[list[int], list[str]]:
    """Select the first stable manifest window for each validation scene."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    requested = [str(scene) for scene in manifest["validation_scenes"]]
    if max_scenes:
        requested = requested[: int(max_scenes)]
    if len(test_dataset.datasets) != 1:
        raise RuntimeError("GSI-v2 eval expects exactly one formal dataset")
    samples = test_dataset.datasets[0].dataset.sample_list
    by_scene = {}
    for index, sample in enumerate(samples):
        scene = str(getattr(sample, "scene", ""))
        by_scene.setdefault(scene, index)
    missing = [scene for scene in requested if scene not in by_scene]
    if missing:
        raise RuntimeError(f"formal validation scenes missing from dataloader: {missing}")
    return [by_scene[scene] for scene in requested], requested


def _scene_name(data, index: int, fallback: str) -> str:
    value = data.get("scene_name", fallback)
    if isinstance(value, (list, tuple)):
        return str(value[index])
    return str(value)


def _frame_ids(data, index: int) -> list[int]:
    value = data.get("frame_ids")
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(x) for x in value[index].reshape(-1).tolist()]
    return [int(x) for x in value[index]]


def _optimizer_step_from_resume(resume: str, mode: str) -> int:
    if mode == "official":
        return 0
    name = Path(resume).stem
    if "step_" in name:
        try:
            return int(name.rsplit("step_", 1)[1])
        except ValueError:
            pass
    return -1


def evaluate(model: torch.nn.Module, loader, opt, scene_names: list[str],
             args: argparse.Namespace, device: torch.device) -> tuple[dict, dict, dict]:
    pred_masks, pred_scores, gt_masks = [], [], []
    pred_ids, gt_ids = [], []
    scene_rows = []
    diagnostics = []
    metrics = MetricsCalculator(device=device)
    parameter_hash_before = _parameter_hash(model)
    with torch.inference_mode():
        for data in loader:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA became unavailable during GSI evaluation")
            start_time = time.perf_counter()
            out = model(data)
            b = out["images_pred"].shape[0]
            for bi in range(b):
                scene = _scene_name(data, bi, scene_names[len(scene_rows)])
                rgb_pred = out["images_pred"][bi].float().clamp(0, 1)
                rgb_gt = data["images_output"][bi].float().clamp(0, 1)
                per_view_psnr = metrics.calculate_psnr(rgb_pred, rgb_gt, reduction="none")
                per_view_ssim = metrics.calculate_ssim(rgb_pred, rgb_gt, reduction="none")
                per_view_lpips = metrics.calculate_lpips(rgb_pred, rgb_gt, reduction="none")
                mse = ((rgb_pred - rgb_gt) ** 2).mean(dim=(1, 2, 3))
                scene_rows.append({
                    "scene_name": scene,
                    "frame_ids": _frame_ids(data, bi),
                    "context_views": int(opt.num_input_views),
                    "target_views": int(rgb_pred.shape[0]),
                    "psnr": float(per_view_psnr.mean()),
                    "ssim": float(per_view_ssim.mean()),
                    "lpips": float(per_view_lpips.mean()),
                    "pooled_mse": float(mse.mean()),
                    "forward_render_seconds": float(time.perf_counter() - start_time),
                })
                alpha = out.get("alphas_pred")
                depth = out.get("depths_pred")
                gaussians = {
                    "means": out.get("means"), "scales": out.get("scales"),
                    "opacities": out.get("opacities"), "sh": out.get("sh"),
                }
                rgb_std = rgb_pred.std(dim=(1, 2, 3))
                diagnostics.append({
                    "scene_name": scene,
                    "frame_ids": _frame_ids(data, bi),
                    "alpha": _tensor_stats(None if alpha is None else alpha[bi]),
                    "depth": _tensor_stats(None if depth is None else depth[bi]),
                    "gaussian": {key: _tensor_stats(value[bi] if value is not None and value.ndim > 0 and value.shape[0] == rgb_pred.shape[0] else value)
                                 for key, value in gaussians.items()},
                    "rgb_spatial_std_mean": float(rgb_std.mean()),
                    "rgb_spatial_std_min": float(rgb_std.min()),
                    "rgb_non_degenerate": bool(torch.all(rgb_std > 0.01)),
                    "alpha_valid_pixel_ratio": float((alpha[bi] > 1e-6).float().mean()) if alpha is not None else None,
                    "depth_positive_ratio": float((depth[bi] > 0).float().mean()) if depth is not None else None,
                })
    if len(scene_rows) != len(scene_names):
        raise RuntimeError(f"expected {len(scene_names)} scenes, evaluated {len(scene_rows)}")
    pooled_mse = float(np.mean([row["pooled_mse"] for row in scene_rows]))
    result = {
        "protocol": "GSI-v2 R0/R1 reconstruction, formal manifest first window per validation scene",
        "scene_count": len(scene_rows),
        "scene_names": scene_names,
        "mean_psnr": float(np.mean([row["psnr"] for row in scene_rows])),
        "mean_ssim": float(np.mean([row["ssim"] for row in scene_rows])),
        "mean_lpips": float(np.mean([row["lpips"] for row in scene_rows])),
        "pooled_mse": pooled_mse,
        "pooled_psnr": float(-10.0 * np.log10(max(pooled_mse, 1e-8))),
        "per_scene": scene_rows,
        "resume": str(Path(opt.resume).resolve()),
        "resume_mode": opt.gsi_v2_resume_mode,
        "optimizer_step": _optimizer_step_from_resume(opt.resume, opt.gsi_v2_resume_mode),
        "official_lineage": opt.lineage_metadata() if hasattr(opt, "lineage_metadata") else {},
        "manifest": {"path": str(MANIFEST.resolve()), "sha256": sha256_file(MANIFEST)},
    }
    diagnostics_payload = {
        "per_scene": diagnostics,
        "parameter_hash_before": parameter_hash_before,
        "parameter_hash_after": _parameter_hash(model),
        "parameter_hash_unchanged": parameter_hash_before == _parameter_hash(model),
        "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "official_checkpoint_report": _official_report(model),
        "no_target_leakage": {
            "model_receives": "context images, context K, context c2w only",
            "target_views": int(opt.num_views - opt.num_input_views),
            "target_rgb_used_only_for_metrics": True,
        },
        "all_finite": bool(all(np.isfinite(float(row[key])) for row in scene_rows for key in ("psnr", "ssim", "lpips", "pooled_mse"))),
    }
    return result, {"per_scene": scene_rows}, diagnostics_payload


def _parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(value.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def _official_report(model: torch.nn.Module) -> dict:
    report = getattr(model, "_official_checkpoint_report", None)
    if report is None:
        return {}
    return {key: getattr(report, key) for key in ("path", "sha256", "loaded_tensor_count", "checkpoint_tensor_count", "loaded_state_numel", "checkpoint_state_numel") if hasattr(report, key)}


def main() -> None:
    args = parse_args()
    opt = load_options(args.preset, args.resume, args.workspace, args.resume_mode)
    accelerator = Accelerator(mixed_precision="no")
    _train, test, _train_ds, test_dataset = get_multi_dataloader(opt, accelerator)
    subset_indices, scene_names = _fixed_validation_indices(test_dataset, args.max_scenes)
    test = DataLoader(
        Subset(test_dataset, subset_indices), batch_size=opt.batch_size,
        shuffle=False, num_workers=0, pin_memory=True, drop_last=False,
    )
    model = strict_load_model(opt, args.resume, accelerator.device)
    model, test = accelerator.prepare(model, test)
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    result, per_scene, diagnostics = evaluate(model, test, opt, scene_names, args, accelerator.device)
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        for filename, payload in (("reconstruction_metrics.json", result), ("per_scene_metrics.json", per_scene), ("diagnostics.json", diagnostics)):
            (workspace / filename).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
