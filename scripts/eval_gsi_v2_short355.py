"""Fixed eight-scene evaluation for GSI-v2 short355 checkpoints.

One invocation evaluates one strict Phase-J checkpoint using the canonical
validation first-window samples.  The launcher can run independent
checkpoints on separate GPUs; this script never changes model parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

torch._dynamo.config.disable = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_instance_lsm_protocol import _mask_diagnostics  # noqa: E402
from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.utils.instance_ap import (  # noqa: E402
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)
from tokengs.utils.metrics import MetricsCalculator  # noqa: E402


MANIFEST = ROOT / "data/scannet_prompt/scannet_prompt_full_wide_8x7.json"
SHORT_CFG = "gsi_v2_joint_scannet_short355_ddp8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--optimizer_step", required=True, type=int)
    parser.add_argument("--eval_precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--gate_override", type=float, default=None)
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument(
        "--preset", choices=("gsi_v2_joint_scannet_eval", "gsi_v2_recon_scannet_eval"),
        default="gsi_v2_joint_scannet_eval",
    )
    return parser.parse_args()


def _first_validation_indices(test_dataset):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    scenes = [str(value) for value in manifest["validation_scenes"]]
    if len(test_dataset.datasets) != 1:
        raise RuntimeError("expected one canonical GSI dataset")
    samples = test_dataset.datasets[0].dataset.sample_list
    first = {}
    for index, sample in enumerate(samples):
        first.setdefault(str(getattr(sample, "scene", "")), index)
    missing = [scene for scene in scenes if scene not in first]
    if missing:
        raise RuntimeError(f"validation scenes missing: {missing}")
    return [first[scene] for scene in scenes], scenes


def _scene_name(data) -> str:
    value = data["scene_name"]
    return str(value[0] if isinstance(value, (tuple, list)) else value)


def _frame_ids(data):
    value = data["frame_ids"]
    if torch.is_tensor(value):
        return [int(item) for item in value[0].reshape(-1).tolist()]
    return [int(item) for item in value[0]]


def _tensor_hash(*named_tensors: tuple[str, torch.Tensor | None]) -> str:
    digest = hashlib.sha256()
    for name, value in named_tensors:
        digest.update(name.encode("utf-8"))
        if value is None:
            digest.update(b"<none>")
            continue
        value = value.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _input_fingerprint(data, scene: str) -> dict[str, object]:
    frames = _frame_ids(data)
    if len(frames) != 15:
        raise RuntimeError(f"expected 15 frame ids for {scene}, got {frames}")
    context_frames, target_frames = frames[:8], frames[8:]
    return {
        "scene_id": scene,
        "context_frame_ids": context_frames,
        "target_frame_ids": target_frames,
        "context_rgb_sha256": _tensor_hash(("context_rgb", data["images_input"])),
        "target_rgb_sha256": _tensor_hash(("target_rgb", data["images_output"])),
        "context_camera_sha256": _tensor_hash(
            ("context_intrinsics", data["intrinsics_input"]),
            ("context_cam_to_world", data["cam_to_world_input"]),
        ),
        "target_camera_sha256": _tensor_hash(
            ("target_intrinsics", data["intrinsics"]),
            ("target_cam_view", data["cam_view"]),
        ),
        "gt_instance_mask_sha256": _tensor_hash(("gt_instance_mask", data.get("instance_label_output"))),
    }


def _metadata_check(checkpoint: Path, optimizer_step: int) -> bool:
    metadata = checkpoint.parent / f"metadata_step_{optimizer_step:06d}.json"
    if not metadata.exists():
        print(f"WARNING: metadata not found for optimizer_step={optimizer_step}: {metadata}")
        return False
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    observed = payload.get("optimizer_step")
    if observed != optimizer_step:
        raise RuntimeError(
            f"checkpoint metadata step mismatch: expected {optimizer_step}, observed {observed} in {metadata}"
        )
    return True


def _scene_instance(out, data):
    probability = out["rendered_instance_group_probability"][0].detach().float().cpu().numpy()[:, :, 0]
    labels = data["instance_label_output"][0].detach().long().cpu().numpy()
    predictions, scores, pred_ids = [], [], []
    ground_truth, gt_ids = [], []
    for view in range(probability.shape[1]):
        image_id = f"{_scene_name(data)}:target:{view}"
        masks, view_scores = masks_from_group_probs(
            probability[:, view], void_channel=probability.shape[0] - 1, min_mask_area=1
        )
        predictions.extend(masks)
        scores.extend(view_scores)
        pred_ids.extend([image_id] * len(masks))
        gt = gt_masks_from_instance_map(labels[view], min_mask_area=1)
        ground_truth.extend(gt)
        gt_ids.extend([image_id] * len(gt))
    ap = instance_ap(
        predictions, scores, ground_truth, thresholds=(.25, .5, .75), vectorized=True,
        pred_image_ids=pred_ids, gt_image_ids=gt_ids,
    )
    diag = _mask_diagnostics(predictions, ground_truth, pred_ids, gt_ids)
    assignment = out["gsi_v2_instance_assignment_probabilities"].detach().float()
    usage = assignment[..., :-1].mean(dim=(0, 1))
    entropy = (-(assignment * assignment.clamp_min(1e-8).log()).sum(-1)).mean()
    return {
        "ap": float(ap["ap_mean"]), "ap25": float(ap["ap_25"]),
        "ap50": float(ap["ap_50"]), "ap75": float(ap["ap_75"]),
        "mean_best_gt_iou": float(diag["mean_best_gt_iou"]),
        "recall_iou25": float(diag["recall_iou25"]),
        "recall_iou50": float(diag["recall_iou50"]),
        "recall_iou75": float(diag["recall_iou75"]),
        "pred_gt": float(len(predictions) / max(1, len(ground_truth))),
        "prediction_count_nonempty": int(len(predictions)),
        "gt_count": int(len(ground_truth)),
        "active_query_count_mass_gt_0.001": int((usage > .001).sum()),
        "effective_query_count": float(torch.exp(entropy).item()),
        "assignment_entropy": float(entropy.item()),
        "void_ratio": float(probability[-1].mean()),
        "predictions": predictions, "scores": scores, "prediction_ids": pred_ids,
        "ground_truth": ground_truth, "ground_truth_ids": gt_ids,
    }


def main():
    cli = args()
    checkpoint = Path(cli.checkpoint).resolve()
    output = Path(cli.output).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evaluation output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if cli.optimizer_step < 0:
        raise ValueError("--optimizer_step must be non-negative")
    metadata_step_verified = _metadata_check(checkpoint, cli.optimizer_step)
    data_opt = config_defaults[SHORT_CFG].evolve(
        resume=str(checkpoint), gsi_v2_resume_mode="strict", evaluating=True,
        workspace=str(output.parent), gsi_v2_return_debug_tensors=True,
        num_workers=0, max_eval_iters=8,
    )
    model_opt = data_opt
    if cli.preset == "gsi_v2_recon_scannet_eval":
        model_opt = config_defaults[cli.preset].evolve(
            resume=str(checkpoint), gsi_v2_resume_mode="strict", evaluating=True,
            workspace=str(output.parent), num_workers=0, max_eval_iters=8,
        )
    accelerator = Accelerator(mixed_precision="no" if cli.eval_precision == "fp32" else "bf16")
    _train, test, _train_ds, test_dataset = get_multi_dataloader(data_opt, accelerator)
    indices, scene_names = _first_validation_indices(test_dataset)
    if cli.max_scenes:
        if cli.max_scenes < 1:
            raise ValueError("--max_scenes must be positive when provided")
        indices, scene_names = indices[:cli.max_scenes], scene_names[:cli.max_scenes]
    test = DataLoader(
        Subset(test_dataset, indices), batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    model = model_registry[model_opt.model_type](model_opt).to(accelerator.device)
    state = load_file(str(checkpoint), device="cpu")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict restore failed: {incompatible}")
    if cli.preset == "gsi_v2_joint_scannet_eval":
        schedule = model.set_eval_schedule(cli.optimizer_step, gate_override=cli.gate_override)
        model.set_eval_stage()
    else:
        schedule = None
    model.eval()
    model, test = accelerator.prepare(model, test)
    metrics = MetricsCalculator(device=accelerator.device)
    rows = []
    all_predictions, all_scores, all_pred_ids = [], [], []
    all_ground_truth, all_gt_ids = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for data in test:
            with accelerator.autocast():
                out = model(data)
            scene = _scene_name(data)
            rgb_pred = out["images_pred"].float().clamp(0, 1)
            rgb_gt = data["images_output"].float().clamp(0, 1)
            psnr_per_view = metrics.calculate_psnr(rgb_pred, rgb_gt, reduction="none")[0]
            ssim_per_view = metrics.calculate_ssim(rgb_pred, rgb_gt, reduction="none")[0]
            lpips_per_view = metrics.calculate_lpips(rgb_pred, rgb_gt, reduction="none")[0]
            row = {
                "scene_name": scene,
                "frame_ids": _frame_ids(data),
                "context_views": 8, "target_views": 7,
                "psnr_per_target_view": [float(x) for x in psnr_per_view],
                "ssim_per_target_view": [float(x) for x in ssim_per_view],
                "lpips_per_target_view": [float(x) for x in lpips_per_view],
                "psnr": float(psnr_per_view.mean()),
                "ssim": float(ssim_per_view.mean()),
                "lpips": float(lpips_per_view.mean()),
                "reconstruction_loss": float(out["loss_reconstruction"].detach()),
                "mse_per_target_view": [float(x) for x in ((rgb_pred - rgb_gt) ** 2).mean(dim=(2, 3, 4))[0]],
                "input_fingerprint": _input_fingerprint(data, scene),
            }
            row["pooled_mse"] = float(np.mean(row["mse_per_target_view"]))
            instance = _scene_instance(out, data) if cli.preset == "gsi_v2_joint_scannet_eval" else {
                "ap": 0.0, "ap25": 0.0, "ap50": 0.0, "ap75": 0.0,
                "mean_best_gt_iou": 0.0, "recall_iou25": 0.0, "recall_iou50": 0.0,
                "recall_iou75": 0.0, "pred_gt": 0.0, "prediction_count_nonempty": 0,
                "gt_count": 0, "active_query_count_mass_gt_0.001": 0,
                "effective_query_count": 0.0, "assignment_entropy": 0.0, "void_ratio": 0.0,
                "predictions": [], "scores": [], "prediction_ids": [],
                "ground_truth": [], "ground_truth_ids": [],
            }
            for key in (
                "ap", "ap25", "ap50", "ap75", "mean_best_gt_iou",
                "recall_iou25", "recall_iou50", "recall_iou75", "pred_gt",
                "prediction_count_nonempty", "gt_count",
                "active_query_count_mass_gt_0.001", "effective_query_count",
                "assignment_entropy", "void_ratio",
            ):
                row[key] = instance[key]
            rows.append(row)
            all_predictions.extend(instance["predictions"])
            all_scores.extend(instance["scores"])
            all_pred_ids.extend(instance["prediction_ids"])
            all_ground_truth.extend(instance["ground_truth"])
            all_gt_ids.extend(instance["ground_truth_ids"])
    pooled_ap = instance_ap(
        all_predictions, all_scores, all_ground_truth, thresholds=(.25, .5, .75), vectorized=True,
        pred_image_ids=all_pred_ids, gt_image_ids=all_gt_ids,
    ) if cli.preset == "gsi_v2_joint_scannet_eval" else {"ap_mean": 0.0, "ap_25": 0.0, "ap_50": 0.0, "ap_75": 0.0}
    pooled_diag = _mask_diagnostics(all_predictions, all_ground_truth, all_pred_ids, all_gt_ids) if cli.preset == "gsi_v2_joint_scannet_eval" else {
        "mean_best_gt_iou": 0.0, "recall_iou25": 0.0, "recall_iou50": 0.0, "recall_iou75": 0.0,
    }
    pooled_mse = float(sum(sum(row["mse_per_target_view"]) for row in rows) / max(1, 7 * len(rows)))
    result = {
        "protocol": "GSI-v2 short355 canonical full_wide_8x7 validation first window",
        "checkpoint": str(checkpoint), "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "optimizer_step": int(cli.optimizer_step),
        "eval_precision": cli.eval_precision,
        "gate_mode": "not_applicable" if cli.preset == "gsi_v2_recon_scannet_eval" else ("override" if cli.gate_override is not None else "native"),
        "gate_override": cli.gate_override,
        "schedule": schedule,
        "metadata_step_verified": metadata_step_verified,
        "formal_metric": cli.gate_override is None,
        "diagnostic_only": cli.gate_override is not None,
        "ablation": "gate_zero" if cli.gate_override == 0.0 else None,
        "scene_names": scene_names, "scene_count": len(rows),
        "mean": {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("psnr", "ssim", "lpips", "reconstruction_loss", "ap", "ap25", "ap50", "ap75", "mean_best_gt_iou", "recall_iou25", "recall_iou50", "recall_iou75", "pred_gt", "effective_query_count", "assignment_entropy", "void_ratio")
        },
        "pooled_mse": pooled_mse,
        "pooled_psnr": float(-10.0 * np.log10(max(pooled_mse, 1e-8))),
        "pooled": {
            "ap": float(pooled_ap["ap_mean"]), "ap25": float(pooled_ap["ap_25"]),
            "ap50": float(pooled_ap["ap_50"]), "ap75": float(pooled_ap["ap_75"]),
            "mean_best_gt_iou": float(pooled_diag["mean_best_gt_iou"]),
            "recall_iou25": float(pooled_diag["recall_iou25"]),
            "recall_iou50": float(pooled_diag["recall_iou50"]),
            "recall_iou75": float(pooled_diag["recall_iou75"]),
        },
        "per_scene": rows,
        "checkpoint_restore": {
            "strict": True, "state_key_count": len(state),
            "reconstruction_keys": sum(key.startswith("reconstruction.") for key in state),
            "fresh_reset": False, "pgsr_absent": not any("pgsr" in key.lower() for key in state),
        },
        "manifest": {"path": str(MANIFEST.resolve()), "sha256": sha256_file(MANIFEST)},
        "input_fingerprints": [row["input_fingerprint"] for row in rows],
        "all_finite": bool(all(np.isfinite(row[key]) for row in rows for key in ("psnr", "ssim", "lpips", "reconstruction_loss", "ap50", "pooled_mse"))),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    output.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
