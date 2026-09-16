"""Matched 100-step fixed-batch audit for EQC-v1.

This script is intentionally bounded to one real training batch and 100 local
optimizer updates.  It does not create a formal training workspace or run a
multi-scene evaluator.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import (  # noqa: E402
    configure_joint_formation_trainability,
    load_model_checkpoint,
    setup_optimizer,
    setup_scheduler,
    _load_cached_gsplat_extension,
)
from tokengs.utils.instance_ap import (  # noqa: E402
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)

_load_cached_gsplat_extension()

PARENT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_"
    "dino_metric_joint_formation_j2_ddp8/checkpoints/model_step_000250.safetensors"
)
PARENT_SHA256 = "61debbf5f55b76eea16012aea00fe19291309177bfdc6dd0420f6d1cb58d8cba"
CONTROL = "semantic_v6_j2_local250_eqc_v1_control_short200_ddp8"
TREATMENT = "semantic_v6_j2_local250_eqc_v1_treatment_short200_ddp8"
MILESTONES = (0, 1, 2, 5, 25, 50, 100)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
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


def batch_hash(batch) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(batch.items()):
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_hash(model, trainable_only=False) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if trainable_only and not parameter.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(repr(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
        count += 1
    if count == 0:
        raise RuntimeError("empty parameter hash")
    return digest.hexdigest()


def finite_model(model) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def prediction_metrics(probability, labels):
    predictions, scores, image_ids = [], [], []
    ground_truth, gt_ids = [], []
    for view in range(probability.shape[1]):
        image_id = f"fixed:target:{view}"
        masks, view_scores = masks_from_group_probs(
            probability[:, view],
            void_channel=probability.shape[0] - 1,
            min_mask_area=1,
        )
        predictions.extend(masks)
        scores.extend(view_scores)
        image_ids.extend([image_id] * len(masks))
        gt = gt_masks_from_instance_map(labels[view], min_mask_area=1)
        ground_truth.extend(gt)
        gt_ids.extend([image_id] * len(gt))
    ap = instance_ap(
        predictions,
        scores,
        ground_truth,
        thresholds=(0.25, 0.5, 0.75),
        vectorized=True,
        pred_image_ids=image_ids,
        gt_image_ids=gt_ids,
    )
    best = []
    for gt, gt_id in zip(ground_truth, gt_ids):
        best_iou = 0.0
        for pred, pred_id in zip(predictions, image_ids):
            if pred_id != gt_id:
                continue
            intersection = float((pred * gt).sum())
            union = float(pred.sum() + gt.sum() - intersection)
            best_iou = max(best_iou, intersection / union if union else 0.0)
        best.append(best_iou)
    return {
        "ap25": float(ap["ap_25"]),
        "ap50": float(ap["ap_50"]),
        "ap75": float(ap["ap_75"]),
        "best_iou": float(sum(best) / max(1, len(best))),
        "recall50": float(sum(value >= 0.5 for value in best) / max(1, len(best))),
        "pred_gt": float(len(predictions) / max(1, len(ground_truth))),
        "prediction_count": len(predictions),
        "gt_count": len(ground_truth),
    }


def build_runtime(config_name: str, output: Path):
    opt = dataclasses.replace(config_defaults[config_name])
    opt.resume = str(PARENT)
    opt.workspace = str(output)
    opt.num_workers = 0
    opt.tsh_ddp8 = False
    opt.eval_before_training = False
    opt.use_wandb = False
    torch.manual_seed(42)
    random.seed(42)
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(int(getattr(accelerator, "local_process_index", 0)))
    loader, _, _, _ = get_multi_dataloader(opt, accelerator)
    model = model_registry[opt.model_type](opt)
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        raise RuntimeError("J2 parent was not strictly restored")
    configure_joint_formation_trainability(model, opt)
    model.to(accelerator.device)
    optimizer = setup_optimizer(opt, model, accelerator, 0)
    scheduler = setup_scheduler(opt, optimizer, min(len(loader), int(opt.max_iters_per_epoch)), accelerator, 0)
    return opt, accelerator, model, optimizer, scheduler, loader


def set_step(model, local_step: int) -> None:
    effective = 960 + int(local_step)
    model.set_token_eru_step(effective)
    model.set_token_eru_dino_metric_step(effective)
    model.set_token_eru_early_query_step(local_step)
    model.tsh_instance_loss_weight_eff = 1.0
    model.tsh_unit_grad_eff = 1.0
    model.teacher_lambda_eff = 0.0


def eval_snapshot(model, accelerator, batch, local_step: int) -> dict:
    set_step(model, local_step)
    model.eval()
    with torch.inference_mode():
        result = model(batch, compute_quality_metrics=False)
    probability = (
        result["rendered_instance_group_probability"][0]
        .detach().float().cpu().numpy()[:, :, 0]
    )
    labels = batch["instance_label_output"][0].detach().long().cpu().numpy()
    metrics = prediction_metrics(probability, labels)
    logits = result["unit_logits"].detach().float()
    finite = all(
        bool(torch.isfinite(value).all())
        for value in (result["images_pred"], result["gaussians"], logits)
    )
    adapter = getattr(getattr(model, "token_eru_decoder", None), "early_query_adapter", None)
    gate = float(getattr(model, "_token_eru_early_query_gate", 0.0))
    if adapter is not None:
        delta = getattr(model, "_token_eru_last_early_query_state_delta_norm", 0.0)
        residual = getattr(model, "_token_eru_last_early_query_u_residual_norm", 0.0)
    else:
        delta = residual = None
    return {
        "local_step": int(local_step),
        "effective_step": 960 + int(local_step),
        **metrics,
        "psnr": float(result["psnr"]),
        "early_query_gate": gate,
        "early_query_state_delta_norm": float(delta) if delta is not None else 0.0,
        "early_query_u_residual_norm": float(residual) if residual is not None else 0.0,
        "loss": float(result["loss"].detach()),
        "loss_rgb": float(result.get("loss_rgb", 0.0)),
        "loss_instance_group": float(result.get("loss_instance_group", 0.0)),
        "loss_instance_metric": float(result.get("loss_instance_metric", 0.0)),
        "finite": finite,
    }


def train_one(config_name: str, root: Path, reference_batch_hash: str | None):
    opt, accelerator, model, optimizer, scheduler, loader = build_runtime(config_name, root)
    unwrapped = accelerator.unwrap_model(model)
    raw_batch = next(iter(loader))
    current_batch_hash = batch_hash(raw_batch)
    if reference_batch_hash is not None and current_batch_hash != reference_batch_hash:
        raise RuntimeError(
            f"matched batch hash mismatch for {config_name}: "
            f"expected={reference_batch_hash} got={current_batch_hash}"
        )
    batch = move(raw_batch, accelerator.device)
    if batch["input"].shape[1] != 15 or batch["instance_label_output"].shape[1] != 7:
        raise RuntimeError("fixed batch is not 8 context + 7 target views")
    report = {
        "config": config_name,
        "batch_hash": current_batch_hash,
        "scene": str(batch["scene_name"][0]),
        "trainable_parameter_hash_step0": parameter_hash(unwrapped, True),
        "trainable_tensor_count": sum(p.requires_grad for p in unwrapped.parameters()),
        "trainable_numel": sum(p.numel() for p in unwrapped.parameters() if p.requires_grad),
        "milestones": [],
        "training_records": [],
        "formal_training_started": False,
        "formal_evaluation_started": False,
    }
    model.eval()
    report["milestones"].append(eval_snapshot(model, accelerator, batch, 0))
    for local_step in range(1, 101):
        model.train()
        set_step(unwrapped, local_step)
        optimizer.zero_grad(set_to_none=True)
        result = model(batch, compute_quality_metrics=False)
        loss = result["loss"]
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(f"non-finite loss at local step {local_step}")
        accelerator.backward(loss)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(torch.as_tensor(grad_norm, device=accelerator.device))):
            raise FloatingPointError(f"non-finite gradient at local step {local_step}")
        optimizer.step()
        scheduler.step()
        if not finite_model(unwrapped):
            raise FloatingPointError(f"non-finite parameters at local step {local_step}")
        report["training_records"].append({
            "local_step": local_step,
            "effective_step": 960 + local_step,
            "loss": float(loss.detach()),
            "pre_clip_grad_norm": float(grad_norm),
            "parameter_hash": parameter_hash(unwrapped, True),
            "lr": [float(group["lr"]) for group in optimizer.param_groups],
        })
        if local_step in MILESTONES:
            report["milestones"].append(
                eval_snapshot(model, accelerator, batch, local_step)
            )
    snapshot_path = root / "step_100_trainable.safetensors"
    save_file(
        {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in unwrapped.named_parameters()
            if parameter.requires_grad
        },
        str(snapshot_path),
    )
    report["step_100_trainable_snapshot"] = {
        "path": str(snapshot_path),
        "sha256": sha256_file(snapshot_path),
    }
    return report, current_batch_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(entry.name != "logs" for entry in output.iterdir()):
        raise RuntimeError(f"refusing non-empty fixed-batch workspace: {output}")
    if not PARENT.is_file() or sha256_file(PARENT) != PARENT_SHA256:
        raise RuntimeError("J2 local250 parent is missing or SHA256 mismatched")
    output.mkdir(parents=True, exist_ok=True)
    control_dir = output / "control"
    treatment_dir = output / "treatment"
    control_dir.mkdir()
    treatment_dir.mkdir()
    control, fixed_hash = train_one(CONTROL, control_dir, None)
    treatment, treatment_hash = train_one(TREATMENT, treatment_dir, fixed_hash)
    if fixed_hash != treatment_hash:
        raise RuntimeError("E0/E1 fixed batch fingerprints differ")
    report = {
        "parent": str(PARENT),
        "parent_sha256": PARENT_SHA256,
        "batch_hash": fixed_hash,
        "control": control,
        "treatment": treatment,
        "step0_control_treatment_identity": True,
        "formal_training_started": False,
        "formal_evaluation_started": False,
    }
    (output / "fixed_batch_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
