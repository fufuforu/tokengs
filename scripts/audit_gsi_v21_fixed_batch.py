"""Matched GSI-v2.1 control/treatment fixed-batch learnability audit."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from accelerate import Accelerator, DataLoaderConfiguration  # noqa: E402
from accelerate.utils import DistributedDataParallelKwargs  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.globalsplat_instance_v2.scene_instance_loss import (  # noqa: E402
    scene_global_hungarian_instance_loss,
)
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import _setup_gsi_v2_optimizer  # noqa: E402
from tokengs.utils.instance_ap import (  # noqa: E402
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)


RESUME = ROOT / "workspace/gsi_v2_recon_scannet_adapt_ddp8/checkpoints/model_step_000250.safetensors"
MILESTONES = (0, 1, 5, 25, 50, 100)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    value = value.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _batch_hashes(value: Any, prefix: str = "") -> dict[str, Any]:
    if torch.is_tensor(value):
        return {
            prefix: {
                "sha256": _tensor_hash(value),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        }
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(_batch_hashes(item, f"{prefix}.{key}" if prefix else str(key)))
        return result
    if isinstance(value, (tuple, list)):
        result = {}
        for index, item in enumerate(value):
            result.update(_batch_hashes(item, f"{prefix}[{index}]"))
        return result
    return {}


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _parameter_hash(model: torch.nn.Module, *, include_semantic: bool = True) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if not include_semantic and name.startswith("semantic_query_head."):
            continue
        value = parameter.detach().float().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _finite_model(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter.detach()).all()) for parameter in model.parameters())


def _grad_norm(model: torch.nn.Module, prefixes: tuple[str, ...]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and name.startswith(prefixes):
            total += float(parameter.grad.detach().float().square().sum())
    return total ** 0.5


def _update_norm(model: torch.nn.Module, before: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes) and name in before:
            total += float((parameter.detach().float().cpu() - before[name]).square().sum())
    return total ** 0.5


def _output_cache(output: dict[str, Any]) -> dict[str, torch.Tensor]:
    keys = (
        "images_pred", "alphas_pred", "depths_pred", "means", "scales", "rotations",
        "sh", "opacities", "rendered_instance_group_probability",
        "rendered_instance_group_alpha", "gsi_v2_instance_assignment_logits",
        "gsi_v2_instance_assignment_probabilities", "semantic_query_logits",
        "semantic_probabilities", "loss", "loss_reconstruction", "loss_instance_group",
        "loss_rgb", "loss_semantic_query_raw", "loss_semantic_query_weighted", "psnr",
    )
    return {key: value.detach().cpu() for key, value in output.items() if key in keys and torch.is_tensor(value)}


def _semantic_metrics(output: dict[str, Any], data: dict[str, Any]) -> dict[str, float]:
    if "semantic_probabilities" not in output:
        return {
            "matched_query_semantic_top1_accuracy": 0.0,
            "thing_miou": 0.0,
            "thing_pixel_accuracy": 0.0,
            "semantic_valid_gt_ratio": 0.0,
        }
    probabilities = output["semantic_probabilities"].detach().float()
    labels = data["semantic_label_output"].long()
    instances = data["instance_label_output"].long()
    predicted = probabilities.argmax(dim=2)
    thing_protocol_ids = torch.tensor((4, 5, 6, 7), device=labels.device)
    thing_ids = thing_protocol_ids - 1
    valid = torch.isin(labels, thing_protocol_ids) & (instances > 0)
    mapped_labels = labels - 1
    correct = (predicted == mapped_labels) & valid
    ious = []
    for class_id in thing_ids.tolist():
        target = valid & (mapped_labels == class_id)
        estimate = predicted == class_id
        union = (target | estimate) & valid
        if union.any():
            ious.append(float(((target & estimate).sum() / union.sum()).item()))
    return {
        "matched_query_semantic_top1_accuracy": float(
            output.get("gsi_v21_matched_semantic_accuracy", torch.zeros(())).detach()
        ),
        "thing_miou": float(sum(ious) / len(ious)) if ious else 0.0,
        "thing_pixel_accuracy": float(correct.sum().item() / max(1, valid.sum().item())),
        "semantic_valid_gt_ratio": float(
            output.get("gsi_v21_semantic_valid_object_ratio", torch.zeros(())).detach()
        ),
    }


def _metrics(output: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    probabilities = output["rendered_instance_group_probability"][0].detach().float().cpu().numpy()[:, :, 0]
    labels = data["instance_label_output"][0].detach().long().cpu().numpy()
    predictions, scores, prediction_ids = [], [], []
    ground_truth, ground_truth_ids = [], []
    nonempty = 0
    for view in range(probabilities.shape[1]):
        image_id = f"{data['scene_name'][0]}:target:{view}"
        masks, view_scores = masks_from_group_probs(
            probabilities[:, view], void_channel=probabilities.shape[0] - 1, min_mask_area=1
        )
        predictions.extend(masks)
        scores.extend(view_scores)
        prediction_ids.extend([image_id] * len(masks))
        nonempty += len(masks)
        view_gts = gt_masks_from_instance_map(labels[view], min_mask_area=1)
        ground_truth.extend(view_gts)
        ground_truth_ids.extend([image_id] * len(view_gts))
    ap = instance_ap(
        predictions, scores, ground_truth, thresholds=(.25, .5, .75), vectorized=True,
        pred_image_ids=prediction_ids, gt_image_ids=ground_truth_ids,
    )
    assignment = output["gsi_v2_instance_assignment_probabilities"].detach().float()
    usage = assignment[..., :-1].mean(dim=(0, 1))
    entropy = (-(assignment * assignment.clamp_min(1e-8).log()).sum(dim=-1)).mean()
    return {
        "instance_loss": float(output["loss_instance_group"].detach()),
        "loss_reconstruction": float(output["loss_reconstruction"].detach()),
        "loss_rgb": float(output["loss_rgb"].detach()),
        "psnr": float(output["psnr"].detach()),
        "ap25": float(ap["ap_25"]), "ap50": float(ap["ap_50"]), "ap75": float(ap["ap_75"]),
        "pred_gt": float(len(predictions) / max(1, len(ground_truth))),
        "nonempty_query_count": int(nonempty),
        "gt_count": int(len(ground_truth)),
        "effective_query_count": float(torch.exp(entropy).item()),
        "assignment_entropy": float(entropy.item()),
        "void_ratio": float(probabilities[-1].mean()),
        **_semantic_metrics(output, data),
    }


def _model_opt(config_name: str, workspace: Path):
    opt = copy.deepcopy(config_defaults[config_name])
    opt.prompt_overfit_single_batch = True
    opt.prompt_overfit_sample_index = 0
    opt.num_workers = 0
    opt.num_epochs = 1
    opt.max_iters_per_epoch = 100
    opt.eval_before_training = False
    opt.use_wandb = False
    opt.log_image_freq = 0
    opt.print_freq = 100000
    opt.gsi_v2_return_debug_tensors = True
    opt.workspace = str(workspace)
    return opt


def _get_fixed_batch(opt, accelerator):
    train_loader, _, train_dataset, _ = get_multi_dataloader(opt, accelerator)
    data = next(iter(train_loader))
    if tuple(data["images_input"].shape[1:]) != (8, 3, 256, 256):
        raise RuntimeError(f"fixed batch context shape is {tuple(data['images_input'].shape)}")
    if tuple(data["images_output"].shape[1:]) != (7, 3, 256, 256):
        raise RuntimeError(f"fixed batch target shape is {tuple(data['images_output'].shape)}")
    if tuple(data["instance_label_output"].shape[1:2]) != (7,):
        raise RuntimeError("fixed batch target instances are not 7 views")
    frames = data["frame_ids"][0].tolist()
    if len(frames) != 15 or len(set(int(item) for item in frames)) != 15:
        raise RuntimeError(f"invalid fixed-batch frames: {frames}")
    if len(set(str(value) for value in data["scene_name"])) != 1:
        raise RuntimeError("fixed batch is not same-scene")
    if int((data["instance_label_output"] > 0).sum()) <= 0:
        raise RuntimeError("fixed batch has no positive target instance")
    return data, train_dataset


def _init_model(opt, accelerator):
    model = model_registry[opt.model_type](opt).to(accelerator.device)
    state = load_file(str(RESUME), device="cpu")
    counts = {
        "reconstruction": sum(key.startswith("reconstruction.") for key in state),
        "pgsr": sum("pgsr" in key.lower() for key in state),
    }
    if counts != {"reconstruction": 454, "pgsr": 0}:
        raise RuntimeError(f"unexpected R1 checkpoint composition: {counts}")
    restore = model.load_phase_r_state_dict(state)
    optimizer = _setup_gsi_v2_optimizer(opt, model, accelerator, 0)
    model, optimizer = accelerator.prepare(model, optimizer)
    base = accelerator.unwrap_model(model)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    trainable_ids = {id(parameter) for parameter in base.parameters() if parameter.requires_grad}
    if optimizer_ids != trainable_ids:
        raise RuntimeError("optimizer/trainable parameter sets differ")
    return model, optimizer, base, restore


def _save_step(out_dir: Path, step: int, model, optimizer, output, metrics, data):
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(state, str(out_dir / f"model_step_{step:06d}.safetensors"))
    torch.save(optimizer.state_dict(), out_dir / f"optimizer_step_{step:06d}.pth")
    torch.save(_output_cache(output), out_dir / f"cache_step_{step:06d}.pt")
    (out_dir / f"metadata_step_{step:06d}.json").write_text(
        json.dumps({
            "optimizer_step": step,
            "scene_id": str(data["scene_name"][0]),
            "parameter_hash": _parameter_hash(model),
            "metrics": metrics,
            "finite": _finite_model(model),
        }, indent=2), encoding="utf-8"
    )


def _semantic_gradient_audit(model, optimizer, base, data, accelerator, step: int) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    base.set_train_step(step)
    model.train()
    with accelerator.autocast():
        audit_output = model(data)
    audit_loss = audit_output.get("loss_semantic_query_weighted")
    if audit_loss is None:
        raise RuntimeError("semantic-only audit requested without semantic loss")
    accelerator.backward(audit_loss)
    values = {
        "semantic_head": _grad_norm(base, ("semantic_query_head.",)),
        "instance_head": _grad_norm(base, ("instance_head.",)),
        "instance_stream": _grad_norm(base, (
            "reconstruction.slot_encoder.slot_to_ins.",
            "reconstruction.slot_encoder.ins_rounds.",
            "reconstruction.slot_encoder.tri_adapters.",
        )),
        "shared_reconstruction": _grad_norm(base, ("reconstruction.",)),
    }
    values["semantic_head_nonzero"] = values["semantic_head"] > 0.0
    values["instance_head_nonzero"] = values["instance_head"] > 0.0
    values["instance_stream_nonzero"] = values["instance_stream"] > 0.0
    values["appearance_zero"] = _grad_norm(base, ("reconstruction.slot_encoder.slot_to_tex.",)) == 0.0
    optimizer.zero_grad(set_to_none=True)
    del audit_output, audit_loss
    return values


def _run_one(config_name: str, root_out: Path, cpu_batch: dict[str, Any], accelerator) -> dict[str, Any]:
    out_dir = root_out / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = _model_opt(config_name, out_dir)
    model, optimizer, base, restore = _init_model(opt, accelerator)
    common_step0_hash = _parameter_hash(base, include_semantic=False)
    data = _to_device(cpu_batch, accelerator.device)
    before_step0 = {name: parameter.detach().float().cpu().clone() for name, parameter in base.named_parameters()}
    base.set_train_step(0)
    model.eval()
    with torch.inference_mode():
        with accelerator.autocast():
            step0 = model(data)
    if config_name.endswith("semantic_short200_ddp8") and "semantic_query_logits" not in step0:
        raise RuntimeError("semantic treatment did not produce semantic_query_logits")
    if config_name.endswith("maskonly_short200_ddp8") and any(
        name.startswith("semantic_query_head.") for name in base.state_dict()
    ):
        raise RuntimeError("mask-only control unexpectedly contains semantic head state")
    step0_metrics = _metrics(step0, data)
    _save_step(out_dir, 0, base, optimizer, step0, step0_metrics, data)
    records: dict[str, Any] = {"0": {"metrics": step0_metrics, "parameter_hash": _parameter_hash(base)}}
    previous = before_step0
    for step in range(1, 101):
        gradient_audit = None
        if config_name.endswith("semantic_short200_ddp8") and step == 1:
            gradient_audit = _semantic_gradient_audit(model, optimizer, base, data, accelerator, step)
            if not all(gradient_audit[key] for key in (
                "semantic_head_nonzero", "instance_head_nonzero", "instance_stream_nonzero"
            )):
                raise RuntimeError(f"semantic gradient did not reach instance stream: {gradient_audit}")
        base.set_train_step(step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            output = model(data)
        loss = output["loss"]
        if not torch.isfinite(loss).all():
            raise RuntimeError(f"non-finite loss at {config_name} step {step}")
        accelerator.backward(loss)
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in base.parameters()):
            raise RuntimeError(f"non-finite gradient at {config_name} step {step}")
        grad_stats = {
            "instance_head": _grad_norm(base, ("instance_head.",)),
            "instance_stream": _grad_norm(base, (
                "reconstruction.slot_encoder.slot_to_ins.",
                "reconstruction.slot_encoder.ins_rounds.",
                "reconstruction.slot_encoder.tri_adapters.",
            )),
            "semantic_head": _grad_norm(base, ("semantic_query_head.",)),
            "reconstruction": _grad_norm(base, ("reconstruction.",)),
        }
        accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)
        optimizer.step()
        if not _finite_model(base):
            raise RuntimeError(f"non-finite parameters at {config_name} step {step}")
        train_record = {
            "train_loss": float(loss.detach()),
            "train_loss_reconstruction": float(output["loss_reconstruction"].detach()),
            "train_loss_instance": float(output["loss_instance_group"].detach()),
            "train_loss_semantic_raw": float(output.get("loss_semantic_query_raw", torch.zeros(())).detach()),
            "train_loss_semantic_weighted": float(output.get("loss_semantic_query_weighted", torch.zeros(())).detach()),
            "gradients": grad_stats,
            "semantic_gradient_audit_step1": gradient_audit,
        }
        del output, loss
        if step in MILESTONES[1:]:
            model.eval()
            base.set_train_step(step)
            with torch.inference_mode():
                with accelerator.autocast():
                    snapshot = model(data)
            metrics = _metrics(snapshot, data)
            update_stats = {
                "instance_head": _update_norm(base, previous, ("instance_head.",)),
                "instance_stream": _update_norm(base, previous, (
                    "reconstruction.slot_encoder.slot_to_ins.",
                    "reconstruction.slot_encoder.ins_rounds.",
                    "reconstruction.slot_encoder.tri_adapters.",
                )),
                "semantic_head": _update_norm(base, previous, ("semantic_query_head.",)),
                "reconstruction": _update_norm(base, previous, ("reconstruction.",)),
            }
            previous = {name: parameter.detach().float().cpu().clone() for name, parameter in base.named_parameters()}
            record = {
                "metrics": metrics,
                "train": train_record,
                "module_updates_from_previous_milestone": update_stats,
                "parameter_hash": _parameter_hash(base),
                "semantic_state_keys": sorted(key for key in base.state_dict() if key.startswith("semantic_query_head.")),
            }
            records[str(step)] = record
            _save_step(out_dir, step, base, optimizer, snapshot, record, data)
            del snapshot
    # Strict independent restore from the last fixed-batch state.
    restored = model_registry[opt.model_type](opt).to(accelerator.device)
    restored.load_phase_r_state_dict(load_file(str(RESUME), device="cpu"))
    restored.load_state_dict(load_file(str(out_dir / "model_step_000100.safetensors"), device="cpu"), strict=True)
    restored.eval()
    restored.set_train_step(100)
    with torch.inference_mode():
        with accelerator.autocast():
            restored_output = restored(data)
    cached = torch.load(out_dir / "cache_step_000100.pt", map_location="cpu", weights_only=False)
    restore_diffs = {
        key: float((restored_output[key].detach().cpu().float() - cached[key].float()).abs().max())
        for key in ("images_pred", "rendered_instance_group_probability")
        if key in cached and key in restored_output
    }
    if any(value > 1e-5 for value in restore_diffs.values()):
        raise RuntimeError(f"independent restore mismatch: {restore_diffs}")
    report = {
        "config_name": config_name,
        "workspace": str(out_dir),
        "checkpoint": {
            "path": str(RESUME),
            "sha256": _hash_file(RESUME),
            "reconstruction_keys": 454,
            "pgsr_absent": True,
            "phase_r_restore": restore,
            "fresh_reset": False,
        },
        "trainable_parameters": {
            "count": int(sum(parameter.numel() for parameter in base.parameters() if parameter.requires_grad)),
            "names": [name for name, parameter in base.named_parameters() if parameter.requires_grad],
        },
        "optimizer_groups": [
            {
                "name": group.get("name"),
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameter_count": int(sum(parameter.numel() for parameter in group["params"])),
            }
            for group in optimizer.param_groups
        ],
        "semantic_head_state_keys": sorted(key for key in base.state_dict() if key.startswith("semantic_query_head.")),
        "common_step0_parameter_hash": common_step0_hash,
        "milestones": records,
        "independent_restore": {
            "strict": True,
            "max_diffs": restore_diffs,
            "match": all(value <= 1e-5 for value in restore_diffs.values()),
        },
        "all_finite": _finite_model(base),
    }
    (out_dir / "fixed_batch_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps != 100:
        raise ValueError("GSI-v2.1 fixed-batch audit is fixed at exactly 100 steps")
    out_dir = (ROOT / args.workspace).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty fixed-batch workspace: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for config_name in ("gsi_v21_maskonly_short200_ddp8", "gsi_v21_semantic_short200_ddp8"):
        if config_defaults[config_name].workspace and Path(config_defaults[config_name].workspace).exists():
            # Existing formal workspaces are never reused or overwritten.
            pass
    _seed_all(int(config_defaults["gsi_v21_maskonly_short200_ddp8"].seed))
    probe_opt = _model_opt("gsi_v21_maskonly_short200_ddp8", out_dir)
    accelerator = Accelerator(
        mixed_precision=probe_opt.mixed_precision,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.num_processes != 1:
        raise RuntimeError("fixed-batch audit must run as a single process")
    raw_batch, dataset = _get_fixed_batch(probe_opt, accelerator)
    cpu_batch = _to_cpu(raw_batch)
    fixed = {
        "scene_id": str(cpu_batch["scene_name"][0]),
        "frame_ids": [int(item) for item in cpu_batch["frame_ids"][0].tolist()],
        "context_views": 8,
        "target_views": 7,
        "same_scene": True,
        "formal_dataloader": "tokengs.data.get_multi_dataloader",
        "batch_hashes": _batch_hashes(cpu_batch),
    }
    torch.save(cpu_batch, out_dir / "fixed_batch_cpu.pt")
    (out_dir / "fixed_batch_manifest.json").write_text(json.dumps(fixed, indent=2), encoding="utf-8")
    (out_dir / "fixed_batch_hashes.json").write_text(json.dumps(fixed["batch_hashes"], indent=2), encoding="utf-8")
    reports = {}
    for config_name in ("gsi_v21_maskonly_short200_ddp8", "gsi_v21_semantic_short200_ddp8"):
        _seed_all(int(config_defaults[config_name].seed))
        reports[config_name] = _run_one(config_name, out_dir, cpu_batch, accelerator)
    common_hashes = {
        config_name: reports[config_name]["common_step0_parameter_hash"]
        for config_name in reports
    }
    report = {
        "fixed_batch": fixed,
        "control": reports["gsi_v21_maskonly_short200_ddp8"],
        "treatment": reports["gsi_v21_semantic_short200_ddp8"],
        "common_initialization_check": {
            "hashes": common_hashes,
            "match": len(set(common_hashes.values())) == 1,
        },
        "ready_for_short200": False,
    }
    (out_dir / "gsi_v21_fixed_batch_audit.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable({
        "workspace": str(out_dir),
        "control": reports["gsi_v21_maskonly_short200_ddp8"]["milestones"].get("100"),
        "treatment": reports["gsi_v21_semantic_short200_ddp8"]["milestones"].get("100"),
        "ready_for_short200": False,
    }), indent=2))


if __name__ == "__main__":
    main()
