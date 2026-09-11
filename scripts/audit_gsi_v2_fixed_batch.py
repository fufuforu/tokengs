"""Diagnostic 8+7 fixed-batch audit for the isolated GSI-v2 short recipe.

This script deliberately uses the formal ``get_multi_dataloader`` path, pins
its first legal training sample, and never writes to a formal training
workspace.  The model is optimized only on that saved batch for the requested
diagnostic budget.  Milestone checkpoints/caches are self-contained so a
second invocation can verify an independent restore.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from accelerate import Accelerator, DataLoaderConfiguration  # noqa: E402
from accelerate.utils import DistributedDataParallelKwargs  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from scripts.eval_instance_lsm_protocol import _mask_diagnostics  # noqa: E402
from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.globalsplat_instance_v2.camera_adapter import (  # noqa: E402
    make_official_context_frustum_meta,
    make_official_target_meta,
)
from tokengs.models.globalsplat_instance_v2.dependency import (  # noqa: E402
    load_globalsplat_symbols,
)
from tokengs.models.globalsplat_instance_v2.scene_instance_loss import (  # noqa: E402
    scene_global_hungarian_instance_loss,
)
from tokengs.models.input_types import split_data  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import _setup_gsi_v2_optimizer  # noqa: E402
from tokengs.utils.instance_ap import (  # noqa: E402
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)


CFG = "gsi_v2_joint_scannet_short355_ddp8"
RESUME = ROOT / "workspace/gsi_v2_recon_scannet_adapt_ddp8/checkpoints/model_step_000250.safetensors"
REFERENCE_BATCH = ROOT / "workspace/tsh_ga_idu1_fixed_batch_reproducible_audit_v13/fixed_batch_cpu.pt"
MANIFEST = ROOT / "data/scannet_prompt/scannet_prompt_full_wide_8x7.json"
MILESTONES = (0, 1, 5, 25, 50, 100)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _batch_hashes(value, prefix="") -> dict[str, object]:
    result = {}
    if torch.is_tensor(value):
        result[prefix] = {
            "sha256": _tensor_hash(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    elif isinstance(value, dict):
        for key, item in value.items():
            result.update(_batch_hashes(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            result.update(_batch_hashes(item, f"{prefix}[{index}]"))
    return result


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _parameter_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        tensor = parameter.detach().float().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _finite_parameters(model) -> bool:
    return all(bool(torch.isfinite(parameter.detach()).all()) for parameter in model.parameters())


def _finite_gradients(model) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad.detach()).all())
        for parameter in model.parameters()
    )


def _module_value(model, prefixes, value_fn) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            total += value_fn(name, parameter)
    return float(total)


def _module_stats(model, before: dict[str, torch.Tensor] | None) -> dict[str, object]:
    modules = {
        "instance_head": ("instance_head.",),
        "instance_stream": (
            "reconstruction.slot_encoder.slot_to_ins.",
            "reconstruction.slot_encoder.ins_rounds.",
        ),
        "tri_adapters": ("reconstruction.slot_encoder.tri_adapters.",),
        "shared_reconstruction": ("reconstruction.",),
    }
    result = {}
    for module, prefixes in modules.items():
        names = [name for name, _ in model.named_parameters() if any(name.startswith(p) for p in prefixes)]
        if module == "shared_reconstruction":
            excluded = set()
            for other in ("instance_stream", "tri_adapters"):
                excluded.update(
                    name for name, _ in model.named_parameters()
                    if any(name.startswith(p) for p in modules[other])
                )
            names = [name for name in names if name not in excluded]
        parameter_map = dict(model.named_parameters())
        grad_sq = sum(
            float(parameter_map[name].grad.detach().float().square().sum())
            for name in names if parameter_map[name].grad is not None
        )
        update_sq = 0.0
        if before is not None:
            update_sq = sum(
                float((parameter_map[name].detach().float().cpu() - before[name]).square().sum())
                for name in names if name in before
            )
        count = sum(parameter_map[name].numel() for name in names)
        result[module] = {
            "parameter_count": int(count),
            "gradient_norm": grad_sq ** 0.5,
            "update_norm_from_previous_milestone": update_sq ** 0.5,
            "gradient_nonzero": bool(grad_sq > 0.0),
        }
    return result


def _saveable_output(output: dict[str, object]) -> dict[str, torch.Tensor]:
    keys = (
        "images_pred", "alphas_pred", "depths_pred", "means", "scales",
        "rotations", "sh", "opacities", "rendered_instance_group_probability",
        "rendered_instance_group_alpha", "gsi_v2_instance_assignment_logits",
        "gsi_v2_instance_assignment_probabilities", "gsi_v2_candidate_gate_logits",
        "loss", "loss_reconstruction", "loss_instance_group", "loss_rgb", "psnr",
    )
    return {
        key: value.detach().cpu()
        for key, value in output.items()
        if key in keys and torch.is_tensor(value)
    }


def _target_meta_and_loss(model, data, output):
    model_input, supervision = split_data(data, model.opt)
    context_K, context_w2c = make_official_context_frustum_meta(model_input)
    symbols = model.symbols
    gaussians = symbols.Gaussians(
        means=output["means"], rotations=output["rotations"], scales=output["scales"],
        sh=output["sh"], opacities=output["opacities"], reg=output["loss_reconstruction"].new_zeros(()),
    )
    reconstruction_loss, reconstruction_stats, _ = model.reconstruction_loss(
        gaussians, output["images_pred"], supervision.images_output,
        context_K, context_w2c,
    )
    instance_loss, instance_stats, _ = scene_global_hungarian_instance_loss(
        output["rendered_instance_group_probability"], data["instance_label_output"].long(),
        num_queries=100,
        min_visible_pixels=int(model.opt.gsi_v2_min_visible_pixels),
        bce_weight=float(model.opt.gsi_v2_match_bce_weight),
        dice_weight=float(model.opt.gsi_v2_match_dice_weight),
        void_weight=float(model.opt.gsi_v2_void_weight),
        unmatched_weight=float(model.opt.gsi_v2_unmatched_weight),
        absent_view_weight=float(model.opt.gsi_v2_absent_view_weight),
    )
    return {
        "reconstruction_loss": float(reconstruction_loss.detach()),
        "instance_loss": float(instance_loss.detach()),
        "reconstruction_stats": {
            key: float(value.detach()) for key, value in reconstruction_stats.items()
            if torch.is_tensor(value)
        },
        "instance_stats": {
            key: float(value.detach()) for key, value in instance_stats.items()
            if torch.is_tensor(value)
        },
    }


def _instance_metrics(output, data, base_output=None) -> dict[str, object]:
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
    diagnostic = _mask_diagnostics(
        predictions, ground_truth, prediction_ids, ground_truth_ids,
        thresholds=(.25, .5, .75),
    )
    assignment = output["gsi_v2_instance_assignment_probabilities"].detach().float()
    usage = assignment[..., :-1].mean(dim=(0, 1))
    entropy = (-(assignment * assignment.clamp_min(1e-8).log()).sum(dim=-1)).mean()
    logits = output["gsi_v2_instance_assignment_logits"].detach().float()
    candidate = output["gsi_v2_candidate_gate_logits"].detach().float()
    result = {
        "instance_loss": float(output["loss_instance_group"].detach()),
        "loss_reconstruction": float(output["loss_reconstruction"].detach()),
        "loss_rgb": float(output["loss_rgb"].detach()),
        "psnr": float(output["psnr"].detach()),
        "ap25": float(ap["ap_25"]), "ap50": float(ap["ap_50"]), "ap75": float(ap["ap_75"]),
        "mean_best_gt_iou": float(diagnostic["mean_best_gt_iou"]),
        "recall_iou25": float(diagnostic["recall_iou25"]),
        "recall_iou50": float(diagnostic["recall_iou50"]),
        "recall_iou75": float(diagnostic["recall_iou75"]),
        "pred_gt": float(len(predictions) / max(1, len(ground_truth))),
        "prediction_count_nonempty": int(nonempty),
        "gt_count": int(len(ground_truth)),
        "active_query_count_mass_gt_0.001": int((usage > .001).sum()),
        "effective_query_count": float(torch.exp(entropy).item()),
        "assignment_entropy": float(entropy.item()),
        "void_ratio": float(probabilities[-1].mean()),
        "assignment_argmax_histogram": torch.bincount(
            assignment[..., :-1].argmax(dim=-1).flatten(), minlength=100
        ).cpu().tolist(),
        "candidate_gate_argmax_histogram": torch.bincount(
            candidate[..., 0].argmax(dim=-1).flatten(), minlength=candidate.shape[-2]
        ).cpu().tolist(),
    }
    if base_output is not None:
        base_assignment = base_output["gsi_v2_instance_assignment_probabilities"].detach().float()
        base_logits = base_output["gsi_v2_instance_assignment_logits"].detach().float()
        base_candidate = base_output["gsi_v2_candidate_gate_logits"].detach().float()
        rendered = output["rendered_instance_group_probability"].detach().float()
        base_rendered = base_output["rendered_instance_group_probability"].detach().float()
        base_margin = torch.topk(base_logits, 2, dim=-1).values
        base_margin = base_margin[..., 0] - base_margin[..., 1]
        delta = logits - base_logits
        base_unit = base_assignment[..., :-1].argmax(dim=-1)
        cur_unit = assignment[..., :-1].argmax(dim=-1)
        base_binary = base_rendered.argmax(dim=1) != (base_rendered.shape[1] - 1)
        cur_binary = rendered.argmax(dim=1) != (rendered.shape[1] - 1)
        result.update({
            "assignment_argmax_changed_vs_step0": float((cur_unit != base_unit).float().mean()),
            "candidate_gate_argmax_changed_vs_step0": float(
                (candidate[..., 0].argmax(dim=-1) != base_candidate[..., 0].argmax(dim=-1)).float().mean()
            ),
            "binary_mask_pixel_change_vs_step0": float((cur_binary != base_binary).float().mean()),
            "rendered_soft_mask_max_diff_vs_step0": float((rendered - base_rendered).abs().max()),
            "assignment_residual_logits": {
                "mean_abs": float(delta.abs().mean()), "std_abs": float(delta.abs().std()),
                "p50_abs": float(torch.quantile(delta.abs().flatten(), .50)),
                "p90_abs": float(torch.quantile(delta.abs().flatten(), .90)),
                "p99_abs": float(torch.quantile(delta.abs().flatten(), .99)),
                "max_abs": float(delta.abs().max()),
            },
            "base_assignment_margin": {
                "mean": float(base_margin.mean()),
                "p50": float(torch.quantile(base_margin.flatten(), .50)),
                "p90": float(torch.quantile(base_margin.flatten(), .90)),
                "p99": float(torch.quantile(base_margin.flatten(), .99)),
            },
            "residual_crosses_base_margin_fraction": float(
                (delta.abs().amax(dim=-1) > base_margin).float().mean()
            ),
            "candidate_gate_residual_abs": {
                "mean": float((candidate - base_candidate).abs().mean()),
                "max": float((candidate - base_candidate).abs().max()),
            },
            "soft_assignment_mean_abs_diff_vs_step0": float((assignment - base_assignment).abs().mean()),
            "soft_assignment_max_abs_diff_vs_step0": float((assignment - base_assignment).abs().max()),
        })
    return result


def _checkpoint_counts(path: Path) -> dict[str, int]:
    state = load_file(str(path), device="cpu")
    return {
        "reconstruction": sum(key.startswith("reconstruction.") for key in state),
        "absolute_gs_head": sum(key.startswith("absolute_gs_head.") for key in state),
        "tsh_instance_head": sum(key.startswith("tsh_instance_head.") for key in state),
        "decoder_tail": sum(key.startswith("enc_dec_backbone.decoder_blocks.") for key in state),
        "pgsr": sum("pgsr" in key.lower() for key in state),
    }


def _make_opt():
    opt = copy.deepcopy(config_defaults[CFG])
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
    return opt


def _prepare(opt):
    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.num_processes != 1:
        raise RuntimeError("fixed-batch GSI audit must run as one process")
    train_loader, _, train_dataset, _ = get_multi_dataloader(opt, accelerator)
    data = next(iter(train_loader))
    if data["images_input"].shape[1:] != (8, 3, 256, 256):
        raise RuntimeError(f"expected 8 context views, got {tuple(data['images_input'].shape)}")
    if data["images_output"].shape[1:] != (7, 3, 256, 256):
        raise RuntimeError(f"expected 7 target views, got {tuple(data['images_output'].shape)}")
    if data["instance_label_output"].shape[1] != 7:
        raise RuntimeError("target instance maps are not aligned to seven target views")
    frames = data["frame_ids"][0].detach().cpu().tolist()
    if len(frames) != 15 or len(set(frames)) != 15:
        raise RuntimeError(f"invalid non-repeating formal frame IDs: {frames}")
    if len(set(str(value) for value in data["scene_name"])) != 1:
        raise RuntimeError("fixed batch is not same-scene")
    if int((data["instance_label_output"] > 0).sum()) <= 0:
        raise RuntimeError("fixed batch has no positive target instance")
    return accelerator, train_loader, train_dataset, data


def _save_milestone(out_dir, step, model, optimizer, output, metrics, data):
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(state, str(out_dir / f"model_step_{step:06d}.safetensors"))
    torch.save(optimizer.state_dict(), out_dir / f"optimizer_step_{step:06d}.pth")
    torch.save(_saveable_output(output), out_dir / f"cache_step_{step:06d}.pt")
    (out_dir / f"metadata_step_{step:06d}.json").write_text(
        json.dumps({
            "optimizer_step": int(step),
            "parameter_hash": _parameter_hash(model),
            "metrics": metrics,
            "finite": _finite_parameters(model),
            "batch_scene": str(data["scene_name"][0]),
        }, indent=2), encoding="utf-8"
    )


def _run(args):
    out_dir = (ROOT / args.workspace).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty audit workspace: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = _make_opt()
    opt.workspace = str(out_dir)
    _seed_all(int(opt.seed))
    accelerator, train_loader, train_dataset, raw_data = _prepare(opt)
    cpu_batch = _to_cpu(raw_data)
    batch_hash = _batch_hashes(cpu_batch)
    torch.save(cpu_batch, out_dir / "fixed_batch_cpu.pt")
    reference_hash_match = None
    if REFERENCE_BATCH.is_file():
        reference_hash_match = batch_hash == _batch_hashes(torch.load(REFERENCE_BATCH, map_location="cpu", weights_only=False))
    (out_dir / "fixed_batch_hashes.json").write_text(json.dumps(batch_hash, indent=2), encoding="utf-8")
    (out_dir / "config_resolved.json").write_text(
        json.dumps(_jsonable(vars(opt)), indent=2, default=str), encoding="utf-8"
    )
    raw_manifest = MANIFEST.read_bytes()
    (out_dir / "manifest_snapshot.json").write_bytes(raw_manifest)
    fixed_manifest = {
        "manifest_path": str(MANIFEST.resolve()),
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "sample_index": 0,
        "scene_id": str(raw_data["scene_name"][0]),
        "frame_ids": raw_data["frame_ids"][0].detach().cpu().tolist(),
        "context_views": 8,
        "target_views": 7,
        "same_scene": True,
        "no_target_leakage": True,
        "fixed_batch_hashes": batch_hash,
        "matches_previous_valid_8x7_batch": reference_hash_match,
        "formal_dataloader": "tokengs.data.get_multi_dataloader",
        "validation_scene_excluded": True,
        "lsm40_excluded": True,
        "selection_scene_excluded": True,
    }
    (out_dir / "fixed_batch_manifest.json").write_text(json.dumps(fixed_manifest, indent=2), encoding="utf-8")

    model = model_registry[opt.model_type](opt).to(accelerator.device)
    raw_resume = load_file(str(RESUME), device="cpu")
    counts = _checkpoint_counts(RESUME)
    if counts != {"reconstruction": 454, "absolute_gs_head": 0, "tsh_instance_head": 0, "decoder_tail": 0, "pgsr": 0}:
        raise RuntimeError(f"unexpected Phase-R checkpoint composition: {counts}")
    restore = model.load_phase_r_state_dict(raw_resume)
    optimizer = _setup_gsi_v2_optimizer(opt, model, accelerator, 0)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    base = accelerator.unwrap_model(model)
    trainable = [name for name, parameter in base.named_parameters() if parameter.requires_grad]
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    trainable_ids = {id(parameter) for parameter in base.parameters() if parameter.requires_grad}
    if optimizer_ids != trainable_ids:
        raise RuntimeError("optimizer/trainable parameter sets differ")
    if not trainable or not all(name.startswith(("reconstruction.", "instance_head.")) for name in trainable):
        raise RuntimeError("unexpected trainable GSI parameters")

    data = _to_device(cpu_batch, accelerator.device)
    base.set_train_step(0)
    base._gate = 0.0
    model.eval()
    with torch.inference_mode():
        with accelerator.autocast():
            step0 = model(data)
    step0_cache = _saveable_output(step0)
    step0_metrics = _instance_metrics(step0, data)
    step0_audit_loss = _target_meta_and_loss(base, data, step0)
    before_milestone = {name: parameter.detach().float().cpu().clone() for name, parameter in base.named_parameters()}
    _save_milestone(out_dir, 0, base, optimizer, step0, {"metrics": step0_metrics, "audit_loss": step0_audit_loss}, data)

    records = {"0": {"metrics": step0_metrics, "audit_loss": step0_audit_loss}}
    step_times = {}
    for iteration in range(1, int(args.steps) + 1):
        started = time.time()
        base.set_train_step(iteration)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            output = model(data)
        train_loss = output["loss"]
        if not torch.isfinite(train_loss).all():
            raise RuntimeError(f"non-finite train loss at step {iteration}")
        accelerator.backward(train_loss)
        if not _finite_gradients(base):
            raise RuntimeError(f"non-finite gradients at step {iteration}")
        gradient_stats = _module_stats(base, None)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
        clipped_norm = float(accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip))
        optimizer.step()
        if not _finite_parameters(base):
            raise RuntimeError(f"non-finite parameters after step {iteration}")
        train_record = {
            "optimizer_step": iteration,
            "train_loss": float(train_loss.detach()),
            "train_loss_reconstruction": float(output["loss_reconstruction"].detach()),
            "train_loss_instance": float(output["loss_instance_group"].detach()),
            "gradient_norm_before_clip": grad_norm,
            "gradient_norm_after_clip": clipped_norm,
            "module_gradient_stats": gradient_stats,
            "finite": True,
        }
        del output, train_loss
        if iteration in MILESTONES[1:]:
            model.eval()
            base.set_train_step(iteration)
            with torch.inference_mode():
                with accelerator.autocast():
                    snapshot = model(data)
            metrics = _instance_metrics(snapshot, data, step0)
            audit_loss = _target_meta_and_loss(base, data, snapshot)
            current_params = {name: parameter.detach().float().cpu().clone() for name, parameter in base.named_parameters()}
            update_stats = _module_stats(base, before_milestone)
            before_milestone = current_params
            metrics_record = {
                **train_record,
                "eval_metrics": metrics,
                "audit_loss": audit_loss,
                "module_update_stats": update_stats,
                "parameter_hash": _parameter_hash(base),
                "cache_keys": sorted(_saveable_output(snapshot)),
            }
            records[str(iteration)] = metrics_record
            _save_milestone(out_dir, iteration, base, optimizer, snapshot, metrics_record, data)
            del snapshot
        step_times[str(iteration)] = time.time() - started

    report = {
        "preset": CFG,
        "workspace": str(out_dir),
        "steps": int(args.steps),
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(accelerator.device),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "checkpoint": {
            "path": str(RESUME),
            "sha256": _sha256_file(RESUME),
            "raw_counts": counts,
            "phase_r_restore": restore,
            "loaded_reconstruction_454_of_454": restore["loaded_reconstruction_keys"] == 454,
            "fresh_instance_stream": True,
            "pgsr_absent": counts["pgsr"] == 0,
            "fresh_reset": False,
        },
        "formal_batch": fixed_manifest,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in base.parameters() if parameter.requires_grad)),
        "trainable_parameter_name_count": len(trainable),
        "optimizer": [
            {"lr": float(group["lr"]), "weight_decay": float(group["weight_decay"]), "parameter_count": int(sum(p.numel() for p in group["params"]))}
            for group in optimizer.param_groups
        ],
        "scheduler": {"state": None, "reason": "fixed-batch diagnostic uses constant optimizer LR"},
        "milestones": records,
        "step_times_sec": step_times,
        "all_finite": True,
        "independent_restore_pending": True,
    }
    (out_dir / "gsi_v2_fixed_batch_audit.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))


def _verify(args):
    out_dir = (ROOT / args.workspace).resolve()
    cache_path = out_dir / "cache_step_000100.pt"
    model_path = out_dir / "model_step_000100.safetensors"
    if not cache_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("step100 cache/checkpoint missing for independent restore")
    opt = _make_opt()
    _seed_all(int(opt.seed))
    accelerator, _loader, _dataset, data = _prepare(opt)
    batch = _to_device(_to_cpu(data), accelerator.device)
    model = model_registry[opt.model_type](opt).to(accelerator.device)
    phase_state = load_file(str(RESUME), device="cpu")
    model.load_phase_r_state_dict(phase_state)
    state = load_file(str(model_path), device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    base = model
    base.set_train_step(100)
    with torch.inference_mode():
        with accelerator.autocast():
            current = model(batch)
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    diffs = {}
    for key in ("images_pred", "rendered_instance_group_probability", "gsi_v2_instance_assignment_logits", "means", "sh"):
        if key in cached and key in current:
            diffs[key] = float((current[key].detach().cpu().float() - cached[key].float()).abs().max())
    result = {
        "workspace": str(out_dir),
        "strict_load": True,
        "cache_max_diffs": diffs,
        "parameter_hash": _parameter_hash(model),
        "cache_parameter_hash": json.loads((out_dir / "metadata_step_000100.json").read_text())["parameter_hash"],
        "restore_match": all(value <= 1e-5 for value in diffs.values()),
        "all_finite": _finite_parameters(model) and all(torch.isfinite(value).all().item() for value in current.values() if torch.is_tensor(value)),
    }
    (out_dir / "independent_restore_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--mode", choices=("run", "verify"), default="run")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.mode == "run":
        if not 1 <= args.steps <= 100:
            raise ValueError("fixed-batch audit steps must be in [1,100]")
        _run(args)
    else:
        _verify(args)


if __name__ == "__main__":
    main()
