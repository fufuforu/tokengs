"""100-step TokenGS-ERU fixed-batch audit.

The script uses the production DataLoader and model forward.  It is a probe
only: it refuses a non-empty output directory and never writes a formal
training workspace.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_instance_lsm_protocol import (  # noqa: E402
    gt_masks_from_instance_map,
    instance_ap,
    _mask_diagnostics,
    masks_from_group_probs,
)
from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import load_model_checkpoint, setup_optimizer  # noqa: E402


CHECKPOINT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_"
    "w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors"
)
BASE_CONFIG = "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
ERU_CONFIG = "semantic_v6_absolute_units_true_shared_token_eru1_ddp8"


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


def _tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(repr(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _batch_manifest(batch: dict) -> dict:
    tensors = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_hash(value),
        }
        for key, value in batch.items()
        if torch.is_tensor(value)
    }
    return {
        "tensor_hashes": tensors,
        "scene_name": [str(x) for x in batch.get("scene_name", [])],
        "frame_ids": batch.get("frame_ids").detach().cpu().tolist()
        if torch.is_tensor(batch.get("frame_ids"))
        else None,
        "input_shape": list(batch["input"].shape),
        "target_instance_shape": list(batch["instance_label_output"].shape)
        if "instance_label_output" in batch
        else None,
    }


def _nonfinite_output_keys(output: dict) -> list[str]:
    # With compute_quality_metrics=False the production forward intentionally
    # returns NaN placeholders for disabled SSIM/LPIPS.  Do not confuse those
    # placeholders with a non-finite model/loss result; report the keys
    # explicitly and check all actual prediction/loss tensors.
    ignored = {"ssim", "lpips"}
    return [
        key
        for key, value in output.items()
        if key not in ignored
        and torch.is_tensor(value)
        and not torch.isfinite(value).all()
    ]


def _model_metrics(
    output: dict,
    data: dict,
    baseline_output: dict | None = None,
) -> dict:
    probability = output["rendered_instance_group_probability"][0].float().detach().cpu()
    labels = data["instance_label_output"][0].long().detach().cpu().numpy()
    predictions, scores, prediction_ids, gts, gt_ids = [], [], [], [], []
    for view in range(probability.shape[1]):
        masks, view_scores = masks_from_group_probs(
            probability[:, view, 0].numpy(),
            void_channel=probability.shape[0] - 1,
            min_mask_area=1,
        )
        image_id = f"{data['scene_name'][0]}:v{view}"
        predictions.extend(masks)
        scores.extend(view_scores)
        prediction_ids.extend([image_id] * len(masks))
        view_gts = gt_masks_from_instance_map(labels[view], min_mask_area=1)
        gts.extend(view_gts)
        gt_ids.extend([image_id] * len(view_gts))
    ap = instance_ap(
        predictions,
        scores,
        gts,
        thresholds=(0.25, 0.5, 0.75),
        vectorized=True,
        pred_image_ids=prediction_ids,
        gt_image_ids=gt_ids,
    )
    mask_diag = _mask_diagnostics(
        predictions,
        gts,
        prediction_ids,
        gt_ids,
        thresholds=(0.25, 0.5, 0.75),
    )
    unit_probability = output["instance_group_probabilities"][0].float().detach()
    entropy = -(
        unit_probability.clamp_min(1e-8)
        * unit_probability.clamp_min(1e-8).log()
    ).sum(dim=-1)
    usage = unit_probability[..., :-1].mean(dim=(0, 1, 2))
    metrics = {
        "instance_loss": float(output["loss_instance_group"].detach()),
        "ap25": float(ap["ap_25"]),
        "ap50": float(ap["ap_50"]),
        "ap75": float(ap["ap_75"]),
        "pred_gt": len(predictions) / max(1, len(gts)),
        "effective_query_count": float(torch.exp(entropy.mean())),
        "assignment_entropy": float(entropy.mean()),
        "nonempty_prediction_count": int(sum(bool(mask.any()) for mask in predictions)),
        "void_ratio": float(unit_probability[..., -1].mean()),
        "psnr": float(output["psnr"].detach()) if "psnr" in output else None,
        "ssim": float(output["ssim"].detach()) if "ssim" in output else None,
        "lpips": float(output["lpips"].detach()) if "lpips" in output else None,
        "finite": not _nonfinite_output_keys(output),
        "nonfinite_keys": _nonfinite_output_keys(output),
    }
    metrics.update(mask_diag)
    if baseline_output is not None:
        base_assignment = baseline_output[
            "instance_group_probabilities"
        ][0].float().to(unit_probability.device)
        metrics["unit_argmax_assignment_change_ratio"] = float(
            (
                unit_probability.argmax(dim=-1)
                != base_assignment.argmax(dim=-1)
            ).float().mean()
        )
        current_pixels = output[
            "rendered_instance_group_probability"
        ][0].float().argmax(dim=0)
        base_pixels = baseline_output[
            "rendered_instance_group_probability"
        ][0].float().to(current_pixels.device).argmax(dim=0)
        metrics["binary_mask_pixel_change_ratio"] = float(
            (current_pixels != base_pixels).float().mean()
        )
    return metrics


def _gradient_norms(model) -> dict[str, float]:
    prefixes = {
        "tsh": ("tsh_instance_head.",),
        "understanding_decoder": (
            "token_eru_decoder.understanding_decoder_blocks.",
        ),
        "understanding_unit_formation": ("token_eru_unit_formation.",),
        "r2u_adapters": (
            "token_eru_decoder.reconstruction_to_understanding.",
        ),
        "u2r_adapters": (
            "token_eru_decoder.understanding_to_reconstruction.",
        ),
        "reconstruction_decoder": ("enc_dec_backbone.decoder_blocks.",),
        "absolute_head": ("absolute_gs_head.",),
    }
    result = {}
    for group, group_prefixes in prefixes.items():
        total = 0.0
        for name, parameter in model.named_parameters():
            if name.startswith(group_prefixes) and parameter.grad is not None:
                total += float(parameter.grad.detach().float().square().sum().sqrt())
        result[group] = total
    return result


def _parameter_group(name: str) -> str | None:
    if name.startswith("tsh_instance_head."):
        return "tsh"
    if name.startswith("token_eru_decoder.understanding_decoder_blocks."):
        return "understanding_decoder"
    if name.startswith("token_eru_unit_formation."):
        return "understanding_unit_formation"
    if name.startswith(
        (
            "token_eru_decoder.reconstruction_to_understanding.",
            "token_eru_decoder.understanding_to_reconstruction.",
        )
    ):
        return "pair_adapters"
    return None


def _update_norms(model, initial: dict[str, torch.Tensor]) -> dict[str, float]:
    sums = {key: 0.0 for key in (
        "tsh",
        "understanding_decoder",
        "understanding_unit_formation",
        "pair_adapters",
    )}
    for name, parameter in model.named_parameters():
        group = _parameter_group(name)
        if group is not None and name in initial:
            delta = parameter.detach().float().cpu() - initial[name]
            sums[group] += float(delta.square().sum().sqrt())
    return sums


def _load_model(opt, accelerator):
    model = model_registry[opt.model_type](opt)
    load_model_checkpoint(opt, model, accelerator, 0)
    if (
        hasattr(model, "initialize_token_eru_from_reconstruction")
        and not getattr(model, "_token_eru_loaded_from_checkpoint", False)
    ):
        model.initialize_token_eru_from_reconstruction()
    return model.to(accelerator.device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    output_dir = ROOT / args.workspace
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    opt = dataclasses.replace(config_defaults[ERU_CONFIG])
    opt.workspace = str(output_dir)
    opt.resume = str(CHECKPOINT)
    opt.num_workers = 0
    opt.batch_size = 1
    opt.eval_before_training = False
    opt.use_wandb = False
    opt.max_iters_per_epoch = int(args.steps)
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    print(
        f"[eru-audit] rank={accelerator.process_index} device={accelerator.device} "
        f"cuda={torch.cuda.is_available()} count={torch.cuda.device_count()}",
        flush=True,
    )
    print("[eru-audit] constructing production dataloader", flush=True)
    loader, _, _, _ = get_multi_dataloader(opt, accelerator)
    print("[eru-audit] fetching fixed batch", flush=True)
    batch_cpu = next(iter(loader))
    if batch_cpu["input"].shape[1] != 15:
        raise RuntimeError(f"expected 8+7 input window, got {batch_cpu['input'].shape}")
    if batch_cpu["instance_label_output"].shape[1] != 7:
        raise RuntimeError(
            "expected seven target instance maps, got "
            f"{batch_cpu['instance_label_output'].shape}"
        )
    (output_dir / "fixed_batch_manifest.json").write_text(
        json.dumps(_batch_manifest(batch_cpu), indent=2), encoding="utf-8"
    )
    batch = _move(batch_cpu, accelerator.device)
    print("[eru-audit] fixed batch ready", flush=True)

    base_opt = dataclasses.replace(config_defaults[BASE_CONFIG])
    base_opt.resume = str(CHECKPOINT)
    base_opt.num_workers = 0
    base_opt.evaluating = False
    base_opt.use_wandb = False
    print("[eru-audit] constructing/loading baseline model", flush=True)
    base = _load_model(base_opt, accelerator)
    print("[eru-audit] baseline model loaded; starting baseline forward", flush=True)
    base.eval()
    with torch.no_grad():
        base_output = base(batch, compute_quality_metrics=False)
    print("[eru-audit] baseline forward complete", flush=True)
    # Keep only detached CPU snapshots needed by the identity check.  Holding
    # the complete rendered baseline output on CUDA while constructing the ERU
    # model unnecessarily retains a large inference graph/tensor set.
    base_snapshot = {
        "q_abs": base._tsh_last_q_abs.detach().cpu().clone(),
        "gaussians": base._tsh_last_student_gaussians.detach().cpu().clone(),
        "output": {
            key: value.detach().cpu().clone()
            for key, value in base_output.items()
            if key
            in {
                "images_pred",
                "unit_logits",
                "rendered_instance_group_probability",
                "instance_group_probabilities",
                "psnr",
            }
        },
    }
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[eru-audit] constructing/loading ERU model", flush=True)
    model = _load_model(opt, accelerator)
    print("[eru-audit] ERU model loaded; starting step0 forward", flush=True)
    model.eval()
    model.set_token_eru_step(0)
    with torch.no_grad():
        step0_output = model(batch, compute_quality_metrics=False)
    print("[eru-audit] ERU step0 forward complete", flush=True)
    identity = {
        "q_abs_max_diff": float(
            (model._tsh_last_q_abs - base_snapshot["q_abs"].to(model._tsh_last_q_abs.device)).abs().max()
        ),
        "student_gs_max_diff": float(
            (model._tsh_last_student_gaussians - base_snapshot["gaussians"].to(model._tsh_last_student_gaussians.device))
            .abs()
            .max()
        ),
        "rgb_max_diff": float(
            (step0_output["images_pred"] - base_snapshot["output"]["images_pred"].to(step0_output["images_pred"].device))
            .abs()
            .max()
        ),
        "unit_logits_max_diff": float(
            (step0_output["unit_logits"] - base_snapshot["output"]["unit_logits"].to(step0_output["unit_logits"].device))
            .abs()
            .max()
        ),
        "rendered_masks_max_diff": float(
            (
                step0_output["rendered_instance_group_probability"]
                - base_snapshot["output"]["rendered_instance_group_probability"].to(step0_output["rendered_instance_group_probability"].device)
            )
            .abs()
            .max()
        ),
        "psnr_max_diff": float(
            (step0_output["psnr"] - base_snapshot["output"]["psnr"].to(step0_output["psnr"].device)).abs().max()
        ),
        "r_u_hidden_max_diff": float(
            (
                model._token_eru_last_reconstruction_tokens
                - model._token_eru_last_understanding_tokens
            )
            .abs()
            .max()
        ),
        "r_u_units_max_diff": float(
            (
                model.absolute_gs_head.form_units(
                    model._token_eru_last_reconstruction_tokens
                )
                - model._token_eru_understanding_units
            )
            .abs()
            .max()
        ),
        "pass": True,
    }
    identity["pass"] = max(
        value for key, value in identity.items() if key != "pass"
    ) <= 1e-6
    (output_dir / "step0_identity.json").write_text(
        json.dumps(identity, indent=2), encoding="utf-8"
    )
    if not identity["pass"]:
        raise RuntimeError(f"TokenGS-ERU step0 identity failed: {identity}")

    optimizer = setup_optimizer(opt, model, accelerator, 0)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    unwrapped = accelerator.unwrap_model(model)
    # The identity probe above intentionally runs in eval mode.  Restore
    # train mode before constructing the first optimization graph.
    model.train()
    initial_trainable = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in unwrapped.named_parameters()
        if parameter.requires_grad and _parameter_group(name) is not None
    }
    milestones = {0, 1, 2, 5, 25, 26, 50, 100}
    results = {"identity": identity, "milestones": {}}
    # Record a train-mode, no-grad step-0 loss separately from the eval-mode
    # identity output.  The production branch only constructs the Hungarian
    # instance loss when self.training is true.
    with torch.no_grad():
        unwrapped.set_token_eru_step(0)
        train_step0_output = model(batch, compute_quality_metrics=False)
    results["milestones"]["0"] = {
        **_model_metrics(
            step0_output,
            batch,
            baseline_output=base_snapshot["output"],
        ),
        "train_instance_loss": float(
            train_step0_output["loss_instance_group"].detach()
        ),
        "train_total_loss": float(train_step0_output["loss"].detach()),
        "train_loss_finite": bool(
            torch.isfinite(train_step0_output["loss"]).item()
            and torch.isfinite(
                train_step0_output["loss_instance_group"]
            ).item()
        ),
        "gates": unwrapped.set_token_eru_step(0),
    }
    results["milestones"]["0"]["instance_loss"] = results["milestones"]["0"]["train_instance_loss"]
    results["milestones"]["0"].update(
        {
            "audit_instance_loss": results["milestones"]["0"]["train_instance_loss"],
            "audit_total_loss": results["milestones"]["0"]["train_total_loss"],
            "parameter_update_norms": {
                "tsh": 0.0,
                "understanding_decoder": 0.0,
                "understanding_unit_formation": 0.0,
                "pair_adapters": 0.0,
            },
            "r_u_units_max_diff": 0.0,
            "student_gs_max_diff_from_baseline": 0.0,
            "rgb_max_diff_from_baseline": 0.0,
        }
    )
    checkpoint = {
        key: value.detach().cpu().contiguous()
        for key, value in unwrapped.state_dict().items()
    }
    save_file(checkpoint, str(output_dir / "model_step_000000.safetensors"))
    (output_dir / "metadata_step_000000.json").write_text(
        json.dumps({"optimizer_step": 0, "gates": results["milestones"]["0"]["gates"]}, indent=2),
        encoding="utf-8",
    )
    del train_step0_output
    for step in range(int(args.steps)):
        completed = step + 1
        unwrapped.set_token_eru_step(completed)
        unwrapped.tsh_instance_loss_weight_eff = 1.0
        unwrapped.tsh_unit_grad_eff = 1.0
        unwrapped.tsh_mbm_u2r_eff = 0.0
        unwrapped.teacher_lambda_eff = 0.0
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            output = model(batch, compute_quality_metrics=False)
        loss = output["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite ERU loss at step {completed}")
        accelerator.backward(loss)
        grads = _gradient_norms(unwrapped)
        accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)
        optimizer.step()
        if completed in milestones:
            model.train()
            with torch.no_grad():
                audit_output = model(batch, compute_quality_metrics=True)
            model.eval()
            unwrapped.set_token_eru_step(completed)
            with torch.no_grad():
                snapshot = model(batch, compute_quality_metrics=True)
                metric = _model_metrics(
                    snapshot,
                    batch,
                    baseline_output=base_snapshot["output"],
                )
            metric["train_instance_loss"] = float(
                output["loss_instance_group"].detach()
            )
            metric["train_total_loss"] = float(output["loss"].detach())
            metric["audit_instance_loss"] = float(
                audit_output["loss_instance_group"].detach()
            )
            metric["audit_total_loss"] = float(audit_output["loss"].detach())
            metric["train_loss_finite"] = bool(
                torch.isfinite(output["loss"]).item()
                and torch.isfinite(output["loss_instance_group"]).item()
            )
            metric["instance_loss"] = metric["train_instance_loss"]
            metric["parameter_update_norms"] = _update_norms(
                unwrapped, initial_trainable
            )
            metric["r_u_units_max_diff"] = float(
                (
                    unwrapped.absolute_gs_head.form_units(
                        unwrapped._token_eru_last_reconstruction_tokens
                    )
                    - unwrapped._token_eru_understanding_units
                )
                .abs()
                .max()
            )
            metric["student_gs_max_diff_from_baseline"] = float(
                (
                    unwrapped._tsh_last_student_gaussians
                    - base_snapshot["gaussians"].to(
                        unwrapped._tsh_last_student_gaussians.device
                    )
                )
                .abs()
                .max()
            )
            metric["rgb_max_diff_from_baseline"] = float(
                (
                    snapshot["images_pred"]
                    - base_snapshot["output"]["images_pred"].to(
                        snapshot["images_pred"].device
                    )
                )
                .abs()
                .max()
            )
            metric["gates"] = unwrapped.set_token_eru_step(completed)
            metric["gradient_norms"] = grads
            metric["r_u_hidden_max_diff"] = float(
                (
                    unwrapped._token_eru_last_reconstruction_tokens
                    - unwrapped._token_eru_last_understanding_tokens
                )
                .abs()
                .max()
            )
            results["milestones"][str(completed)] = metric
            checkpoint = {
                key: value.detach().cpu().contiguous()
                for key, value in unwrapped.state_dict().items()
            }
            save_file(checkpoint, str(output_dir / f"model_step_{completed:06d}.safetensors"))
            (output_dir / f"metadata_step_{completed:06d}.json").write_text(
                json.dumps({"optimizer_step": completed, "gates": metric["gates"]}, indent=2),
                encoding="utf-8",
            )
            model.train()
    if int(args.steps) >= 100:
        restore_opt = dataclasses.replace(
            opt,
            resume=str(output_dir / "model_step_000100.safetensors"),
        )
        restore_model = _load_model(restore_opt, accelerator)
        saved_state = load_file(
            str(output_dir / "model_step_000100.safetensors"),
            device="cpu",
        )
        restored_state = restore_model.state_dict()
        required_prefixes = (
            "token_eru_decoder.",
            "token_eru_unit_formation.",
            "absolute_gs_head.",
            "tsh_instance_head.",
            "enc_dec_backbone.decoder_blocks.",
        )
        required_saved_keys = [
            key
            for key in saved_state
            if key.startswith(required_prefixes)
        ]
        saved_keys_missing = [
            key for key in required_saved_keys if key not in restored_state
        ]
        restore_diffs = {
            key: float(
                (restored_state[key].detach().cpu() - value.detach().cpu())
                .float()
                .abs()
                .max()
            )
            for key, value in saved_state.items()
            if key.startswith(required_prefixes)
            if key in restored_state
        }
        restore_report = {
            "missing_keys": saved_keys_missing,
            "unexpected_keys": [],
            "max_parameter_diff": max(restore_diffs.values(), default=0.0),
            "parameter_count": len(required_saved_keys),
            "all_saved_keys": len(saved_state),
            "checked_prefixes": list(required_prefixes),
            "pass": (
                not saved_keys_missing
                and max(restore_diffs.values(), default=0.0) == 0.0
            ),
        }
        results["independent_restore"] = restore_report
        (output_dir / "independent_restore.json").write_text(
            json.dumps(restore_report, indent=2), encoding="utf-8"
        )
    (output_dir / "fixed_batch_audit.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
