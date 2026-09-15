"""Matched 100-step fixed-batch audit for QueryMetric-Coupling-v1.

The heavy TokenGS setup is reused from the already audited ERU fixed-batch
runner; this wrapper changes only the treatment configuration and local QMC
step setter.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
from pathlib import Path

import torch

from scripts import audit_token_eru_unit_3d_anchor_fixed_batch as base
from tokengs.options import config_defaults

_ORIGINAL_SCHEDULE = base.set_training_schedule


CONTROL = (
    "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_local250_control_short200_ddp8"
)
QMC = (
    "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_j2_local250_query_metric_v1_short200_ddp8"
)
PARENT = Path(
    "/space/mawb/tokengs/workspace/semantic_v6_absolute_units_true_shared_"
    "token_eru1_dino_metric_joint_formation_j2_ddp8/checkpoints/"
    "model_step_000250.safetensors"
)
PARENT_SHA = "61debbf5f55b76eea16012aea00fe19291309177bfdc6dd0420f6d1cb58d8cba"


def _set_schedule(model, opt, effective_step: int) -> None:
    _ORIGINAL_SCHEDULE(model, opt, effective_step)
    if hasattr(model, "set_token_eru_query_metric_step"):
        model.set_token_eru_query_metric_step(max(0, int(effective_step) - 960))


def _tensor_max_abs(left, right) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def _permutation_sha256(permutation: torch.Tensor) -> str:
    return hashlib.sha256(
        permutation.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def run_causal_ablation(source_dir: Path, output_dir: Path) -> None:
    """Evaluate an existing QMC step-100 checkpoint without optimizer steps."""
    from safetensors.torch import load_file

    checkpoint = source_dir / "qmc" / "checkpoints" / "model_step_000100_trainable.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing QMC step-100 diagnostic checkpoint: {checkpoint}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty causal audit workspace: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base.PARENT = PARENT
    base.PARENT_SHA = PARENT_SHA
    base.set_training_schedule = _set_schedule
    loader_opt = dataclasses.replace(config_defaults[QMC])
    loader_opt.resume = str(PARENT)
    loader_opt.workspace = str(output_dir / "batch_loader")
    loader_opt.num_workers = 0
    loader_opt.tsh_ddp8 = False
    loader_opt.use_wandb = False
    loader_opt.eval_before_training = False
    loader_accelerator = base.Accelerator(
        mixed_precision="no",
        dataloader_config=base.DataLoaderConfiguration(use_seedable_sampler=True),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(int(loader_accelerator.local_process_index))
    loader, _, _, _ = base.get_multi_dataloader(loader_opt, loader_accelerator)
    batch = base.move(next(iter(loader)), loader_accelerator.device)
    if batch["input"].shape[1] != 15:
        raise RuntimeError("causal audit requires 8 context + 7 target")

    opt, accelerator, model, _, _ = base.build_runtime(QMC, output_dir / "runtime")
    state = load_file(str(checkpoint), device="cpu")
    expected = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(state) != expected:
        raise RuntimeError("QMC step-100 trainable checkpoint keys are not exact")
    model.load_state_dict(state, strict=False)
    _set_schedule(model, opt, 1060)
    model.eval()
    coupling = getattr(model, "token_eru_query_metric_coupling", None)
    if coupling is None:
        raise RuntimeError("QMC coupling module is missing")
    original_forward = coupling.forward
    variants = {}
    permutation_hashes = {}
    for mode in ("full", "off", "shuffle_metric"):
        if mode == "full":
            coupling.forward = original_forward
        else:
            def overridden_forward(
                unit_metric_embeddings,
                query_features,
                base_unit_logits,
                *,
                gate,
                _mode=mode,
            ):
                if _mode == "off":
                    return original_forward(
                        unit_metric_embeddings,
                        query_features,
                        base_unit_logits,
                        gate=0.0,
                    )
                generator = torch.Generator(device="cpu")
                generator.manual_seed(42)
                permutation = torch.randperm(
                    1024 * 8, generator=generator, device="cpu"
                )
                permutation_hashes[_mode] = _permutation_sha256(permutation)
                batch_size = unit_metric_embeddings.shape[0]
                shuffled = unit_metric_embeddings.reshape(
                    batch_size, 1024 * 8, -1
                )[:, permutation.to(unit_metric_embeddings.device)].reshape_as(
                    unit_metric_embeddings
                )
                return original_forward(
                    shuffled,
                    query_features,
                    base_unit_logits,
                    gate=gate,
                )

            coupling.forward = overridden_forward
        with torch.no_grad():
            output = model(batch, compute_quality_metrics=False)
        variants[mode] = {
            "metrics": base.evaluate_native(output, batch),
            "unit_logits": output["unit_logits"].detach().cpu(),
            "soft_masks": output["rendered_instance_group_probability"].detach().cpu(),
            "images_pred": output["images_pred"].detach().cpu(),
            "gaussians": output["gaussians"].detach().cpu(),
            "qmc_gate": float(output["query_metric_gate"]),
            "qmc_residual_abs_mean": float(output["query_metric_residual_abs_mean"]),
            "qmc_residual_abs_max": float(output["query_metric_residual_abs_max"]),
        }
    coupling.forward = original_forward

    full = variants["full"]
    differences = {}
    for mode in ("off", "shuffle_metric"):
        other = variants[mode]
        differences[mode] = {
            "unit_logits_max_diff": _tensor_max_abs(full["unit_logits"], other["unit_logits"]),
            "soft_mask_max_diff": _tensor_max_abs(full["soft_masks"], other["soft_masks"]),
            "rgb_max_diff": _tensor_max_abs(full["images_pred"], other["images_pred"]),
            "gaussian_max_diff": _tensor_max_abs(full["gaussians"], other["gaussians"]),
            "metrics": {
                key: float(full["metrics"][key] - other["metrics"][key])
                for key in ("ap25", "ap50", "ap75", "best_iou", "recall50", "pred_gt")
            },
        }
    if any(
        differences[mode][key] != 0.0
        for mode in differences
        for key in ("rgb_max_diff", "gaussian_max_diff")
    ):
        raise RuntimeError("QMC eval ablation changed RGB or Gaussian outputs")
    payload = {
        "source_checkpoint": str(checkpoint),
        "parent_path": str(PARENT),
        "parent_sha256": PARENT_SHA,
        "batch_fingerprint": {
            "batch_hash": base.batch_hash(batch),
            "scene_name": str(batch.get("scene_name", "unknown")),
            "frame_ids": batch["frame_ids"].detach().cpu().tolist(),
        },
        "variants": {
            mode: {
                key: value
                for key, value in item.items()
                if key not in ("unit_logits", "soft_masks", "images_pred", "gaussians")
            }
            for mode, item in variants.items()
        },
        "differences_full_vs_variant": differences,
        "permutation_hashes": permutation_hashes,
        "optimizer_step_executed": False,
        "formal_evaluation_started": False,
    }
    (output_dir / "causal_ablation.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "status").mkdir(exist_ok=True)
    (output_dir / "status" / "COMPLETE").write_text("ok\n", encoding="utf-8")


def run_gradient_audit(output_dir: Path) -> None:
    """Run independent instance-only and RGB-only backward probes."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty gradient audit workspace: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base.PARENT = PARENT
    base.PARENT_SHA = PARENT_SHA
    base.set_training_schedule = _set_schedule
    loader_opt = dataclasses.replace(config_defaults[QMC])
    loader_opt.resume = str(PARENT)
    loader_opt.workspace = str(output_dir / "batch_loader")
    loader_opt.num_workers = 0
    loader_opt.tsh_ddp8 = False
    loader_opt.use_wandb = False
    loader_opt.eval_before_training = False
    loader_accelerator = base.Accelerator(
        mixed_precision="no",
        dataloader_config=base.DataLoaderConfiguration(use_seedable_sampler=True),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(int(loader_accelerator.local_process_index))
    loader, _, _, _ = base.get_multi_dataloader(loader_opt, loader_accelerator)
    batch = base.move(next(iter(loader)), loader_accelerator.device)
    opt, _, model, _, _ = base.build_runtime(QMC, output_dir / "runtime")
    model.train()
    _set_schedule(model, opt, 961)

    def norm(parameter) -> float:
        if parameter is None or parameter.grad is None:
            return 0.0
        return float(parameter.grad.detach().float().norm())

    def collect(prefixes):
        values = []
        for name, parameter in model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                values.append(norm(parameter))
        return float(sum(value * value for value in values) ** 0.5)

    model.zero_grad(set_to_none=True)
    instance_output = model(batch, compute_quality_metrics=False)
    instance_loss = instance_output["loss_instance_group"] * float(
        getattr(opt, "tsh_lambda_instance", 0.05)
    ) * float(getattr(model, "tsh_instance_loss_weight_eff", 1.0))
    instance_loss.backward()
    instance_probe = {
        "loss": float(instance_loss.detach()),
        "qmc_query_projection_grad_norm": norm(
            model.token_eru_query_metric_coupling.query_projection.weight
        ),
        "qmc_log_temperature_grad_norm": norm(
            model.token_eru_query_metric_coupling.log_temperature
        ),
        "metric_head_grad_norm": collect(("token_eru_metric_head.",)),
        "understanding_stream_grad_norm": collect(
            ("token_eru_decoder.understanding_decoder_blocks.", "token_eru_unit_formation.")
        ),
        "dino_grad_norm": collect(("token_eru_dino_encoder._dino_model.",)),
    }
    model.zero_grad(set_to_none=True)
    rgb_output = model(batch, compute_quality_metrics=False)
    rgb_loss = rgb_output["loss_rgb"]
    rgb_loss.backward()
    rgb_probe = {
        "loss": float(rgb_loss.detach()),
        "qmc_query_projection_grad_norm": norm(
            model.token_eru_query_metric_coupling.query_projection.weight
        ),
        "qmc_log_temperature_grad_norm": norm(
            model.token_eru_query_metric_coupling.log_temperature
        ),
        "metric_head_grad_norm": collect(("token_eru_metric_head.",)),
        "understanding_stream_grad_norm": collect(
            ("token_eru_decoder.understanding_decoder_blocks.", "token_eru_unit_formation.")
        ),
        "dino_grad_norm": collect(("token_eru_dino_encoder._dino_model.",)),
    }
    if not (
        instance_probe["qmc_query_projection_grad_norm"] > 0.0
        and instance_probe["qmc_log_temperature_grad_norm"] > 0.0
        and instance_probe["metric_head_grad_norm"] > 0.0
        and instance_probe["understanding_stream_grad_norm"] > 0.0
        and instance_probe["dino_grad_norm"] == 0.0
        and rgb_probe["qmc_query_projection_grad_norm"] == 0.0
        and rgb_probe["qmc_log_temperature_grad_norm"] == 0.0
        and rgb_probe["metric_head_grad_norm"] == 0.0
    ):
        raise RuntimeError(f"QMC gradient boundary failed: {instance_probe}; {rgb_probe}")
    payload = {
        "batch_hash": base.batch_hash(batch),
        "instance_only": instance_probe,
        "rgb_only": rgb_probe,
        "optimizer_step_executed": False,
    }
    (output_dir / "gradient_audit.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "status").mkdir(exist_ok=True)
    (output_dir / "status" / "COMPLETE").write_text("ok\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--causal-only", action="store_true")
    parser.add_argument("--gradient-only", action="store_true")
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    if args.gradient_only:
        run_gradient_audit(args.output_dir.resolve())
        return
    if args.causal_only:
        if args.source_dir is None:
            raise ValueError("--causal-only requires --source-dir")
        run_causal_ablation(args.source_dir.resolve(), args.output_dir.resolve())
        return
    if args.steps != 100:
        raise ValueError("QMC fixed-batch audit is fixed at 100 optimizer steps")
    if not PARENT.is_file() or base.sha256_file(PARENT) != PARENT_SHA:
        raise RuntimeError("QMC parent checkpoint is missing or SHA mismatched")
    root = args.output_dir.resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing non-empty QMC audit workspace: {root}")
    root.mkdir(parents=True, exist_ok=True)

    # Patch only the helper module's runtime constants; no model or training
    # implementation is changed by this audit wrapper.
    base.PARENT = PARENT
    base.PARENT_SHA = PARENT_SHA
    base.set_training_schedule = _set_schedule

    loader_opt = dataclasses.replace(config_defaults[CONTROL])
    loader_opt.resume = str(PARENT)
    loader_opt.workspace = str(root / "batch_loader")
    loader_opt.num_workers = 0
    loader_opt.tsh_ddp8 = False
    loader_opt.use_wandb = False
    loader_opt.eval_before_training = False
    accelerator = base.Accelerator(
        mixed_precision="no",
        dataloader_config=base.DataLoaderConfiguration(use_seedable_sampler=True),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(int(accelerator.local_process_index))
    loader, _, _, _ = base.get_multi_dataloader(loader_opt, accelerator)
    batch = base.move(next(iter(loader)), accelerator.device)
    if batch["input"].shape[1] != 15:
        raise RuntimeError("fixed batch is not 8 context + 7 target")
    fingerprint = {
        "batch_hash": base.batch_hash(batch),
        "scene_name": str(batch.get("scene_name", "unknown")),
        "frame_ids": (
            batch["frame_ids"].detach().cpu().tolist()
            if torch.is_tensor(batch.get("frame_ids"))
            else None
        ),
    }
    (root / "batch_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2), encoding="utf-8"
    )

    reports = {}
    identities = {}
    final_qmc = None
    for label, config_name in (("control", CONTROL), ("qmc", QMC)):
        item, model, optimizer, saved, identity, final_eval = base.run_one(
            config_name, root / label, batch, args.steps
        )
        reports[label] = item
        identities[label] = identity
        if label == "qmc":
            final_qmc = final_eval
        del model, optimizer, saved
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    identity_diffs = {
        key: float(
            (identities["control"][key].float() - identities["qmc"][key].float())
            .abs()
            .max()
        )
        for key in sorted(set(identities["control"]) & set(identities["qmc"]))
    }
    qmc_report = reports["qmc"].get(100, {})
    payload = {
        "parent_path": str(PARENT),
        "parent_sha256": PARENT_SHA,
        "control_config": CONTROL,
        "qmc_config": QMC,
        "batch_fingerprint": fingerprint,
        "milestones": [0, 1, 2, 5, 25, 50, 100],
        "reports": reports,
        "step0_control_qmc_identity_diffs": identity_diffs,
        "qmc_step100": qmc_report,
        "formal_native_query_only": True,
        "p_u_used": False,
        "metric_cluster_used": False,
        "training_steps": 100,
    }
    (root / "fixed_batch_learning_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (root / "status").mkdir(exist_ok=True)
    (root / "status" / "COMPLETE").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
