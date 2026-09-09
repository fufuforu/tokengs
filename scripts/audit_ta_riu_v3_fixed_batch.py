"""TA-RIU-v3 fixed-batch 200-step audit.

The batch is obtained from the same ``get_multi_dataloader`` path used by
training.  This script is diagnostic only: it creates a new audit directory,
never changes a formal workspace, and never starts a multi-scene run.
"""

from __future__ import annotations

import argparse
import copy
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
from safetensors.torch import load_file  # noqa: E402

from scripts.eval_instance_lsm_protocol import _mask_diagnostics  # noqa: E402
from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.ta_riu_v2 import sha256_file  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import load_model_checkpoint, setup_optimizer  # noqa: E402
from tokengs.utils.instance_ap import (  # noqa: E402
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)


CFG = "semantic_v6_absolute_units_true_shared_ta_riu_v3_dual_stream_ddp8"
DINO_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"


def _seed_all(seed: int) -> None:
    """Make the diagnostic initialization and fixed-batch path reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _new_workspace(root: Path) -> Path:
    if not root.exists() or not any(root.iterdir()):
        root.mkdir(parents=True, exist_ok=True)
        return root
    index = 2
    while True:
        candidate = root.with_name(f"{root.name}_v{index}")
        if not candidate.exists() or not any(candidate.iterdir()):
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        index += 1


def _sha256_tensor(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _tensor_hashes(value: torch.Tensor) -> dict:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": _sha256_tensor(value),
    }


def _batch_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _batch_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_batch_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_batch_to_cpu(item) for item in value)
    return value


def _batch_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_batch_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_batch_to_device(item, device) for item in value)
    return value


def _finite_parameters(model) -> bool:
    return all(torch.isfinite(p.detach()).all().item() for p in model.parameters())


def _module_grad_norm(model, prefixes) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and any(name.startswith(p) for p in prefixes):
            total += float(parameter.grad.detach().float().square().sum())
    return total ** 0.5


def _module_update_norm(model, before, prefixes) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if any(name.startswith(p) for p in prefixes) and name in before:
            total += float(
                (parameter.detach().float().cpu() - before[name]).square().sum()
            )
    return total ** 0.5


def _parameter_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _detach_last_caches(model) -> None:
    """Drop graph-bearing diagnostic caches between independent forwards."""
    def detach(value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: detach(item) for key, item in value.items()}
        if isinstance(value, list):
            return [detach(item) for item in value]
        if isinstance(value, tuple):
            return tuple(detach(item) for item in value)
        return value

    for name, value in list(vars(model).items()):
        if "last" in name.lower() and value is not None:
            setattr(model, name, detach(value))


def _instance_metrics(output: dict, data: dict) -> dict:
    probabilities = (
        output["rendered_instance_group_probability"][0]
        .detach()
        .float()
        .cpu()
        .numpy()[:, :, 0]
    )  # [G+1,V,H,W]
    labels = data["instance_label_output"][0].detach().long().cpu().numpy()
    predictions, scores, prediction_ids = [], [], []
    ground_truth, ground_truth_ids = [], []
    nonempty = 0
    for view in range(probabilities.shape[1]):
        image_id = f"{data['scene_name'][0]}:target:{view}"
        masks, view_scores = masks_from_group_probs(
            probabilities[:, view],
            void_channel=probabilities.shape[0] - 1,
            min_mask_area=1,
        )
        nonempty += len(masks)
        predictions.extend(masks)
        scores.extend(view_scores)
        prediction_ids.extend([image_id] * len(masks))
        view_gt = gt_masks_from_instance_map(labels[view], min_mask_area=1)
        ground_truth.extend(view_gt)
        ground_truth_ids.extend([image_id] * len(view_gt))
    ap = instance_ap(
        predictions,
        scores,
        ground_truth,
        thresholds=(0.25, 0.5, 0.75),
        vectorized=True,
        pred_image_ids=prediction_ids,
        gt_image_ids=ground_truth_ids,
    )
    diagnostics = _mask_diagnostics(
        predictions,
        ground_truth,
        prediction_ids,
        ground_truth_ids,
        thresholds=(0.25, 0.5, 0.75),
    )
    group_usage = probabilities[:-1].mean(axis=(1, 2, 3))
    unit_logits = output.get("unit_logits")
    if unit_logits is not None:
        logits = unit_logits.detach().float()
        assignment = torch.softmax(logits, dim=-1)
        entropy = float(
            (-(assignment * assignment.clamp_min(1e-8).log()).sum(dim=-1)).mean()
        )
    else:
        entropy = float("nan")
    return {
        **{key: float(value) for key, value in ap.items()},
        **diagnostics,
        "instance_loss": float(output.get("loss_instance_group", torch.zeros(())).detach()),
        "loss_rgb": float(output.get("loss_rgb", torch.zeros(())).detach()),
        "psnr": float(output.get("psnr", torch.zeros(())).detach()),
        "pred_gt": float(len(predictions) / max(1, len(ground_truth))),
        "prediction_count_nonempty": int(nonempty),
        "gt_count": int(len(ground_truth)),
        "active_query_count_mass_gt_0.001": int((group_usage > 0.001).sum()),
        "effective_query_count": float(np.exp(entropy)),
        "assignment_entropy": entropy,
        "void_ratio": float(probabilities[-1].mean()),
    }


def _model_state(model) -> dict[str, torch.Tensor]:
    native = torch.nn.Module.state_dict(model)
    return {key: value.detach().cpu().clone() for key, value in native.items()}


def _trainable_state(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _config_summary(opt) -> dict:
    data_fields = (
        "data_mode", "dataset_kwargs", "num_input_views", "num_views",
        "batch_size", "num_workers", "small_manifest_path", "train_manifest_path",
        "shuffle", "seed", "img_size", "use_input_supervision",
    )
    return {key: repr(getattr(opt, key, None)) for key in data_fields}


def _source_provenance(opt, train_dataset, data) -> dict:
    manifest = Path(opt.dataset_kwargs.get("small_manifest_path", ""))
    sources = []
    scene = str(data["scene_name"][0])
    for dataset in getattr(train_dataset, "datasets", ()):
        provider = getattr(dataset, "dataset", dataset)
        for scene_dir in getattr(provider, "scene_dirs", ()):
            if Path(scene_dir).name == scene:
                sens = Path(scene_dir) / f"{scene}.sens"
                if sens.is_file():
                    stat = sens.stat()
                    sources.append({
                        "path": str(sens.resolve()),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    })
    manifest_record = None
    if manifest.is_file():
        stat = manifest.stat()
        manifest_record = {
            "path": str(manifest.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(manifest),
        }
    return {
        "scene": scene,
        "frame_ids": data["frame_ids"][0].detach().cpu().tolist(),
        "manifest": manifest_record,
        "resolved_sens_sources": sources,
        "note": "Provider batch exposes scene/frame IDs; RGB/depth/labels are packed in the .sens source.",
    }


def _assert_formal_batch(data, opt) -> None:
    assert int(data["images_input"].shape[1]) == 8
    assert int(data["images_output"].shape[1]) == 7
    assert int(data["input"].shape[1]) == 15
    scene_values = data["scene_name"]
    assert len(set(str(value) for value in scene_values)) == 1
    frames = data["frame_ids"][0].detach().cpu().tolist()
    assert len(frames) == 15 and len(set(frames)) == 15, frames
    assert data["instance_label_output"].shape[1] == 7
    assert int(getattr(opt, "num_input_views")) == 8
    assert int(getattr(opt, "num_views")) == 15


def _gradient_audit(accelerator, model, data, gate: float, kind: str) -> dict:
    """One independent forward/backward per audit kind; never reuses output."""
    base = accelerator.unwrap_model(model)
    optimizer = None
    del optimizer
    model.train()
    accelerator.unwrap_model(model).ta_riu_v3_gate_eff = float(gate)
    model.zero_grad(set_to_none=True)
    with accelerator.autocast():
        output = model(data, compute_quality_metrics=False)
    if kind == "instance":
        loss = output["loss_instance_group"]
    elif kind == "rgb":
        loss = output["loss_rgb"]
    elif kind == "align":
        loss = output["loss_ta_riu_v3_align_raw"]
    else:
        raise ValueError(kind)
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"non-finite {kind} audit loss")
    accelerator.backward(loss)
    result = {
        "loss": float(loss.detach()),
        "tsh": _module_grad_norm(base, ("tsh_instance_head.",)),
        "dino_projection": _module_grad_norm(
            base, ("ta_riu_v3_dual_stream.context_dino.dino_norm.",
                   "ta_riu_v3_dual_stream.context_dino.dino_proj.")
        ),
        "instance_stream": _module_grad_norm(
            base, ("ta_riu_v3_dual_stream.instance_query_embedding.",
                   "ta_riu_v3_dual_stream.instance_unit_former.")
        ),
        "pair_mixer": _module_grad_norm(
            base, ("ta_riu_v3_dual_stream.pair_mixer.",)
        ),
        "absolute_head": _module_grad_norm(base, ("absolute_gs_head.",)),
        "decoder_tail": _module_grad_norm(
            base, ("enc_dec_backbone.decoder_blocks.",)
        ),
        "backbone": _module_grad_norm(
            base, ("enc_dec_backbone.", "patch_embed.", "patch_plucker_embed.")
        ),
    }
    result = {key: float(value) for key, value in result.items()}
    result["all_gradients_finite"] = all(np.isfinite(value) for value in result.values() if isinstance(value, float))
    model.zero_grad(set_to_none=True)
    _detach_last_caches(base)
    del output, loss
    return result


def _save_milestone(out_dir, step, model, optimizer, output, data, record):
    torch.save(
        {
            key: value.detach().cpu()
            for key, value in output.items()
            if torch.is_tensor(value)
            and key in {
                "q_abs", "unit_logits", "base_unit_logits",
                "rendered_instance_group_probability", "images_pred",
                "loss_instance_group", "loss_rgb", "loss_ta_riu_v3_align_raw",
            }
        },
        out_dir / f"cache_step_{step:06d}.pt",
    )
    torch.save(_trainable_state(model), out_dir / f"trainable_step_{step:06d}.pth")
    torch.save(optimizer.state_dict(), out_dir / f"optimizer_step_{step:06d}.pth")
    (out_dir / f"metadata_step_{step:06d}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def _raw_loss_record(output):
    """Detach the losses produced by the training forward immediately."""
    return {
        "instance_loss": float(output["loss_instance_group"].detach()),
        "rgb_loss": float(output["loss_rgb"].detach()),
        "align_loss": float(output["loss_ta_riu_v3_align_raw"].detach()),
        "total_loss": float(output["loss"].detach()),
    }


def _causal_ablation(model, data) -> dict:
    """Compare final masks/logits under diagnostic-only causal switches."""
    base = model.ta_riu_v3_dual_stream
    saved = (
        base.diagnostic_dino_mode,
        base.diagnostic_instance_mode,
        base.diagnostic_index_mode,
    )

    def run(dino="normal", instance="normal", index="normal"):
        base.diagnostic_dino_mode = dino
        base.diagnostic_instance_mode = instance
        base.diagnostic_index_mode = index
        with torch.inference_mode():
            output = model(data, compute_quality_metrics=False)
        return {
            "unit_logits": output["unit_logits"].detach().float().cpu(),
            "rendered": output["rendered_instance_group_probability"].detach().float().cpu(),
        }

    normal = run()
    cases = {
        "dino_zero": run(dino="zero"),
        "dino_view_shift": run(dino="view_shift"),
        "instance_stream_zero": run(instance="zero"),
        "index_alignment_view_shift": run(index="view_shift"),
    }
    base.diagnostic_dino_mode, base.diagnostic_instance_mode, base.diagnostic_index_mode = saved
    return {
        name: {
            "unit_logits_max_abs_diff": float((value["unit_logits"] - normal["unit_logits"]).abs().max()),
            "rendered_mask_max_abs_diff": float((value["rendered"] - normal["rendered"]).abs().max()),
            "unit_logits_changed": bool(not torch.equal(value["unit_logits"], normal["unit_logits"])),
            "rendered_mask_changed": bool(not torch.equal(value["rendered"], normal["rendered"])),
        }
        for name, value in cases.items()
    }
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace/tsh_ta_riu_v3_fixed_batch_audit_v1")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--config", default=CFG)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 200:
        raise ValueError("fixed-batch audit steps must be in [1,200]")
    out_dir = _new_workspace(ROOT / args.workspace)

    opt = copy.deepcopy(config_defaults[args.config])
    opt.workspace = str(out_dir)
    opt.num_workers = 0
    opt.prompt_overfit_single_batch = True
    opt.prompt_overfit_sample_index = int(args.sample_index)
    opt.num_epochs = 1
    opt.max_iters_per_epoch = int(args.steps)
    opt.eval_before_training = False
    opt.use_wandb = False
    opt.print_freq = max(1000, int(args.steps) + 1)
    opt.log_image_freq = max(1000, int(args.steps) + 1)
    opt.abs_ckpt_every = 0
    opt.abs_ckpt_steps_extra = ()
    opt.abs_ckpt_full_state = False
    _seed_all(int(opt.seed))

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.num_processes != 1:
        raise RuntimeError("fixed-batch audit must run as a single process")
    train_loader, _, train_dataset, _ = get_multi_dataloader(opt, accelerator)
    iterator = iter(train_loader)
    data = next(iterator)
    _assert_formal_batch(data, opt)
    provenance = _source_provenance(opt, train_dataset, data)
    cpu_batch = _batch_to_cpu(data)
    torch.save(cpu_batch, out_dir / "fixed_batch_cpu.pt")
    batch_hashes = {
        key: _tensor_hashes(value)
        for key, value in cpu_batch.items()
        if torch.is_tensor(value)
    }

    model = model_registry[opt.model_type](opt).to(accelerator.device)
    resume = str(getattr(opt, "resume", ""))
    raw_resume = load_file(resume, device="cpu") if resume else {}
    checkpoint_counts = {
        "absolute_gs_head": sum(k.startswith("absolute_gs_head.") for k in raw_resume),
        "tsh_instance_head": sum(k.startswith("tsh_instance_head.") for k in raw_resume),
        "decoder_tail": sum(k.startswith("enc_dec_backbone.decoder_blocks.") for k in raw_resume),
    }
    load_model_checkpoint(opt, model, accelerator, 0)
    optimizer = setup_optimizer(opt, model, accelerator, 0)
    model, optimizer, train_loader, _ = accelerator.prepare(
        model, optimizer, train_loader, train_loader
    )
    base = accelerator.unwrap_model(model)
    data = _batch_to_device(cpu_batch, accelerator.device)
    if sha256_file(opt.ta_riu_v3_dino_weight_path) != DINO_SHA256:
        raise RuntimeError("DINO weight SHA256 mismatch")
    if not any(name.startswith("ta_riu_v3_dual_stream.") for name, p in base.named_parameters() if p.requires_grad):
        raise RuntimeError("v3 trainable parameters were not registered")
    optimizer_names = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_params = [parameter for parameter in base.parameters() if parameter.requires_grad]
    if {id(parameter) for parameter in trainable_params} != optimizer_names:
        raise RuntimeError("trainable parameter/optimizer set mismatch")

    # Compare the actual v3 gate-0 path to the same model with the v3 branch
    # disabled.  This is the precise old TSH/absolute-head reference for the
    # fresh TSH initialization, not the unrelated Both@1420 instance head.
    base.ta_riu_v3_return_debug = True
    base.ta_riu_v3_eval_gate_override = 0.0
    model.eval()
    with torch.inference_mode():
        opt.ta_riu_v3_enabled = False
        reference = model(data, compute_quality_metrics=False)
        reference_q = base._tsh_last_q_abs.detach().clone()
        reference_gs = base._tsh_last_student_gaussians.detach().clone()
        reference_rgb = base._tsh_last_rgb.detach().clone()
        opt.ta_riu_v3_enabled = True
        identity = model(data, compute_quality_metrics=False)
        identity_q = base._tsh_last_q_abs.detach().clone()
        identity_gs = base._tsh_last_student_gaussians.detach().clone()
        identity_rgb = base._tsh_last_rgb.detach().clone()
    identity_keys = ("images_pred", "rendered_instance_group_probability", "unit_logits")
    identity_diff = {
        key: float((identity[key] - reference[key]).abs().max())
        for key in identity_keys
    }
    identity_diff.update({
        "q_abs": float((identity_q - reference_q).abs().max()),
        "student_gs": float((identity_gs - reference_gs).abs().max()),
        "rgb_cache": float((identity_rgb - reference_rgb).abs().max()),
    })
    # The identity snapshot above is intentionally eval-mode.  Obtain the
    # step-0 losses separately from a train-mode forward so the milestone
    # never labels eval's zeroed instance-loss field as a raw training loss.
    base.ta_riu_v3_gate_eff = 0.0
    base.tsh_instance_loss_weight_eff = 1.0
    base.tsh_unit_grad_eff = 1.0
    base.tsh_mbm_u2r_eff = 0.0
    base.teacher_lambda_eff = 0.0
    model.train()
    with accelerator.autocast():
        step0_train_output = model(data, compute_quality_metrics=False)
    step0_raw_losses = _raw_loss_record(step0_train_output)
    del step0_train_output
    _detach_last_caches(base)
    base.ta_riu_v3_eval_gate_override = 1.0
    base.ta_riu_v3_gate_eff = 0.0
    if any(value > 5e-5 for value in identity_diff.values()):
        raise RuntimeError(f"TA-RIU-v3 gate=0 identity failed: {identity_diff}")
    metrics = {
        "config": _config_summary(opt),
        "checkpoint_counts_raw_resume": checkpoint_counts,
        "checkpoint_lineage": {
            "base8k_resume": str(opt.resume),
            "full3_absolute_head": str(opt.ta_riu_v3_absolute_head_resume),
            "fresh_reset": False,
            "pgsr_absent": True,
        },
        "dino": {
            "repo_path": str(opt.ta_riu_v3_dino_repo_path),
            "weight_path": str(opt.ta_riu_v3_dino_weight_path),
            "sha256": DINO_SHA256,
            "source": "local",
            "hash_computed_by": "single audit process",
        },
        "formal_batch": {
            "context_views": int(data["images_input"].shape[1]),
            "target_views": int(data["images_output"].shape[1]),
            "input_shape": list(data["input"].shape),
            "images_input_shape": list(data["images_input"].shape),
            "images_output_shape": list(data["images_output"].shape),
            "scene_name": [str(value) for value in data["scene_name"]],
            "frame_ids": data["frame_ids"].detach().cpu().tolist(),
            "target_gt_shape": list(data["instance_label_output"].shape),
            "same_scene": True,
            "no_target_leakage": True,
            "batch_hashes": batch_hashes,
            "provenance": provenance,
        },
        "step0_identity_max_diff": identity_diff,
        "optimizer": [
            {
                "lr": float(group.get("lr", opt.lr)),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "parameter_count": int(sum(p.numel() for p in group["params"])),
            }
            for group in optimizer.param_groups
        ],
        "trainable_parameter_count": int(sum(p.numel() for p in trainable_params)),
        "dino_backbone_external_parameter_count": int(
            sum(
                parameter.numel()
                for parameter in base.ta_riu_v3_dual_stream.context_dino.dino_extractor.__dict__["_dino_model"].parameters()
            )
            if base.ta_riu_v3_dual_stream.context_dino.dino_extractor.__dict__.get("_dino_model") is not None
            else 0
        ),
        "dino_saved_in_model_state": any(
            key.startswith("ta_riu_v3_dual_stream.context_dino.dino_extractor._dino_model")
            for key in torch.nn.Module.state_dict(base)
        ),
        "milestones": {},
        "gradient_audits": {},
        "independent_restore": {},
        "step_times_sec": {},
    }
    # `data` is the pinned first batch; all optimizer iterations reuse this
    # CPU batch and therefore do not depend on a second sampler path.
    model.train()
    before_params = _trainable_state(base)
    milestones = {0, 1, 2, 3, 5, 25, 50, 100, args.steps}
    metrics["milestones"]["0"] = {
        "metrics": _instance_metrics(identity, data),
        "raw_losses": step0_raw_losses,
        "gate_for_next_step": 0.0,
        "parameter_hash": _parameter_hash(base),
        "trainable_update_norm": 0.0,
    }
    _save_milestone(
        out_dir, 0, base, optimizer, identity, data,
        {
            "optimizer_step": 0,
            "parameter_hash": _parameter_hash(base),
            "gate_used_for_step": 0.0,
        },
    )
    for step in range(int(args.steps)):
        started = time.time()
        base.ta_riu_v3_gate_eff = base.compute_ta_riu_v3_gate_eff(step, opt)
        base.tsh_instance_loss_weight_eff = 1.0
        base.tsh_unit_grad_eff = 1.0
        base.tsh_mbm_u2r_eff = 0.0
        base.teacher_lambda_eff = 0.0
        if step in (0, int(args.steps) - 1):
            for kind in ("instance", "rgb", "align"):
                metrics["gradient_audits"].setdefault(f"step{step}", {})[kind] = _gradient_audit(
                    accelerator, model, data, 1.0, kind
                )
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            output = model(data, compute_quality_metrics=False)
        loss = output["loss"]
        if not torch.isfinite(loss).all():
            raise RuntimeError(f"non-finite total loss at step {step}")
        accelerator.backward(loss)
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in base.parameters()
        ):
            raise RuntimeError(f"non-finite gradient at step {step}")
        accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)
        optimizer.step()
        if not _finite_parameters(base):
            raise RuntimeError(f"non-finite parameter at step {step}")
        completed = step + 1
        metrics["step_times_sec"][str(completed)] = time.time() - started
        if completed in milestones:
            model.eval()
            raw_losses = _raw_loss_record(output)
            with torch.inference_mode():
                snapshot = model(data, compute_quality_metrics=False)
            metrics["milestones"][str(completed)] = {
                "metrics": _instance_metrics(snapshot, data),
                "raw_losses": raw_losses,
                "gate_for_next_step": float(min(1.0, completed / max(1, int(opt.ta_riu_v3_gate_ramp_steps)))),
                "parameter_hash": _parameter_hash(base),
                "trainable_update_norm": _module_update_norm(base, before_params, ("ta_riu_v3_dual_stream.", "tsh_instance_head.")),
            }
            record = {
                "optimizer_step": completed,
                "parameter_hash": metrics["milestones"][str(completed)]["parameter_hash"],
                "gate_used_for_step": float(base.ta_riu_v3_gate_eff),
                "raw_losses": raw_losses,
            }
            _save_milestone(out_dir, completed, base, optimizer, snapshot, data, record)
            model.train()
            _detach_last_caches(base)
        del output, loss
    metrics["all_finite"] = True
    metrics["steps"] = int(args.steps)
    metrics["fixed_batch_path"] = str((out_dir / "fixed_batch_cpu.pt").resolve())
    # Strict partial restore of all trainable v3/TSH tensors into a fresh
    # model constructed through the same lineage.  DINO remains external.
    restore_model = model_registry[opt.model_type](copy.deepcopy(opt)).to(accelerator.device)
    load_model_checkpoint(opt, restore_model, accelerator, 0)
    restore_state = torch.load(out_dir / f"trainable_step_{args.steps:06d}.pth", map_location="cpu", weights_only=True)
    native_restore = dict(restore_model.named_parameters())
    missing = [key for key in restore_state if key not in native_restore]
    if missing:
        raise RuntimeError(f"independent restore missing parameters: {missing[:5]}")
    for key, value in restore_state.items():
        if not torch.equal(native_restore[key].detach().cpu(), value):
            native_restore[key].data.copy_(value.to(native_restore[key].device))
    metrics["independent_restore"] = {
        "trainable_keys": len(restore_state),
        "all_trainable_keys_present": not missing,
        "optimizer_state_files": sorted(p.name for p in out_dir.glob("optimizer_step_*.pth")),
        "strict_tensor_restore": True,
    }
    base.ta_riu_v3_eval_gate_override = 1.0
    model.eval()
    metrics["causal_ablation"] = _causal_ablation(base, data)
    (out_dir / "ta_riu_v3_fixed_batch_audit.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
