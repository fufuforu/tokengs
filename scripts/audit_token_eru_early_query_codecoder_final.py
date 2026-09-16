"""Final read-only EQC readiness audit.

This audit uses one real training batch only.  It never calls an optimizer or
the formal evaluator.  The step-100 fixed-batch artifact is a trainable-only
overlay, so restoration is explicitly parent-plus-overlay.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import (  # noqa: E402
    _load_cached_gsplat_extension,
    configure_joint_formation_trainability,
    load_model_checkpoint,
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
PARENT_SHA = "61debbf5f55b76eea16012aea00fe19291309177bfdc6dd0420f6d1cb58d8cba"
E1_OVERLAY = ROOT / (
    "workspace/token_eru_early_query_codecoder_v1_fixed_batch_v5/"
    "treatment/step_100_trainable.safetensors"
)
J2 = "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_ddp8"
E0 = "semantic_v6_j2_local250_eqc_v1_control_short200_ddp8"
E1 = "semantic_v6_j2_local250_eqc_v1_treatment_short200_ddp8"
MODES = ("full", "no_final_query_override", "no_u_write", "no_query_update", "off", "shuffle_query")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(x: torch.Tensor) -> str:
    x = x.detach().cpu().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def finite(x) -> bool:
    return torch.is_tensor(x) and bool(torch.isfinite(x).all())


def maxdiff(a, b) -> float | None:
    if a is None or b is None:
        return None
    if tuple(a.shape) != tuple(b.shape):
        return float("inf")
    return float((a.float() - b.float()).abs().max().item())


def move(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move(v, device) for v in x)
    return x


def batch_hash(batch) -> str:
    h = hashlib.sha256()
    for k, v in sorted(batch.items()):
        if torch.is_tensor(v):
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def model_hash(model) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prediction_metrics(prob, labels):
    predictions, scores, image_ids, ground_truth, gt_ids = [], [], [], [], []
    for view in range(prob.shape[1]):
        image_id = f"fixed:target:{view}"
        masks, view_scores = masks_from_group_probs(
            prob[:, view], void_channel=prob.shape[0] - 1, min_mask_area=1
        )
        predictions.extend(masks)
        scores.extend(view_scores)
        image_ids.extend([image_id] * len(masks))
        gt = gt_masks_from_instance_map(labels[view], min_mask_area=1)
        ground_truth.extend(gt)
        gt_ids.extend([image_id] * len(gt))
    ap = instance_ap(
        predictions, scores, ground_truth,
        thresholds=(0.25, 0.5, 0.75), vectorized=True,
        pred_image_ids=image_ids, gt_image_ids=gt_ids,
    )
    best = []
    for gt, gid in zip(ground_truth, gt_ids):
        best.append(max(
            [
                float((pred * gt).sum()) /
                float(pred.sum() + gt.sum() - (pred * gt).sum())
                if float(pred.sum() + gt.sum() - (pred * gt).sum()) else 0.0
                for pred, pid in zip(predictions, image_ids) if pid == gid
            ] or [0.0]
        ))
    return {
        "ap25": float(ap["ap_25"]), "ap50": float(ap["ap_50"]), "ap75": float(ap["ap_75"]),
        "best_iou": float(sum(best) / max(1, len(best))),
        "recall50": float(sum(x >= 0.5 for x in best) / max(1, len(best))),
        "pred_gt": float(len(predictions) / max(1, len(ground_truth))),
        "prediction_count": len(predictions), "gt_count": len(ground_truth),
    }


def build(config_name: str, accelerator, output: Path, overlay: Path | None = None):
    opt = dataclasses.replace(config_defaults[config_name])
    opt.resume = str(PARENT)
    opt.workspace = str(output)
    opt.num_workers = 0
    opt.eval_before_training = False
    opt.use_wandb = False
    torch.manual_seed(42)
    random.seed(42)
    model = model_registry[opt.model_type](opt)
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        raise RuntimeError(f"parent restore was not recorded for {config_name}")
    configure_joint_formation_trainability(model, opt)
    if overlay is not None:
        saved = load_file(str(overlay), device="cpu")
        current = model.state_dict()
        missing = sorted(k for k in saved if k not in current)
        mismatch = sorted(k for k in saved if k in current and tuple(saved[k].shape) != tuple(current[k].shape))
        trainable = {k for k, p in model.named_parameters() if p.requires_grad}
        nontrainable = sorted(k for k in saved if k not in trainable)
        if missing or mismatch or nontrainable or set(saved) != trainable:
            raise RuntimeError(f"trainable overlay is not exact: missing={missing[:3]} mismatch={mismatch[:3]} nontrainable={nontrainable[:3]} saved={len(saved)} trainable={len(trainable)}")
        result = model.load_state_dict(saved, strict=False)
        # A trainable-only overlay deliberately omits frozen parent keys.
        # The exactness condition is therefore: every overlay key is applied,
        # with no unexpected key; parent loading already checked the full
        # active namespace strictly.
        if result.unexpected_keys or set(saved) - set(current):
            raise RuntimeError(f"overlay restore mismatch: {result}")
    model.to(accelerator.device)
    model.eval()
    return opt, model


def set_step(model, local_step: int):
    effective = 960 + int(local_step)
    model.set_token_eru_step(effective)
    model.set_token_eru_dino_metric_step(effective)
    model.set_token_eru_early_query_step(local_step)
    model.tsh_instance_loss_weight_eff = 1.0
    model.tsh_unit_grad_eff = 1.0
    model.teacher_lambda_eff = 0.0


def capture(model, batch, local_step: int, mode: str = "full", metrics: bool = True):
    set_step(model, local_step)
    model.eval()
    model.set_eqc_ablation(mode)
    decoder_input = {}

    def capture_decoder_input(_module, args):
        if args and torch.is_tensor(args[0]):
            decoder_input["reconstruction_query_tokens"] = args[0].detach().clone()

    hook = model.token_eru_decoder.register_forward_pre_hook(capture_decoder_input)
    try:
        with torch.inference_mode():
            out = model(batch, compute_quality_metrics=False)
    finally:
        hook.remove()
    # The model flattens the DINO auxiliary dictionary into its public output
    # dictionary; there is intentionally no nested eval-only container.
    dino = out
    head = model.tsh_instance_head
    values = {
        "encoder_latent": getattr(model, "_last_encoder_values", None),
        "reconstruction_query_tokens": decoder_input.get("reconstruction_query_tokens"),
        "r_hidden": getattr(model, "_token_eru_last_reconstruction_tokens", None),
        "u_hidden": getattr(model, "_token_eru_last_understanding_tokens", None),
        "native_query_seed": head.get_object_query_seed(int(batch["input"].shape[0])),
        "final_query_state": (
            getattr(model, "_token_eru_last_early_query_state", None)
            if getattr(model, "_token_eru_last_early_query_state", None) is not None
            else head.get_object_query_seed(int(batch["input"].shape[0]))
        ),
        "reconstruction_units": getattr(model, "_tsh_last_q_abs_live", None),
        "understanding_units": getattr(model, "_token_eru_understanding_units", None),
        "q_abs": getattr(model, "_tsh_last_q_abs_live", None),
        "gaussians": getattr(model, "_tsh_last_student_gaussians_live", None),
        "unit_logits": out.get("unit_logits"),
        "soft_masks": out.get("rendered_instance_group_probability"),
        "rgb": out.get("images_pred"),
        "alpha": out.get("alphas_pred"),
        "depth": out.get("depths_pred"),
        "fused_understanding_units": dino.get("fused_understanding_units"),
        "unit_metric_embeddings": dino.get("unit_metric_embeddings"),
    }
    if torch.is_tensor(values["gaussians"]):
        # These slices are the actual renderer-facing Gaussian tensor fields;
        # they are audit views only and do not enter the model graph.
        values["gaussian_xyz"] = values["gaussians"][..., :3]
        values["gaussian_opacity"] = values["gaussians"][..., 3:4]
        values["gaussian_sh"] = values["gaussians"][..., 11:]
    cpu_values = {k: (v.detach().float().cpu().clone() if torch.is_tensor(v) else None) for k, v in values.items()}
    if metrics:
        prob = cpu_values["soft_masks"][0].numpy()[:, :, 0]
        labels = batch["instance_label_output"][0].detach().long().cpu().numpy()
        scalars = prediction_metrics(prob, labels)
    else:
        scalars = {}
    finite_values = [v for v in values.values() if torch.is_tensor(v)]
    return {
        "mode": mode, "local_step": local_step, "effective_step": 960 + local_step,
        "early_gate": float(getattr(model, "_token_eru_early_query_gate", 0.0)),
        "parameter_hash": model_hash(model),
        "finite": all(finite(v) for v in finite_values),
        "tensor_hashes": {k: tensor_sha(v) for k, v in cpu_values.items() if v is not None},
        "tensors": cpu_values, **scalars,
        "loss": float(out["loss"].detach().cpu()),
        "loss_rgb": float(out.get("loss_rgb", torch.zeros(())).detach().cpu()),
        "loss_instance_group": float(out.get("loss_instance_group", torch.zeros(())).detach().cpu()),
        "loss_instance_metric": float(out.get("loss_instance_metric", torch.zeros(())).detach().cpu()),
        "psnr": float(out["psnr"].detach().cpu()),
        "permutation": getattr(model, "_token_eru_eqc_eval_permutation", None),
    }


def compact(record):
    return {k: v for k, v in record.items() if k not in ("tensors", "permutation")}


def diff_records(a, b):
    out = {}
    for key in sorted(set(a["tensors"]) | set(b["tensors"])):
        out[key + "_max_diff"] = maxdiff(a["tensors"].get(key), b["tensors"].get(key))
    for key in ("unit_logits", "soft_masks"):
        x, y = a["tensors"].get(key), b["tensors"].get(key)
        if x is not None and y is not None:
            out[key + "_argmax_change_ratio"] = float((x.argmax(-1) != y.argmax(-1)).float().mean()) if key == "unit_logits" else None
    return out


def grad_norms(model):
    groups = {
        "native_query_seed": ["tsh_instance_head.group_tokens"],
        "eqc_query_read": ["query_attention", "query_output"],
        "eqc_query_ffn": ["query_ffn"],
        "eqc_u_write": ["understanding_attention", "understanding_output"],
        "eqc_u_ffn": ["understanding_ffn"],
        "understanding_decoder": ["token_eru_decoder.understanding_decoder_blocks."],
        "r2u_adapters": ["token_eru_decoder.reconstruction_to_understanding."],
        "u2r_adapters": ["token_eru_decoder.understanding_to_reconstruction."],
        "reconstruction_decoder": ["enc_dec_backbone.decoder_blocks."],
        "reconstruction_unit": ["token_eru_unit_formation."],
        "absolute_gs_head": ["absolute_gs_head."],
        "dino_metric_projector": ["token_eru_dino_encoder.unit_projector."],
    }
    named = dict(model.named_parameters())
    result = {}
    for group, prefixes in groups.items():
        vals = [p.grad.detach().float().norm().item() for n, p in named.items() if any(x in n for x in prefixes) and p.grad is not None]
        result[group] = float(math.sqrt(sum(x * x for x in vals))) if vals else 0.0
    return result


def gradient_probe(model, batch, local_step, loss_key):
    model.train()
    model.set_eqc_ablation("full")
    set_step(model, local_step)
    model.zero_grad(set_to_none=True)
    out = model(batch, compute_quality_metrics=False)
    live_gaussians = getattr(model, "_tsh_last_student_gaussians_live", None)
    if torch.is_tensor(live_gaussians) and live_gaussians.requires_grad:
        live_gaussians.retain_grad()
    loss = out[loss_key]
    loss.backward()
    result = grad_norms(model)
    if torch.is_tensor(live_gaussians) and live_gaussians.grad is not None:
        result["actual_gaussian_xyz"] = float(live_gaussians.grad[..., :3].float().norm().item())
        result["actual_gaussian_opacity"] = float(live_gaussians.grad[..., 3:4].float().norm().item())
        result["actual_gaussian_sh"] = float(live_gaussians.grad[..., 11:].float().norm().item())
    else:
        result["actual_gaussian_xyz"] = 0.0
        result["actual_gaussian_opacity"] = 0.0
        result["actual_gaussian_sh"] = 0.0
    dino_extractor = getattr(getattr(model, "token_eru_dino_encoder", None), "dino_extractor", None)
    dino_model = None if dino_extractor is None else dino_extractor.__dict__.get("_dino_model")
    dino_grads = [] if dino_model is None else [p.grad for p in dino_model.parameters()]
    result["dino_backbone"] = float(
        math.sqrt(sum(float(g.detach().float().norm().item()) ** 2 for g in dino_grads if g is not None))
    ) if dino_grads else 0.0
    result.update({"loss": float(loss.detach().cpu()), "local_step": local_step, "loss_key": loss_key})
    model.zero_grad(set_to_none=True)
    model.eval()
    return result


def jsonable(value):
    if torch.is_tensor(value):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items() if k != "tensors"}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty audit directory: {output}")
    if not PARENT.is_file() or file_sha(PARENT) != PARENT_SHA:
        raise RuntimeError("J2 parent missing or SHA mismatch")
    if not E1_OVERLAY.is_file():
        raise RuntimeError(f"E1 step100 trainable-only snapshot missing: {E1_OVERLAY}")
    # An interrupted launcher may have created an empty directory before
    # Python reached the audit body.  Reusing that empty directory is safe;
    # any actual artifact still makes the run refuse to overwrite.
    output.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(mixed_precision="no", dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    if torch.cuda.is_available():
        torch.cuda.set_device(int(getattr(accelerator, "local_process_index", 0)))
    opt = dataclasses.replace(config_defaults[E0])
    opt.num_workers = 0
    loader, _, _, _ = get_multi_dataloader(opt, accelerator)
    raw_batch = next(iter(loader))
    batch = move(raw_batch, accelerator.device)
    if batch["input"].shape[1] != 15 or batch["instance_label_output"].shape[1] != 7:
        raise RuntimeError("expected one real 8+7 batch")
    bh = batch_hash(raw_batch)

    parent_opt, parent = build(J2, accelerator, output / "parent", None)
    e0_opt, e0 = build(E0, accelerator, output / "e0", None)
    e1_opt, e1 = build(E1, accelerator, output / "e1", None)
    parent0 = capture(parent, batch, 0)
    e00 = capture(e0, batch, 0)
    e10 = capture(e1, batch, 0)
    identity_keys = sorted(set(parent0["tensors"]) & set(e00["tensors"]) & set(e10["tensors"]))
    identity = {
        "batch_hash": bh,
        "parent_vs_e0": {k: maxdiff(parent0["tensors"].get(k), e00["tensors"].get(k)) for k in identity_keys},
        "e0_vs_e1": {k: maxdiff(e00["tensors"].get(k), e10["tensors"].get(k)) for k in identity_keys},
        "parent": compact(parent0), "e0": compact(e00), "e1": compact(e10),
        "early_gate_step0": e10["early_gate"],
        "q_state_equals_seed": maxdiff(e10["tensors"].get("final_query_state"), e10["tensors"].get("native_query_seed")),
        "qmc_enabled": bool(getattr(e1_opt, "token_eru_query_metric_enabled", False)),
        "anchor_enabled": bool(getattr(e1_opt, "token_eru_3d_anchor_enabled", False)),
        "scalar_diffs": {
            key: abs(float(parent0[key]) - float(e00[key]))
            for key in ("loss", "loss_rgb", "loss_instance_group", "loss_instance_metric", "psnr")
        },
        "e0_e1_scalar_diffs": {
            key: abs(float(e00[key]) - float(e10[key]))
            for key in ("loss", "loss_rgb", "loss_instance_group", "loss_instance_metric", "psnr")
        },
    }
    (output / "step0_identity.json").write_text(json.dumps(jsonable(identity), indent=2), encoding="utf-8")
    del parent, e0, parent_opt, e0_opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _, e1 = build(E1, accelerator, output / "e1_step100", E1_OVERLAY)
    step100 = {}
    for mode in MODES:
        record = capture(e1, batch, 100, mode)
        step100[mode] = compact(record)
        if record["permutation"] is not None:
            step100[mode]["permutation_sha256"] = tensor_sha(record["permutation"])
        step100[mode]["_record"] = record
    diffs = {mode: diff_records(step100["full"]["_record"], step100[mode]["_record"]) for mode in MODES if mode != "full"}
    (output / "step100_causal_ablation.json").write_text(json.dumps(jsonable({"modes": {k: compact(v["_record"]) for k, v in step100.items()}, "full_vs": diffs}), indent=2), encoding="utf-8")
    before_hash = model_hash(e1)
    for mode in MODES:
        e1.set_eqc_ablation(mode)
        if model_hash(e1) != before_hash:
            raise RuntimeError(f"parameter hash changed while selecting mode {mode}")

    gradients = {}
    for step in (1, 2, 25, 100):
        gradients[str(step)] = {
            "instance": gradient_probe(e1, batch, step, "loss_instance_group"),
            "rgb": gradient_probe(e1, batch, step, "loss_rgb"),
        }
    (output / "gradient_boundary.json").write_text(json.dumps(jsonable(gradients), indent=2), encoding="utf-8")

    # Compare the inherited reconstruction/geometry boundary against the E0
    # control.  The EQC-specific gradients are intentionally not required to
    # match: this comparison is only for the pre-existing geometry boundary.
    e0_for_grad = build(E0, accelerator, output / "e0_gradient", None)[1]
    e0_geometry = {}
    for step in (1, 2, 25, 100):
        probe = gradient_probe(e0_for_grad, batch, step, "loss_instance_group")
        e0_geometry[str(step)] = {
            key: probe.get(key, 0.0)
            for key in (
                "reconstruction_decoder", "reconstruction_unit", "absolute_gs_head",
                "r2u_adapters", "u2r_adapters", "actual_gaussian_xyz",
                "actual_gaussian_opacity", "actual_gaussian_sh",
            )
        }
    eqc_geometry = {
        step: {
            key: gradients[step]["instance"].get(key, 0.0)
            for key in (
                "reconstruction_decoder", "reconstruction_unit", "absolute_gs_head",
                "r2u_adapters", "u2r_adapters", "actual_gaussian_xyz",
                "actual_gaussian_opacity", "actual_gaussian_sh",
            )
        }
        for step in gradients
    }
    # “Boundary” means which namespaces receive a non-zero edge, not that
    # two independently evaluated losses have identical magnitudes.
    geometry_boundary_match = all(
        (float(e0_geometry[step][key]) > 1e-12)
        == (float(eqc_geometry[step][key]) > 1e-12)
        for step in e0_geometry for key in e0_geometry[step]
    )
    (output / "geometry_boundary_comparison.json").write_text(
        json.dumps({"e0": e0_geometry, "e1_eqc": eqc_geometry, "match": geometry_boundary_match}, indent=2),
        encoding="utf-8",
    )

    restored = build(E1, accelerator, output / "e1_restore", E1_OVERLAY)[1]
    restored_record = capture(restored, batch, 100, "full")
    restore_diff = diff_records(step100["full"]["_record"], restored_record)
    restore_report = {
        "restore_source": "strict J2 parent + exact E1 trainable-only overlay",
        "overlay_sha256": file_sha(E1_OVERLAY),
        "strict_parent_restore": True,
        "strict_overlay_restore": True,
        "full_state_checkpoint_present": False,
        "max_diffs": restore_diff,
        "default_mode_after_restore": getattr(restored, "_token_eru_eqc_eval_ablation", None),
        "local_step": 100,
        "effective_step": 1060,
        "early_gate": float(getattr(restored, "_token_eru_early_query_gate", 0.0)),
    }
    (output / "independent_restore.json").write_text(json.dumps(jsonable(restore_report), indent=2), encoding="utf-8")

    report = {
        "git_head": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "parent": {"path": str(PARENT), "sha256": file_sha(PARENT), "expected_sha256": PARENT_SHA},
        "e1_overlay": {"path": str(E1_OVERLAY), "sha256": file_sha(E1_OVERLAY)},
        "batch_hash": bh,
        "identity": identity,
        "causal": {"modes": {k: compact(v["_record"]) for k, v in step100.items()}, "diffs": diffs},
        "gradient_boundary": gradients,
        "geometry_boundary_comparison": {
            "e0": e0_geometry,
            "e1_eqc": eqc_geometry,
            "match": geometry_boundary_match,
        },
        "independent_restore": restore_report,
        "formal_training_started": False,
        "formal_evaluation_started": False,
    }
    (output / "readiness_report.json").write_text(json.dumps(jsonable(report), indent=2), encoding="utf-8")
    print(json.dumps(jsonable({k: report[k] for k in ("git_head", "parent", "batch_hash", "formal_training_started", "formal_evaluation_started")}), indent=2))


if __name__ == "__main__":
    main()
