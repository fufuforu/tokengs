"""Fixed-batch 100-step learnability audit for GA-IDU-1.

This is a diagnostic only: it never writes a formal checkpoint and never
changes the GA-IDU training configuration used by the formal trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
torch._dynamo.config.disable = True
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.options import config_defaults
from tokengs.train import load_model_checkpoint, setup_optimizer
from scripts.eval_instance_lsm_protocol import (
    _mask_diagnostics,
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)

CKPT = "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors"


def tensor_hash(value):
    h = hashlib.sha256()
    h.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:24]


def parameter_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:24]


def loss_parts(output):
    parts = {"total": float(output["loss_instance_group"].detach())}
    for name, value in output.items():
        if name.startswith("loss_instance_group_"):
            parts[name.removeprefix("loss_instance_group_")] = float(value.detach())
    return parts


def parameter_stats(model, initial, previous):
    out = {}
    for module in ("memory_resampler", "instance_decoder", "group_residual_decoder", "assignment_residual"):
        prefix = f"ga_idu1_head.{module}."
        grad_sq = update_sq = cumulative_sq = initial_sq = nonzero = count = 0.0
        for name, p in model.named_parameters():
            if not name.startswith(prefix):
                continue
            now = p.detach().float()
            old = initial[name]
            prev = previous[name]
            count += p.numel()
            initial_sq += float(old.square().sum())
            update_sq += float((now - prev).square().sum())
            cumulative_sq += float((now - old).square().sum())
            if p.grad is not None:
                g = p.grad.detach().float()
                grad_sq += float(g.square().sum())
                nonzero += float((g != 0).sum())
        raw = grad_sq ** 0.5
        out[module] = {
            "raw_grad_norm": raw,
            "grad_norm_sqrt_param": raw / max(count, 1.0) ** 0.5,
            "parameter_update_norm": update_sq ** 0.5,
            "cumulative_update_norm": cumulative_sq ** 0.5,
            "update_norm_initial_param_norm": (cumulative_sq / max(initial_sq, 1e-30)) ** 0.5,
            "nonzero_grad_parameter_fraction": nonzero / max(count, 1.0),
            "parameter_count": int(count),
        }
    return out


def module_grad_norm(model, module):
    prefix = f"ga_idu1_head.{module}."
    value = 0.0
    for name, param in model.named_parameters():
        if name.startswith(prefix) and param.grad is not None:
            value += float(param.grad.detach().float().square().sum())
    return value ** 0.5


def evaluate(output, data, base_logits):
    prob = output["rendered_instance_group_probability"][0].float().detach().cpu().numpy()
    labels = data["instance_label_output"][0].long().cpu().numpy()
    preds, scores, pred_ids, gts, gt_ids = [], [], [], [], []
    entropy = []
    for view in range(prob.shape[1]):
        p = prob[:, view, 0]
        nonvoid = p[:-1]
        q = nonvoid / np.maximum(nonvoid.sum(0, keepdims=True), 1e-8)
        entropy.append(float(-(q * np.log(np.maximum(q, 1e-8))).sum(0).mean()))
        masks, view_scores = masks_from_group_probs(
            p, void_channel=prob.shape[0] - 1, min_mask_area=1
        )
        image_id = f"{data['scene_name'][0]}:b0"
        preds.extend(masks); scores.extend(view_scores); pred_ids.extend([image_id] * len(masks))
        view_gts = gt_masks_from_instance_map(labels[view], min_mask_area=1)
        gts.extend(view_gts); gt_ids.extend([image_id] * len(view_gts))
    ap = instance_ap(
        preds, scores, gts, thresholds=(0.25, 0.5, 0.75), vectorized=True,
        pred_image_ids=pred_ids, gt_image_ids=gt_ids,
    )
    diag = _mask_diagnostics(preds, gts, pred_ids, gt_ids)
    pi = output["instance_group_probabilities"][0].float().detach()
    usage = pi[..., :-1].mean(dim=(0, 1, 2))
    logits = output["unit_logits"][0].float().detach()
    base = base_logits[0].float().detach()
    changed = (logits[..., :-1].argmax(-1) != base[..., :-1].argmax(-1)).float().mean()
    delta = logits[..., :-1] - base[..., :-1]
    base_margin = torch.topk(base[..., :-1], 2, dim=-1).values
    base_margin = base_margin[..., 0] - base_margin[..., 1]
    final_groups = logits[..., :-1].argmax(-1)
    base_groups = base[..., :-1].argmax(-1)
    crossed = delta.abs().amax(-1) > base_margin
    pi_final = torch.softmax(logits.float(), dim=-1)
    pi_base = torch.softmax(base.float(), dim=-1)
    kl = (pi_final * (pi_final.clamp_min(1e-8).log() - pi_base.clamp_min(1e-8).log())).sum(-1).mean()
    rendered = output["rendered_instance_group_probability"].float().detach()
    # Compare against the base-rendered probability if it is supplied by the
    # caller; the normal audit path computes this separately.
    # GA-IDU does not export its internal delta as a public model output;
    # the exact observable residual is final unit logits minus the anchored
    # BaseTSH logits (void is unchanged and is excluded below).
    flat = (
        output["unit_logits"][..., :-1].float().detach()
        - output["base_unit_logits"][..., :-1].float().detach()
    ).abs().flatten().cpu().numpy()
    return {
        "instance_loss": float(output["loss_instance_group"].detach()),
        "ap25": float(ap["ap_25"]), "ap50": float(ap["ap_50"]), "ap75": float(ap["ap_75"]),
        **diag,
        "pred_gt": len(preds) / max(1, len(gts)),
        "mass_query_count_gt_0.001": int((usage > 0.001).sum()),
        "nonempty_prediction_count": int(len(preds)),
        "effective_query_count": float(np.exp(np.mean(entropy))),
        "void_ratio": float(prob[-1].mean()),
        "assignment_entropy": float(np.mean(entropy)),
        "unit_argmax_changed_vs_base": float(changed),
        "base_margin": {
            "mean": float(base_margin.mean()),
            "p50": float(torch.quantile(base_margin.flatten(), .50)),
            "p75": float(torch.quantile(base_margin.flatten(), .75)),
            "p90": float(torch.quantile(base_margin.flatten(), .90)),
            "p95": float(torch.quantile(base_margin.flatten(), .95)),
            "p99": float(torch.quantile(base_margin.flatten(), .99)),
        },
        "delta_crosses_base_margin_fraction": float(crossed.float().mean()),
        "group_argmax_changed_fraction": float((final_groups != base_groups).float().mean()),
        "group_to_void_migrations": int(((logits[..., :-1].argmax(-1) != base_groups) & (pi_final[..., -1] > pi_base[..., -1])).sum()),
        "soft_probability_mean_abs_diff": float((pi_final - pi_base).abs().mean()),
        "soft_probability_max_abs_diff": float((pi_final - pi_base).abs().max()),
        "assignment_kl_final_base": float(kl),
        "logit_hashes": {
            "base": tensor_hash(output["base_unit_logits"]),
            "delta": tensor_hash(output["unit_logits"] - output["base_unit_logits"]),
            "final": tensor_hash(output["unit_logits"]),
        },
        "delta_logits_abs": {
            "mean": float(flat.mean()), "std": float(flat.std()),
            "p50": float(np.percentile(flat, 50)), "p90": float(np.percentile(flat, 90)),
            "p99": float(np.percentile(flat, 99)), "max": float(flat.max()),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--config", default="semantic_v6_absolute_units_true_shared_ga_idu1_ddp8")
    parser.add_argument("--resume", default=CKPT)
    parser.add_argument("--scene-id", default="scene0198_00")
    args = parser.parse_args()
    out = ROOT / args.workspace
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing non-empty workspace: {out}")
    out.mkdir(parents=True, exist_ok=True)
    opt = config_defaults[args.config]
    opt.workspace = str(out)
    opt.resume = str(ROOT / args.resume) if not os.path.isabs(args.resume) else args.resume
    opt.num_workers = 0; opt.num_epochs = 1; opt.max_iters_per_epoch = args.steps
    opt.eval_before_training = False; opt.use_wandb = False
    acc = Accelerator(
        mixed_precision=opt.mixed_precision,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    loader, _, _, _ = get_multi_dataloader(opt, acc)
    model = model_registry[opt.model_type](opt).cuda().train()
    raw = load_file(opt.resume, device="cpu")
    counts = {
        "absolute_gs_head": sum(k.startswith("absolute_gs_head.") for k in raw),
        "tsh_instance_head": sum(k.startswith("tsh_instance_head.") for k in raw),
        "decoder_tail": sum(k.startswith("enc_dec_backbone.decoder_blocks.") for k in raw),
    }
    assert counts == {"absolute_gs_head": 24, "tsh_instance_head": 50, "decoder_tail": 324}
    assert not any("pgsr" in k.lower() or "query_memory_refiner" in k.lower() for k in raw)
    load_model_checkpoint(opt, model, acc, 0)
    optimizer = setup_optimizer(opt, model, acc, 0)
    model, optimizer, loader, _ = acc.prepare(model, optimizer, loader, loader)
    base = acc.unwrap_model(model)
    trainable = [n for n, p in base.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("ga_idu1_head.") for n in trainable), trainable[:10]
    iterator = iter(loader)
    data = None
    for _ in range(10000):
        candidate = next(iterator)
        candidate_scene = str(candidate["scene_name"][0])
        if not args.scene_id or candidate_scene == args.scene_id:
            data = candidate
            break
    if data is None:
        raise RuntimeError(f"fixed scene not found: {args.scene_id}")
    initial_params = {
        n: p.detach().float().clone()
        for n, p in base.named_parameters()
        if n.startswith("ga_idu1_head.")
    }

    base.ga_idu_eval_gate_override = 0.0
    model.eval()
    with torch.no_grad():
        zero = model(data, compute_quality_metrics=False)
    base_logits = zero["base_unit_logits"].detach().clone()
    assert float((zero["unit_logits"] - zero["base_unit_logits"]).abs().max()) <= 1e-6
    metrics = {
        "step0": evaluate(zero, data, base_logits),
        "checkpoint_counts": counts,
        "fresh_reset": False,
        "pgsr_absent": True,
        "trainable": trainable,
        "fixed_scene": str(data["scene_name"][0]),
        "optimizer": [
            {"lr": group.get("lr"), "parameter_count": sum(p.numel() for p in group["params"])}
            for group in optimizer.param_groups
        ],
        "step_records": [],
    }
    delattr(base, "ga_idu_eval_gate_override")
    model.train()
    previous_params = {n: v.clone() for n, v in initial_params.items()}
    for step in range(args.steps):
        base.ga_idu_gate_eff = min(1.0, (step + 1) / float(max(1, opt.ga_idu_gate_steps)))
        base.tsh_instance_loss_weight_eff = 1.0
        base.tsh_unit_grad_eff = 0.0; base.tsh_mbm_u2r_eff = 0.0; base.teacher_lambda_eff = 0.0
        optimizer.zero_grad(set_to_none=True)
        with acc.autocast():
            current = model(data, compute_quality_metrics=False)
        loss = current["loss_instance_group"]
        assert torch.isfinite(loss).all()
        current_loss_parts = loss_parts(current)
        acc.backward(loss)
        grad_before_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
        gradients = {
            name: module_grad_norm(base, name)
            for name in ("memory_resampler", "instance_decoder", "group_residual_decoder", "assignment_residual")
        }
        grad_after_clip = float(acc.clip_grad_norm_(model.parameters(), opt.gradient_clip))
        optimizer.step()
        if step + 1 in (1, 5, 25, 50, 100):
            current_params = {
                n: p.detach().float().clone()
                for n, p in base.named_parameters()
                if n.startswith("ga_idu1_head.")
            }
            record = {
                "optimizer_step": step + 1,
                "scene_id": str(data["scene_name"][0]),
                "s_inst": float(base.ga_idu_gate_eff),
                "parameter_hash": parameter_hash(base),
                "loss_parts": current_loss_parts,
                "gradient_norm_before_clip": grad_before_clip,
                "gradient_norm_after_clip": grad_after_clip,
                "optimizer_step_executed": True,
                "module_stats": parameter_stats(base, initial_params, previous_params),
            }
            previous_params = current_params
            model.eval()
            with torch.no_grad(): snapshot = model(data, compute_quality_metrics=False)
            metrics[f"step{step + 1}"] = evaluate(snapshot, data, base_logits)
            metrics[f"step{step + 1}"]["gate"] = float(base.ga_idu_gate_eff)
            metrics[f"step{step + 1}"]["gradient_norms"] = gradients
            metrics[f"step{step + 1}"]["loss_parts_train"] = current_loss_parts
            metrics[f"step{step + 1}"]["parameter_hash"] = parameter_hash(base)
            record["eval_parameter_hash"] = parameter_hash(base)
            record["eval_logits"] = {
                "base": metrics[f"step{step + 1}"]["logit_hashes"]["base"],
                "delta": metrics[f"step{step + 1}"]["logit_hashes"]["delta"],
                "final": metrics[f"step{step + 1}"]["logit_hashes"]["final"],
            }
            model.train()
            metrics["step_records"].append(record)
    metrics["steps"] = args.steps
    metrics["all_finite"] = True
    if acc.is_main_process:
        (out / "ga_idu_fixed_batch_audit.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
