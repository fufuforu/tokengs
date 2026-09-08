"""Fixed-sample overfit smoke for the SIU3R-mapped MBM joint run.

Loads the True-Shared development checkpoint (m0_ddp8 @ 1420), fixes one
real scene + one context/target choice, and runs a compressed schedule:

  0-30    head-only-ish warm-up (instance head trains, unit gradient 0)
  30-90   instance -> unit gradient ramp to the configured multiplier
  90-180  U->R (mask-guided depth smoothness) weight ramps 0 -> 0.05
  180-end full joint

Every 25 steps it logs instance loss / AP25 / AP50 / best-GT IoU / pred vs
gt counts / void share / assignment entropy / PSNR / U->R loss + effective
weights.  No formal training is started; output JSON lives in --workspace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safetensors.torch import load_file  # noqa: E402

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.utils.instance_ap import (  # noqa: E402
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)


class _LocalAccelerator:
    is_main_process = True

    @staticmethod
    def print(*args, **kwargs):
        print(*args, **kwargs)


def _load_heads_strict(model, ckpt, opt):
    abs_state = {
        key.split(".", 1)[1]: value
        for key, value in ckpt.items()
        if key.startswith("absolute_gs_head.")
    }
    res = model.absolute_gs_head.load_state_dict(abs_state, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    if any(key.startswith("tsh_instance_head.") for key in ckpt):
        tsh_state = {
            key.split(".", 1)[1]: value
            for key, value in ckpt.items()
            if key.startswith("tsh_instance_head.")
        }
        res = model.tsh_instance_head.load_state_dict(tsh_state, strict=True)
        assert not res.missing_keys and not res.unexpected_keys
        tsh_count = len(tsh_state)
    else:
        tsh_count = 0
    tail_state = {
        key: value
        for key, value in ckpt.items()
        if key.startswith("enc_dec_backbone.decoder_blocks.")
    }
    if float(getattr(opt, "tsh_mbm_decoder_tail_lr", 0.0)) > 0.0:
        expected = {
            key: value
            for key, value in model.state_dict().items()
            if key.startswith("enc_dec_backbone.decoder_blocks.")
        }
        if set(tail_state) != set(expected):
            raise RuntimeError(
                f"decoder tail strict load failed: {len(tail_state)}/"
                f"{len(expected)} keys"
            )
        res = torch.nn.Module.load_state_dict(model, tail_state, strict=False)
        tail_missing = [
            key for key in res.missing_keys
            if key.startswith("enc_dec_backbone.decoder_blocks.")
        ]
        tail_unexpected = [
            key for key in res.unexpected_keys
            if key.startswith("enc_dec_backbone.decoder_blocks.")
        ]
        if tail_missing or tail_unexpected:
            raise RuntimeError(
                f"decoder tail load mismatch: missing={tail_missing} "
                f"unexpected={tail_unexpected}"
            )
    return tsh_count


def _iou(a, b) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _instance_metrics(model, data, opt) -> dict:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(data, compute_quality_metrics=False)
    prob = out["rendered_instance_group_probability"].detach().float()
    labels = data["instance_label_output"].long()
    masks, scores = [], []
    gt_all = []
    active_ids = set()
    for v in range(prob.shape[2]):
        p = prob[0, :, v, 0].cpu().numpy()
        pred_id = p[:-1].argmax(axis=0)
        for g in range(p.shape[0] - 1):
            sel = pred_id == g
            if bool(sel.sum() > 0) and bool(p[g][sel].mean() > 0.5):
                active_ids.add(int(g))
        pm, ps = masks_from_group_probs(
            p,
            void_channel=prob.shape[1] - 1,
            min_mask_area=1,
        )
        masks.extend(pm)
        scores.extend(ps)
        gt_all.extend(
            gt_masks_from_instance_map(
                labels[0, v].cpu().numpy(), min_mask_area=1
            )
        )
    ap = instance_ap(
        masks, scores, gt_all, thresholds=(0.25, 0.5), vectorized=True
    )
    matched = []
    for g in gt_all:
        ious = [_iou(g, m) for m in masks]
        matched.append(max(ious) if ious else 0.0)
    entropy = -(
        out["pi_unit"][..., :-1]
        * out["pi_unit"][..., :-1].clamp_min(1e-8).log()
    ).sum(-1).mean().item()
    return {
        "ap25": float(ap["ap_25"]),
        "ap50": float(ap["ap_50"]),
        "mean_best_gt_iou": float(np.mean(matched)) if matched else 0.0,
        "num_pred": len(masks),
        "num_gt": len(gt_all),
        "active_queries": len(active_ids),
        "void_share": float(prob[0, -1].mean()),
        "assignment_entropy": entropy,
        "psnr": float(out["psnr"]),
        "instance_loss": float(out["loss_instance_group"]),
    }


def _unit_grad_norm(model, prefixes):
    total = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if any(name.startswith(p) for p in prefixes):
            total += float(param.grad.double().norm().item() ** 2)
    return float(total ** 0.5)


_NAN_PLACEHOLDERS = {"ssim", "lpips"}


def _out_finite(result) -> bool:
    for key, value in result.items():
        if key in _NAN_PLACEHOLDERS:
            continue
        if torch.is_tensor(value) and value.is_floating_point():
            if not bool(torch.isfinite(value).all().item()):
                return False
    return True


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace", default="workspace/tsh_mbm_overfit"
    )
    parser.add_argument(
        "--resume",
        default=(
            "workspace/semantic_v6_absolute_units_true_shared_head_only_m0_ddp8/"
            "checkpoints/model_step_001420.safetensors"
        ),
    )
    parser.add_argument("--config-name",
        default=(
            "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_ddp8"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--report-every", type=int, default=25)
    args = parser.parse_args()
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    opt = config_defaults[args.config_name]
    opt.workspace = str(out_dir)
    opt.num_workers = 0
    opt.prompt_overfit_single_batch = True
    opt.prompt_overfit_sample_index = args.sample_index
    opt.tsh_instance_warmup_steps = 30
    opt.tsh_instance_ramp_end_steps = 90
    opt.tsh_mbm_u2r_warmup_start_step = 90
    opt.tsh_mbm_u2r_warmup_steps = 90
    opt.tsh_per_gs_ramp_steps = 60
    opt.abs_bootstrap_steps = 4
    opt.abs_teacher_decay_steps = 6

    loader, _, train_dataset, _ = get_multi_dataloader(
        opt, _LocalAccelerator()
    )
    train_dataset.set_rng_epoch(0)
    data = next(iter(loader))
    data = {
        k: (v.cuda() if torch.is_tensor(v) else v)
        for k, v in data.items()
    }
    print(
        "[mbm-overfit] fixed sample",
        data.get("scene_name"),
        "index", args.sample_index,
    )

    model = model_registry[opt.model_type](opt).cuda().train()
    ckpt = load_file(args.resume, device="cpu")
    tsh_keys = _load_heads_strict(model, ckpt, opt)
    if tsh_keys == 0:
        raise RuntimeError("expected tsh_instance_head keys in resume ckpt")

    from tokengs.train import setup_optimizer

    optimizer = setup_optimizer(opt, model, _LocalAccelerator(), 0)
    mode = opt.tsh_mbm_mode
    records = []
    base_metrics = _instance_metrics(model, data, opt)
    base_metrics["step"] = 0
    base_metrics["mode"] = mode
    records.append(base_metrics)

    for step in range(1, args.steps + 1):
        step_effs = model.compute_tsh_effs(step, opt)
        model.tsh_instance_loss_weight_eff, model.tsh_unit_grad_eff = step_effs
        model.tsh_mbm_u2r_eff = model.compute_tsh_mbm_u2r_eff(step, opt)
        if bool(getattr(opt, "tsh_per_gs_refine", False)) and hasattr(
            model, "compute_tsh_per_gs_gate_eff"
        ):
            model.tsh_per_gs_gate_eff = (
                model.compute_tsh_per_gs_gate_eff(step, opt)
            )
        model.teacher_lambda_eff = model.compute_teacher_lambda_eff(step, opt)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(data, compute_quality_metrics=False)
        loss = out["loss"]
        loss.backward()
        total_norm = _unit_grad_norm(
            model, tuple(p for p, _ in model.named_parameters())
        )
        # instance-only contribution to unit formation (multiplier already
        # applied in the head path) is measured through grad_scale -> q_abs;
        # approximate with the weighted joint ratio from audited rows instead
        # of another backward here.
        optimizer.step()

        if step % args.report_every == 0:
            metrics = _instance_metrics(model, data, opt)
            metrics.update(
                {
                    "step": step,
                    "mode": mode,
                    "head_eff": step_effs[0],
                    "unit_eff": step_effs[1],
                    "u2r_eff": model.tsh_mbm_u2r_eff,
                    "teacher_eff": model.teacher_lambda_eff,
                    "loss": float(out["loss"]),
                    "loss_rgb": float(out["loss_rgb"]),
                    "loss_instance": float(out["loss_instance_group"]),
                    "loss_mbm_u2r_weighted": (
                        float(model._tsh_last_mbm_u2r_loss.item())
                        if model._tsh_last_mbm_u2r_loss is not None
                        else 0.0
                    ),
                    "pgsr_gate_eff": float(
                        getattr(model, "tsh_per_gs_gate_eff", 0.0)
                    ),
                    "pgsr_unit_logit_diff": (
                        float(out["tsh_per_gs_unit_logit_diff_mean"])
                        if "tsh_per_gs_unit_logit_diff_mean" in out
                        else 0.0
                    ),
                    "pgsr_alpha": (
                        float(out["tsh_per_gs_alpha"])
                        if "tsh_per_gs_alpha" in out
                        else 0.0
                    ),
                    "mbm_interior_x": (
                        float(out["mbm_u2r_interior_share_x"])
                        if "mbm_u2r_interior_share_x" in out
                        else 0.0
                    ),
                    "pre_clip_total_grad_norm": total_norm,
                    "finite": _out_finite(out),
                }
            )
            records.append(metrics)
            print(
                f"[mbm-overfit] step={step} mode={mode} "
                f"ap25={metrics['ap25']:.3f} ap50={metrics['ap50']:.3f} "
                f"iou={metrics['mean_best_gt_iou']:.3f} "
                f"pred/gt={metrics['num_pred']}/{metrics['num_gt']} "
                f"psnr={metrics['psnr']:.3f} "
                f"inst_loss={metrics['loss_instance']:.4f} "
                f"u2r={metrics['loss_mbm_u2r_weighted']:.6f} "
                f"eff(h/u/u2r/t)="
                f"{metrics['head_eff']:.2f}/{metrics['unit_eff']:.2f}/"
                f"{metrics['u2r_eff']:.3f}/{metrics['teacher_eff']:.2f} "
                f"clip_norm={total_norm:.3f} "
                f"finite={metrics['finite']}",
                flush=True,
            )

    report = {
        "mode": mode,
        "config": args.config_name,
        "resume": str(args.resume),
        "records": records,
    }
    with open(out_dir / "overfit_true_shared_mbm.json", "w") as handle:
        json.dump(report, handle, indent=2, default=float)
    print(
        f"[mbm-overfit] wrote {out_dir / 'overfit_true_shared_mbm.json'}"
    )


if __name__ == "__main__":
    _main()
