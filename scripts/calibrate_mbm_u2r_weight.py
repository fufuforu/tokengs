"""Empirical U->R weight calibration for the SIU3R-mapped MBM joint run.

For one candidate max weight W (0.5 / 1.0 / 2.5 / 5.0 / 10.0), on the same
checkpoint and the same 16 real batches (deterministic seed/sample order),
this script reports:

  * weighted U->R / GT-RGB grad ratios on
      - unit formation (tok_norm/tok_proj/unit_queries/unit_readout),
      - GS decoder (center_mlp/gs_decoder/slot_emb),
      - shared decoder tail (decoder_blocks);
  * grad cosine of the same three groups (U->R vs GT-RGB);
  * total joint pre-clip grad norm / clip trigger;
  * U->R loss, rendered-depth range and GS position/scale/opacity stats;
  * NaN/abnormal-GS checks;
  * predicted-instance region diagnostics (region count, void share,
    confidence-gated valid share, boundary-pixel share).

The U->R formula / predicted-mask policy / depth source / warm-up schedule /
True Shared structure are unchanged; only the effective max weight differs.
No model, config or checkpoint is modified and no optimizer step is taken
here (real Adam steps are measured by measure_mbm_optimizer_updates.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safetensors.torch import load_file  # noqa: E402

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


class _LocalAccelerator:
    is_main_process = True

    @staticmethod
    def print(*args, **kwargs):
        print(*args, **kwargs)


def _load_heads_strict(model, ckpt, opt) -> int:
    abs_state = {
        key.split(".", 1)[1]: value
        for key, value in ckpt.items()
        if key.startswith("absolute_gs_head.")
    }
    res = model.absolute_gs_head.load_state_dict(abs_state, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    tsh_keys = 0
    if any(key.startswith("tsh_instance_head.") for key in ckpt):
        tsh_state = {
            key.split(".", 1)[1]: value
            for key, value in ckpt.items()
            if key.startswith("tsh_instance_head.")
        }
        res = model.tsh_instance_head.load_state_dict(tsh_state, strict=True)
        assert not res.missing_keys and not res.unexpected_keys
        tsh_keys = len(tsh_state)
    return tsh_keys


_FORM = (
    "absolute_gs_head.tok_norm.",
    "absolute_gs_head.tok_proj.",
    "absolute_gs_head.unit_queries",
    "absolute_gs_head.unit_readout.",
)
_GS_DEC = (
    "absolute_gs_head.center_mlp.",
    "absolute_gs_head.gs_decoder.",
    "absolute_gs_head.slot_emb",
)
_TAIL = "enc_dec_backbone.decoder_blocks."
_TSH = "tsh_instance_head."
_NAN_PLACEHOLDERS = {"ssim", "lpips"}


def _group_norm(model, prefixes) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if any(name.startswith(p) for p in prefixes):
            total += float(param.grad.double().norm().item() ** 2)
    return float(total ** 0.5)


def _group_vec(model, prefixes):
    parts = []
    for name, param in model.named_parameters():
        if not any(name.startswith(p) for p in prefixes):
            continue
        if param.grad is not None:
            parts.append(param.grad.detach().float().flatten())
        else:
            parts.append(torch.zeros_like(param).flatten())
    return torch.cat(parts) if parts else torch.zeros(1)


def _cosine(a, b) -> float | None:
    if a.numel() == 0 or b.numel() == 0:
        return None
    return float(
        torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)
        ).item()
    )


def _out_finite(result) -> bool:
    for key, value in result.items():
        if key in _NAN_PLACEHOLDERS:
            continue
        if torch.is_tensor(value) and value.is_floating_point():
            if not bool(torch.isfinite(value).all().item()):
                return False
    return True


def _region_diagnostics(result) -> dict:
    probs = result["rendered_instance_group_probability"].detach().float()
    alpha = result["rendered_instance_group_alpha"].detach().float()
    conf = probs.max(dim=1).values
    ids = probs.argmax(dim=1)
    g_plus_1 = probs.shape[1]
    void_idx = g_plus_1 - 1
    valid = (conf >= 0.3) & (ids != void_idx) & (alpha >= 0.05)
    v = valid[:, :, 0]  # [B,V,H,W]
    idm = ids[:, :, 0]
    b, view, h, w = idm.shape
    region_counts = []
    large_masks = []
    max_area_shares = []
    pixels_per_image = h * w
    for bi in range(b):
        for vi in range(view):
            ids_v = idm[bi, vi]
            counts = torch.bincount(
                ids_v.flatten(), minlength=g_plus_1
            )
            region_counts.append(
                int((counts[:void_idx] >= 25).sum().item())
            )
            fg_counts = counts[:void_idx].float()
            if fg_counts.numel():
                max_area_shares.append(
                    float(fg_counts.max().item() / pixels_per_image)
                )
                large_masks.append(
                    int((fg_counts > 0.25 * pixels_per_image).sum().item())
                )
    # Approximate boundary-pixel share among valid pixels: a valid pixel is a
    # boundary if its right or bottom 4-neighbor is invalid or a different id.
    right_diff = (idm[..., 1:] != idm[..., :-1]) | (
        ~v[..., 1:] | ~v[..., :-1]
    )
    bottom_diff = (
        idm[:, :, 1:, :] != idm[:, :, :-1, :]
    ) | (~v[:, :, 1:, :] | ~v[:, :, :-1, :])
    right_valid = v[..., 1:] & v[..., :-1]
    bottom_valid = v[:, :, 1:, :] & v[:, :, :-1, :]
    boundary_px = (
        (right_diff & right_valid).float().sum()
        + (bottom_diff & bottom_valid).float().sum()
    ) / max(2.0, (v.float().sum() * 2.0).item())
    return {
        "region_count_mean": float(
            torch.as_tensor(region_counts).float().mean().item()
        ),
        "region_count_max": max(region_counts),
        "region_count_min": min(region_counts),
        "max_mask_area_share_mean": (
            float(torch.as_tensor(max_area_shares).mean().item())
            if max_area_shares else None
        ),
        "max_mask_area_share_max": max(max_area_shares) if max_area_shares else None,
        "large_mask_count_mean": (
            float(torch.as_tensor(large_masks).float().mean().item())
            if large_masks else None
        ),
        "void_share": float(probs[:, void_idx].mean().item()),
        "valid_share": float(v.float().mean().item()),
        "boundary_pixel_share": float(boundary_px),
    }


def _gs_stats(gaussians) -> dict:
    pos = gaussians[..., :3].float()
    scale = gaussians[..., 4:7].float()
    opacity = gaussians[..., 3:4].float()
    return {
        "pos_norm_mean": float(pos.norm(dim=-1).mean().item()),
        "pos_norm_std": float(pos.norm(dim=-1).std().item()),
        "pos_max_abs": float(pos.abs().max().item()),
        "scale_mean": float(scale.mean().item()),
        "scale_std": float(scale.std().item()),
        "scale_max": float(scale.max().item()),
        "opacity_mean": float(opacity.mean().item()),
        "opacity_std": float(opacity.std().item()),
        "opacity_max": float(opacity.max().item()),
    }


def _depth_stats(result) -> dict:
    depth = result["depths_pred"].detach().float()
    alpha = result["alphas_pred"].detach().float()
    valid = (depth > 0) & (alpha >= 0.05) & torch.isfinite(depth)
    vals = depth[valid]
    if vals.numel() == 0:
        return {"depth_min": None, "depth_max": None, "depth_mean": None,
                "depth_std": None, "valid_share": 0.0}
    return {
        "depth_min": float(vals.min().item()),
        "depth_max": float(vals.max().item()),
        "depth_mean": float(vals.mean().item()),
        "depth_std": float(vals.std().item()),
        "valid_share": float(valid.float().mean().item()),
    }


def _backward_component_metrics(model, result, opt, inst_weight, W) -> dict:
    loss_rgb = result["loss_rgb"]
    loss_inst = result["loss_instance_group"]
    u2r_term = getattr(model, "_tsh_last_mbm_u2r_loss", None)
    metrics = {}
    # GT RGB gradients.
    model.zero_grad(set_to_none=True)
    loss_rgb.backward(retain_graph=True)
    gt = {
        "unit": _group_norm(model, _FORM),
        "gs_decoder": _group_norm(model, _GS_DEC),
        "tail": _group_norm(model, _TAIL),
    }
    gt_vecs = {
        "unit": _group_vec(model, _FORM),
        "gs_decoder": _group_vec(model, _GS_DEC),
        "tail": _group_vec(model, _TAIL),
    }
    # Weighted instance gradients.
    model.zero_grad(set_to_none=True)
    (loss_inst * inst_weight).backward(retain_graph=True)
    inst = {
        "unit": _group_norm(model, _FORM),
        "gs_decoder": _group_norm(model, _GS_DEC),
        "tail": _group_norm(model, _TAIL),
        "tsh": _group_norm(model, _TSH),
    }
    inst_vecs = {
        "unit": _group_vec(model, _FORM),
        "tail": _group_vec(model, _TAIL),
    }
    # Weighted U->R gradients (u2r_term already includes W).
    if u2r_term is not None and W > 0.0:
        model.zero_grad(set_to_none=True)
        u2r_term.backward(retain_graph=True)
        u2r = {
            "unit": _group_norm(model, _FORM),
            "gs_decoder": _group_norm(model, _GS_DEC),
            "tail": _group_norm(model, _TAIL),
            "tsh": _group_norm(model, _TSH),
        }
        u2r_vecs = {
            "unit": _group_vec(model, _FORM),
            "gs_decoder": _group_vec(model, _GS_DEC),
            "tail": _group_vec(model, _TAIL),
        }
    else:
        u2r = {k: 0.0 for k in ("unit", "gs_decoder", "tail", "tsh")}
        u2r_vecs = {}
    for key in ("unit", "gs_decoder", "tail"):
        metrics[f"u2r_gt_ratio_{key}"] = (
            u2r[key] / max(gt[key], 1e-12) if gt[key] > 0 else None
        )
        metrics[f"inst_gt_ratio_{key}"] = (
            inst[key] / max(gt[key], 1e-12) if gt[key] > 0 else None
        )
        metrics[f"cosine_inst_gt_{key}"] = (
            _cosine(inst_vecs[key], gt_vecs[key])
            if key in inst_vecs and gt_vecs[key].numel() > 1
            else None
        )
        metrics[f"cosine_u2r_gt_{key}"] = (
            _cosine(u2r_vecs[key], gt_vecs[key])
            if key in u2r_vecs and gt_vecs[key].numel() > 1
            else None
        )
    metrics["inst_gt_ratio_tsh"] = inst["tsh"] / max(
        gt["unit"], 1e-12
    )
    metrics["gt_grad_norms"] = gt
    metrics["u2r_grad_norms"] = u2r
    metrics["inst_grad_norms"] = inst
    return metrics


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--u2r-weight", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-batches", type=int, default=16)
    parser.add_argument(
        "--config-name",
        default=(
            "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_ddp8"
        ),
    )
    args = parser.parse_args()
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    W = float(args.u2r_weight)

    opt = config_defaults[args.config_name]
    opt.workspace = str(out_dir)
    opt.num_workers = 0
    opt.tsh_mbm_u2r_weight = W
    torch.manual_seed(args.seed)
    loader, _, _, _ = get_multi_dataloader(opt, _LocalAccelerator())

    model = model_registry[opt.model_type](opt).cuda().train()
    ckpt = load_file(args.resume, device="cpu")
    tsh_keys = _load_heads_strict(model, ckpt, opt)
    model.tsh_instance_loss_weight_eff = 1.0
    model.tsh_unit_grad_eff = float(
        getattr(opt, "tsh_unit_gradient_multiplier_max", 0.0)
    )
    model.tsh_mbm_u2r_eff = W
    model.teacher_lambda_eff = 0.0

    inst_weight = float(getattr(opt, "tsh_lambda_instance", 0.05)) * 1.0
    report = {
        "u2r_weight": W,
        "resume": str(args.resume),
        "config": args.config_name,
        "tsh_keys_loaded": tsh_keys,
        "multiplier": opt.tsh_unit_gradient_multiplier_max,
    }
    batches = []
    iterator = iter(loader)
    for batch_id in range(args.n_batches):
        try:
            data = next(iterator)
        except StopIteration:
            break
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        scene = str(data.get("scene_name", ("?",))[0])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(data, compute_quality_metrics=False)
        metrics = _backward_component_metrics(
            model, result, opt, inst_weight, W
        )
        # Joint loss pre-clip norm / clip trigger.
        model.zero_grad(set_to_none=True)
        result["loss"].backward()
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                total_norm += float(
                    param.grad.double().norm().item() ** 2
                )
        total_norm = float(total_norm ** 0.5)
        metrics.update(
            {
                "batch_id": batch_id,
                "scene": scene,
                "u2r_raw_loss": float(result["loss_mbm_u2r"].item()),
                "u2r_effective_loss": (
                    float(model._tsh_last_mbm_u2r_loss.item())
                    if model._tsh_last_mbm_u2r_loss is not None
                    else 0.0
                ),
                "joint_preclip_total_norm": total_norm,
                "clip_triggered": bool(
                    total_norm
                    > float(getattr(opt, "gradient_clip", 1.0))
                ),
                "loss_rgb": float(result["loss_rgb"].item()),
                "loss_instance": float(
                    result["loss_instance_group"].item()
                ),
                "psnr": float(result["psnr"].item()),
                "depth": _depth_stats(result),
                "gs": _gs_stats(result["gaussians"]),
                "regions": _region_diagnostics(result),
                "finite": _out_finite(result),
            }
        )
        batches.append(metrics)
        print(
            f"[calib] W={W} batch={batch_id} scene={scene} "
            f"r_unit={metrics['u2r_gt_ratio_unit']:.6f} "
            f"r_gsdec={metrics['u2r_gt_ratio_gs_decoder']:.6f} "
            f"r_tail={metrics['u2r_gt_ratio_tail']:.6f} "
            f"cos_unit={metrics['cosine_u2r_gt_unit']} "
            f"clip={metrics['clip_triggered']} "
            f"finite={metrics['finite']}",
            flush=True,
        )
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    def _summ(fn):
        vals = [fn(b) for b in batches if fn(b) is not None]
        if not vals:
            return None
        ordered = sorted(vals)
        return {
            "min": ordered[0],
            "median": statistics.median(ordered),
            "p90": ordered[
                min(len(ordered) - 1, int(0.90 * len(ordered)))
            ],
            "max": ordered[-1],
            "mean": statistics.fmean(ordered),
        }

    report["n_batches"] = len(batches)
    report["summary"] = {
        "u2r_gt_ratio_unit": _summ(lambda b: b["u2r_gt_ratio_unit"]),
        "u2r_gt_ratio_gs_decoder": _summ(
            lambda b: b["u2r_gt_ratio_gs_decoder"]
        ),
        "u2r_gt_ratio_tail": _summ(lambda b: b["u2r_gt_ratio_tail"]),
        "inst_gt_ratio_unit": _summ(lambda b: b["inst_gt_ratio_unit"]),
        "inst_gt_ratio_gs_decoder": _summ(
            lambda b: b["inst_gt_ratio_gs_decoder"]
        ),
        "inst_gt_ratio_tail": _summ(lambda b: b["inst_gt_ratio_tail"]),
        "cosine_u2r_gt_unit": _summ(lambda b: b["cosine_u2r_gt_unit"]),
        "cosine_u2r_gt_gs_decoder": _summ(
            lambda b: b["cosine_u2r_gt_gs_decoder"]
        ),
        "cosine_u2r_gt_tail": _summ(lambda b: b["cosine_u2r_gt_tail"]),
        "cosine_inst_gt_unit": _summ(lambda b: b["cosine_inst_gt_unit"]),
        "cosine_inst_gt_tail": _summ(lambda b: b["cosine_inst_gt_tail"]),
        "joint_preclip_total_norm": _summ(
            lambda b: b["joint_preclip_total_norm"]
        ),
        "clip_triggered_count": sum(
            1 for b in batches if b["clip_triggered"]
        ),
        "nan_count": sum(1 for b in batches if not b["finite"]),
        "u2r_effective_loss": _summ(lambda b: b["u2r_effective_loss"]),
        "loss_rgb": _summ(lambda b: b["loss_rgb"]),
        "psnr": _summ(lambda b: b["psnr"]),
        "region_count_mean": _summ(lambda b: b["regions"]["region_count_mean"]),
        "region_count_max": max(b["regions"]["region_count_max"] for b in batches)
        if batches else None,
        "void_share": _summ(lambda b: b["regions"]["void_share"]),
        "boundary_pixel_share": _summ(
            lambda b: b["regions"]["boundary_pixel_share"]
        ),
        "max_mask_area_share": _summ(
            lambda b: b["regions"]["max_mask_area_share_mean"]
        ),
        "large_mask_count": _summ(
            lambda b: b["regions"]["large_mask_count_mean"]
        ),
    }
    report["per_batch"] = batches
    out_path = out_dir / f"calibration_w{W:g}.json"
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(
        f"[calib] W={W} wrote {out_path} ({len(batches)} batches)"
    )


if __name__ == "__main__":
    _main()
