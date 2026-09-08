"""SIU3R-mapped MBM audit for the True-Shared joint run.

Read-only-ish audit that loads a True-Shared checkpoint (development init:
m0_ddp8 model_step_001420), runs real batches and reports, for the RGB /
instance (raw and weighted) / U->R (mask-guided depth smoothness) losses,
per-module gradient norms:

  - absolute unit formation  (tok_norm/tok_proj/unit_queries/unit_readout)
  - absolute GS decoder      (center_mlp/gs_decoder/slot_emb)
  - full absolute_gs_head
  - tsh_instance_head
  - decoder tail (decoder_blocks, when enabled)
  - image encoder / activation-head teacher / semantic modules (must be 0)

It also computes the GT vs instance r_unit ratio, the GT/U->R ratio,
gradient cosine, total pre-clip grad norm, and reports min/median/P90/max
over n batches.  No model/config/checkpoint is modified and no training is
started.

Example (240, one 3090 GPU):
    python -u scripts/audit_true_shared_mbm.py \
      --workspace workspace/tsh_mbm_audit \
      --resume workspace/semantic_v6_absolute_units_true_shared_head_only_m0_ddp8/checkpoints/model_step_001420.safetensors \
      --config-name semantic_v6_absolute_units_true_shared_siu3r_mbm_both_ddp8
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


def _sha(t: torch.Tensor) -> str:
    data = t.detach().float().cpu().contiguous()
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()[:16]


def _head_hashes(model) -> dict:
    return {
        name: _sha(param)
        for name, param in model.named_parameters()
        if name.startswith(("absolute_gs_head.", "tsh_instance_head."))
    }


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
    else:
        fresh_seed = (int(opt.seed) + 987654321) % (2**31)
        torch.manual_seed(fresh_seed)
        model.tsh_instance_head.reset_parameters_fresh()
    tail_state = {
        key: value
        for key, value in ckpt.items()
        if key.startswith("enc_dec_backbone.decoder_blocks.")
    }
    if float(getattr(opt, "tsh_mbm_decoder_tail_lr", 0.0)) > 0.0:
        expected = {
            key for key in model.state_dict()
            if key.startswith("enc_dec_backbone.decoder_blocks.")
        }
        if set(tail_state) != expected:
            raise RuntimeError(
                f"decoder tail strict load failed: {len(tail_state)}/"
                f"{len(expected)} keys"
            )
        res = torch.nn.Module.load_state_dict(model, tail_state, strict=False)
        missing = [k for k in res.missing_keys if k in expected]
        unexpected = [k for k in res.unexpected_keys if k.startswith("enc_dec_backbone.decoder_blocks.")]
        if missing or unexpected:
            raise RuntimeError(
                f"decoder tail load mismatch: missing={missing} "
                f"unexpected={unexpected}"
            )
    return tsh_keys


def _group_norm(model, prefix: tuple[str, ...] | str) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if isinstance(prefix, str):
            hit = name.startswith(prefix)
        else:
            hit = any(name.startswith(p) for p in prefix)
        if hit:
            total += float(param.grad.double().norm().item() ** 2)
    return float(total ** 0.5)


def _set_full_joint(model, opt) -> None:
    model.tsh_instance_loss_weight_eff = 1.0
    model.tsh_unit_grad_eff = float(
        getattr(opt, "tsh_unit_gradient_multiplier_max", 0.0)
    )
    model.tsh_mbm_u2r_eff = float(
        getattr(opt, "tsh_mbm_u2r_weight", 0.05)
    )
    model.teacher_lambda_eff = 0.0


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
_TSH = "tsh_instance_head."
_DEC_TAIL = "enc_dec_backbone.decoder_blocks."
_FROZEN = (
    "enc_dec_backbone.encoder_blocks.",
    "enc_dec_backbone.encoder_norm.",
    "patch_embed.",
    "patch_plucker_embed.",
    "activation_head.",
    "anchor_pos_encoder.",
    "prompt_matcher.",
    "semantic_lifting_head.",
    "semantic_projector.",
    "prompt_semantic_adapter.",
    "semantic_head.",
    "lseg_teacher.",
)

# Intentionally-NaN metric placeholders produced by the model when
# compute_quality_metrics=False (ssim/lpips are set to NaN).
_NAN_PLACEHOLDERS = {"ssim", "lpips"}


def _out_finite(result) -> bool:
    for key, value in result.items():
        if key in _NAN_PLACEHOLDERS:
            continue
        if torch.is_tensor(value) and value.is_floating_point():
            if not bool(torch.isfinite(value).all().item()):
                return False
    return True


def _backward_metrics(model, data, opt, batch_id, scene) -> dict:
    torch.cuda.empty_cache()
    out = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(data, compute_quality_metrics=False)
    loss_rgb = result["loss_rgb"]
    loss_inst_raw = result["loss_instance_group"]
    inst_weight = float(getattr(opt, "tsh_lambda_instance", 0.05)) * 1.0
    u2r_term = getattr(model, "_tsh_last_mbm_u2r_loss", None)

    rows = {}
    targets = {
        "gt_rgb": (loss_rgb, 1.0),
        "instance_raw": (loss_inst_raw, 1.0),
        "instance_weighted": (loss_inst_raw, inst_weight),
        "u2r_weighted": (
            u2r_term if u2r_term is not None else loss_rgb.new_zeros(()),
            1.0,
        ),
    }
    for label, (loss, scale) in targets.items():
        if label == "u2r_weighted" and u2r_term is None:
            continue
        model.zero_grad(set_to_none=True)
        (loss * scale).backward(retain_graph=True)
        rows[label] = {
            "unit_formation": _group_norm(model, _FORM),
            "gs_decoder": _group_norm(model, _GS_DEC),
            "absolute_head": _group_norm(model, "absolute_gs_head."),
            "tsh_head": _group_norm(model, _TSH),
            "decoder_tail": _group_norm(model, _DEC_TAIL),
            "frozen": _group_norm(model, _FROZEN),
            "total": _group_norm(
                model, tuple(p for p, _ in model.named_parameters())
            ),
        }
    if u2r_term is None:
        rows["u2r_weighted"] = {
            key: 0.0 for key in ("unit_formation", "gs_decoder",
                                 "absolute_head", "tsh_head",
                                 "decoder_tail", "frozen", "total")
        }
    # instance vs rgb gradient cosine at the unit-formation parameters.
    model.zero_grad(set_to_none=True)
    (loss_inst_raw * inst_weight).backward(retain_graph=True)
    inst_vec = torch.cat(
        [
            param.grad.detach().float().flatten()
            for name, param in model.named_parameters()
            if param.grad is not None
            and any(name.startswith(p) for p in _FORM)
        ]
    )
    model.zero_grad(set_to_none=True)
    loss_rgb.backward(retain_graph=True)
    rgb_vec = torch.cat(
        [
            param.grad.detach().float().flatten()
            for name, param in model.named_parameters()
            if param.grad is not None
            and any(name.startswith(p) for p in _FORM)
        ]
    )
    cosine_inst_unit = (
        float(
            torch.nn.functional.cosine_similarity(
                inst_vec.unsqueeze(0), rgb_vec.unsqueeze(0)
            ).item()
        )
        if inst_vec.numel() and rgb_vec.numel()
        else None
    )
    model.zero_grad(set_to_none=True)

    grad_rgb_unit = rows["gt_rgb"]["unit_formation"]
    grad_rgb_abs = rows["gt_rgb"]["absolute_head"]
    grad_inst_unit = rows["instance_weighted"]["unit_formation"]
    grad_u2r_unit = (
        rows["u2r_weighted"]["unit_formation"]
        if u2r_term is not None
        else 0.0
    )
    grad_u2r_abs = (
        rows["u2r_weighted"]["absolute_head"]
        if u2r_term is not None
        else 0.0
    )

    # Fresh forward for the real joint loss / total pre-clip norm / stats.
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(data, compute_quality_metrics=False)
    joint = result["loss"]
    joint.backward()
    pre_clip_total = _group_norm(
        model, tuple(p for p, _ in model.named_parameters())
    )
    clip_triggered = bool(pre_clip_total > float(getattr(opt, "gradient_clip", 1.0)))
    r_unit_inst = (
        grad_inst_unit / max(grad_rgb_unit, 1e-8)
        if grad_rgb_unit > 0
        else None
    )
    r_unit_u2r = (
        grad_u2r_unit / max(grad_rgb_unit, 1e-8)
        if grad_rgb_unit > 0 and u2r_term is not None
        else None
    )
    r_abs_u2r = (
        grad_u2r_abs / max(grad_rgb_abs, 1e-8)
        if grad_rgb_abs > 0 and u2r_term is not None
        else None
    )
    instance_ratio = inst_weight
    out.update(
        {
            "batch_id": batch_id,
            "scene": scene,
            "rows": rows,
            "grad_inst_unit": grad_inst_unit,
            "grad_gt_unit": grad_rgb_unit,
            "grad_u2r_unit": grad_u2r_unit,
            "r_unit_instance_raw": (
                rows["instance_raw"]["unit_formation"]
                / max(rows["gt_rgb"]["unit_formation"], 1e-8)
                if grad_rgb_unit > 0
                else None
            ),
            "r_unit_instance_weighted": (
                rows["instance_weighted"]["unit_formation"]
                / max(rows["gt_rgb"]["unit_formation"], 1e-8)
                if grad_rgb_unit > 0
                else None
            ),
            "r_unit_u2r": r_unit_u2r,
            "r_abs_u2r": r_abs_u2r,
            "cosine_inst_unit": cosine_inst_unit,
            "cosine_u2r_unit": None,
            "pre_clip_total_grad_norm": pre_clip_total,
            "clip_triggered": clip_triggered,
            "instance_loss": float(loss_inst_raw.item()),
            "void_share": float(
                result.get("void_share", torch.zeros(())).detach().item()
            )
            if torch.is_tensor(result.get("void_share"))
            else float(result.get("void_share", 0.0)),
            "max_group_prob_share": float(
                result.get("max_group_prob_share", torch.zeros(()))
                .detach()
                .item()
            )
            if torch.is_tensor(result.get("max_group_prob_share"))
            else float(result.get("max_group_prob_share", 0.0)),
            "psnr": float(result["psnr"].item()),
            "loss_mbm_u2r_raw_weighted": (
                float(u2r_term.item()) if u2r_term is not None else 0.0
            ),
            "mbm_u2r_raw": (
                float(u2r_term.item())
                if u2r_term is not None
                else 0.0
            ),
            "mbm_u2r_interior_x": float(
                result.get("mbm_u2r_interior_share_x", torch.zeros(()))
                .detach()
                .item()
            )
            if torch.is_tensor(result.get("mbm_u2r_interior_share_x"))
            else 0.0,
            "finite": _out_finite(result),
        }
    )
    return out


def _summarize(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace/tsh_mbm_audit")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-batches", type=int, default=16)
    parser.add_argument(
        "--config-name",
        default="semantic_v6_absolute_units_true_shared_siu3r_mbm_both_ddp8",
    )
    args = parser.parse_args()
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    opt = config_defaults[args.config_name]
    opt.workspace = str(out_dir)
    opt.num_workers = 0
    torch.manual_seed(args.seed)
    loader, _, _, _ = get_multi_dataloader(opt, _LocalAccelerator())

    model = model_registry[opt.model_type](opt).cuda().train()
    ckpt = load_file(args.resume, device="cpu")
    tsh_keys = _load_heads_strict(model, ckpt, opt)
    hashes_before = _head_hashes(model)
    mode = opt.tsh_mbm_mode
    _set_full_joint(model, opt)

    requires_grad = {
        name: param.requires_grad
        for name, param in model.named_parameters()
    }
    legacy_instance_branch = any(
        name.startswith("instance_branch.") for name in requires_grad
    )
    dec_tail_trainable = any(
        name.startswith("enc_dec_backbone.decoder_blocks.")
        and flag
        for name, flag in requires_grad.items()
    )
    activation_trainable = any(
        name.startswith("activation_head.") and flag
        for name, flag in requires_grad.items()
    )
    unit_form_count = sum(
        1
        for name in requires_grad
        if name.startswith(("absolute_gs_head.tok_norm.",
                            "absolute_gs_head.tok_proj.",
                            "absolute_gs_head.unit_queries",
                            "absolute_gs_head.unit_readout."))
    )
    report["static"] = {
        "mode": mode,
        "mbm_u2r_weight": opt.tsh_mbm_u2r_weight,
        "u2r_warmup_start": opt.tsh_mbm_u2r_warmup_start_step,
        "u2r_warmup_steps": opt.tsh_mbm_u2r_warmup_steps,
        "unit_gradient_multiplier_max": opt.tsh_unit_gradient_multiplier_max,
        "decoder_tail_lr": opt.tsh_mbm_decoder_tail_lr,
        "resume_ckpt": str(args.resume),
        "has_tsh_keys": tsh_keys > 0,
        "tsh_keys_loaded": tsh_keys,
        "abs_24": sum(
            1
            for name in hashes_before
            if name.startswith("absolute_gs_head.")
        ),
        "legacy_dual_unit_params": legacy_instance_branch,
        "dec_tail_trainable": dec_tail_trainable,
        "activation_head_trainable": activation_trainable,
        "unit_formation_params": unit_form_count,
        "frozen_semantic": all(
            not flag
            for name, flag in requires_grad.items()
            if name.startswith(("prompt_matcher.", "semantic_lifting_head.",
                                "semantic_projector.", "semantic_head.",
                                "lseg_teacher."))
        ),
    }

    batches = []
    loader_iter = iter(loader)
    for batch_id in range(args.n_batches):
        try:
            data = next(loader_iter)
        except StopIteration:
            break
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        scene = str(data.get("scene_name", ("?",))[0])
        metrics = _backward_metrics(
            model, data, opt, batch_id, scene
        )
        batches.append(metrics)
        print(
            f"[mbm-audit] batch={batch_id} scene={scene} "
            f"r_inst={metrics['r_unit_instance_weighted']} "
            f"r_u2r={metrics['r_unit_u2r']} "
            f"cos_inst={metrics['cosine_inst_unit']} "
            f"clip={metrics['clip_triggered']} "
            f"psnr={metrics['psnr']:.2f}",
            flush=True,
        )

    def _field(fn):
        vals = [fn(b) for b in batches if fn(b) is not None]
        return _summarize(vals) if vals else None

    report["n_batches"] = len(batches)
    report["summary"] = {
        "r_unit_instance_weighted": _field(
            lambda b: b["r_unit_instance_weighted"]
        ),
        "r_unit_instance_raw": _field(lambda b: b["r_unit_instance_raw"]),
        "r_unit_u2r": _field(lambda b: b["r_unit_u2r"]),
        "r_abs_u2r": _field(lambda b: b["r_abs_u2r"]),
        "cosine_instance_unit": _field(lambda b: b["cosine_inst_unit"]),
        "pre_clip_total_grad_norm": _field(
            lambda b: b["pre_clip_total_grad_norm"]
        ),
        "clip_triggered_count": sum(
            1 for b in batches if b["clip_triggered"]
        ),
        "instance_loss": _field(lambda b: b["instance_loss"]),
        "void_share": _field(lambda b: b["void_share"]),
        "max_group_prob_share": _field(
            lambda b: b["max_group_prob_share"]
        ),
        "psnr": _field(lambda b: b["psnr"]),
        "mbm_u2r_raw": _field(lambda b: b["mbm_u2r_raw"]),
        "mbm_u2r_interior_x": _field(lambda b: b["mbm_u2r_interior_x"]),
    }
    report["per_batch"] = batches
    report["head_hashes"] = hashes_before
    report["has_nan"] = any(not b["finite"] for b in batches)

    with open(out_dir / "audit_true_shared_mbm.json", "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(
        f"[mbm-audit] wrote {out_dir / 'audit_true_shared_mbm.json'} "
        f"({len(batches)} batches)"
    )


if __name__ == "__main__":
    _main()
