"""Step-0 identity and gradient-boundary audit for per-GS refinement.

Compares a model with the zero-initialized PerGSSlotRefineHead (gate=0)
against the identical non-refined True-Shared model on the same real batch,
both loaded from the calibrated Both@1420 checkpoint (abs 24/24 + tsh 50/50,
no fresh reset), then opens the gate and audits:

  * RGB / unit logits / per-GS logits / rendered masks identical at gate=0;
  * gate-open: within-unit GS logit differences appear;
  * instance BCE/Dice gradients reach the new refine head and q_abs unit
    formation but never center_mlp / gs_decoder / reconstruction slot_emb;
  * U->R still updates geometry (and not the refine head);
  * RGB path unchanged, no NaN.
"""

from __future__ import annotations

import argparse
import json
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


_RESUME = (
    "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_"
    "t3e6_ddp8/checkpoints/model_step_001420.safetensors"
)
_REFINE_CFG = (
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_w10_t3e6_pgsr_ddp8"
)
_BASE_CFG = (
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
)


def _load_heads(model, ckpt):
    abs_state = {
        key.split(".", 1)[1]: value
        for key, value in ckpt.items()
        if key.startswith("absolute_gs_head.")
    }
    res = model.absolute_gs_head.load_state_dict(abs_state, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    tsh_state = {
        key.split(".", 1)[1]: value
        for key, value in ckpt.items()
        if key.startswith("tsh_instance_head.")
    }
    res = model.tsh_instance_head.load_state_dict(tsh_state, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    return len(abs_state), len(tsh_state)


def _grad_norm(params, prefixes):
    total = 0.0
    for name, param in params:
        if param.grad is None:
            continue
        if isinstance(prefixes, str):
            hit = name.startswith(prefixes)
        else:
            hit = any(name.startswith(p) for p in prefixes)
        if hit:
            total += float(param.grad.double().norm().item() ** 2)
    return float(total ** 0.5)


def _set_full_joint(model, opt):
    model.tsh_instance_loss_weight_eff = 1.0
    model.tsh_unit_grad_eff = float(
        getattr(opt, "tsh_unit_gradient_multiplier_max", 0.0)
    )
    model.tsh_mbm_u2r_eff = float(
        getattr(opt, "tsh_mbm_u2r_weight", 10.0)
    )
    model.teacher_lambda_eff = 0.0


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace/tsh_pgsr_audit")
    parser.add_argument("--resume", default=_RESUME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-index", type=int, default=0)
    args = parser.parse_args()
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    torch.manual_seed(args.seed)
    opt = config_defaults[_REFINE_CFG]
    opt.workspace = str(out_dir)
    opt.num_workers = 0
    loader, _, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    iterator = iter(loader)
    for _ in range(args.batch_index + 1):
        data = next(iterator)
    data = {
        k: (v.cuda() if torch.is_tensor(v) else v)
        for k, v in data.items()
    }
    scene = str(data.get("scene_name", ("?",))[0])

    ckpt = load_file(args.resume, device="cpu")

    def _build(cfg_name):
        o = config_defaults[cfg_name]
        o.workspace = str(out_dir)
        o.num_workers = 0
        m = model_registry[o.model_type](o).cuda().train()
        abs_keys, tsh_keys = _load_heads(m, ckpt)
        _set_full_joint(m, o)
        return m, o, abs_keys, tsh_keys

    base, base_opt, abs_keys, tsh_keys = _build(_BASE_CFG)
    refine_model, refine_opt, _, _ = _build(_REFINE_CFG)
    refine_model.tsh_per_gs_gate_eff = 0.0
    report["loaded"] = {
        "abs_keys": abs_keys,
        "tsh_keys": tsh_keys,
        "refine_head_exists": refine_model.tsh_slot_refine_head is not None,
        "scene": scene,
    }

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_base = base(data, compute_quality_metrics=False)
        out0 = refine_model(data, compute_quality_metrics=False)

    def _close(a, b, atol):
        return bool(torch.allclose(a.float(), b.float(), atol=atol))

    identity = {
        "rgb": _close(
            out_base["images_pred"], out0["images_pred"], 1e-5
        ),
        "pi_unit": _close(
            out_base["pi_unit"], out0["pi_unit"], 1e-5
        ),
        "pi_gs": _close(
            out_base["pi_gs_refined"],
            out0["pi_gs_refined"],
            1e-5,
        ),
        "rendered_masks": _close(
            out_base["rendered_instance_group_probability"],
            out0["rendered_instance_group_probability"],
            1e-5,
        ),
        "loss": _close(out_base["loss"], out0["loss"], 1e-4),
        "max_abs_pi_gs_diff": float(
            (
                out_base["pi_gs_refined"].float()
                - out0["pi_gs_refined"].float()
            ).abs().max().item()
        ),
    }
    report["gate0_identity"] = identity
    assert identity["rgb"] and identity["pi_unit"] and identity["pi_gs"]
    assert identity["rendered_masks"] and identity["loss"]

    # Open the gate on a fresh forward.
    refine_model.zero_grad(set_to_none=True)
    with torch.no_grad():
        refine_model.tsh_slot_refine_head.alpha_param.fill_(1.0)
        refine_model.tsh_slot_refine_head.logit_proj.weight.uniform_(
            -0.05, 0.05
        )
        refine_model.tsh_slot_refine_head.logit_proj.bias.uniform_(
            -0.05, 0.05
        )
    refine_model.tsh_per_gs_gate_eff = 1.0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out1 = refine_model(data, compute_quality_metrics=False)
    block = out1["pi_gs_refined"].reshape(
        out1["pi_gs_refined"].shape[0],
        int(out1["pi_unit"].shape[1]),
        int(out1["pi_unit"].shape[2]),
        8,
        out1["pi_gs_refined"].shape[-1],
    )
    within_diff = (block - block[..., :1, :]).abs()
    report["gate_open"] = {
        "within_unit_logit_diff_mean": float(
            out1["tsh_per_gs_unit_logit_diff_mean"]
        ),
        "within_unit_logit_diff_max": float(
            out1["tsh_per_gs_unit_logit_diff_max"]
        ),
        "within_unit_prob_diff_mean": float(
            out1["tsh_per_gs_unit_prob_diff_mean"]
        ),
        "alpha_eff": float(out1["tsh_per_gs_alpha"]),
        "nonuniform_unit_count": int(
            (within_diff.amax(dim=(-1, -2)) > 1e-5).sum().item()
        ),
        "rgb_unchanged_vs_gate0": _close(
            out0["images_pred"], out1["images_pred"], 1e-5
        ),
    }
    assert report["gate_open"]["nonuniform_unit_count"] > 0
    assert report["gate_open"]["rgb_unchanged_vs_gate0"]

    groups = {
        "abs_unit": (
            "absolute_gs_head.tok_norm.",
            "absolute_gs_head.tok_proj.",
            "absolute_gs_head.unit_queries",
            "absolute_gs_head.unit_readout.",
        ),
        "abs_gs_decoder": (
            "absolute_gs_head.center_mlp.",
            "absolute_gs_head.gs_decoder.",
            "absolute_gs_head.slot_emb",
        ),
        "refine": "tsh_slot_refine_head.",
        "tsh": "tsh_instance_head.",
        "tail": "enc_dec_backbone.decoder_blocks.",
    }
    params = list(refine_model.named_parameters())
    inst_weight = float(
        getattr(refine_opt, "tsh_lambda_instance", 0.05)
    ) * 1.0
    grads = {}
    scenarios = {
        "rgb": out1["loss_rgb"],
        "instance": out1["loss_instance_group"] * inst_weight,
        "u2r": getattr(
            refine_model, "_tsh_last_mbm_u2r_loss", None
        ),
    }
    for label, loss in scenarios.items():
        if loss is None:
            continue
        refine_model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        grads[label] = {
            key: _grad_norm(params, prefixes)
            for key, prefixes in groups.items()
        }
        if label == "rgb":
            grads["rgb_refine_param_names"] = [
                name
                for name, param in params
                if name.startswith("tsh_slot_refine_head.")
                and param.grad is not None
                and float(param.grad.abs().sum()) > 0
            ]
        if label == "u2r":
            grads["u2r_refine_param_names"] = [
                name
                for name, param in params
                if name.startswith("tsh_slot_refine_head.")
                and param.grad is not None
                and float(param.grad.abs().sum()) > 0
            ]
    report["grads"] = grads
    assert grads["instance"]["refine"] > 0
    assert grads["instance"]["abs_unit"] > 0
    assert grads["instance"]["abs_gs_decoder"] == 0.0
    assert grads["instance"]["tsh"] > 0
    assert grads["rgb"]["refine"] == 0.0
    if grads.get("u2r") is not None:
        assert grads["u2r"]["abs_gs_decoder"] > 0
        assert grads["u2r"]["refine"] == 0.0
    finite = {
        k: bool(torch.isfinite(v.float()).all())
        for k, v in {
            "loss": out1["loss"],
            "rgb": out1["images_pred"],
            "pi_gs": out1["pi_gs_refined"],
            "mask": out1["rendered_instance_group_probability"],
            "depth": out1["depths_pred"],
        }.items()
    }
    report["finite"] = finite
    assert all(finite.values())

    with open(out_dir / "audit_pgsr_step0.json", "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"[pgsr-audit] wrote {out_dir / 'audit_pgsr_step0.json'}")


if __name__ == "__main__":
    _main()
