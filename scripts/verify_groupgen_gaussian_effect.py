"""Verify Experiment B's core property (offline, no official eval).

The claim that distinguishes B from the old post-hoc mask heads:
GroupToken enters the decoder BEFORE Gaussian generation and can change
Gaussian parameters -- it is not merely used for mask assignment afterward.

Checks:
1. Gradient path: instance loss -> final Gaussians -> group tokens /
   geometry head / token rewrite / unfrozen decoder must all have nonzero
   gradients (the path exists and is exercised).
2. On/off effect: after a few training steps, forward with the generator's
   scales forced to 0 vs active; report mean |dxyz| and |dopacity| on the
   final Gaussians. Both must be > 0, i.e. the group tokens changed the
   Gaussian parameters (xyz and opacity).
3. Mask path sanity: the rendered instance probability map is still produced
   for supervision.

Usage:
    python scripts/verify_groupgen_gaussian_effect.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

torch._dynamo.config.disable = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.input_types import split_data
from tokengs.options import config_defaults
from safetensors.torch import load_file


class _LocalAccelerator:
    is_main_process = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="semantic_v6_groupgen_smoke")
    parser.add_argument(
        "--backbone-resume",
        default=(
            "/space0/mawb/tokengs/workspace/"
            "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
            "checkpoints/model_step_008000.safetensors"
        ),
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--decoder-lr", type=float, default=1e-5)
    parser.add_argument("--workspace", default="workspace/verify_groupgen_effect")
    args = parser.parse_args()

    opt = config_defaults[args.config]
    # Skip the lambda warm-up so the instance loss is nonzero from step 0.
    opt.instance_group_lambda_warmup_steps = 0
    opt.mixed_precision = "no"
    Path(args.workspace).mkdir(parents=True, exist_ok=True)

    train_loader, _, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    batch = next(iter(train_loader))
    batch = {
        k: (v.cuda() if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    model_input, _ = split_data(batch, opt)

    model = model_registry[opt.model_type](opt)
    backbone_ckpt = load_file(args.backbone_resume, device="cpu")
    frozen_prefixes = (
        "enc_dec_backbone.",
        "patch_embed.",
        "patch_plucker_embed.",
        "activation_head.",
        "anchor_pos_encoder.",
    )
    native_state = torch.nn.Module.state_dict(model)
    loadable = {
        key: value
        for key, value in backbone_ckpt.items()
        if (key.startswith(frozen_prefixes) or key == "gs_tokens")
        and key in native_state
        and native_state[key].shape == value.shape
    }
    torch.nn.Module.load_state_dict(model, loadable, strict=False)
    print(f"[verify-groupgen] loaded {len(loadable)} backbone keys")
    model.train()
    model = model.cuda()
    gen = model.instance_group_head

    # --- 1. gradient-path check on the very first forward ---
    out = model(batch, compute_quality_metrics=False)
    assert "loss_instance_group" in out, list(out)
    loss = out["loss_instance_group"]
    assert loss.requires_grad and float(loss) > 0, float(loss)
    loss.backward()
    grad_report = {
        "group_tokens": float(gen.group_tokens.grad.norm()),
        "geometry_head_w": float(gen.geometry_head[-1].weight.grad.norm()),
        "token_residual_w": float(gen.token_residual[-1].weight.grad.norm()),
        "decoder_blocks": float(
            sum(
                p.grad.norm().item()
                for p in model.enc_dec_backbone.decoder_blocks.parameters()
                if p.grad is not None
            )
        ),
    }
    print("[verify-groupgen] grad norms:", grad_report)
    for key, value in grad_report.items():
        assert value > 0, f"no gradient to {key}: {value}"

    # --- 2. short training so the group effect heads leave zero-init ---
    head_params = list(gen.parameters())
    decoder_params = [
        p
        for p in model.enc_dec_backbone.decoder_blocks.parameters()
        if p.requires_grad
    ]
    optim = torch.optim.Adam(
        [
            {"params": head_params, "lr": args.head_lr},
            {"params": decoder_params, "lr": args.decoder_lr},
        ]
    )
    for step in range(args.steps):
        optim.zero_grad()
        out = model(batch, compute_quality_metrics=False)
        loss = out["loss_instance_group"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        torch.nn.utils.clip_grad_norm_(decoder_params, 1.0)
        optim.step()
        if step % 5 == 0:
            print(
                f"[verify-groupgen] step {step} "
                f"instance={float(loss):.4f} "
                f"geo_scale={gen.log_geometry_scale.exp().item():.4f} "
                f"opac_scale={gen.log_opacity_scale.exp().item():.4f}"
            )

    # --- 3. on/off effect on the final Gaussians ---
    with torch.no_grad():
        recon_on, _, _ = model._forward_prompt_reconstruction(model_input)
        gauss_on = recon_on.gaussians.clone()
        saved = {
            name: param.detach().clone()
            for name, param in gen.named_parameters()
            if "scale" in name
        }
        for name in saved:
            param = dict(gen.named_parameters())[name]
            param.data.fill_(-20.0)  # exp -> ~0, disable all group effects
        recon_off, _, _ = model._forward_prompt_reconstruction(model_input)
        gauss_off = recon_off.gaussians.clone()
        for name, value in saved.items():
            dict(gen.named_parameters())[name].data.copy_(value)

    dxyz = (gauss_on[..., :3] - gauss_off[..., :3]).abs().mean().item()
    dop = (gauss_on[..., 3:4] - gauss_off[..., 3:4]).abs().mean().item()
    print(f"[verify-groupgen] dxyz={dxyz:.6f} dopacity={dop:.6f}")
    assert dxyz > 1e-6, "GroupToken did NOT change Gaussian positions"
    assert dop > 1e-6, "GroupToken did NOT change Gaussian opacity"

    # --- 4. mask path sanity ---
    prob = out["rendered_instance_group_probability"]
    payload = {
        "config": args.config,
        "steps": args.steps,
        "grad_report": grad_report,
        "dxyz": dxyz,
        "dopacity": dop,
        "mask_prob_shape": list(prob.shape),
        "gaussians_shape": list(gauss_on.shape),
        "passed": True,
    }
    out_path = Path(args.workspace) / "verify_groupgen_effect.json"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[verify-groupgen] PASSED -> {out_path}")


if __name__ == "__main__":
    main()
