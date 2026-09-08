"""Verify Experiment C's core property (offline, no official eval).

The independent instance branch must be FULLY detached from the frozen
TokenGS reconstruction branch:
1. After the instance-loss backward, gradients exist ONLY on
   ``instance_branch.*`` parameters; every backbone parameter
   (``enc_dec_backbone.*``, ``patch_*``, ``activation_head.*``, ``gs_tokens``)
   has no gradient.
2. The RGB reconstruction is byte-identical before/after branch training
   (the branch never modifies it; eval PSNR stays at the frozen level).
3. The branch generates its OWN Gaussians (different from the frozen ones
   after a few training steps) and renders its own instance masks.
4. Rendered instance probability maps are produced (mask path alive).

Usage:
    python scripts/verify_independent_instance_branch.py
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
    parser.add_argument("--config", default="semantic_v6_instance_branch_smoke")
    parser.add_argument(
        "--backbone-resume",
        default=(
            "/space0/mawb/tokengs/workspace/"
            "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
            "checkpoints/model_step_008000.safetensors"
        ),
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--workspace", default="workspace/verify_independent_branch"
    )
    args = parser.parse_args()

    opt = config_defaults[args.config]
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
    assert model.instance_group_head is None
    branch = model.instance_branch
    assert branch is not None
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
    print(f"[verify-ibranch] loaded {len(loadable)} backbone keys")
    model.train()
    model = model.cuda()

    # --- 1. gradient isolation on the first forward ---
    out = model(batch, compute_quality_metrics=False)
    loss = out["loss_instance_group"]
    assert loss.requires_grad and float(loss) > 0, float(loss)
    loss.backward()
    backbone_has_grad = [
        name
        for name, param in model.named_parameters()
        if not name.startswith("instance_branch.")
        and param.grad is not None
        and param.grad.abs().sum() > 0
    ]
    branch_grad_norm = sum(
        p.grad.norm().item()
        for p in branch.parameters()
        if p.grad is not None
    )
    print("[verify-ibranch] backbone grads:", backbone_has_grad[:10])
    print("[verify-ibranch] branch grad norm:", branch_grad_norm)
    assert not backbone_has_grad, (
        "instance loss reached the reconstruction branch: "
        + str(backbone_has_grad[:10])
    )
    assert branch_grad_norm > 0

    # --- 2. RGB reconstruction byte-identity ---
    with torch.no_grad():
        recon_ref, hidden_ref, _ = model._forward_prompt_reconstruction(
            model_input
        )
        rgb_ref = recon_ref.gaussians.detach().clone()
        frozen_ref = recon_ref.gaussians.detach().clone()
        # Warm-start check: at init the branch reproduces the frozen geometry.
        init_branch = branch.last_instance_gaussians
        init_diff = (init_branch - frozen_ref).abs().max().item()
        print(f"[verify-ibranch] init branch vs frozen max diff: {init_diff:.2e}")
        assert init_diff < 1e-4, (
            "branch did not warm-start from the frozen geometry"
        )

    # --- 3. short training of the branch only ---
    optim = torch.optim.Adam(
        [p for p in branch.parameters() if p.requires_grad], lr=args.lr
    )
    for step in range(args.steps):
        optim.zero_grad()
        out = model(batch, compute_quality_metrics=False)
        loss = out["loss_instance_group"]
        rgb_loss = float(out.get("loss_instance_branch_rgb", -1.0))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0)
        optim.step()
        if step % 3 == 0:
            print(
                f"[verify-ibranch] step {step} instance={float(loss):.4f} "
                f"branch_rgb={rgb_loss:.4f}"
            )

    with torch.no_grad():
        recon_after, _, _ = model._forward_prompt_reconstruction(model_input)
        rgb_after = recon_after.gaussians.detach().clone()
        branch_gaussians = branch.last_instance_gaussians
        frozen_gaussians = recon_ref.gaussians.detach()
    rgb_diff = (rgb_ref - rgb_after).abs().max().item()
    # Branch has its own (smaller) Gaussian set; compare per-anchor means.
    branch_anchor_means = (
        branch_gaussians[..., :3]
        .view(branch_gaussians.shape[0], -1, branch.num_gaussians_per_anchor, 3)
        .mean(dim=2)
    )
    frozen_anchor_means = frozen_gaussians[..., :3].view(
        frozen_gaussians.shape[0], -1, 64, 3
    ).mean(dim=2)
    branch_vs_frozen = (
        branch_anchor_means - frozen_anchor_means
    ).abs().mean().item()
    print(f"[verify-ibranch] rgb_max_diff={rgb_diff:.2e}")
    print(f"[verify-ibranch] branch_vs_frozen_dxyz={branch_vs_frozen:.6f}")
    assert rgb_diff == 0.0, "RGB reconstruction changed; branch is NOT independent"
    assert branch_vs_frozen > 0.0, "branch did not produce its own geometry"

    prob = out["rendered_instance_group_probability"]
    payload = {
        "config": args.config,
        "backbone_grads_nonzero": backbone_has_grad,
        "branch_grad_norm": branch_grad_norm,
        "rgb_max_diff": rgb_diff,
        "init_frozen_diff": init_diff,
        "branch_vs_frozen_dxyz": branch_vs_frozen,
        "mask_prob_shape": list(prob.shape),
        "passed": True,
    }
    out_path = Path(args.workspace) / "verify_independent_branch.json"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[verify-ibranch] PASSED -> {out_path}")


if __name__ == "__main__":
    main()
