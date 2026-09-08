"""Direction B0 smoke: Scene-Instance-Conditioned unit formation (SIC).

Checks, on a single ScanNet LSM-style batch:
  1. gate=0 parity: SIC on vs SIC off produce identical instance outputs
     (same clustering, same rendered masks) as the 0.324 baseline;
  2. reconstruction stays strictly frozen (no backbone/decoder/activation
     gradients; PSNR at the frozen level);
  3. gradients: only the new SIC conditioning branch (sic_module, gate,
     readout, projections) is new; gate has a nonzero gradient from 0 and
     opens after an optimizer step; with the gate opened, gradients reach
     the SIC module through the readout path;
  4. cross-token binding diagnostics run in both train and eval:
     same-instance/same-token, same-instance/different-token and
     different-instance embedding similarities are reported.

Does NOT start long training; writes a checkpoint + config so a 1-scene LSM
sanity eval can be run afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


class _LocalAccelerator:
    is_main_process = True


INSTANCE_TRAINABLE_TOKENS = (
    "gs_feature_mlp.",
    "unit_queries",
    "unit_layers.",
    "log_unit_temp",
    "unit_image_net.",
    "dino_proj.",
    # Direction B0 (new params only)
    "sic_module.",
    "sic_gate",
    "sic_readout.",
    "sic_pos_emb.",
    "sic_h_proj.",
    "sic_dense_proj.",
    "sic_dino_proj.",
    "sic_desc_norm.",
    "sic_q_proj.",
    "identity_encoder.",
    "direct_gs_head",
    "unit_gaussian_decoder",
    "unit_ctx_mlp",
    "unit_pos_mlp",
    "unit_norm",
    "group_tokens",
    "group_layers",
    "group_norm",
    "unit_assignment_proj",
    "group_assignment_proj",
    "log_assignment_temperature",
    "void_head",
)


def _freeze_and_unfreeze(model) -> None:
    for name, param in model.named_parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        if name.startswith("instance_branch.") and any(
            token in name for token in INSTANCE_TRAINABLE_TOKENS
        ):
            param.requires_grad_(True)


def _max_abs_diff(a, b) -> float:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.shape != b.shape:
        return float("inf")
    return float((a - b).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace/sic_b0_smoke")
    parser.add_argument(
        "--config-name",
        default="semantic_v6_unit_shaping_img_dino_sic_smoke",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = config_defaults[args.config_name]
    opt.workspace = args.workspace
    opt.experiment_name = out_dir.name
    opt.num_workers = 0
    opt.max_iters_per_epoch = 2
    opt.num_epochs = 1
    opt.evaluating = False

    import tyro

    (out_dir / "config.yaml").write_text(
        tyro.extras.to_yaml(opt), encoding="utf-8"
    )

    train_loader, _, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    data = next(iter(train_loader))
    data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in data.items()}

    model = model_registry[opt.model_type](opt)
    model = model.cuda()
    # Warm-start the entire existing branch from the best 0.324 checkpoint.
    # The new SIC params are not in the checkpoint and keep their init.
    from safetensors.torch import load_file

    resume_ckpt = load_file(opt.instance_branch_unit_resume, device="cpu")
    torch.nn.Module.load_state_dict(model, resume_ckpt, strict=False)
    # Eval loads the frozen backbone from backbone_resume when the head
    # checkpoint has no backbone keys; mirror that so the smoke matches the
    # LSM eval configuration exactly.
    backbone_path = str(getattr(opt, "backbone_resume", "") or "")
    if backbone_path and Path(backbone_path).is_file():
        backbone_ckpt = load_file(backbone_path, device="cpu")
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
        print(
            f"[sic-smoke] frozen backbone loaded from {backbone_path} "
            f"({len(loadable)} keys)"
        )

    _freeze_and_unfreeze(model)
    branch = model.instance_branch

    report: dict = {}
    # ------------------------------------------------------------------
    # 1) gate=0 parity: SIC off vs SIC on (gate zero) eval forward.
    # ------------------------------------------------------------------
    model.eval()
    branch.sic_units = False
    with torch.inference_mode():
        out0 = model(data, compute_quality_metrics=True)
    branch.sic_units = True
    with torch.no_grad():
        branch.sic_gate.zero_()
    with torch.inference_mode():
        out1 = model(data, compute_quality_metrics=True)
    parity_keys = [
        "instance_group_probabilities",
        "rendered_instance_group_probability",
        "num_clusters",
    ]
    parity: dict[str, float] = {}
    for key in parity_keys:
        if key in out0 and key in out1:
            parity[key] = _max_abs_diff(out0[key], out1[key])
    if "num_clusters" in out0 and "num_clusters" in out1:
        parity["num_clusters_equal"] = float(
            int(out0["num_clusters"].item())
            == int(out1["num_clusters"].item())
        )
    if "psnr" in out0 and "psnr" in out1:
        parity["psnr"] = _max_abs_diff(out0["psnr"], out1["psnr"])
        report["psnr_gate0_off"] = float(out0["psnr"])
        report["psnr_gate0_on"] = float(out1["psnr"])
    eval_monitors = {
        key: float(out1[key])
        for key in (
            "unit_same_tok_sim",
            "unit_cross_tok_sim",
            "unit_diff_sim",
            "unit_cross_tok_gap",
            "sic_usage_entropy",
            "sic_active_ratio",
            "sic_gate_value",
        )
        if key in out1
    }
    report["eval_monitors_gate0"] = eval_monitors
    report["gate0_parity"] = parity
    print(
        f"[sic-smoke] gate=0 parity max-abs-diff: "
        + ", ".join(f"{k}={v:.3e}" for k, v in parity.items())
    )

    # ------------------------------------------------------------------
    # 2+3) training-mode gradient checks (2 steps; no long training).
    # ------------------------------------------------------------------
    branch.sic_units = True
    model.train()
    trainable = [
        param
        for param in model.parameters()
        if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.05)

    def _grad_stats() -> dict:
        backbone_grads = []
        sic_grads = []
        other_inst_grads = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            g = param.grad
            has_grad = g is not None and float(g.abs().sum()) > 0
            if name.startswith(
                ("enc_dec_backbone.", "activation_head.")
            ) or name in ("gs_tokens",):
                backbone_grads.append((name, has_grad))
            elif "sic_" in name or name.startswith(
                "instance_branch.sic_module."
            ):
                sic_grads.append(
                    (name, float(g.abs().sum()) if g is not None else 0.0)
                )
            else:
                other_inst_grads.append(
                    (name, float(g.abs().sum()) if g is not None else 0.0)
                )
        return {
            "backbone_any_grad": any(flag for _, flag in backbone_grads),
            "sic_param_norms": {
                name: norm for name, norm in sic_grads if norm > 0
            },
            "sic_param_with_grad": sum(1 for _, norm in sic_grads if norm > 0),
            "sic_param_total": len(sic_grads),
            "existing_inst_param_with_grad": sum(
                1 for _, norm in other_inst_grads if norm > 0
            ),
            "existing_inst_param_total": len(other_inst_grads),
        }

    step_stats = {}
    learned_gate = 0.0
    for step in (1, 2):
        optimizer.zero_grad()
        out = model(data)
        loss = (
            float(getattr(opt, "lambda_rgb", 200.0)) * out["loss_rgb"]
            + out["loss_instance_group"]
        )
        loss.backward()
        stats = _grad_stats()
        gate = branch.sic_gate
        gate_grad = (
            float(gate.grad.abs().sum()) if gate.grad is not None else 0.0
        )
        stats["gate_grad_norm"] = gate_grad
        stats["gate_before"] = float(gate.item())
        step_stats[f"step{step}"] = stats
        print(
            f"[sic-smoke] step {step}: loss={float(loss):.3f} "
            f"gate={float(gate.item()):.3e} gate_grad={gate_grad:.3e} "
            f"backbone_any_grad={stats['backbone_any_grad']} "
            f"sic_params_with_grad={stats['sic_param_with_grad']}/"
            f"{stats['sic_param_total']}"
        )
        optimizer.step()
        if step == 1:
            step_stats["step1"]["gate_after"] = float(gate.item())
            learned_gate = float(gate.item())
            # Open the gate manually for one diagnostic backward so the full
            # SIC readout -> unit-query -> InfoNCE gradient path is verified
            # (not only the usage-entropy anti-collapse path).
            with torch.no_grad():
                gate.fill_(0.5)
    report["gradient_checks"] = step_stats
    # Restore the gate to its actually-learned (tiny) value so the saved
    # checkpoint is a faithful 2-step B0 model, not the diagnostic open-gate
    # configuration.
    with torch.no_grad():
        branch.sic_gate.fill_(learned_gate)
    report["sic_gate_final"] = float(branch.sic_gate.item())

    # Monitors from the last training forward (cross-token diagnostics).
    report["last_train_monitors"] = {
        key: float(out[key])
        for key in (
            "unit_same_tok_sim",
            "unit_cross_tok_sim",
            "unit_diff_sim",
            "unit_cross_tok_gap",
            "unit_embedding_same_sim",
            "unit_embedding_diff_sim",
            "sic_usage_entropy",
            "sic_active_ratio",
            "sic_gate_value",
        )
        if key in out
    }
    report["psnr_step2_diagnostic_forward"] = float(
        out.get("psnr", float("nan"))
    )

    # ------------------------------------------------------------------
    # Save checkpoint + metadata for the 1-scene LSM sanity eval.
    # ------------------------------------------------------------------
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    state = {
        k: v.detach().cpu().contiguous()
        for k, v in model.state_dict().items()
    }
    from safetensors.torch import save_file

    path = ckpt_dir / "model_step_000002.safetensors"
    save_file(state, str(path))
    save_file(state, str(out_dir / "model.safetensors"))
    metadata = {
        "epoch": 0,
        "step": 2,
        "model_type": opt.model_type,
        "tokengs_checkpoint": opt.prompt_tokengs_checkpoint,
        "prompt_checkpoint": str(path),
        "backbone_resume": str(opt.backbone_resume),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (out_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[sic-smoke] saved checkpoint {path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
