"""Smoke for generative-unit teacher bootstrap (v3 token-aligned structure).

Checks, on one ScanNet LSM-style batch:
  1. model builds from PURE tokengs_re10k (strict load, encoder + token
     transformer kept; old activation head frozen as teacher);
  2. Token -> 8 local units -> 8 GS/unit decode the FINAL Gaussians
     (zero-init residual => step-0 student GS == frozen teacher GS);
  3. teacher distill (GS-param + rendered RGB) is active with the expected
     decay weight; joint loss = RGB + unit-level query mask loss + teacher;
  4. backward works: gradients reach the unit Gaussian decoder / unit
     formation / group tokens; encoder + decoder + old GS head stay zero;
  5. PSNR stays at the frozen-teacher level at step 0.

No formal training is started.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        default="semantic_v6_generative_units_teacher_train",
    )
    parser.add_argument("--workspace", default="workspace/gen_units_teacher_smoke")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = config_defaults[args.config_name]
    opt.workspace = args.workspace
    opt.experiment_name = out_dir.name
    opt.num_workers = 0
    opt.evaluating = False

    _LocalAccelerator()
    train_loader, _, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    data = next(iter(train_loader))
    data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in data.items()}

    model = model_registry[opt.model_type](opt)
    model = model.cuda()
    model.train()

    report: dict = {}
    report["config"] = args.config_name
    report["gen_teacher_distill"] = bool(opt.gen_teacher_distill)
    report["gen_teacher_decay_steps"] = int(opt.gen_teacher_decay_steps)

    teacher_eff = model.compute_teacher_lambda_eff(1, opt)
    model.teacher_lambda_eff = teacher_eff
    report["teacher_lambda_eff_step1"] = teacher_eff

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(data)
        loss = out["loss"]
        loss.backward()

    report["loss_total"] = float(loss.detach().float())
    report["loss_rgb"] = float(out["loss_rgb"].detach().float())
    report["loss_teacher_gs"] = float(out["loss_teacher_gs"].detach().float())
    report["loss_teacher_rgb"] = float(out["loss_teacher_rgb"].detach().float())
    report["psnr"] = float(out["psnr"].detach().float())
    inst_loss = out.get("loss_instance_group")
    report["loss_instance_group"] = (
        float(inst_loss.detach().float()) if inst_loss is not None else None
    )

    grad_stats = {"backbone_any_grad": False, "gen_decoder_grad": 0.0,
                  "unit_formation_grad": 0.0, "group_token_grad": 0.0}
    for name, param in model.named_parameters():
        g = param.grad
        has_grad = g is not None and float(g.abs().sum()) > 0
        if name.startswith(("enc_dec_backbone.", "activation_head.",
                            "patch_embed.", "patch_plucker_embed.")) or name == "gs_tokens":
            if has_grad:
                grad_stats["backbone_any_grad"] = True
        elif name.startswith("instance_branch.unit_gaussian_decoder."):
            grad_stats["gen_decoder_grad"] += (
                float(g.abs().sum()) if g is not None else 0.0
            )
        elif name.startswith("instance_branch.gs_feature_mlp.") or name.startswith(
            "instance_branch.unit_layers."
        ) or name.startswith("instance_branch.unit_queries"):
            grad_stats["unit_formation_grad"] += (
                float(g.abs().sum()) if g is not None else 0.0
            )
        elif "group_tokens" in name or "group_layers" in name or "void_head" in name:
            grad_stats["group_token_grad"] += (
                float(g.abs().sum()) if g is not None else 0.0
            )
    report["grad_stats"] = grad_stats

    # Step-0 identity: the zero-initialized unit decoder means the final
    # Gaussians should still equal the frozen teacher GS (small residual).
    new_gs = out["gaussians"].detach().float()
    teacher_gs = model._teacher_gaussians.detach().float()
    if teacher_gs is not None and teacher_gs.shape == new_gs.shape:
        report["student_teacher_gs_maxdiff"] = float(
            (new_gs - teacher_gs).abs().max()
        )

    (out_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
