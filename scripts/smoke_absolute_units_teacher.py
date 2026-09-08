"""Smoke for the absolute token-aligned student (no residual / no old-GS
runtime dependence).

Checks on one ScanNet LSM-style batch:
  1. student GS shape [B, 65536, 14], values finite;
  2. teacher-on bootstrap step: teacher (old activation head) called under
     no_grad, teacher GS/RGB distill losses finite, student decoder gets
     gradients, teacher/backbone get none;
  3. teacher-off step: old GS head is NOT called (call counter stays 0),
     forward/backward works purely Token->Unit->GS, student decoder grads
     are nonzero;
  4. stage schedule helpers return the expected ramp (bootstrap 1 -> decay
     -> 0 -> instance warm-up).

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
        default="semantic_v6_absolute_units_teacher_train",
    )
    parser.add_argument(
        "--workspace", default="workspace/abs_units_teacher_smoke"
    )
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
    report["abs_mode"] = bool(opt.instance_branch_abs_units)

    # Stage schedule helpers.
    report["schedule"] = {
        "teacher_eff@0": model.compute_teacher_lambda_eff(0, opt),
        "teacher_eff@599": model.compute_teacher_lambda_eff(599, opt),
        "teacher_eff@800": model.compute_teacher_lambda_eff(800, opt),
        "teacher_eff@999": model.compute_teacher_lambda_eff(999, opt),
        "teacher_eff@1000": model.compute_teacher_lambda_eff(1000, opt),
        "inst_eff@1000": model.compute_instance_stage_eff(1000, opt),
        "inst_eff@1800": model.compute_instance_stage_eff(1800, opt),
        "inst_eff@2600": model.compute_instance_stage_eff(2600, opt),
    }

    def _grads() -> dict:
        stats = {"backbone_or_teacher_grad": 0.0, "abs_decoder_grad": 0.0}
        for name, param in model.named_parameters():
            g = param.grad
            norm = float(g.abs().sum()) if g is not None else 0.0
            if name.startswith(
                ("enc_dec_backbone.", "activation_head.", "patch_embed.",
                 "patch_plucker_embed.")
            ) or name == "gs_tokens":
                stats["backbone_or_teacher_grad"] += norm
            elif name.startswith("absolute_gs_head."):
                stats["abs_decoder_grad"] += norm
        return stats

    # 1) teacher-on bootstrap step (step 599 -> eff 1.0).
    model.teacher_lambda_eff = 1.0
    model.instance_stage_eff = 0.0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(data)
        loss = out["loss"]
        loss.backward()
    gs = out["gaussians"].detach().float()
    report["teacher_on"] = {
        "teacher_called": bool(model.teacher_called),
        "gs_shape": list(gs.shape),
        "gs_finite": bool(torch.isfinite(gs).all()),
        "loss_total": float(loss.detach().float()),
        "loss_rgb": float(out["loss_rgb"].detach().float()),
        "loss_teacher_gs": float(out["loss_teacher_gs"].detach().float()),
        "loss_teacher_rgb": float(out["loss_teacher_rgb"].detach().float()),
        "grads": _grads(),
    }
    model.zero_grad(set_to_none=True)

    # 2) teacher-off step (>=1000): old GS head must not be called.
    calls = {"n": 0}
    orig_forward = model.activation_head.forward

    def _counting_forward(*a, **k):
        calls["n"] += 1
        return orig_forward(*a, **k)

    model.activation_head.forward = _counting_forward
    model.teacher_lambda_eff = 0.0
    model.instance_stage_eff = 1.0
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out2 = model(data)
            loss2 = out2["loss"]
            loss2.backward()
    finally:
        model.activation_head.forward = orig_forward
    gs2 = out2["gaussians"].detach().float()
    report["teacher_off"] = {
        "teacher_called": bool(model.teacher_called),
        "old_head_calls": calls["n"],
        "gs_shape": list(gs2.shape),
        "gs_finite": bool(torch.isfinite(gs2).all()),
        "loss_total": float(loss2.detach().float()),
        "loss_rgb": float(out2["loss_rgb"].detach().float()),
        "loss_teacher_gs": float(out2["loss_teacher_gs"].detach().float()),
        "grads": _grads(),
    }

    (out_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
