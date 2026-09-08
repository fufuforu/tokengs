"""Deterministic re-evaluation + GS statistics for overfit checkpoints.

Loads a checkpoint with the same fixed single sample as
``train_abs_single_overfit.py`` and, per checkpoint:
  * repeats the student-only eval N times (identical PSNR expected);
  * reports the train-mode loss decomposition and pre-clip abs-gradient norm;
  * reports Gaussian position/scale/opacity min/max/mean and outlier ratios
    for both student and teacher GS.
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
from tokengs.models.input_types import split_data  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


class _Accelerator:
    is_main_process = True

    def print(self, *args, **kwargs):
        print(*args, **kwargs)


def _gs_stats(gs: torch.Tensor) -> dict:
    pos = gs[..., :3]
    opacity = gs[..., 3]
    scale = gs[..., 4:7]
    teacher_ref = None
    return {
        "pos_min": float(pos.min()),
        "pos_max": float(pos.max()),
        "pos_mean": float(pos.mean()),
        "opacity_min": float(opacity.min()),
        "opacity_max": float(opacity.max()),
        "opacity_mean": float(opacity.mean()),
        "scale_min": float(scale.min()),
        "scale_max": float(scale.max()),
        "scale_mean": float(scale.mean()),
        "pos_abs_gt_10_ratio": float((pos.abs() > 10.0).float().mean()),
        "opacity_gt_0.99_ratio": float((opacity > 0.99).float().mean()),
        "scale_out_1e-5_1e0_ratio": float(
            ((scale < 1e-5) | (scale > 1.0)).float().mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--workspace", default="workspace/audit_overfit_ckpts")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = config_defaults["semantic_v6_absolute_units_recon_continue"]
    opt.num_workers = 0
    opt.prompt_overfit_single_batch = True
    opt.prompt_overfit_sample_index = int(args.sample_index)
    opt.evaluating = False

    _Accelerator()
    train_loader, _, train_dataset, _ = get_multi_dataloader(
        opt, _Accelerator()
    )

    def _fetch():
        train_dataset.set_rng_epoch(0)
        data = next(iter(train_loader))
        return {
            key: (value.cuda() if torch.is_tensor(value) else value)
            for key, value in data.items()
        }

    report = {}
    from safetensors.torch import load_file

    for ckpt in args.checkpoints:
        model = model_registry[opt.model_type](opt)
        model = model.cuda()
        state = load_file(ckpt, device="cpu")
        torch.nn.Module.load_state_dict(model, state, strict=False)
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith("absolute_gs_head."))
        model.teacher_lambda_eff = 1.0
        model.instance_stage_eff = 0.0
        model._quality_metrics.device = "cuda"

        data = _fetch()
        # 1) repeated deterministic student evals
        psnrs = []
        for _ in range(args.repeats):
            data = _fetch()
            model.eval()
            old_calls = 0
            orig = model.activation_head.forward

            def _cnt(*a, **k):
                nonlocal old_calls
                old_calls += 1
                return orig(*a, **k)

            model.activation_head.forward = _cnt
            with torch.inference_mode():
                out = model(data, compute_quality_metrics=True)
            calls = old_calls
            model.activation_head.forward = orig
            model.train()
            psnrs.append(float(out["psnr"].detach()))

        # 2) train-mode loss decomposition + grad norm (one backward)
        data = _fetch()
        model.train()
        opt_params = [
            p for p in model.parameters() if p.requires_grad
        ]
        for p in opt_params:
            p.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(data)
            loss = out["loss"]
            loss.backward()
        loss_train_total = float(loss.detach())
        loss_train_rgb = float(out["loss_rgb"].detach())
        loss_train_gs = float(out["loss_teacher_gs"].detach())
        loss_train_rgb_t = float(out["loss_teacher_rgb"].detach())
        abs_grad = 0.0
        sem_grad = 0.0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            norm = float(param.grad.abs().sum())
            if name.startswith("absolute_gs_head."):
                abs_grad += norm
            elif name.startswith(
                (
                    "prompt_matcher.",
                    "semantic_",
                    "prompt_semantic_adapter.",
                    "instance_branch.",
                )
            ):
                sem_grad += norm

        # 3) teacher GS + student GS statistics
        data = _fetch()
        model.eval()
        with torch.inference_mode():
            out = model(data, compute_quality_metrics=True)
            model_input, _ = split_data(data, opt)
            with torch.no_grad():
                teacher_recon, _, _ = model._forward_prompt_reconstruction(
                    model_input
                )
        student_gs = out["gaussians"].detach().float()
        teacher_gs = teacher_recon.gaussians.detach().float()

        report[str(Path(ckpt).stem)] = {
            "repeated_student_psnr": psnrs,
            "deterministic": len(set(psnrs)) == 1,
            "loss_total_train": loss_train_total,
            "loss_rgb_train": loss_train_rgb,
            "loss_teacher_gs_train": loss_train_gs,
            "loss_teacher_rgb_train": loss_train_rgb_t,
            "lr": 1e-4,
            "abs_grad_sum": abs_grad,
            "sem_grad_sum": sem_grad,
            "student_gs": _gs_stats(student_gs),
            "teacher_gs": _gs_stats(teacher_gs),
            "student_teacher_gs_meandiff": float(
                (student_gs - teacher_gs).abs().mean()
            ),
        }
        print(
            f"[audit] {Path(ckpt).stem} psnr={psnrs} "
            f"loss_rgb_train={loss_train_rgb:.4f} "
            f"abs_grad={abs_grad:.1f} sem_grad={sem_grad:.1f}"
        )

    (out_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
