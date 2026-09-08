"""Single-scene / fixed-batch reconstruction overfit for the absolute-unit
student (diagnostic; no instance/semantic/MBM).

Loads a checkpoint (default: the recon-continuation model_best), pins one
fixed ScanNet training sample (same context/target views every step), freezes
backbone + teacher + everything except ``absolute_gs_head`` (unit formation +
unit GS decoder), keeps the frozen base8k teacher ON at full weight, and
trains only:
    RGB reconstruction + teacher-RGB distill + low-weight GS distill.

Every ``--eval-every`` steps the SAME sample is evaluated in eval mode and we
print student/teacher PSNR/SSIM/LPIPS, student-vs-teacher GS distance and the
old-GS-head call count (must be 0 on the student path).
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


def _psnr_from_rgb(pred, gt) -> torch.Tensor:
    mse = (pred.clamp(0, 1) - gt.clamp(0, 1)).square().mean()
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=(
            "/space0/mawb/tokengs/workspace/"
            "semantic_v6_absolute_units_recon_continue_4000/model_best.safetensors"
        ),
    )
    parser.add_argument(
        "--workspace", default="workspace/abs_single_overfit_diag"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = config_defaults["semantic_v6_absolute_units_recon_continue"]
    opt.workspace = args.workspace
    opt.experiment_name = out_dir.name
    opt.num_workers = 0
    opt.prompt_overfit_single_batch = True
    opt.prompt_overfit_sample_index = int(args.sample_index)
    opt.evaluating = False

    train_loader, _, train_dataset, _ = get_multi_dataloader(
        opt, _Accelerator()
    )
    print(
        "[abs-overfit] fixed sample index "
        f"{opt.prompt_overfit_sample_index} (train size "
        f"{len(train_dataset)})"
    )

    from safetensors.torch import load_file, save_file

    resume_state = load_file(args.resume, device="cpu")
    abs_ckpt_keys = sum(
        1 for key in resume_state if key.startswith("absolute_gs_head.")
    )
    print(f"[abs-overfit] resume {args.resume} (absolute_gs_head keys "
          f"{abs_ckpt_keys})")

    model = model_registry[opt.model_type](opt)
    model = model.cuda()
    torch.nn.Module.load_state_dict(model, resume_state, strict=False)
    # Freeze everything except the absolute unit former + unit GS decoder.
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("absolute_gs_head."))
    trainable = [
        param
        for param in model.parameters()
        if param.requires_grad
    ]
    print(
        f"[abs-overfit] trainable params: {sum(p.numel() for p in trainable)}"
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.05)

    model.teacher_lambda_eff = 1.0
    model.instance_stage_eff = 0.0

    old_head_calls = 0
    _orig_act = model.activation_head.forward

    def _counting_act(*a, **k):
        nonlocal old_head_calls
        old_head_calls += 1
        return _orig_act(*a, **k)

    model.activation_head.forward = _counting_act

    def _fetch_sample():
        train_dataset.set_rng_epoch(0)
        data = next(iter(train_loader))
        return {
            key: (value.cuda() if torch.is_tensor(value) else value)
            for key, value in data.items()
        }

    def _student_metrics(data):
        nonlocal old_head_calls
        model.eval()
        old_head_calls = 0
        with torch.inference_mode():
            out = model(data, compute_quality_metrics=True)
        calls = old_head_calls
        model.train()
        return out, calls

    log = {"steps": [], "evals": []}
    for step in range(1, args.steps + 1):
        data = _fetch_sample()
        model.train()
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(data)
            loss = out["loss"]
            loss.backward()
        sem_grad = 0.0
        abs_grad = 0.0
        abs_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
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
            if name.startswith("absolute_gs_head."):
                abs_norm += float(param.detach().float().abs().sum())
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        inst_value = out.get("loss_instance_group")
        inst_loss = (
            float(inst_value.detach().float())
            if inst_value is not None
            else 0.0
        )
        log["steps"].append(
            {
                "step": step,
                "loss": float(loss.detach().float()),
                "loss_rgb": float(out["loss_rgb"].detach().float()),
                "loss_teacher_gs": float(
                    out["loss_teacher_gs"].detach().float()
                ),
                "loss_teacher_rgb": float(
                    out["loss_teacher_rgb"].detach().float()
                ),
                "teacher_called": bool(model.teacher_called),
                "instance_loss": inst_loss,
                "psnr_train": float(out["psnr"].detach().float()),
                "abs_grad": abs_grad,
                "sem_grad": sem_grad,
                "abs_norm": abs_norm,
                "old_head_calls": old_head_calls,
            }
        )
        if step == 1 or step % max(1, args.eval_every) == 0:
            data = _fetch_sample()
            student_out, calls = _student_metrics(data)
            model_input, supervision = split_data(data, opt)
            with torch.no_grad():
                teacher_recon, _, teacher_rgb_res = (
                    model._forward_prompt_reconstruction(model_input)
                )
            teacher_rgb = teacher_rgb_res["images_pred"]
            gt_rgb = supervision.images_output
            student_gs = student_out["gaussians"].detach().float()
            teacher_gs = teacher_recon.gaussians.detach().float()
            model._quality_metrics.device = str(gt_rgb.device)
            teacher_ssim = model._quality_metrics.calculate_ssim(
                teacher_rgb, gt_rgb
            )
            teacher_lpips = model._quality_metrics.calculate_lpips(
                teacher_rgb, gt_rgb
            )
            log["evals"].append(
                {
                    "step": step,
                    "student_psnr": float(
                        student_out["psnr"].detach().float()
                    ),
                    "student_ssim": float(
                        student_out["ssim"].detach().float()
                    ),
                    "student_lpips": float(
                        student_out["lpips"].detach().float()
                    ),
                    "teacher_psnr": float(
                        _psnr_from_rgb(teacher_rgb, gt_rgb).detach().float()
                    ),
                    "teacher_ssim": float(teacher_ssim),
                    "teacher_lpips": float(teacher_lpips),
                    "student_teacher_gs_meandiff": float(
                        (student_gs - teacher_gs).abs().mean()
                    ),
                    "old_head_calls_student_eval": calls,
                }
            )
            print(
                f"[abs-overfit-eval] step={step} "
                f"student_psnr={log['evals'][-1]['student_psnr']:.3f} "
                f"teacher_psnr={log['evals'][-1]['teacher_psnr']:.3f} "
                f"gs_diff={log['evals'][-1]['student_teacher_gs_meandiff']:.4f} "
                f"old_calls={calls}"
            )
        if step % max(1, args.save_every) == 0 or step == args.steps:
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            path = ckpt_dir / f"model_step_{step:06d}.safetensors"
            save_file(
                {
                    key: value.detach().cpu().contiguous()
                    for key, value in model.state_dict().items()
                },
                str(path),
            )
            print(f"[abs-overfit] saved {path}")

    (out_dir / "overfit_report.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )
    print(f"[abs-overfit] report -> {out_dir / 'overfit_report.json'}")


if __name__ == "__main__":
    main()
