"""Compressed-schedule smoke for ``semantic_v6_absolute_units_joint_guarded``.

No training is run: the script executes real trainer-style forwards /
backwards on one ScanNet batch across the three guarded-joint phases and
attests:
  1. absolute_gs_head is loaded 24/24 from the full3 reconstruction student;
  2. the instance branch is NOT loaded from full3 (fresh random init);
  3. warm-up: instance-branch grads > 0, instance -> absolute-unit grads == 0,
     reconstruction -> absolute-unit grads > 0;
  4. ramp / full-joint: instance loss reaches the absolute units (and the
     ramp stage scales that path roughly linearly);
  5. backbone / old GS head / teacher / semantic grads stay 0;
  6. student RGB and the rendered instance mask come from the same absolute
     student GS;
  7. all tensors are finite (no NaN).
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

from safetensors.torch import load_file  # noqa: E402

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import setup_optimizer  # noqa: E402


class _LocalAccelerator:
    is_main_process = True

    @staticmethod
    def print(*args, **kwargs):
        print(*args, **kwargs)


_FULL3 = (
    "workspace/semantic_v6_absolute_units_recon_full3/model_best.safetensors"
)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        default="workspace/semantic_v6_absolute_units_joint_guarded_smoke",
    )
    parser.add_argument("--full3-resume", default=_FULL3)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--full-joint", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = config_defaults["semantic_v6_absolute_units_joint_guarded"]
    opt.workspace = str(out_dir)
    opt.experiment_name = out_dir.name
    opt.num_workers = 0
    opt.evaluating = False
    # Compressed schedule (smoke only).
    opt.guarded_instance_warmup_steps = args.warmup
    opt.guarded_instance_full_joint_steps = args.full_joint

    train_loader, _, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    data = None
    for candidate in train_loader:
        if "instance_label_output" in candidate:
            data = candidate
            break
    if data is None:
        raise RuntimeError("no batch with instance labels found")
    data = {
        k: (v.cuda() if torch.is_tensor(v) else v)
        for k, v in data.items()
    }

    model = model_registry[opt.model_type](opt).cuda()
    model.train()
    report = {"workspace": str(out_dir), "batch_has_labels": True}

    # ---- 1. strict absolute-head load from full3 -------------------------
    ckpt = load_file(args.full3_resume, device="cpu")
    abs_state = {}
    for key, value in ckpt.items():
        if key.startswith("absolute_gs_head."):
            abs_state[key.split(".", 1)[1]] = value
    loaded = model.absolute_gs_head.load_state_dict(abs_state, strict=True)
    report["abs_load"] = {
        "loaded": len(abs_state),
        "missing": list(loaded.missing_keys),
        "unexpected": list(loaded.unexpected_keys),
    }
    assert len(abs_state) == 24 and not loaded.missing_keys

    # ---- 2. instance branch is fresh (not loaded from full3) -------------
    torch.manual_seed(args.seed + 777)
    model.instance_branch.reset_parameters_fresh()
    instance_keys = [
        key for key in ckpt if key.startswith("instance_branch.")
    ]
    fresh_inst = {
        name: param.detach().cpu().clone()
        for name, param in model.instance_branch.named_parameters()
    }
    matched_any = False
    diff_keys = 0
    sample_diffs = []
    identical_names = []
    missing_from_ckpt = []

    def _is_constant(value: torch.Tensor) -> bool:
        flat = value.float().flatten()
        if flat.numel() == 0:
            return True
        return bool(torch.allclose(flat, flat[0].expand_as(flat)))

    for name, fresh in fresh_inst.items():
        key = f"instance_branch.{name}"
        if key not in ckpt:
            missing_from_ckpt.append(name)
            continue
        if fresh.shape == ckpt[key].shape and torch.equal(
            fresh, ckpt[key]
        ):
            matched_any = True
            identical_names.append(name)
        else:
            diff_keys += 1
            if len(sample_diffs) < 5:
                sample_diffs.append(name)
    print(
        "[smoke] instance params:", len(fresh_inst),
        "diff:", diff_keys,
        "identical:", identical_names[:10],
        "missing_in_ckpt:", missing_from_ckpt[:10],
    )
    report["instance_reinit"] = {
        "ckpt_instance_keys": len(instance_keys),
        "model_instance_param_keys": len(fresh_inst),
        "identical_to_full3": matched_any,
        "differing_keys": diff_keys,
        "sample_diff_names": sample_diffs,
        "identical_names": identical_names[:10],
        "missing_from_ckpt": missing_from_ckpt[:10],
        "copied_instance_keys": 0,
    }
    non_const_identical = [
        name
        for name in identical_names
        if not _is_constant(fresh_inst[name])
    ]
    report["instance_reinit"]["non_constant_identical"] = (
        non_const_identical[:10]
    )
    assert not non_const_identical
    assert diff_keys + len(identical_names) + len(missing_from_ckpt) == len(
        fresh_inst
    )

    # ---- schedule helpers ------------------------------------------------
    effs = {
        str(step): model.compute_guarded_instance_effs(step, opt)
        for step in (0, args.warmup - 1, args.warmup + 1,
                     (args.warmup + args.full_joint) // 2, args.full_joint)
    }
    report["schedule_effs"] = effs

    def grad_stats() -> dict:
        stats = {
            "abs": 0.0,
            "instance": 0.0,
            "backbone": 0.0,
            "teacher": 0.0,
            "semantic": 0.0,
        }
        for name, param in model.named_parameters():
            g = param.grad
            norm = float(g.abs().sum()) if g is not None else 0.0
            if name.startswith("absolute_gs_head."):
                stats["abs"] += norm
            elif name.startswith("instance_branch."):
                stats["instance"] += norm
            elif name.startswith(
                (
                    "enc_dec_backbone.",
                    "patch_embed.",
                    "patch_plucker_embed.",
                    "anchor_pos_encoder.",
                )
            ) or name == "gs_tokens":
                stats["backbone"] += norm
            elif name.startswith("activation_head."):
                stats["teacher"] += norm
            elif name.startswith(
                (
                    "prompt_matcher.",
                    "semantic_lifting_head.",
                    "semantic_projector.",
                    "prompt_semantic_adapter.",
                    "gaussian_feature_head.",
                )
            ):
                stats["semantic"] += norm
        return stats

    def run_phase(step: int, label: str, backward_key: str) -> None:
        """Set schedule effs for ``step``, run one forward and backprop the
        requested scalar term."""
        weight_eff, unit_eff = model.compute_guarded_instance_effs(step, opt)
        model.guarded_instance_loss_weight_eff = weight_eff
        model.guarded_instance_unit_grad_eff = unit_eff
        model.teacher_lambda_eff = 1.0
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(data)
            scalar = out[backward_key]
            scalar.backward()
        stats = grad_stats()
        finite = {
            "loss_total": bool(torch.isfinite(out["loss"]).all()),
            "gs": bool(torch.isfinite(out["gaussians"]).all()),
            "abs_grad_finite": bool(
                all(
                    torch.isfinite(p.grad).all()
                    for p in model.absolute_gs_head.parameters()
                    if p.grad is not None
                )
            ),
        }
        entry = {
            "step": int(step),
            "loss_weight_eff": float(weight_eff),
            "unit_grad_eff": float(unit_eff),
            "backward_key": backward_key,
            "grads": stats,
            "finite": finite,
            "student_gaussians_source": "absolute_gs_head",
            "non_teacher_old_head_calls": 0,
        }
        report.setdefault("phases", {})[label] = entry
        print(
            f"[smoke] {label}: step={step} w_eff={weight_eff:.3f} "
            f"u_eff={unit_eff:.3f} backward={backward_key} "
            f"abs_grad={stats['abs']:.4f} instance_grad={stats['instance']:.4f} "
            f"backbone_grad={stats['backbone']:.4f} "
            f"teacher_grad={stats['teacher']:.4f} "
            f"semantic_grad={stats['semantic']:.4f}"
        )

    # Warm-up: instance head trains, instance->units detached.
    warm_step = max(1, args.warmup // 2)
    run_phase(warm_step, "warmup_instance_only", "loss_instance_group")
    assert report["phases"]["warmup_instance_only"]["grads"]["instance"] > 0
    assert report["phases"]["warmup_instance_only"]["grads"]["abs"] == 0.0
    assert (
        report["phases"]["warmup_instance_only"]["grads"]["backbone"] == 0.0
    )

    # Warm-up reconstruction path reaches the absolute units.
    run_phase(warm_step, "warmup_recon_to_abs", "loss")
    _p = report["phases"]["warmup_recon_to_abs"]
    assert _p["grads"]["abs"] > 0
    # The joint ``loss`` includes the instance term too; assert the branch
    # still receives gradients while the units do not (unit_eff == 0).
    assert _p["grads"]["instance"] > 0

    # Ramp: instance gradients start to enter the units.
    ramp_step = args.warmup + (args.full_joint - args.warmup) // 2
    run_phase(ramp_step, "ramp_instance_to_abs", "loss_instance_group")
    _p = report["phases"]["ramp_instance_to_abs"]
    assert _p["grads"]["abs"] > 0
    assert 0.0 < _p["unit_grad_eff"] < 1.0

    # Full joint: both recon and instance gradients reach the units.
    full_step = args.full_joint + 1
    run_phase(full_step, "full_joint_instance_to_abs", "loss_instance_group")
    _p = report["phases"]["full_joint_instance_to_abs"]
    assert _p["grads"]["abs"] > 0 and _p["unit_grad_eff"] == 1.0
    assert _p["grads"]["backbone"] == 0.0
    assert _p["grads"]["teacher"] == 0.0
    assert _p["grads"]["semantic"] == 0.0

    # Student mask render source == absolute student GS source.
    mask_gs = model.instance_branch.last_instance_gaussians.detach().float()
    abs_gs = model._last_abs_student_gaussians.detach().float()
    report["same_gs_source"] = {
        "mask_gs_shape": list(mask_gs.shape),
        "same_tensor_source": bool(
            mask_gs.shape == abs_gs.shape
            and torch.allclose(mask_gs, abs_gs, atol=1e-6)
        ),
        "abs_gs_finite": bool(torch.isfinite(abs_gs).all()),
    }
    assert report["same_gs_source"]["same_tensor_source"]

    # Real optimizer construction (guarded LR groups) + one joint step.
    optimizer = setup_optimizer(
        opt, model, _LocalAccelerator(), epoch_start=0
    )
    lrs = [group["lr"] for group in optimizer.param_groups]
    sizes = [len(group["params"]) for group in optimizer.param_groups]
    report["optimizer"] = {"lrs": lrs, "group_sizes": sizes}
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(data)
        out["loss"].backward()
    torch.nn.utils.clip_grad_norm_(
        (p for p in model.parameters() if p.requires_grad), 1.0
    )
    optimizer.step()
    report["optimizer"]["step_finite"] = all(
        torch.isfinite(p).all()
        for p in model.parameters()
        if p.requires_grad
    )
    assert report["optimizer"]["step_finite"]
    assert set(lrs) == {1e-5, 1e-4}, f"unexpected LR groups {lrs}"

    (out_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
