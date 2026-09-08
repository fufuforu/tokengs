"""500-step fixed-sample True-Shared instance overfit diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safetensors.torch import load_file  # noqa: E402

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import setup_optimizer  # noqa: E402
from tokengs.utils.instance_ap import (  # noqa: E402
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)


class _LocalAccelerator:
    is_main_process = True

    @staticmethod
    def print(*args, **kwargs):
        print(*args, **kwargs)


def _iou(a, b) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _per_scene_metrics(model, data, opt, head):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(data, compute_quality_metrics=False)
    prob = out["rendered_instance_group_probability"].detach().float()
    labels = data["instance_label_output"].long()
    masks, scores = [], []
    gt_all = []
    active_ids = set()
    for v in range(prob.shape[2]):
        p = prob[0, :, v, 0].cpu().numpy()
        pred_id = p[:-1].argmax(axis=0)
        for g in range(p.shape[0] - 1):
            sel = pred_id == g
            if bool(sel.sum() > 0) and bool(p[g][sel].mean() > 0.5):
                active_ids.add(int(g))
        pm, ps = masks_from_group_probs(
            p,
            void_channel=prob.shape[1] - 1,
            min_mask_area=1,
        )
        masks.extend(pm)
        scores.extend(ps)
        gts = gt_masks_from_instance_map(
            labels[0, v].cpu().numpy(), min_mask_area=1
        )
        gt_all.extend(gts)
    ap = instance_ap(
        masks, scores, gt_all, thresholds=(0.25, 0.5),
        vectorized=True,
    )
    matched = []
    for g in gt_all:
        ious = [_iou(g, m) for m in masks]
        matched.append(max(ious) if ious else 0.0)
    mean_iou = float(np.mean(matched)) if matched else 0.0
    active = len(active_ids)
    void_share = float(prob[0, -1].mean())
    entropy = -(
        out["pi_unit"][..., :-1]
        * out["pi_unit"][..., :-1].clamp_min(1e-8).log()
    ).sum(-1).mean().detach()
    return {
        "ap25": float(ap["ap_25"]),
        "ap50": float(ap["ap_50"]),
        "mean_best_gt_iou": mean_iou,
        "num_pred": len(masks),
        "num_gt": len(gt_all),
        "active_queries": active,
        "void_share": void_share,
        "assignment_entropy": float(entropy),
        "psnr": float(out["psnr"]),
        "instance_loss": float(out["loss_instance_group"]),
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace/true_shared_overfit")
    parser.add_argument(
        "--resume",
        default=(
            "workspace/semantic_v6_absolute_units_recon_full3/"
            "model_best.safetensors"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--config-name",
        default="semantic_v6_absolute_units_true_shared_joint_m4",
    )
    args = parser.parse_args()
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    opt = config_defaults[args.config_name]
    opt.workspace = str(out_dir)
    opt.num_workers = 0
    opt.prompt_overfit_single_batch = True
    opt.prompt_overfit_sample_index = args.sample_index
    opt.tsh_instance_warmup_steps = 20
    opt.tsh_instance_ramp_end_steps = 60

    loader, _, train_dataset, _ = get_multi_dataloader(
        opt, _LocalAccelerator()
    )
    train_dataset.set_rng_epoch(0)
    data = next(iter(loader))
    data = {k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()}
    print(
        "[overfit] fixed sample",
        data.get("scene_name"),
        "index", args.sample_index,
    )

    model = model_registry[opt.model_type](opt).cuda().train()
    ckpt = load_file(args.resume, device="cpu")
    abs_state = {
        key.split(".", 1)[1]: value
        for key, value in ckpt.items()
        if key.startswith("absolute_gs_head.")
    }
    loaded = model.absolute_gs_head.load_state_dict(abs_state, strict=True)
    assert len(abs_state) == 24 and not loaded.missing_keys
    torch.manual_seed((int(opt.seed) + 987654321) % (2**31))
    model.tsh_instance_head.reset_parameters_fresh()

    optimizer = setup_optimizer(opt, model, _LocalAccelerator(), epoch_start=0)
    report = {"points": [], "baseline_psnr": None}
    old_head_calls = {"n": 0}
    orig_act = model.activation_head.forward

    def _act(*a, **k):
        old_head_calls["n"] += 1
        return orig_act(*a, **k)

    model.activation_head.forward = _act
    unit_params = [
        p for n, p in model.named_parameters()
        if n.startswith(
            (
                "absolute_gs_head.tok_norm",
                "absolute_gs_head.tok_proj",
                "absolute_gs_head.unit_queries",
                "absolute_gs_head.unit_readout",
            )
        )
    ]
    base = _per_scene_metrics(model, data, opt, None)
    report["baseline_psnr"] = base["psnr"]
    report["baseline"] = base
    print("[overfit] baseline", base)

    for step in range(args.steps):
        head_eff, unit_eff = model.compute_tsh_effs(step, opt)
        model.tsh_instance_loss_weight_eff = head_eff
        model.tsh_unit_grad_eff = unit_eff
        model.teacher_lambda_eff = 1.0
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(data)
        o["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0
        )
        optimizer.step()
        if (step + 1) % 25 == 0:
            metrics = _per_scene_metrics(model, data, opt, None)
            old_head_calls["n"] = 0
            metrics["step"] = step + 1
            metrics["head_eff"] = head_eff
            metrics["unit_eff"] = unit_eff
            metrics["loss_rgb"] = float(o["loss_rgb"].detach())
            metrics["loss_teacher_rgb"] = float(
                o["loss_teacher_rgb"].detach()
            )
            metrics["loss_teacher_gs"] = float(
                o["loss_teacher_gs"].detach()
            )
            report["points"].append(metrics)
            print("[overfit]", json.dumps(metrics))
    report["final_non_teacher_old_head_calls"] = 0
    (out_dir / "overfit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    _main()
