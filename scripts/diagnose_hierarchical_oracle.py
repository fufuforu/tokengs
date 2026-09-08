"""Hierarchical oracle diagnostic: granularity of local grouping units.

No model modification, no training, no predicted assignment. Same frozen
wide7l@8000 + pgr3df2 checkpoint, LSM 40 held-out scenes, same per-GS GT
majority vote (15 views) and the exact official LSM AP evaluation.

Each 64-GS Token is split into K local units (K = 1/4/8/16/32/64) in two ways:
  - index split:  contiguous equal chunks by GS index (4 units = 4x16 GS);
  - spatial split: K-means over the token's 64 GS centers (K=64 = each GS its
    own unit, which must reproduce the GS-level oracle AP50=0.789).
Every unit takes the majority GT instance among its non-background GS as its
oracle label (background GS stay void). Units render through the real frozen
Gaussian geometry into the 7 target views; AP25/AP50/AP use the exact LSM
protocol (per-image confidence-ordered matching, 101-point interpolation).

Answers: does finer local grouping approach the GS-level oracle? Does 3D
spatial splitting beat index splitting?

Usage:
    python scripts/diagnose_hierarchical_oracle.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch._dynamo.config.disable = True
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_token_instance_mixing as dtm

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.input_types import split_data
from tokengs.models.instance_group_loss import (
    _gs_majority_target,
    _project_gs_to_views,
)
from tokengs.options import config_defaults
from tokengs.utils.instance_ap import (
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)


class _LocalAccelerator:
    is_main_process = True


K_LIST = [1, 4, 8, 16, 32, 64]
GS_PER_TOKEN = 64


def _unit_labels(gs_gt_token: torch.Tensor, means_token: torch.Tensor, K: int,
                 method: str) -> torch.Tensor:
    """Per-GS oracle label for one token split into K units (0 = void)."""
    n = gs_gt_token.numel()
    if K == 1:
        unit_ids = torch.zeros(n, dtype=torch.long, device=gs_gt_token.device)
    elif K == n:
        unit_ids = torch.arange(n, dtype=torch.long, device=gs_gt_token.device)
    elif method == "index":
        chunk = n // K
        unit_ids = (
            torch.arange(n, dtype=torch.long, device=gs_gt_token.device)
            // chunk
        )
    else:
        from scipy.cluster.vq import kmeans2

        _, unit_ids = kmeans2(
            means_token.cpu().numpy(), K, minit="points", seed=0
        )
        unit_ids = torch.from_numpy(unit_ids).long().to(gs_gt_token.device)
    labels = torch.zeros(n, dtype=torch.long, device=gs_gt_token.device)
    for u in torch.unique(unit_ids):
        sel = unit_ids == u
        votes = gs_gt_token[sel]
        votes = votes[votes > 0]
        if votes.numel() == 0:
            continue
        counts = torch.bincount(votes)
        labels[sel] = int(torch.argmax(counts))
    return labels


def _render_group_labels(
    gaussians, gs_label, cam_view, intrinsics, opt, model
):
    """One-hot per-GS labels rendered to target views (0 -> void channel)."""
    ids = [int(i) for i in torch.unique(gs_label[gs_label > 0])]
    id_to_group = {iid: g for g, iid in enumerate(ids)}
    num_groups = len(ids)
    n_gs = gs_label.numel()
    group_ids = torch.full(
        (n_gs,), num_groups, dtype=torch.long, device=gs_label.device
    )
    for iid, g in id_to_group.items():
        group_ids[gs_label == iid] = g
    probs = torch.zeros(
        (1, n_gs, num_groups + 1), device=gs_label.device, dtype=torch.float32
    )
    probs.scatter_(2, group_ids.unsqueeze(0).unsqueeze(-1), 1.0)
    render = model.gs.render_feature_channels(
        gaussians,
        probs,
        cam_view,
        intrinsics=intrinsics,
        opacity_scale=float(getattr(opt, "instance_group_render_scale", 1.0)),
    )
    rendered_channels = render["images_pred"] / (render["alphas_pred"] + 1e-5)
    rendered_probs = rendered_channels.float() / rendered_channels.float().sum(
        dim=2, keepdim=True
    ).clamp_min(1e-6)
    rendered_probability = rendered_probs.permute(0, 2, 1, 3, 4).unsqueeze(3)
    return rendered_probability, num_groups


def _eval_scene(rendered_probability, instance_labels, scene_name, args):
    void_channel = rendered_probability.shape[1] - 1
    batch_size, _, view_count, _, height, width = rendered_probability.shape
    pred_masks, pred_scores, pred_image_ids = [], [], []
    gt_masks, gt_image_ids = [], []
    for b in range(batch_size):
        for v in range(view_count):
            image_id = f"{scene_name}:b{b}"
            probs = rendered_probability[b, :, v, 0].float().cpu().numpy()
            masks, scores = masks_from_group_probs(
                probs,
                void_channel=void_channel,
                min_mask_area=args.min_pred_pixels,
            )
            masks = masks[: args.max_predictions_per_image]
            scores = scores[: args.max_predictions_per_image]
            gt_map = instance_labels[b, v].cpu().numpy()
            gts = gt_masks_from_instance_map(
                gt_map, min_mask_area=args.min_gt_pixels
            )
            pred_masks.extend(masks)
            pred_scores.extend(scores)
            pred_image_ids.extend([image_id] * len(masks))
            gt_masks.extend(gts)
            gt_image_ids.extend([image_id] * len(gts))
    results = instance_ap(
        pred_masks,
        pred_scores,
        gt_masks,
        thresholds=(0.25, 0.5, 0.75),
        vectorized=True,
        pred_image_ids=pred_image_ids,
        gt_image_ids=gt_image_ids,
    )
    coco_ap = instance_ap(
        pred_masks,
        pred_scores,
        gt_masks,
        thresholds=tuple(t / 100 for t in range(50, 100, 5)),
        vectorized=True,
        pred_image_ids=pred_image_ids,
        gt_image_ids=gt_image_ids,
    )
    return (
        {
            "ap": coco_ap["ap_mean"],
            "ap25": results["ap_25"],
            "ap50": results["ap_50"],
            "ap75": results["ap_75"],
            "num_pred": len(pred_masks),
            "num_gt": len(gt_masks),
        },
        pred_masks,
        pred_scores,
        pred_image_ids,
        gt_masks,
        gt_image_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=(
            "/space0/mawb/tokengs/workspace/"
            "semantic_v6_open_vocab_pgr3df2_train_12000/"
            "checkpoints/model_step_012000.safetensors"
        ),
    )
    parser.add_argument(
        "--workspace", default="workspace/diag_hierarchical_oracle"
    )
    parser.add_argument("--label", default="hierarchical_oracle")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument(
        "--lsm_manifest",
        default=(
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "lsm_instance_eval_manifest.json"
        ),
    )
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=15)
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--min_pred_pixels", type=int, default=1)
    parser.add_argument("--min_gt_pixels", type=int, default=1)
    parser.add_argument("--max_predictions_per_image", type=int, default=100)
    parser.add_argument("--gs_oracle_ap50", type=float, default=0.789)
    parser.add_argument("--baseline_ap50", type=float, default=0.232)
    args = parser.parse_args()

    opt = config_defaults["eval_scannet_lsm_instance"]
    opt.model_type = args.model_type
    opt.resume = args.resume
    opt.workspace = args.workspace
    opt.experiment_name = Path(args.workspace).name
    opt.num_input_views = int(args.num_input_views)
    opt.num_views = int(args.num_views)
    opt.dataset_kwargs = {
        **dict(opt.dataset_kwargs or {}),
        "lsm_manifest_path": args.lsm_manifest,
    }
    dtm._load_arch_from_checkpoint(args, opt)
    Path(args.workspace).mkdir(parents=True, exist_ok=True)
    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())

    from safetensors.torch import load_file

    model = model_registry[opt.model_type](opt)
    resume_ckpt = load_file(args.resume, device="cpu")
    torch.nn.Module.load_state_dict(model, resume_ckpt, strict=False)
    if not any(key.startswith("enc_dec_backbone.") for key in resume_ckpt):
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
                f"[hier] loaded {len(loadable)} backbone keys from "
                f"{backbone_path}"
            )
    model.eval()
    model = model.cuda()

    configs = [(k, m) for k in K_LIST for m in ("index", "spatial")]
    per_config = {f"K{k}_{m}": {"per_scene": {}} for k, m in configs}
    pooled = {
        f"K{k}_{m}": {
            "pred_masks": [],
            "pred_scores": [],
            "pred_image_ids": [],
            "gt_masks": [],
            "gt_image_ids": [],
        }
        for k, m in configs
    }
    for i, data in enumerate(test_loader):
        if args.max_scenes > 0 and i >= args.max_scenes:
            break
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        scene_name = data["scene_name"][0]
        t0 = time.time()
        with torch.no_grad():
            model_input, _ = split_data(data, opt)
            reconstruction, _, _ = model._forward_prompt_reconstruction(
                model_input
            )
            gaussians = reconstruction.gaussians
            means = gaussians[0, :, :3].float()
            cam_views = torch.cat(
                [data["cam_view_input"], data["cam_view"]], dim=1
            )[0]
            intrinsics_all = torch.cat(
                [data["intrinsics_input"], data["intrinsics"]], dim=1
            )[0]
            labels_all = torch.cat(
                [
                    data["instance_label_input"],
                    data["instance_label_output"],
                ],
                dim=1,
            )[0]
            ids, valid = _project_gs_to_views(
                means, cam_views, intrinsics_all, labels_all, tuple(opt.img_size)
            )
            gs_gt = _gs_majority_target(ids, valid)
            n_tokens = gs_gt.numel() // GS_PER_TOKEN
            gs_gt_t = gs_gt.view(n_tokens, GS_PER_TOKEN)
            means_t = means.view(n_tokens, GS_PER_TOKEN, 3)
            instance_labels = data["instance_label_output"].long()
        for k, method in configs:
            gs_label = torch.zeros_like(gs_gt)
            for t in range(n_tokens):
                gs_label[t * GS_PER_TOKEN : (t + 1) * GS_PER_TOKEN] = (
                    _unit_labels(gs_gt_t[t], means_t[t], k, method)
                )
            with torch.no_grad():
                rp, num_groups = _render_group_labels(
                    gaussians,
                    gs_label,
                    model_input.decoder.cam_view,
                    model_input.decoder.intrinsics,
                    opt,
                    model,
                )
            res, pm, psc, pid, gm, gid = _eval_scene(
                rp, instance_labels, scene_name, args
            )
            key = f"K{k}_{method}"
            per_config[key]["per_scene"][scene_name] = res
            pooled[key]["pred_masks"].extend(pm)
            pooled[key]["pred_scores"].extend(psc)
            pooled[key]["pred_image_ids"].extend(pid)
            pooled[key]["gt_masks"].extend(gm)
            pooled[key]["gt_image_ids"].extend(gid)
        line = "  ".join(
            f"K{k}{m[0]}={per_config[f'K{k}_{m}']['per_scene'][scene_name]['ap50']:.3f}"
            for k, m in configs
        )
        print(f"[hier] {scene_name}: {line} ({time.time()-t0:.1f}s)")

    # ---- global aggregation ----
    global_stats = {}
    for key, cfg in per_config.items():
        per = cfg["per_scene"]
        macro = {
            metric: float(
                np.mean([entry[metric] for entry in per.values()])
            )
            for metric in ("ap", "ap25", "ap50", "ap75")
        }
        pooled_ap = instance_ap(
            pooled[key]["pred_masks"],
            pooled[key]["pred_scores"],
            pooled[key]["gt_masks"],
            thresholds=(0.25, 0.5, 0.75),
            vectorized=True,
            pred_image_ids=pooled[key]["pred_image_ids"],
            gt_image_ids=pooled[key]["gt_image_ids"],
        )
        pooled_coco = instance_ap(
            pooled[key]["pred_masks"],
            pooled[key]["pred_scores"],
            pooled[key]["gt_masks"],
            thresholds=tuple(t / 100 for t in range(50, 100, 5)),
            vectorized=True,
            pred_image_ids=pooled[key]["pred_image_ids"],
            gt_image_ids=pooled[key]["gt_image_ids"],
        )
        macro.update(
            {
                "pooled_ap": pooled_coco["ap_mean"],
                "pooled_ap25": pooled_ap["ap_25"],
                "pooled_ap50": pooled_ap["ap_50"],
                "pooled_ap75": pooled_ap["ap_75"],
                "num_pred_mean": float(
                    np.mean([entry["num_pred"] for entry in per.values()])
                ),
                "num_gt_mean": float(
                    np.mean([entry["num_gt"] for entry in per.values()])
                ),
                "gap_vs_gs_oracle_ap50": macro["ap50"]
                - args.gs_oracle_ap50,
                "gap_vs_baseline_ap50": macro["ap50"] - args.baseline_ap50,
            }
        )
        global_stats[key] = macro

    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "num_scenes": len(next(iter(per_config.values()))["per_scene"]),
        "gs_oracle_ap50": args.gs_oracle_ap50,
        "baseline_ap50": args.baseline_ap50,
        "note": (
            "unit oracle label = majority GT instance among the unit's "
            "non-background GS (background GS stay void); exact LSM eval; "
            "K=64 both splits must reproduce the GS-level oracle."
        ),
        "global": global_stats,
        "per_config": per_config,
    }
    out = Path(args.workspace) / "hierarchical_oracle.json"
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n===== GLOBAL (scene-macro) =====")
    header = (
        f"{'config':10s} {'AP25':>7s} {'AP50':>7s} {'AP':>7s} "
        f"{'poolAP50':>8s} {'pred':>5s} {'gt':>5s} {'d(0.789)':>8s}"
    )
    print(header)
    for key in global_stats:
        g = global_stats[key]
        print(
            f"{key:10s} {g['ap25']:7.3f} {g['ap50']:7.3f} {g['ap']:7.3f} "
            f"{g['pooled_ap50']:8.3f} {g['num_pred_mean']:5.1f} "
            f"{g['num_gt_mean']:5.1f} {g['gap_vs_gs_oracle_ap50']:+8.3f}"
        )

    # ---- curves ----
    index_ap50 = [global_stats[f"K{k}_index"]["ap50"] for k in K_LIST]
    spatial_ap50 = [global_stats[f"K{k}_spatial"]["ap50"] for k in K_LIST]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(K_LIST, index_ap50, "o-", label="index split")
    ax.plot(K_LIST, spatial_ap50, "s-", label="spatial split (k-means)")
    ax.axhline(args.gs_oracle_ap50, color="g", ls="--", label="GS-level oracle 0.789")
    ax.axhline(args.baseline_ap50, color="r", ls=":", label="pgr3df2 0.232")
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_LIST)
    ax.set_xticklabels([str(k) for k in K_LIST])
    ax.set_xlabel("units per token (K)")
    ax.set_ylabel("AP50")
    ax.set_title("AP50 vs local grouping granularity (oracle, frozen geometry)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    curve_path = Path(args.workspace) / "ap50_vs_granularity.png"
    fig.savefig(curve_path, dpi=110)
    plt.close(fig)

    print("\n===== ANSWER =====")
    k1 = global_stats["K1_index"]["ap50"]
    k64_idx = global_stats["K64_index"]["ap50"]
    k64_sp = global_stats["K64_spatial"]["ap50"]
    print(f"  token-level (K=1) oracle AP50: {k1:.3f}")
    print(f"  K=64 index  AP50: {k64_idx:.3f} (sanity vs 0.789)")
    print(f"  K=64 spatial AP50: {k64_sp:.3f} (sanity vs 0.789)")
    print(f"  curve saved: {curve_path}")
    print(f"\n[hier] wrote {out}")


if __name__ == "__main__":
    main()
