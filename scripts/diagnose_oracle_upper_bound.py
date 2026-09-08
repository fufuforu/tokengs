"""Oracle upper-bound diagnostic: perfect token-internal instance mixing.

No model modification, no training, no predicted assignment. Using the frozen
wide7l@8000 + pgr3df2 best checkpoint on the LSM 40 held-out scenes:

1. Per-GS GT instance id is obtained with the existing 3D-loss convention:
   project each GS center into all 15 labeled views and majority-vote.
2. The oracle assignment is EXACTLY that GT id (token-internal mixing is
   assumed perfectly solved -- each GS knows its true instance).
3. Every GT instance's GS set is rendered as a one-hot group through the same
   Gaussian renderer into the 7 target views (alpha-composited, same
   opacity_scale as the real eval).
4. AP25 / AP50 / AP are computed with the exact LSM protocol (per-image
   confidence-ordered matching inside each scene, 101-point interpolation).
5. Extra stats: oracle mask IoU vs GT per instance, pred/gt counts, and how
   many GT instances have zero (or too few) Gaussians to be represented.

Answer: if token-internal mixing were perfectly resolved, how high could the
frozen TokenGS Gaussian geometry reach on instance AP?

Usage:
    python scripts/diagnose_oracle_upper_bound.py
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


def _render_oracle(
    gaussians: torch.Tensor,
    gs_gt: torch.Tensor,
    cam_view: torch.Tensor,
    intrinsics: torch.Tensor,
    opt,
    model,
) -> tuple[torch.Tensor, dict]:
    """One-hot oracle groups rendered to target views."""
    ids = [int(i) for i in torch.unique(gs_gt[gs_gt > 0])]
    id_to_group = {iid: g for g, iid in enumerate(ids)}
    num_groups = len(ids)
    n_gs = gs_gt.numel()
    group_ids = torch.full(
        (n_gs,), num_groups, dtype=torch.long, device=gs_gt.device
    )
    for iid, g in id_to_group.items():
        group_ids[gs_gt == iid] = g
    probs = torch.zeros(
        (1, n_gs, num_groups + 1), device=gs_gt.device, dtype=torch.float32
    )
    probs.scatter_(2, group_ids.unsqueeze(0).unsqueeze(-1), 1.0)
    render = model.gs.render_feature_channels(
        gaussians,
        probs,
        cam_view,
        intrinsics=intrinsics,
        opacity_scale=float(getattr(opt, "instance_group_render_scale", 1.0)),
    )
    rendered_groups = render["images_pred"]  # [B,V,G,H,W]
    rendered_alpha = render["alphas_pred"]
    rendered_channels = rendered_groups / (rendered_alpha + 1e-5)
    rendered_probs = rendered_channels.float() / rendered_channels.float().sum(
        dim=2, keepdim=True
    ).clamp_min(1e-6)
    rendered_probability = rendered_probs.permute(0, 2, 1, 3, 4).unsqueeze(
        3
    )  # [B,G+1,V,1,H,W]
    # iid -> group index (the direction used by _oracle_instance_stats).
    mapping = id_to_group
    return rendered_probability, {"num_groups": num_groups, "mapping": mapping}


def _oracle_instance_stats(
    rendered_probability: torch.Tensor,
    instance_labels: torch.Tensor,
    mapping: dict,
    gs_gt: torch.Tensor,
    min_gt_pixels: int,
    min_gs: int,
) -> dict:
    """Per-GT-instance oracle mask IoU and representability stats."""
    instance_ids = [int(i) for i in torch.unique(instance_labels) if i > 0]
    probs = rendered_probability[0].float().cpu().numpy()  # [G+1,V,H,W]
    labels = instance_labels[0].cpu().numpy()
    argmax = probs.argmax(axis=0)  # [V,H,W]
    stats = []
    for iid in instance_ids:
        n_gs = int((gs_gt == iid).sum())
        ious = []
        for v in range(labels.shape[0]):
            gt_mask = labels[v] == iid
            if int(gt_mask.sum()) < min_gt_pixels:
                continue
            group = mapping.get(int(iid))
            if group is None:
                ious.append(0.0)
                continue
            pred = argmax[v] == group
            inter = float(np.logical_and(pred, gt_mask).sum())
            union = float(np.logical_or(pred, gt_mask).sum())
            ious.append(inter / max(union, 1.0))
        stats.append(
            {
                "instance_id": iid,
                "n_gs": n_gs,
                "representable": n_gs >= 1,
                "representable_16": n_gs >= min_gs,
                "oracle_iou": float(np.mean(ious)) if ious else None,
                "n_views_with_gt": len(ious),
            }
        )
    n_total = len(stats)
    n_no_gs = sum(1 for s in stats if s["n_gs"] == 0)
    n_lt16 = sum(1 for s in stats if 0 < s["n_gs"] < min_gs)
    ious = [s["oracle_iou"] for s in stats if s["oracle_iou"] is not None]
    return {
        "n_gt_instances": n_total,
        "n_zero_gs": n_no_gs,
        "n_lt_min_gs": n_lt16,
        "n_representable": sum(1 for s in stats if s["representable"]),
        "n_representable_16": sum(
            1 for s in stats if s["representable_16"]
        ),
        "oracle_iou_mean": float(np.mean(ious)) if ious else None,
        "oracle_iou_median": float(np.median(ious)) if ious else None,
        "per_instance": stats,
    }


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
        "--workspace", default="workspace/diag_oracle_upper_bound"
    )
    parser.add_argument("--label", default="oracle_upper_bound")
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
    parser.add_argument("--min_gs", type=int, default=16)
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
                f"[oracle] loaded {len(loadable)} backbone keys from "
                f"{backbone_path}"
            )
    model.eval()
    model = model.cuda()

    per_scene = {}
    pooled_pred_masks, pooled_pred_scores, pooled_pred_image_ids = [], [], []
    pooled_gt_masks, pooled_gt_image_ids = [], []
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
            rendered_probability, meta = _render_oracle(
                gaussians,
                gs_gt,
                model_input.decoder.cam_view,
                model_input.decoder.intrinsics,
                opt,
                model,
            )
        instance_labels = data["instance_label_output"].long()
        void_channel = meta["num_groups"]
        batch_size, _, view_count, _, height, width = (
            rendered_probability.shape
        )
        pred_masks, pred_scores, pred_image_ids = [], [], []
        gt_masks, gt_image_ids = [], []
        for b in range(batch_size):
            for v in range(view_count):
                image_id = f"{scene_name}:b{b}"
                probs = (
                    rendered_probability[b, :, v, 0].float().cpu().numpy()
                )
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
        pooled_pred_masks.extend(pred_masks)
        pooled_pred_scores.extend(pred_scores)
        pooled_pred_image_ids.extend(pred_image_ids)
        pooled_gt_masks.extend(gt_masks)
        pooled_gt_image_ids.extend(gt_image_ids)
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
        istats = _oracle_instance_stats(
            rendered_probability,
            instance_labels,
            meta["mapping"],
            gs_gt,
            args.min_gt_pixels,
            args.min_gs,
        )
        per_scene[scene_name] = {
            **results,
            "ap": coco_ap["ap_mean"],
            "num_gt_instances": len(gt_masks),
            "num_pred_instances": len(pred_masks),
            "num_images": view_count * batch_size,
            "oracle_groups": meta["num_groups"],
            "instance_stats": istats,
        }
        print(
            f"[oracle] {scene_name}: AP={coco_ap['ap_mean']:.4f} "
            f"AP50={results['ap_50']:.4f} AP25={results['ap_25']:.4f} "
            f"gt={len(gt_masks)} pred={len(pred_masks)} "
            f"zero_gs={istats['n_zero_gs']}/{istats['n_gt_instances']} "
            f"iou={istats['oracle_iou_mean']:.3f} ({time.time()-t0:.1f}s)"
        )

    def _mean(key):
        return float(
            np.mean([entry[key] for entry in per_scene.values()])
        )

    pooled_ap = instance_ap(
        pooled_pred_masks,
        pooled_pred_scores,
        pooled_gt_masks,
        thresholds=(0.25, 0.5, 0.75),
        vectorized=True,
        pred_image_ids=pooled_pred_image_ids,
        gt_image_ids=pooled_gt_image_ids,
    )
    pooled_coco = instance_ap(
        pooled_pred_masks,
        pooled_pred_scores,
        pooled_gt_masks,
        thresholds=tuple(t / 100 for t in range(50, 100, 5)),
        vectorized=True,
        pred_image_ids=pooled_pred_image_ids,
        gt_image_ids=pooled_gt_image_ids,
    )
    macro = {
        "ap": _mean("ap"),
        "ap25": _mean("ap_25"),
        "ap50": _mean("ap_50"),
        "ap75": _mean("ap_75"),
        "pooled_ap": pooled_coco["ap_mean"],
        "pooled_ap25": pooled_ap["ap_25"],
        "pooled_ap50": pooled_ap["ap_50"],
        "pooled_ap75": pooled_ap["ap_75"],
        "pgr3df2_baseline_ap50": 0.232,
        "gap_vs_baseline_ap50": _mean("ap_50") - 0.232,
        "n_gt_instances": _mean("num_gt_instances"),
        "n_pred_instances": _mean("num_pred_instances"),
        "zero_gs_fraction": float(
            np.mean(
                [
                    entry["instance_stats"]["n_zero_gs"]
                    / max(entry["instance_stats"]["n_gt_instances"], 1)
                    for entry in per_scene.values()
                ]
            )
        ),
        "lt_min_gs_fraction": float(
            np.mean(
                [
                    entry["instance_stats"]["n_lt_min_gs"]
                    / max(entry["instance_stats"]["n_gt_instances"], 1)
                    for entry in per_scene.values()
                ]
            )
        ),
        "oracle_iou_mean": float(
            np.mean(
                [
                    entry["instance_stats"]["oracle_iou_mean"]
                    for entry in per_scene.values()
                    if entry["instance_stats"]["oracle_iou_mean"] is not None
                ]
            )
        ),
    }
    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "num_scenes": len(per_scene),
        "protocol_note": (
            "exact LSM eval protocol (per-image confidence-ordered matching "
            "inside scene, 101-point AP); oracle groups = GT instance id "
            "majority-voted per GS from 15 views; rendered through the frozen "
            "Gaussian renderer on the 7 target views."
        ),
        "global": macro,
        "per_scene": per_scene,
    }
    out = Path(args.workspace) / "oracle_upper_bound.json"
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n===== ORACLE UPPER BOUND (GLOBAL) =====")
    print(f"  AP25 : {macro['ap25']:.4f} (pooled {macro['pooled_ap25']:.4f})")
    print(f"  AP50 : {macro['ap50']:.4f} (pooled {macro['pooled_ap50']:.4f})")
    print(f"  AP75 : {macro['ap75']:.4f} (pooled {macro['pooled_ap75']:.4f})")
    print(f"  AP   : {macro['ap']:.4f} (pooled {macro['pooled_ap']:.4f})")
    print(f"  pred/gt masks per scene: {macro['n_pred_instances']:.1f} / "
          f"{macro['n_gt_instances']:.1f}")
    print(f"  GT instances with ZERO GS: {macro['zero_gs_fraction']*100:.1f}%")
    print(f"  GT instances with <{args.min_gs} GS: "
          f"{macro['lt_min_gs_fraction']*100:.1f}%")
    print(f"  oracle mask IoU vs GT (mean): {macro['oracle_iou_mean']:.3f}")
    print(f"  gap vs pgr3df2 AP50=0.232: {macro['gap_vs_baseline_ap50']:+.3f}")
    print("\n===== ANSWER =====")
    print(
        f"If token-internal mixing were perfectly solved, the frozen "
        f"TokenGS Gaussian geometry reaches AP50 ~ {macro['ap50']:.3f} "
        f"(AP ~ {macro['ap']:.3f}), i.e. "
        f"{macro['ap50']/0.232:.2f}x the current pgr3df2 AP50=0.232."
    )
    print(f"\n[oracle] wrote {out}")


if __name__ == "__main__":
    main()
