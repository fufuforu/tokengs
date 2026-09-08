"""Read-only oracle diagnostics for the multi-view instance binding problem.

On fixed validation scenes (LSM manifest, first N scenes) with the Both@1420
checkpoint, reports, per scene-global GT instance:

  * best-IoU prediction(s) per visible view;
  * number of predicted query masks with IoU > 0.25 / > 0.5;
  * Recall@IoU50 and an oracle "merge duplicate queries for this GT" IoU;
  * an oracle "keep only the single best cross-view query" coverage ratio;
  * same-view duplicate-prediction ratio (pred-pred IoU > 0.5).

No weights/checkpoints/config are modified and no metric is tuned.
"""

from __future__ import annotations

import argparse
import json
import statistics
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
from tokengs.utils.instance_ap import gt_masks_from_instance_map  # noqa: E402
from _240_path_remap import remap_opt  # noqa: E402


class _LocalAccelerator:
    is_main_process = True

    @staticmethod
    def print(*args, **kwargs):
        print(*args, **kwargs)


def _load_heads(model, ckpt):
    abs_state = {
        key.split(".", 1)[1]: value
        for key, value in ckpt.items()
        if key.startswith("absolute_gs_head.")
    }
    res = model.absolute_gs_head.load_state_dict(abs_state, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    tsh_state = {
        key.split(".", 1)[1]: value
        for key, value in ckpt.items()
        if key.startswith("tsh_instance_head.")
    }
    res = model.tsh_instance_head.load_state_dict(tsh_state, strict=True)
    assert not res.missing_keys and not res.unexpected_keys


def _iou(a, b):
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _group_preds(probs: np.ndarray, void_channel: int):
    group_ids = np.argmax(probs, axis=0)
    scores = np.max(probs, axis=0)
    items = []
    for g in range(probs.shape[0]):
        if g == void_channel:
            continue
        mask = group_ids == g
        if int(mask.sum()) < 1:
            continue
        items.append((g, mask, float(scores[mask].mean())))
    items.sort(key=lambda x: -x[2])
    return items


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace/tsh_mv_oracle")
    parser.add_argument(
        "--resume",
        default=(
            "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_"
            "w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors"
        ),
    )
    parser.add_argument("--n-scenes", type=int, default=4)
    args = parser.parse_args()
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = config_defaults["eval_scannet_lsm_instance"]
    opt.workspace = str(out_dir)
    opt.num_workers = 0
    opt.dataset_kwargs = {
        **dict(opt.dataset_kwargs or {}),
        "lsm_manifest_path": (
            "/space/mawb/tokengs/data/scannet_prompt/"
            "lsm_instance_eval_manifest.json"
        ),
    }
    remap_opt(opt)
    opt.prompt_tokengs_checkpoint = (
        "/space/mawb/tokengs/workspace/scannet_recon_finetune_base_8k/"
        "tokengs_backbone_step_008000.safetensors"
    )
    opt.backbone_resume = opt.prompt_tokengs_checkpoint
    opt.prompt_clip_model_path = (
        "/space/mawb/tokengs/checkpoints/clip-vit-large-patch14"
    )
    opt.instance_branch_abs_units = True
    opt.abs_true_shared_units = True
    opt.instance_branch_token_units = False
    opt.instance_branch_independent = False
    opt.tsh_num_groups = 100
    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    model = model_registry[opt.model_type](opt).cuda().eval()
    ckpt = load_file(args.resume, device="cpu")
    _load_heads(model, ckpt)
    model.teacher_lambda_eff = 0.0

    scenes_report = []
    agg = {
        "recall_iou50": [],
        "best_iou": [],
        "preds_over_025_per_gt": [],
        "preds_over_05_per_gt": [],
        "merge_oracle_iou": [],
        "best_query_iou": [],
        "visible_views_per_gt": [],
        "duplicate_pred_ratio_view": [],
    }
    for scene_i, data in enumerate(test_loader):
        if scene_i >= args.n_scenes:
            break
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        scene = str(data["scene_name"][0])
        with torch.inference_mode():
            out = model(data, compute_quality_metrics=True)
        probability = out["rendered_instance_group_probability"]
        void_channel = probability.shape[1] - 1
        labels = data["instance_label_output"].long().cpu().numpy()
        if labels.ndim == 5:
            labels = labels[0, :, 0]
        elif labels.ndim == 4:
            labels = labels[0]
        probs = probability[0].float().cpu().numpy()  # [G+1,V,1,H,W]
        probs = probs[:, :, 0]  # [G+1,V,H,W]
        if labels.ndim == 3:
            label_view_count = labels.shape[0]
        else:
            label_view_count = labels.shape[1]
        view_count = min(probs.shape[1], label_view_count)

        preds_by_view = []  # list of list[(group, mask, score)]
        gt_by_view = []  # list of list[(gt_raw_id, mask)]
        for v in range(view_count):
            preds_by_view.append(
                _group_preds(probs[:, v], void_channel)
            )
            gt_items = gt_masks_from_instance_map(
                labels[0, v], min_mask_area=1
            )
            # map raw GT id from map: reconstruct via gt_masks_from_instance_map
            # returns masks without ids; recompute ids directly.
            gt_entries = []
            gt_map = (
                labels[v]
                if labels.ndim == 3
                else labels[0, v]
            )
            for gid in np.unique(gt_map):
                if int(gid) in (0, 255):
                    continue
                mask = gt_map == gid
                if int(mask.sum()) < 1:
                    continue
                gt_entries.append((int(gid), mask))
            gt_by_view.append(gt_entries)

        # visibility per raw GT id
        id_views = {}
        for v, entries in enumerate(gt_by_view):
            for gid, mask in entries:
                id_views.setdefault(gid, []).append((v, mask))
        # duplicate pred masks within each view
        dup_ratios = []
        for v, preds in enumerate(preds_by_view):
            dup = 0
            count = len(preds)
            for i in range(count):
                for j in range(i + 1, count):
                    if _iou(preds[i][1], preds[j][1]) > 0.5:
                        dup += 1
                        break
            dup_ratios.append(dup / max(1, count))
        agg["duplicate_pred_ratio_view"].extend(dup_ratios)

        scene_gt = {}
        for gid, views in id_views.items():
            visible = len(views)
            best = 0.0
            over25 = 0
            over50 = 0
            gt_view_masks = []
            pred_by_gt_query = {}
            for v, gt_mask in views:
                gt_view_masks.append((v, gt_mask))
                for g, pm, score in preds_by_view[v]:
                    iou = _iou(pm, gt_mask)
                    best = max(best, iou)
                    if iou > 0.25:
                        over25 += 1
                    if iou > 0.5:
                        over50 += 1
                    if iou > 0.25:
                        pred_by_gt_query.setdefault(g, []).append(
                            (v, pm, iou)
                        )
            # oracle: merge all query masks >0.25 for this GT per visible
            # view, then best view-union IoU
            merged_best = 0.0
            for v, gt_mask in gt_view_masks:
                union = np.zeros_like(gt_mask, dtype=bool)
                any_mask = False
                for g, entries in pred_by_gt_query.items():
                    for ev, pm, iou in entries:
                        if ev == v:
                            union |= pm
                            any_mask = True
                if any_mask:
                    merged_best = max(merged_best, _iou(union, gt_mask))
            # oracle: keep only the best single cross-view query for the GT
            best_query_total = 0.0
            best_query_iou = 0.0
            for g, entries in pred_by_gt_query.items():
                total = sum(iou for _, _, iou in entries)
                if total > best_query_total:
                    best_query_total = total
                    best_query_iou = max(iou for _, _, iou in entries)
            scene_gt[gid] = {
                "visible_views": visible,
                "best_iou": best,
                "preds_over_025": over25,
                "preds_over_05": over50,
                "recall50": bool(best >= 0.5),
                "merge_oracle_iou": merged_best,
                "best_query_iou": (
                    best_query_iou if pred_by_gt_query else 0.0
                ),
                "distinct_query_count": len(pred_by_gt_query),
            }
        agg["recall_iou50"].extend(
            [int(v["recall50"]) for v in scene_gt.values()]
        )
        agg["best_iou"].extend([v["best_iou"] for v in scene_gt.values()])
        agg["preds_over_025_per_gt"].extend(
            [v["preds_over_025"] for v in scene_gt.values()]
        )
        agg["preds_over_05_per_gt"].extend(
            [v["preds_over_05"] for v in scene_gt.values()]
        )
        agg["merge_oracle_iou"].extend(
            [v["merge_oracle_iou"] for v in scene_gt.values()]
        )
        agg["best_query_iou"].extend(
            [v["best_query_iou"] for v in scene_gt.values()]
        )
        agg["visible_views_per_gt"].extend(
            [v["visible_views"] for v in scene_gt.values()]
        )
        scenes_report.append(
            {
                "scene": scene,
                "gt_instances": scene_gt,
                "pred_count_per_view": [
                    len(preds_by_view[v]) for v in range(view_count)
                ],
                "duplicate_pred_ratio_per_view": dup_ratios,
                "psnr": float(out["psnr"]),
            }
        )
        print(
            f"[oracle] scene={scene} gt={len(scene_gt)} "
            f"recall50={sum(v['recall50'] for v in scene_gt.values())}",
            flush=True,
        )

    def _summ(values):
        if not values:
            return None
        o = sorted(values)
        return {
            "min": o[0],
            "median": statistics.median(o),
            "p90": o[min(len(o) - 1, int(0.90 * len(o)))],
            "max": o[-1],
            "mean": statistics.fmean(o),
            "count": len(o),
        }

    report = {
        "resume": str(args.resume),
        "n_scenes": len(scenes_report),
        "summary": {
            "recall_iou50": sum(agg["recall_iou50"]),
            "recall_iou50_ratio": (
                sum(agg["recall_iou50"]) / len(agg["recall_iou50"])
                if agg["recall_iou50"]
                else None
            ),
            "best_iou": _summ(agg["best_iou"]),
            "preds_over_025_per_gt": _summ(agg["preds_over_025_per_gt"]),
            "preds_over_05_per_gt": _summ(agg["preds_over_05_per_gt"]),
            "merge_oracle_iou": _summ(agg["merge_oracle_iou"]),
            "best_query_iou": _summ(agg["best_query_iou"]),
            "visible_views_per_gt": _summ(agg["visible_views_per_gt"]),
            "duplicate_pred_ratio_view": _summ(
                agg["duplicate_pred_ratio_view"]
            ),
        },
        "scenes": scenes_report,
    }
    out_path = out_dir / "oracle_multiview_query_audit.json"
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"[oracle] wrote {out_path}")


if __name__ == "__main__":
    _main()
