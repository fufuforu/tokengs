"""LSM evaluation for the PointGroup-style instance-center pipeline.

One-shot feed-forward: the frozen TokenGS + 8-local-unit + DINO model runs
once; a small center-offset head predicts each unit's instance-center
offset; the offset-adjusted unit centers are tight per instance and are
grouped by a simple 3D center clustering (DBSCAN or agglomerative on the
centers), propagated to the frozen Gaussians through the GS->unit
assignment, rendered, and scored with the standard LSM AP protocol.

Also reports: predicted-center same/diff distances (diagnostic), cluster
count vs GT, pred/gt, PSNR.  No GT is used at inference.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tokengs.data import get_multi_dataloader
from tokengs.options import config_defaults
from tokengs.utils.instance_ap import (
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)

import cluster_affinity as ca  # noqa: E402
from ablate_region_growing import agglomerative_labels  # noqa: E402
from eval_instance_lsm_protocol import (  # noqa: E402
    _LocalAccelerator,
    _audit_lsm_manifest,
    _load_checkpoint_arch,
)


def center_labels(
    centers: np.ndarray,
    fg: np.ndarray,
    method: str,
    eps: float,
    min_samples: int = 1,
) -> np.ndarray:
    """Cluster the offset-adjusted unit centers -> per-unit labels."""
    u = centers.shape[0]
    labels = np.full(u, -1, dtype=np.int64)
    idx = np.where(fg)[0]
    if idx.size < 2:
        return labels
    c = centers[idx]
    if method == "dbscan":
        from sklearn.cluster import DBSCAN

        lab = DBSCAN(
            eps=float(eps), min_samples=int(min_samples),
            metric="euclidean", algorithm="kd_tree", n_jobs=-1,
        ).fit_predict(c)
        lab = np.where(lab < 0, np.arange(idx.size) + 10_000_000, lab)
        uniq = {x: i for i, x in enumerate(np.unique(lab))}
        labels[idx] = [uniq[x] for x in lab]
    else:
        lab = agglomerative_labels(
            np.zeros((u, 1), dtype=np.float32), centers,
            eps=eps, pos_w=1.0,
        )
        labels = lab
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--resume", default=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_center_offset_train_3000/checkpoints/"
        "model_step_003000.safetensors"
    ))
    parser.add_argument("--label", default="center_offset")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument("--lsm_manifest", default=(
        "data/scannet_prompt/lsm_instance_eval_manifest.json"
    ))
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=15)
    parser.add_argument("--method", choices=("dbscan", "agg"), default="dbscan")
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min_samples", type=int, default=1)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    opt = config_defaults["eval_scannet_lsm_instance"]
    opt.model_type = args.model_type
    opt.resume = args.resume
    opt.workspace = args.workspace
    opt.experiment_name = Path(args.workspace).name
    opt.instance_group_num_groups = int(args.num_groups)
    opt.num_input_views = int(args.num_input_views)
    opt.num_views = int(args.num_views)
    opt.dataset_kwargs = {
        **dict(opt.dataset_kwargs or {}),
        "lsm_manifest_path": str(ROOT / args.lsm_manifest),
    }

    class _A:
        resume = args.resume
        workspace = args.workspace
        num_groups = args.num_groups
        num_input_views = args.num_input_views
        num_views = args.num_views
        lsm_manifest = args.lsm_manifest
        max_scenes = 0
        model_type = args.model_type
        backbone_resume = ""
        instance_branch_cluster_eps = None
        instance_branch_cluster_pos_weight = None
        instance_branch_void_fg_share = None

    _load_checkpoint_arch(_A(), opt)
    ca.patch_model_forward()
    model = ca.build_frozen_model(args.resume, opt)
    br = model.instance_branch
    renderer = br.renderer
    _, loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_scene: dict[str, dict] = {}
    t0 = time.time()
    for i, data in enumerate(loader):
        if args.max_scenes > 0 and i >= args.max_scenes:
            break
        data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in data.items()}
        scene_name = str(data["scene_name"][0])
        ex = ca.extract_scene(model, data)
        centers = br.last_center_pred[0].float().cpu().numpy()  # [U,3]
        pos_norm = ex["pos"]
        p_u = ex["p_u"]
        bg = ex["bg_idx"]
        fg = np.ones(ex["u_count"], dtype=bool)
        if bg >= 0:
            fg = (1.0 - p_u[:, bg]) > 0.05
        labels = center_labels(
            centers, fg, args.method, args.eps, args.min_samples
        )
        # diagnostic: predicted-center same/diff distance
        diag = {}
        if fg.sum() >= 10 and bg >= 0:
            cf = centers[fg]
            pff = p_u[fg]
            fg_cols = [c for c in range(pff.shape[1]) if c != bg]
            pf = pff[:, fg_cols]
            pf = pf / np.maximum(pf.sum(1, keepdims=True), 1e-6)
            dom = pf.argmax(1)
            n = len(cf)
            r = np.arange(n)
            same_mask = (
                (dom[:, None] == dom[None, :])
                & (r[:, None] < r[None, :])
            )
            diff_mask = (dom[:, None] != dom[None, :]) & (r[:, None] < r[None, :])
            d = np.linalg.norm(cf[:, None, :] - cf[None, :, :], axis=-1)
            diag["center_same_dist"] = (
                float(d[same_mask].mean()) if same_mask.any() else 0.0
            )
            diag["center_diff_dist"] = (
                float(d[diff_mask].mean()) if diff_mask.any() else 0.0
            )
        # render + score (same block as cluster_affinity eval)
        void_fg_share = float(getattr(opt, "instance_branch_void_fg_share", 0.5))
        fg_share_u = (
            1.0 - p_u[:, bg] if bg >= 0
            else np.ones(ex["u_count"], dtype=np.float32)
        )
        cluster_list = []
        for c in np.unique(labels):
            if c < 0:
                continue
            sel = labels == c
            if float(fg_share_u[sel].mean()) >= void_fg_share:
                cluster_list.append(int(c))
        if not cluster_list:
            per_scene[scene_name] = {
                "ap25": 0.0, "ap50": 0.0, "ap_mean": 0.0,
                "num_pred": 0, "clusters": 0, "num_gt": 0,
                "psnr": 0.0, **diag,
            }
            continue
        used = {c: j for j, c in enumerate(cluster_list)}
        num = len(cluster_list)
        onehot = np.zeros((ex["u_count"], num + 1), dtype=np.float32)
        for uu in range(ex["u_count"]):
            c = int(labels[uu])
            if c in used:
                onehot[uu, used[c]] = 1.0
        onehot[:, num] = 1.0 - onehot.sum(1)
        a3 = ex["a"][0].cpu().numpy()
        unit_probs_t = onehot.reshape(ex["token_count"], ex["k"], num + 1)
        group_probs = np.einsum(
            "tpk,tkl->tpl", a3, unit_probs_t
        ).reshape(ex["n_gs"], num + 1)
        gpt = torch.from_numpy(group_probs).unsqueeze(0).cuda()
        with torch.inference_mode():
            render = renderer.render_feature_channels(
                ex["gaussians"], gpt, ex["cam_view"],
                intrinsics=ex["intrinsics"],
            )
        rc = render["images_pred"].cpu().numpy()[0] / (
            render["alphas_pred"].cpu().numpy()[0] + 1e-5
        )
        rp = rc / np.maximum(rc.sum(1, keepdims=True), 1e-6)
        pred_masks, pred_scores, pred_ids, gt_masks, gt_ids = [], [], [], [], []
        gt_maps = data["instance_label_output"][0].cpu().numpy()
        for v in range(rp.shape[0]):
            image_id = f"{scene_name}:b0"
            masks, scores = masks_from_group_probs(
                rp[v], void_channel=num, min_mask_area=1
            )
            masks, scores = masks[:100], scores[:100]
            gts = gt_masks_from_instance_map(gt_maps[v], min_mask_area=1)
            pred_masks.extend(masks)
            pred_scores.extend(scores)
            pred_ids.extend([image_id] * len(masks))
            gt_masks.extend(gts)
            gt_ids.extend([image_id] * len(gts))
        res = instance_ap(
            pred_masks, pred_scores, gt_masks,
            thresholds=(0.25, 0.5, 0.75), vectorized=True,
            pred_image_ids=pred_ids, gt_image_ids=gt_ids,
        )
        coco = instance_ap(
            pred_masks, pred_scores, gt_masks,
            thresholds=tuple(t / 100 for t in range(50, 100, 5)),
            vectorized=True, pred_image_ids=pred_ids, gt_image_ids=gt_ids,
        )
        per_scene[scene_name] = {
            "ap25": res["ap_25"],
            "ap50": res["ap_50"],
            "ap_mean": coco["ap_mean"],
            "num_pred": len(pred_masks),
            "num_gt": len(gt_masks),
            "clusters": num,
            "psnr": float(out_psnr) if (out_psnr := _psnr(data, model)) else 0.0,
            **diag,
        }
        print(
            f"[center-eval] {scene_name}: AP50={per_scene[scene_name]['ap50']:.3f} "
            f"clusters={num} pred={len(pred_masks)} gt={len(gt_masks)} "
            f"({i + 1}/{min(args.max_scenes or 40, 40)})",
            flush=True,
        )

    def _mean(keys, entries):
        return {
            k: float(np.mean([entries[s][k] for s in entries])) if entries else 0.0
            for k in keys
        }

    summary = {
        "num_scenes": len(per_scene),
        **_mean(
            [
                "ap25", "ap50", "ap_mean", "num_pred", "clusters", "num_gt",
                "center_same_dist", "center_diff_dist",
            ],
            per_scene,
        ),
    }
    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "fixed": vars(args),
        "summary": summary,
        "per_scene": per_scene,
    }
    (out_dir / "center_eval.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def _psnr(data, model) -> float:
    """PSNR of the frozen model on this scene (constant ~19.8)."""
    try:
        with torch.inference_mode():
            out = model(data, compute_quality_metrics=False)
        return float(out["psnr"])
    except Exception:
        return 0.0


if __name__ == "__main__":
    main()
