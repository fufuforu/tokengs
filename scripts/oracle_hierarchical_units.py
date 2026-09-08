"""Offline hierarchical local-unit oracle (8 -> 16 -> 32 units/token).

Keeps the current learned 8-unit formation and Gaussian geometry fixed.  For
each original token the 8 learned units are refined by spatial K-means on the
3D Gaussian centers *inside each learned unit* (2 or 4 sub-units -> 16 or 32
units per token).  Every sub-unit is assigned its internal majority GT
instance label as the oracle identity; sub-units sharing a GT label merge
into one predicted instance.  Masks are rendered through the ORIGINAL frozen
Gaussian geometry and scored with the exact LSM protocol used everywhere
(masks_from_group_probs + instance_ap).

No training, no checkpoint modification.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.instance_group_head import TokenLocalUnitGrouping
from tokengs.options import config_defaults
from tokengs.utils.instance_ap import (
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)

from ablate_unit_clustering import (  # noqa: E402
    _gs_level_stats,
    _max_iou_recall,
)
from eval_instance_lsm_protocol import (  # noqa: E402
    _LocalAccelerator,
    _audit_lsm_manifest,
    _load_checkpoint_arch,
)


def _split_unit_gs(
    means3: np.ndarray, gs_idx: np.ndarray, sub: int
) -> list[np.ndarray]:
    """Spatial K-means of a learned unit's GS centers into ``sub`` groups."""
    from scipy.cluster.vq import kmeans2

    pts = means3[gs_idx]
    n = len(gs_idx)
    if n <= sub:
        return [gs_idx[i : i + 1] for i in range(n)]
    _, labels = kmeans2(
        pts.astype(np.float32), sub, minit="++", iter=100, seed=0
    )
    return [gs_idx[labels == c] for c in range(sub)]


def _build_hier_labels(
    a3: np.ndarray, means3: np.ndarray, gs_gt3: np.ndarray, sub_per_unit: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gs_cluster [T*P], cluster_gt [C], sub_purity).

    GS are hard-assigned to their argmax learned unit, then each learned
    unit's GS are spatially K-means'd into ``sub_per_unit`` sub-units (or
    fewer when the unit has fewer GS).  Each sub-unit's oracle identity is
    its internal majority GT instance; sub-units sharing a GT label merge
    into one cluster.  Background-majority sub-units go to void (-1).
    """
    t, p, k = a3.shape
    n_gs = t * p
    uid = a3.argmax(axis=-1)  # [T,P] hard learned-unit id
    gs_cluster = np.full(n_gs, -1, dtype=np.int64)
    sub_gt: list[int] = []
    purity_list: list[tuple[float, int]] = []  # (purity, n_gs)
    cluster_map: dict[int, int] = {}
    for tk in range(t):
        for kk in range(k):
            gs_idx = np.where(uid[tk] == kk)[0]
            if gs_idx.size == 0:
                continue
            subs = _split_unit_gs(means3[tk], gs_idx, sub_per_unit)
            for sub in subs:
                if sub.size == 0:
                    continue
                sub_gt_ids = gs_gt3[tk][sub]
                counts = np.bincount(sub_gt_ids)
                dom = int(counts.argmax())
                purity = float(counts[dom] / max(1, sub_gt_ids.size))
                if dom == 0:
                    continue  # background-majority sub-unit -> void
                cid = cluster_map.setdefault(dom, len(cluster_map))
                gs_cluster[tk * p + sub] = cid
                sub_gt.append(dom)
                purity_list.append((purity, int(sub.size)))
    cluster_gt = np.asarray(
        [gt for gt, _ in sorted(cluster_map.items(), key=lambda kv: kv[1])],
        dtype=np.int64,
    )
    if purity_list:
        w = np.asarray([x[1] for x in purity_list], dtype=np.float64)
        purity = float(
            np.average(
                [x[0] for x in purity_list], weights=w
            )
        )
    else:
        purity = 0.0
    return gs_cluster, cluster_gt, purity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=(
            "workspace/semantic_v6_unit_shaping_img_train_3000/"
            "checkpoints/model_step_002000.safetensors"
        ),
    )
    parser.add_argument(
        "--workspace",
        default="workspace/oracle_hierarchical_units",
    )
    parser.add_argument("--label", default="unit_shaping_img_2000")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument("--lsm_manifest", default=(
        "data/scannet_prompt/lsm_instance_eval_manifest.json"
    ))
    parser.add_argument("--max_scenes", type=int, default=0)
    args = parser.parse_args()

    manifest_audit = _audit_lsm_manifest(str(ROOT / args.lsm_manifest))
    opt = config_defaults["eval_scannet_lsm_instance"]
    opt.model_type = args.model_type
    opt.resume = args.resume
    opt.workspace = args.workspace
    opt.experiment_name = Path(args.workspace).name
    opt.num_input_views = 8
    opt.num_views = 15
    opt.dataset_kwargs = {
        **dict(opt.dataset_kwargs or {}),
        "lsm_manifest_path": str(ROOT / args.lsm_manifest),
    }
    _load_checkpoint_arch(args, opt)
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    _orig = TokenLocalUnitGrouping._forward_unit_embedding

    def _patched(
        self,
        a,
        unit_feat,
        unit_center,
        means,
        gaussians,
        data,
        opt_inner,
        training,
        model_input,
        batch_size,
        token_count,
        n_gs,
        p,
        dense_features=None,
    ):
        out = _orig(
            self,
            a,
            unit_feat,
            unit_center,
            means,
            gaussians,
            data,
            opt_inner,
            training,
            model_input,
            batch_size,
            token_count,
            n_gs,
            p,
            dense_features,
        )
        if not training:
            self._hier_buf = {
                "a": a.detach().float(),
                "means": means.detach().float(),
                "gaussians": gaussians.detach().float(),
                "cam_view": model_input.decoder.cam_view.detach().float(),
                "intrinsics": model_input.decoder.intrinsics.detach().float(),
            }
        return out

    TokenLocalUnitGrouping._forward_unit_embedding = _patched

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
                f"[hier-oracle] frozen-backbone eval: loaded "
                f"{len(loadable)} backbone keys from {backbone_path}"
            )
    model.eval()
    model = model.cuda()
    br = model.instance_branch
    renderer = br.renderer
    render_scale = float(getattr(opt, "instance_group_render_scale", 1.0))

    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())

    granularities = {
        "units8": 1,
        "units16": 2,
        "units32": 4,
    }
    per_scene: dict[str, dict] = {}
    t_start = time.time()
    for i, data in enumerate(test_loader):
        if args.max_scenes > 0 and i >= args.max_scenes:
            break
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        scene_name = str(data["scene_name"][0])
        with torch.inference_mode():
            model(data, compute_quality_metrics=False)
        buf = br._hier_buf
        T, P, K = buf["a"].shape[1], buf["a"].shape[2], br.units_per_token
        n_gs = T * P
        a3 = buf["a"][0].reshape(T, P, K).cpu().numpy()
        means3 = buf["means"][0].reshape(T, P, 3).cpu().numpy()
        gaussians = buf["gaussians"].float()
        cam_view = buf["cam_view"].float()
        intrinsics = buf["intrinsics"].float()
        gs_gt, _, _ = br._pseudo_gs_labels(
            torch.from_numpy(means3.reshape(1, n_gs, 3)).cuda(), data, opt
        )
        gs_gt3 = gs_gt[0].cpu().numpy().reshape(T, P)
        gt_maps = data["instance_label_output"][0].cpu().numpy()

        for name, sub_per in granularities.items():
            gs_cluster, cluster_gt, purity = _build_hier_labels(
                a3, means3, gs_gt3, sub_per
            )
            n_clusters = len(cluster_gt)
            onehot = np.zeros((n_gs, n_clusters + 1), dtype=np.float32)
            valid = gs_cluster >= 0
            onehot[np.arange(n_gs)[valid], gs_cluster[valid]] = 1.0
            # Background GS go to the void channel (zero rows would inflate
            # foreground masks through the per-channel normalization).
            onehot[np.arange(n_gs)[~valid], n_clusters] = 1.0
            if n_clusters == 0:
                onehot[:, 0] = 1.0
            probs_t = torch.from_numpy(onehot).unsqueeze(0).cuda()
            with torch.inference_mode():
                render = renderer.render_feature_channels(
                    gaussians,
                    probs_t,
                    cam_view,
                    intrinsics=intrinsics,
                    opacity_scale=render_scale,
                )
            rendered_channels = (
                render["images_pred"].cpu().numpy()[0]
                / (render["alphas_pred"].cpu().numpy()[0] + 1e-5)
            )
            rendered_probs = rendered_channels / np.maximum(
                rendered_channels.sum(axis=1, keepdims=True), 1e-6
            )
            view_count = rendered_probs.shape[0]
            pred_masks, pred_scores, pred_image_ids = [], [], []
            gt_masks, gt_image_ids = [], []
            for v in range(view_count):
                image_id = f"{scene_name}:b0"
                masks, scores = masks_from_group_probs(
                    rendered_probs[v],
                    void_channel=n_clusters,
                    min_mask_area=1,
                )
                masks = masks[:100]
                scores = scores[:100]
                gts = gt_masks_from_instance_map(gt_maps[v], min_mask_area=1)
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
            recall = _max_iou_recall(pred_masks, gt_masks)
            gs_stats = _gs_level_stats(
                onehot, gs_gt3.reshape(-1), n_clusters
            )
            per_scene.setdefault(scene_name, {})[name] = {
                "ap25": results["ap_25"],
                "ap50": results["ap_50"],
                "ap75": results["ap_75"],
                "ap_mean": coco_ap["ap_mean"],
                "num_pred": len(pred_masks),
                "num_gt": len(gt_masks),
                "num_clusters": n_clusters,
                "recall_05": recall[0.5],
                "purity": purity,
                **gs_stats,
            }
        elapsed = time.time() - t_start
        print(
            f"[hier-oracle] {scene_name} done "
            f"({i + 1}/{min(args.max_scenes or 40, 40)}) elapsed={elapsed:.0f}s"
        )

    summary = {}
    for name in granularities:
        entries = {s: per_scene[s][name] for s in per_scene}
        summary[name] = {
            "num_scenes": len(entries),
            **{
                key: float(
                    np.mean([entries[s][key] for s in entries])
                )
                for key in (
                    "ap25",
                    "ap50",
                    "ap75",
                    "ap_mean",
                    "num_pred",
                    "num_gt",
                    "num_clusters",
                    "recall_05",
                    "purity",
                    "merging_frac",
                    "mean_instances_per_cluster",
                    "fragmentation_frac",
                    "mean_clusters_per_instance",
                )
            },
        }
    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "num_scenes": len(per_scene),
        "protocol": manifest_audit,
        "reference": {
            "learned_unit_oracle_ap50": 0.633,
            "gs_level_oracle_ap50": 0.789,
        },
        "granularities": summary,
    }
    out_json = out_dir / "hierarchical_oracle.json"
    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    per_scene_json = out_dir / "hierarchical_oracle_per_scene.json"
    per_scene_json.write_text(
        json.dumps(per_scene, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[hier-oracle] summary (scene-macro mean):")
    print(
        f"{'granularity':10s} {'AP25':>6s} {'AP50':>6s} {'AP':>6s} "
        f"{'pred/gt':>9s} {'recall.5':>8s} {'purity':>6s} {'merge':>6s} "
        f"{'frag':>6s}"
    )
    for name, entry in summary.items():
        print(
            f"{name:10s} {entry['ap25']:6.3f} {entry['ap50']:6.3f} "
            f"{entry['ap_mean']:6.3f} "
            f"{int(entry['num_pred'])}/{int(entry['num_gt']):<4d} "
            f"{entry['recall_05']:8.3f} {entry['purity']:6.3f} "
            f"{entry['merging_frac']:6.2f} {entry['fragmentation_frac']:6.2f}"
        )
    print(f"[hier-oracle] wrote {out_json}")


if __name__ == "__main__":
    main()
