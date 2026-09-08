"""Offline oracle: GT instance count + maxclust on DINO-trained embeddings.

Uses the trained `unit-shaping + patch + DINO` checkpoint (embedding is
normalize(concat(unit_feat, unit_img, proj(dino_unit)))).  The clustering is
Agglomerative with maxclust = GT instance count (per scene), isolating the
"instance-count estimation" link from the embedding quality.  Everything
else (frozen geometry, rendering, LSM evaluation) is identical.

Reference points: DINO-trained + eps=0.5 = 0.324; per-scene best-eps oracle
= 0.369; DINO GT-prototype oracle = 0.534.
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

from ablate_unit_clustering import _gs_level_stats, _max_iou_recall  # noqa: E402
from eval_instance_lsm_protocol import (  # noqa: E402
    _LocalAccelerator,
    _audit_lsm_manifest,
    _load_checkpoint_arch,
)


def _maxclust_labels(
    x: np.ndarray, k: int, linkage: str = "average"
) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage as slink

    z = slink(x, method=linkage)
    return fcluster(z, t=int(k), criterion="maxclust").astype(np.int64) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=(
            "workspace/semantic_v6_unit_shaping_img_dino_train_3000/"
            "checkpoints/model_step_003000.safetensors"
        ),
    )
    parser.add_argument("--workspace", default="workspace/dino_maxclust_gtcount")
    parser.add_argument("--label", default="dino_maxclust_gt")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument("--lsm_manifest", default=(
        "data/scannet_prompt/lsm_instance_eval_manifest.json"
    ))
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=15)
    parser.add_argument("--linkage", default="average")
    args = parser.parse_args()

    if args.num_input_views != 8 or args.num_views != 15:
        raise ValueError("LSM protocol requires 8 context + 7 target views")
    manifest_audit = _audit_lsm_manifest(str(ROOT / args.lsm_manifest))

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
    _load_checkpoint_arch(args, opt)
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    _orig_forward = TokenLocalUnitGrouping._forward_unit_embedding

    def _patched_forward(
        self, a, unit_feat, unit_center, means, gaussians, data, opt_inner,
        training, model_input, batch_size, token_count, n_gs, p,
        dense_features=None,
    ):
        out = _orig_forward(
            self, a, unit_feat, unit_center, means, gaussians, data,
            opt_inner, training, model_input, batch_size, token_count, n_gs,
            p, dense_features,
        )
        if not training:
            u_count = token_count * self.units_per_token
            center = unit_center.reshape(batch_size, u_count, 3).float()
            scene_center = center.mean(dim=1, keepdim=True)
            scene_scale = (
                (center - scene_center).square().mean(dim=(1, 2), keepdim=True)
                .sqrt().clamp_min(1e-3)
            )
            self._mc_buffers = {
                "a": a.detach().float(),
                "means": means.detach().float(),
                "pos_norm": ((center - scene_center) / scene_scale)
                .detach().float(),
                "gaussians": gaussians.detach().float(),
                "cam_view": model_input.decoder.cam_view.detach().float(),
                "intrinsics": model_input.decoder.intrinsics.detach().float(),
            }
        return out

    TokenLocalUnitGrouping._forward_unit_embedding = _patched_forward

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
                f"[dino-maxclust] frozen-backbone eval: loaded "
                f"{len(loadable)} backbone keys from {backbone_path}"
            )
    model.eval()
    model = model.cuda()
    br = model.instance_branch
    renderer = br.renderer
    render_scale = float(getattr(opt, "instance_group_render_scale", 1.0))
    void_fg_share = float(getattr(opt, "instance_branch_void_fg_share", 0.5))
    min_pred_pixels = 1
    min_gt_pixels = 1
    max_pred = 100

    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())

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
        buf = br._mc_buffers
        batch_size = 1
        token_count = buf["a"].shape[1]
        k = br.units_per_token
        p = buf["a"].shape[2]
        u_count = token_count * k
        n_gs = token_count * p
        a4 = buf["a"][0].reshape(token_count, p, k).cpu().numpy()
        e = br.last_unit_embeddings[0].cpu().numpy()  # [U,D]
        pos_norm = buf["pos_norm"][0].cpu().numpy()
        gaussians = buf["gaussians"].float()
        cam_view = buf["cam_view"].float()
        intrinsics = buf["intrinsics"].float()
        means_np = buf.get("means")
        if means_np is None:
            means_np = (
                buf["a"].new_zeros(batch_size, n_gs, 3)
            )
        p_u = br.last_unit_pu[0].cpu().numpy()
        bg_idx = int(br.last_unit_bg_idx)
        fg_share_u = (
            1.0 - p_u[:, bg_idx] if bg_idx >= 0
            else np.ones(u_count, dtype=np.float32)
        )
        # GT instance count from the 15-view projection (oracle count).
        gs_gt, _, _ = br._pseudo_gs_labels(
            buf["means"].reshape(batch_size, n_gs, 3), data, opt
        )
        gs_gt = gs_gt[0].cpu().numpy()
        gt_count = int(len(np.unique(gs_gt[gs_gt > 0])))

        feat = np.concatenate([e, pos_norm], axis=-1).astype(np.float32)
        labels = _maxclust_labels(
            feat, max(1, gt_count), linkage=args.linkage
        )

        cluster_list = []
        for c in np.unique(labels):
            if c < 0:
                continue
            sel = labels == c
            if fg_share_u[sel].mean() >= void_fg_share:
                cluster_list.append(int(c))
        if not cluster_list:
            per_scene[scene_name] = {
                "ap25": 0.0, "ap50": 0.0, "ap75": 0.0, "ap_mean": 0.0,
                "num_pred": 0, "num_gt": 0, "num_clusters": 0,
                "gt_count": gt_count,
                "recall_025": 0.0, "recall_05": 0.0,
                "merging_frac": 0.0, "mean_instances_per_cluster": 0.0,
                "fragmentation_frac": 0.0, "mean_clusters_per_instance": 0.0,
            }
            continue
        used = {c: j for j, c in enumerate(cluster_list)}
        num = len(cluster_list)
        onehot = np.zeros((u_count, num + 1), dtype=np.float32)
        for uu in range(u_count):
            c = int(labels[uu])
            onehot[uu, used[c] if c in used else num] = 1.0
        unit_probs_t = onehot.reshape(token_count, k, num + 1)
        group_probs = np.einsum(
            "tpk,tkl->tpl", a4, unit_probs_t
        ).reshape(n_gs, num + 1)
        group_probs_t = torch.from_numpy(group_probs).unsqueeze(0).cuda()
        with torch.inference_mode():
            render = renderer.render_feature_channels(
                gaussians, group_probs_t, cam_view,
                intrinsics=intrinsics, opacity_scale=render_scale,
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
        gt_maps = data["instance_label_output"][0].cpu().numpy()
        for v in range(view_count):
            image_id = f"{scene_name}:b0"
            masks, scores = masks_from_group_probs(
                rendered_probs[v],
                void_channel=num,
                min_mask_area=min_pred_pixels,
            )
            masks = masks[:max_pred]
            scores = scores[:max_pred]
            gts = gt_masks_from_instance_map(
                gt_maps[v], min_mask_area=min_gt_pixels
            )
            pred_masks.extend(masks)
            pred_scores.extend(scores)
            pred_image_ids.extend([image_id] * len(masks))
            gt_masks.extend(gts)
            gt_image_ids.extend([image_id] * len(gts))
        results = instance_ap(
            pred_masks, pred_scores, gt_masks,
            thresholds=(0.25, 0.5, 0.75),
            vectorized=True,
            pred_image_ids=pred_image_ids,
            gt_image_ids=gt_image_ids,
        )
        coco_ap = instance_ap(
            pred_masks, pred_scores, gt_masks,
            thresholds=tuple(t / 100 for t in range(50, 100, 5)),
            vectorized=True,
            pred_image_ids=pred_image_ids,
            gt_image_ids=gt_image_ids,
        )
        recall = _max_iou_recall(pred_masks, gt_masks)
        gs_stats = _gs_level_stats(group_probs, gs_gt, num)
        per_scene[scene_name] = {
            "ap25": results["ap_25"],
            "ap50": results["ap_50"],
            "ap75": results["ap_75"],
            "ap_mean": coco_ap["ap_mean"],
            "num_pred": len(pred_masks),
            "num_gt": len(gt_masks),
            "num_clusters": num,
            "gt_count": gt_count,
            "recall_025": recall[0.25],
            "recall_05": recall[0.5],
            **gs_stats,
        }
        elapsed = time.time() - t_start
        print(
            f"[dino-maxclust] {scene_name} done "
            f"({i + 1}/{min(args.max_scenes or 40, 40)}) "
            f"AP50={results['ap_50']:.3f} gt={gt_count} "
            f"clusters={num} elapsed={elapsed:.0f}s"
        )

    def _mean(keys: list[str]) -> dict:
        out = {}
        for key in keys:
            vals = [per_scene[s][key] for s in per_scene]
            out[key] = float(np.mean(vals)) if vals else 0.0
        return out

    summary = {
        "num_scenes": len(per_scene),
        "linkage": args.linkage,
        **_mean(
            [
                "ap25", "ap50", "ap75", "ap_mean",
                "num_pred", "num_gt", "num_clusters", "gt_count",
                "recall_025", "recall_05",
                "merging_frac", "mean_instances_per_cluster",
                "fragmentation_frac", "mean_clusters_per_instance",
            ]
        ),
        "reference": {
            "dino_eps05_ap50": 0.324,
            "per_scene_best_eps_oracle_ap50": 0.369,
            "dino_gt_prototype_oracle_ap50": 0.534,
        },
    }
    (out_dir / "dino_maxclust_gtcount.json").write_text(
        json.dumps(
            {"summary": summary, "per_scene": per_scene},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\n[dino-maxclust] summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
