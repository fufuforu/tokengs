"""Offline ablation: DINO unit features + Agglomerative clustering.

Frozen checkpoint (unit-shaping + patch @2000), LSM 40 scenes, no training.
Replaces the GT-prototype matching of ``dino_dense_feature_oracle.py`` with
the standard Agglomerative (average linkage) used by the current pipeline:

  feature = [L2-normalized DINO unit feature, pw * scene-normalized unit 3D
  center], distance threshold eps.

Sweeps eps x position weight.  Everything else (DINO projection, frozen
GS->unit assignment, frozen geometry, rendering, LSM evaluation) is
identical to the oracle script.  Reference points: current embedding +
Agglomerative AP50=0.279, DINO GT-prototype oracle AP50=0.534.
"""

from __future__ import annotations

import argparse
import json
import os
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
from tokengs.models.instance_group_head import (
    TokenLocalUnitGrouping,
    _project_dense_features,
)
from tokengs.options import config_defaults
from tokengs.utils.instance_ap import (
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)

from ablate_unit_clustering import _gs_level_stats, _max_iou_recall  # noqa: E402
from dino_dense_feature_oracle import (  # noqa: E402
    _dino_frame_hw,
    _rescale_intrinsics,
    dinov2_patch_features,
    load_dinov2,
)
from eval_instance_lsm_protocol import (  # noqa: E402
    _LocalAccelerator,
    _audit_lsm_manifest,
    _load_checkpoint_arch,
)


def _agg_labels(x: np.ndarray, eps: float) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage

    z = linkage(x, method="average")
    return fcluster(z, t=eps, criterion="distance").astype(np.int64) - 1


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
        "--workspace", default="workspace/dino_agg_ablation"
    )
    parser.add_argument("--label", default="dino_agg")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument("--lsm_manifest", default=(
        "data/scannet_prompt/lsm_instance_eval_manifest.json"
    ))
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=15)
    parser.add_argument(
        "--eps_list", default="0.3,0.4,0.5,0.6,0.7,0.9",
        help="Comma-separated Agglomerative distance thresholds.",
    )
    parser.add_argument(
        "--pos_w_list", default="0.0,1.0,2.0",
        help="Comma-separated position weights.",
    )
    args = parser.parse_args()

    if args.num_input_views != 8 or args.num_views != 15:
        raise ValueError("LSM protocol requires 8 context + 7 target views")
    manifest_audit = _audit_lsm_manifest(str(ROOT / args.lsm_manifest))

    eps_list = [float(v) for v in args.eps_list.split(",") if v]
    pw_list = [float(v) for v in args.pos_w_list.split(",") if v]
    variants: list[dict] = [
        {"name": f"dino_agg_eps{eps:.1f}_pw{pw:.0f}",
         "eps": eps, "pos_w": pw}
        for eps in eps_list
        for pw in pw_list
    ]
    print(f"[dino-agg] {len(variants)} variants")

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

    device = torch.device("cuda")
    dinov2 = load_dinov2(device)

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
            self._dino_agg_buffers = {
                "a": a.detach().float(),
                "pos_norm": ((center - scene_center) / scene_scale)
                .detach().float(),
                "means": means.detach().float(),
                "gaussians": gaussians.detach().float(),
                "cam_view": model_input.decoder.cam_view.detach().float(),
                "intrinsics": model_input.decoder.intrinsics.detach().float(),
                "cam_to_world_input": data["cam_to_world_input"].detach().float(),
                "intrinsics_input": data["intrinsics_input"].detach().float(),
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
                f"[dino-agg] frozen-backbone eval: loaded "
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
        buf = br._dino_agg_buffers
        batch_size = 1
        token_count = buf["a"].shape[1]
        k = br.units_per_token
        p = buf["a"].shape[2]
        u_count = token_count * k
        n_gs = token_count * p
        a3 = buf["a"][0].reshape(n_gs, k).cpu().numpy()
        a4 = a3.reshape(token_count, p, k)
        pos_norm = buf["pos_norm"][0].cpu().numpy()
        means_np = buf["means"].reshape(batch_size, n_gs, 3)

        # DINO unit features (identical to the oracle script).
        dino_grid = dinov2_patch_features(
            dinov2, data["images_input"], device
        )
        with torch.inference_mode():
            fused, _ = _project_dense_features(
                means_np,
                dino_grid,
                buf["cam_to_world_input"],
                _rescale_intrinsics(
                    buf["intrinsics_input"],
                    tuple(opt.img_size),
                    _dino_frame_hw(),
                ),
                _dino_frame_hw(),
            )
        gs_dino = fused[0].cpu().numpy()
        unit_dino = np.einsum(
            "tpk,tpd->tkd", a4, gs_dino.reshape(token_count, p, -1)
        )
        mass = np.einsum("tpk->tk", a4).clip(min=1e-6)
        unit_dino = (unit_dino / mass[:, :, None]).reshape(u_count, -1)
        unit_dino = unit_dino / np.maximum(
            np.linalg.norm(unit_dino, axis=-1, keepdims=True), 1e-8
        )

        # GT instance labels + foreground share (for void filtering).
        gs_inst, _, _ = br._pseudo_gs_labels(means_np, data, opt)
        gs_inst = gs_inst[0].cpu().numpy()
        p_u = br.last_unit_pu[0].cpu().numpy()
        bg_idx = int(br.last_unit_bg_idx)
        fg_share_u = (
            1.0 - p_u[:, bg_idx] if bg_idx >= 0
            else np.ones(u_count, dtype=np.float32)
        )
        fg = fg_share_u > 0.05

        gaussians = buf["gaussians"].float()
        cam_view = buf["cam_view"].float()
        intrinsics = buf["intrinsics"].float()

        for variant in variants:
            pw = float(variant["pos_w"])
            if os.environ.get("DINO_AGG_DEBUG"):
                print(
                    f"  [dbg] {scene_name} variant={variant['name']}",
                    flush=True,
                )
            feat = np.concatenate(
                [unit_dino, pw * pos_norm], axis=-1
            ).astype(np.float32) if pw > 0 else unit_dino.astype(np.float32)
            labels = _agg_labels(feat, float(variant["eps"]))
            if os.environ.get("DINO_AGG_DEBUG"):
                print(
                    f"  [dbg]   clusters(raw)={int(labels.max()) + 1} "
                    f"nan={bool(np.isnan(labels).any())}",
                    flush=True,
                )

            cluster_list = []
            for c in np.unique(labels):
                if c < 0:
                    continue
                sel = labels == c
                if fg_share_u[sel].mean() >= void_fg_share:
                    cluster_list.append(int(c))
            if not cluster_list:
                per_scene.setdefault(scene_name, {})[variant["name"]] = {
                    "ap25": 0.0, "ap50": 0.0, "ap75": 0.0, "ap_mean": 0.0,
                    "num_pred": 0, "num_gt": 0, "num_clusters": 0,
                    "recall_025": 0.0, "recall_05": 0.0,
                    "merging_frac": 0.0, "mean_instances_per_cluster": 0.0,
                    "fragmentation_frac": 0.0,
                    "mean_clusters_per_instance": 0.0,
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
            gs_stats = _gs_level_stats(group_probs, gs_inst, num)
            per_scene.setdefault(scene_name, {})[variant["name"]] = {
                "ap25": results["ap_25"],
                "ap50": results["ap_50"],
                "ap75": results["ap_75"],
                "ap_mean": coco_ap["ap_mean"],
                "num_pred": len(pred_masks),
                "num_gt": len(gt_masks),
                "num_clusters": num,
                "recall_025": recall[0.25],
                "recall_05": recall[0.5],
                **gs_stats,
            }
        elapsed = time.time() - t_start
        if os.environ.get("DINO_AGG_DEBUG"):
            print(
                f"  [dbg] per_scene[{scene_name}] variants="
                f"{list(per_scene.get(scene_name, {}).keys())}",
                flush=True,
            )
        print(
            f"[dino-agg] {scene_name} done "
            f"({i + 1}/{min(args.max_scenes or 40, 40)}) "
            f"elapsed={elapsed:.0f}s"
        )

    def _mean(keys: list[str], variant_name: str) -> dict:
        out = {}
        for key in keys:
            vals = []
            for s in per_scene:
                try:
                    vals.append(per_scene[s][variant_name][key])
                except KeyError:
                    print(
                        f"  [dbg-keyerr] scene={s} key={key} "
                        f"present={key in per_scene[s].get(variant_name, {})} "
                        f"scene_type={type(per_scene[s]).__name__} "
                        f"entry_type={type(per_scene[s]).__name__}",
                        flush=True,
                    )
                    raise
            out[key] = float(np.mean(vals)) if vals else 0.0
        return out

    if os.environ.get("DINO_AGG_DEBUG"):
        for s in per_scene:
            print(
                f"  [dbg-summary] {s}: "
                f"{list(per_scene[s].keys())} "
                f"first={list(per_scene[s][list(per_scene[s].keys())[0]].keys()) if per_scene[s] else None}",
                flush=True,
            )

    summary = {}
    if os.environ.get("DINO_AGG_DEBUG"):
        for s in per_scene:
            e = per_scene[s]
            print(
                f"  [dbg-summary2] {s} type={type(e).__name__} "
                f"variants={list(e.keys()) if isinstance(e, dict) else '?'}",
                flush=True,
            )
    for variant in variants:
        name = variant["name"]
        scene_entries = {
            s: per_scene[s][name] for s in per_scene if name in per_scene[s]
        }
        summary[name] = {
            "variant": variant,
            "num_scenes": len(scene_entries),
            **_mean(
                [
                    "ap25", "ap50", "ap75", "ap_mean",
                    "num_pred", "num_gt", "num_clusters",
                    "recall_025", "recall_05",
                    "merging_frac", "mean_instances_per_cluster",
                    "fragmentation_frac", "mean_clusters_per_instance",
                ],
                name,
            ),
        }
    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "protocol": manifest_audit,
        "reference": {
            "current_embedding_agg_ap50": 0.279,
            "dino_gt_prototype_oracle_ap50": 0.534,
        },
        "variants": summary,
    }
    (out_dir / "dino_agg_ablation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "dino_agg_per_scene.json").write_text(
        json.dumps(per_scene, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[dino-agg] summary:")
    print(
        f"{'variant':28s} {'AP25':>6s} {'AP50':>6s} {'AP':>6s} "
        f"{'pred/gt':>10s} {'recall.5':>8s} {'merge':>6s} {'frag':>6s}"
    )
    for name, entry in summary.items():
        print(
            f"{name:28s} {entry['ap25']:6.3f} {entry['ap50']:6.3f} "
            f"{entry['ap_mean']:6.3f} "
            f"{int(entry['num_pred'])}/{int(entry['num_gt']):<4d} "
            f"{entry['recall_05']:8.3f} {entry['merging_frac']:6.2f} "
            f"{entry['fragmentation_frac']:6.2f}"
        )


if __name__ == "__main__":
    main()
