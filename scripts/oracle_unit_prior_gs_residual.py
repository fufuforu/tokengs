"""Offline diagnostic: unit identity prior + GS-level residual oracle.

For each learned 8-unit the dominant identity is taken from the GT majority
instance (oracle prior).  Inside a unit each GS may be decided as belonging
to that dominant instance or not, using the EXISTING per-GS features:
GS feature (gs_feature_mlp) + patch feature (unit_image_net output) +
relative 3D position + GS scale/opacity.  Variants:

  - unit8_oracle: every GS of the unit goes to the dominant identity
    (reproduces the 8-unit oracle).
  - res_perfect: every GS goes to its true GT instance (GS-level oracle).
  - res_gt_drop: perfect residual decision; GS whose GT != dominant are
    dropped to void (upper bound of the binary residual with a perfect
    classifier).
  - res_feature_loo: leave-one-out prototype residual with the existing
    per-GS features (a realistic, non-trivial feature decision).

Masks are rendered through the ORIGINAL frozen Gaussian geometry and scored
with the exact LSM protocol.  No training, no checkpoint modification.
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


def _unit_dominant(a3: np.ndarray, gs_gt3: np.ndarray) -> np.ndarray:
    """GT-majority dominant instance id per learned unit [T,K]."""
    t, p, k = a3.shape
    uid = a3.argmax(axis=-1)
    dom = np.zeros((t, k), dtype=np.int64)
    for tk in range(t):
        for kk in range(k):
            sel = uid[tk] == kk
            if not sel.any():
                dom[tk, kk] = 0
                continue
            counts = np.bincount(gs_gt3[tk][sel])
            dom[tk, kk] = int(counts.argmax())
    return dom


def _gs_feature(
    f_gs: np.ndarray,
    img_gs: np.ndarray,
    means: np.ndarray,
    gaussians: np.ndarray,
) -> np.ndarray:
    """Per-GS residual feature [N,D]: GS feat + patch feat + pos/attrs."""
    t, p, _ = means.shape
    means3 = means.reshape(t, p, 3)
    anchor = means3.mean(axis=1, keepdims=True)
    scene_scale = (
        ((means3 - anchor) ** 2).mean(axis=(1, 2), keepdims=True) ** 0.5
    ).clip(min=1e-3)
    rel_pos = ((means3 - anchor) / scene_scale).reshape(t * p, 3)
    g14 = gaussians.reshape(t, p, 14)
    log_scale = np.log(
        g14[..., 4:7].reshape(t * p, 3).clip(min=1e-4)
    )
    opacity = g14[..., 3:4].reshape(t * p, 1)
    return np.concatenate(
        [
            f_gs.reshape(t * p, -1),
            img_gs.reshape(t * p, -1),
            rel_pos,
            log_scale,
            opacity,
        ],
        axis=-1,
    )


def _residual_labels(
    a3: np.ndarray,
    gs_gt3: np.ndarray,
    feat: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (gs_cluster [N], cluster_gt [C], kept_gs_frac)."""
    t, p, k = a3.shape
    n_gs = t * p
    uid = a3.argmax(axis=-1)
    dom = _unit_dominant(a3, gs_gt3)
    gs_cluster = np.full(n_gs, -1, dtype=np.int64)
    cluster_map: dict[int, int] = {}
    kept = 0
    if mode == "unit8_oracle":
        for tk in range(t):
            for kk in range(k):
                if dom[tk, kk] == 0:
                    continue
                cid = cluster_map.setdefault(int(dom[tk, kk]), len(cluster_map))
                idx = np.where(uid[tk] == kk)[0]
                gs_cluster[tk * p + idx] = cid
                kept += int(idx.size)
    elif mode == "res_perfect":
        fg_ids = np.unique(gs_gt3[gs_gt3 > 0])
        for gid in fg_ids.tolist():
            cluster_map[int(gid)] = len(cluster_map)
        mask = gs_gt3.reshape(-1) > 0
        gs_cluster[mask] = np.asarray(
            [cluster_map[int(g)] for g in gs_gt3.reshape(-1)[mask]],
            dtype=np.int64,
        )
        kept = int(mask.sum())
    elif mode in ("res_gt_drop", "res_feature_loo"):
        feats = feat.reshape(t, p, -1)
        for tk in range(t):
            for kk in range(k):
                d = int(dom[tk, kk])
                if d == 0:
                    continue
                sel = np.where(uid[tk] == kk)[0]
                if sel.size == 0:
                    continue
                gt_sel = gs_gt3[tk][sel]
                if mode == "res_gt_drop":
                    belong = gt_sel == d
                else:
                    # Leave-one-out prototype residual on existing features.
                    f_loc = feats[tk][sel]
                    f_norm = f_loc / np.maximum(
                        np.linalg.norm(f_loc, axis=1, keepdims=True), 1e-8
                    )
                    belong = np.zeros(sel.size, dtype=bool)
                    for gi in range(sel.size):
                        others = np.arange(sel.size) != gi
                        dom_mask = others & (gt_sel == d)
                        other_mask = others & (gt_sel != d)
                        if dom_mask.sum() == 0 and other_mask.sum() == 0:
                            belong[gi] = True
                            continue
                        if dom_mask.sum() == 0:
                            belong[gi] = False
                            continue
                        if other_mask.sum() == 0:
                            belong[gi] = True
                            continue
                        dom_p = f_norm[dom_mask].mean(axis=0)
                        other_p = f_norm[other_mask].mean(axis=0)
                        dom_p = dom_p / max(float(np.linalg.norm(dom_p)), 1e-8)
                        other_p = other_p / max(
                            float(np.linalg.norm(other_p)), 1e-8
                        )
                        belong[gi] = float(f_norm[gi] @ dom_p) >= float(
                            f_norm[gi] @ other_p
                        )
                cid = cluster_map.setdefault(d, len(cluster_map))
                gs_cluster[tk * p + sel[belong]] = cid
                kept += int(belong.sum())
    else:
        raise ValueError(mode)
    cluster_gt = np.asarray(
        [gt for gt, _ in sorted(cluster_map.items(), key=lambda kv: kv[1])],
        dtype=np.int64,
    )
    return gs_cluster, cluster_gt, kept / max(1, n_gs)


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
        "--workspace", default="workspace/oracle_unit_prior_gs_residual"
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
            self._res_buf = {
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
                f"[gs-residual] frozen-backbone eval: loaded "
                f"{len(loadable)} backbone keys from {backbone_path}"
            )
    model.eval()
    model = model.cuda()
    br = model.instance_branch
    renderer = br.renderer
    render_scale = float(getattr(opt, "instance_group_render_scale", 1.0))

    # Feature hooks (no model modification).
    f_gs_store: dict[str, torch.Tensor] = {}
    img_gs_store: dict[str, torch.Tensor] = {}
    h1 = br.gs_feature_mlp.register_forward_hook(
        lambda mod, inp, out: f_gs_store.__setitem__("v", out.detach())
    )
    h2 = br.unit_image_net.register_forward_hook(
        lambda mod, inp, out: img_gs_store.__setitem__("v", out.detach())
    )

    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    variants = ["unit8_oracle", "res_perfect", "res_gt_drop", "res_feature_loo"]
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
        buf = br._res_buf
        T, P, K = buf["a"].shape[1], buf["a"].shape[2], br.units_per_token
        n_gs = T * P
        a3 = buf["a"][0].reshape(T, P, K).cpu().numpy()
        means3 = buf["means"][0].reshape(T, P, 3).cpu().numpy()
        gaussians = buf["gaussians"][0].cpu().numpy()
        cam_view = buf["cam_view"].float()
        intrinsics = buf["intrinsics"].float()
        gs_gt, _, _ = br._pseudo_gs_labels(
            torch.from_numpy(means3.reshape(1, n_gs, 3)).cuda(), data, opt
        )
        gs_gt3 = gs_gt[0].cpu().numpy().reshape(T, P)
        f_gs = f_gs_store["v"][0].cpu().numpy()
        img_gs = img_gs_store["v"][0].cpu().numpy()
        feat = _gs_feature(f_gs, img_gs, means3, gaussians)
        gt_maps = data["instance_label_output"][0].cpu().numpy()

        for mode in variants:
            gs_cluster, cluster_gt, kept_frac = _residual_labels(
                a3, gs_gt3, feat, mode
            )
            n_clusters = len(cluster_gt)
            onehot = np.zeros((n_gs, n_clusters + 1), dtype=np.float32)
            valid = gs_cluster >= 0
            onehot[np.arange(n_gs)[valid], gs_cluster[valid]] = 1.0
            # Background GS go to the void channel (matches the reference
            # oracle rendering; zero rows would inflate foreground masks).
            onehot[np.arange(n_gs)[~valid], n_clusters] = 1.0
            if n_clusters == 0:
                onehot[:, 0] = 1.0
            probs_t = torch.from_numpy(onehot).unsqueeze(0).cuda()
            with torch.inference_mode():
                render = renderer.render_feature_channels(
                    buf["gaussians"].float(),
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
            per_scene.setdefault(scene_name, {})[mode] = {
                "ap25": results["ap_25"],
                "ap50": results["ap_50"],
                "ap75": results["ap_75"],
                "ap_mean": coco_ap["ap_mean"],
                "num_pred": len(pred_masks),
                "num_gt": len(gt_masks),
                "num_clusters": n_clusters,
                "recall_05": recall[0.5],
                "kept_gs_frac": kept_frac,
                **gs_stats,
            }
        elapsed = time.time() - t_start
        print(
            f"[gs-residual] {scene_name} done "
            f"({i + 1}/{min(args.max_scenes or 40, 40)}) elapsed={elapsed:.0f}s"
        )

    h1.remove()
    h2.remove()
    summary = {}
    for mode in variants:
        entries = {s: per_scene[s][mode] for s in per_scene}
        summary[mode] = {
            "num_scenes": len(entries),
            **{
                key: float(np.mean([entries[s][key] for s in entries]))
                for key in (
                    "ap25",
                    "ap50",
                    "ap75",
                    "ap_mean",
                    "num_pred",
                    "num_gt",
                    "num_clusters",
                    "recall_05",
                    "kept_gs_frac",
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
            "8unit_oracle_ap50": 0.624,
            "32unit_oracle_ap50": 0.651,
            "gs_level_oracle_ap50": 0.789,
        },
        "variants": summary,
    }
    out_json = out_dir / "unit_prior_gs_residual.json"
    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    per_scene_json = out_dir / "unit_prior_gs_residual_per_scene.json"
    per_scene_json.write_text(
        json.dumps(per_scene, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[gs-residual] summary (scene-macro mean):")
    print(
        f"{'variant':16s} {'AP25':>6s} {'AP50':>6s} {'AP':>6s} "
        f"{'pred/gt':>9s} {'recall.5':>8s} {'keptGS':>6s} {'merge':>6s} "
        f"{'frag':>6s}"
    )
    for mode, entry in summary.items():
        print(
            f"{mode:16s} {entry['ap25']:6.3f} {entry['ap50']:6.3f} "
            f"{entry['ap_mean']:6.3f} "
            f"{int(entry['num_pred'])}/{int(entry['num_gt']):<4d} "
            f"{entry['recall_05']:8.3f} {entry['kept_gs_frac']:6.3f} "
            f"{entry['merging_frac']:6.2f} {entry['fragmentation_frac']:6.2f}"
        )
    print(f"[gs-residual] wrote {out_json}")


if __name__ == "__main__":
    main()
