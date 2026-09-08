"""Offline oracle: GT instance count + prune top-K agglomerative clusters.

Uses the trained `unit-shaping + patch + DINO` checkpoint with the standard
Agglomerative eps=0.5 inference (AP50=0.324).  For each scene the rendered
cluster probabilities are pruned to the top-K most-used clusters (K = GT
instance count from the 15-view projection), dumping the rest into void.
This isolates the "instance-count estimation" link: if the count were
perfect, how much would AP improve over the fixed-threshold baseline?

Reference: DINO+agg eps=0.5 AP50=0.324; per-scene best-eps oracle 0.369.
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


def _agg_labels(x: np.ndarray, eps: float) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage

    z = linkage(x, method="average")
    return fcluster(z, t=eps, criterion="distance").astype(np.int64) - 1


def _prune_top_k(
    probs: np.ndarray, active: int, void_channel: int
) -> np.ndarray:
    """Keep only the top-``active`` most-used groups; dump the rest to void."""
    num_groups = int(void_channel)
    if active >= num_groups:
        return probs
    probs = np.asarray(probs).copy()
    usage = probs[:num_groups].mean(axis=(1, 2))
    active_ids = np.argsort(-usage)[: int(active)]
    keep = np.zeros(num_groups, dtype=bool)
    keep[active_ids] = True
    inactive_sum = probs[:num_groups][~keep].sum(axis=0)
    probs[:num_groups][~keep] = 0.0
    probs[void_channel] = probs[void_channel] + inactive_sum
    total = probs.sum(axis=0)
    return probs / np.maximum(total, 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=(
            "workspace/semantic_v6_unit_shaping_img_dino_train_3000/"
            "checkpoints/model_step_003000.safetensors"
        ),
    )
    parser.add_argument("--workspace", default="workspace/dino_gtcount_prune")
    parser.add_argument("--label", default="dino_gtcount_prune")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument("--lsm_manifest", default=(
        "data/scannet_prompt/lsm_instance_eval_manifest.json"
    ))
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=15)
    parser.add_argument("--eps", type=float, default=0.5)
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
            self._pr_buffers = {
                "a": a.detach().float(),
                "pos_norm": ((center - scene_center) / scene_scale)
                .detach().float(),
                "means": means.detach().float(),
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
                f"[dino-prune] frozen-backbone eval: loaded "
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
        buf = br._pr_buffers
        batch_size = 1
        token_count = buf["a"].shape[1]
        k = br.units_per_token
        p = buf["a"].shape[2]
        u_count = token_count * k
        n_gs = token_count * p
        a4 = buf["a"][0].reshape(token_count, p, k).cpu().numpy()
        e = br.last_unit_embeddings[0].cpu().numpy()
        pos_norm = buf["pos_norm"][0].cpu().numpy()
        gaussians = buf["gaussians"].float()
        cam_view = buf["cam_view"].float()
        intrinsics = buf["intrinsics"].float()
        gs_gt, _, _ = br._pseudo_gs_labels(
            buf["means"].reshape(batch_size, n_gs, 3), data, opt
        )
        gs_gt = gs_gt[0].cpu().numpy()
        gt_count = int(len(np.unique(gs_gt[gs_gt > 0])))
        p_u = br.last_unit_pu[0].cpu().numpy()
        bg_idx = int(br.last_unit_bg_idx)
        fg_share_u = (
            1.0 - p_u[:, bg_idx] if bg_idx >= 0
            else np.ones(u_count, dtype=np.float32)
        )

        # Agglomerative eps (same as the 0.324 baseline), then prune.
        feat = np.concatenate([e, pos_norm], axis=-1).astype(np.float32)
        labels = _agg_labels(feat, float(args.eps))

        cluster_list = []
        for c in np.unique(labels):
            if c < 0:
                continue
            sel = labels == c
            if fg_share_u[sel].mean() >= void_fg_share:
                cluster_list.append(int(c))
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

        # ---- variants: no prune / prune to GT count / prune to 2x GT ----
        variant_results = {}
        for vname, active in (
            ("no_prune", 0),
            ("gt_count", gt_count),
            ("gt2x", 2 * gt_count),
            ("gt_half", max(1, gt_count // 2)),
        ):
            view_count = rendered_probs.shape[0]
            pred_masks, pred_scores, pred_image_ids = [], [], []
            gt_masks, gt_image_ids = [], []
            gt_maps = data["instance_label_output"][0].cpu().numpy()
            for v in range(view_count):
                image_id = f"{scene_name}:b0"
                probs = rendered_probs[v]
                if active > 0:
                    probs = _prune_top_k(probs, active, void_channel=num)
                masks, scores = masks_from_group_probs(
                    probs, void_channel=num, min_mask_area=min_pred_pixels
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
            variant_results[vname] = {
                "ap25": results["ap_25"],
                "ap50": results["ap_50"],
                "ap": coco_ap["ap_mean"],
                "num_pred": len(pred_masks),
            }
        per_scene[scene_name] = {
            "gt_count": gt_count,
            **{f"{k}_{v}": val for v, d in variant_results.items()
               for k, val in d.items()},
        }
        elapsed = time.time() - t_start
        print(
            f"[dino-prune] {scene_name} done "
            f"({i + 1}/{min(args.max_scenes or 40, 40)}) "
            f"gt={gt_count} no_prune={variant_results['no_prune']['ap50']:.3f} "
            f"gt_count={variant_results['gt_count']['ap50']:.3f} "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

    def _mean(key: str) -> float:
        vals = [per_scene[s][key] for s in per_scene]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "num_scenes": len(per_scene),
        "eps": args.eps,
        "no_prune_ap50": _mean("ap50_no_prune"),
        "gt_count_ap50": _mean("ap50_gt_count"),
        "gt2x_ap50": _mean("ap50_gt2x"),
        "gt_half_ap50": _mean("ap50_gt_half"),
        "no_prune_ap25": _mean("ap25_no_prune"),
        "gt_count_ap25": _mean("ap25_gt_count"),
        "no_prune_pred": _mean("num_pred_no_prune"),
        "gt_count_pred": _mean("num_pred_gt_count"),
        "mean_gt_count": _mean("gt_count"),
        "reference": {
            "dino_eps05_ap50": 0.324,
            "per_scene_best_eps_oracle_ap50": 0.369,
        },
    }
    (out_dir / "dino_gtcount_prune.json").write_text(
        json.dumps(
            {"summary": summary, "per_scene": per_scene},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\n[dino-prune] summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
