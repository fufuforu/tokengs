"""Offline oracle: render the unit identity embedding and cluster per view
with k-means (k = GT-count oracle), instead of the 3D Agglomerative step.

Question: the unit embedding's rendered discriminability (pixel gap) keeps
improving but Agglomerative over-segments (pred/gt 1.56 -> 4.46) and AP
drops.  Does switching the inference to per-view k-means on the rendered
embedding recover AP above 0.324?

Two checkpoints are compared:
  - DINO@3000 (Agglomerative AP50=0.324)
  - render_space_info@3000 (stronger embedding, Agglomerative AP50=0.229)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--label", default="unit_kmeans")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument("--lsm_manifest", default=(
        "data/scannet_prompt/lsm_instance_eval_manifest.json"
    ))
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=15)
    parser.add_argument("--kmeans_dim", type=int, default=0,
                        help="Project rendered embedding to this dim (0=raw).")
    parser.add_argument("--kmeans_iters", type=int, default=20)
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
            self._km_buffers = {
                "a": a.detach().float(),
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
                f"[unit-kmeans] frozen-backbone eval: loaded "
                f"{len(loadable)} backbone keys from {backbone_path}"
            )
    model.eval()
    model = model.cuda()
    br = model.instance_branch
    renderer = br.renderer
    render_scale = float(getattr(opt, "instance_group_render_scale", 1.0))
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
        buf = br._km_buffers
        batch_size = 1
        token_count = buf["a"].shape[1]
        k = br.units_per_token
        p = buf["a"].shape[2]
        u_count = token_count * k
        n_gs = token_count * p
        a = buf["a"]
        e = br.last_unit_embeddings  # [B,U,D]
        D = e.shape[-1]
        # GS embedding = frozen assignment @ unit embedding, rendered.
        e_t = e.reshape(batch_size, token_count, k, D)
        gs_emb = torch.einsum("btpk,btkd->btpd", a, e_t).reshape(
            batch_size, n_gs, D
        )
        gaussians = buf["gaussians"].float()
        cam_view = buf["cam_view"].float()
        intrinsics = buf["intrinsics"].float()
        with torch.inference_mode():
            render = renderer.render_feature_channels(
                gaussians, gs_emb, cam_view,
                intrinsics=intrinsics, opacity_scale=render_scale,
            )
        rendered = render["images_pred"] / (
            render["alphas_pred"] + 1e-5
        )
        rendered = torch.nn.functional.normalize(rendered, dim=2)
        if args.kmeans_dim and args.kmeans_dim < D:
            # Project per view with a fixed random orthonormal matrix
            # (deterministic, no training) to reduce k-means dimensionality.
            rng = np.random.default_rng(0)
            proj = rng.standard_normal((D, args.kmeans_dim)).astype(np.float32)
            proj = torch.from_numpy(proj).cuda()
            proj = proj / proj.norm(dim=0, keepdim=True)
            rendered = torch.einsum("bvchw,cd->bvdhw", rendered, proj)
            D = args.kmeans_dim
        # GT count oracle.
        gs_gt, _, _ = br._pseudo_gs_labels(
            buf["means"].reshape(batch_size, n_gs, 3), data, opt
        )
        K = int(gs_gt[0].unique().numel() - 1)
        K = max(1, K)
        from scipy.cluster.vq import kmeans2

        V = rendered.shape[1]
        probs = torch.zeros(
            (batch_size, K + 1, V, 1, rendered.shape[3], rendered.shape[4]),
            device=rendered.device,
            dtype=rendered.dtype,
        )
        for b in range(batch_size):
            for v in range(V):
                feat = rendered[b, v].permute(1, 2, 0).reshape(
                    -1, D
                ).float().cpu().numpy()
                centroids, labels = kmeans2(
                    feat, K, minit="++", iter=args.kmeans_iters, seed=0
                )
                labels = labels.reshape(rendered.shape[3], rendered.shape[4])
                for kk in range(K):
                    probs[b, kk, v, 0][labels == kk] = 1.0
                probs[b, K, v, 0][labels >= K] = 1.0

        view_count = V
        pred_masks, pred_scores, pred_image_ids = [], [], []
        gt_masks, gt_image_ids = [], []
        gt_maps = data["instance_label_output"][0].cpu().numpy()
        for v in range(view_count):
            image_id = f"{scene_name}:b0"
            masks, scores = masks_from_group_probs(
                probs[0, :, v, 0].cpu().numpy(),
                void_channel=K,
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
        per_scene[scene_name] = {
            "ap25": results["ap_25"],
            "ap50": results["ap_50"],
            "ap": coco_ap["ap_mean"],
            "num_pred": len(pred_masks),
            "num_gt": len(gt_masks),
            "num_clusters": K,
            "recall_05": recall[0.5],
        }
        elapsed = time.time() - t_start
        print(
            f"[unit-kmeans] {scene_name} done "
            f"({i + 1}/{min(args.max_scenes or 40, 40)}) "
            f"AP50={results['ap_50']:.3f} K={K} elapsed={elapsed:.0f}s",
            flush=True,
        )

    def _mean(keys: list[str]) -> dict:
        out = {}
        for key in keys:
            vals = [per_scene[s][key] for s in per_scene]
            out[key] = float(np.mean(vals)) if vals else 0.0
        return out

    summary = {
        "num_scenes": len(per_scene),
        "kmeans_dim": D,
        **_mean(
            ["ap25", "ap50", "ap", "num_pred", "num_gt",
             "num_clusters", "recall_05"]
        ),
        "reference": {
            "dino_agglomerative_ap50": 0.324,
            "render_space_info_agglomerative_ap50": 0.229,
        },
    }
    (out_dir / "unit_kmeans_oracle.json").write_text(
        json.dumps(
            {"summary": summary, "per_scene": per_scene},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\n[unit-kmeans] summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
