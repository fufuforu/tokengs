"""Offline DINOv2 dense-feature instance-discrimination oracle.

Fixed checkpoint (unit-shaping + patch @2000), LSM 40 scenes, no training.

Pipeline per scene:
  1. Run the frozen model forward (patched) to obtain GS centers, frozen
     Gaussian geometry, cameras, and the frozen GS->8-unit assignment.
  2. Extract DINOv2 ViT-B/14 patch tokens (last layer, CLS dropped) for the
     8 input views and project them onto the GS centers with the same
     camera/projection machinery used by the current patch-feature branch.
  3. Aggregate multi-view DINO features per GS, then per unit with the
     frozen soft GS->unit assignment.
  4. Statistics with the 15-view majority-voted GT instance labels:
     same-instance / different-instance cosine, same-class DI vs diff-class
     DI (reuses the c3g8 semantic labels).
  5. GT-instance prototype oracle: unit DINO feature -> per-instance mean
     prototype -> cosine nearest assignment -> propagate to GS through the
     frozen assignment -> render with frozen geometry -> LSM AP25/AP50/AP.

Outputs JSON + a same/diff cosine histogram figure.  No model change.
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

import torch.nn.functional as F

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
from diagnose_pair_class_similarity import (  # noqa: E402
    _project_gs_labels,
    _quantiles,
)
from eval_instance_lsm_protocol import (  # noqa: E402
    _LocalAccelerator,
    _audit_lsm_manifest,
    _load_checkpoint_arch,
)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_dinov2(device: torch.device) -> torch.nn.Module:
    torch.hub.set_dir(str(Path.home() / ".cache/torch/hub"))
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14",
        source="github",
        force_reload=False,
    )
    model.eval()
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def dinov2_patch_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Last-layer DINOv2 patch tokens -> [B,V,D,Hf,Wf] grid.

    ``images``: [B,V,3,H,W] in [0,1].  Returns L2-normalized features.
    """
    b, v, c, h, w = images.shape
    # DINOv2 patch=14 requires dimensions divisible by 14; rescale the
    # 256x256 input to 252x252 (18x18 patches).
    target = 252
    x = F.interpolate(
        images.reshape(b * v, c, h, w).to(device),
        size=(target, target),
        mode="bilinear",
        align_corners=False,
    )
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    with torch.no_grad():
        out = model.get_intermediate_layers(
            x, n=1, return_class_token=False
        )  # list of [B*V, np, D] (CLS/registers already removed)
    tokens = out[0]
    dim = tokens.shape[-1]
    hf = wf = int(round(tokens.shape[1] ** 0.5))
    if tokens.shape[1] != hf * wf:
        raise RuntimeError(
            f"DINO tokens {tokens.shape} are not square after CLS drop "
            f"(input {tuple(x.shape)}); check register tokens / resolution."
        )
    feats = tokens.reshape(b * v, hf, wf, dim).permute(0, 3, 1, 2)
    feats = F.normalize(feats, dim=1)
    return feats.reshape(b, v, dim, hf, wf)


def _dino_frame_hw() -> tuple[int, int]:
    return (252, 252)


def _rescale_intrinsics(
    intrinsics: torch.Tensor, src_hw: tuple[int, int], dst_hw: tuple[int, int]
) -> torch.Tensor:
    """Scale fx/fy/cx/cy from the src image frame to the dst frame."""
    s = torch.tensor(
        [dst_hw[1] / src_hw[1], dst_hw[0] / src_hw[0],
         dst_hw[1] / src_hw[1], dst_hw[0] / src_hw[0]],
        dtype=intrinsics.dtype,
        device=intrinsics.device,
    )
    return intrinsics * s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=(
            "workspace/semantic_v6_unit_shaping_img_train_3000/"
            "checkpoints/model_step_002000.safetensors"
        ),
    )
    parser.add_argument("--workspace", default="workspace/dino_dense_oracle")
    parser.add_argument("--label", default="dino_vitb14_oracle")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument("--lsm_manifest", default=(
        "data/scannet_prompt/lsm_instance_eval_manifest.json"
    ))
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=15)
    parser.add_argument("--pair_samples", type=int, default=400000)
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

    device = torch.device("cuda")
    dinov2 = load_dinov2(device)
    print(f"[dino-oracle] DINOv2 ViT-B/14 loaded "
          f"({sum(p.numel() for p in dinov2.parameters())/1e6:.1f}M)")

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
            self._dino_buffers = {
                "a": a.detach().float(),
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
                f"[dino-oracle] frozen-backbone eval: loaded "
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
    same_cos_all: list[float] = []
    diff_cos_all: list[float] = []
    same_cls_cos_all: list[float] = []
    diff_cls_cos_all: list[float] = []
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
        buf = br._dino_buffers
        batch_size = 1
        token_count = buf["a"].shape[1]
        k = br.units_per_token
        p = buf["a"].shape[2]
        u_count = token_count * k
        n_gs = token_count * p
        a3 = buf["a"][0].reshape(n_gs, k).cpu().numpy()
        means_np = buf["means"].reshape(batch_size, n_gs, 3)

        # --- DINO dense features per GS (multi-view projection) ---
        dino_grid = dinov2_patch_features(
            dinov2, data["images_input"], device
        )  # [B,V,D,Hf,Wf]
        with torch.inference_mode():
            fused, has_source = _project_dense_features(
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
        gs_dino = fused[0].cpu().numpy()  # [N,D]
        vis = has_source[0, :, 0].cpu().numpy().astype(bool)
        # Unit-level DINO feature (frozen soft assignment, mass-weighted).
        a4 = a3.reshape(token_count, p, k)
        unit_dino = np.einsum(
            "tpk,tpd->tkd", a4, gs_dino.reshape(token_count, p, -1)
        )
        mass = np.einsum("tpk->tk", a4).clip(min=1e-6)
        unit_dino = (unit_dino / mass[:, :, None]).reshape(u_count, -1)
        unit_dino = unit_dino / np.maximum(
            np.linalg.norm(unit_dino, axis=-1, keepdims=True), 1e-8
        )

        # --- GT instance + semantic class per GS (15-view vote) ---
        gs_inst, _, _ = br._pseudo_gs_labels(means_np, data, opt)
        gs_inst = gs_inst[0].cpu().numpy()
        gs_cls, _ = _project_gs_labels(means_np, data, opt, "semantic_label")
        gs_cls = gs_cls[0]
        inst_onehot = np.zeros(
            (n_gs, int(gs_inst.max()) + 1), dtype=np.float64
        )
        inst_onehot[np.arange(n_gs), gs_inst] = 1.0
        cls_onehot = np.zeros(
            (n_gs, int(gs_cls.max()) + 1), dtype=np.float64
        )
        cls_onehot[np.arange(n_gs), gs_cls] = 1.0
        u_inst_mass = np.einsum(
            "tpk,tpm->tkm", a4,
            inst_onehot.reshape(token_count, p, -1),
        ).reshape(u_count, -1)
        u_cls_mass = np.einsum(
            "tpk,tpc->tkc", a4,
            cls_onehot.reshape(token_count, p, -1),
        ).reshape(u_count, -1)
        dom_inst = u_inst_mass.argmax(axis=1)
        dom_cls = u_cls_mass.argmax(axis=1)
        fg = (
            (u_inst_mass.max(axis=1) / mass.reshape(-1).clip(min=1e-6) > 0.5)
            & (dom_inst != 0)
        )
        cls_valid = (
            u_cls_mass.max(axis=1) / mass.reshape(-1).clip(min=1e-6) > 0.3
        )
        fg_idx = np.where(fg)[0]

        # --- cosine statistics (sampled pairs) ---
        rng = np.random.default_rng(0)
        n_fg = fg_idx.size
        n_cand = int(min(args.pair_samples * 3 + 5000, n_fg * n_fg))
        cand_a = rng.integers(0, n_fg, size=n_cand)
        cand_b = rng.integers(0, n_fg, size=n_cand)
        keep = cand_a != cand_b
        cand_a = fg_idx[cand_a[keep]]
        cand_b = fg_idx[cand_b[keep]]
        di = dom_inst[cand_a] != dom_inst[cand_b]
        si = ~di
        aa_d = cand_a[di][: args.pair_samples]
        bb_d = cand_b[di][: args.pair_samples]
        aa_s = cand_a[si][: args.pair_samples]
        bb_s = cand_b[si][: args.pair_samples]
        aa = np.concatenate([aa_d, aa_s])
        bb = np.concatenate([bb_d, bb_s])
        same_cos = []
        diff_cos = []
        same_cls_cos = []
        diff_cls_cos = []
        if aa.size:
            cos = (unit_dino[aa] * unit_dino[bb]).sum(axis=1)
            same = dom_inst[aa] == dom_inst[bb]
            same_cos = cos[same][: args.pair_samples]
            diff_cos = cos[~same][: args.pair_samples]
            vc = (
                cls_valid[aa]
                & cls_valid[bb]
                & (dom_cls[aa] != 0)
                & (dom_cls[bb] != 0)
            )
            sc = vc & (dom_cls[aa] == dom_cls[bb])
            same_cls_cos = cos[sc]
            diff_cls_cos = cos[vc & ~sc]
        same_cos_all.extend(same_cos.tolist())
        diff_cos_all.extend(diff_cos.tolist())
        same_cls_cos_all.extend(same_cls_cos.tolist())
        diff_cls_cos_all.extend(diff_cls_cos.tolist())

        # --- GT-instance prototype oracle ---
        fg_ids = np.unique(dom_inst[fg])
        proto = np.stack(
            [unit_dino[fg & (dom_inst == gid)].mean(axis=0) for gid in fg_ids]
        )
        proto = proto / np.maximum(
            np.linalg.norm(proto, axis=-1, keepdims=True), 1e-8
        )
        sim = unit_dino @ proto.T  # [U, F]
        labels = np.full(u_count, -1, dtype=np.int64)
        labels[fg] = np.argmax(sim[fg], axis=1)
        fg_share_u = np.zeros(u_count, dtype=np.float64)
        fg_share_u[fg] = 1.0

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
                buf["gaussians"].float(),
                group_probs_t,
                buf["cam_view"].float(),
                intrinsics=buf["intrinsics"].float(),
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
        per_scene[scene_name] = {
            "ap25": results["ap_25"],
            "ap50": results["ap_50"],
            "ap75": results["ap_75"],
            "ap_mean": coco_ap["ap_mean"],
            "num_pred": len(pred_masks),
            "num_gt": len(gt_masks),
            "num_clusters": num,
            "recall_025": recall[0.25],
            "recall_05": recall[0.5],
            "same_cos": _quantiles(np.asarray(same_cos)),
            "diff_cos": _quantiles(np.asarray(diff_cos)),
            "same_class_cos": _quantiles(np.asarray(same_cls_cos)),
            "diff_class_cos": _quantiles(np.asarray(diff_cls_cos)),
            **gs_stats,
        }
        elapsed = time.time() - t_start
        print(
            f"[dino-oracle] {scene_name} done "
            f"({i + 1}/{min(args.max_scenes or 40, 40)}) AP50="
            f"{results['ap_50']:.3f} elapsed={elapsed:.0f}s"
        )

    # ---- histogram figure ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(
            same_cos_all, bins=80, alpha=0.55, density=True,
            label="same-instance DINO cosine",
        )
        ax.hist(
            diff_cos_all, bins=80, alpha=0.55, density=True,
            label="different-instance DINO cosine",
        )
        ax.set_xlabel("unit-level DINO cosine")
        ax.set_ylabel("density")
        ax.set_title(
            f"DINOv2 unit cosine ({args.label})\n"
            f"same={np.mean(same_cos_all) if same_cos_all else float('nan'):.3f} "
            f"diff={np.mean(diff_cos_all) if diff_cos_all else float('nan'):.3f}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "dino_same_diff_hist.png", dpi=130)
        print("[dino-oracle] figure saved")
    except Exception as exc:  # pragma: no cover
        print(f"[dino-oracle] figure failed: {exc}")

    def _mean(keys: list[str]) -> dict:
        out = {}
        for key in keys:
            vals = [per_scene[s][key] for s in per_scene]
            out[key] = float(np.mean(vals)) if vals else 0.0
        return out

    summary = {
        "num_scenes": len(per_scene),
        "dino_same_cos": _quantiles(np.asarray(same_cos_all)),
        "dino_diff_cos": _quantiles(np.asarray(diff_cos_all)),
        "dino_separation": (
            float(np.mean(diff_cos_all) - np.mean(same_cos_all))
            if same_cos_all and diff_cos_all else None
        ),
        "dino_same_class_cos": _quantiles(np.asarray(same_cls_cos_all)),
        "dino_diff_class_cos": _quantiles(np.asarray(diff_cls_cos_all)),
        **_mean(
            [
                "ap25", "ap50", "ap75", "ap_mean",
                "num_pred", "num_gt", "num_clusters",
                "recall_025", "recall_05",
                "merging_frac", "mean_instances_per_cluster",
                "fragmentation_frac", "mean_clusters_per_instance",
            ]
        ),
        "reference": {
            "current_embedding_ap50": 0.279,
            "current_embedding_same_cos": 0.852,
            "current_embedding_diff_cos": 0.69,
            "current_embedding_gt_identity_unit_oracle_ap50": 0.70,
            "current_embedding_same_class_di_cos": 0.713,
            "current_embedding_diff_class_di_cos": 0.659,
        },
    }
    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "protocol": manifest_audit,
        "summary": summary,
        "per_scene": per_scene,
    }
    (out_dir / "dino_oracle.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[dino-oracle] summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
