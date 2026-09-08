"""Offline diagnostics for the one-shot scene-conditioned prototype extractor.

No training, no model changes.  Frozen wide7l@8000 + DINO 8-local-unit
pipeline, LSM 40 held-out scenes.  Three diagnostics:

  1. deterministic floor -- 3D-FPS seeds + prototype mean-shift (no GT in
     the assignment): can a pure one-shot seeded grouping beat Agglomerative
     0.324?  (FPS-3D vs FPS-embedding, GT-count vs fixed K=100.)
  2. 3D-anchored prototype upper bound -- 3D-FPS seeds whose prototypes are
     refined with GT instance means (perfect refinement): how much of the
     0.534 GT-prototype oracle survives 3D seeding?
  3. cross-view consistency of the 8 context views -- per-unit per-view DINO
     features; is same-instance cross-view consistency higher than
     different-instance?  (validates using context views for prototype
     extraction.)

Usage:
    python scripts/diagnose_3d_seeded_prototypes.py \
        --workspace workspace/diagnose_3d_seeded_prototypes --gpu 0
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

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.instance_group_head import (  # noqa: E402
    TokenLocalUnitGrouping,
    _project_dense_features,
)
from tokengs.options import config_defaults  # noqa: E402
from tokengs.utils.instance_ap import (  # noqa: E402
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


def _fps_3d(centers: np.ndarray, num: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy farthest-point sampling on 3D centers (deterministic start)."""
    n = centers.shape[0]
    num = int(max(1, min(num, n)))
    first = int(np.argmin(np.linalg.norm(centers - centers.mean(0), axis=1)))
    idx = [first]
    dists = np.linalg.norm(centers - centers[first], axis=1)
    for _ in range(1, num):
        nxt = int(np.argmax(dists))
        idx.append(nxt)
        d = np.linalg.norm(centers - centers[nxt], axis=1)
        dists = np.minimum(dists, d)
    return np.asarray(idx, dtype=np.int64)


def _fps_emb(emb: np.ndarray, num: int) -> np.ndarray:
    """Greedy farthest-point sampling on L2-normalized embeddings (cosine)."""
    n = emb.shape[0]
    num = int(max(1, min(num, n)))
    first = int(np.argmax(np.linalg.norm(emb, axis=1)))  # most confident seed
    idx = [first]
    dists = 1.0 - emb @ emb[first]
    for _ in range(1, num):
        nxt = int(np.argmax(dists))
        idx.append(nxt)
        dists = np.minimum(dists, 1.0 - emb @ emb[nxt])
    return np.asarray(idx, dtype=np.int64)


def _prototype_iteration(
    emb: np.ndarray,
    seed_idx: np.ndarray,
    fg: np.ndarray,
    iterations: int,
    proto_init: np.ndarray | None = None,
) -> np.ndarray:
    """Hard cosine assignment + prototype mean re-estimation (mean-shift).

    Returns per-unit labels (-1 = void); only foreground units get labels.
    """
    u = emb.shape[0]
    k = seed_idx.size
    labels = np.full(u, -1, dtype=np.int64)
    proto = emb[seed_idx] if proto_init is None else proto_init
    proto = proto / np.maximum(
        np.linalg.norm(proto, axis=-1, keepdims=True), 1e-8
    )
    for _ in range(iterations):
        sim = emb @ proto.T  # [U,K]
        hard = np.argmax(sim, axis=1)
        new_proto = np.zeros_like(proto)
        for gi in range(k):
            members = np.where(fg & (hard == gi))[0]
            if members.size:
                new_proto[gi] = emb[members].mean(axis=0)
            else:
                new_proto[gi] = proto[gi]
        new_proto = new_proto / np.maximum(
            np.linalg.norm(new_proto, axis=-1, keepdims=True), 1e-8
        )
        proto = new_proto
    sim = emb @ proto.T
    labels[fg] = np.argmax(sim[fg], axis=1)
    return labels


def _gt_refined_prototypes(
    emb: np.ndarray,
    seed_idx: np.ndarray,
    dom_inst: np.ndarray,
    fg: np.ndarray,
) -> np.ndarray:
    """Each 3D seed's prototype = mean embedding of its GT instance units."""
    proto = np.zeros((seed_idx.size, emb.shape[1]), dtype=np.float64)
    for gi, s in enumerate(seed_idx.tolist()):
        gid = int(dom_inst[s])
        members = np.where(fg & (dom_inst == gid))[0]
        if members.size:
            proto[gi] = emb[members].mean(axis=0)
        else:
            proto[gi] = emb[s]
    return proto / np.maximum(
        np.linalg.norm(proto, axis=-1, keepdims=True), 1e-8
    )


def _render_and_eval(
    labels: np.ndarray,
    fg_share_u: np.ndarray,
    a4: np.ndarray,
    gaussians: torch.Tensor,
    cam_view: torch.Tensor,
    intrinsics: torch.Tensor,
    renderer,
    render_scale: float,
    void_fg_share: float,
    gs_inst: np.ndarray,
    gt_maps: np.ndarray,
    scene_name: str,
    min_pred_pixels: int = 1,
    min_gt_pixels: int = 1,
    max_pred: int = 100,
) -> dict:
    """One-hot group probs -> render -> LSM AP (same as seeded oracle)."""
    token_count, p, k = a4.shape
    n_gs = token_count * p
    cluster_list = []
    for c in np.unique(labels):
        if c < 0:
            continue
        sel = labels == c
        if fg_share_u[sel].mean() >= void_fg_share:
            cluster_list.append(int(c))
    used = {c: j for j, c in enumerate(cluster_list)}
    num = len(cluster_list)
    u_all = token_count * k
    onehot = np.zeros((u_all, num + 1), dtype=np.float32)
    for uu in range(u_all):
        c = int(labels[uu])
        onehot[uu, used[c] if c in used else num] = 1.0
    unit_probs_t = onehot.reshape(token_count, k, num + 1)
    group_probs = np.einsum("tpk,tkl->tpl", a4, unit_probs_t).reshape(
        n_gs, num + 1
    )
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
    return {
        "ap25": results["ap_25"],
        "ap50": results["ap_50"],
        "ap": coco_ap["ap_mean"],
        "num_pred": len(pred_masks),
        "num_gt": len(gt_masks),
        "num_clusters": num,
        "recall_05": recall[0.5],
        **gs_stats,
    }


def _cross_view_consistency(
    unit_dino_v: np.ndarray,  # [V,U,F] normalized per view
    visible: np.ndarray,  # [V,U] bool
    dom_inst: np.ndarray,
    fg: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Same- vs different-instance cross-view consistency (diagnostic 3)."""
    n_views, u_count, _ = unit_dino_v.shape
    n_vis = visible.sum(axis=0)
    cons = np.full(u_count, np.nan, dtype=np.float64)
    for uu in range(u_count):
        idx = np.where(visible[:, uu])[0]
        if idx.size < 2:
            continue
        feats = unit_dino_v[idx, uu]
        sim = feats @ feats.T
        off = sim[np.triu_indices(idx.size, k=1)]
        if off.size:
            cons[uu] = float(off.mean())
    same_vals, diff_vals = [], []
    inst_list = np.unique(dom_inst[fg])
    for gid in inst_list.tolist():
        members = np.where(fg & (dom_inst == gid))[0]
        members = members[~np.isnan(cons[members])]
        if members.size < 2:
            continue
        for _ in range(min(200, members.size * 2)):
            i, j = rng.choice(members, 2, replace=False)
            same_vals.append(min(cons[i], cons[j]))
    fg_units = np.where(fg)[0]
    fg_units = fg_units[~np.isnan(cons[fg_units])]
    if fg_units.size >= 2:
        for _ in range(4000):
            i, j = rng.choice(fg_units, 2, replace=False)
            if dom_inst[i] != dom_inst[j]:
                diff_vals.append(min(cons[i], cons[j]))
    same = float(np.mean(same_vals)) if same_vals else np.nan
    diff = float(np.mean(diff_vals)) if diff_vals else np.nan
    # P(same consistency > diff consistency) over sampled pairs.
    auc = np.nan
    if same_vals and diff_vals:
        n = min(len(same_vals), len(diff_vals), 4000)
        s = np.asarray(same_vals[:n])
        d = np.asarray(diff_vals[:n])
        auc = float((s[:, None] > d[None, :]).mean())
    return {
        "same_consistency": same,
        "diff_consistency": diff,
        "gap": (same - diff) if same == same and diff == diff else np.nan,
        "p_same_gt_diff": auc,
        "units_with_2plus_views": int((n_vis >= 2).sum()),
        "fg_units": int(fg.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=(
            "workspace/semantic_v6_unit_shaping_img_dino_train_3000/"
            "checkpoints/model_step_003000.safetensors"
        ),
    )
    parser.add_argument(
        "--workspace", default="workspace/diagnose_3d_seeded_prototypes"
    )
    parser.add_argument("--label", default="3d_seeded_prototypes")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument("--lsm_manifest", default=(
        "data/scannet_prompt/lsm_instance_eval_manifest.json"
    ))
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=15)
    args = parser.parse_args()

    if args.num_input_views != 8 or args.num_views != 15:
        raise ValueError("LSM protocol requires 8 context + 7 target views")
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
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
            self._anchor_buffers = {
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
                f"[3d-seed] loaded {len(loadable)} backbone keys from "
                f"{backbone_path}"
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

    d1_variants = [
        ("d1_fps3d_gtcount_iter1", "fps3d", 1, 0),
        ("d1_fps3d_gtcount_iter3", "fps3d", 3, 0),
        ("d1_fpsemb_gtcount_iter3", "fpsemb", 3, 0),
        ("d1_fps3d_K100_iter3", "fps3d_k100", 3, 0),
    ]
    d2_variants = [
        ("d2_seed3d_gtmean_iter0", "gtmean", 0, 0),
        ("d2_seed3d_gtmean_iter1", "gtmean", 1, 0),
        ("d2_seed3d_gtmean_iter3", "gtmean", 3, 0),
    ]
    per_scene: dict[str, dict] = {}
    rng = np.random.default_rng(42)
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
        buf = br._anchor_buffers
        batch_size = 1
        token_count = buf["a"].shape[1]
        k = br.units_per_token
        p = buf["a"].shape[2]
        u_count = token_count * k
        n_gs = token_count * p
        a4 = buf["a"][0].reshape(token_count, p, k).cpu().numpy()
        means_np = buf["means"].reshape(batch_size, n_gs, 3)
        means_t = means_np.reshape(token_count, p, 3).cpu().numpy()

        # Unit 3D centers.
        u_mass = a4.sum(axis=1).clip(min=1e-6)  # [T,K]
        unit_center = np.einsum("tpk,tpx->tkx", a4, means_t) / u_mass[:, :, None]
        unit_center = unit_center.reshape(u_count, 3)

        # DINO unit features + per-view DINO unit features (8 context views).
        dino_grid = dinov2_patch_features(dinov2, data["images_input"], device)
        dino_hw = _dino_frame_hw()
        intrin_dino = _rescale_intrinsics(
            buf["intrinsics_input"], tuple(opt.img_size), dino_hw
        )
        with torch.inference_mode():
            fused, _ = _project_dense_features(
                means_np, dino_grid, buf["cam_to_world_input"],
                intrin_dino, dino_hw,
            )
        gs_dino = fused[0].cpu().numpy()
        unit_dino = np.einsum(
            "tpk,tpd->tkd", a4, gs_dino.reshape(token_count, p, -1)
        ) / u_mass[:, :, None]
        unit_dino = unit_dino.reshape(u_count, -1)
        unit_dino = unit_dino / np.maximum(
            np.linalg.norm(unit_dino, axis=-1, keepdims=True), 1e-8
        )
        # Per-view per-unit DINO features + visibility (diagnostic 3).
        n_views_in = int(opt.num_input_views)
        unit_dino_v = np.zeros((n_views_in, u_count, unit_dino.shape[1]))
        visible_v = np.zeros((n_views_in, u_count), dtype=bool)
        for v in range(n_views_in):
            with torch.inference_mode():
                fv, valid_v = _project_dense_features(
                    means_np,
                    dino_grid[:, v : v + 1],
                    buf["cam_to_world_input"][:, v : v + 1],
                    intrin_dino[:, v : v + 1],
                    dino_hw,
                )
            gs_v = fv[0].cpu().numpy()
            vis_np = valid_v[0, :, 0].cpu().numpy()  # [N]
            ud_v = np.einsum(
                "tpk,tpd->tkd", a4, gs_v.reshape(token_count, p, -1)
            ) / u_mass[:, :, None]
            ud_v = ud_v.reshape(u_count, -1)
            ud_v = ud_v / np.maximum(
                np.linalg.norm(ud_v, axis=-1, keepdims=True), 1e-8
            )
            unit_dino_v[v] = ud_v
            uv = np.einsum(
                "tpk,tp->tk", a4, vis_np.reshape(token_count, p)
            ) / u_mass
            visible_v[v] = uv.reshape(u_count) > 0.5

        # GT instance + fg units (15-view vote).
        gs_inst, _, _ = br._pseudo_gs_labels(means_np, data, opt)
        gs_inst = gs_inst[0].cpu().numpy()
        inst_onehot = np.zeros(
            (n_gs, int(gs_inst.max()) + 1), dtype=np.float64
        )
        inst_onehot[np.arange(n_gs), gs_inst] = 1.0
        u_inst_mass = np.einsum(
            "tpk,tpm->tkm", a4,
            inst_onehot.reshape(token_count, p, -1),
        ).reshape(u_count, -1)
        dom_inst = u_inst_mass.argmax(axis=1)
        fg = (
            (u_inst_mass.max(axis=1) / u_mass.reshape(-1).clip(min=1e-6) > 0.5)
            & (dom_inst != 0)
        )
        fg_ids = np.unique(dom_inst[fg])
        gt_count = int(fg_ids.size)
        fg_share_u = np.zeros(u_count, dtype=np.float64)
        fg_share_u[fg] = 1.0

        gaussians = buf["gaussians"].float()
        cam_view = buf["cam_view"].float()
        intrinsics = buf["intrinsics"].float()
        gt_maps = data["instance_label_output"][0].cpu().numpy()
        entry: dict = {}

        # --- Diagnostic 1: deterministic FPS + mean-shift (no GT) ---
        for vname, seed_mode, iters, _seed in d1_variants:
            if seed_mode == "fps3d":
                seeds = _fps_3d(unit_center, gt_count, rng)
            elif seed_mode == "fpsemb":
                seeds = _fps_emb(unit_dino, gt_count)
            elif seed_mode == "fps3d_k100":
                seeds = _fps_3d(unit_center, 100, rng)
            labels = _prototype_iteration(unit_dino, seeds, fg, iters)
            entry[vname] = _render_and_eval(
                labels, fg_share_u, a4, gaussians, cam_view, intrinsics,
                renderer, render_scale, void_fg_share, gs_inst, gt_maps,
                scene_name, min_pred_pixels, min_gt_pixels, max_pred,
            )
            entry[vname]["num_seeds"] = int(seeds.size)

        # --- Diagnostic 2: 3D-FPS seeds + GT-refined prototypes ---
        seeds = _fps_3d(unit_center, gt_count, rng)
        gt_proto = _gt_refined_prototypes(unit_dino, seeds, dom_inst, fg)
        for vname, _mode, iters, _seed in d2_variants:
            labels = _prototype_iteration(
                unit_dino, seeds, fg, iters, proto_init=gt_proto
            )
            entry[vname] = _render_and_eval(
                labels, fg_share_u, a4, gaussians, cam_view, intrinsics,
                renderer, render_scale, void_fg_share, gs_inst, gt_maps,
                scene_name, min_pred_pixels, min_gt_pixels, max_pred,
            )
            entry[vname]["num_seeds"] = int(seeds.size)

        # --- Diagnostic 3: cross-view consistency ---
        entry["d3_crossview_consistency"] = _cross_view_consistency(
            unit_dino_v, visible_v, dom_inst, fg, rng
        )
        per_scene[scene_name] = entry
        elapsed = time.time() - t_start
        print(
            f"[3d-seed] {scene_name} done "
            f"({i + 1}/{min(args.max_scenes or 40, 40)}) "
            f"gt_inst={gt_count} elapsed={elapsed:.0f}s",
            flush=True,
        )

    def _mean(keys: list[str], scene_entries: dict) -> dict:
        out = {}
        for key in keys:
            vals = [scene_entries[s][key] for s in scene_entries]
            vals = [v for v in vals if v == v]  # drop nan
            out[key] = float(np.mean(vals)) if vals else float("nan")
        return out

    summary = {}
    all_variants = [v[0] for v in d1_variants + d2_variants]
    for vname in all_variants:
        scene_entries = {
            s: per_scene[s][vname] for s in per_scene if vname in per_scene[s]
        }
        summary[vname] = {
            "num_scenes": len(scene_entries),
            **_mean(
                [
                    "ap25", "ap50", "ap", "num_pred", "num_gt",
                    "num_clusters", "recall_05", "num_seeds",
                    "merging_frac", "mean_instances_per_cluster",
                    "fragmentation_frac", "mean_clusters_per_instance",
                ],
                scene_entries,
            ),
        }
    d3 = {
        s: per_scene[s]["d3_crossview_consistency"] for s in per_scene
    }
    summary["d3_crossview_consistency"] = {
        "num_scenes": len(d3),
        **_mean(
            [
                "same_consistency", "diff_consistency", "gap",
                "p_same_gt_diff", "units_with_2plus_views", "fg_units",
            ],
            d3,
        ),
    }
    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "protocol": manifest_audit,
        "reference": {
            "dino_agglomerative_ap50": 0.324,
            "dino_gt_prototype_oracle_ap50": 0.534,
            "seeded_best_center_unit_ap50": 0.44,
            "seeded_random_unit_ap50": 0.32,
        },
        "variants": summary,
    }
    (out_dir / "3d_seeded_prototypes.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "3d_seeded_per_scene.json").write_text(
        json.dumps(per_scene, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[3d-seed] summary:")
    for vname, vals in summary.items():
        print(
            f"  {vname:28s} AP25={vals.get('ap25', float('nan')):.3f} "
            f"AP50={vals.get('ap50', float('nan')):.3f} "
            f"AP={vals.get('ap', float('nan')):.3f} "
            f"pred/gt={vals.get('num_pred', 0) / max(vals.get('num_gt', 1), 1):.2f}"
        )


if __name__ == "__main__":
    main()
