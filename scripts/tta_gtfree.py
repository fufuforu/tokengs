"""GT-free test-time adaptation (TTA) of the instance representation.

Frozen: TokenGS encoder/decoder, 8-local-unit formation, DINO+patch unit
embedding, Gaussian geometry (the 0.324 ``semantic_v6_unit_shaping_img_dino
_train_3000`` checkpoint).

Adapted per scene: a tiny residual MLP on the unit instance features,
optimized on the 8 CONTEXT views only (no GT mask / instance label):

  1. cross-view rendered-feature consistency: the shared 3D Gaussians are
     the anchors - the frontmost GS of an anchor pixel in view i is
     projected into every other context view and the rendered adapted
     instance features at the corresponding pixels are pulled together
     (occluded correspondences skipped via z-buffer);
  2. diversity push: spatially distant units are pushed apart (hinge on
     cosine) so the scene features do not collapse;
  3. stability regularizer: keep the adapter's change from the frozen base
     small.

The 7 target views are used ONLY for the final LSM AP evaluation (same
rendering/scoring as the feed-forward protocol).  Reconstruction is
untouched by construction (the adapter never enters the RGB path), so PSNR
must be identical before/after.

Outputs per scene (and global): same/diff cosine, collapse, cluster count,
AP25/AP50/AP, PSNR - before and after TTA.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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


DINO_RESUME = (
    "/space0/mawb/tokengs/workspace/"
    "semantic_v6_unit_shaping_img_dino_train_3000/model.safetensors"
)


class SceneAdapter(torch.nn.Module):
    """Tiny per-scene residual adapter on the unit instance features."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + 3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, dim),
        )
        # init near zero so the adapted feature starts at the frozen base
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, f: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        delta = 0.05 * self.net(torch.cat([f, pos], dim=-1))
        return F.normalize(f + delta, dim=-1)


def compute_frontmost_maps(
    means: torch.Tensor,
    cam_view: torch.Tensor,
    intrinsics: torch.Tensor,
    img_size: tuple,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel frontmost (min-z) GS id and depth per context view.

    Geometry/camera-only -> computed once per scene and cached.  Returns
    ``gid_map`` [V,H,W] int64 (-1 = no GS) and ``z_map`` [V,H,W].
    """
    V = cam_view.shape[0]
    N = means.shape[0]
    H, W = int(img_size[0]), int(img_size[1])
    w2c = cam_view.transpose(-1, -2).float()  # [V,4,4]
    homo = torch.cat(
        [means.float(), torch.ones_like(means[..., :1])], dim=-1
    )
    cam = torch.einsum("vij,nj->vni", w2c, homo)  # [V,N,3]
    z = cam[..., 2]
    fx = intrinsics[:, 0, None]
    fy = intrinsics[:, 1, None]
    cx = intrinsics[:, 2, None]
    cy = intrinsics[:, 3, None]
    px = fx * cam[..., 0] / z.clamp_min(1e-6) + cx
    py = fy * cam[..., 1] / z.clamp_min(1e-6) + cy
    valid = (z > 0.01) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    gid_map = torch.full((V, H, W), -1, dtype=torch.long, device=means.device)
    z_map = torch.full((V, H, W), float("inf"), device=means.device)
    flat = torch.arange(N, device=means.device)
    for v in range(V):
        ok = valid[v]
        if not ok.any():
            continue
        pix = (py[v][ok].long() * W + px[v][ok].long())  # [M]
        zv = z[v][ok]
        gid_v = flat[ok]
        # min-z per pixel
        z_min = torch.full((H * W,), float("inf"), device=means.device)
        z_min.scatter_reduce_(
            0, pix, zv, reduce="amin", include_self=True
        )
        match = zv == z_min[pix]
        if match.any():
            gid_map[v].view(-1).scatter_(
                0, pix[match], gid_v[match]
            )
        z_map[v].view(-1).copy_(z_min)
    return gid_map, z_map


def project_points(
    pts: torch.Tensor,
    cam_view: torch.Tensor,
    intrinsics: torch.Tensor,
    img_size: tuple,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project 3D points into views; returns (uv [V,P,2], z [V,P])."""
    w2c = cam_view.transpose(-1, -2).float()
    homo = torch.cat([pts.float(), torch.ones_like(pts[..., :1])], dim=-1)
    cam = torch.einsum("vij,pj->vpi", w2c, homo)
    z = cam[..., 2]
    fx = intrinsics[:, 0, None]
    fy = intrinsics[:, 1, None]
    cx = intrinsics[:, 2, None]
    cy = intrinsics[:, 3, None]
    px = fx * cam[..., 0] / z.clamp_min(1e-6) + cx
    py = fy * cam[..., 1] / z.clamp_min(1e-6) + cy
    return torch.stack([px, py], dim=-1), z


def render_context_features(
    renderer,
    gaussians: torch.Tensor,
    gs_feat: torch.Tensor,
    cam_view: torch.Tensor,
    intrinsics: torch.Tensor,
    img_size: tuple,
):
    """Render per-GS instance features into the context views."""
    render = renderer.render_feature_channels(
        gaussians,
        gs_feat,
        cam_view.unsqueeze(0),
        intrinsics=intrinsics.unsqueeze(0),
    )
    rendered = render["images_pred"][0] / (render["alphas_pred"][0] + 1e-5)
    return F.normalize(rendered, dim=2), render


def tta_consistency_loss(
    renderer,
    gaussians: torch.Tensor,
    gs_feat: torch.Tensor,
    means: torch.Tensor,
    cam_view: torch.Tensor,
    intrinsics: torch.Tensor,
    img_size: tuple,
    gid_map: torch.Tensor,
    anchors_per_view: int,
    rng: np.random.Generator,
    margin: float = 0.5,
):
    """Cross-view rendered-feature consistency at shared-3D-Gaussian anchors.

    Sample anchor pixels in each context view, take their FRONTMOST GS from
    the cached z-buffer map, project that GS center into every other view,
    and only pull when the same GS is also the frontmost there (occlusion
    check).  The rendered adapted features at the two pixels are pulled
    together.
    """
    Fm, _ = render_context_features(
        renderer, gaussians, gs_feat, cam_view, intrinsics, img_size
    )
    V = cam_view.shape[0]
    H, W = int(img_size[0]), int(img_size[1])
    losses = []
    n_pairs = 0
    for i in range(V):
        gmap_i = gid_map[i]  # [H,W]
        ok = gmap_i >= 0
        idx = torch.where(ok.view(-1))[0]
        if idx.numel() == 0:
            continue
        if idx.numel() > anchors_per_view:
            sel = idx[
                torch.randperm(idx.numel(), device=idx.device)[
                    :anchors_per_view
                ]
            ]
        else:
            sel = idx
        yi = (sel // W).long()
        xi = (sel % W).long()
        gid_i = gmap_i[yi, xi]  # [A]
        gid_center = means[gid_i]  # [A,3]
        proj, zj = project_points(gid_center, cam_view, intrinsics, img_size)
        fi = Fm[i, :, yi, xi]  # [D,A]
        for j in range(V):
            if i == j:
                continue
            u = proj[j, :, 0]
            vv = proj[j, :, 1]
            inside = (u >= 0) & (u < W) & (vv >= 0) & (vv < H) & (zj[j] > 0.01)
            if not inside.any():
                continue
            ui = u[inside].long()
            vi = vv[inside].long()
            # occlusion check: same GS is frontmost in view j at the target
            same_gs = gid_map[j][vi, ui] == gid_i[inside]
            if not same_gs.any():
                continue
            s = inside.clone()
            s[inside] = same_gs
            fj = Fm[j, :, vi[same_gs], ui[same_gs]]  # [D,K]
            fia = fi[:, inside][:, same_gs]
            cos = (fia * fj).sum(0)
            losses.append(F.relu(margin - cos).mean())
            n_pairs += int(same_gs.sum())
    if losses:
        return torch.stack(losses).mean(), n_pairs
    return torch.zeros((), device=gs_feat.device), 0


def tta_diversity_loss(f: torch.Tensor, pos: torch.Tensor, margin: float = 0.3):
    """Push spatially distant units apart (anti-collapse)."""
    U = f.shape[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(U, size=min(4096, U), replace=False)
    a = torch.from_numpy(idx).to(f.device)
    b = torch.from_numpy(rng.choice(U, size=len(idx), replace=False)).to(f.device)
    dist = (pos[a] - pos[b]).norm(dim=-1)
    far = dist > 1.0
    if not far.any():
        return torch.zeros((), device=f.device)
    cos = (f[a[far]] * f[b[far]]).sum(-1)
    return F.relu(cos - margin).mean()


def cluster_and_score(
    ex: dict,
    e: np.ndarray,
    data: dict,
    renderer,
    opt,
    eps: float,
    pos_w: float,
    scene_name: str,
    max_pred: int = 100,
) -> dict:
    """Agglomerative clustering of unit features + render + LSM AP."""
    p_u = ex["p_u"]
    bg = ex["bg_idx"]
    fg_share_u = (
        1.0 - p_u[:, bg] if bg >= 0 else np.ones(ex["u_count"], dtype=np.float32)
    )
    labels = agglomerative_labels(
        e, ex["pos"], eps=eps, pos_w=pos_w
    )
    void_fg_share = float(getattr(opt, "instance_branch_void_fg_share", 0.5))
    cluster_list = []
    for c in np.unique(labels):
        if c < 0:
            continue
        sel = labels == c
        if float(fg_share_u[sel].mean()) >= void_fg_share:
            cluster_list.append(int(c))
    if not cluster_list:
        return {"ap25": 0.0, "ap50": 0.0, "ap_mean": 0.0, "num_pred": 0, "clusters": 0}
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
    group_probs = np.einsum("tpk,tkl->tpl", a3, unit_probs_t).reshape(
        ex["n_gs"], num + 1
    )
    gpt = torch.from_numpy(group_probs).unsqueeze(0).cuda()
    with torch.inference_mode():
        render = renderer.render_feature_channels(
            ex["gaussians"], gpt, ex["cam_view"], intrinsics=ex["intrinsics"]
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
        masks, scores = masks[:max_pred], scores[:max_pred]
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
    return {
        "ap25": res["ap_25"],
        "ap50": res["ap_50"],
        "ap_mean": coco["ap_mean"],
        "num_pred": len(pred_masks),
        "clusters": num,
    }


def same_diff_stats(e: np.ndarray, p_u: np.ndarray, bg_idx: int) -> dict:
    """Diagnostics on unit features vs the (GT-derived) soft instance
    distributions - used ONLY for reporting, never in the TTA objective."""
    fg = np.ones(e.shape[0], dtype=bool)
    if bg_idx >= 0:
        fg = (1.0 - p_u[:, bg_idx]) > 0.05
    if fg.sum() < 10:
        return {"same": 0.0, "diff": 0.0, "collapse": 0.0}
    ef, pf = e[fg], p_u[fg]
    m = pf.shape[1]
    fg_cols = [c for c in range(m) if c != bg_idx]
    pfn = pf[:, fg_cols]
    pfn = pfn / np.maximum(pfn.sum(1, keepdims=True), 1e-6)
    dom = pfn.argmax(1)
    pur = pfn.max(1)
    n = len(ef)
    r = np.arange(n)
    same_mask = (
        (dom[:, None] == dom[None, :])
        & (np.minimum(pur[:, None], pur[None, :]) >= 0.5)
        & (r[:, None] < r[None, :])
    )
    diff_mask = (dom[:, None] != dom[None, :]) & (r[:, None] < r[None, :])
    cos = ef @ ef.T
    same = cos[same_mask]
    diff = cos[diff_mask]
    collapse = cos[np.triu_indices(n, 1)].mean()
    return {
        "same": float(same.mean()) if same.size else 0.0,
        "diff": float(diff.mean()) if diff.size else 0.0,
        "collapse": float(collapse),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--resume", default=DINO_RESUME)
    parser.add_argument("--label", default="tta_gtfree")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument("--lsm_manifest", default=(
        "data/scannet_prompt/lsm_instance_eval_manifest.json"
    ))
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=15)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--pos_w", type=float, default=1.0)
    parser.add_argument("--tta_steps", type=int, default=30)
    parser.add_argument("--tta_lr", type=float, default=1e-3)
    parser.add_argument("--anchors_per_view", type=int, default=2048)
    parser.add_argument("--consist_margin", type=float, default=0.5)
    parser.add_argument("--div_margin", type=float, default=0.3)
    parser.add_argument("--div_weight", type=float, default=1.0)
    parser.add_argument("--reg_weight", type=float, default=0.1)
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
    model.eval()
    renderer = model.instance_branch.renderer
    _, loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    per_scene: dict[str, dict] = {}
    t0 = time.time()
    for i, data in enumerate(loader):
        if args.max_scenes > 0 and i >= args.max_scenes:
            break
        data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in data.items()}
        scene_name = str(data["scene_name"][0])
        ex = ca.extract_scene(model, data)
        e_base = ex["e"].astype(np.float32)
        pos = torch.from_numpy(ex["pos"]).float().cuda()
        e0 = torch.from_numpy(e_base).float().cuda()
        D = e_base.shape[1]
        adapter = SceneAdapter(D).cuda()
        optim = torch.optim.Adam(adapter.parameters(), lr=args.tta_lr)

        # context-view cameras
        c2w_ctx = data["cam_to_world_input"][0]  # [V,4,4]
        cam_view_ctx = torch.inverse(c2w_ctx).transpose(-1, -2)
        intr_ctx = data["intrinsics_input"][0]
        # clone inference-mode tensors so they can participate in autograd
        means = ex["means"].reshape(-1, 3).clone()
        gaussians = ex["gaussians"].clone()
        a = ex["a"].clone()
        # cached z-buffer frontmost maps (geometry-only, TTA-invariant)
        with torch.no_grad():
            gid_map, _ = compute_frontmost_maps(
                means,
                cam_view_ctx,
                intr_ctx,
                tuple(opt.img_size),
            )

        def gs_feat_from_units(fu: torch.Tensor) -> torch.Tensor:
            fu_t = fu.reshape(1, ex["token_count"], ex["k"], D)
            return torch.einsum("btpk,btkd->btpd", a, fu_t).reshape(
                1, ex["n_gs"], D
            )

        def cluster_and_score_adapted(fu: np.ndarray) -> dict:
            return cluster_and_score(
                ex, fu, data, renderer, opt, args.eps, args.pos_w, scene_name
            )

        before_stats = same_diff_stats(e_base, ex["p_u"], ex["bg_idx"])
        before_ap = cluster_and_score_adapted(e_base)
        with torch.no_grad():
            psnr = float(model(data, compute_quality_metrics=False)["psnr"])

        # ---- TTA on context views ----
        for _s in range(args.tta_steps):
            optim.zero_grad()
            f_adapt = adapter(e0, pos)
            gs_f = gs_feat_from_units(f_adapt)
            l_cyc, n_pairs = tta_consistency_loss(
                renderer,
                gaussians,
                gs_f,
                means,
                cam_view_ctx,
                intr_ctx,
                tuple(opt.img_size),
                gid_map,
                args.anchors_per_view,
                rng,
                margin=args.consist_margin,
            )
            l_div = tta_diversity_loss(
                f_adapt, pos, margin=args.div_margin
            )
            with torch.no_grad():
                base_delta = (f_adapt - e0).norm(dim=-1).mean()
            l_reg = base_delta
            loss = l_cyc + args.div_weight * l_div + args.reg_weight * l_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optim.step()

        with torch.no_grad():
            f_adapt_final = adapter(e0, pos)
        e_after = f_adapt_final.float().cpu().numpy()
        after_stats = same_diff_stats(e_after, ex["p_u"], ex["bg_idx"])
        after_ap = cluster_and_score_adapted(e_after)

        per_scene[scene_name] = {
            "before": {**before_stats, **before_ap},
            "after": {**after_stats, **after_ap},
            "psnr": psnr,
        }
        print(
            f"[tta] {scene_name}: before AP50={before_ap['ap50']:.3f} "
            f"same/diff={before_stats['same']:.3f}/{before_stats['diff']:.3f} "
            f"| after AP50={after_ap['ap50']:.3f} "
            f"same/diff={after_stats['same']:.3f}/{after_stats['diff']:.3f} "
            f"clusters {before_ap['clusters']}->{after_ap['clusters']} "
            f"psnr={psnr:.2f} ({i + 1}/{min(args.max_scenes or 40, 40)})",
            flush=True,
        )

    def _mean(keys, entries):
        return {
            k: float(np.mean([entries[s][k] for s in entries])) if entries else 0.0
            for k in keys
        }

    summary = {}
    for stage in ("before", "after"):
        entries = {s: per_scene[s][stage] for s in per_scene}
        summary[stage] = {
            "num_scenes": len(entries),
            **_mean(
                [
                    "ap25", "ap50", "ap_mean", "num_pred", "clusters",
                    "same", "diff", "collapse",
                ],
                entries,
            ),
        }
    summary["psnr"] = float(
        np.mean([per_scene[s]["psnr"] for s in per_scene])
    )
    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "num_scenes": len(per_scene),
        "fixed": vars(args),
        "summary": summary,
        "per_scene": per_scene,
    }
    (out_dir / "tta_gtfree.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
