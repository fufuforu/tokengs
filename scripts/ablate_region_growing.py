"""Offline 3D-aware region-growing grouping for the frozen unit-embedding
instance branch (pure inference, no training).

The checkpoint, identity embeddings, 8-local-unit formation and frozen
Gaussian geometry are all fixed.  Only the scene-level grouping step is
replaced:

  - Seeds are picked by farthest-point sampling over the foreground units'
    3D centers (no GT).
  - Each seed grows greedily: the highest-scoring unassigned unit among the
    3D-kNN of current cluster members is absorbed if
        cos(e_u, prototype_c) - w_d * d3d(u, centroid_c) >= thr
    where prototype_c is the running mean of member embeddings and d3d is
    the scene-normalized 3D distance to the cluster centroid.  Cluster
    prototypes/centroids are updated as the cluster grows, so identity is
    enforced at the cluster level (no per-unit push, no giant component).
  - Left-over foreground units become singleton clusters.

The resulting unit labels are propagated to GS through the frozen
GS->unit assignment, rendered through the frozen Gaussians, and scored with
the exact LSM machinery (AP25/AP50/AP, pred/gt, merge, fragmentation).

The reference to beat is the Agglomerative eps=0.5 baseline (AP50=0.279).
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch._dynamo.config.disable = True  # eager fallback for eval

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


# ---------------------------------------------------------------------------
# Region-growing clustering (numpy, no GT)
# ---------------------------------------------------------------------------


def _knn_graph(x: np.ndarray, k: int) -> np.ndarray:
    """3D k-NN indices via scipy cKDTree (k self-inclusive)."""
    from scipy.spatial import cKDTree

    n = x.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.int64)
    k = int(max(1, min(k + 1, n)))
    tree = cKDTree(x)
    _, idx = tree.query(x, k=k, workers=-1)
    idx = np.atleast_2d(idx)
    if idx.shape[1] < k:
        idx = np.tile(np.arange(n)[:, None], (1, k))
    # Drop self-neighbors (first column is the point itself).
    idx = idx[:, 1:] if idx.shape[1] > 1 else idx
    return idx.astype(np.int64)


def _fps_seeds(pos: np.ndarray, num: int, seed: int = 0) -> list[int]:
    """Farthest-point sampling over unit 3D centers (deterministic)."""
    n = pos.shape[0]
    if n == 0:
        return []
    num = int(max(1, min(num, n)))
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    sel = [first]
    d = np.full(n, np.inf)
    for _ in range(num - 1):
        d = np.minimum(d, np.linalg.norm(pos - pos[sel[-1]], axis=-1))
        sel.append(int(np.argmax(d)))
    return sel


def region_grow_labels(
    e: np.ndarray,
    pos_norm: np.ndarray,
    fg_mask: np.ndarray,
    n_seeds: int,
    thr: float,
    w_d: float,
    knn_k: int = 32,
) -> np.ndarray:
    """Region growing on foreground units -> per-unit labels (-1 = void).

    ``e``: [U, D] L2-normalized identity embeddings (foreground rows used).
    ``pos_norm``: [U, 3] scene-normalized unit centers.
    ``fg_mask``: [U] boolean, True = unit has foreground mass.
    """
    u = e.shape[0]
    labels = np.full(u, -1, dtype=np.int64)
    fg_idx = np.where(fg_mask)[0]
    if fg_idx.size == 0:
        return labels
    fg_pos = pos_norm[fg_idx]
    fg_e = e[fg_idx]
    m = fg_idx.size

    # 3D kNN only among foreground units.
    knn = _knn_graph(fg_pos, knn_k)  # [m, k]

    # Seeds: FPS over the foreground 3D centers (no GT).
    seeds = _fps_seeds(fg_pos, n_seeds)
    assigned = np.zeros(m, dtype=bool)
    cluster_id = 0
    members: list[list[int]] = []
    member_sets: list[set[int]] = []
    proto_sum: list[np.ndarray] = []
    cent_sum: list[np.ndarray] = []
    count: list[float] = []

    heap: list[tuple[float, int, int, int]] = []  # (-score, cid, unit, ver)
    ver = np.zeros(m, dtype=np.int64)
    heap_owner = np.full(m, -1, dtype=np.int64)  # best cluster for unit
    heap_score = np.full(m, -np.inf)

    def _push_candidates(cid: int) -> None:
        c = np.asarray(cent_sum[cid]) / max(count[cid], 1e-8)
        p = np.asarray(proto_sum[cid])
        pn = p / max(float(np.linalg.norm(p)), 1e-8)
        mset = member_sets[cid]
        for uid in members[cid]:
            for v in knn[uid]:
                if assigned[v] or v in mset:
                    continue
                sim = float(fg_e[v] @ pn)
                dist = float(np.linalg.norm(fg_pos[v] - c))
                score = sim - w_d * dist
                if score > heap_score[v]:
                    heap_score[v] = score
                    heap_owner[v] = cid
                    ver[v] += 1
                    heapq.heappush(heap, (-score, cid, v, int(ver[v])))

    for s in seeds:
        if assigned[s]:
            continue
        assigned[s] = True
        members.append([s])
        member_sets.append({s})
        proto_sum.append(fg_e[s].copy())
        cent_sum.append(fg_pos[s].copy())
        count.append(1.0)
        labels[fg_idx[s]] = cluster_id
        _push_candidates(cluster_id)
        cluster_id += 1

    while heap:
        neg_score, cid, v, vv = heapq.heappop(heap)
        if assigned[v]:
            continue
        if heap_owner[v] != cid or ver[v] != vv:
            continue  # stale entry
        # Re-score against the current cluster prototype/centroid.
        c = np.asarray(cent_sum[cid]) / max(count[cid], 1e-8)
        p = np.asarray(proto_sum[cid])
        pn = p / max(float(np.linalg.norm(p)), 1e-8)
        sim = float(fg_e[v] @ pn)
        dist = float(np.linalg.norm(fg_pos[v] - c))
        score = sim - w_d * dist
        if score < thr:
            continue  # keep in heap in case the cluster moves closer later
        assigned[v] = True
        members[cid].append(v)
        member_sets[cid].add(v)
        count[cid] += 1.0
        proto_sum[cid] = proto_sum[cid] + fg_e[v]
        cent_sum[cid] = cent_sum[cid] + fg_pos[v]
        labels[fg_idx[v]] = cid
        # New candidates from the absorbed unit's neighbors.
        mset = member_sets[cid]
        for nb in knn[v]:
            if assigned[nb] or nb in mset:
                continue
            sim2 = float(fg_e[nb] @ pn)
            dist2 = float(np.linalg.norm(fg_pos[nb] - c))
            score2 = sim2 - w_d * dist2
            if score2 > heap_score[nb]:
                heap_score[nb] = score2
                heap_owner[nb] = cid
                ver[nb] += 1
                heapq.heappush(heap, (-score2, cid, nb, int(ver[nb])))

    # Left-over foreground units become singleton clusters so no instance
    # coverage is silently dropped.
    for i in range(m):
        if not assigned[i]:
            assigned[i] = True
            labels[fg_idx[i]] = cluster_id
            cluster_id += 1
    return labels


def agglomerative_labels(
    e: np.ndarray, pos_norm: np.ndarray, eps: float, pos_w: float = 1.0
) -> np.ndarray:
    """Reference baseline: agglomerative average linkage on [e, w*pos]."""
    from scipy.cluster.hierarchy import fcluster, linkage

    feat = np.concatenate([e, pos_w * pos_norm], axis=-1).astype(np.float32)
    z = linkage(feat, method="average")
    return fcluster(z, t=eps, criterion="distance").astype(np.int64) - 1


def oracle_labels(
    a3: np.ndarray, gs_gt3: np.ndarray
) -> np.ndarray:
    """Per-unit dominant GT-instance label (upper-bound reference)."""
    t, p, k = a3.shape
    fg_ids = np.unique(gs_gt3[gs_gt3 > 0])
    u = t * k
    if fg_ids.size == 0:
        return np.full(u, -1, dtype=np.int64)
    mass = np.zeros((t, k, fg_ids.size), dtype=np.float64)
    for gi, gid in enumerate(fg_ids.tolist()):
        mask = (gs_gt3 == gid)[:, :, None].astype(np.float64)
        mass[:, :, gi] = (a3 * mask).sum(axis=1)
    mass = mass.reshape(u, fg_ids.size)
    total = mass.sum(axis=1)
    fg = total > 0
    dom = mass.argmax(axis=1)
    labels = np.full(u, -1, dtype=np.int64)
    labels[fg] = dom[fg]
    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        default="workspace/region_growing_ablation_2000",
    )
    parser.add_argument("--label", default="region_growing_2000")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument("--lsm_manifest", default=(
        "data/scannet_prompt/lsm_instance_eval_manifest.json"
    ))
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=15)
    parser.add_argument(
        "--instances_per_seed", type=int, default=24,
        help="Expected units per instance -> seed count = ceil(U_fg / this).",
    )
    parser.add_argument("--thr", type=float, default=0.15)
    parser.add_argument("--w_d", type=float, default=1.0)
    parser.add_argument("--knn_k", type=int, default=32)
    parser.add_argument("--sweep", action="store_true",
                        help="Run a small grid over seeds/thr/w_d.")
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

    if args.sweep:
        variants: list[dict] = [
            {"name": f"rg_ips{ips}_thr{thr:.2f}_wd{w_d:.1f}",
             "kind": "rg", "ips": ips, "thr": thr, "w_d": w_d,
             "knn_k": args.knn_k}
            for ips, thr, w_d in (
                (16, 0.15, 1.0),   # loose -> merge dominated
                (32, 0.30, 1.0),   # default
                (32, 0.55, 3.0),   # strict
            )
        ]
    else:
        variants = [{
            "name": "region_growing",
            "kind": "rg",
            "ips": args.instances_per_seed,
            "thr": args.thr,
            "w_d": args.w_d,
            "knn_k": args.knn_k,
        }]
    variants += [
        {"name": "agg_avg_eps0.5_pw1", "kind": "agg",
         "eps": 0.5, "pos_w": 1.0},
        {"name": "oracle_gt", "kind": "oracle"},
    ]
    print(f"[region-growing] {len(variants)} variants, "
          f"scenes={args.max_scenes or 40}")

    # ------------------------------------------------------------------
    # Patch the model forward to expose the frozen internals (same as the
    # clustering ablation; formation/embedding/geometry untouched).
    # ------------------------------------------------------------------
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
            self._rg_buffers = {
                "a": a.detach().float(),
                "unit_center": unit_center.detach().float(),
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
                f"[region-growing] frozen-backbone eval: loaded "
                f"{len(loadable)} backbone keys from {backbone_path}"
            )
    model.eval()
    model = model.cuda()
    br = model.instance_branch
    renderer = br.renderer
    render_scale = float(getattr(opt, "instance_group_render_scale", 1.0))
    void_fg_share = float(
        getattr(opt, "instance_branch_void_fg_share", 0.5)
    )
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
        buf = br._rg_buffers
        batch_size = 1
        token_count = buf["a"].shape[1]
        k = br.units_per_token
        p = buf["a"].shape[2]
        u_count = token_count * k
        n_gs = token_count * p
        a_flat = buf["a"][0].reshape(n_gs, k).cpu().numpy()
        e = br.last_unit_embeddings[0].cpu().numpy()  # [U,D]
        pos_norm = buf["pos_norm"][0].cpu().numpy()  # [U,3]
        gaussians = buf["gaussians"].float()
        cam_view = buf["cam_view"].float()
        intrinsics = buf["intrinsics"].float()
        gs_gt, _, _ = br._pseudo_gs_labels(
            buf["means"].reshape(batch_size, n_gs, 3), data, opt
        )
        gs_gt_np = gs_gt[0].cpu().numpy()
        p_u = br.last_unit_pu[0].cpu().numpy()  # [U,m]
        bg_idx = int(br.last_unit_bg_idx)
        if bg_idx >= 0:
            fg_share_u = 1.0 - p_u[:, bg_idx]
        else:
            fg_share_u = np.ones(u_count, dtype=np.float32)
        fg_mask = fg_share_u > 0.05
        a3 = a_flat.reshape(token_count, p, k)

        for variant in variants:
            kind = variant["kind"]
            if kind == "rg":
                n_fg = int(fg_mask.sum())
                n_seeds = max(1, int(np.ceil(n_fg / variant["ips"])))
                labels = region_grow_labels(
                    e, pos_norm, fg_mask,
                    n_seeds=n_seeds,
                    thr=float(variant["thr"]),
                    w_d=float(variant["w_d"]),
                    knn_k=int(variant["knn_k"]),
                )
            elif kind == "agg":
                labels = agglomerative_labels(
                    e, pos_norm,
                    eps=float(variant["eps"]),
                    pos_w=float(variant["pos_w"]),
                )
            elif kind == "oracle":
                labels = oracle_labels(a3, gs_gt_np.reshape(token_count, p))
            else:
                raise ValueError(f"unknown kind {kind}")

            # Shared rendering: build one-hot unit->cluster assignment per
            # variant (pure numpy), then render and score each.
            cluster_list: list[int] = []
            for c in np.unique(labels):
                if c < 0:
                    continue
                sel = labels == c
                fg_share = float(fg_share_u[sel].mean())
                if fg_share >= void_fg_share:
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
            # Cap rendered channels (diagnostics only): fragment-explosion
            # variants keep the largest clusters, mirroring the eval's
            # max-predictions budget instead of paying for hundreds of
            # singleton render channels.
            max_clusters = 64
            if len(cluster_list) > max_clusters:
                sizes = np.asarray(
                    [int((labels == c).sum()) for c in cluster_list]
                )
                order = np.argsort(-sizes, kind="stable")[:max_clusters]
                cluster_list = [cluster_list[j] for j in order]
            used = {c: j for j, c in enumerate(cluster_list)}
            num = len(cluster_list)
            onehot = np.zeros((u_count, num + 1), dtype=np.float32)
            for uu in range(u_count):
                c = int(labels[uu])
                if c in used:
                    onehot[uu, used[c]] = 1.0
                else:
                    onehot[uu, num] = 1.0
            unit_probs_t = onehot.reshape(token_count, k, num + 1)
            group_probs = np.einsum(
                "tpk,tkl->tpl", a3, unit_probs_t
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
            pred_masks: list[np.ndarray] = []
            pred_scores: list[float] = []
            pred_image_ids: list[str] = []
            gt_masks: list[np.ndarray] = []
            gt_image_ids: list[str] = []
            gt_maps = data["instance_label_output"][0].cpu().numpy()
            for v in range(view_count):
                image_id = f"{scene_name}:b0"
                probs = rendered_probs[v]
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
            recall = _max_iou_recall(pred_masks, gt_masks)
            gs_stats = _gs_level_stats(group_probs, gs_gt_np, num)
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
        print(
            f"[region-growing] {scene_name} done "
            f"({i + 1}/{min(args.max_scenes or 40, 40)}) "
            f"elapsed={elapsed:.0f}s"
        )

    def _mean(keys: list[str], scene_entries: dict) -> dict:
        out = {}
        for key in keys:
            vals = [scene_entries[s][key] for s in scene_entries]
            out[key] = float(np.mean(vals)) if vals else 0.0
        return out

    summary = {}
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
                scene_entries,
            ),
        }

    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "num_scenes": len(per_scene),
        "protocol": manifest_audit,
        "fixed": {
            "unit_embedding": True,
            "units_per_token": k,
            "num_gs_tokens": token_count,
            "gaussians_per_token": p,
            "embed_dim": int(e.shape[-1]),
            "void_fg_share": void_fg_share,
            "render_scale": render_scale,
            "min_pred_pixels": min_pred_pixels,
            "min_gt_pixels": min_gt_pixels,
            "max_predictions_per_image": max_pred,
        },
        "reference_AP50": 0.279,
        "variants": summary,
    }
    (out_dir / "region_growing.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "region_growing_per_scene.json").write_text(
        json.dumps(per_scene, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n[region-growing] summary (scene-macro mean):")
    print(
        f"{'variant':36s} {'AP25':>6s} {'AP50':>6s} {'AP':>6s} "
        f"{'pred/gt':>10s} {'recall.5':>8s} {'merge':>6s} {'frag':>6s}"
    )
    for name, entry in summary.items():
        print(
            f"{name:36s} {entry['ap25']:6.3f} {entry['ap50']:6.3f} "
            f"{entry['ap_mean']:6.3f} "
            f"{int(entry['num_pred'])}/{int(entry['num_gt']):<4d} "
            f"{entry['recall_05']:8.3f} {entry['merging_frac']:6.2f} "
            f"{entry['fragmentation_frac']:6.2f}"
        )


if __name__ == "__main__":
    main()
