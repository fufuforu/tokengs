"""Learned cluster-level affinity merging on top of the frozen 0.324
unit-embedding representation.

Freezes: TokenGS, 8-local-unit formation, DINO+patch identity embedding,
Gaussian geometry (the whole ``semantic_v6_unit_shaping_img_dino_train_3000``
checkpoint).

Learns: a small pairwise MLP predicting P(two spatial clusters belong to the
same instance).  The GT soft affinity is NOT a bare overlap dot product; it
combines
  - dominant-instance agreement (gate): argmax of the foreground instance
    distribution must match,
  - shared quality: the fraction of the smaller cluster's foreground mass
    that is shared (``sum min(p_i,p_j) / min(mass_i,mass_j)``),
  - purity confidence: ``min(purity_i, purity_j)``,
and training uses hard-negative sampling (spatially close clusters with
different dominant instances) plus random negatives.

Inference (deliberately anti chain-merge / giant-component):
  1. over-segment the scene by 3D position only (DBSCAN + size cap) into
     many small spatial clusters;
  2. predict affinities only for pairs within a locality radius;
  3. constrained greedy merging with learned affinities on the CURRENT
     cluster features, gated by an affinity threshold, a max cluster size
     and a max spatial extent (complete-linkage-like re-evaluation every
     round, so weak links cannot chain into a giant component);
  4. propagate the final unit->cluster labels to GS via the frozen
     GS->unit soft assignment, render, and score with the LSM protocol.

Mode ``train``: iterates the 0.324 recipe's ScanNet training windows,
extracts per-window units (frozen), builds spatial clusters + soft-labelled
pairs, and trains the affinity MLP online.
Mode ``eval``: scores LSM held-out scenes with the trained MLP (plus the
agg-eps0.5 and GT-oracle references).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

torch._dynamo.config.disable = True  # eager fallback for eval

ROOT = Path(__file__).resolve().parents[1]
sys_path = ROOT
import sys

sys.path.insert(0, str(sys_path))
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
from ablate_region_growing import (  # noqa: E402
    _fps_seeds,
    agglomerative_labels,
    oracle_labels,
)
from eval_instance_lsm_protocol import (  # noqa: E402
    _LocalAccelerator,
    _audit_lsm_manifest,
    _load_checkpoint_arch,
)


DINO_RESUME = (
    "/space0/mawb/tokengs/workspace/"
    "semantic_v6_unit_shaping_img_dino_train_3000/model.safetensors"
)
WIDE7L_BACKBONE = (
    "/space0/mawb/tokengs/workspace/"
    "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
    "checkpoints/model_step_008000.safetensors"
)
WIDE_8X7_MANIFEST = (
    "/space0/mawb/tokengs/data/scannet_prompt/"
    "scannet_prompt_full_wide_8x7.json"
)


# ---------------------------------------------------------------------------
# Spatial over-segmentation + cluster features
# ---------------------------------------------------------------------------


def spatial_clusters(
    pos: np.ndarray, eps: float, max_units: int, seed: int = 0
) -> np.ndarray:
    """Over-segment units by 3D position (no instance info).

    DBSCAN (density, min_samples=1 so nothing is dropped) followed by a size
    cap: oversized clusters are split by farthest-point seeds + nearest
    assignment, guaranteeing small, spatially compact atoms.
    """
    from sklearn.cluster import DBSCAN

    n = pos.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    labels = DBSCAN(
        eps=float(eps), min_samples=1, metric="euclidean",
        algorithm="kd_tree", n_jobs=-1,
    ).fit_predict(pos)
    labels = np.where(labels < 0, np.arange(n) + 10_000_000, labels)
    uniq = {c: i for i, c in enumerate(np.unique(labels))}
    labels = np.array([uniq[c] for c in labels], dtype=np.int64)
    # Split oversized clusters.
    new_labels = labels.copy()
    nxt = int(labels.max()) + 1
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if idx.size <= max_units:
            continue
        n_seeds = int(np.ceil(idx.size / max_units))
        seeds = _fps_seeds(pos[idx], n_seeds, seed=seed)
        d = np.linalg.norm(pos[idx][:, None, :] - pos[idx][None, seeds, :], axis=-1)
        sub = np.argmin(d, axis=1)
        for s in range(n_seeds):
            new_labels[idx[sub == s]] = nxt
            nxt += 1
    uniq2 = {c: i for i, c in enumerate(np.unique(new_labels))}
    return np.array([uniq2[c] for c in new_labels], dtype=np.int64)


def cluster_summaries(
    e: np.ndarray,
    pos: np.ndarray,
    p_u: np.ndarray,
    bg_idx: int,
    labels: np.ndarray,
    n_clusters: int,
) -> dict:
    """Per-cluster statistics.

    Returns arrays (all length ``n_clusters``): e_c (normalized mean
    embedding), pos_c, size, extent (max pairwise 3D distance), p_fg
    (foreground-normalized mean instance distribution), purity, dom
    (dominant foreground instance id; -1 if no foreground mass), valid.
    """
    m = p_u.shape[1]
    fg_cols = [c for c in range(m) if c != bg_idx]
    e_c = np.zeros((n_clusters, e.shape[1]), dtype=np.float32)
    pos_c = np.zeros((n_clusters, 3), dtype=np.float32)
    size = np.zeros(n_clusters, dtype=np.int64)
    extent = np.zeros(n_clusters, dtype=np.float32)
    p_c = np.zeros((n_clusters, len(fg_cols)), dtype=np.float32)
    purity = np.zeros(n_clusters, dtype=np.float32)
    dom = np.full(n_clusters, -1, dtype=np.int64)
    valid = np.zeros(n_clusters, dtype=bool)
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        ee = e[idx]
        e_c[c] = ee.mean(0)
        nrm = np.linalg.norm(e_c[c])
        if nrm > 1e-6:
            e_c[c] /= nrm
        pos_c[c] = pos[idx].mean(0)
        size[c] = idx.size
        pd = np.linalg.norm(pos[idx][:, None, :] - pos[idx][None, :, :], axis=-1)
        extent[c] = float(pd.max()) if idx.size > 1 else 0.0
        pf = p_u[idx][:, fg_cols].mean(0)
        s = float(pf.sum())
        if s > 1e-4:
            pf = pf / s
            p_c[c] = pf
            purity[c] = float(pf.max())
            dom[c] = int(fg_cols[int(np.argmax(pf))])
            valid[c] = True
    return {
        "e": e_c,
        "pos": pos_c,
        "size": size,
        "extent": extent,
        "p": p_c,
        "purity": purity,
        "dom": dom,
        "valid": valid,
        "fg_cols": fg_cols,
    }


def soft_affinity_label(
    p_i: np.ndarray, p_j: np.ndarray, purity_i: float, purity_j: float,
    dom_i: int, dom_j: int,
) -> float:
    """GT soft affinity for a cluster pair.

    ``dom`` are indices into the foreground instance space (already
    foreground-normalized).  Combines:
      y = 1[dom_i == dom_j] * min(purity_i, purity_j) * share_ratio
    with share_ratio = sum_c min(p_i[c], p_j[c]) / min(sum p_i, sum p_j),
    i.e. dominant-agreement gate * confidence * shared quality.
    """
    if dom_i != dom_j or dom_i < 0 or dom_j < 0:
        return 0.0
    shared = float(np.minimum(p_i, p_j).sum())
    max_shared = max(float(min(p_i.sum(), p_j.sum())), 1e-6)
    share_ratio = min(1.0, shared / max_shared)
    return float((dom_i == dom_j)) * float(min(purity_i, purity_j)) * share_ratio


def pair_features(
    s: dict, i: int, j: int, max_size: float, max_extent: float
) -> np.ndarray:
    """Symmetric (order-invariant) pair features for the affinity MLP."""
    e_i, e_j = s["e"][i], s["e"][j]
    p_i, p_j = s["pos"][i], s["pos"][j]
    e_mean = (e_i + e_j) / 2.0
    nrm = np.linalg.norm(e_mean)
    if nrm > 1e-6:
        e_mean = e_mean / nrm
    return np.concatenate(
        [
            e_mean,
            np.abs(e_i - e_j),
            [float(e_i @ e_j)],
            (p_i + p_j) / 2.0,
            np.abs(p_i - p_j),
            [
                min(s["size"][i], s["size"][j]) / max_size,
                max(s["size"][i], s["size"][j]) / max_size,
                min(s["extent"][i], s["extent"][j]) / max_extent,
                max(s["extent"][i], s["extent"][j]) / max_extent,
                min(s["purity"][i], s["purity"][j]),
                max(s["purity"][i], s["purity"][j]),
            ],
        ],
        dtype=np.float32,
    )


def sample_pairs(
    s: dict,
    n_pos: int,
    n_hard: int,
    n_easy: int,
    hard_dist: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (i, j, y) training pairs.

    Positives: same dominant instance, purity_min >= 0.5.
    Hard negatives: different dominant instances, spatially close.
    Easy negatives: different dominant instances, random.
    """
    idx = np.where(s["valid"])[0]
    if idx.size < 2:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float32)
    pos = s["pos"][idx]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    dom = s["dom"][idx]
    pur = s["purity"][idx]
    pairs, ys = [], []
    # positives
    pos_idx = np.argwhere((dom[:, None] == dom[None, :]) & (np.arange(len(idx))[:, None] < np.arange(len(idx))[None, :]))
    keep = np.minimum(pur[pos_idx[:, 0]], pur[pos_idx[:, 1]]) >= 0.5
    pos_idx = pos_idx[keep]
    if pos_idx.size > 0:
        sel = rng.choice(pos_idx.shape[0], size=min(n_pos, pos_idx.shape[0]), replace=False)
        for a, b in pos_idx[sel]:
            pi, pj = s["p"][idx[a]], s["p"][idx[b]]
            y = soft_affinity_label(pi, pj, pur[a], pur[b], dom[a], dom[b])
            pairs.append((int(idx[a]), int(idx[b])))
            ys.append(y)
    # hard negatives: different dom, close
    hard_mask = (dom[:, None] != dom[None, :]) & (d < hard_dist) & (
        np.arange(len(idx))[:, None] < np.arange(len(idx))[None, :]
    )
    hard_i, hard_j = np.where(hard_mask)
    if hard_i.size > 0:
        order = np.argsort(d[hard_i, hard_j])[: n_hard * 4]
        hard_i, hard_j = hard_i[order], hard_j[order]
        if hard_i.size > n_hard:
            sel = rng.choice(hard_i.size, size=n_hard, replace=False)
            hard_i, hard_j = hard_i[sel], hard_j[sel]
        for a, b in zip(hard_i, hard_j):
            pairs.append((int(idx[a]), int(idx[b])))
            ys.append(0.0)
    # easy negatives
    easy_mask = (dom[:, None] != dom[None, :]) & (
        np.arange(len(idx))[:, None] < np.arange(len(idx))[None, :]
    )
    easy_i, easy_j = np.where(easy_mask)
    if easy_i.size > 0:
        sel = rng.choice(easy_i.size, size=min(n_easy, easy_i.size), replace=False)
        for a, b in zip(easy_i[sel], easy_j[sel]):
            pairs.append((int(idx[a]), int(idx[b])))
            ys.append(0.0)
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float32)
    return np.asarray(pairs, dtype=np.int64), np.asarray(ys, dtype=np.float32)


# ---------------------------------------------------------------------------
# Affinity MLP + normalization
# ---------------------------------------------------------------------------


class AffinityMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


class RunningNorm:
    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0.0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        x = x.reshape(-1, self.dim).astype(np.float64)
        n = x.shape[0]
        new_n = self.n + n
        x_mean = x.mean(axis=0)
        mean_new = self.mean + (x_mean - self.mean) * n / new_n
        self.m2 = (
            self.m2
            + ((x - x_mean) ** 2).sum(axis=0)
            + n * (x_mean - self.mean) * (x_mean - mean_new)
        )
        self.mean = mean_new
        self.n = new_n

    def stats(self) -> tuple[np.ndarray, np.ndarray]:
        std = np.sqrt(self.m2 / max(self.n - 1, 1.0))
        std = np.where(std < 1e-6, 1.0, std)
        return self.mean.astype(np.float32), std.astype(np.float32)


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Constrained greedy merging (anti chain-merge / giant component)
# ---------------------------------------------------------------------------


def greedy_merge(
    s: dict,
    mlp: AffinityMLP,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    theta: float,
    r_aff: float,
    max_units: int,
    max_extent: float,
    max_rounds: int = 6,
    device: str = "cuda",
) -> np.ndarray:
    """Merge spatial clusters into instances.

    Repeatedly: predict affinities for current-group pairs within a locality
    radius, merge the strongest pair above ``theta`` subject to size/extent
    caps, then recompute group features (so affinity is evaluated on the
    CURRENT groups, preventing weak-link chains).
    """
    n = int(s["valid"].sum())
    if n == 0:
        return np.zeros(s["valid"].shape[0], dtype=np.int64)
    orig_idx = np.where(s["valid"])[0]
    labels = np.arange(n, dtype=np.int64)
    members = [[i] for i in range(n)]
    max_size = float(s["size"].max(initial=1))
    max_extent = float(s["extent"].max(initial=1.0))

    def _feats(lab: np.ndarray) -> dict:
        ids = np.unique(lab)
        ids = ids[ids >= 0]
        e_c = np.zeros((len(ids), s["e"].shape[1]), dtype=np.float32)
        pos_c = np.zeros((len(ids), 3), dtype=np.float32)
        size_c = np.zeros(len(ids), dtype=np.int64)
        extent_c = np.zeros(len(ids), dtype=np.float32)
        pur_c = np.zeros(len(ids), dtype=np.float32)
        for gi, g in enumerate(ids):
            mem = np.where(lab == g)[0]
            src = orig_idx[mem]
            e_c[gi] = s["e"][src].mean(0)
            nrm = np.linalg.norm(e_c[gi])
            if nrm > 1e-6:
                e_c[gi] /= nrm
            pos_c[gi] = s["pos"][src].mean(0)
            size_c[gi] = int(s["size"][src].sum())
            pd = np.linalg.norm(s["pos"][src][:, None, :] - s["pos"][src][None, :, :], axis=-1)
            extent_c[gi] = float(pd.max()) if len(src) > 1 else 0.0
            pur_c[gi] = float(s["purity"][src].mean())
        return {
            "ids": ids, "e": e_c, "pos": pos_c, "size": size_c,
            "extent": extent_c, "purity": pur_c,
        }

    feats = _feats(labels)
    torch.cuda.empty_cache()
    for _round in range(max_rounds):
        ids = feats["ids"]
        C = len(ids)
        if C <= 1:
            break
        d = np.linalg.norm(feats["pos"][:, None, :] - feats["pos"][None, :, :], axis=-1)
        cand_i, cand_j = np.where((d < r_aff) & (np.arange(C)[:, None] < np.arange(C)[None, :]))
        if cand_i.size == 0:
            break
        feats_all = []
        for a, b in zip(cand_i, cand_j):
            e_mean = (feats["e"][a] + feats["e"][b]) / 2.0
            nrm = np.linalg.norm(e_mean)
            if nrm > 1e-6:
                e_mean = e_mean / nrm
            feats_all.append(
                np.concatenate(
                    [
                        e_mean,
                        np.abs(feats["e"][a] - feats["e"][b]),
                        [float(feats["e"][a] @ feats["e"][b])],
                        (feats["pos"][a] + feats["pos"][b]) / 2.0,
                        np.abs(feats["pos"][a] - feats["pos"][b]),
                        [
                            min(feats["size"][a], feats["size"][b]) / max_size,
                            max(feats["size"][a], feats["size"][b]) / max_size,
                            min(feats["extent"][a], feats["extent"][b]) / max_extent,
                            max(feats["extent"][a], feats["extent"][b]) / max_extent,
                            min(feats["purity"][a], feats["purity"][b]),
                            max(feats["purity"][a], feats["purity"][b]),
                        ],
                    ],
                    dtype=np.float32,
                )
            )
        X = np.stack(feats_all, axis=0)
        X = apply_norm(X, norm_mean, norm_std)
        with torch.no_grad():
            logits = []
            for b0 in range(0, X.shape[0], 8192):
                xb = torch.from_numpy(X[b0 : b0 + 8192]).to(device)
                logits.append(mlp(xb).float().cpu().numpy())
            aff = np.concatenate(logits)
        order = np.argsort(-aff, kind="stable")
        parent = {g: g for g in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merged_any = False
        for oi in order:
            if aff[oi] < theta:
                break
            a, b = int(ids[cand_i[oi]]), int(ids[cand_j[oi]])
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            # size + extent caps (union extent from members is expensive;
            # use a conservative sphere bound).
            ma, mb = members[ra], members[rb]
            if len(ma) + len(mb) > max_units:
                continue
            # extent cap on the union (centroid distance + both radii)
            ra_idx = np.where(ids == a)[0][0]
            rb_idx = np.where(ids == b)[0][0]
            dist_c = float(np.linalg.norm(feats["pos"][ra_idx] - feats["pos"][rb_idx]))
            union_r = dist_c + feats["extent"][ra_idx] / 2.0 + feats["extent"][rb_idx] / 2.0
            if union_r > max_extent:
                continue
            parent[rb] = ra
            members[ra] = ma + mb
            merged_any = True
        if not merged_any:
            break
        # propagate parent to labels
        for g in ids:
            labels[members[g]] = find(g)
        labels = np.array([find(g) for g in labels], dtype=np.int64)
        feats = _feats(labels)
    final = np.zeros(s["valid"].shape[0], dtype=np.int64)
    final[orig_idx] = labels
    # remap to compact ids
    uniq = {c: i for i, c in enumerate(np.unique(final[final >= 0]))}
    out = np.full(s["valid"].shape[0], -1, dtype=np.int64)
    for u in np.where(s["valid"])[0]:
        out[u] = uniq[final[u]]
    return out


# ---------------------------------------------------------------------------
# Model scaffolding (same as the region-growing ablation)
# ---------------------------------------------------------------------------


def patch_model_forward() -> None:
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
            self._aff_buffers = {
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


def build_frozen_model(
    resume: str,
    opt,
) -> torch.nn.Module:
    from safetensors.torch import load_file

    model = model_registry[opt.model_type](opt)
    resume_ckpt = load_file(resume, device="cpu")
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
    model.eval()
    return model.cuda()


def extract_scene(model, data) -> dict:
    """Run one window and return the frozen unit-level internals."""
    batch_size = 1
    with torch.inference_mode():
        model(data, compute_quality_metrics=False)
    br = model.instance_branch
    buf = br._aff_buffers
    token_count = buf["a"].shape[1]
    k = br.units_per_token
    p = buf["a"].shape[2]
    u_count = token_count * k
    n_gs = token_count * p
    e = br.last_unit_embeddings[0].cpu().numpy()  # [U,D]
    pos_norm = buf["pos_norm"][0].cpu().numpy()  # [U,3]
    p_u = br.last_unit_pu[0].cpu().numpy()  # [U,m]
    bg_idx = int(br.last_unit_bg_idx)
    return {
        "e": e,
        "pos": pos_norm,
        "p_u": p_u,
        "bg_idx": bg_idx,
        "a": buf["a"].float(),
        "means": buf["means"].float(),
        "gaussians": buf["gaussians"].float(),
        "cam_view": buf["cam_view"].float(),
        "intrinsics": buf["intrinsics"].float(),
        "token_count": token_count,
        "k": k,
        "p": p,
        "u_count": u_count,
        "n_gs": n_gs,
        "scene_name": str(data["scene_name"][0]),
    }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train_main(args: argparse.Namespace) -> None:
    opt = config_defaults["semantic_v6_unit_shaping_img_dino_train"]
    opt.workspace = args.workspace
    opt.evaluating = False
    opt.batch_size = 1
    opt.num_workers = args.num_workers
    opt.prompt_unfreeze_tokengs = False
    opt.data_mode = (("scannet_prompt_small", 1),)
    opt.dataset_kwargs = {
        "small_manifest_path": WIDE_8X7_MANIFEST,
        "wide_target_subsample": 0,
    }
    patch_model_forward()
    model = build_frozen_model(args.resume, opt)
    _, loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    it = iter(loader)

    mlp: AffinityMLP | None = None
    norm: RunningNorm | None = None
    optim: torch.optim.Optimizer | None = None
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    step = 0
    total_loss = 0.0
    n_windows = 0
    for wi in range(args.train_windows):
        try:
            data = next(it)
        except StopIteration:
            it = iter(loader)
            data = next(it)
        data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in data.items()}
        ex = extract_scene(model, data)
        if mlp is None:
            e_dim = int(ex["e"].shape[1])
            in_dim = 2 * e_dim + 13
            mlp = AffinityMLP(in_dim, hidden=args.hidden).cuda()
            norm = RunningNorm(in_dim)
            optim = torch.optim.AdamW(
                mlp.parameters(), lr=args.lr, weight_decay=1e-4
            )
        fg = np.ones(ex["u_count"], dtype=bool)
        if ex["bg_idx"] >= 0:
            fg = 1.0 - ex["p_u"][:, ex["bg_idx"]] > 0.05
        if not fg.any():
            continue
        lab = spatial_clusters(ex["pos"], args.eps, args.max_units, seed=args.seed)
        n_cl = int(lab.max()) + 1
        s = cluster_summaries(ex["e"], ex["pos"], ex["p_u"], ex["bg_idx"], lab, n_cl)
        pairs, ys = sample_pairs(
            s, args.n_pos, args.n_hard, args.n_easy, args.hard_dist, rng
        )
        if pairs.shape[0] == 0:
            continue
        feats = np.stack(
            [
                pair_features(
                    s, int(i), int(j),
                    max_size=float(s["size"].max(initial=1)),
                    max_extent=float(s["extent"].max(initial=1.0)),
                )
                for i, j in pairs
            ],
            axis=0,
        )
        assert norm is not None and optim is not None and mlp is not None
        norm.update(feats)
        X = torch.from_numpy(apply_norm(feats, *norm.stats())).cuda()
        y = torch.from_numpy(ys).cuda()
        n_windows += 1
        for _ in range(args.steps_per_window):
            perm = torch.randperm(X.shape[0], device=X.device)
            for b0 in range(0, X.shape[0], 256):
                b = perm[b0 : b0 + 256]
                logit = mlp(X[b])
                loss = torch.nn.functional.binary_cross_entropy(logit, y[b])
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
                optim.step()
                total_loss += float(loss)
                step += 1
        if (wi + 1) % args.print_freq == 0:
            print(
                f"[cluster-affinity] win={wi + 1}/{args.train_windows} "
                f"pairs={pairs.shape[0]} y_mean={ys.mean():.3f} "
                f"loss={total_loss / max(step, 1):.4f} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    mean, std = norm.stats()
    torch.save(
        {
            "model": mlp.state_dict(),
            "norm_mean": mean,
            "norm_std": std,
            "in_dim": in_dim,
            "hidden": args.hidden,
            "eps": args.eps,
            "max_units": args.max_units,
            "windows": n_windows,
            "steps": step,
        },
        str(out_dir / "affinity_mlp.pt"),
    )
    (out_dir / "config.yaml").write_text(
        json.dumps(
            {k: v for k, v in vars(args).items() if k != "func"},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[cluster-affinity] trained {step} steps over {n_windows} windows; "
        f"saved {out_dir / 'affinity_mlp.pt'}"
    )


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def load_mlp(path: Path, device: str) -> tuple[AffinityMLP, np.ndarray, np.ndarray]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mlp = AffinityMLP(int(ck["in_dim"]), hidden=int(ck["hidden"])).to(device)
    mlp.load_state_dict(ck["model"])
    mlp.eval()
    return mlp, ck["norm_mean"], ck["norm_std"]


def eval_main(args: argparse.Namespace) -> None:
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
    patch_model_forward()
    model = build_frozen_model(args.resume, opt)
    mlp, norm_mean, norm_std = load_mlp(Path(args.mlp), "cuda")
    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    br = model.instance_branch
    renderer = br.renderer
    render_scale = float(getattr(opt, "instance_group_render_scale", 1.0))
    void_fg_share = float(getattr(opt, "instance_branch_void_fg_share", 0.5))
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [{"name": "learned_affinity", "kind": "aff"}]
    if args.baselines:
        variants += [
            {"name": "agg_avg_eps0.5_pw1", "kind": "agg", "eps": 0.5, "pos_w": 1.0},
            {"name": "oracle_gt", "kind": "oracle"},
        ]
    per_scene: dict[str, dict] = {}
    t_start = time.time()
    for i, data in enumerate(test_loader):
        if args.max_scenes > 0 and i >= args.max_scenes:
            break
        data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in data.items()}
        scene_name = str(data["scene_name"][0])
        ex = extract_scene(model, data)
        fg = np.ones(ex["u_count"], dtype=bool)
        if ex["bg_idx"] >= 0:
            fg = 1.0 - ex["p_u"][:, ex["bg_idx"]] > 0.05
        gs_gt, _, _ = br._pseudo_gs_labels(
            ex["means"].reshape(1, ex["n_gs"], 3), data, opt
        )
        gs_gt_np = gs_gt[0].cpu().numpy()
        a3 = ex["a"][0].cpu().numpy()  # [T, p_gs, k_units]
        u_count = ex["u_count"]
        for variant in variants:
            kind = variant["kind"]
            if kind == "aff":
                lab = spatial_clusters(ex["pos"], args.eps, args.max_units, seed=0)
                s = cluster_summaries(
                    ex["e"], ex["pos"], ex["p_u"], ex["bg_idx"], lab,
                    int(lab.max()) + 1,
                )
                clu_labels = greedy_merge(
                    s, mlp, norm_mean, norm_std,
                    theta=args.theta, r_aff=args.r_aff,
                    max_units=args.max_merge_units,
                    max_extent=args.max_extent,
                    max_rounds=args.max_rounds,
                )
                # propagate cluster-level labels back to units
                labels = np.where(
                    s["valid"][lab], clu_labels[lab], -1
                )
            elif kind == "agg":
                labels = agglomerative_labels(
                    ex["e"], ex["pos"],
                    eps=float(variant["eps"]), pos_w=float(variant["pos_w"]),
                )
            elif kind == "oracle":
                labels = oracle_labels(
                    a3, gs_gt_np.reshape(ex["token_count"], ex["p"])
                )
            else:
                raise ValueError(kind)
            fg_share_u = (
                1.0 - ex["p_u"][:, ex["bg_idx"]]
                if ex["bg_idx"] >= 0
                else np.ones(u_count, dtype=np.float32)
            )
            cluster_list = []
            for c in np.unique(labels):
                if c < 0:
                    continue
                sel = labels == c
                if float(fg_share_u[sel].mean()) >= void_fg_share:
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
                if c in used:
                    onehot[uu, used[c]] = 1.0
            onehot[:, num] = 1.0 - onehot.sum(axis=1)
            unit_probs_t = onehot.reshape(ex["token_count"], ex["k"], num + 1)
            group_probs = np.einsum(
                "tpk,tkl->tpl", a3, unit_probs_t
            ).reshape(ex["n_gs"], num + 1)
            group_probs_t = torch.from_numpy(group_probs).unsqueeze(0).cuda()
            with torch.inference_mode():
                render = renderer.render_feature_channels(
                    ex["gaussians"], group_probs_t, ex["cam_view"],
                    intrinsics=ex["intrinsics"], opacity_scale=render_scale,
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
                    rendered_probs[v], void_channel=num, min_mask_area=1
                )
                masks, scores = masks[:100], scores[:100]
                gts = gt_masks_from_instance_map(gt_maps[v], min_mask_area=1)
                pred_masks.extend(masks)
                pred_scores.extend(scores)
                pred_image_ids.extend([image_id] * len(masks))
                gt_masks.extend(gts)
                gt_image_ids.extend([image_id] * len(gts))
            results = instance_ap(
                pred_masks, pred_scores, gt_masks,
                thresholds=(0.25, 0.5, 0.75), vectorized=True,
                pred_image_ids=pred_image_ids, gt_image_ids=gt_image_ids,
            )
            coco_ap = instance_ap(
                pred_masks, pred_scores, gt_masks,
                thresholds=tuple(t / 100 for t in range(50, 100, 5)),
                vectorized=True,
                pred_image_ids=pred_image_ids, gt_image_ids=gt_image_ids,
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
        print(
            f"[cluster-affinity] {scene_name} done "
            f"({i + 1}/{min(args.max_scenes or 40, 40)}) "
            f"elapsed={time.time() - t_start:.0f}s",
            flush=True,
        )

    def _mean(keys, entries):
        return {
            key: float(np.mean([entries[s][key] for s in entries])) if entries else 0.0
            for key in keys
        }

    summary = {}
    for variant in variants:
        name = variant["name"]
        entries = {s: per_scene[s][name] for s in per_scene if name in per_scene[s]}
        summary[name] = {
            "variant": variant,
            "num_scenes": len(entries),
            **_mean(
                [
                    "ap25", "ap50", "ap75", "ap_mean", "num_pred", "num_gt",
                    "num_clusters", "recall_025", "recall_05",
                    "merging_frac", "mean_instances_per_cluster",
                    "fragmentation_frac", "mean_clusters_per_instance",
                ],
                entries,
            ),
        }
    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "mlp": args.mlp,
        "num_scenes": len(per_scene),
        "fixed": {
            "eps": args.eps,
            "max_units": args.max_units,
            "theta": args.theta,
            "r_aff": args.r_aff,
            "max_merge_units": args.max_merge_units,
            "max_extent": args.max_extent,
            "max_rounds": args.max_rounds,
        },
        "summary": summary,
    }
    (out_dir / "cluster_affinity_eval.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in summary}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--workspace", required=True)
    pt.add_argument("--resume", default=DINO_RESUME)
    pt.add_argument("--gpu", type=int, default=0)
    pt.add_argument("--train_windows", type=int, default=800)
    pt.add_argument("--steps_per_window", type=int, default=1)
    pt.add_argument("--n_pos", type=int, default=1024)
    pt.add_argument("--n_hard", type=int, default=1024)
    pt.add_argument("--n_easy", type=int, default=1024)
    pt.add_argument("--hard_dist", type=float, default=0.9)
    pt.add_argument("--eps", type=float, default=0.35)
    pt.add_argument("--max_units", type=int, default=48)
    pt.add_argument("--hidden", type=int, default=256)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--print_freq", type=int, default=50)
    pt.add_argument("--num_workers", type=int, default=4)
    pt.set_defaults(func=train_main)

    pe = sub.add_parser("eval")
    pe.add_argument("--workspace", required=True)
    pe.add_argument("--resume", default=DINO_RESUME)
    pe.add_argument("--mlp", required=True)
    pe.add_argument("--label", default="cluster_affinity")
    pe.add_argument("--gpu", type=int, default=0)
    pe.add_argument("--model_type", default="semantic_tokengs_v6")
    pe.add_argument("--num_groups", type=int, default=64)
    pe.add_argument("--lsm_manifest", default=(
        "data/scannet_prompt/lsm_instance_eval_manifest.json"
    ))
    pe.add_argument("--max_scenes", type=int, default=0)
    pe.add_argument("--num_input_views", type=int, default=8)
    pe.add_argument("--num_views", type=int, default=15)
    pe.add_argument("--eps", type=float, default=0.35)
    pe.add_argument("--max_units", type=int, default=48)
    pe.add_argument("--theta", type=float, default=0.5)
    pe.add_argument("--r_aff", type=float, default=1.0)
    pe.add_argument("--max_merge_units", type=int, default=512)
    pe.add_argument("--max_extent", type=float, default=1.5)
    pe.add_argument("--max_rounds", type=int, default=6)
    pe.add_argument("--baselines", action="store_true")
    pe.set_defaults(func=eval_main)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.func(args)


if __name__ == "__main__":
    main()
