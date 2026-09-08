"""Offline clustering ablation for the frozen unit-embedding instance branch.

The checkpoint, unit embeddings, unit formation, and frozen Gaussian geometry
are all fixed.  Only the scene-level clustering step is swept:

  - Agglomerative: average / complete / single linkage, distance-threshold eps
  - DBSCAN: eps x min_samples (implemented with scipy cKDTree; scikit-learn is
    not installed in this environment)
  - HDBSCAN: reported as unavailable when the package is missing
  - Embedding-only vs embedding + scene-normalized 3D position (pos weight)
  - Oracle: per-unit dominant GT-instance label (upper bound)

Every variant is rendered through the original frozen Gaussians and scored
with the exact per-scene LSM machinery used by
scripts/eval_instance_lsm_protocol.py (masks_from_group_probs + instance_ap).
No training, no model modification, no new loss.
"""

from __future__ import annotations

import argparse
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

from eval_instance_lsm_protocol import (  # noqa: E402
    _LocalAccelerator,
    _audit_lsm_manifest,
    _load_checkpoint_arch,
)


# ---------------------------------------------------------------------------
# Clustering backends
# ---------------------------------------------------------------------------


def _agg_labels(
    x: np.ndarray, eps: float, linkage_method: str
) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage

    z = linkage(x, method=linkage_method)
    return fcluster(z, t=eps, criterion="distance").astype(np.int64) - 1


def _dbscan_labels(
    x: np.ndarray, eps: float, min_samples: int
) -> np.ndarray:
    """DBSCAN via scipy cKDTree (no scikit-learn dependency)."""
    from scipy.spatial import cKDTree

    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    tree = cKDTree(x)
    nb = tree.query_ball_point(x, r=eps, workers=-1)
    core = np.array([len(s) >= min_samples for s in nb], dtype=bool)
    labels = np.full(n, -1, dtype=np.int64)
    cid = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        labels[i] = cid
        stack = [i]
        while stack:
            j = stack.pop()
            for kk in nb[j]:
                if labels[kk] == -1:
                    labels[kk] = cid
                    if core[kk]:
                        stack.append(kk)
        cid += 1
    return labels


def _oracle_labels(
    a3: np.ndarray, gs_gt3: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit dominant GT-instance label (upper-bound oracle).

    ``a3`` is [T, P, K] soft GS->unit weights, ``gs_gt3`` is [T, P] with 0
    as background.  Units with zero foreground mass are labeled -1 (void).
    Returns (labels, unit_fg_share).
    """
    t, p, k = a3.shape
    fg_ids = np.unique(gs_gt3[gs_gt3 > 0])
    unit_fg = np.ones((t, k), dtype=np.float64)
    if fg_ids.size == 0:
        return np.full(t * k, -1, dtype=np.int64), unit_fg.reshape(-1)
    bg_mass = (a3 * (gs_gt3 == 0)[:, :, None]).sum(axis=1)  # [T,K]
    total = a3.sum(axis=1).clip(min=1e-6)  # [T,K]
    unit_fg = 1.0 - bg_mass / total
    mass = np.zeros((t, k, fg_ids.size), dtype=np.float64)
    for gi, gid in enumerate(fg_ids.tolist()):
        mask = (gs_gt3 == gid)[:, :, None].astype(np.float64)
        mass[:, :, gi] = (a3 * mask).sum(axis=1)
    dom = mass.argmax(axis=2)
    max_mass = mass.max(axis=2)
    labels = np.where(max_mass > 0, dom, -1).astype(np.int64).reshape(-1)
    return labels, unit_fg.reshape(-1)


def _pairwise_labels(
    a3: np.ndarray,
    gs_gt3: np.ndarray,
    mode: str = "share",
    share_eps: float = 0.1,
) -> np.ndarray:
    """GT pairwise-affinity graph oracle -> connected-component labels.

    Per-unit soft mass over GT instances is computed from the frozen
    GS->unit assignment and per-GS pseudo-GT instance ids.  Two modes:

    - "dom": edge iff two units share the same dominant GT instance
      (equivalent to the label-based oracle, validates the graph path).
    - "share": edge iff both units have relative mass >= share_eps on some
      common GT instance (co-occurrence through mixed units).

    Returns per-unit component ids (-1 for units with no foreground mass).
    """
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
    labels = np.full(u, -1, dtype=np.int64)

    def _find(parent: np.ndarray, x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def _union(parent: np.ndarray, a: int, b: int) -> None:
        ra, rb = _find(parent, a), _find(parent, b)
        if ra != rb:
            parent[rb] = ra

    if mode == "dom":
        dom = mass.argmax(axis=1)
        comp: dict[int, int] = {}
        cid = 0
        for i in np.where(fg)[0]:
            key = int(dom[i])
            if key not in comp:
                comp[key] = cid
                cid += 1
            labels[i] = comp[key]
        return labels

    rel = mass / np.maximum(total[:, None], 1e-6)
    parent = np.arange(u)
    for gi in range(fg_ids.size):
        members = np.where(rel[:, gi] >= share_eps)[0]
        for j in range(1, members.size):
            _union(parent, int(members[0]), int(members[j]))
    comp_map: dict[int, int] = {}
    cid = 0
    for i in np.where(fg)[0]:
        r = _find(parent, int(i))
        if r not in comp_map:
            comp_map[r] = cid
            cid += 1
        labels[i] = comp_map[r]
    return labels


def _unit_rel_mass(
    a3: np.ndarray, gs_gt3: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit relative mass over foreground GT instances.

    Returns (rel [U,F], fg_mask [U]) where rel[u,i] is the fraction of
    unit u's GS mass that belongs to foreground instance i.
    """
    t, p, k = a3.shape
    fg_ids = np.unique(gs_gt3[gs_gt3 > 0])
    u = t * k
    if fg_ids.size == 0:
        return np.zeros((u, 0), dtype=np.float64), np.zeros(u, dtype=bool)
    mass = np.zeros((t, k, fg_ids.size), dtype=np.float64)
    for gi, gid in enumerate(fg_ids.tolist()):
        mask = (gs_gt3 == gid)[:, :, None].astype(np.float64)
        mass[:, :, gi] = (a3 * mask).sum(axis=1)
    mass = mass.reshape(u, fg_ids.size)
    total = mass.sum(axis=1)
    rel = mass / np.maximum(total[:, None], 1e-6)
    fg = total > 0
    return rel, fg


def _spectral_from_factor(x: np.ndarray, k: int) -> np.ndarray:
    """Normalized spectral clustering on affinity A = x @ x.T.

    The normalized Laplacian's smallest eigenvectors are computed exactly
    through the low-rank factorization B = D^-1/2 A D^-1/2 = X X^T with
    X = D^-1/2 x, avoiding the U x U matrix (U = 8192).  Rows of the top-k
    eigenvectors are then k-means clustered (scipy kmeans2).
    """
    from scipy.cluster.vq import kmeans2

    u, f = x.shape
    if u == 0:
        return np.zeros(u, dtype=np.int64)
    k = int(max(1, min(k, f, u)))
    row_sums = x @ x.sum(axis=0)  # degree of A
    x_scaled = x / np.sqrt(np.maximum(row_sums[:, None], 1e-6))
    u_svd, s, _ = np.linalg.svd(x_scaled, full_matrices=False)
    vecs = u_svd[:, :k]
    vecs = vecs / np.maximum(
        np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8
    )
    if k == 1:
        return np.zeros(u, dtype=np.int64)
    centroids, labels = kmeans2(
        vecs, k, minit="++", iter=100, seed=0
    )
    return labels.astype(np.int64)


def _spectral_labels(rel: np.ndarray, k: int) -> np.ndarray:
    """Spectral clustering on the soft-overlap affinity rel @ rel.T."""
    return _spectral_from_factor(rel, k)


def _spectral_auto_k_from_factor(x: np.ndarray) -> int:
    """Eigengap estimate of the number of clusters from the affinity graph."""
    u, f = x.shape
    if f == 0:
        return 1
    row_sums = x @ x.sum(axis=0)
    x_scaled = x / np.sqrt(np.maximum(row_sums[:, None], 1e-6))
    _, s, _ = np.linalg.svd(x_scaled, full_matrices=False)
    s = s[: min(f, 60)]
    if s.size < 2:
        return int(f)
    floor = 0.05 * s[0]
    gap_idx = np.where(s[:-1] - s[1:] > floor * 0.1)[0]
    if gap_idx.size == 0:
        return int(min(f, 20))
    k = int(gap_idx[np.argmax(s[gap_idx] - s[gap_idx + 1])] + 1)
    return int(max(3, min(k, f)))


def _spectral_auto_k(rel: np.ndarray) -> int:
    return _spectral_auto_k_from_factor(rel)


def _maxclust_labels(
    rel: np.ndarray, k: int, linkage_method: str
) -> np.ndarray:
    """Agglomerative with maxclust=K on distance 1 - soft overlap."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    u, f = rel.shape
    if f == 0 or u == 0:
        return np.zeros(u, dtype=np.int64)
    aff = rel @ rel.T
    d = np.clip(1.0 - aff, 0.0, None).astype(np.float32)
    condensed = squareform(d, checks=False)
    z = linkage(condensed, method=linkage_method)
    labels = fcluster(z, t=int(k), criterion="maxclust").astype(np.int64) - 1
    return labels


def _cc_greedy_from_aff(aff: np.ndarray) -> np.ndarray:
    """Greedy correlation clustering on a signed affinity matrix.

    Units are processed by decreasing total affinity and joined to the
    cluster with the largest positive agreement, otherwise a new cluster
    starts.  No cluster count is required.
    """
    u = aff.shape[0]
    if u == 0:
        return np.zeros(u, dtype=np.int64)
    signed = aff
    order = np.argsort(-aff.sum(axis=1), kind="stable")
    labels = np.full(u, -1, dtype=np.int64)
    members: list[list[int]] = []
    for idx in order:
        best_c, best_gain = -1, 0.0
        for c, mem in enumerate(members):
            gain = float(signed[idx, mem].sum())
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c < 0:
            best_c = len(members)
            members.append([])
        members[best_c].append(int(idx))
        labels[idx] = best_c
    return labels


def _cc_greedy_labels(rel: np.ndarray) -> np.ndarray:
    """Greedy correlation clustering on soft-overlap affinity."""
    if rel.shape[0] == 0 or rel.shape[1] == 0:
        return np.zeros(rel.shape[0], dtype=np.int64)
    return _cc_greedy_from_aff(rel @ rel.T)


def _cluster_scene(
    variant: dict,
    e: np.ndarray,
    pos_norm: np.ndarray,
    a3: np.ndarray,
    gs_gt3: np.ndarray,
) -> np.ndarray:
    """Run one clustering variant and return per-unit labels (-1 = void)."""
    kind = variant["kind"]
    if kind == "oracle":
        labels, _ = _oracle_labels(a3, gs_gt3)
        return labels
    if kind == "oracle_pair":
        return _pairwise_labels(
            a3,
            gs_gt3,
            mode=variant.get("pair_mode", "share"),
            share_eps=float(variant.get("share_eps", 0.1)),
        )
    rel, fg = _unit_rel_mass(a3, gs_gt3)
    labels = np.full(rel.shape[0], -1, dtype=np.int64)
    if not fg.any():
        return labels
    rel_fg = rel[fg]
    if kind == "oracle_spec":
        if variant.get("k") is not None:
            k = int(variant["k"])
        elif variant.get("auto_k", False):
            k = _spectral_auto_k(rel_fg)
        else:
            k = rel_fg.shape[1]  # GT instance count
        labels[fg] = _spectral_labels(rel_fg, k)
        return labels
    if kind == "oracle_maxclust":
        labels[fg] = _maxclust_labels(
            rel_fg,
            int(variant["k"]) if variant.get("k") is not None
            else rel_fg.shape[1],
            variant.get("linkage", "average"),
        )
        return labels
    if kind == "oracle_cc":
        labels[fg] = _cc_greedy_labels(rel_fg)
        return labels
    if kind == "emb_cc":
        # Greedy correlation clustering on the identity-embedding cosine
        # affinity (signed; negative cosine = repulsion).
        aff = e @ e.T
        return _cc_greedy_from_aff(aff)
    if kind == "emb_cen_cc":
        # Mean-centered cosine (Pearson correlation): removes the common
        # mode that makes raw cosine affinity dense and monolithically
        # positive (same=0.85 / diff=0.67 on this embedding).
        ec = e - e.mean(axis=0, keepdims=True)
        return _cc_greedy_from_aff(ec @ ec.T)
    if kind == "emb_spec":
        # Normalized spectral clustering on the non-negative affinity
        # (cos+1)/2 = X X^T with X = [e/sqrt(2), 1/sqrt(2)].
        x = np.concatenate(
            [
                e / np.sqrt(2.0),
                np.ones((e.shape[0], 1), dtype=e.dtype) / np.sqrt(2.0),
            ],
            axis=1,
        )
        if variant.get("k") is not None:
            k = int(variant["k"])
        elif variant.get("auto_k", False):
            k = _spectral_auto_k_from_factor(x)
        else:
            rel, _ = _unit_rel_mass(a3, gs_gt3)
            k = max(1, int(rel.shape[1]))  # GT-count reference
        return _spectral_from_factor(x, k)
    if kind == "emb_cen_spec":
        # Spectral on (correlation+1)/2 = X X^T with
        # X = [e_cn/sqrt(2), 1/sqrt(2)] (non-negative, centered).
        ec = e - e.mean(axis=0, keepdims=True)
        e_cn = ec / np.maximum(
            np.linalg.norm(ec, axis=1, keepdims=True), 1e-8
        )
        x = np.concatenate(
            [
                e_cn / np.sqrt(2.0),
                np.ones((e_cn.shape[0], 1), dtype=e_cn.dtype)
                / np.sqrt(2.0),
            ],
            axis=1,
        )
        if variant.get("k") is not None:
            k = int(variant["k"])
        elif variant.get("auto_k", False):
            k = _spectral_auto_k_from_factor(x)
        else:
            rel, _ = _unit_rel_mass(a3, gs_gt3)
            k = max(1, int(rel.shape[1]))
        return _spectral_from_factor(x, k)
    pw = float(variant.get("pos_w", 1.0))
    feat = (
        np.concatenate([e, pw * pos_norm], axis=-1)
        if pw > 0
        else e
    ).astype(np.float32)
    if kind == "agg":
        return _agg_labels(
            feat, float(variant["eps"]), variant.get("linkage", "average")
        )
    if kind == "dbscan":
        return _dbscan_labels(
            feat, float(variant["eps"]), int(variant.get("min_samples", 5))
        )
    if kind == "hdbscan":
        try:
            import hdbscan  # type: ignore

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(variant.get("min_cluster_size", 5)),
                min_samples=int(variant.get("min_samples", 5)),
                metric="euclidean",
            )
            clusterer.fit(feat)
            return clusterer.labels_.astype(np.int64)
        except Exception as exc:  # pragma: no cover - env fallback
            print(f"[clustering-ablation] hdbscan unavailable: {exc}")
            return np.zeros(feat.shape[0], dtype=np.int64)
    raise ValueError(f"unknown cluster kind {kind}")


# ---------------------------------------------------------------------------
# Variant grid
# ---------------------------------------------------------------------------


def _build_variants(include_hdbscan: bool = True) -> list[dict]:
    variants: list[dict] = []
    for eps in (0.4, 0.5, 0.6, 0.7):
        for pw in (0.0, 1.0, 2.0):
            variants.append(
                {
                    "name": f"agg_avg_eps{eps:.1f}_pw{pw:.0f}",
                    "kind": "agg",
                    "linkage": "average",
                    "eps": eps,
                    "pos_w": pw,
                }
            )
    for eps in (0.5, 0.7, 0.9):
        variants.append(
            {
                "name": f"agg_complete_eps{eps:.1f}_pw1",
                "kind": "agg",
                "linkage": "complete",
                "eps": eps,
                "pos_w": 1.0,
            }
        )
    for eps in (0.5, 0.7):
        variants.append(
            {
                "name": f"agg_single_eps{eps:.1f}_pw1",
                "kind": "agg",
                "linkage": "single",
                "eps": eps,
                "pos_w": 1.0,
            }
        )
    for pw in (0.0, 1.0):
        for eps in (0.5, 0.7, 0.9):
            for ms in (5, 20):
                variants.append(
                    {
                        "name": f"dbscan_eps{eps:.1f}_ms{ms}_pw{pw:.0f}",
                        "kind": "dbscan",
                        "eps": eps,
                        "min_samples": ms,
                        "pos_w": pw,
                    }
                )
    if include_hdbscan:
        try:
            import hdbscan  # noqa: F401
        except Exception:
            print("[clustering-ablation] hdbscan not installed; skipping")
        else:
            for pw in (0.0, 1.0):
                for mcs in (5, 20):
                    variants.append(
                        {
                            "name": f"hdbscan_mcs{mcs}_pw{pw:.0f}",
                            "kind": "hdbscan",
                            "min_cluster_size": mcs,
                            "min_samples": 5,
                            "pos_w": pw,
                        }
                    )
    variants.append({"name": "oracle_gt", "kind": "oracle"})
    variants.append(
        {
            "name": "oracle_gt_anyfg",
            "kind": "oracle",
            "no_void_filter": True,
        }
    )
    variants.append(
        {
            "name": "oracle_pair_dom",
            "kind": "oracle_pair",
            "pair_mode": "dom",
            "no_void_filter": True,
        }
    )
    for eps in (0.05, 0.1, 0.2):
        variants.append(
            {
                "name": f"oracle_pair_share_{eps:.2f}",
                "kind": "oracle_pair",
                "pair_mode": "share",
                "share_eps": eps,
                "no_void_filter": True,
            }
        )
    # GT pairwise affinity + global constrained clustering (no model
    # predictions): known-count and auto-count settings.
    variants.append(
        {
            "name": "oracle_spec_gtk",
            "kind": "oracle_spec",
            "k": None,  # resolved to GT instance count per scene
            "no_void_filter": True,
        }
    )
    variants.append(
        {
            "name": "oracle_spec_auto",
            "kind": "oracle_spec",
            "k": None,  # eigengap estimate
            "auto_k": True,
            "no_void_filter": True,
        }
    )
    variants.append(
        {
            "name": "oracle_agg_avg_maxclust_gt",
            "kind": "oracle_maxclust",
            "linkage": "average",
            "k": None,
            "no_void_filter": True,
        }
    )
    variants.append(
        {
            "name": "oracle_agg_single_maxclust_gt",
            "kind": "oracle_maxclust",
            "linkage": "single",
            "k": None,
            "no_void_filter": True,
        }
    )
    variants.append(
        {
            "name": "oracle_cc_greedy",
            "kind": "oracle_cc",
            "no_void_filter": True,
        }
    )
    # Identity-embedding cosine affinity + global grouping (replaces the
    # Agglomerative; no GT in the grouping itself).
    variants.append(
        {
            "name": "emb_cos_cc_greedy",
            "kind": "emb_cc",
        }
    )
    variants.append(
        {
            "name": "emb_cos_spec_auto",
            "kind": "emb_spec",
            "auto_k": True,
        }
    )
    variants.append(
        {
            "name": "emb_cos_spec_gtk",
            "kind": "emb_spec",
            "k": None,  # GT-instance count reference
        }
    )
    variants.append(
        {
            "name": "emb_cen_cc_greedy",
            "kind": "emb_cen_cc",
        }
    )
    variants.append(
        {
            "name": "emb_cen_spec_auto",
            "kind": "emb_cen_spec",
            "auto_k": True,
        }
    )
    variants.append(
        {
            "name": "emb_cen_spec_gtk",
            "kind": "emb_cen_spec",
            "k": None,
        }
    )
    return variants


# ---------------------------------------------------------------------------
# Mask-level + GS-level diagnostics
# ---------------------------------------------------------------------------


def _max_iou_recall(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    thresholds=(0.25, 0.5),
) -> dict:
    """Fraction of GT masks with any prediction IoU >= t (same image)."""
    recalls = {t: 0.0 for t in thresholds}
    if not pred_masks or not gt_masks:
        for t in thresholds:
            recalls[t] = 0.0
        return recalls
    n_g = len(gt_masks)
    n_p = len(pred_masks)
    h, w = gt_masks[0].shape
    pm = np.stack(pred_masks).reshape(n_p, -1).astype(np.float32)
    gm = np.stack(gt_masks).reshape(n_g, -1).astype(np.float32)
    p_sum = pm.sum(1)
    g_sum = gm.sum(1)
    overlap = gm @ pm.T  # [n_g, n_p]
    union = g_sum[:, None] + p_sum[None, :] - overlap
    iou = overlap / np.maximum(union, 1e-6)
    best = iou.max(axis=1)
    for t in thresholds:
        recalls[t] = float((best >= t).mean()) if n_g else 0.0
    return recalls


def _gs_level_stats(
    group_probs: np.ndarray,
    gs_gt: np.ndarray,
    n_clusters: int,
) -> dict:
    """Merging / fragmentation computed on per-GS pseudo-GT instance ids."""
    stats = {
        "merging_frac": 0.0,
        "mean_instances_per_cluster": 0.0,
        "fragmentation_frac": 0.0,
        "mean_clusters_per_instance": 0.0,
    }
    if n_clusters <= 0:
        return stats
    probs = group_probs[:, :n_clusters]  # [N, C]
    total_mass = probs.sum(0)  # [C]
    fg_ids = np.unique(gs_gt[gs_gt > 0])
    per_cluster = []
    for c in range(n_clusters):
        w = probs[:, c]
        t = max(float(total_mass[c]), 1e-6)
        if t < 1e-4:
            per_cluster.append(0)
            continue
        distinct = 0
        for gid in fg_ids.tolist():
            m = float(w[gs_gt == gid].sum())
            if m / t >= 0.05:
                distinct += 1
        per_cluster.append(distinct)
    per_cluster = np.asarray(per_cluster, dtype=np.float64)
    stats["mean_instances_per_cluster"] = float(per_cluster.mean())
    stats["merging_frac"] = float((per_cluster >= 2).mean())

    per_instance = []
    for gid in fg_ids.tolist():
        sel = gs_gt == gid
        w_g = probs[sel]
        t_g = max(float(w_g.sum()), 1e-6)
        share = w_g.sum(0) / t_g
        per_instance.append(int((share >= 0.05).sum()))
    if per_instance:
        per_instance = np.asarray(per_instance, dtype=np.float64)
        stats["mean_clusters_per_instance"] = float(per_instance.mean())
        stats["fragmentation_frac"] = float((per_instance >= 2).mean())
    return stats


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
        default="workspace/clustering_ablation_unit_shaping_img_2000",
    )
    parser.add_argument("--label", default="unit_shaping_img_2000")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument("--lsm_manifest", default=(
        "data/scannet_prompt/lsm_instance_eval_manifest.json"
    ))
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=15)
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated variant names; empty runs the full grid.",
    )
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

    variants = _build_variants(include_hdbscan=True)
    if args.variants:
        wanted = set(name.strip() for name in args.variants.split(","))
        variants = [v for v in variants if v["name"] in wanted]
    if not variants:
        raise ValueError("no variants selected")
    print(f"[clustering-ablation] {len(variants)} variants")

    # ------------------------------------------------------------------
    # Patch the model forward to expose the frozen internals.  This only
    # records activations; formation/embedding/geometry are untouched.
    # ------------------------------------------------------------------
    _orig_forward = TokenLocalUnitGrouping._forward_unit_embedding

    def _patched_forward(
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
        out = _orig_forward(
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
            u_count = token_count * self.units_per_token
            center = unit_center.reshape(batch_size, u_count, 3).float()
            scene_center = center.mean(dim=1, keepdim=True)
            scene_scale = (
                (center - scene_center).square().mean(dim=(1, 2), keepdim=True)
                .sqrt()
                .clamp_min(1e-3)
            )
            self._ablation_buffers = {
                "a": a.detach().float(),  # [B,T,P,K]
                "unit_center": unit_center.detach().float(),  # [B,T,K,3]
                "pos_norm": (
                    (center - scene_center) / scene_scale
                ).detach().float(),  # [B,U,3]
                "means": means.detach().float(),  # [B,T,P,3]
                "gaussians": gaussians.detach().float(),  # [B,N,14]
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
                f"[clustering-ablation] frozen-backbone eval: loaded "
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
        buf = br._ablation_buffers
        batch_size = 1
        token_count = buf["a"].shape[1]
        k = br.units_per_token
        p = buf["a"].shape[2]
        u_count = token_count * k
        n_gs = token_count * p
        a_flat = buf["a"][0].reshape(n_gs, k).cpu().numpy()
        e = br.last_unit_embeddings[0].cpu().numpy()  # [U,D]
        pos_norm = buf["pos_norm"][0].cpu().numpy()  # [U,3]
        gaussians = buf["gaussians"].float()  # [1,N,14] (CUDA)
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

        for variant in variants:
            labels = _cluster_scene(
                variant,
                e,
                pos_norm,
                a_flat.reshape(token_count, p, k),
                gs_gt_np.reshape(token_count, p),
            )
            # One-hot with void filtering (same convention as the model).
            cluster_list: list[int] = []
            for c in np.unique(labels):
                if c < 0:
                    continue
                sel = labels == c
                if variant.get("no_void_filter"):
                    fg_share = 1.0
                else:
                    fg_share = float(fg_share_u[sel].mean())
                if fg_share >= void_fg_share:
                    cluster_list.append(int(c))
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
                "tpk,tkl->tpl",
                a_flat.reshape(token_count, p, k),
                unit_probs_t,
            ).reshape(n_gs, num + 1)  # [N, C+1]
            group_probs_t = torch.from_numpy(group_probs).unsqueeze(0).cuda()
            with torch.inference_mode():
                render = renderer.render_feature_channels(
                    gaussians,
                    group_probs_t,
                    cam_view,
                    intrinsics=intrinsics,
                    opacity_scale=render_scale,
                )
            rendered_channels = (
                render["images_pred"].cpu().numpy()[0]
                / (render["alphas_pred"].cpu().numpy()[0] + 1e-5)
            )  # [V, C+1, H, W]
            rendered_probs = rendered_channels / np.maximum(
                rendered_channels.sum(axis=1, keepdims=True), 1e-6
            )  # [V, C+1, H, W]
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
                    probs,
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
            f"[clustering-ablation] {scene_name} done "
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
                    "ap25",
                    "ap50",
                    "ap75",
                    "ap_mean",
                    "num_pred",
                    "num_gt",
                    "num_clusters",
                    "recall_025",
                    "recall_05",
                    "merging_frac",
                    "mean_instances_per_cluster",
                    "fragmentation_frac",
                    "mean_clusters_per_instance",
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
            "clustering_feature": "concat([L2-norm embedding, pw * scene-normalized pos])",
        },
        "reference_AP50": 0.279,
        "variants": summary,
    }
    out_json = out_dir / "clustering_ablation.json"
    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    per_scene_json = out_dir / "clustering_ablation_per_scene.json"
    per_scene_json.write_text(
        json.dumps(per_scene, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n[clustering-ablation] summary (scene-macro mean):")
    print(
        f"{'variant':32s} {'AP25':>6s} {'AP50':>6s} {'AP':>6s} "
        f"{'pred/gt':>9s} {'recall.5':>8s} {'merge':>6s} {'frag':>6s}"
    )
    for name, entry in summary.items():
        print(
            f"{name:32s} {entry['ap25']:6.3f} {entry['ap50']:6.3f} "
            f"{entry['ap_mean']:6.3f} "
            f"{int(entry['num_pred'])}/{int(entry['num_gt']):<4d} "
            f"{entry['recall_05']:8.3f} {entry['merging_frac']:6.2f} "
            f"{entry['fragmentation_frac']:6.2f}"
        )
    print(f"[clustering-ablation] wrote {out_json}")


if __name__ == "__main__":
    main()
