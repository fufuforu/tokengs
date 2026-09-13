"""Historical metric-embedding agglomerative inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MetricClusterOutput:
    gaussian_cluster_ids: torch.Tensor
    cluster_count: list[int]
    cluster_confidence: list[torch.Tensor]
    rendered_masks: torch.Tensor
    rendered_scores: torch.Tensor


def _cluster_labels(features: np.ndarray, eps: float) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage

    if features.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if features.shape[0] == 1:
        return np.zeros((1,), dtype=np.int64)
    # This is the exact recovered historical algorithm for the 8192-unit
    # regime: average linkage and a distance cut, with no learned threshold.
    tree = linkage(features, method="average")
    return fcluster(tree, t=float(eps), criterion="distance").astype(np.int64) - 1


def historical_metric_cluster(
    unit_embeddings: torch.Tensor,
    unit_positions: torch.Tensor,
    unit_objectness: torch.Tensor,
    gaussians: object,
    cameras: object,
    *,
    eps: float = 0.5,
    foreground_threshold: float = 0.5,
    foreground_share: float = 0.5,
) -> MetricClusterOutput:
    """GT-free historical metric clustering with predicted objectness.

    ``gaussians`` and ``cameras`` may carry an optional renderer.  Clustering
    itself is always deterministic CPU scipy inference.  When no renderer is
    supplied, the returned render tensors are empty rather than inventing a
    second rendering protocol; the evaluator supplies the existing renderer.
    """
    if unit_embeddings.ndim != 3 or unit_positions.ndim != 3:
        raise ValueError("metric cluster inputs must be [B,U,D] and [B,U,3]")
    if unit_objectness.ndim != 2:
        raise ValueError("unit_objectness must be [B,U]")
    if unit_embeddings.shape[:2] != unit_positions.shape[:2] or unit_positions.shape[-1] != 3:
        raise ValueError("metric cluster layout mismatch")
    if unit_objectness.shape != unit_embeddings.shape[:2]:
        raise ValueError("unit_objectness layout mismatch")
    if not np.isfinite(float(eps)) or float(eps) <= 0:
        raise ValueError("cluster eps must be finite and positive")
    if not 0.0 <= float(foreground_threshold) <= 1.0:
        raise ValueError("foreground_threshold must be in [0,1]")
    if not 0.0 <= float(foreground_share) <= 1.0:
        raise ValueError("foreground_share must be in [0,1]")
    b, units, _ = unit_positions.shape
    if not torch.isfinite(unit_objectness).all():
        raise FloatingPointError("unit_objectness contains NaN/Inf")
    emb = F.normalize(unit_embeddings.float(), dim=-1).detach().cpu().numpy()
    objectness = unit_objectness.float().detach().cpu().numpy()
    pos = unit_positions.float()
    center = pos.mean(dim=1, keepdim=True)
    scale = (pos - center).square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-3)
    pos_norm = ((pos - center) / scale).detach().cpu().numpy()
    labels = np.stack(
        [_cluster_labels(np.concatenate([emb[i], pos_norm[i]], axis=-1), float(eps)) for i in range(b)],
        axis=0,
    ) if units else np.zeros((b, 0), dtype=np.int64)
    filtered_labels = []
    counts = []
    cluster_confidence = []
    for batch_index in range(b):
        remap = {}
        confidence = []
        for raw_cluster in np.unique(labels[batch_index]):
            members = labels[batch_index] == raw_cluster
            member_foreground = objectness[batch_index, members] >= float(
                foreground_threshold
            )
            share = float(member_foreground.mean()) if members.any() else 0.0
            if share >= float(foreground_share):
                remap[int(raw_cluster)] = len(confidence)
                confidence.append(float(objectness[batch_index, members].mean()))
        mapped = np.full((units,), len(confidence), dtype=np.int64)
        for raw_cluster, cluster_id in remap.items():
            mapped[labels[batch_index] == raw_cluster] = cluster_id
        filtered_labels.append(mapped)
        counts.append(len(confidence))
        cluster_confidence.append(
            torch.tensor(confidence, dtype=torch.float32, device=unit_embeddings.device)
        )
    filtered_labels = (
        np.stack(filtered_labels, axis=0)
        if b
        else np.zeros((0, units), dtype=np.int64)
    )
    gaussian_ids = torch.from_numpy(
        np.repeat(filtered_labels, 8, axis=1)
    ).to(unit_embeddings.device)
    renderer = gaussians.get("renderer") if isinstance(gaussians, dict) else None
    rendered_masks = unit_embeddings.new_empty((0,))
    rendered_scores = unit_embeddings.new_empty((0,))
    if renderer is not None and units:
        values = gaussians["values"] if isinstance(gaussians, dict) else gaussians
        if values.ndim != 3 or values.shape[1] != units * 8:
            raise ValueError("Gaussian order is not aligned to 8 children per unit")
        n_clusters = max(counts) if counts else 0
        channel_count = n_clusters + 1
        onehot = F.one_hot(gaussian_ids, num_classes=channel_count).to(values.dtype)
        render = renderer.render_feature_channels(
            values.float(),
            onehot,
            cameras["cam_view"] if isinstance(cameras, dict) else cameras.cam_view,
            intrinsics=(cameras.get("intrinsics") if isinstance(cameras, dict) else cameras.intrinsics),
        )
        images = render["images_pred"]
        alpha = render["alphas_pred"]
        probs = (images / (alpha + 1e-5)).float()
        probs = probs / probs.sum(dim=2, keepdim=True).clamp_min(1e-6)
        rendered_masks = probs.permute(0, 2, 1, 3, 4).unsqueeze(3)
        # Match masks_from_group_probs: confidence is the mean winning
        # probability over the pixels assigned to that cluster, per target
        # view.  Keeping this alongside the masks makes the output usable by
        # an evaluator without introducing a second score convention.
        winning = probs.argmax(dim=2)  # [B,V,H,W]
        score_rows = []
        for batch_index in range(probs.shape[0]):
            per_cluster = []
            for cluster_index in range(n_clusters):
                per_view = []
                for view_index in range(probs.shape[1]):
                    mask = winning[batch_index, view_index] == cluster_index
                    values = probs[batch_index, view_index, cluster_index][mask]
                    per_view.append(
                        values.mean()
                        if values.numel()
                        else probs.new_zeros(())
                    )
                per_cluster.append(torch.stack(per_view))
            score_rows.append(torch.stack(per_cluster))
        rendered_scores = torch.stack(score_rows)
    return MetricClusterOutput(
        gaussian_cluster_ids=gaussian_ids,
        cluster_count=counts,
        cluster_confidence=cluster_confidence,
        rendered_masks=rendered_masks,
        rendered_scores=rendered_scores,
    )


def historical_metric_cluster_oracle_audit(
    unit_embeddings: torch.Tensor,
    unit_positions: torch.Tensor,
    p_u_gt: torch.Tensor,
    gaussians: object,
    cameras: object,
    *,
    background_index: int,
    eps: float = 0.5,
    foreground_threshold: float = 0.5,
    foreground_share: float = 0.5,
) -> MetricClusterOutput:
    """GT-derived diagnostic-only reproduction of the historical filter.

    This function is intentionally named ``oracle_audit`` and is not called
    by the model forward or the formal evaluator.  The historical p_u tensor
    is a GT-derived soft instance distribution; converting its background
    column to an oracle foreground share is retained solely to quantify how
    much the old GT-assisted filtering helped.
    """
    if p_u_gt.ndim != 3 or p_u_gt.shape[:2] != unit_embeddings.shape[:2]:
        raise ValueError("p_u_gt must be [B,U,M] aligned with unit embeddings")
    if not 0 <= int(background_index) < p_u_gt.shape[-1]:
        raise ValueError("background_index is outside p_u_gt")
    if not torch.isfinite(p_u_gt).all():
        raise FloatingPointError("p_u_gt contains NaN/Inf")
    oracle_objectness = 1.0 - p_u_gt[..., int(background_index)].float()
    return historical_metric_cluster(
        unit_embeddings,
        unit_positions,
        oracle_objectness,
        gaussians,
        cameras,
        eps=eps,
        foreground_threshold=foreground_threshold,
        foreground_share=foreground_share,
    )
