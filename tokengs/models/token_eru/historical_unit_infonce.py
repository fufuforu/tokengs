"""The recovered historical soft unit InfoNCE objective.

The implementation mirrors ``TokenLocalUnitGrouping._soft_info_nce`` in
``instance_group_head.py``: soft pseudo-instance distributions define pair
weights, the diagonal is excluded from positives, and all valid units are
used (the recovered 0.322 configuration has no embedding subsampling).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from tokengs.models.instance_group_loss import (
    _gs_majority_target,
    _project_gs_to_views,
)


@dataclass(frozen=True)
class MetricLossOutput:
    loss: torch.Tensor
    valid_unit_count: int
    positive_pair_count: int
    negative_pair_count: int
    mean_positive_similarity: torch.Tensor
    mean_negative_similarity: torch.Tensor
    target_entropy: torch.Tensor


def build_historical_soft_unit_targets(
    base_gaussians: torch.Tensor,
    data: dict,
    image_size: tuple[int, int],
    *,
    num_tokens: int = 1024,
    units_per_token: int = 8,
    gaussians_per_unit: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the historical soft pseudo-instance distribution in ERU order.

    Historical pseudo labels are the majority vote of projected labels across
    the 8 context and 7 target views.  The ERU unit layout is exactly eight
    consecutive child Gaussians per unit, so the recovered soft target is the
    normalized histogram over those children.  Background (0) remains a
    valid pseudo class; ignore/void labels 255 and -1 are excluded.
    """
    if base_gaussians.ndim != 3 or base_gaussians.shape[1] != num_tokens * units_per_token * gaussians_per_unit:
        raise ValueError("base_gaussians is not in the historical 1024x8x8 layout")
    views = torch.cat([data["cam_view_input"], data["cam_view"]], dim=1)
    intrinsics = torch.cat([data["intrinsics_input"], data["intrinsics"]], dim=1)
    labels = torch.cat(
        [data["instance_label_input"], data["instance_label_output"]], dim=1
    ).long()
    all_targets = []
    all_valid = []
    for b in range(base_gaussians.shape[0]):
        ids, valid = _project_gs_to_views(
            base_gaussians[b, ..., :3],
            views[b],
            intrinsics[b],
            labels[b],
            image_size,
        )
        # The recovered majority helper operates on [V,N].  Only after the
        # per-Gaussian vote is complete may the result be grouped into the
        # historical 1024x8x8 child layout.
        projected_ids = ids
        projected_valid = valid
        gs_ids = _gs_majority_target(projected_ids, projected_valid)
        ids = projected_ids.reshape(-1, num_tokens, units_per_token, gaussians_per_unit)
        valid = projected_valid.reshape_as(ids)
        gs_ids = gs_ids.reshape(num_tokens, units_per_token, gaussians_per_unit)
        # Treat the majority target as the recovered historical label, while
        # retaining the historical visibility requirement for each child.
        gs_valid = projected_valid.any(dim=0).reshape_as(gs_ids) & (gs_ids != 255) & (gs_ids != -1)
        present = torch.unique(gs_ids[gs_valid])
        present = present[(present != 255) & (present != -1)]
        if present.numel() == 0:
            present = torch.zeros(1, dtype=torch.long, device=ids.device)
        counts = gs_ids.reshape(num_tokens, units_per_token, gaussians_per_unit, 1).eq(
            present.view(1, 1, 1, -1)
        )
        counts = (counts & gs_valid.reshape(num_tokens, units_per_token, gaussians_per_unit, 1)).sum(dim=2).float()
        mass = counts.sum(dim=-1, keepdim=True)
        target = counts / mass.clamp_min(1.0)
        unit_valid = mass.squeeze(-1) > 0
        all_targets.append(target.reshape(num_tokens * units_per_token, -1))
        all_valid.append(unit_valid.reshape(-1))
    width = max(target.shape[-1] for target in all_targets)
    padded = [F.pad(target, (0, width - target.shape[-1])) for target in all_targets]
    return torch.stack(padded), torch.stack(all_valid)


def historical_soft_unit_infonce(
    unit_embeddings: torch.Tensor,
    unit_targets: torch.Tensor,
    valid_units: torch.Tensor,
    *,
    temperature: float,
) -> MetricLossOutput:
    """Compute the recovered distribution-level soft InfoNCE objective."""
    if unit_embeddings.ndim != 3 or unit_targets.ndim != 3 or valid_units.ndim != 2:
        raise ValueError("metric inputs must be [B,U,D], [B,U,M], and [B,U]")
    if unit_embeddings.shape[:2] != unit_targets.shape[:2] or unit_embeddings.shape[:2] != valid_units.shape:
        raise ValueError("metric input layouts do not match")
    temperature = float(temperature)
    if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    losses, valid_counts, positive_counts, negative_counts = [], [], [], []
    positive_sims, negative_sims, entropies = [], [], []
    for embedding, target, valid in zip(unit_embeddings, unit_targets, valid_units):
        e = F.normalize(embedding.float(), dim=-1)
        p = target.float().clamp_min(0.0)
        u = e.shape[0]
        diag = torch.eye(u, dtype=torch.bool, device=e.device)
        sim = e @ e.transpose(0, 1) / temperature
        weights = (p @ p.transpose(0, 1)).masked_fill(diag, 0.0)
        valid_anchor = (p.sum(-1) > 0.1) & (weights.sum(-1) > 0) & valid
        log_soft = F.log_softmax(sim, dim=-1).masked_fill(diag, 0.0)
        normalized = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        per_anchor = -(normalized * log_soft).sum(-1)
        losses.append((per_anchor * valid_anchor).sum() / valid_anchor.sum().clamp_min(1))
        valid_counts.append(valid_anchor.sum())
        positive_mask = (weights > 0) & ~diag
        negative_mask = (weights < 0.1) & ~diag
        positive_counts.append(positive_mask.sum())
        negative_counts.append(negative_mask.sum())
        cosine = e @ e.transpose(0, 1)
        positive_sims.append(cosine[positive_mask].mean() if positive_mask.any() else e.new_zeros(()))
        negative_sims.append(cosine[negative_mask].mean() if negative_mask.any() else e.new_zeros(()))
        p_norm = p / p.sum(-1, keepdim=True).clamp_min(1e-8)
        entropies.append((-(p_norm * p_norm.clamp_min(1e-8).log()).sum(-1)[valid]).mean() if valid.any() else e.new_zeros(()))
    loss = torch.stack(losses).mean() if losses else unit_embeddings.sum() * 0.0
    return MetricLossOutput(
        loss=loss,
        valid_unit_count=int(torch.stack(valid_counts).sum().item()) if valid_counts else 0,
        positive_pair_count=int(torch.stack(positive_counts).sum().item()) if positive_counts else 0,
        negative_pair_count=int(torch.stack(negative_counts).sum().item()) if negative_counts else 0,
        mean_positive_similarity=torch.stack(positive_sims).mean() if positive_sims else loss.detach() * 0.0,
        mean_negative_similarity=torch.stack(negative_sims).mean() if negative_sims else loss.detach() * 0.0,
        target_entropy=torch.stack(entropies).mean() if entropies else loss.detach() * 0.0,
    )
