"""Visibility-aware scene-window Hungarian loss for TokenGS-ERU.

This module is intentionally separate from the legacy instance-group loss.
The legacy per-view path remains unchanged; ERU scene matching calls the
functions below with canonical ``[B,V,Q,H,W]`` probability maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from tokengs.models.instance_group_loss import (
    _dice_loss,
    _group_pair_loss,
    _soft_match_cost,
)


@dataclass(frozen=True)
class SceneHungarianAssignment:
    query_indices: torch.Tensor
    gt_instance_ids: torch.Tensor
    gt_columns: torch.Tensor
    visibility: torch.Tensor
    pairwise_cost: torch.Tensor


def _check_scene_inputs(gt_instance_maps: torch.Tensor) -> None:
    if gt_instance_maps.ndim != 3:
        raise ValueError(
            "gt_instance_maps must be [V,H,W], got "
            f"{tuple(gt_instance_maps.shape)}"
        )
    if gt_instance_maps.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("gt_instance_maps must have an integer dtype")


def build_scene_gt_masks(
    gt_instance_maps: torch.Tensor,
    *,
    valid_instance_ids: Optional[torch.Tensor] = None,
    min_visible_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a deterministic scene GT union and visibility matrix."""
    _check_scene_inputs(gt_instance_maps)
    if min_visible_pixels < 1:
        raise ValueError("min_visible_pixels must be positive")
    maps = gt_instance_maps.long()
    ignored = {0, 255, -1}
    valid_set = None
    if valid_instance_ids is not None:
        valid_set = {int(x) for x in valid_instance_ids.detach().cpu().flatten()}
    ids = set()
    for value in torch.unique(maps).detach().cpu().tolist():
        value = int(value)
        if value in ignored or (valid_set is not None and value not in valid_set):
            continue
        visible = (maps == value).sum(dim=(-2, -1))
        if bool((visible >= int(min_visible_pixels)).any()):
            ids.add(value)
    scene_gt_ids = torch.tensor(
        sorted(ids), dtype=torch.long, device=maps.device
    )
    views, height, width = maps.shape
    if scene_gt_ids.numel() == 0:
        return (
            scene_gt_ids,
            torch.zeros(
                (views, 0, height, width), dtype=torch.float32, device=maps.device
            ),
            torch.zeros((0, views), dtype=torch.bool, device=maps.device),
        )
    gt_masks = torch.stack(
        [(maps == int(instance_id)).float() for instance_id in scene_gt_ids],
        dim=1,
    )
    visible_pixels = gt_masks.sum(dim=(-2, -1))
    visibility = visible_pixels.ge(int(min_visible_pixels)).transpose(0, 1).contiguous()
    return scene_gt_ids, gt_masks, visibility


def compute_scene_pairwise_cost(
    predicted_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    visibility: torch.Tensor,
    *,
    bce_weight: float,
    dice_weight: float,
    eps: float,
) -> torch.Tensor:
    """Aggregate the unchanged per-view BCE+Dice cost over visible views."""
    if predicted_masks.ndim != 4 or gt_masks.ndim != 4:
        raise ValueError("predicted_masks and gt_masks must be [V,Q,H,W]/[V,O,H,W]")
    views, queries, height, width = predicted_masks.shape
    if gt_masks.shape[0] != views or gt_masks.shape[-2:] != (height, width):
        raise ValueError("predicted and GT view/spatial shapes do not match")
    objects = gt_masks.shape[1]
    if visibility.shape != (objects, views):
        raise ValueError(
            f"visibility must be {(objects, views)}, got {tuple(visibility.shape)}"
        )
    with torch.autocast(device_type=predicted_masks.device.type, enabled=False):
        pred = predicted_masks.float()
        gt = gt_masks.float()
        cost = torch.zeros((queries, objects), dtype=torch.float32, device=pred.device)
        counts = visibility.float().sum(dim=1)
        for view in range(views):
            object_columns = torch.nonzero(visibility[:, view], as_tuple=False).flatten()
            if object_columns.numel() == 0:
                continue
            view_cost = _soft_match_cost(
                pred[view].detach(),
                [gt[view, int(column)].bool() for column in object_columns],
                float(dice_weight),
                float(bce_weight),
                area_norm_bce=False,
                eps=float(eps),
            )
            cost[:, object_columns] += view_cost
        return cost / counts.clamp_min(1.0).unsqueeze(0)


def solve_scene_hungarian(
    pairwise_cost: torch.Tensor,
    scene_gt_ids: torch.Tensor,
    visibility: torch.Tensor,
) -> SceneHungarianAssignment:
    """Perform exactly one detached CPU Hungarian solve for one scene."""
    if pairwise_cost.ndim != 2:
        raise ValueError("pairwise_cost must be [Q,O]")
    if scene_gt_ids.ndim != 1 or scene_gt_ids.numel() != pairwise_cost.shape[1]:
        raise ValueError("scene_gt_ids does not match pairwise_cost columns")
    if visibility.ndim != 2 or visibility.shape[0] != scene_gt_ids.numel():
        raise ValueError("visibility does not match scene GT columns")
    cost = pairwise_cost.detach().float().cpu()
    if cost.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=scene_gt_ids.device)
        return SceneHungarianAssignment(
            query_indices=empty,
            gt_instance_ids=empty.clone(),
            gt_columns=empty.clone(),
            visibility=visibility.detach().bool(),
            pairwise_cost=cost,
        )
    rows, columns = linear_sum_assignment(cost.numpy())
    query_indices = torch.as_tensor(rows, dtype=torch.long, device=scene_gt_ids.device)
    gt_columns = torch.as_tensor(columns, dtype=torch.long, device=scene_gt_ids.device)
    return SceneHungarianAssignment(
        query_indices=query_indices,
        gt_instance_ids=scene_gt_ids.detach().long()[gt_columns],
        gt_columns=gt_columns,
        visibility=visibility.detach().bool(),
        pairwise_cost=cost,
    )


def _cfg(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def scene_hungarian_instance_group_loss(
    predicted_masks: torch.Tensor,
    gt_instance_maps: torch.Tensor,
    *,
    existing_loss_config: Any,
    valid_instance_ids: Optional[torch.Tensor] = None,
    return_diagnostics: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int | float]]:
    """Scene-global assignment with the legacy BCE/Dice/void terms.

    ``predicted_masks`` is the non-void probability tensor ``[B,V,Q,H,W]``.
    The caller owns the unchanged rendering and 101-channel void semantics.
    """
    if predicted_masks.ndim != 5 or gt_instance_maps.ndim != 4:
        raise ValueError("expected predicted [B,V,Q,H,W] and GT [B,V,H,W]")
    batch, views, queries, height, width = predicted_masks.shape
    if gt_instance_maps.shape != (batch, views, height, width):
        raise ValueError("GT shape must match [B,V,H,W]")
    if int(_cfg(existing_loss_config, "instance_group_match_topk", 1)) != 1:
        raise ValueError("scene matching requires the fixed existing match_topk=1")
    bce_weight = float(_cfg(existing_loss_config, "lambda_instance_group_mask", 1.0))
    dice_weight = float(_cfg(existing_loss_config, "lambda_instance_group_dice", 1.0))
    void_weight = float(_cfg(existing_loss_config, "lambda_instance_group_void", 0.1))
    unmatched_weight = float(
        _cfg(existing_loss_config, "lambda_instance_group_unmatched", 0.1)
    )
    ce_weight = float(_cfg(existing_loss_config, "lambda_instance_group_ce", 0.0))
    entropy_weight = float(_cfg(existing_loss_config, "instance_group_usage_entropy", 0.0))
    min_pixels = int(_cfg(existing_loss_config, "instance_group_min_instance_pixels", 32))
    eps = 1e-6
    void_channel = queries
    # The void probability is passed separately by the ERU caller only for
    # the unchanged void term.  A zero placeholder is intentionally not used:
    # callers attach it below through ``_scene_void_probability``.
    void_probability = getattr(existing_loss_config, "_scene_void_probability", None)
    if void_probability is None and isinstance(existing_loss_config, dict):
        void_probability = existing_loss_config.get("_scene_void_probability")
    if void_probability is None:
        raise ValueError("existing_loss_config must provide _scene_void_probability")
    if void_probability.shape != (batch, views, height, width):
        raise ValueError("_scene_void_probability must be [B,V,H,W]")

    total_loss = predicted_masks.new_zeros(())
    stats: dict[str, torch.Tensor | int | float] = {
        "instance_group_gt_count": predicted_masks.new_zeros(()),
        "instance_group_matched_count": predicted_masks.new_zeros(()),
        "loss_instance_group_dice": predicted_masks.new_zeros(()),
        "loss_instance_group_mask": predicted_masks.new_zeros(()),
        "loss_instance_group_void": predicted_masks.new_zeros(()),
        "loss_instance_group_unmatched": predicted_masks.new_zeros(()),
        "loss_instance_group_ce": predicted_masks.new_zeros(()),
        "loss_instance_group_entropy": predicted_masks.new_zeros(()),
        "instance_group_active_count": predicted_masks.new_zeros(()),
        "instance_group_hungarian_calls": 0,
        "instance_group_same_assignment_all_views": True,
        "instance_group_scene_gt_count": predicted_masks.new_zeros(()),
        "instance_group_visible_object_view_pairs": predicted_masks.new_zeros(()),
    }
    valid_view_count = 0
    for batch_index in range(batch):
        ids, gt_masks, visibility = build_scene_gt_masks(
            gt_instance_maps[batch_index],
            valid_instance_ids=valid_instance_ids,
            min_visible_pixels=min_pixels,
        )
        stats["instance_group_hungarian_calls"] += 1
        stats["instance_group_scene_gt_count"] += float(ids.numel())
        stats["instance_group_visible_object_view_pairs"] += float(visibility.sum())
        if ids.numel() == 0:
            continue
        cost = compute_scene_pairwise_cost(
            predicted_masks[batch_index],
            gt_masks,
            visibility,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            eps=eps,
        )
        assignment = solve_scene_hungarian(cost, ids, visibility)
        matched_by_id = {
            int(gt_id): int(query_id)
            for query_id, gt_id in zip(
                assignment.query_indices.tolist(), assignment.gt_instance_ids.tolist()
            )
        }
        matched_queries = set(matched_by_id.values())
        for view in range(views):
            visible_columns = torch.nonzero(visibility[:, view], as_tuple=False).flatten()
            if visible_columns.numel() == 0:
                continue
            valid_view_count += 1
            gt_map = gt_instance_maps[batch_index, view]
            view_ids = ids[visible_columns]
            view_masks = gt_masks[view, visible_columns].bool()
            mask_by_column = {
                int(column): view_masks[index]
                for index, column in enumerate(visible_columns.tolist())
            }
            primary = [
                (matched_by_id[int(gt_id)], int(column), 1.0)
                for gt_id, column in zip(view_ids.tolist(), visible_columns.tolist())
                if int(gt_id) in matched_by_id
            ]
            stats["instance_group_gt_count"] += float(visible_columns.numel())
            stats["instance_group_matched_count"] += float(len(primary))
            stats["instance_group_active_count"] += float(queries)
            probabilities = predicted_masks[batch_index, view]
            full_probabilities = torch.cat(
                [probabilities, void_probability[batch_index, view].unsqueeze(0)],
                dim=0,
            )
            if ce_weight > 0.0 and primary:
                target_map = torch.full_like(gt_map, queries, dtype=torch.long)
                for query_id, column, _ in primary:
                    target_map[mask_by_column[column]] = query_id
                ce = -full_probabilities.clamp(1e-6, 1.0 - 1e-6).gather(
                    0, target_map.unsqueeze(0)
                ).log().mean()
                total_loss = total_loss + ce_weight * ce
                stats["loss_instance_group_ce"] += ce_weight * ce.detach()
            pair_loss = predicted_masks.new_zeros(())
            for query_id, column, role_weight in primary:
                gt_mask = gt_masks[view, column].bool()
                pair = _group_pair_loss(
                    full_probabilities,
                    query_id,
                    gt_mask,
                    bce_weight,
                    dice_weight,
                )
                pair_loss = pair_loss + role_weight * pair
                stats["loss_instance_group_mask"] += role_weight * bce_weight * F.binary_cross_entropy(
                    probabilities[query_id].clamp(1e-6, 1.0 - 1e-6), gt_mask.float(), reduction="mean"
                ).detach()
                stats["loss_instance_group_dice"] += role_weight * dice_weight * _dice_loss(
                    probabilities[query_id], gt_mask
                ).detach()
            if primary:
                pair_loss = pair_loss / len(primary)
            total_loss = total_loss + pair_loss
            if entropy_weight > 0.0:
                usage = probabilities.clamp_min(1e-8).mean(dim=(1, 2))
                entropy_loss = torch.log(
                    torch.tensor(float(queries), device=usage.device)
                ) + (usage * usage.log()).sum()
                total_loss = total_loss + entropy_weight * entropy_loss
                stats["loss_instance_group_entropy"] += entropy_weight * entropy_loss.detach()
            gt_union = view_masks.any(dim=0) if view_masks.numel() else torch.zeros_like(gt_map, dtype=torch.bool)
            void_target = ~gt_union
            if void_target.any():
                void = void_probability[batch_index, view].clamp(1e-6, 1.0 - 1e-6)
                void_loss = F.binary_cross_entropy(void, void_target.float(), reduction="mean") + _dice_loss(void, void_target)
                total_loss = total_loss + void_weight * void_loss
                stats["loss_instance_group_void"] += void_weight * F.binary_cross_entropy(
                    void, void_target.float(), reduction="mean"
                ).detach()
            unmatched_mass = probabilities.new_zeros(())
            for query_id in range(queries):
                if query_id not in matched_queries:
                    unmatched_mass = unmatched_mass + probabilities[query_id].mean()
            if unmatched_mass.detach() > 0:
                total_loss = total_loss + unmatched_weight * unmatched_mass
                stats["loss_instance_group_unmatched"] += unmatched_weight * unmatched_mass.detach()
    if valid_view_count > 0:
        total_loss = total_loss / valid_view_count
        for key, value in list(stats.items()):
            if torch.is_tensor(value):
                stats[key] = value / valid_view_count
    if not return_diagnostics:
        # Keep scalar stats available to the existing trainer for logging;
        # all tensors are detached exactly as in the legacy loss caller.
        pass
    return total_loss, stats
