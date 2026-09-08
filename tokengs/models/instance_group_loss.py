"""Hungarian-matched instance group losses (InstOk3D-style slot grouping).

The model renders one probability channel per learned instance group (plus
an explicit void channel). Supervision follows InstOk3D / standard 2D mask
transformers:

1. All groups (including currently empty ones) are candidates; the matching
   cost between group g and GT instance n is a detached Dice + BCE cost on
   the soft probability maps, and Hungarian assigns every GT instance to its
   best group (full assignment, no IoU threshold). Empty groups can
   therefore be pushed onto uncovered instances.
2. Each matched pair gets a BCE + Dice loss on the (non-detached) soft map,
   so every GT instance receives a gradient.
3. The void channel is supervised by the background, and groups left
   unmatched by the assignment are pushed toward zero.

The whole loss runs in FP32 so it is safe under bf16 autocast.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _gt_masks_from_map(
    instance_map: torch.Tensor,
    ignore_ids: tuple[int, ...] = (0, 255, -1),
    min_pixels: int = 32,
) -> list[torch.Tensor]:
    return [
        mask
        for _, mask in _gt_masks_with_ids(
            instance_map,
            ignore_ids=ignore_ids,
            min_pixels=min_pixels,
        )
    ]


def _gt_masks_with_ids(
    instance_map: torch.Tensor,
    ignore_ids: tuple[int, ...] = (0, 255, -1),
    min_pixels: int = 32,
) -> list[tuple[int, torch.Tensor]]:
    masks = []
    for instance_id in torch.unique(instance_map):
        if int(instance_id) in ignore_ids:
            continue
        mask = instance_map == instance_id
        if mask.sum() >= min_pixels:
            masks.append((int(instance_id), mask))
    return masks


def _soft_match_cost(
    pred_probs: torch.Tensor,
    gt_masks: list[torch.Tensor],
    dice_weight: float,
    mask_weight: float,
    area_norm_bce: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Detached [G, N] matching cost: lambda_dice*(1-Dice) + lambda_bce*BCE."""
    pred_flat = pred_probs.reshape(pred_probs.shape[0], -1)  # [G, HW]
    gt_flat = torch.stack(
        [mask.reshape(-1).float() for mask in gt_masks], dim=0
    )  # [N, HW]
    intersection = pred_flat @ gt_flat.T  # [G, N]
    dice = (2.0 * intersection) / (
        pred_flat.sum(dim=1, keepdim=True)
        + gt_flat.sum(dim=1, keepdim=True).T
    ).clamp_min(eps)
    prob_clamped = pred_flat.clamp(eps, 1.0 - eps)
    log_prob = prob_clamped.log()
    if area_norm_bce:
        # Foreground-only BCE normalized by GT area, so small instances get
        # matching costs of the same scale as large ones (Dice already does
        # this; the BCE term no longer drowns small masks in background).
        gt_areas = gt_flat.sum(dim=1, keepdim=True).T.clamp_min(1.0)  # [1, N]
        bce = -(log_prob @ gt_flat.T) / gt_areas  # [G, N]
    else:
        log_one_minus = (1.0 - prob_clamped).log()
        bce = -(
            log_prob @ gt_flat.T + log_one_minus @ (1.0 - gt_flat).T
        ) / pred_flat.shape[1]  # [G, N]
    return dice_weight * (1.0 - dice) + mask_weight * bce


def _hungarian_matches(
    probabilities: torch.Tensor,
    gt_masks: list[torch.Tensor],
    dice_weight: float,
    mask_weight: float,
    area_norm_bce: bool = False,
    topk: int = 1,
    secondary_weight: float = 0.3,
    num_active_groups: int | None = None,
    active_ids: list[int] | None = None,
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Bipartite assignment over soft group probability maps.

    Every GT instance is matched to its best group (primary, weight 1.0),
    even if the group is currently empty (the pair then receives a strong
    BCE+Dice gradient). With ``topk > 1``, each GT instance additionally
    claims up to ``topk - 1`` unused groups with a lower role weight, so
    more groups receive positive gradient per view -- the direct mechanism
    to break the coarse-group collapse (only ~N winners per view).

    Returns (primary_pairs, extra_pairs), each pair is (group_id, gt_idx,
    role_weight).
    """
    if active_ids is not None:
        pred_probs = probabilities[active_ids]
        id_map = active_ids
    else:
        pred_probs = probabilities[:num_active_groups or (probabilities.shape[0] - 1)]
        id_map = list(range(pred_probs.shape[0]))
    cost = _soft_match_cost(
        pred_probs.detach(),
        gt_masks,
        dice_weight,
        mask_weight,
        area_norm_bce=area_norm_bce,
    )
    return _matches_from_cost(
        cost,
        id_map,
        topk=topk,
        secondary_weight=secondary_weight,
    )


def _matches_from_cost(
    cost: torch.Tensor,
    id_map: list[int],
    topk: int = 1,
    secondary_weight: float = 0.3,
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Convert a detached [groups, instances] cost matrix to assignments."""
    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    used_groups: set[int] = set()
    primary: list[tuple[int, int, float]] = []
    for r, c in zip(row_ind, col_ind):
        group_id = id_map[int(r)]
        primary.append((group_id, int(c), 1.0))
        used_groups.add(group_id)
    extra: list[tuple[int, int, float]] = []
    if topk > 1:
        cost_np = cost.detach().cpu().numpy()
        for c in col_ind:
            c = int(c)
            added = 0
            for r in np.argsort(cost_np[:, c]):
                group_id = id_map[int(r)]
                if group_id in used_groups:
                    continue
                extra.append((group_id, c, float(secondary_weight)))
                used_groups.add(group_id)
                added += 1
                if added >= topk - 1:
                    break
    return primary, extra


def _scene_hungarian_matches(
    probabilities: torch.Tensor,
    view_instances: list[list[tuple[int, torch.Tensor]]],
    dice_weight: float,
    mask_weight: float,
    area_norm_bce: bool = False,
    topk: int = 1,
    secondary_weight: float = 0.3,
    num_active_groups: int | None = None,
    active_ids: list[int] | None = None,
) -> tuple[
    list[int],
    list[tuple[int, int, float]],
    list[tuple[int, int, float]],
]:
    """Match scene-global instance ids once using all views where they appear."""
    scene_instance_ids = sorted(
        {
            instance_id
            for instances in view_instances
            for instance_id, _ in instances
        }
    )
    if not scene_instance_ids:
        return [], [], []

    if active_ids is not None:
        pred_probs = probabilities[active_ids]
        id_map = active_ids
    else:
        active_count = num_active_groups or (probabilities.shape[0] - 1)
        pred_probs = probabilities[:active_count]
        id_map = list(range(pred_probs.shape[0]))

    instance_to_column = {
        instance_id: column
        for column, instance_id in enumerate(scene_instance_ids)
    }
    cost_sum = torch.zeros(
        (pred_probs.shape[0], len(scene_instance_ids)),
        device=pred_probs.device,
        dtype=torch.float32,
    )
    visible_view_count = torch.zeros(
        len(scene_instance_ids),
        device=pred_probs.device,
        dtype=torch.float32,
    )
    for view_index, instances in enumerate(view_instances):
        if not instances:
            continue
        view_masks = [mask for _, mask in instances]
        view_cost = _soft_match_cost(
            pred_probs[:, view_index].detach(),
            view_masks,
            dice_weight,
            mask_weight,
            area_norm_bce=area_norm_bce,
        )
        columns = torch.tensor(
            [instance_to_column[instance_id] for instance_id, _ in instances],
            device=pred_probs.device,
            dtype=torch.long,
        )
        cost_sum.index_add_(1, columns, view_cost)
        visible_view_count.index_add_(
            0, columns, torch.ones_like(columns, dtype=torch.float32)
        )
    cost = cost_sum / visible_view_count.clamp_min(1.0).unsqueeze(0)
    primary, extra = _matches_from_cost(
        cost,
        id_map,
        topk=topk,
        secondary_weight=secondary_weight,
    )
    return scene_instance_ids, primary, extra


def _dice_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    pred_flat = pred.float().reshape(-1)
    target_flat = target.float().reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + eps) / (
        pred_flat.sum() + target_flat.sum() + eps
    )


def _group_pair_loss(
    probabilities: torch.Tensor,
    group_id: int,
    gt_mask: torch.Tensor,
    mask_weight: float,
    dice_weight: float,
) -> torch.Tensor:
    pred = probabilities[group_id].clamp(1e-6, 1.0 - 1e-6)
    bce = F.binary_cross_entropy(pred, gt_mask.float(), reduction="mean")
    dice = _dice_loss(pred, gt_mask)
    return mask_weight * bce + dice_weight * dice


def _hungarian_instance_group_loss_impl(
    rendered_probability: torch.Tensor,
    instance_labels: torch.Tensor,
    num_groups: int,
    min_instance_pixels: int = 64,
    dice_weight: float = 1.0,
    mask_weight: float = 1.0,
    void_weight: float = 0.1,
    unmatched_weight: float = 0.1,
    lambda_eff: float = 1.0,
    area_alpha: float = 0.0,
    match_area_norm: bool = False,
    ce_weight: float = 0.0,
    match_topk: int = 1,
    secondary_pair_weight: float = 0.3,
    usage_entropy_weight: float = 0.0,
    use_adaptive_groups: bool = False,
    scene_level_matching: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Hungarian-matched BCE + Dice over rendered group probability maps.

    ``rendered_probability`` is [B, G+1, V, 1, H, W] (softmaxed over groups
    plus the final void channel) and ``instance_labels`` is [B, V, H, W] with
    raw ScanNet instance ids (0/255 ignored).
    """
    batch_size, channels, view_count, _, height, width = (
        rendered_probability.shape
    )
    if channels != num_groups + 1:
        raise ValueError(
            f"Expected {num_groups + 1} channels, got {channels}"
        )
    void_channel = num_groups
    total_loss = torch.zeros((), device=rendered_probability.device)
    stats = {
        "instance_group_gt_count": torch.zeros(
            (), device=rendered_probability.device
        ),
        "instance_group_matched_count": torch.zeros(
            (), device=rendered_probability.device
        ),
        "loss_instance_group_dice": torch.zeros(
            (), device=rendered_probability.device
        ),
        "loss_instance_group_mask": torch.zeros(
            (), device=rendered_probability.device
        ),
        "loss_instance_group_void": torch.zeros(
            (), device=rendered_probability.device
        ),
        "loss_instance_group_unmatched": torch.zeros(
            (), device=rendered_probability.device
        ),
        "loss_instance_group_ce": torch.zeros(
            (), device=rendered_probability.device
        ),
        "loss_instance_group_entropy": torch.zeros(
            (), device=rendered_probability.device
        ),
        "instance_group_active_count": torch.zeros(
            (), device=rendered_probability.device
        ),
    }
    num_views_total = 0
    for b in range(batch_size):
        scene_primary_by_id: dict[int, tuple[int, float]] | None = None
        scene_extra_by_id: dict[int, list[tuple[int, float]]] | None = None
        scene_active_ids = None
        scene_active_count = num_groups
        view_instances = [
            _gt_masks_with_ids(
                instance_labels[b, v], min_pixels=min_instance_pixels
            )
            for v in range(view_count)
        ]
        if scene_level_matching:
            scene_instance_count = len(
                {
                    instance_id
                    for instances in view_instances
                    for instance_id, _ in instances
                }
            )
            if scene_instance_count == 0:
                continue
            if use_adaptive_groups:
                scene_active_count = max(
                    1, min(num_groups, scene_instance_count)
                )
                scene_usage = rendered_probability[
                    b, :num_groups, :, 0
                ].mean(dim=(1, 2, 3)).detach()
                scene_active_ids = torch.topk(
                    scene_usage, scene_active_count
                ).indices.tolist()
            scene_ids, scene_primary, scene_extra = _scene_hungarian_matches(
                rendered_probability[b, :, :, 0],
                view_instances,
                dice_weight,
                mask_weight,
                area_norm_bce=match_area_norm,
                topk=int(match_topk),
                secondary_weight=float(secondary_pair_weight),
                num_active_groups=scene_active_count,
                active_ids=scene_active_ids,
            )
            scene_primary_by_id = {
                scene_ids[gt_idx]: (group_id, role_weight)
                for group_id, gt_idx, role_weight in scene_primary
            }
            scene_extra_by_id = {instance_id: [] for instance_id in scene_ids}
            for group_id, gt_idx, role_weight in scene_extra:
                scene_extra_by_id[scene_ids[gt_idx]].append(
                    (group_id, role_weight)
                )

        for v in range(view_count):
            probabilities = rendered_probability[b, :, v, 0]  # [G+1,H,W]
            gt_map = instance_labels[b, v]
            instances = view_instances[v]
            if not instances:
                continue
            gt_instance_ids = [instance_id for instance_id, _ in instances]
            gt_masks = [mask for _, mask in instances]
            active_ids = None
            num_active_groups = num_groups
            if scene_level_matching:
                active_ids = scene_active_ids
                num_active_groups = scene_active_count
                primary = []
                extra = []
                assert scene_primary_by_id is not None
                assert scene_extra_by_id is not None
                for gt_idx, instance_id in enumerate(gt_instance_ids):
                    assignment = scene_primary_by_id.get(instance_id)
                    # If a scene contains more instances than the configured
                    # group budget, Hungarian can only assign the first G.
                    # Leave excess instances unsupervised instead of creating
                    # a contradictory slot target or raising a KeyError.
                    if assignment is None:
                        continue
                    group_id, role_weight = assignment
                    primary.append((group_id, gt_idx, role_weight))
                    extra.extend(
                        (extra_group_id, gt_idx, extra_weight)
                        for extra_group_id, extra_weight in scene_extra_by_id[
                            instance_id
                        ]
                    )
            elif use_adaptive_groups:
                # Scene-adaptive budget: supervise only the ``G`` most-used
                # groups (G = GT instance count in this view). The count
                # head learns G; groups outside the active set are pushed
                # toward void by the unmatched penalty and CE competition.
                num_active_groups = max(
                    1, min(num_groups, len(gt_masks))
                )
                usage = (
                    probabilities[:num_groups].mean(dim=(1, 2)).detach()
                )
                active_ids = torch.topk(
                    usage, num_active_groups
                ).indices.tolist()
                primary, extra = _hungarian_matches(
                    probabilities,
                    gt_masks,
                    dice_weight,
                    mask_weight,
                    area_norm_bce=match_area_norm,
                    topk=int(match_topk),
                    secondary_weight=float(secondary_pair_weight),
                    num_active_groups=num_active_groups,
                    active_ids=active_ids,
                )
            else:
                primary, extra = _hungarian_matches(
                    probabilities,
                    gt_masks,
                    dice_weight,
                    mask_weight,
                    area_norm_bce=match_area_norm,
                    topk=int(match_topk),
                    secondary_weight=float(secondary_pair_weight),
                    num_active_groups=num_active_groups,
                    active_ids=active_ids,
                )
            matches = primary + extra
            matched_group_ids = {pred_idx for pred_idx, _, _ in matches}
            num_gt_matched = len(primary)
            num_views_total += 1
            stats["instance_group_gt_count"] += len(gt_masks)
            stats["instance_group_matched_count"] += num_gt_matched
            stats["instance_group_active_count"] += num_active_groups

            # Per-pixel cross-entropy over the G+1 group channels: each
            # pixel's target is the group matched to its GT instance (void
            # for non-instance pixels). Unlike independent BCE+Dice per
            # group, CE makes groups compete for pixels, so a group that
            # covers another group's instance is directly penalized. This
            # is what prevents the coarse "4-group" collapse.
            if ce_weight > 0.0 and primary:
                target_map = torch.full(
                    gt_map.shape,
                    void_channel,
                    dtype=torch.long,
                    device=gt_map.device,
                )
                for group_id, gt_idx, _ in primary:
                    target_map[gt_masks[gt_idx]] = group_id
                prob_clamped = probabilities.clamp(1e-6, 1.0 - 1e-6)
                ce_loss = -prob_clamped.gather(
                    0, target_map.unsqueeze(0)
                ).log().mean()
                total_loss = total_loss + lambda_eff * ce_weight * ce_loss
                with torch.no_grad():
                    stats["loss_instance_group_ce"] += (
                        ce_weight * ce_loss.detach()
                    )

            pair_loss = torch.zeros((), device=rendered_probability.device)
            if area_alpha > 0.0 and primary:
                gt_areas = torch.stack(
                    [
                        gt_masks[gt_idx].sum().float()
                        for _, gt_idx, _ in primary
                    ]
                ).clamp_min(1.0)
                pair_weights = (
                    gt_areas.mean() / gt_areas
                ).pow(area_alpha)
                pair_weights = pair_weights / pair_weights.mean().clamp_min(1e-6)
                area_weight_by_gt = {
                    gt_idx: float(pair_weights[i])
                    for i, (_, gt_idx, _) in enumerate(primary)
                }
            else:
                area_weight_by_gt = None
            for group_id, gt_idx, role_weight in matches:
                gt_mask = gt_masks[gt_idx]
                pair = _group_pair_loss(
                    probabilities, group_id, gt_mask, mask_weight, dice_weight
                )
                weight = role_weight * (
                    area_weight_by_gt[gt_idx]
                    if area_weight_by_gt is not None
                    else 1.0
                )
                pair_loss = pair_loss + weight * pair
                with torch.no_grad():
                    stats["loss_instance_group_mask"] += (
                        weight
                        * mask_weight
                        * F.binary_cross_entropy(
                            probabilities[group_id].clamp(1e-6, 1 - 1e-6),
                            gt_mask.float(),
                            reduction="mean",
                        )
                    )
                    stats["loss_instance_group_dice"] += (
                        weight
                        * dice_weight
                        * _dice_loss(probabilities[group_id], gt_mask)
                    )
            if num_gt_matched > 0:
                pair_loss = pair_loss / num_gt_matched
            total_loss = total_loss + lambda_eff * pair_loss

            # Group usage entropy (KL from uniform over non-void groups):
            # penalizes a few groups hoarding all tokens, the anti-collapse
            # regularizer from slot-attention style training.
            if usage_entropy_weight > 0.0:
                usage = probabilities[:num_groups].mean(dim=(1, 2)).clamp_min(
                    1e-8
                )
                entropy = -(usage * usage.log()).sum()
                entropy_loss = math.log(float(num_groups)) - entropy
                total_loss = (
                    total_loss
                    + lambda_eff * usage_entropy_weight * entropy_loss
                )
                with torch.no_grad():
                    stats["loss_instance_group_entropy"] += (
                        usage_entropy_weight * entropy_loss.detach()
                    )

            # Void channel: pixels with no GT instance.
            gt_union = torch.zeros_like(gt_map, dtype=torch.bool)
            for gt_mask in gt_masks:
                gt_union = gt_union | gt_mask
            void_target = ~gt_union
            if void_target.any():
                void_loss = _group_pair_loss(
                    probabilities,
                    void_channel,
                    void_target,
                    mask_weight,
                    dice_weight,
                )
                total_loss = total_loss + lambda_eff * void_weight * void_loss
                with torch.no_grad():
                    stats["loss_instance_group_void"] += (
                        void_weight
                        * F.binary_cross_entropy(
                            probabilities[void_channel].clamp(1e-6, 1 - 1e-6),
                            void_target.float(),
                            reduction="mean",
                        )
                    )

            # Groups left unmatched by the assignment should be empty.
            unmatched_mass = torch.zeros((), device=rendered_probability.device)
            for group_id in range(num_groups):
                if group_id in matched_group_ids:
                    continue
                unmatched_mass = unmatched_mass + probabilities[group_id].mean()
            if unmatched_mass > 0:
                total_loss = (
                    total_loss
                    + lambda_eff * unmatched_weight * unmatched_mass
                )
                with torch.no_grad():
                    stats["loss_instance_group_unmatched"] += (
                        unmatched_weight * unmatched_mass.detach()
                    )

    if num_views_total > 0:
        total_loss = total_loss / num_views_total
        for key in stats:
            stats[key] = stats[key] / num_views_total
    return total_loss, stats


def hungarian_instance_group_loss(
    rendered_probability: torch.Tensor,
    instance_labels: torch.Tensor,
    num_groups: int,
    min_instance_pixels: int = 64,
    dice_weight: float = 1.0,
    mask_weight: float = 1.0,
    void_weight: float = 0.1,
    unmatched_weight: float = 0.1,
    lambda_eff: float = 1.0,
    area_alpha: float = 0.0,
    match_area_norm: bool = False,
    ce_weight: float = 0.0,
    match_topk: int = 1,
    secondary_pair_weight: float = 0.3,
    usage_entropy_weight: float = 0.0,
    use_adaptive_groups: bool = False,
    scene_level_matching: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """FP32 wrapper so BCE runs safely under bf16 autocast."""
    with torch.autocast(
        device_type=rendered_probability.device.type, enabled=False
    ):
        return _hungarian_instance_group_loss_impl(
            rendered_probability,
            instance_labels,
            num_groups,
            min_instance_pixels=min_instance_pixels,
            dice_weight=dice_weight,
            mask_weight=mask_weight,
            void_weight=void_weight,
            unmatched_weight=unmatched_weight,
            lambda_eff=lambda_eff,
            area_alpha=area_alpha,
            match_area_norm=match_area_norm,
            ce_weight=ce_weight,
            match_topk=match_topk,
            secondary_pair_weight=secondary_pair_weight,
            usage_entropy_weight=usage_entropy_weight,
            use_adaptive_groups=use_adaptive_groups,
            scene_level_matching=scene_level_matching,
        )


def _project_gs_to_views(
    means: torch.Tensor,
    cam_views: torch.Tensor,
    intrinsics: torch.Tensor,
    labels: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project Gaussian centers into views, returning (ids, valid).

    ``means`` is [N,3] world positions, ``cam_views`` is [V,4,4] stored in
    the provider's transposed convention (same as the renderer input) and
    ``intrinsics`` is [V,4] flattened as (fx, fy, cx, cy) -- the provider
    format. ``labels`` is [V,H,W] GT instance ids. The returned ``ids`` is
    [V,N] with the instance id at each projected pixel (0 = background,
    255/-1 = ignored) and ``valid`` is [V,N] masking in-bounds, in-front,
    non-ignored votes.
    """
    height, width = int(image_size[0]), int(image_size[1])
    view_count = cam_views.shape[0]
    world_to_cam = cam_views.transpose(-1, -2).float()  # [V,4,4]
    homo = torch.cat(
        [
            means.float(),
            torch.ones_like(means[..., :1]),
        ],
        dim=-1,
    )  # [N,4]
    cam = torch.einsum("vij,nj->vni", world_to_cam, homo)  # [V,N,4]
    z = cam[..., 2]
    fx = intrinsics[:, 0]
    fy = intrinsics[:, 1]
    cx = intrinsics[:, 2]
    cy = intrinsics[:, 3]
    px = fx[:, None] * cam[..., 0] / z.clamp_min(1e-6) + cx[:, None]
    py = fy[:, None] * cam[..., 1] / z.clamp_min(1e-6) + cy[:, None]
    valid = (
        (z > 0.01)
        & (px >= 0)
        & (px < width)
        & (py >= 0)
        & (py < height)
    )
    px_l = px.clamp(0, width - 1).long()
    py_l = py.clamp(0, height - 1).long()
    ids = labels[
        torch.arange(view_count, device=labels.device)[:, None],
        py_l,
        px_l,
    ]  # [V,N]
    valid = valid & (ids != 255) & (ids != -1)
    return ids, valid


def _gs_majority_target(
    ids: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Per-Gaussian majority-vote instance target.

    ``ids`` is [V,N] (0 = background), ``valid`` masks usable votes.
    Returns [N] long with the majority instance id per Gaussian
    (0 = background for Gaussians with no valid votes).
    """
    votes = torch.where(valid, ids, torch.full_like(ids, -1))
    num_gs = ids.shape[1]
    candidates = torch.unique(ids[valid])
    best_id = torch.zeros(
        num_gs, dtype=torch.long, device=ids.device
    )
    if candidates.numel() == 0:
        return best_id
    best_count = torch.zeros(
        num_gs, dtype=torch.float32, device=ids.device
    )
    for cid in candidates:
        count = (votes == cid).sum(dim=0).float()
        replace = (count > best_count) | (
            (count == best_count) & (best_id > cid)
        )
        best_id = torch.where(replace, cid.expand_as(best_id), best_id)
        best_count = torch.where(replace, count, best_count)
    return best_id


def _hungarian_matches_3d(
    group_probs: torch.Tensor,
    target: torch.Tensor,
    min_instance_gs: int,
    topk: int,
    secondary_weight: float,
    num_active_groups: int | None = None,
    active_ids: list[int] | None = None,
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Match groups to GT instances by 3D overlap.

    Coverage of group ``g`` on instance ``i`` is the mean probability of
    group ``g`` over the Gaussians voted to instance ``i`` (InstOk3D-style
    anchor-level matching). Instances with fewer than ``min_instance_gs``
    Gaussians are excluded (their Gaussians fall to the void channel, the
    3D analogue of the 2D ``min_instance_pixels`` filter).

    Returns (primary_pairs, extra_pairs), each (group_id, instance_id,
    role_weight).
    """
    if active_ids is not None:
        group_probs = group_probs[:, active_ids]
        id_map = active_ids
    else:
        id_map = list(range(group_probs.shape[1]))
    instance_ids = torch.unique(target[target > 0])
    valid_pairs: list[tuple[int, int]] = []
    for j, iid in enumerate(instance_ids.tolist()):
        count = int((target == iid).sum())
        if count >= min_instance_gs:
            valid_pairs.append((j, iid))
    if not valid_pairs:
        return [], []
    coverage = torch.zeros(
        group_probs.shape[1], len(valid_pairs), device=group_probs.device
    )
    for k, (_, iid) in enumerate(valid_pairs):
        mask = target == iid
        coverage[:, k] = group_probs[mask].mean(dim=0)
    cost = -coverage.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    used_groups: set[int] = set()
    primary: list[tuple[int, int, float]] = []
    for r, c in zip(row_ind, col_ind):
        group_id = id_map[int(r)]
        primary.append((group_id, int(valid_pairs[int(c)][1]), 1.0))
        used_groups.add(group_id)
    extra: list[tuple[int, int, float]] = []
    if topk > 1:
        for c in col_ind:
            added = 0
            for r in np.argsort(cost[:, c]):
                group_id = id_map[int(r)]
                if group_id in used_groups:
                    continue
                extra.append(
                    (group_id, int(valid_pairs[int(c)][1]), float(secondary_weight))
                )
                used_groups.add(group_id)
                added += 1
                if added >= topk - 1:
                    break
    return primary, extra


def _instance_group_3d_loss_impl(
    gaussian_group_probs: torch.Tensor,
    gaussians: torch.Tensor,
    cam_views: torch.Tensor,
    intrinsics: torch.Tensor,
    instance_labels: torch.Tensor,
    num_groups: int,
    image_size: tuple[int, int],
    min_instance_gs: int = 16,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    mask_weight: float = 1.0,
    void_weight: float = 1.0,
    unmatched_weight: float = 0.1,
    match_topk: int = 1,
    secondary_pair_weight: float = 0.3,
    use_adaptive_groups: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """InstOk3D-style 3D anchor-level instance grouping loss.

    Projects every Gaussian center into all labeled views, takes a
    majority-vote instance id per Gaussian (cross-view consistent by
    construction), Hungarian-matches groups to instances by 3D overlap,
    then supervises the per-Gaussian soft assignment directly: CE toward
    the matched group, BCE+Dice on the 3D group masks, a void channel for
    background Gaussians, and a penalty on unmatched groups. This is the
    dense, render-free counterpart of the rendered 2D Hungarian loss and
    teaches 3D-consistent grouping instead of view-dependent pixels.
    """
    batch_size, num_gs, channels = gaussian_group_probs.shape
    void_channel = num_groups
    total_loss = torch.zeros((), device=gaussian_group_probs.device)
    stats = {
        "instance_group_3d_gt_count": torch.zeros(
            (), device=gaussian_group_probs.device
        ),
        "instance_group_3d_matched_count": torch.zeros(
            (), device=gaussian_group_probs.device
        ),
        "loss_instance_group_3d_ce": torch.zeros(
            (), device=gaussian_group_probs.device
        ),
        "loss_instance_group_3d_mask": torch.zeros(
            (), device=gaussian_group_probs.device
        ),
        "loss_instance_group_3d_dice": torch.zeros(
            (), device=gaussian_group_probs.device
        ),
        "loss_instance_group_3d_void": torch.zeros(
            (), device=gaussian_group_probs.device
        ),
        "loss_instance_group_3d_unmatched": torch.zeros(
            (), device=gaussian_group_probs.device
        ),
        "instance_group_3d_active_count": torch.zeros(
            (), device=gaussian_group_probs.device
        ),
    }
    for b in range(batch_size):
        ids, valid = _project_gs_to_views(
            gaussians[b, :, :3],
            cam_views[b],
            intrinsics[b],
            instance_labels[b],
            image_size,
        )
        target = _gs_majority_target(ids, valid)  # [N]
        probs = gaussian_group_probs[b].float()  # [N,G+1]
        group_probs = probs[:, :num_groups]
        active_ids = None
        num_active_groups = num_groups
        if use_adaptive_groups:
            instance_ids = torch.unique(target[target > 0])
            num_valid = sum(
                1
                for iid in instance_ids.tolist()
                if int((target == iid).sum()) >= min_instance_gs
            )
            num_active_groups = max(1, min(num_groups, num_valid))
            usage = group_probs.mean(dim=0).detach()  # [G]
            active_ids = torch.topk(
                usage, num_active_groups
            ).indices.tolist()
        primary, extra = _hungarian_matches_3d(
            group_probs,
            target,
            min_instance_gs,
            match_topk,
            secondary_pair_weight,
            num_active_groups=num_active_groups,
            active_ids=active_ids,
        )
        matches = primary + extra
        matched_groups = {group_id for group_id, _, _ in matches}

        # Per-Gaussian CE toward the matched group (void for background /
        # tiny instances) -- the dense 3D equivalent of the 2D pixel CE.
        target_group = torch.full(
            (num_gs,), void_channel, dtype=torch.long, device=probs.device
        )
        for group_id, instance_id, _ in matches:
            target_group[target == instance_id] = group_id
        prob_clamped = probs.clamp(1e-6, 1.0 - 1e-6)
        ce_loss = -prob_clamped.gather(
            1, target_group.unsqueeze(1)
        ).log().mean()
        total_loss = total_loss + ce_weight * ce_loss
        stats["loss_instance_group_3d_ce"] += ce_weight * ce_loss.detach()

        pair_loss = torch.zeros((), device=probs.device)
        for group_id, instance_id, role_weight in matches:
            mask = (target == instance_id).float()
            pair_loss = pair_loss + role_weight * (
                mask_weight
                * F.binary_cross_entropy(
                    probs[:, group_id].clamp(1e-6, 1.0 - 1e-6),
                    mask,
                    reduction="mean",
                )
                + dice_weight * _dice_loss(probs[:, group_id], mask)
            )
        if primary:
            pair_loss = pair_loss / len(primary)
            total_loss = total_loss + pair_loss
            with torch.no_grad():
                stats["loss_instance_group_3d_mask"] += (
                    mask_weight
                    * sum(
                        F.binary_cross_entropy(
                            probs[:, group_id].clamp(1e-6, 1.0 - 1e-6),
                            (target == instance_id).float(),
                            reduction="mean",
                        )
                        for group_id, instance_id, _ in matches
                    )
                    / len(primary)
                )
                stats["loss_instance_group_3d_dice"] += (
                    dice_weight
                    * sum(
                        _dice_loss(
                            probs[:, group_id],
                            (target == instance_id).float(),
                        )
                        for group_id, instance_id, _ in matches
                    )
                    / len(primary)
                )

        # Void channel for background Gaussians (and tiny instances).
        background_mask = (target == 0).float()
        if background_mask.any():
            void_loss = (
                void_weight
                * mask_weight
                * F.binary_cross_entropy(
                    probs[:, void_channel].clamp(1e-6, 1.0 - 1e-6),
                    background_mask,
                    reduction="mean",
                )
                + void_weight
                * dice_weight
                * _dice_loss(probs[:, void_channel], background_mask)
            )
            total_loss = total_loss + void_loss
            with torch.no_grad():
                stats["loss_instance_group_3d_void"] += void_loss.detach()

        # Groups left unmatched should be empty.
        unmatched_mass = torch.zeros((), device=probs.device)
        for group_id in range(num_groups):
            if group_id in matched_groups:
                continue
            unmatched_mass = unmatched_mass + probs[:, group_id].mean()
        total_loss = total_loss + unmatched_weight * unmatched_mass
        with torch.no_grad():
            stats["loss_instance_group_3d_unmatched"] += (
                unmatched_weight * unmatched_mass.detach()
            )
        stats["instance_group_3d_gt_count"] += len(primary)
        stats["instance_group_3d_matched_count"] += len(primary)
        stats["instance_group_3d_active_count"] += num_active_groups

    total_loss = total_loss / max(1, batch_size)
    for key in stats:
        stats[key] = stats[key] / max(1, batch_size)
    return total_loss, stats


def instance_group_3d_loss(
    gaussian_group_probs: torch.Tensor,
    gaussians: torch.Tensor,
    cam_views: torch.Tensor,
    intrinsics: torch.Tensor,
    instance_labels: torch.Tensor,
    num_groups: int,
    image_size: tuple[int, int] = (256, 256),
    min_instance_gs: int = 16,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    mask_weight: float = 1.0,
    void_weight: float = 1.0,
    unmatched_weight: float = 0.1,
    match_topk: int = 1,
    secondary_pair_weight: float = 0.3,
    use_adaptive_groups: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """FP32 wrapper so the 3D instance loss runs safely under bf16."""
    # Accept both batched [B,...] and unbatched [V,...]/[N,...] inputs.
    if gaussian_group_probs.ndim == 2:
        gaussian_group_probs = gaussian_group_probs.unsqueeze(0)
    if gaussians.ndim == 2:
        gaussians = gaussians.unsqueeze(0)
    if cam_views.ndim == 3:
        cam_views = cam_views.unsqueeze(0)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if instance_labels.ndim == 3:
        instance_labels = instance_labels.unsqueeze(0)
    with torch.autocast(
        device_type=gaussian_group_probs.device.type, enabled=False
    ):
        return _instance_group_3d_loss_impl(
            gaussian_group_probs,
            gaussians,
            cam_views,
            intrinsics,
            instance_labels,
            num_groups,
            image_size=image_size,
            min_instance_gs=min_instance_gs,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            mask_weight=mask_weight,
            void_weight=void_weight,
            unmatched_weight=unmatched_weight,
            match_topk=match_topk,
            secondary_pair_weight=secondary_pair_weight,
            use_adaptive_groups=use_adaptive_groups,
        )


def count_instance_masks(
    instance_labels: torch.Tensor,
    min_pixels: int = 32,
) -> torch.Tensor:
    """Return the number of valid GT instance masks per view.

    ``instance_labels`` is [B,V,H,W] with raw ScanNet instance ids
    (0/255/-1 ignored). Used to supervise the scene-adaptive count head
    (target = max count over the labeled views of the sample).
    """
    batch, views = instance_labels.shape[:2]
    counts = torch.zeros(
        (batch, views), dtype=torch.long, device=instance_labels.device
    )
    for b in range(batch):
        for v in range(views):
            masks = _gt_masks_from_map(
                instance_labels[b, v], min_pixels=min_pixels
            )
            counts[b, v] = len(masks)
    return counts
