from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .types import SceneAssignment


def collect_scene_instances(labels: torch.Tensor, min_visible_pixels: int = 32,
                           ignore_ids: tuple[int, ...] = (0, 255, -1), max_objects: int = 100):
    if labels.ndim != 3:
        raise ValueError(f"labels must be [7,H,W], got {tuple(labels.shape)}")
    ignore = set(ignore_ids)
    candidates = {}
    masks = {}
    visible = {}
    ambiguous = {}
    totals = {}
    for v in range(labels.shape[0]):
        for value in torch.unique(labels[v]).tolist():
            raw = int(value)
            if raw in ignore:
                continue
            count = int((labels[v] == raw).sum().item())
            totals[raw] = totals.get(raw, 0) + count
            masks.setdefault(raw, {})[v] = labels[v] == raw
            if count >= min_visible_pixels:
                visible.setdefault(raw, []).append(v)
            elif count > 0:
                ambiguous.setdefault(raw, []).append(v)
    instance_ids = sorted(raw for raw, views in visible.items() if views)
    dropped = []
    if len(instance_ids) > max_objects:
        kept = sorted(instance_ids, key=lambda raw: (-totals[raw], raw))[:max_objects]
        dropped = sorted(set(instance_ids) - set(kept))
        instance_ids = sorted(kept)
    for raw in instance_ids:
        visible[raw] = tuple(sorted(visible[raw]))
        ambiguous.setdefault(raw, [])
        for v in range(labels.shape[0]):
            masks.setdefault(raw, {})
            masks[raw].setdefault(v, labels[v] == raw)
    return instance_ids, masks, {k: tuple(v) for k, v in visible.items() if k in instance_ids}, {k: tuple(v) for k, v in ambiguous.items() if k in instance_ids}, tuple(dropped)


def _bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Probability BCE is intentionally kept (the head exposes probabilities),
    # but must run outside CUDA autocast; PyTorch marks this kernel unsafe.
    with torch.autocast(device_type=pred.device.type, enabled=False):
        return F.binary_cross_entropy(pred.float().clamp(1e-6, 1.0 - 1e-6), target.float(), reduction="mean")


def _bce_per_query(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=pred.device.type, enabled=False):
        return F.binary_cross_entropy(pred.float().clamp(1e-6, 1.0 - 1e-6), target.float(), reduction="none")


def _dice(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float().reshape(-1)
    target = target.float().reshape(-1)
    return 1.0 - (2 * (pred * target).sum() + 1e-6) / (pred.sum() + target.sum() + 1e-6)


def build_multiview_cost(query_probability: torch.Tensor, instance_ids: list[int], masks: dict[int, dict[int, torch.Tensor]],
                         visible_views: dict[int, tuple[int, ...]], bce_weight: float, dice_weight: float) -> torch.Tensor:
    if query_probability.ndim != 4:
        raise ValueError("query_probability must be [100,7,H,W]")
    costs = []
    for raw in instance_ids:
        view_costs = []
        for v in visible_views[raw]:
            gt = masks[raw][v].to(device=query_probability.device)
            pred = query_probability[:, v]
            bce = _bce_per_query(pred, gt.float().expand_as(pred)).mean(dim=(-1, -2))
            inter = (pred * gt.float()).flatten(1).sum(-1)
            dice = 1.0 - (2 * inter + 1e-6) / (pred.flatten(1).sum(-1) + gt.float().sum() + 1e-6)
            view_costs.append(float(bce_weight) * bce + float(dice_weight) * dice)
        costs.append(torch.stack(view_costs, dim=0).mean(dim=0))
    return torch.stack(costs, dim=1).float().detach()


def derive_scene_object_semantic_targets(
    instance_ids_vhw: torch.Tensor,
    semantic_ids_vhw: torch.Tensor,
    object_ids: torch.Tensor,
    raw_to_contiguous: dict[int, int],
    *,
    ignore_ids: tuple[int, ...],
    min_purity: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive one semantic class for each already-ordered scene object."""
    if instance_ids_vhw.ndim != 3 or semantic_ids_vhw.ndim != 3:
        raise ValueError("instance and semantic IDs must be [V,H,W]")
    if instance_ids_vhw.shape != semantic_ids_vhw.shape:
        raise ValueError("instance and semantic ID shapes must match")
    if object_ids.ndim != 1:
        raise ValueError("object_ids must be [O]")
    classes = torch.full_like(object_ids, -1, dtype=torch.long)
    valid = torch.zeros_like(object_ids, dtype=torch.bool)
    purity = torch.zeros(object_ids.shape, dtype=torch.float32, device=object_ids.device)
    ignore = set(int(value) for value in ignore_ids)
    for object_index, object_id in enumerate(object_ids.tolist()):
        counts: dict[int, int] = {}
        total = 0
        object_mask = instance_ids_vhw == int(object_id)
        for value in semantic_ids_vhw[object_mask].tolist():
            raw_value = int(value)
            if raw_value in ignore:
                continue
            mapped = raw_to_contiguous.get(raw_value)
            if mapped is None or int(mapped) in ignore:
                continue
            mapped = int(mapped)
            counts[mapped] = counts.get(mapped, 0) + 1
            total += 1
        if total:
            class_id, count = max(counts.items(), key=lambda item: (item[1], -item[0]))
            classes[object_index] = class_id
            purity[object_index] = float(count) / float(total)
            valid[object_index] = bool(purity[object_index] >= float(min_purity))
    return classes, valid, purity


def semantic_class_matching_cost(
    semantic_logits_qc: torch.Tensor,
    target_classes_o: torch.Tensor,
    target_valid_o: torch.Tensor,
) -> torch.Tensor:
    if semantic_logits_qc.ndim != 2 or target_classes_o.ndim != 1 or target_valid_o.ndim != 1:
        raise ValueError("semantic matching inputs must be [Q,C+1], [O], [O]")
    if target_classes_o.shape != target_valid_o.shape:
        raise ValueError("target class and validity shapes must match")
    with torch.autocast(device_type=semantic_logits_qc.device.type, enabled=False):
        probabilities = F.softmax(semantic_logits_qc.float(), dim=-1)
        cost = probabilities.new_zeros((semantic_logits_qc.shape[0], target_classes_o.shape[0]))
        valid_indices = torch.nonzero(target_valid_o, as_tuple=False).flatten()
        if valid_indices.numel():
            target_classes = target_classes_o[valid_indices].long()
            if (target_classes < 0).any() or (target_classes >= semantic_logits_qc.shape[1] - 1).any():
                raise ValueError("valid semantic target class is outside logits class range")
            cost[:, valid_indices] = -torch.log(
                probabilities[:, target_classes].clamp_min(1e-8)
            )
    return cost.float()


def semantic_query_classification_loss(
    semantic_logits_qc: torch.Tensor,
    matched_query_indices: torch.Tensor,
    matched_object_indices: torch.Tensor,
    target_classes_o: torch.Tensor,
    target_valid_o: torch.Tensor,
    *,
    no_object_index: int,
    eos_coef: float = 0.1,
) -> torch.Tensor:
    if semantic_logits_qc.ndim != 2:
        raise ValueError("semantic_logits_qc must be [Q,C+1]")
    if matched_query_indices.shape != matched_object_indices.shape:
        raise ValueError("matched query/object index shapes must match")
    with torch.autocast(device_type=semantic_logits_qc.device.type, enabled=False):
        logits = semantic_logits_qc.float()
        targets = torch.full(
            (logits.shape[0],), int(no_object_index), dtype=torch.long, device=logits.device
        )
        sample_weight = torch.full_like(logits[:, 0], float(eos_coef))
        if matched_query_indices.numel():
            q = matched_query_indices.long().to(logits.device)
            o = matched_object_indices.long().to(logits.device)
            valid_matches = target_valid_o[o].to(logits.device)
            if valid_matches.any():
                targets[q[valid_matches]] = target_classes_o[o[valid_matches]].long().to(logits.device)
                sample_weight[q[valid_matches]] = 1.0
            if (~valid_matches).any():
                sample_weight[q[~valid_matches]] = 0.0
        per_query = F.cross_entropy(logits, targets, reduction="none")
        normalization = sample_weight.sum().clamp_min(1.0)
        return (per_query * sample_weight).sum() / normalization


def hungarian_once(cost: torch.Tensor, instance_ids: list[int]) -> dict[int, int]:
    if cost.numel() == 0 or not instance_ids:
        return {}
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return {int(row): int(instance_ids[int(col)]) for row, col in zip(rows, cols)}


def scene_global_hungarian_instance_loss(rendered_probability: torch.Tensor, instance_labels: torch.Tensor, *,
                                         num_queries: int = 100, min_visible_pixels: int = 32,
                                         bce_weight: float = 1.0, dice_weight: float = 1.0,
                                         void_weight: float = 0.1, unmatched_weight: float = 0.1,
                                         absent_view_weight: float = 0.25,
                                         ignore_ids: tuple[int, ...] = (0, 255, -1),
                                         semantic_query_logits: torch.Tensor | None = None,
                                         semantic_labels: torch.Tensor | None = None,
                                         semantic_match_cost_weight_eff: float = 0.0,
                                         semantic_loss_weight_eff: float = 0.0,
                                         semantic_eos_coef: float = 0.1,
                                         semantic_min_purity: float = 0.95,
                                         semantic_raw_to_contiguous: dict[int, int] | None = None):
    if rendered_probability.ndim != 6 or instance_labels.ndim != 4:
        raise ValueError("probability must be [B,101,7,1,H,W] and labels [B,7,H,W]")
    b, channels, views, one, h, w = rendered_probability.shape
    if channels != num_queries + 1 or views != 7 or one != 1 or instance_labels.shape != (b, 7, h, w):
        raise ValueError(f"invalid scene instance loss shapes: probability={tuple(rendered_probability.shape)} labels={tuple(instance_labels.shape)}")
    semantic_enabled = semantic_query_logits is not None
    if semantic_enabled:
        if semantic_labels is None:
            raise ValueError("semantic_labels are required when semantic logits are provided")
        if semantic_query_logits.ndim != 3 or semantic_query_logits.shape[:2] != (b, num_queries):
            raise ValueError("semantic_query_logits must be [B,100,C+1]")
        if semantic_labels.shape != instance_labels.shape:
            raise ValueError("semantic_labels must match instance_labels shape")
        if semantic_raw_to_contiguous is None:
            semantic_raw_to_contiguous = {
                value: value for value in range(int(semantic_query_logits.shape[-1]))
            }
    elif semantic_labels is not None:
        raise ValueError("semantic_labels cannot be provided without semantic logits")
    probs = rendered_probability.float()[:, :num_queries, :, 0]
    total = rendered_probability.float().sum() * 0.0
    values = {name: rendered_probability.new_zeros(()) for name in (
        "gsi_v2_instance_gt_count", "gsi_v2_instance_matched_count", "gsi_v2_hungarian_calls",
        "gsi_v2_loss_matched", "gsi_v2_loss_void", "gsi_v2_loss_unmatched", "gsi_v2_dropped_gt_count",
        "gsi_v2_visible_object_views", "gsi_v2_target_view_count",
        "gsi_v21_loss_semantic_query_raw", "gsi_v21_matched_semantic_accuracy",
        "gsi_v21_semantic_valid_object_ratio", "gsi_v21_no_object_prediction_ratio")}
    values["gsi_v2_target_view_count"] = rendered_probability.new_tensor(float(b * 7))
    assignments = []
    for bi in range(b):
        labels = instance_labels[bi].long()
        ids, masks, visible, ambiguous, dropped = collect_scene_instances(labels, min_visible_pixels, ignore_ids, num_queries)
        values["gsi_v2_instance_gt_count"] += len(ids)
        values["gsi_v2_dropped_gt_count"] += len(dropped)
        values["gsi_v2_visible_object_views"] += sum(len(v) for v in visible.values())
        target_classes = target_valid = None
        if semantic_enabled:
            target_classes, target_valid, _purity = derive_scene_object_semantic_targets(
                labels,
                semantic_labels[bi].long(),
                torch.tensor(ids, dtype=torch.long, device=semantic_query_logits.device),
                semantic_raw_to_contiguous,
                ignore_ids=ignore_ids,
                min_purity=semantic_min_purity,
            )
            values["gsi_v21_semantic_valid_object_ratio"] += (
                target_valid.float().mean() if target_valid.numel() else rendered_probability.new_zeros(())
            )
        cost = build_multiview_cost(probs[bi], ids, masks, visible, bce_weight, dice_weight) if ids else None
        if semantic_enabled and ids:
            semantic_cost = semantic_class_matching_cost(
                semantic_query_logits[bi], target_classes, target_valid
            ).to(device=probs.device)
            cost = cost + float(semantic_match_cost_weight_eff) * semantic_cost
        mapping = hungarian_once(cost, ids) if cost is not None else {}
        values["gsi_v2_hungarian_calls"] += int(bool(ids))
        values["gsi_v2_instance_matched_count"] += len(mapping)
        assignments.append(SceneAssignment(bi, mapping, visible, tuple(dropped)))
        if semantic_enabled:
            matched_queries = torch.tensor(list(mapping.keys()), dtype=torch.long, device=semantic_query_logits.device)
            matched_objects = torch.tensor(
                [ids.index(mapping[query]) for query in mapping],
                dtype=torch.long,
                device=semantic_query_logits.device,
            )
            semantic_raw = semantic_query_classification_loss(
                semantic_query_logits[bi], matched_queries, matched_objects,
                target_classes, target_valid,
                no_object_index=semantic_query_logits.shape[-1] - 1,
                eos_coef=semantic_eos_coef,
            )
            values["gsi_v21_loss_semantic_query_raw"] += semantic_raw
            valid_matched = target_valid[matched_objects] if matched_objects.numel() else target_valid.new_zeros((0,))
            if valid_matched.any():
                predicted = semantic_query_logits[bi, matched_queries[valid_matched]].argmax(dim=-1)
                expected = target_classes[matched_objects[valid_matched]]
                values["gsi_v21_matched_semantic_accuracy"] += (predicted == expected).float().mean()
            values["gsi_v21_no_object_prediction_ratio"] += (
                (semantic_query_logits[bi].argmax(dim=-1) == semantic_query_logits.shape[-1] - 1).float().mean()
            )
        matched_terms = []
        for query, raw in mapping.items():
            for v in visible[raw]:
                pred = probs[bi, query, v]
                gt = masks[raw][v].to(pred.device)
                matched_terms.append(float(bce_weight) * _bce(pred, gt) + float(dice_weight) * _dice(pred, gt))
            for v in range(views):
                if v not in visible[raw] and v not in ambiguous.get(raw, ()):
                    matched_terms.append(float(absent_view_weight) * _bce(probs[bi, query, v], torch.zeros_like(probs[bi, query, v])))
        matched = torch.stack(matched_terms).mean() if matched_terms else rendered_probability.new_zeros(())
        ignored = (labels == 255) | (labels == -1)
        void_mask = labels == 0
        void_terms = [_bce(rendered_probability[bi, num_queries, v, 0], void_mask[v].float()) for v in range(views)]
        void_loss = torch.stack(void_terms).mean()
        matched_queries = set(mapping)
        unmatched = [q for q in range(num_queries) if q not in matched_queries]
        valid = ~ignored
        unmatched_loss = (_bce(probs[bi, unmatched], torch.zeros_like(probs[bi, unmatched])) if unmatched else rendered_probability.new_zeros(()))
        # Ignore pixels marked by ignore IDs for the unmatched term.
        if unmatched:
            pred = probs[bi, unmatched]
            valid_expanded = valid.unsqueeze(0).expand_as(pred)
            unmatched_loss = _bce_per_query(pred, torch.zeros_like(pred))[valid_expanded].mean()
        total = total + matched + float(void_weight) * void_loss + float(unmatched_weight) * unmatched_loss
        values["gsi_v2_loss_matched"] += matched.detach()
        values["gsi_v2_loss_void"] += void_loss.detach()
        values["gsi_v2_loss_unmatched"] += unmatched_loss.detach()
    total = total / max(1, b)
    values = {key: value / max(1, b) for key, value in values.items()}
    values["gsi_v2_target_view_count"] = rendered_probability.new_tensor(float(b * 7))
    return total, values, assignments
