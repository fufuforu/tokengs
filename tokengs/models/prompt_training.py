"""Small, renderer-independent utilities for prompt mask supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_scores_to_gaussians(
    token_logits: torch.Tensor, num_gaussians_per_token: int
) -> torch.Tensor:
    """Expand [B,Q,T] token logits to [B,Q,T*P] without reordering tokens."""
    if token_logits.ndim != 3:
        raise ValueError("token_logits must have shape [B,Q,T]")
    if num_gaussians_per_token <= 0:
        raise ValueError("num_gaussians_per_token must be positive")
    return token_logits.sigmoid().repeat_interleave(num_gaussians_per_token, dim=-1)


def _broadcast_prompt_target(
    rendered_probability: torch.Tensor,
    target_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize prompt renderer [B,Q,V,1,H,W] and dataset [B,V,H,W] shapes."""
    if rendered_probability.ndim != 6:
        raise ValueError("rendered_probability must have shape [B,Q,V,1,H,W]")
    if target_mask.ndim == 4:
        target_mask = target_mask[:, None, :, None]
    elif target_mask.ndim == 5:
        target_mask = target_mask[:, None]
    if valid_mask.ndim == 4:
        valid_mask = valid_mask[:, None, :, None]
    elif valid_mask.ndim == 5:
        valid_mask = valid_mask[:, None]
    target_mask = target_mask.expand_as(rendered_probability)
    valid_mask = valid_mask.expand_as(rendered_probability)
    return rendered_probability, target_mask, valid_mask


def instance_contrastive_loss(
    rendered_features: torch.Tensor,
    instance_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 0.07,
    min_pixels: int = 32,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Cross-view instance-level InfoNCE on rendered per-pixel features.

    ``rendered_features`` is [B, V, D, H, W] (alpha-composited token
    features) and ``instance_labels`` is [B, V, H, W] with raw ScanNet
    instance ids (0/255 ignored). ScanNet instance ids are consistent across
    the frames of a scene, so pixels of the same instance id are aggregated
    across ALL target views into one prototype; every valid pixel is then
    classified against all prototypes (cross-entropy over instances). This
    directly teaches cross-view feature consistency and instance
    discriminability -- exactly what the feed-forward group head needs to
    separate instances without test-time optimization.
    """
    batch_size, view_count, dim, height, width = rendered_features.shape
    total = torch.zeros((), device=rendered_features.device)
    n_batches = 0
    for b in range(batch_size):
        feat = rendered_features[b]  # [V,D,H,W]
        labels = instance_labels[b]  # [V,H,W]
        valid = (
            valid_mask[b, :, 0]
            if valid_mask.ndim == 5
            else valid_mask[b]
        )  # [V,H,W]
        feat_flat = F.normalize(
            feat.reshape(view_count, dim, -1)
            .transpose(1, 2)
            .reshape(-1, dim),
            dim=-1,
        )  # [V*HW, D]
        labels_flat = labels.reshape(-1)
        valid_flat = valid.reshape(-1)
        unique_ids = torch.unique(labels_flat)
        keep = [
            int(value)
            for value in unique_ids.tolist()
            if int(value) not in (0, 255, -1)
        ]
        if len(keep) < 2:
            continue
        protos: list[torch.Tensor] = []
        id_to_index: dict[int, int] = {}
        for instance_id in keep:
            mask = valid_flat & (labels_flat == instance_id)
            if mask.sum() < min_pixels:
                continue
            proto = F.normalize(
                feat_flat[mask].mean(dim=0, keepdim=True), dim=-1
            )
            id_to_index[instance_id] = len(protos)
            protos.append(proto)
        if len(protos) < 2:
            continue
        prototype_matrix = torch.cat(protos, dim=0)  # [K,D]
        pixel_index = torch.full(
            (view_count * height * width,),
            -1,
            dtype=torch.long,
            device=labels.device,
        )
        for instance_id, index in id_to_index.items():
            mask = valid_flat & (labels_flat == instance_id)
            pixel_index[mask] = index
        selected = pixel_index >= 0
        if selected.sum() < min_pixels:
            continue
        logits = (
            feat_flat[selected] @ prototype_matrix.t() / temperature
        )  # [n,K]
        targets = pixel_index[selected]
        total = total + F.cross_entropy(logits, targets)
        n_batches += 1
    if n_batches == 0:
        return torch.zeros((), device=rendered_features.device)
    return total / n_batches


def compute_prompt_mask_loss(
    rendered_probability: torch.Tensor,
    target_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    lambda_bce: float = 1.0,
    lambda_dice: float = 1.0,
    balance_classes: bool = False,
    pos_weight: float = 0.5,
    class_weights: tuple[float, ...] | None = None,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Compute valid-pixel BCE + Dice in FP32 outside mixed-precision autocast."""
    with torch.autocast(device_type=rendered_probability.device.type, enabled=False):
        probability, target, valid = _broadcast_prompt_target(
            rendered_probability, target_mask, valid_mask
        )
        probability = probability.float()
        target = target.float()
        valid = valid.float()
        bce_map = F.binary_cross_entropy(
            probability.clamp(eps, 1.0 - eps), target, reduction="none"
        )
        if balance_classes:
            positive = target * valid
            negative = (1.0 - target) * valid
            class_reduce_dims = (0, 2, 3, 4, 5)
            positive_count = positive.sum(class_reduce_dims)
            negative_count = negative.sum(class_reduce_dims)
            positive_loss = (bce_map * positive).sum(
                class_reduce_dims
            ) / positive_count.clamp_min(eps)
            negative_loss = (bce_map * negative).sum(
                class_reduce_dims
            ) / negative_count.clamp_min(eps)
            class_bce = torch.where(
                positive_count > 0,
                float(pos_weight) * positive_loss
                + (1.0 - float(pos_weight)) * negative_loss,
                negative_loss,
            )
            if class_weights is not None:
                if len(class_weights) != class_bce.shape[0]:
                    raise ValueError(
                        "class_weights must match the number of classes "
                        f"({class_bce.shape[0]}), got {len(class_weights)}"
                    )
                weights = torch.as_tensor(
                    class_weights,
                    dtype=class_bce.dtype,
                    device=class_bce.device,
                )
                bce = (class_bce * weights).sum() / weights.sum().clamp_min(eps)
            else:
                bce = class_bce.mean()
        else:
            valid_count = valid.sum().clamp_min(eps)
            bce = (bce_map * valid).sum() / valid_count

        intersection = (probability * target * valid).sum(dim=(-1, -2, -3))
        pred_sum = (probability * valid).sum(dim=(-1, -2, -3))
        target_sum = (target * valid).sum(dim=(-1, -2, -3))
        dice = 1.0 - (
            (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        ).mean()
        total = float(lambda_bce) * bce + float(lambda_dice) * dice
    return {
        "loss": total,
        "loss_bce": bce,
        "loss_dice": dice,
    }


def compute_prompt_mask_metrics(
    rendered_probability: torch.Tensor,
    target_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Return FP32 segmentation diagnostics over valid target pixels."""
    with torch.autocast(device_type=rendered_probability.device.type, enabled=False):
        probability, target, valid = _broadcast_prompt_target(
            rendered_probability, target_mask, valid_mask
        )
        probability = probability.float()
        target = target.float()
        valid = valid.float()
        valid_bool = valid > 0.5
        predicted = probability >= float(threshold)
        target_bool = target >= 0.5
        intersection = (predicted & target_bool & valid_bool).sum().float()
        union = ((predicted | target_bool) & valid_bool).sum().float()
        iou = intersection / union.clamp_min(eps)
        true_positive = (predicted & target_bool & valid_bool).sum().float()
        target_count = (target_bool & valid_bool).sum().float()
        mask_accuracy = true_positive / target_count.clamp_min(eps)
        foreground_probability = (probability * target * valid).sum() / (
            (target * valid).sum().clamp_min(eps)
        )
        background_probability = (
            (1.0 - probability) * (1.0 - target) * valid
        ).sum() / (((1.0 - target) * valid).sum().clamp_min(eps))
        valid_count = valid.sum().clamp_min(eps)
        predicted_foreground_ratio = (predicted.float() * valid).sum() / valid_count
        gt_foreground_ratio = (target * valid).sum() / valid_count
    return {
        "mask_iou": iou,
        "mask_accuracy": mask_accuracy,
        "foreground_probability": foreground_probability,
        "background_probability": background_probability,
        "predicted_foreground_ratio": predicted_foreground_ratio,
        "gt_foreground_ratio": gt_foreground_ratio,
    }
