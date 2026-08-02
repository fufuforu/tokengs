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


def compute_prompt_mask_loss(
    rendered_probability: torch.Tensor,
    target_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    lambda_bce: float = 1.0,
    lambda_dice: float = 1.0,
    balance_classes: bool = False,
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
                0.5 * (positive_loss + negative_loss),
                negative_loss,
            )
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
        "foreground_probability": foreground_probability,
        "background_probability": background_probability,
        "predicted_foreground_ratio": predicted_foreground_ratio,
        "gt_foreground_ratio": gt_foreground_ratio,
    }
