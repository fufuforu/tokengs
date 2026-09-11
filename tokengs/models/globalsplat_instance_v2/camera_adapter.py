from __future__ import annotations

from typing import Any

import torch


def intrinsics_vec_to_matrix(intrinsics: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(intrinsics) or intrinsics.ndim != 3 or intrinsics.shape[-1] != 4:
        raise ValueError(f"intrinsics must be [B,V,4], got {getattr(intrinsics, 'shape', None)}")
    if not torch.isfinite(intrinsics).all() or (intrinsics[..., :2] <= 0).any():
        raise ValueError("intrinsics must be finite and fx/fy must be positive")
    matrix = torch.zeros(*intrinsics.shape[:-1], 3, 3, device=intrinsics.device, dtype=intrinsics.dtype)
    matrix[..., 0, 0] = intrinsics[..., 0]
    matrix[..., 1, 1] = intrinsics[..., 1]
    matrix[..., 0, 2] = intrinsics[..., 2]
    matrix[..., 1, 2] = intrinsics[..., 3]
    matrix[..., 2, 2] = 1
    return matrix


def _field(obj: Any, name: str) -> torch.Tensor:
    value = getattr(obj, name, None)
    if value is None:
        raise ValueError(f"missing model input field {name}")
    return value


def make_official_context_input(model_input: Any) -> dict[str, torch.Tensor]:
    encoder = _field(model_input, "encoder")
    images = _field(encoder, "images_rgb_unnormalized")
    intrinsics = _field(encoder, "intrinsics_input")
    c2w = _field(encoder, "cam_to_world_input")
    if images.ndim != 5 or images.shape[1] != 8 or images.shape[2] != 3:
        raise ValueError(f"context RGB must be [B,8,3,H,W], got {tuple(images.shape)}")
    if not torch.isfinite(images).all() or images.amin() < -1e-4 or images.amax() > 1.0001:
        raise ValueError("context RGB must be finite and in [0,1] within tolerance")
    if c2w.shape != (images.shape[0], 8, 4, 4):
        raise ValueError(f"context c2w shape mismatch: {tuple(c2w.shape)}")
    K = intrinsics_vec_to_matrix(intrinsics)
    if not torch.isfinite(c2w).all():
        raise ValueError("context c2w must be finite")
    return {"images": images, "intrinsic": K, "c2w": c2w}


def split_eight_context_anchor_alternating(
    context: dict[str, torch.Tensor], flip: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build the fixed R1 two-branch context split from the formal 8 views."""
    if context["images"].ndim != 5 or context["images"].shape[1] != 8:
        raise ValueError("subset consistency requires an 8-view context")
    a = [0, 7, 1, 3, 5]
    b = [0, 7, 2, 4, 6]
    if flip:
        a, b = b, a
    return (
        {key: value[:, a] for key, value in context.items()},
        {key: value[:, b] for key, value in context.items()},
    )


def make_official_target_meta(model_input: Any, image_hw: tuple[int, int]) -> dict[str, torch.Tensor]:
    decoder = _field(model_input, "decoder")
    cam_view = _field(decoder, "cam_view")
    intrinsics = _field(decoder, "intrinsics")
    if cam_view.ndim != 4 or cam_view.shape[1] != 7 or cam_view.shape[-2:] != (4, 4):
        raise ValueError(f"target cam_view must be [B,7,4,4], got {tuple(cam_view.shape)}")
    if intrinsics.ndim != 3 or intrinsics.shape[1] != 7:
        raise ValueError(f"target intrinsics must be [B,7,4], got {tuple(intrinsics.shape)}")
    H, W = image_hw
    images = torch.zeros(cam_view.shape[0], 7, 3, H, W, device=cam_view.device, dtype=cam_view.dtype)
    extrinsic = cam_view.transpose(-1, -2).contiguous()
    if not torch.isfinite(extrinsic).all():
        raise ValueError("target camera must be finite")
    return {"images": images, "intrinsic": intrinsics_vec_to_matrix(intrinsics), "extrinsic": extrinsic}


def make_official_context_frustum_meta(model_input: Any) -> tuple[torch.Tensor, torch.Tensor]:
    context = make_official_context_input(model_input)
    return context["intrinsic"], torch.linalg.inv(context["c2w"])
