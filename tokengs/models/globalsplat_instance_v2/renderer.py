from __future__ import annotations

import torch

from .dependency import GlobalSplatSymbols


def render_rgb_sh(symbols: GlobalSplatSymbols, gaussians, target_meta: dict[str, torch.Tensor], *, render_depth: bool = True) -> dict[str, torch.Tensor]:
    rendered = symbols.render_static_batched(gaussians, target_meta, render_depth=render_depth)
    b, t, _, h, w = target_meta["images"].shape
    out = {
        "images_pred": rendered["img"].reshape(b, t, 3, h, w),
        "alphas_pred": rendered["acc"].reshape(b, t, 1, h, w),
    }
    if render_depth:
        out["depths_pred"] = rendered["depth"].reshape(b, t, h, w, 1).permute(0, 1, 4, 2, 3).contiguous()
    else:
        out["depths_pred"] = None
    return out


def _quaternion(symbols: GlobalSplatSymbols, rotations: torch.Tensor) -> torch.Tensor:
    if rotations.shape[-1] == 4:
        return rotations.contiguous()
    if rotations.shape[-1] != 6:
        raise ValueError(f"rotations must end in 6 or 4, got {tuple(rotations.shape)}")
    matrices = symbols.rotation_6d_to_matrix(rotations.reshape(-1, 6).contiguous())
    return symbols.matrix_to_quaternion(matrices).reshape(rotations.shape[0], rotations.shape[1], 4).contiguous()


def render_feature_channels_sh_geometry(symbols: GlobalSplatSymbols, gaussians, features: torch.Tensor,
                                        target_meta: dict[str, torch.Tensor], *, near_plane: float = 1e-2,
                                        eps2d: float = 0.05, packed: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 3 or features.shape[:2] != gaussians.means.shape[:2]:
        raise ValueError("features must be [B,N,D] aligned with Gaussians")
    from gsplat import rasterization
    images = target_meta["images"]
    b, t, _, h, w = images.shape
    device = gaussians.means.device
    means = gaussians.means.float().contiguous()
    scales = gaussians.scales.float().contiguous()
    rotations = gaussians.rotations.float().contiguous()
    quats = _quaternion(symbols, rotations)
    opacities = gaussians.opacities.float()
    if opacities.shape[-1] == 1:
        opacities = opacities.squeeze(-1)
    colors = features.float().contiguous()
    Ks = target_meta["intrinsic"].to(device=device, dtype=torch.float32).contiguous()
    viewmats = target_meta["extrinsic"].to(device=device, dtype=torch.float32).contiguous()
    # gsplat's packed path consumes a channel-only background because packed
    # means2d has no batch/camera dimensions at rasterize_to_pixels time.
    backgrounds = torch.zeros(colors.shape[-1], device=device, dtype=torch.float32) if packed else torch.zeros(b, t, colors.shape[-1], device=device, dtype=torch.float32)
    with torch.autocast(device_type=device.type, enabled=False):
        rendered, alpha, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=viewmats, Ks=Ks, width=w, height=h,
            packed=packed, render_mode="RGB", backgrounds=backgrounds,
            camera_model="pinhole", eps2d=eps2d, near_plane=near_plane,
        )
    composited = rendered.permute(0, 1, 4, 2, 3).contiguous()
    alpha_out = alpha.permute(0, 1, 4, 2, 3).contiguous()
    return composited, alpha_out


def normalize_rendered_assignment(composited: torch.Tensor, alpha: torch.Tensor,
                                  eps_alpha: float = 1e-5, eps_prob: float = 1e-6) -> torch.Tensor:
    if composited.ndim != 5 or alpha.ndim != 5 or composited.shape[:2] != alpha.shape[:2] or composited.shape[2] != 101:
        raise ValueError("composited must be [B,V,101,H,W] and alpha [B,V,1,H,W]")
    conditional = composited / alpha.clamp_min(eps_alpha)
    conditional = conditional / conditional.sum(dim=2, keepdim=True).clamp_min(eps_prob)
    foreground = alpha >= eps_alpha
    conditional[:, :, -1:] = torch.where(
        foreground, conditional[:, :, -1:], torch.ones_like(conditional[:, :, -1:])
    )
    conditional[:, :, :-1] = torch.where(
        foreground, conditional[:, :, :-1], torch.zeros_like(conditional[:, :, :-1])
    )
    result = conditional.permute(0, 2, 1, 3, 4).unsqueeze(3).contiguous()
    if not torch.isfinite(result).all():
        raise RuntimeError("non-finite normalized assignment")
    return result
