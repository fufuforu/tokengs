"""C3G-style semantic feature lifting (ported from the earlier tokengs_c3g
experiment that reached ~0.536 C3G8 mIoU).

The recipe: extract LSeg features from the context (source) views, project
them onto the 3D Gaussian centers via the source cameras (with an optional
depth filter), fuse across views, render the per-Gaussian features to the
source/target views, and finally decode with LSeg's own ``output_conv`` +
text decoder. This is pure lifting -- no learned semantic decoder -- so it
can be evaluated on any existing checkpoint (e.g. the instance-grouping
wide7l model) without retraining.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SourceFeatureProjector(nn.Module):
    """Project 3D Gaussian centers onto source LSeg feature maps and fuse."""

    def __init__(
        self,
        image_hw: tuple[int, int],
        chunk_size: int = 8192,
        depth_rel_tol: float = 0.10,
        znear: float = 0.025,
    ):
        super().__init__()
        self.image_hw = tuple(image_hw)
        self.chunk_size = int(chunk_size)
        self.depth_rel_tol = float(depth_rel_tol)
        self.znear = float(znear)

    @torch.no_grad()
    def forward(
        self,
        xyz_world: torch.Tensor,
        source_features: torch.Tensor,
        source_c2w: torch.Tensor,
        source_intrinsics: torch.Tensor,
        source_depth: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse source-view features onto Gaussian centers.

        Args:
            xyz_world: [B,N,3] Gaussian centers in world coordinates.
            source_features: [B,V,C,Hf,Wf] LSeg features of the source views.
            source_c2w: [B,V,4,4].
            source_intrinsics: [B,V,4] (fx, fy, cx, cy) at ``image_hw``.
            source_depth: optional [B,V,1,H,W] for the depth filter.

        Returns:
            fused_features [B,N,C], has_source [B,N,1] bool,
            confidence [B,N,1] (#valid sources / V).
        """
        if xyz_world.ndim != 3:
            raise ValueError(f"xyz_world must be [B,N,3], got {xyz_world.shape}")
        batch, num_points, _ = xyz_world.shape
        _, num_views, channels, feat_h, feat_w = source_features.shape
        img_h, img_w = self.image_hw

        w2c = torch.linalg.inv(source_c2w.float())
        feature_flat = source_features.reshape(
            batch * num_views, channels, feat_h, feat_w
        ).float()

        if source_depth is not None:
            depth_feature = F.interpolate(
                source_depth.reshape(
                    batch * num_views,
                    1,
                    source_depth.shape[-2],
                    source_depth.shape[-1],
                ).float(),
                size=(feat_h, feat_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, num_views, 1, feat_h, feat_w)
        else:
            depth_feature = None

        fused_chunks: list[torch.Tensor] = []
        valid_chunks: list[torch.Tensor] = []
        conf_chunks: list[torch.Tensor] = []
        for start in range(0, num_points, self.chunk_size):
            end = min(start + self.chunk_size, num_points)
            xyz = xyz_world[:, start:end].float()
            chunk_n = xyz.shape[1]
            xyz_h = torch.cat(
                [
                    xyz,
                    torch.ones(
                        batch, chunk_n, 1, device=xyz.device, dtype=xyz.dtype
                    ),
                ],
                dim=-1,
            )
            xyz_cam_h = torch.einsum(
                "bvij,bnj->bvni", w2c, xyz_h
            )
            xyz_cam = xyz_cam_h[..., :3]
            x, y, z = xyz_cam[..., 0], xyz_cam[..., 1], xyz_cam[..., 2]
            fx = source_intrinsics[..., 0, None]
            fy = source_intrinsics[..., 1, None]
            cx = source_intrinsics[..., 2, None]
            cy = source_intrinsics[..., 3, None]
            z_safe = z.clamp_min(1e-6)
            u = fx * x / z_safe + cx
            v = fy * y / z_safe + cy
            grid_x = 2.0 * u / max(img_w - 1, 1) - 1.0
            grid_y = 2.0 * v / max(img_h - 1, 1) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1)
            valid = (
                (z > self.znear)
                & (u >= 0)
                & (u <= img_w - 1)
                & (v >= 0)
                & (v <= img_h - 1)
            )
            grid_flat = grid.reshape(batch * num_views, chunk_n, 1, 2)
            sampled = F.grid_sample(
                feature_flat,
                grid_flat,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled = (
                sampled.squeeze(-1)
                .permute(0, 2, 1)
                .reshape(batch, num_views, chunk_n, channels)
            )
            if depth_feature is not None:
                sampled_depth = F.grid_sample(
                    depth_feature.reshape(batch * num_views, 1, feat_h, feat_w),
                    grid_flat,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                sampled_depth = (
                    sampled_depth.squeeze(1)
                    .squeeze(-1)
                    .reshape(batch, num_views, chunk_n)
                )
                valid = valid & (sampled_depth > self.znear) & (
                    (z - sampled_depth).abs()
                    / sampled_depth.clamp_min(1e-6)
                    < self.depth_rel_tol
                )
            weights = valid.float()
            denominator = weights.sum(dim=1, keepdim=False).unsqueeze(-1)
            fused = (
                sampled * weights[..., None]
            ).sum(dim=1) / denominator.clamp_min(1.0)
            fused_chunks.append(fused)
            valid_chunks.append(denominator > 0)
            conf_chunks.append((denominator / float(num_views)).clamp(0.0, 1.0))

        return (
            torch.cat(fused_chunks, dim=1),
            torch.cat(valid_chunks, dim=1),
            torch.cat(conf_chunks, dim=1),
        )


class GaussianSemanticHeadV2(nn.Module):
    """Fusion head: projected LSeg features + token/geometry residual.

    Ported from the tokengs_c3g experiment (source_projected branch).
    Initial residual is zero, so the head starts as pure lifting and the
    token/geometry terms learn to refine the projected LSeg features.
    """

    def __init__(
        self,
        token_dim: int = 1024,
        feature_dim: int = 512,
        local_dim: int = 128,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.local_dim = int(local_dim)
        self.residual_scale = float(residual_scale)
        self.token_fallback = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim)),
            nn.GELU(),
            nn.Linear(int(token_dim), int(feature_dim)),
        )
        self.token_local = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(local_dim)),
        )
        self.gaussian_local = nn.Sequential(
            nn.Linear(14, int(local_dim)),
            nn.LayerNorm(int(local_dim)),
            nn.GELU(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(int(local_dim) * 2 + 1, int(local_dim) * 2),
            nn.GELU(),
            nn.Linear(int(local_dim) * 2, int(feature_dim)),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        decoded_tokens: torch.Tensor,
        gaussians: torch.Tensor,
        projected_features: torch.Tensor,
        has_source: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        batch, token_count, _ = decoded_tokens.shape
        _, gaussian_count, gaussian_dim = gaussians.shape
        if gaussian_dim != 14:
            raise ValueError(f"Expected Gaussian dim 14, got {gaussian_dim}")
        if gaussian_count % token_count != 0:
            raise ValueError(
                f"N_gaussian={gaussian_count} not divisible by "
                f"N_token={token_count}"
            )
        gs_per_token = gaussian_count // token_count
        fallback = self.token_fallback(decoded_tokens).repeat_interleave(
            gs_per_token, dim=1
        )
        base_feature = torch.where(has_source, projected_features, fallback)
        token_local = self.token_local(decoded_tokens).repeat_interleave(
            gs_per_token, dim=1
        )
        gaussian_local = self.gaussian_local(gaussians)
        residual_input = torch.cat(
            [token_local, gaussian_local, confidence], dim=-1
        )
        residual = self.residual_head(residual_input)
        return base_feature + self.residual_scale * residual


class RenderedSemanticClassifier(nn.Module):
    """1x1-conv closed-set classifier over rendered 512D LSeg features.

    Ported from the tokengs_c3g recipe (``semantic_classifier_hidden_dim``).
    Used only during training: its pseudo-label CE loss gives the semantic
    lifting head strong, class-discriminative gradients (the plain
    cosine/L1 feature loss alone leaves the learned residual near zero).
    """

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 8,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                f"features must be [N,C,H,W], got {tuple(features.shape)}"
            )
        return self.net(features)


def lseg_features_to_pseudo_labels(
    teacher,
    features: torch.Tensor,
    labelset: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """LSeg text-decode half-res features into pseudo labels + confidence.

    Args:
        teacher: LSegTeacher (exposes ``.extractor``).
        features: [B,V,512,Hf,Wf] half-resolution LSeg features.
        labelset: ScanNet class names.

    Returns:
        labels: [B,V,Hf,Wf], values 0..K-1.
        confidence: [B,V,Hf,Wf].
    """
    if features.ndim != 5:
        raise ValueError(
            f"features must be [B,V,C,H,W], got {tuple(features.shape)}"
        )
    batch, views, channels, height, width = features.shape
    flat = features.reshape(batch * views, channels, height, width).float()
    decode_features = teacher.extractor.scratch.output_conv(flat)
    logits = teacher.extractor.decode_feature(
        decode_features, labelset=labelset
    )
    probability = torch.softmax(logits.float(), dim=1)
    confidence, labels = probability.max(dim=1)
    # ``output_conv`` may upsample to the full image resolution, so keep
    # whatever spatial size the text decoder returns.
    labels = labels.reshape(batch, views, *labels.shape[-2:])
    confidence = confidence.reshape(batch, views, *confidence.shape[-2:])
    return labels.long(), confidence.float()


def semantic_feature_loss(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    alpha: torch.Tensor,
    lambda_cosine: float = 1.0,
    lambda_l1: float = 0.05,
    alpha_threshold: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Cosine + smooth-L1 between rendered and teacher LSeg features."""
    if teacher.shape[-2:] != prediction.shape[-2:]:
        batch, views, channels, height, width = teacher.shape
        teacher = F.interpolate(
            teacher.reshape(batch * views, channels, height, width),
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(
            batch,
            views,
            channels,
            prediction.shape[-2],
            prediction.shape[-1],
        )
    pred_n = F.normalize(prediction.float(), dim=2, eps=eps)
    teacher_n = F.normalize(teacher.float(), dim=2, eps=eps)
    valid = (alpha.detach() > float(alpha_threshold)).float()
    denominator = valid.sum().clamp_min(1.0)
    cosine = 1.0 - (pred_n * teacher_n).sum(dim=2, keepdim=True)
    loss_cosine = (cosine * valid).sum() / denominator
    l1_map = F.smooth_l1_loss(
        prediction.float(), teacher.float(), reduction="none"
    ).mean(dim=2, keepdim=True)
    loss_l1 = (l1_map * valid).sum() / denominator
    return float(lambda_cosine) * loss_cosine + float(lambda_l1) * loss_l1
