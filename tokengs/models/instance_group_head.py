"""Learned anchor-to-instance-group assignment head (InstOk3D-style)."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokengs.models.instance_group_loss import (
    count_instance_masks,
    hungarian_instance_group_loss,
    instance_group_3d_loss,
    _gs_majority_target,
    _project_gs_to_views,
)
from tokengs.rendering.gs import GaussianRenderer


class _ObjectQueryDecoderLayer(nn.Module):
    """DETR-style object-query refinement over scene anchor features."""

    def __init__(self, dim: int, num_heads: int, mlp_hidden: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_query_norm = nn.LayerNorm(dim)
        self.cross_anchor_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, queries: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        normalized = self.query_norm(queries)
        attended, _ = self.self_attn(normalized, normalized, normalized)
        queries = queries + attended
        attended, _ = self.cross_attn(
            self.cross_query_norm(queries),
            self.cross_anchor_norm(anchors),
            self.cross_anchor_norm(anchors),
        )
        queries = queries + attended
        return queries + self.mlp(self.mlp_norm(queries))


class GroupConditionedGaussianAdapter(nn.Module):
    """Shared-query group-conditioned anchor decoder used by GC2.

    Frozen TokenGS tokens and their provisional Gaussian centers form the
    scene anchors. One set of object queries produces both the token-to-group
    assignment supervised by the instance objective and the object context
    injected before the final Gaussian activation. The zero-initialized
    residual preserves the pretrained reconstruction exactly at init.
    """

    def __init__(
        self,
        token_dim: int = 1024,
        num_groups: int = 128,
        condition_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        residual_scale: float = 1.0,
        num_gaussians_per_token: int = 64,
        assignment_temperature: float = 10.0,
        per_gaussian_assignment: bool = False,
        per_gaussian_opacity_scale: float = 0.05,
        image_aligned_anchors: bool = False,
        image_feature_dim: int = 64,
        image_upsample: int = 2,
        image_multiscale: bool = True,
        image_scale: float = 1.0,
    ):
        super().__init__()
        if condition_dim % num_heads != 0:
            raise ValueError(
                f"condition_dim={condition_dim} must be divisible by "
                f"num_heads={num_heads}"
            )
        self.token_dim = int(token_dim)
        self.num_groups = int(num_groups)
        self.condition_dim = int(condition_dim)
        self.residual_scale = float(residual_scale)
        self.num_gaussians_per_token = int(num_gaussians_per_token)
        self.per_gaussian_assignment = bool(per_gaussian_assignment)
        self.per_gaussian_opacity_scale = float(per_gaussian_opacity_scale)
        self.image_aligned_anchors = bool(image_aligned_anchors)
        self.image_feature_dim = int(image_feature_dim)
        self.image_upsample = int(image_upsample)
        self.image_multiscale = bool(image_multiscale)
        self.image_scale = float(image_scale)
        self.token_in = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.condition_dim),
        )
        self.position_in = nn.Sequential(
            nn.Linear(3, self.condition_dim),
            nn.GELU(),
            nn.Linear(self.condition_dim, self.condition_dim),
        )
        if self.image_aligned_anchors:
            encoder_dim = self.token_dim * (2 if self.image_multiscale else 1)
            self.image_dense_net = nn.Sequential(
                nn.LayerNorm(encoder_dim),
                nn.Linear(encoder_dim, 512),
                nn.GELU(),
                nn.Linear(
                    512,
                    self.image_feature_dim * self.image_upsample**2,
                ),
            )
            self.image_in = nn.Sequential(
                nn.LayerNorm(self.image_feature_dim),
                nn.Linear(self.image_feature_dim, self.condition_dim),
            )
            # Keep the warm-started token/position path dominant initially.
            # The gate can open as rendered instance supervision proves that
            # the projected image evidence is useful.
            self.image_gate_logit = nn.Parameter(torch.tensor(-2.0))
            with torch.no_grad():
                self.image_dense_net[-1].weight.mul_(0.01)
                self.image_dense_net[-1].bias.zero_()
        else:
            self.image_dense_net = None
            self.image_in = None
            self.image_gate_logit = None
        self.group_tokens = nn.Parameter(
            0.02 * torch.randn(self.num_groups, self.condition_dim)
        )
        self.layers = nn.ModuleList(
            [
                _ObjectQueryDecoderLayer(
                    self.condition_dim,
                    num_heads=num_heads,
                    mlp_hidden=self.condition_dim * 4,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.group_norm = nn.LayerNorm(self.condition_dim)
        self.anchor_norm = nn.LayerNorm(self.condition_dim)
        self.group_assignment_proj = nn.Linear(
            self.condition_dim, self.condition_dim, bias=False
        )
        self.anchor_assignment_proj = nn.Linear(
            self.condition_dim, self.condition_dim, bias=False
        )
        self.void_head = nn.Linear(self.condition_dim, 1)
        if self.per_gaussian_assignment:
            self.gaussian_local_in = nn.Sequential(
                nn.LayerNorm(self.condition_dim + 3 + 14),
                nn.Linear(self.condition_dim + 3 + 14, self.condition_dim),
                nn.GELU(),
                nn.Linear(self.condition_dim, self.condition_dim),
            )
            self.gaussian_local_norm = nn.LayerNorm(self.condition_dim)
            self.gaussian_assignment_proj = nn.Linear(
                self.condition_dim, self.condition_dim, bias=False
            )
            self.gaussian_void_head = nn.Linear(self.condition_dim, 1)
            self.gaussian_opacity_head = nn.Sequential(
                nn.LayerNorm(self.condition_dim * 2),
                nn.Linear(self.condition_dim * 2, self.condition_dim),
                nn.GELU(),
                nn.Linear(self.condition_dim, 1),
            )
        else:
            self.gaussian_local_in = None
            self.gaussian_local_norm = None
            self.gaussian_assignment_proj = None
            self.gaussian_void_head = None
            self.gaussian_opacity_head = None
        self.residual = nn.Sequential(
            nn.LayerNorm(self.token_dim + self.condition_dim),
            nn.Linear(self.token_dim + self.condition_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        # Identity at initialization: the old Gaussian decoder is preserved
        # exactly while gradients train the new object-conditioned pathway.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.void_head.weight)
        nn.init.zeros_(self.void_head.bias)
        if self.per_gaussian_assignment:
            nn.init.zeros_(self.gaussian_void_head.weight)
            nn.init.zeros_(self.gaussian_void_head.bias)
            nn.init.zeros_(self.gaussian_opacity_head[-1].weight)
            nn.init.zeros_(self.gaussian_opacity_head[-1].bias)
        self.log_assignment_temperature = nn.Parameter(
            torch.tensor(float(assignment_temperature)).log()
        )

    def forward(
        self,
        token_hidden: torch.Tensor,
        proposal_gaussians: torch.Tensor,
        dense_features: torch.Tensor | None = None,
        source_c2w: torch.Tensor | None = None,
        source_intrinsics: torch.Tensor | None = None,
        image_hw: tuple[int, int] | None = None,
        num_views: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if token_hidden.ndim != 3 or token_hidden.shape[-1] != self.token_dim:
            raise ValueError(
                f"token_hidden must be [B,T,{self.token_dim}], got "
                f"{tuple(token_hidden.shape)}"
            )
        batch_size, token_count, _ = token_hidden.shape
        expected_gaussians = token_count * self.num_gaussians_per_token
        if proposal_gaussians.shape[:2] != (batch_size, expected_gaussians):
            raise ValueError(
                "proposal_gaussians must be [B,T*P,14], got "
                f"{tuple(proposal_gaussians.shape)} for T={token_count}, "
                f"P={self.num_gaussians_per_token}"
            )

        positions = proposal_gaussians[..., :3].float().view(
            batch_size, token_count, self.num_gaussians_per_token, 3
        ).mean(dim=2)
        center = positions.mean(dim=1, keepdim=True)
        centered = positions - center
        scene_scale = centered.square().sum(dim=-1).mean(dim=1, keepdim=True)
        scene_scale = scene_scale.sqrt().clamp_min(1e-3).unsqueeze(-1)
        normalized_positions = centered / scene_scale

        anchor_context = (
            self.token_in(token_hidden.float())
            + self.position_in(normalized_positions)
        )
        self.last_image_anchor_valid_share = None
        self.last_image_anchor_gate = None
        if self.image_aligned_anchors:
            if any(
                value is None
                for value in (
                    dense_features,
                    source_c2w,
                    source_intrinsics,
                    image_hw,
                    num_views,
                )
            ):
                raise ValueError(
                    "image-aligned anchors require dense_features, source "
                    "cameras, intrinsics, image_hw and num_views"
                )
            batch, seq_len, _ = dense_features.shape
            patches_per_view = seq_len // int(num_views)
            spatial = int(round(patches_per_view**0.5))
            if spatial * spatial != patches_per_view:
                raise ValueError(
                    "dense patch features are not square per view: "
                    f"{patches_per_view} patches/view"
                )
            image_features = self.image_dense_net(dense_features.float())
            image_features = image_features.view(
                batch,
                int(num_views),
                spatial,
                spatial,
                self.image_upsample,
                self.image_upsample,
                self.image_feature_dim,
            )
            image_features = image_features.permute(0, 1, 6, 2, 4, 3, 5)
            image_features = image_features.reshape(
                batch,
                int(num_views),
                self.image_feature_dim,
                spatial * self.image_upsample,
                spatial * self.image_upsample,
            )
            fused_image, has_source = _project_dense_features(
                positions,
                image_features,
                source_c2w,
                source_intrinsics,
                image_hw,
            )
            gate = torch.sigmoid(self.image_gate_logit) * self.image_scale
            anchor_context = anchor_context + gate * self.image_in(fused_image)
            self.last_image_anchor_valid_share = has_source.float().mean()
            self.last_image_anchor_gate = gate
        anchor_context = self.anchor_norm(anchor_context)
        groups = self.group_tokens.unsqueeze(0).expand(
            token_hidden.shape[0], -1, -1
        )
        for layer in self.layers:
            groups = layer(groups, anchor_context)
        groups = self.group_norm(groups)

        anchor_assignment = F.normalize(
            self.anchor_assignment_proj(anchor_context), dim=-1
        )
        group_assignment = F.normalize(
            self.group_assignment_proj(groups), dim=-1
        )
        temperature = self.log_assignment_temperature.exp().clamp(1.0, 100.0)
        group_logits = temperature * torch.einsum(
            "btd,bgd->btg", anchor_assignment, group_assignment
        )
        void_logits = self.void_head(anchor_context)
        logits = torch.cat([group_logits, void_logits], dim=-1)
        probabilities = F.softmax(logits.float(), dim=-1)
        self.last_token_probabilities = probabilities
        self.last_gaussian_probabilities = None
        self.last_gaussian_logits = None
        self.last_gaussian_opacity_delta = None
        if self.per_gaussian_assignment:
            local_positions = proposal_gaussians[..., :3].float().view(
                batch_size, token_count, self.num_gaussians_per_token, 3
            )
            local_positions = (
                local_positions - positions.unsqueeze(2)
            ) / scene_scale.unsqueeze(2)
            local_positions = local_positions.reshape(
                batch_size, expected_gaussians, 3
            )
            anchor_context_per_gaussian = anchor_context.repeat_interleave(
                self.num_gaussians_per_token, dim=1
            )
            local_input = torch.cat(
                [
                    anchor_context_per_gaussian,
                    local_positions,
                    proposal_gaussians[..., :14].float(),
                ],
                dim=-1,
            )
            local_features = self.gaussian_local_norm(
                self.gaussian_local_in(local_input)
            )
            gaussian_assignment = F.normalize(
                self.gaussian_assignment_proj(local_features), dim=-1
            )
            gaussian_group_logits = temperature * torch.einsum(
                "bnd,bgd->bng", gaussian_assignment, group_assignment
            )
            gaussian_void_logits = self.gaussian_void_head(local_features)
            gaussian_logits = torch.cat(
                [gaussian_group_logits, gaussian_void_logits], dim=-1
            )
            gaussian_probabilities = F.softmax(
                gaussian_logits.float(), dim=-1
            )
            object_context_per_gaussian = torch.einsum(
                "bng,bgd->bnd",
                gaussian_probabilities[..., : self.num_groups],
                groups,
            )
            opacity_delta = torch.tanh(
                self.gaussian_opacity_head(
                    torch.cat(
                        [local_features, object_context_per_gaussian], dim=-1
                    )
                )
            ) * self.per_gaussian_opacity_scale
            self.last_gaussian_probabilities = gaussian_probabilities
            self.last_gaussian_logits = gaussian_logits
            self.last_gaussian_opacity_delta = opacity_delta
        assignment = probabilities[..., : self.num_groups]
        object_context = torch.einsum(
            "btg,bgd->btd", assignment, groups
        )
        delta = torch.tanh(
            self.residual(
                torch.cat([token_hidden.float(), object_context], dim=-1)
            )
        )
        conditioned = token_hidden.float() + self.residual_scale * delta
        return (
            conditioned.to(dtype=token_hidden.dtype),
            probabilities,
            logits,
            groups,
            positions,
        )


class GroupConditionedGaussianGenerator(nn.Module):
    """InstOk3D-style group tokens that participate in Gaussian *generation*.

    This is deliberately NOT a post-hoc mask-assignment head. The pipeline:

    1. Anchors: TokenGS decoder token hidden ``h`` plus normalized 3D anchor
       positions (mean of the token's decoded Gaussian centers) form the
       scene anchors.
    2. Group tokens: L learnable embeddings cross-attend to the anchors
       (anchor-grouping transformer) and compete for anchor ownership via a
       slot-competition softmax over L groups + a void channel.
    3. Token rewrite (BEFORE the Gaussian activation head): the per-anchor
       group context ``c_t = sum_l pi_tl * proj(G_l)`` is injected into the
       token hidden ``h_cond = h + residual_scale * tanh(mlp([h, c]))``, so
       the activation head outputs different Gaussian parameters.
    4. Explicit group-driven Gaussian deltas: bounded per-anchor geometry
       offsets ``dxyz_t`` and opacity deltas ``dop_t`` are produced from the
       group context and applied to the decoded Gaussians (consumed by
       ``_finalize_conditioned_gaussians`` in semantic_tokengs_v4).

    The assignment is rendered for supervision as well, but the module's
    primary role is to make instance identity influence Gaussian generation.
    All effect heads are zero-initialized so the pretrained reconstruction is
    preserved exactly at init.
    """

    def __init__(
        self,
        token_dim: int = 1024,
        num_groups: int = 128,
        condition_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        residual_scale: float = 0.3,
        geometry_scale: float = 0.05,
        opacity_scale: float = 0.05,
        num_gaussians_per_token: int = 64,
        assignment_temperature: float = 10.0,
    ):
        super().__init__()
        if condition_dim % num_heads != 0:
            raise ValueError(
                f"condition_dim={condition_dim} must be divisible by "
                f"num_heads={num_heads}"
            )
        self.token_dim = int(token_dim)
        self.num_groups = int(num_groups)
        self.condition_dim = int(condition_dim)
        self.num_gaussians_per_token = int(num_gaussians_per_token)
        self.token_in = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.condition_dim),
        )
        self.position_in = nn.Sequential(
            nn.Linear(3, self.condition_dim),
            nn.GELU(),
            nn.Linear(self.condition_dim, self.condition_dim),
        )
        self.group_tokens = nn.Parameter(
            0.02 * torch.randn(self.num_groups, self.condition_dim)
        )
        self.layers = nn.ModuleList(
            [
                _ObjectQueryDecoderLayer(
                    self.condition_dim,
                    num_heads=num_heads,
                    mlp_hidden=self.condition_dim * 4,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.group_norm = nn.LayerNorm(self.condition_dim)
        self.anchor_norm = nn.LayerNorm(self.condition_dim)
        self.group_assignment_proj = nn.Linear(
            self.condition_dim, self.condition_dim, bias=False
        )
        self.anchor_assignment_proj = nn.Linear(
            self.condition_dim, self.condition_dim, bias=False
        )
        self.void_head = nn.Linear(self.condition_dim, 1)

        # Group context -> token rewrite before the Gaussian activation head.
        self.token_residual = nn.Sequential(
            nn.LayerNorm(self.token_dim + self.condition_dim),
            nn.Linear(self.token_dim + self.condition_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        # Group context -> explicit bounded Gaussian deltas.
        self.geometry_head = nn.Sequential(
            nn.LayerNorm(self.condition_dim),
            nn.Linear(self.condition_dim, self.condition_dim),
            nn.GELU(),
            nn.Linear(self.condition_dim, 3),
        )
        self.opacity_head = nn.Sequential(
            nn.LayerNorm(self.condition_dim),
            nn.Linear(self.condition_dim, self.condition_dim),
            nn.GELU(),
            nn.Linear(self.condition_dim, 1),
        )
        nn.init.zeros_(self.void_head.weight)
        nn.init.zeros_(self.void_head.bias)
        nn.init.zeros_(self.token_residual[-1].weight)
        nn.init.zeros_(self.token_residual[-1].bias)
        nn.init.zeros_(self.geometry_head[-1].weight)
        nn.init.zeros_(self.geometry_head[-1].bias)
        nn.init.zeros_(self.opacity_head[-1].weight)
        nn.init.zeros_(self.opacity_head[-1].bias)
        self.log_residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale)).log()
        )
        self.log_geometry_scale = nn.Parameter(
            torch.tensor(float(geometry_scale)).log()
        )
        self.log_opacity_scale = nn.Parameter(
            torch.tensor(float(opacity_scale)).log()
        )
        self.log_assignment_temperature = nn.Parameter(
            torch.tensor(float(assignment_temperature)).log()
        )
        self.last_token_probabilities = None
        self.last_geometry_offset = None
        self.last_opacity_delta = None

    def forward(
        self,
        token_hidden: torch.Tensor,
        proposal_gaussians: torch.Tensor,
        dense_features: torch.Tensor | None = None,
        source_c2w: torch.Tensor | None = None,
        source_intrinsics: torch.Tensor | None = None,
        image_hw: tuple[int, int] | None = None,
        num_views: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        del dense_features, source_c2w, source_intrinsics, image_hw, num_views
        if token_hidden.ndim != 3 or token_hidden.shape[-1] != self.token_dim:
            raise ValueError(
                f"token_hidden must be [B,T,{self.token_dim}], got "
                f"{tuple(token_hidden.shape)}"
            )
        batch_size, token_count, _ = token_hidden.shape
        expected_gaussians = token_count * self.num_gaussians_per_token
        if proposal_gaussians.shape[:2] != (batch_size, expected_gaussians):
            raise ValueError(
                f"proposal_gaussians must be [B,N,14], got "
                f"{tuple(proposal_gaussians.shape)}"
            )

        positions = proposal_gaussians[..., :3].float().view(
            batch_size, token_count, self.num_gaussians_per_token, 3
        ).mean(dim=2)  # [B,T,3]
        center = positions.mean(dim=1, keepdim=True)
        centered = positions - center
        scene_scale = centered.square().sum(dim=-1).mean(dim=1, keepdim=True)
        scene_scale = scene_scale.sqrt().clamp_min(1e-3).unsqueeze(-1)
        normalized_positions = centered / scene_scale

        anchor_context = (
            self.token_in(token_hidden.float())
            + self.position_in(normalized_positions)
        )
        anchor_context = self.anchor_norm(anchor_context)
        groups = self.group_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            groups = layer(groups, anchor_context)
        groups = self.group_norm(groups)

        anchor_assignment = F.normalize(
            self.anchor_assignment_proj(anchor_context), dim=-1
        )
        group_assignment = F.normalize(
            self.group_assignment_proj(groups), dim=-1
        )
        temperature = self.log_assignment_temperature.exp().clamp(1.0, 100.0)
        group_logits = temperature * torch.einsum(
            "btd,bgd->btg", anchor_assignment, group_assignment
        )
        void_logits = self.void_head(anchor_context)
        logits = torch.cat([group_logits, void_logits], dim=-1)
        probabilities = F.softmax(logits.float(), dim=-1)

        # Per-anchor group context (group tokens -> token hidden, BEFORE the
        # Gaussian activation head).
        group_context = torch.einsum(
            "btg,bgd->btd",
            probabilities[..., : self.num_groups].float(),
            groups,
        )  # [B,T,condition_dim]
        token_rewrite = torch.tanh(
            self.token_residual(
                torch.cat([token_hidden.float(), group_context], dim=-1)
            )
        )
        residual_scale = self.log_residual_scale.exp().clamp(0.0, 2.0)
        conditioned = token_hidden.float() + residual_scale * token_rewrite

        # Explicit group-driven Gaussian deltas (bounded).
        geometry_scale = self.log_geometry_scale.exp().clamp(0.0, 1.0)
        opacity_scale = self.log_opacity_scale.exp().clamp(0.0, 1.0)
        geometry_offset = geometry_scale * torch.tanh(
            self.geometry_head(group_context)
        )  # [B,T,3]
        opacity_delta = opacity_scale * torch.tanh(
            self.opacity_head(group_context)
        )  # [B,T,1]
        self.last_token_probabilities = probabilities
        self.last_geometry_offset = geometry_offset
        self.last_opacity_delta = opacity_delta
        return (
            conditioned.to(dtype=token_hidden.dtype),
            probabilities,
            logits,
            groups,
            positions,
        )


class IndependentInstanceBranch(nn.Module):
    """Fully independent instance-structured branch (Experiment C).

    The frozen TokenGS reconstruction branch (encoder + decoder + activation
    head) is untouched: its hidden states and Gaussians are consumed only as
    detached inputs. This branch builds its OWN representation:

    1. Scene-adaptive 3D anchors: frozen token hidden + normalized 3D anchor
       positions (mean of the token's frozen Gaussian centers).
    2. 100 learnable group tokens cross-attend to the anchors and compete for
       anchor ownership via slot-competition softmax (L groups + void).
    3. Instance-aware latent per anchor (anchor context + assigned group
       context), decoded by an INDEPENDENT Gaussian decoder into its own
       Gaussians. The decoder predicts bounded DELTAS around the frozen
       TokenGS Gaussian parameters (position/scale/rotation/opacity/RGB), so
       at init the branch reproduces the frozen geometry exactly, then learns
       independently.
    4. Supervision: the exact existing recipe (rendered Hungarian BCE+Dice
       + CE/void/unmatched + 3D anchor-level loss) PLUS a branch-only RGB
       reconstruction loss on the branch's own Gaussians (the frozen TokenGS
       RGB branch stays untouched).

    Instance loss gradients can only touch this branch's parameters; nothing
    flows back into ``enc_dec_backbone`` / ``patch_*`` / ``activation_head`` /
    ``gs_tokens`` (verified by grad-presence checks).
    """

    def __init__(
        self,
        opt,
        token_dim: int = 1024,
        num_groups: int = 100,
        anchor_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        num_gaussians_per_anchor: int = 64,
        pos_offset_scale: float = 0.2,
        scale_delta_amp: float = 0.5,
        opacity_delta_amp: float = 0.3,
        rgb_delta_amp: float = 0.5,
        rot_delta_amp: float = 0.5,
        gs_z_offset: float = 1.0,
    ):
        super().__init__()
        if anchor_dim % num_heads != 0:
            raise ValueError(
                f"anchor_dim={anchor_dim} must be divisible by num_heads"
            )
        self.opt = opt
        self.token_dim = int(token_dim)
        self.num_groups = int(num_groups)
        self.anchor_dim = int(anchor_dim)
        self.num_gaussians_per_anchor = int(num_gaussians_per_anchor)
        self.pos_offset_scale = float(pos_offset_scale)
        self.scale_delta_amp = float(scale_delta_amp)
        self.opacity_delta_amp = float(opacity_delta_amp)
        self.rgb_delta_amp = float(rgb_delta_amp)
        self.rot_delta_amp = float(rot_delta_amp)
        self.gs_z_offset = float(gs_z_offset)
        self.renderer = GaussianRenderer(opt)

        self.anchor_mlp = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.anchor_dim),
        )
        self.position_mlp = nn.Sequential(
            nn.Linear(3, self.anchor_dim),
            nn.GELU(),
            nn.Linear(self.anchor_dim, self.anchor_dim),
        )
        self.group_tokens = nn.Parameter(
            0.02 * torch.randn(self.num_groups, self.anchor_dim)
        )
        self.group_layers = nn.ModuleList(
            [
                _ObjectQueryDecoderLayer(
                    self.anchor_dim,
                    num_heads=num_heads,
                    mlp_hidden=self.anchor_dim * 4,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.group_norm = nn.LayerNorm(self.anchor_dim)
        self.anchor_norm = nn.LayerNorm(self.anchor_dim)
        self.anchor_assignment_proj = nn.Linear(
            self.anchor_dim, self.anchor_dim, bias=False
        )
        self.group_assignment_proj = nn.Linear(
            self.anchor_dim, self.anchor_dim, bias=False
        )
        self.void_head = nn.Linear(self.anchor_dim, 1)
        self.group_context_proj = nn.Linear(
            self.anchor_dim, self.anchor_dim, bias=False
        )
        # Deltas for xyz(3)+opacity(1)+scale(3)+rot(4)+rgb(3) = 14 dims/GS.
        gaussian_dim = self.num_gaussians_per_anchor * 14
        self.instance_decoder = nn.Sequential(
            nn.LayerNorm(self.anchor_dim),
            nn.Linear(self.anchor_dim, self.anchor_dim),
            nn.GELU(),
            nn.Linear(self.anchor_dim, gaussian_dim),
        )
        nn.init.zeros_(self.void_head.weight)
        nn.init.zeros_(self.void_head.bias)
        # Zero-init decoder output: instance Gaussians start exactly at the
        # scene anchors with default scale/opacity (reconstruction unchanged).
        nn.init.zeros_(self.instance_decoder[-1].weight)
        nn.init.zeros_(self.instance_decoder[-1].bias)
        self.log_assignment_temperature = nn.Parameter(
            torch.tensor(10.0).log()
        )
        self.last_instance_gaussians = None
        self.last_group_probs = None
        self.last_unit_assignment = None
        self.last_unit_centers = None

    def _render_probability(
        self,
        gaussians: torch.Tensor,
        gaussian_group_probs: torch.Tensor,
        cam_view: torch.Tensor,
        intrinsics: torch.Tensor,
        opt,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        render = self.renderer.render_feature_channels(
            gaussians,
            gaussian_group_probs,
            cam_view,
            intrinsics=intrinsics,
            opacity_scale=float(
                getattr(opt, "instance_group_render_scale", 1.0)
            ),
        )
        rendered_groups = render["images_pred"]  # [B,V,G,H,W]
        rendered_alpha = render["alphas_pred"]
        rendered_channels = rendered_groups / (rendered_alpha + 1e-5)
        rendered_probs = rendered_channels.float() / rendered_channels.float().sum(
            dim=2, keepdim=True
        ).clamp_min(1e-6)
        rendered_probability = rendered_probs.permute(0, 2, 1, 3, 4).unsqueeze(
            3
        )  # [B,G+1,V,1,H,W]
        return rendered_probability, rendered_alpha

    def forward(
        self,
        token_hidden: torch.Tensor,
        frozen_gaussians: torch.Tensor,
        data: dict,
        model_input,
        opt,
        lambda_eff: float = 1.0,
        training: bool = True,
        dense_features: torch.Tensor | None = None,
    ) -> dict:
        """Run the independent branch and compute the full instance loss."""
        batch_size, token_count, _ = token_hidden.shape
        num_groups = self.num_groups
        ng = self.num_gaussians_per_anchor
        token_hidden = token_hidden.detach().float()
        frozen_gaussians = frozen_gaussians.detach().float()
        num_frozen_per_token = frozen_gaussians.shape[1] // token_count
        if num_frozen_per_token * token_count != frozen_gaussians.shape[1]:
            raise ValueError(
                f"frozen gaussians {frozen_gaussians.shape} not divisible by "
                f"token_count {token_count}"
            )
        if num_frozen_per_token != ng:
            raise ValueError(
                "IndependentInstanceBranch v2 requires "
                f"num_gaussians_per_anchor ({ng}) == frozen GS per token "
                f"({num_frozen_per_token}) so the branch can warm-start from "
                "the frozen geometry"
            )

        # --- scene-adaptive 3D anchors ---
        anchor_pos = frozen_gaussians[..., :3].view(
            batch_size, token_count, num_frozen_per_token, 3
        ).mean(dim=2)  # [B,T,3]
        center = anchor_pos.mean(dim=1, keepdim=True)
        scene_scale = (
            (anchor_pos - center).square().mean(dim=(1, 2), keepdim=True)
            .sqrt()
            .clamp_min(1e-3)
        )
        pos_norm = (anchor_pos - center) / scene_scale
        anchor_context = self.anchor_norm(
            self.anchor_mlp(token_hidden) + self.position_mlp(pos_norm)
        )

        # --- group tokens interact with anchors ---
        groups = self.group_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.group_layers:
            groups = layer(groups, anchor_context)
        groups = self.group_norm(groups)
        anchor_assignment = F.normalize(
            self.anchor_assignment_proj(anchor_context), dim=-1
        )
        group_assignment = F.normalize(
            self.group_assignment_proj(groups), dim=-1
        )
        temperature = self.log_assignment_temperature.exp().clamp(1.0, 100.0)
        group_logits = temperature * torch.einsum(
            "btd,bgd->btg", anchor_assignment, group_assignment
        )
        void_logits = self.void_head(anchor_context)
        logits = torch.cat([group_logits, void_logits], dim=-1)
        probabilities = F.softmax(logits.float(), dim=-1)  # [B,T,G+1]

        # --- instance-aware latent + independent Gaussian decoder ---
        group_context = torch.einsum(
            "btg,bgd->btd",
            probabilities[..., :num_groups].float(),
            self.group_context_proj(groups),
        )
        latent = anchor_context + group_context
        raw = self.instance_decoder(latent).view(
            batch_size, token_count, ng, 14
        )
        frozen_gs = frozen_gaussians.view(
            batch_size, token_count, num_frozen_per_token, 14
        )
        # Warm-started deltas around the frozen Gaussian parameters:
        # zero-init decoder output reproduces the frozen geometry exactly.
        d_pos = torch.tanh(raw[..., :3]) * self.pos_offset_scale
        d_opacity = torch.tanh(raw[..., 3:4]) * self.opacity_delta_amp
        d_scale = torch.tanh(raw[..., 4:7]) * self.scale_delta_amp
        d_rot = torch.tanh(raw[..., 7:11]) * self.rot_delta_amp
        d_rgb = torch.tanh(raw[..., 11:14]) * self.rgb_delta_amp
        means = frozen_gs[..., :3] + d_pos
        opacity = (frozen_gs[..., 3:4] + d_opacity).clamp(0.0, 1.0)
        scale = (frozen_gs[..., 4:7] * torch.exp(d_scale)).clamp_max(
            float(getattr(opt, "gaussian_scale_cap", 0.075))
        )
        rotation = F.normalize(frozen_gs[..., 7:11] + d_rot, dim=-1)
        rgb = (frozen_gs[..., 11:14] + d_rgb).clamp(0.0, 1.0)
        gaussians = torch.cat(
            [means, opacity, scale, rotation, rgb], dim=-1
        ).reshape(batch_size, token_count * ng, 14)
        gaussian_group_probs = probabilities.repeat_interleave(ng, dim=1)
        self.last_instance_gaussians = gaussians.detach()
        self.last_group_probs = probabilities.detach()

        # --- render instance masks with the branch's OWN Gaussians ---
        rendered_probability, rendered_alpha = self._render_probability(
            gaussians,
            gaussian_group_probs,
            model_input.decoder.cam_view,
            model_input.decoder.intrinsics,
            opt,
        )
        outputs = {
            "instance_group_probabilities": probabilities,
            "instance_group_logits": logits,
            "gaussian_group_probabilities": gaussian_group_probs,
            "rendered_instance_group_probability": rendered_probability,
            "rendered_instance_group_alpha": rendered_alpha,
            "loss_instance_group": torch.zeros(
                (), device=rendered_probability.device
            ),
            "loss_instance_branch_rgb": torch.zeros(
                (), device=rendered_probability.device
            ),
        }
        if not (training and "instance_label_output" in data):
            return outputs

        # --- branch-only RGB reconstruction loss (own Gaussians) ---
        if "images_output" in data:
            bg_value = {
                "white": 1.0,
                "black": 0.0,
                "grey": 0.5,
            }.get(str(getattr(opt, "bg_color", "grey")), 0.5)
            bg_color = torch.full(
                (3,), bg_value, device=gaussians.device, dtype=gaussians.dtype
            )
            rgb_render = self.renderer.render(
                gaussians,
                model_input.decoder.cam_view,
                bg_color=bg_color,
                intrinsics=model_input.decoder.intrinsics,
            )
            pred_rgb = rgb_render["images_pred"]
            gt_rgb = data["images_output"].float()
            branch_rgb_mse = (pred_rgb - gt_rgb).square().mean()
            lambda_rgb = float(getattr(opt, "lambda_rgb", 200.0))
            outputs["loss_instance_branch_rgb"] = branch_rgb_mse.detach()
            outputs["loss_instance_branch_rgb_weighted"] = (
                lambda_rgb * branch_rgb_mse
            ).detach()
        else:
            lambda_rgb = float(getattr(opt, "lambda_rgb", 200.0))
            branch_rgb_mse = torch.zeros(
                (), device=rendered_probability.device
            )
        outputs["loss_instance_group"] = (
            outputs["loss_instance_group"] + lambda_rgb * branch_rgb_mse
        )

        # --- supervision: existing Hungarian BCE/Dice + 3D consistency ---
        supervision_probability = rendered_probability
        supervision_labels = data["instance_label_output"].long()
        if bool(
            getattr(
                opt, "instance_group_supervise_input_views", False
            )
        ):
            source_probability, _ = self._render_probability(
                gaussians,
                gaussian_group_probs,
                data["cam_view_input"],
                data["intrinsics_input"],
                opt,
            )
            supervision_probability = torch.cat(
                [source_probability, rendered_probability], dim=2
            )
            supervision_labels = torch.cat(
                [
                    data["instance_label_input"].long(),
                    data["instance_label_output"].long(),
                ],
                dim=1,
            )
        loss, stats = hungarian_instance_group_loss(
            supervision_probability,
            supervision_labels,
            num_groups=num_groups,
            min_instance_pixels=int(
                getattr(opt, "instance_group_min_instance_pixels", 32)
            ),
            dice_weight=float(getattr(opt, "lambda_instance_group_dice", 1.0)),
            mask_weight=float(getattr(opt, "lambda_instance_group_mask", 1.0)),
            void_weight=float(getattr(opt, "lambda_instance_group_void", 0.1)),
            unmatched_weight=float(
                getattr(opt, "lambda_instance_group_unmatched", 0.1)
            ),
            lambda_eff=lambda_eff,
            area_alpha=float(getattr(opt, "instance_group_area_alpha", 0.0)),
            match_area_norm=bool(
                getattr(opt, "instance_group_match_area_norm", False)
            ),
            ce_weight=float(getattr(opt, "lambda_instance_group_ce", 0.0)),
            match_topk=int(getattr(opt, "instance_group_match_topk", 1)),
            secondary_pair_weight=float(
                getattr(opt, "instance_group_secondary_pair_weight", 0.3)
            ),
            usage_entropy_weight=float(
                getattr(opt, "instance_group_usage_entropy", 0.0)
            ),
            use_adaptive_groups=bool(
                getattr(opt, "instance_group_adaptive_count", False)
            ),
            scene_level_matching=bool(
                getattr(opt, "instance_group_scene_level_matching", False)
            ),
        )
        outputs["loss_instance_group"] = loss
        outputs.update(stats)

        lambda_3d = float(getattr(opt, "lambda_instance_group_3d", 0.0))
        if lambda_3d > 0.0 and "instance_label_input" in data:
            cam_views = torch.cat(
                [data["cam_view_input"], data["cam_view"]], dim=1
            )
            intrinsics_all = torch.cat(
                [data["intrinsics_input"], data["intrinsics"]], dim=1
            )
            labels_all = torch.cat(
                [
                    data["instance_label_input"],
                    data["instance_label_output"],
                ],
                dim=1,
            )
            loss_3d, stats_3d = instance_group_3d_loss(
                gaussian_group_probs,
                gaussians,
                cam_views,
                intrinsics_all,
                labels_all.long(),
                num_groups=num_groups,
                image_size=tuple(opt.img_size),
                min_instance_gs=int(
                    getattr(opt, "instance_group_3d_min_gs", 16)
                ),
                ce_weight=float(
                    getattr(opt, "lambda_instance_group_3d_ce", 1.0)
                ),
                dice_weight=float(
                    getattr(opt, "lambda_instance_group_dice", 1.0)
                ),
                mask_weight=float(
                    getattr(opt, "lambda_instance_group_mask", 1.0)
                ),
                void_weight=float(
                    getattr(opt, "lambda_instance_group_void", 0.1)
                ),
                unmatched_weight=float(
                    getattr(opt, "lambda_instance_group_unmatched", 0.1)
                ),
                match_topk=int(
                    getattr(opt, "instance_group_3d_match_topk", 1)
                ),
                secondary_pair_weight=float(
                    getattr(opt, "instance_group_secondary_pair_weight", 0.3)
                ),
                use_adaptive_groups=bool(
                    getattr(opt, "instance_group_adaptive_count", False)
                ),
            )
            outputs["loss_instance_group"] = (
                outputs["loss_instance_group"]
                + lambda_3d * lambda_eff * loss_3d
            )
            outputs.update(stats_3d)
        return outputs

def _supervise_rendered_masks(
    gaussian_group_probs: torch.Tensor,
    gaussians: torch.Tensor,
    rendered_probability: torch.Tensor,
    num_groups: int,
    data: dict,
    opt,
    lambda_eff: float,
) -> dict:
    """Shared instance supervision: Hungarian BCE/Dice + 3D anchor loss."""
    outputs = {
        "loss_instance_group": torch.zeros(
            (), device=rendered_probability.device
        )
    }
    if not (data and "instance_label_output" in data):
        return outputs
    supervision_probability = rendered_probability
    supervision_labels = data["instance_label_output"].long()
    if bool(getattr(opt, "instance_group_supervise_input_views", False)):
        raise NotImplementedError(
            "input-view supervision not supported in this module"
        )
    loss, stats = hungarian_instance_group_loss(
        supervision_probability,
        supervision_labels,
        num_groups=num_groups,
        min_instance_pixels=int(
            getattr(opt, "instance_group_min_instance_pixels", 32)
        ),
        dice_weight=float(getattr(opt, "lambda_instance_group_dice", 1.0)),
        mask_weight=float(getattr(opt, "lambda_instance_group_mask", 1.0)),
        void_weight=float(getattr(opt, "lambda_instance_group_void", 0.1)),
        unmatched_weight=float(
            getattr(opt, "lambda_instance_group_unmatched", 0.1)
        ),
        lambda_eff=lambda_eff,
        area_alpha=float(getattr(opt, "instance_group_area_alpha", 0.0)),
        match_area_norm=bool(
            getattr(opt, "instance_group_match_area_norm", False)
        ),
        ce_weight=float(getattr(opt, "lambda_instance_group_ce", 0.0)),
        match_topk=int(getattr(opt, "instance_group_match_topk", 1)),
        secondary_pair_weight=float(
            getattr(opt, "instance_group_secondary_pair_weight", 0.3)
        ),
        usage_entropy_weight=float(
            getattr(opt, "instance_group_usage_entropy", 0.0)
        ),
        use_adaptive_groups=bool(
            getattr(opt, "instance_group_adaptive_count", False)
        ),
        scene_level_matching=bool(
            getattr(opt, "instance_group_scene_level_matching", False)
        ),
    )
    outputs["loss_instance_group"] = loss
    outputs.update(stats)
    lambda_3d = float(getattr(opt, "lambda_instance_group_3d", 0.0))
    if lambda_3d > 0.0 and "instance_label_input" in data:
        cam_views = torch.cat(
            [data["cam_view_input"], data["cam_view"]], dim=1
        )
        intrinsics_all = torch.cat(
            [data["intrinsics_input"], data["intrinsics"]], dim=1
        )
        labels_all = torch.cat(
            [
                data["instance_label_input"],
                data["instance_label_output"],
            ],
            dim=1,
        )
        loss_3d, stats_3d = instance_group_3d_loss(
            gaussian_group_probs,
            gaussians,
            cam_views,
            intrinsics_all,
            labels_all.long(),
            num_groups=num_groups,
            image_size=tuple(opt.img_size),
            min_instance_gs=int(getattr(opt, "instance_group_3d_min_gs", 16)),
            ce_weight=float(getattr(opt, "lambda_instance_group_3d_ce", 1.0)),
            dice_weight=float(getattr(opt, "lambda_instance_group_dice", 1.0)),
            mask_weight=float(getattr(opt, "lambda_instance_group_mask", 1.0)),
            void_weight=float(getattr(opt, "lambda_instance_group_void", 0.1)),
            unmatched_weight=float(
                getattr(opt, "lambda_instance_group_unmatched", 0.1)
            ),
            match_topk=int(getattr(opt, "instance_group_3d_match_topk", 1)),
            secondary_pair_weight=float(
                getattr(opt, "instance_group_secondary_pair_weight", 0.3)
            ),
            use_adaptive_groups=bool(
                getattr(opt, "instance_group_adaptive_count", False)
            ),
        )
        outputs["loss_instance_group"] = (
            outputs["loss_instance_group"] + lambda_3d * lambda_eff * loss_3d
        )
        outputs.update(stats_3d)
    return outputs


def _fps_indices(positions: torch.Tensor, num: int) -> torch.Tensor:
    """Farthest-point sampling indices over [B,N,3] positions."""
    B, N, _ = positions.shape
    device = positions.device
    idx = torch.zeros(B, num, dtype=torch.long, device=device)
    first = torch.randint(0, N, (B,), device=device)
    idx[:, 0] = first
    dists = torch.full((B, N), float("inf"), device=device)
    for i in range(num):
        chosen = idx[:, i]
        pt = positions.gather(
            1, chosen[:, None, None].expand(B, 1, 3)
        )
        d = (positions - pt).square().sum(-1)
        dists = torch.minimum(dists, d)
        if i + 1 < num:
            idx[:, i + 1] = dists.argmax(-1)
    return idx


def _fps_indices_d(embeddings: torch.Tensor, num: int) -> torch.Tensor:
    """Farthest-point sampling over [B,N,D] L2-normalized embeddings."""
    B, N, D = embeddings.shape
    device = embeddings.device
    idx = torch.zeros(B, num, dtype=torch.long, device=device)
    idx[:, 0] = torch.randint(0, N, (B,), device=device)
    dists = torch.full((B, N), float("inf"), device=device)
    for i in range(num):
        chosen = idx[:, i]
        pt = embeddings.gather(
            1, chosen[:, None, None].expand(B, 1, D)
        )
        d = 1.0 - (embeddings * pt).sum(-1)  # cosine distance
        dists = torch.minimum(dists, d)
        if i + 1 < num:
            idx[:, i + 1] = dists.argmax(-1)
    return idx


class ScenePrototypeGrouping(nn.Module):
    """Scene-specific dynamic instance prototypes / slots.

    No global learnable instance queries: the M prototypes are formed per
    scene from the 8192 local units (unit feature + 3D center) by iterative
    soft k-means-style slot grouping (FPS init -> assign -> update, T iters).
    Each unit is assigned to the M slots + a void channel. The unit feature
    embedding MLP and temperature are learned; prototypes themselves are
    purely scene-derived (weighted means of the scene's own unit embeddings
    and positions). This replaces the global Group Tokens for the final
    instance identity.
    """

    def __init__(
        self,
        feat_dim: int = 128,
        num_slots: int = 100,
        slot_dim: int = 128,
        iterations: int = 3,
        temp: float = 5.0,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.num_slots = int(num_slots)
        self.iterations = int(iterations)
        self.unit_embed = nn.Sequential(
            nn.Linear(int(feat_dim) + 3, int(slot_dim)),
            nn.GELU(),
            nn.Linear(int(slot_dim), int(slot_dim)),
        )
        self.void_head = nn.Linear(int(slot_dim), 1)
        nn.init.zeros_(self.void_head.weight)
        nn.init.zeros_(self.void_head.bias)
        self.log_temp = nn.Parameter(torch.tensor(float(temp)).log())
        self.log_pos_weight = nn.Parameter(
            torch.tensor(float(pos_weight)).log()
        )
        self.last_slot_centers = None

    def forward(
        self,
        unit_feat: torch.Tensor,
        unit_center: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, U, _ = unit_feat.shape
        center = unit_center.float()
        scene_center = center.mean(dim=1, keepdim=True)
        scene_scale = (
            (center - scene_center).square().mean(dim=(1, 2), keepdim=True)
            .sqrt()
            .clamp_min(1e-3)
        )
        pos_norm = (center - scene_center) / scene_scale
        embed = self.unit_embed(
            torch.cat([unit_feat.float(), pos_norm], dim=-1)
        )
        embed_n = F.normalize(embed, dim=-1)
        pos_n = F.normalize(pos_norm, dim=-1)
        idx = _fps_indices(pos_norm, self.num_slots)
        proto_embed = embed_n.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, embed_n.shape[-1])
        )
        proto_pos = pos_n.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
        temp = self.log_temp.exp().clamp(0.5, 50.0)
        pos_w = self.log_pos_weight.exp().clamp(0.0, 10.0)
        void = self.void_head(embed)  # [B,U,1]
        for _ in range(self.iterations):
            sim = temp * torch.bmm(
                embed_n, proto_embed.transpose(-1, -2)
            ) + pos_w * torch.bmm(pos_n, proto_pos.transpose(-1, -2))
            logits = torch.cat([sim, void], dim=-1)
            assign = F.softmax(logits.float(), dim=-1)
            w = assign[..., : self.num_slots]  # [B,U,M]
            denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
            proto_embed = F.normalize(
                torch.bmm(w.transpose(1, 2), embed_n) / denom.transpose(1, 2),
                dim=-1,
            )
            proto_pos = F.normalize(
                torch.bmm(w.transpose(1, 2), pos_n) / denom.transpose(1, 2),
                dim=-1,
            )
        sim = temp * torch.bmm(
            embed_n, proto_embed.transpose(-1, -2)
        ) + pos_w * torch.bmm(pos_n, proto_pos.transpose(-1, -2))
        logits = torch.cat([sim, void], dim=-1)
        pi_unit = F.softmax(logits.float(), dim=-1)  # [B,U,M+1]
        slot_mass = pi_unit[..., : self.num_slots].sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)  # [B,1,M]
        self.last_slot_centers = (
            pos_norm.unsqueeze(-2)
            * pi_unit[..., : self.num_slots].unsqueeze(-1)
        ).sum(dim=1) / slot_mass.transpose(1, 2)
        return pi_unit, logits


class IdentityEncoder(nn.Module):
    """Instance-aware unit embedding (the only trainable module).

    Maps (frozen unit feature, normalized 3D unit center) to a 128-D
    L2-normalized embedding. Trained by a soft InfoNCE over per-unit pseudo
    instance distributions (no fixed instance class per unit). At inference
    the scene units are grouped by deterministic agglomerative clustering on
    [embedding, pos].
    """

    def __init__(self, feat_dim: int = 128, embed_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(int(feat_dim) + 3, int(embed_dim)),
            nn.GELU(),
            nn.Linear(int(embed_dim), int(embed_dim)),
        )

    def forward(
        self, unit_feat: torch.Tensor, unit_pos_norm: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([unit_feat.float(), unit_pos_norm], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class SceneUnitAssignmentHead(nn.Module):
    """Directly learn unit -> scene-specific instance assignment.

    Replaces both the global GroupTokens and the 'embedding -> clustering'
    inference path.  M slots are initialized per scene by farthest-point
    sampling on the unit 3D centers (scene-derived, no shared global query
    memory), refined by iterative soft assignment, and the final
    unit -> slot (+ void) assignment is produced directly.

    Supervision is permutation-invariant and acts directly on the unit
    assignment (not on an intermediate embedding):
      - unit-level Hungarian: match each slot's unit-mass distribution to the
        GT instance unit-mass distributions (built from the 15-view majority
        voted pseudo labels) and apply BCE + Dice on the matched
        distributions; soft distributions handle units that mix instances;
      - void BCE against the unit background share from the pseudo labels;
      - slot-usage entropy so unused slots cannot collapse into one dominant
        instance.

    Unit formation, Gaussian geometry and TokenGS reconstruction are frozen
    inputs only.
    """

    def __init__(
        self,
        feat_dim: int = 192,
        num_slots: int = 100,
        slot_dim: int = 128,
        iterations: int = 3,
        temp: float = 5.0,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.num_slots = int(num_slots)
        self.iterations = int(iterations)
        self.unit_embed = nn.Sequential(
            nn.LayerNorm(int(feat_dim) + 3),
            nn.Linear(int(feat_dim) + 3, int(slot_dim)),
            nn.GELU(),
            nn.Linear(int(slot_dim), int(slot_dim)),
        )
        self.void_head = nn.Linear(int(slot_dim), 1)
        nn.init.zeros_(self.void_head.weight)
        nn.init.zeros_(self.void_head.bias)
        self.log_temp = nn.Parameter(torch.tensor(float(temp)).log())
        self.log_pos_weight = nn.Parameter(
            torch.tensor(float(pos_weight)).log()
        )

    def forward(
        self,
        unit_feat: torch.Tensor,
        unit_center: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pi_unit [B,U,M+1], logits [B,U,M+1])."""
        B, U, _ = unit_feat.shape
        center = unit_center.float()
        scene_center = center.mean(dim=1, keepdim=True)
        scene_scale = (
            (center - scene_center).square().mean(dim=(1, 2), keepdim=True)
            .sqrt()
            .clamp_min(1e-3)
        )
        pos_norm = (center - scene_center) / scene_scale
        embed = self.unit_embed(
            torch.cat([unit_feat.float(), pos_norm], dim=-1)
        )
        embed_n = F.normalize(embed, dim=-1)
        pos_n = F.normalize(pos_norm, dim=-1)
        idx = _fps_indices(pos_norm, self.num_slots)
        proto_embed = embed_n.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, embed_n.shape[-1])
        )
        proto_pos = pos_n.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
        temp = self.log_temp.exp().clamp(0.5, 50.0)
        pos_w = self.log_pos_weight.exp().clamp(0.0, 10.0)
        void = self.void_head(embed)  # [B,U,1]
        for _ in range(self.iterations):
            sim = temp * torch.bmm(
                embed_n, proto_embed.transpose(-1, -2)
            ) + pos_w * torch.bmm(pos_n, proto_pos.transpose(-1, -2))
            logits = torch.cat([sim, void], dim=-1)
            assign = F.softmax(logits.float(), dim=-1)
            w = assign[..., : self.num_slots]  # [B,U,M]
            denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
            proto_embed = F.normalize(
                torch.bmm(w.transpose(1, 2), embed_n) / denom.transpose(1, 2),
                dim=-1,
            )
            proto_pos = F.normalize(
                torch.bmm(w.transpose(1, 2), pos_n) / denom.transpose(1, 2),
                dim=-1,
            )
        sim = temp * torch.bmm(
            embed_n, proto_embed.transpose(-1, -2)
        ) + pos_w * torch.bmm(pos_n, proto_pos.transpose(-1, -2))
        logits = torch.cat([sim, void], dim=-1)
        pi_unit = F.softmax(logits.float(), dim=-1)  # [B,U,M+1]
        return pi_unit, logits

    @staticmethod
    def unit_assignment_loss(
        pi_unit: torch.Tensor,
        p_u: torch.Tensor,
        bg_idx: int,
        void_weight: float = 0.1,
        unmatched_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict]:
        """Permutation-invariant BCE + Dice on unit->instance distributions.

        Args:
            pi_unit: [B, U, M+1] predicted unit->slot assignment.
            p_u: [B, U, m] unit soft GT-instance distribution (0=background).
            bg_idx: column of p_u holding background (or -1).
        Returns (loss, stats).
        """
        from scipy.optimize import linear_sum_assignment

        B, U, m1 = pi_unit.shape
        M = int(m1 - 1)
        pi = pi_unit[..., :M]
        r = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-6)  # [B,U,M]
        fg_ids = [
            i for i in range(p_u.shape[-1]) if i != bg_idx
        ] if bg_idx >= 0 else list(range(p_u.shape[-1]))
        matched_losses: list[torch.Tensor] = []
        agreements: list[float] = []
        for b in range(B):
            qf = p_u[b][:, fg_ids]  # [U,F]
            if qf.shape[1] == 0:
                continue
            qf = qf / qf.sum(dim=0, keepdim=True).clamp_min(1e-6)
            cost = 1.0 - torch.einsum(
                "uf,us->fs",
                F.normalize(qf, dim=0),
                F.normalize(r[b], dim=0),
            )
            cost_np = cost.detach().float().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            for f, s in zip(row_ind.tolist(), col_ind.tolist()):
                qv = qf[:, f].clamp(1e-6, 1.0 - 1e-6)
                rv = r[b][:, s].clamp(1e-6, 1.0 - 1e-6)
                # Manual BCE in float32: F.binary_cross_entropy is rejected
                # by autocast (bf16 training).
                qvf = qv.float()
                rvf = rv.float()
                bce = -(
                    qvf * rvf.log() + (1.0 - qvf) * (1.0 - rvf).log()
                ).mean()
                inter = (qv * rv).sum()
                dice = 1.0 - (2.0 * inter + 1.0) / (
                    qv.sum() + rv.sum() + 1.0
                )
                matched_losses.append(bce + dice)
            # assignment agreement: unit argmax slot matches the Hungarian-
            # matched instance of its dominant GT instance.
            dom_gt = qf.argmax(dim=1)  # [U]
            fg_mass = qf.sum(dim=1)  # [U]
            map_arr = torch.full(
                (qf.shape[1],), -1, dtype=torch.long, device=pi.device
            )
            for f, s in zip(row_ind.tolist(), col_ind.tolist()):
                map_arr[f] = s
            dom_slot = pi[b].argmax(dim=1)
            agree_mask = (map_arr[dom_gt] == dom_slot) & (fg_mass > 0)
            valid_count = (fg_mass > 0).sum().clamp_min(1.0)
            agreements.append(
                float(agree_mask.sum() / valid_count)
            )
        if matched_losses:
            match_loss = torch.stack(matched_losses).mean()
        else:
            match_loss = torch.zeros((), device=pi.device)
        if bg_idx >= 0:
            bg_target = p_u[..., bg_idx].clamp(0.0, 1.0)
        else:
            bg_target = torch.zeros_like(pi_unit[..., 0])
        void_pred = pi_unit[..., M].clamp(1e-6, 1.0 - 1e-6)
        vpf = void_pred.float()
        btf = bg_target.float()
        void_loss = -(
            btf * vpf.log() + (1.0 - btf) * (1.0 - vpf).log()
        ).mean()
        # Unmatched-slot constraint: slots that no GT instance matched
        # (M ~ 100 slots vs ~13 instances) must not accumulate foreground
        # unit mass; penalize their mean assignment so they drift toward
        # empty and get filtered at inference.
        unmatched_mass = torch.zeros((), device=pi.device)
        matched_slots = set()
        for f, s in zip(row_ind.tolist(), col_ind.tolist()):
            matched_slots.add(s)
        if unmatched_weight > 0:
            unmatched = [
                s for s in range(M) if s not in matched_slots
            ]
            if unmatched:
                unmatched_mass = pi[..., unmatched].mean()
        usage = pi.mean(dim=1)  # [B,M]
        entropy = -(
            usage * usage.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        stats = {
            "unit_assignment_match": match_loss.detach(),
            "unit_assignment_void": void_loss.detach(),
            "unit_assignment_unmatched": unmatched_mass.detach(),
            "unit_assignment_agreement": (
                torch.tensor(
                    float(np.mean(agreements)) if agreements else 0.0,
                    device=pi.device,
                )
            ),
            "slot_usage_entropy": entropy.detach(),
        }
        return (
            match_loss
            + void_weight * void_loss
            + unmatched_weight * unmatched_mass,
            stats,
        )


class ScenePrototypeLearner(nn.Module):
    """Learn scene-specific instance prototypes (DPG).

    K prototypes are initialized per scene by farthest-point sampling on the
    unit identity embeddings (K = instance count, GT at train / GT-count
    oracle at eval).  They are refined by cross-attention over the unit
    embeddings -- a soft/confidence-weighted aggregation that does NOT
    bootstrap from hard assignments.  Supervision comes from the GT
    instance prototypes (p_u-weighted unit-embedding means) through a
    Hungarian-aligned cosine pull, so the learned prototypes are pulled
    toward the GT instance centers that the 0.534 oracle uses.

    At inference the units are hard-assigned by cosine nearest prototype
    (same as the oracle), and the masks are rendered through the frozen
    Gaussian geometry.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        num_heads: int = 4,
        layers: int = 2,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.norm = nn.LayerNorm(int(feat_dim))
        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(int(layers)):
            self.attn_layers.append(
                nn.MultiheadAttention(
                    int(feat_dim),
                    num_heads=int(num_heads),
                    batch_first=True,
                )
            )
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(int(feat_dim), int(feat_dim) * 2),
                    nn.GELU(),
                    nn.Linear(int(feat_dim) * 2, int(feat_dim)),
                )
            )

    def forward(
        self, e_u: torch.Tensor, K: int
    ) -> torch.Tensor:
        """Return normalized prototypes [B,K,D]."""
        B, U, D = e_u.shape
        K = int(max(1, min(K, U)))
        idx = _fps_indices_d(e_u, K)
        proto = e_u.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, D)
        )  # [B,K,D]
        for attn, ffn in zip(self.attn_layers, self.ffn_layers):
            attn_out, _ = attn(proto, e_u, e_u)
            proto = proto + attn_out
            proto = proto + ffn(self.norm(proto))
        return F.normalize(proto, dim=-1)


class DynamicInstanceQueryHead(nn.Module):
    """Local Unit -> Dynamic Scene Instance Query -> Direct Mask Prediction.

    Frozen inputs: 8-local-unit features, scene-normalized unit 3D centers,
    DINO unit features.  Pipeline (one-shot feed-forward, no clustering):

      1. PointGroup-style center prior: a small MLP predicts a per-unit 3D
         offset toward its instance center; farthest-point sampling over the
         adjusted centers proposes K scene-specific query anchors.
      2. Query features start from the seed units' (unit + DINO) features
         and are refined by Mask3D-style cross-attention over all units.
      3. Mask prediction: query-unit logits = query x projected-unit dot
         product (+ a learned void bias), propagated to Gaussians through
         the frozen GS->unit assignment, rendered, softmaxed over K+void.

    Supervision (in the enclosing module): existing Hungarian + BCE/Dice +
    void/unmatched + usage entropy + 3D consistency.  Unmatched queries are
    pushed to the void channel (anti slot-collapse / anti fragmentation).
    """

    def __init__(
        self,
        unit_dim: int = 128,
        dino_dim: int = 64,
        query_dim: int = 128,
        hidden: int = 256,
        num_queries: int = 128,
        num_heads: int = 4,
        layers: int = 2,
    ):
        super().__init__()
        self.num_queries = int(num_queries)
        self.query_dim = int(query_dim)
        self.offset_head = nn.Sequential(
            nn.Linear(int(unit_dim) + int(dino_dim) + 3, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 3),
        )
        self.query_proj = nn.Sequential(
            nn.Linear(int(unit_dim) + int(dino_dim), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(query_dim)),
        )
        self.unit_proj = nn.Linear(int(unit_dim), int(query_dim))
        self.norm = nn.LayerNorm(int(query_dim))
        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(int(layers)):
            self.attn_layers.append(
                nn.MultiheadAttention(
                    int(query_dim), num_heads=int(num_heads), batch_first=True
                )
            )
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(int(query_dim), int(query_dim) * 2),
                    nn.GELU(),
                    nn.Linear(int(query_dim) * 2, int(query_dim)),
                )
            )
        self.void_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        unit_feat: torch.Tensor,
        unit_pos: torch.Tensor,
        dino_feat: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (unit_logits [B,K,U], q [B,K,Dq], seed_idx [B,K])."""
        B, U, D = unit_feat.shape
        dd = dino_feat.shape[-1] if dino_feat is not None else 0
        x = torch.cat(
            [unit_feat, dino_feat if dino_feat is not None else torch.zeros(
                B, U, 0, device=unit_feat.device, dtype=unit_feat.dtype
            ), unit_pos],
            dim=-1,
        )
        offset = self.offset_head(x)  # [B,U,3]
        center = unit_pos + offset  # PointGroup-style adjusted centers
        K = min(self.num_queries, U)
        idx = _fps_indices(center, K)  # [B,K]
        seed = torch.cat(
            [
                unit_feat,
                dino_feat if dino_feat is not None else torch.zeros(
                    B, U, 0, device=unit_feat.device, dtype=unit_feat.dtype
                ),
            ],
            dim=-1,
        ).gather(1, idx.unsqueeze(-1).expand(B, K, D + dd))
        q = self.query_proj(seed)  # [B,K,Dq]
        kv = self.unit_proj(unit_feat)  # [B,U,Dq]
        for attn, ffn in zip(self.attn_layers, self.ffn_layers):
            attn_out, _ = attn(q, kv, kv)
            q = q + attn_out
            q = q + ffn(self.norm(q))
        unit_logits = (
            torch.einsum("bkd,bud->bku", q, kv) + self.void_bias
        )  # [B,K,U]
        return unit_logits, q, idx

    def query_diversity_loss(
        self, q: torch.Tensor, margin: float = 0.1
    ) -> torch.Tensor:
        """Push the K query features apart (anti set-collapse)."""
        B, K, _ = q.shape
        if K <= 1:
            return torch.zeros((), device=q.device)
        qn = F.normalize(q, dim=-1)
        sim = qn @ qn.transpose(-1, -2)  # [B,K,K]
        eye = torch.eye(K, dtype=torch.bool, device=q.device)
        off = ~eye
        return F.relu(
            sim.masked_fill(eye, -1e9) - margin
        ).masked_select(off.unsqueeze(0)).mean()


class InstanceCenterOffsetHead(nn.Module):
    """PointGroup-style per-unit 3D center-offset prediction.

    For every 8-local-unit, predict the 3D offset from the unit center
    toward its instance center.  At inference the offset-adjusted unit
    centers are tight per instance, so a simple center clustering (DBSCAN /
    agglomerative on the 3D centers) recovers the instances -- replacing the
    threshold-sensitive high-dim embedding clustering whose per-scene
    oracle-eps ceiling is only 0.369.
    """

    def __init__(self, unit_dim: int = 128, dino_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(unit_dim) + int(dino_dim) + 3, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 3),
        )
        # zero-init -> predicted center starts at the unit center
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        unit_feat: torch.Tensor,
        dino_feat: torch.Tensor | None,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        dd = dino_feat.shape[-1] if dino_feat is not None else 0
        x = torch.cat(
            [
                unit_feat,
                dino_feat
                if dino_feat is not None
                else torch.zeros(
                    unit_feat.shape[0],
                    unit_feat.shape[1],
                    0,
                    device=unit_feat.device,
                    dtype=unit_feat.dtype,
                ),
                pos,
            ],
            dim=-1,
        )
        return self.net(x.float())


class DirectGSInstanceHead(nn.Module):
    """InstanceSplat-style per-GS compact instance embedding (no unit
    clustering).

    A small MLP maps (token hidden expanded per GS + relative position +
    scale + opacity) to a compact D-dim instance embedding attached to each
    frozen Gaussian.  The embedding is rendered (alpha compositing) into the
    target views and supervised directly by the 2D GT instance masks:
      - pull: rendered pixels pulled toward their instance prototype;
      - push: prototype-level hinge between different instances (no per-pixel
        push);
      - cross: same instance's prototypes aligned across views (uses the
        3D-consistent instance maps from the 15-view per-GS labels).

    At inference the rendered 8-D embedding is clustered per view (k-means,
    k = GT-count oracle in this first version) into a per-pixel instance
    probability map.  No unit-level grouping anywhere.
    """

    def __init__(
        self,
        token_dim: int = 1024,
        embed_dim: int = 8,
        hidden: int = 128,
        dino_dim: int = 768,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.output_scale = nn.Parameter(torch.tensor(5.0))
        self.mlp = nn.Sequential(
            nn.Linear(int(token_dim) + 7 + int(dino_dim), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(embed_dim)),
        )

    def forward(
        self,
        token_hidden: torch.Tensor,
        gaussians: torch.Tensor,
        means: torch.Tensor,
        dino_gs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-GS instance embedding [B,N,D] on the unit sphere scaled
        by ``output_scale`` (L2-normalized direction, so alpha compositing
        preserves instance direction instead of collapsing toward zero)."""
        batch_size, token_count, _ = token_hidden.shape
        p = gaussians.shape[1] // token_count
        scene_center = means.mean(dim=(1, 2), keepdim=True)
        scene_scale = (
            (means - scene_center).square().mean(dim=(1, 2, 3), keepdim=True)
            .sqrt()
            .clamp_min(1e-3)
        )
        anchor = means.mean(dim=2, keepdim=True)
        rel_pos = (means - anchor) / scene_scale  # [B,T,P,3]
        log_scale = gaussians[..., 4:7].log().view(
            batch_size, token_count, p, 3
        )
        opacity = gaussians[..., 3:4].view(batch_size, token_count, p, 1)
        h_expand = token_hidden.unsqueeze(2).expand(
            batch_size, token_count, p, -1
        )
        x = torch.cat([h_expand, rel_pos, log_scale, opacity], dim=-1)
        if dino_gs is not None:
            x = torch.cat(
                [x, dino_gs.reshape(batch_size, token_count, p, -1)],
                dim=-1,
            )
        emb = self.mlp(x.float()).reshape(batch_size, token_count * p, -1)
        return self.output_scale * F.normalize(emb, dim=-1)


class UnitImageFeature(nn.Module):
    """Trainable per-unit image-evidence branch.

    Maps frozen multi-view encoder patch features (projected onto GS centers
    and aggregated per unit by the frozen GS->unit assignment) into a
    low-dim unit-level image feature, concatenated with the shaped unit
    feature for the instance embedding. Adds image-level instance evidence
    beyond the reconstruction-oriented token hidden.
    """

    def __init__(self, dense_dim: int = 1024, out_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(int(dense_dim)),
            nn.Linear(int(dense_dim), int(out_dim) * 4),
            nn.GELU(),
            nn.Linear(int(out_dim) * 4, int(out_dim)),
        )

    def forward(self, gs_image_feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(gs_image_feat.float())


class SceneInstanceConditioning(nn.Module):
    """Direction B0: scene-conditioned instance queries for unit formation.

    The M queries are initialized deterministically in 3D: FPS over the
    per-token anchor centers selects spatial seeds (coverage only -- FPS is
    NOT an instance prototype), each seed query is content-initialized from
    the token descriptor at that position plus a positional embedding, and
    refined by ``num_layers`` DETR-style cross-attention layers over ALL
    token descriptors.  The resulting scene-conditioned queries are consumed
    by the (zero-gated) unit-query readout in ``TokenLocalUnitGrouping`` so
    instance-level context enters unit formation while step 0 stays exactly
    equal to the non-conditioned baseline.
    """

    def __init__(
        self,
        desc_dim: int = 256,
        num_queries: int = 128,
        dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError(
                f"SceneInstanceConditioning requires dim={dim} divisible "
                f"by num_heads={num_heads}"
            )
        self.num_queries = int(num_queries)
        self.dim = int(dim)
        self.seed_ctx = nn.Linear(int(desc_dim), int(dim))
        self.seed_pos = nn.Sequential(
            nn.Linear(3, int(dim)),
            nn.GELU(),
            nn.Linear(int(dim), int(dim)),
        )
        self.layers = nn.ModuleList(
            [
                _ObjectQueryDecoderLayer(
                    int(dim),
                    num_heads=int(num_heads),
                    mlp_hidden=int(dim) * 4,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(
        self,
        token_desc: torch.Tensor,
        token_pos_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (queries, seed_positions, seed_indices)."""
        batch_size, token_count, _ = token_desc.shape
        device = token_desc.device
        num = min(self.num_queries, token_count)
        # Deterministic 3D-FPS over the normalized token anchor positions.
        # The first seed is index 0 and the rest are farthest points, so the
        # same scene always produces the same spatial coverage.
        with torch.no_grad():
            pos = token_pos_norm.detach().float()
            idx = torch.zeros(
                batch_size, num, dtype=torch.long, device=device
            )
            dists = torch.full(
                (batch_size, token_count), float("inf"), device=device
            )
            for i in range(num):
                if i == 0:
                    chosen = torch.zeros(
                        batch_size, dtype=torch.long, device=device
                    )
                else:
                    chosen = dists.argmax(dim=-1)
                idx[:, i] = chosen
                pt = pos.gather(
                    1, chosen[:, None, None].expand(batch_size, 1, 3)
                )
                d = (pos - pt).square().sum(-1)
                dists = torch.minimum(dists, d)
        gather_idx = idx.unsqueeze(-1).expand(batch_size, num, token_desc.shape[-1])
        seed_desc = token_desc.gather(1, gather_idx)
        seed_pos = pos.gather(1, idx.unsqueeze(-1).expand(batch_size, num, 3))
        queries = self.seed_ctx(seed_desc) + self.seed_pos(seed_pos)
        for layer in self.layers:
            queries = layer(queries, token_desc)
        return queries, seed_pos, idx


class TokenLocalUnitGrouping(nn.Module):
    """Token -> local spatial units -> Group (Experiment D).

    Frozen TokenGS encoder/decoder/geometry are inputs only. Each 64-GS token
    learns K=8 3D-local units (soft k-means over per-GS features built from
    the token hidden + relative 3D position + scale + opacity). Units get
    their own feature and 3D center, then 100 learnable group tokens
    cross-attend to all units (T*K) and produce a unit->group assignment with
    a void channel. Every GS inherits its unit's group; instance masks are
    rendered through the ORIGINAL frozen Gaussian geometry.

    Only unit formation (GS-feature MLP + unit queries), the group tokens and
    the assignment heads train; nothing flows back into the reconstruction.
    Supervision: Hungarian BCE/Dice + 3D anchor-level loss (+ a small unit
    entropy term so the soft assignment does not collapse to one unit).
    """

    def __init__(
        self,
        opt,
        token_dim: int = 1024,
        units_per_token: int = 8,
        gs_feat_dim: int = 128,
        unit_layers: int = 2,
        num_groups: int = 100,
        group_anchor_dim: int = 256,
        num_heads: int = 8,
        group_layers: int = 2,
        unit_temp: float = 5.0,
        assignment_temperature: float = 10.0,
        unit_entropy_weight: float = 0.05,
        unit_compactness_weight: float = 0.1,
        unit_purity: bool = False,
        unit_purity_weight: float = 0.1,
        gaussians_per_token: int = 64,
        gs_refine: bool = False,
        gs_refine_scale: float = 0.1,
        gs_refine_dim: int = 64,
        scene_prototypes: bool = False,
        num_slots: int = 100,
        slot_dim: int = 128,
        slot_iterations: int = 3,
        slot_temp: float = 5.0,
        unit_embedding: bool = False,
        unit_encoder: bool = False,
        embed_dim: int = 128,
        embed_temp: float = 0.1,
        embed_loss: float = 1.0,
        embed_loss_mode: str = "info_nce",
        embed_sample: int = 0,
        embed_proto_temp: float = 0.1,
        embed_margin: float = 0.2,
        embed_margin_weight: float = 1.0,
        embed_center_push_margin: float = 0.2,
        embed_center_push_weight: float = 0.1,
        scene_assignment: bool = False,
        scene_slots: int = 100,
        scene_slot_iters: int = 3,
        scene_slot_temp: float = 5.0,
        scene_slot_pos_weight: float = 1.0,
        scene_unit_loss_weight: float = 1.0,
        scene_slot_entropy: float = 0.05,
        scene_slot_void: float = 0.1,
        scene_slot_unmatched: float = 0.1,
        scene_slot_min_mass: float = 1.0,
        dpg: bool = False,
        dpg_proto_dim: int = 256,
        dpg_heads: int = 4,
        dpg_layers: int = 2,
        dpg_proto_weight: float = 1.0,
        render_space: bool = False,
        render_space_pull: float = 1.0,
        render_space_push: float = 0.5,
        render_space_cross: float = 1.0,
        render_space_margin_push: float = 0.5,
        render_space_margin_cross: float = 0.2,
        render_space_info_nce: float = 0.0,
        render_space_info_temp: float = 0.2,
        render_space_info_samples: int = 32,
        direct_gs: bool = False,
        direct_gs_embed_dim: int = 8,
        direct_gs_hidden: int = 128,
        direct_gs_dino: bool = True,
        direct_gs_pull: float = 1.0,
        direct_gs_push: float = 2.0,
        direct_gs_cross: float = 1.0,
        direct_gs_margin_push: float = 1.0,
        direct_gs_margin_cross: float = 0.2,
        direct_gs_info_nce: float = 1.0,
        direct_gs_info_temp: float = 0.1,
        direct_gs_info_samples: int = 32,
        backprop_token: bool = False,
        unit_image: bool = False,
        unit_image_dim: int = 64,
        dino_unit: bool = False,
        dino_unit_dim: int = 64,
        grounding: bool = False,
        grounding_dim: int = 64,
        grounding_3d_pull: float = 0.0,
        grounding_3d_push: float = 0.0,
        grounding_3d_margin: float = 0.2,
        dynamic_queries: bool = False,
        num_queries: int = 128,
        query_dim: int = 128,
        query_layers: int = 2,
        query_diversity: float = 0.0,
        query_diversity_margin: float = 0.1,
        center_offset: bool = False,
        center_offset_hidden: int = 256,
        pseudo_conf: float = 0.0,
        pseudo_min_views: int = 0,
        pseudo_unit_min_mass: float = 0.0,
        cluster_pos_weight: float = 1.0,
        cluster_eps: float = 1.0,
        void_fg_share: float = 0.5,
        sic_units: bool = False,
        sic_queries: int = 128,
        sic_dim: int = 256,
        sic_heads: int = 4,
        sic_layers: int = 2,
        sic_usage: float = 0.02,
        generative_units: bool = False,
    ):
        super().__init__()
        self.opt = opt
        self.token_dim = int(token_dim)
        self.units_per_token = int(units_per_token)
        self.gs_feat_dim = int(gs_feat_dim)
        self.num_groups = int(num_groups)
        self.group_anchor_dim = int(group_anchor_dim)
        self.gaussians_per_token = int(gaussians_per_token)
        self.unit_entropy_weight = float(unit_entropy_weight)
        self.unit_compactness_weight = float(unit_compactness_weight)
        self.unit_purity = bool(unit_purity)
        self.unit_purity_weight = float(unit_purity_weight)
        self.gs_refine = bool(gs_refine)
        self.scene_prototypes = bool(scene_prototypes)
        self.unit_embedding = bool(unit_embedding)
        self.unit_encoder = bool(unit_encoder)
        self.unit_image = bool(unit_image)
        self.dino_unit = bool(dino_unit)
        self.dino_unit_dim = int(dino_unit_dim)
        self._dino_model = None
        if self.dino_unit:
            self.dino_proj = nn.Sequential(
                nn.Linear(768, int(dino_unit_dim)),
                nn.GELU(),
                nn.Linear(int(dino_unit_dim), int(dino_unit_dim)),
            )
        else:
            self.dino_proj = None
        self.grounding = bool(grounding)
        self.grounding_dim = int(grounding_dim)
        self.grounding_3d_pull = float(grounding_3d_pull)
        self.grounding_3d_push = float(grounding_3d_push)
        self.grounding_3d_margin = float(grounding_3d_margin)
        self.dynamic_queries = bool(dynamic_queries)
        self.query_diversity = float(query_diversity)
        self.query_diversity_margin = float(query_diversity_margin)
        self.center_offset = bool(center_offset)
        if self.center_offset:
            self.center_offset_head = InstanceCenterOffsetHead(
                unit_dim=int(gs_feat_dim),
                dino_dim=int(dino_unit_dim),
                hidden=int(center_offset_hidden),
            )
        else:
            self.center_offset_head = None
        if self.dynamic_queries:
            self.dynamic_query_head = DynamicInstanceQueryHead(
                unit_dim=int(gs_feat_dim),
                dino_dim=int(dino_unit_dim),
                query_dim=int(query_dim),
                num_queries=int(num_queries),
                layers=int(query_layers),
            )
        else:
            self.dynamic_query_head = None
        if self.grounding:
            # Explicit instance feature per local unit, consumed by the
            # rendered-space grounding loss (and by eval clustering).
            self.grounding_net = nn.Sequential(
                nn.Linear(
                    int(gs_feat_dim) + int(dino_unit_dim) + 3,
                    256,
                ),
                nn.GELU(),
                nn.Linear(256, int(grounding_dim)),
            )
        else:
            self.grounding_net = None
        self.pseudo_conf = float(pseudo_conf)
        self.pseudo_min_views = int(pseudo_min_views)
        self.pseudo_unit_min_mass = float(pseudo_unit_min_mass)
        self.embed_temp = float(embed_temp)
        self.embed_loss = float(embed_loss)
        self.embed_loss_mode = str(embed_loss_mode)
        self.embed_sample = int(embed_sample)
        self.embed_proto_temp = float(embed_proto_temp)
        self.embed_margin = float(embed_margin)
        self.embed_margin_weight = float(embed_margin_weight)
        self.embed_center_push_margin = float(embed_center_push_margin)
        self.embed_center_push_weight = float(embed_center_push_weight)
        self.scene_assignment = bool(scene_assignment)
        self.scene_slots = int(scene_slots)
        self.scene_unit_loss_weight = float(scene_unit_loss_weight)
        self.scene_slot_entropy = float(scene_slot_entropy)
        self.scene_slot_void = float(scene_slot_void)
        self.scene_slot_unmatched = float(scene_slot_unmatched)
        self.scene_slot_min_mass = float(scene_slot_min_mass)
        self.dpg = bool(dpg)
        self.dpg_proto_weight = float(dpg_proto_weight)
        self.render_space = bool(render_space)
        self.render_space_pull = float(render_space_pull)
        self.render_space_push = float(render_space_push)
        self.render_space_cross = float(render_space_cross)
        self.render_space_margin_push = float(render_space_margin_push)
        self.render_space_margin_cross = float(render_space_margin_cross)
        self.render_space_info_nce = float(render_space_info_nce)
        self.render_space_info_temp = float(render_space_info_temp)
        self.render_space_info_samples = int(render_space_info_samples)
        self.direct_gs = bool(direct_gs)
        self.direct_gs_dino = bool(direct_gs_dino)
        self.direct_gs_pull = float(direct_gs_pull)
        self.direct_gs_push = float(direct_gs_push)
        self.direct_gs_cross = float(direct_gs_cross)
        self.direct_gs_margin_push = float(direct_gs_margin_push)
        self.direct_gs_margin_cross = float(direct_gs_margin_cross)
        self.direct_gs_info_nce = float(direct_gs_info_nce)
        self.direct_gs_info_temp = float(direct_gs_info_temp)
        self.direct_gs_info_samples = int(direct_gs_info_samples)
        self.backprop_token = bool(backprop_token)
        self.direct_gs_head = (
            DirectGSInstanceHead(
                token_dim=int(token_dim),
                embed_dim=int(direct_gs_embed_dim),
                hidden=int(direct_gs_hidden),
                dino_dim=(768 if direct_gs_dino else 0),
            )
            if direct_gs
            else None
        )
        self.generative_units = bool(generative_units)
        if self.generative_units:
            # Token -> K local units -> per-unit Gaussian decoder.  The
            # decoder outputs a RESIDUAL on the frozen per-GS parameters
            # (zero-initialized, so step 0 reproduces the frozen TokenGS
            # geometry exactly and PSNR starts at the frozen level).  The
            # units are then the shared intermediate that reconstruction
            # (RGB through the decoded GS) and instance supervision
            # (InfoNCE on unit features) jointly optimize.
            gen_in_dim = (
                int(gs_feat_dim) * 2 + 3 + 14
            )  # f_gs + unit-context + rel-center + frozen GS params
            self.unit_gaussian_decoder = nn.Sequential(
                nn.Linear(gen_in_dim, 256),
                nn.GELU(),
                nn.Linear(256, 256),
                nn.GELU(),
                nn.Linear(256, 14),
            )
            nn.init.zeros_(self.unit_gaussian_decoder[-1].weight)
            nn.init.zeros_(self.unit_gaussian_decoder[-1].bias)
        else:
            self.unit_gaussian_decoder = None
        self.dpg_learner = (
            ScenePrototypeLearner(
                feat_dim=int(dpg_proto_dim),
                num_heads=int(dpg_heads),
                layers=int(dpg_layers),
            )
            if dpg
            else None
        )
        self.cluster_pos_weight = float(cluster_pos_weight)
        self.cluster_eps = float(cluster_eps)
        self.void_fg_share = float(void_fg_share)
        self.renderer = GaussianRenderer(opt)
        # Per-GS feature: token hidden + rel position(3) + log scale(3) +
        # opacity(1).
        self.gs_feature_mlp = nn.Sequential(
            nn.LayerNorm(self.token_dim + 7),
            nn.Linear(self.token_dim + 7, self.gs_feat_dim),
            nn.GELU(),
            nn.Linear(self.gs_feat_dim, self.gs_feat_dim),
        )
        self.unit_queries = nn.Parameter(
            0.02 * torch.randn(self.units_per_token, self.gs_feat_dim)
        )
        self.unit_layers = nn.ModuleList(
            [
                _ObjectQueryDecoderLayer(
                    self.gs_feat_dim,
                    num_heads=max(1, min(num_heads, self.gs_feat_dim // 8)),
                    mlp_hidden=self.gs_feat_dim * 4,
                )
                for _ in range(int(unit_layers))
            ]
        )
        self.log_unit_temp = nn.Parameter(torch.tensor(float(unit_temp)).log())
        # Unit -> group.
        self.unit_ctx_mlp = nn.Sequential(
            nn.LayerNorm(self.gs_feat_dim),
            nn.Linear(self.gs_feat_dim, group_anchor_dim),
        )
        self.unit_pos_mlp = nn.Sequential(
            nn.Linear(3, group_anchor_dim),
            nn.GELU(),
            nn.Linear(group_anchor_dim, group_anchor_dim),
        )
        self.unit_norm = nn.LayerNorm(group_anchor_dim)
        self.group_tokens = nn.Parameter(
            0.02 * torch.randn(self.num_groups, group_anchor_dim)
        )
        self.group_layers = nn.ModuleList(
            [
                _ObjectQueryDecoderLayer(
                    group_anchor_dim,
                    num_heads=num_heads,
                    mlp_hidden=group_anchor_dim * 4,
                )
                for _ in range(int(group_layers))
            ]
        )
        self.group_norm = nn.LayerNorm(group_anchor_dim)
        self.unit_assignment_proj = nn.Linear(
            group_anchor_dim, group_anchor_dim, bias=False
        )
        self.group_assignment_proj = nn.Linear(
            group_anchor_dim, group_anchor_dim, bias=False
        )
        self.void_head = nn.Linear(group_anchor_dim, 1)
        nn.init.zeros_(self.void_head.weight)
        nn.init.zeros_(self.void_head.bias)
        self.log_assignment_temperature = nn.Parameter(
            torch.tensor(float(assignment_temperature)).log()
        )
        # Per-GS group-logit refinement on top of the unit proposal
        # (zero-init residual; never touches Gaussian parameters).
        if self.gs_refine:
            self.gs_refine_feat = nn.Sequential(
                nn.LayerNorm(self.gs_feat_dim + 3),
                nn.Linear(self.gs_feat_dim + 3, int(gs_refine_dim)),
                nn.GELU(),
                nn.Linear(int(gs_refine_dim), int(gs_refine_dim)),
            )
            self.gs_refine_group_proj = nn.Linear(
                self.group_anchor_dim, int(gs_refine_dim), bias=False
            )
            nn.init.zeros_(self.gs_refine_feat[-1].weight)
            nn.init.zeros_(self.gs_refine_feat[-1].bias)
            self.log_gs_refine_scale = nn.Parameter(
                torch.tensor(float(gs_refine_scale)).log()
            )
        else:
            self.gs_refine_feat = None
            self.gs_refine_group_proj = None
            self.log_gs_refine_scale = None
        # Scene-specific dynamic prototypes replace the global Group Tokens.
        if self.scene_prototypes:
            self.prototype_grouping = ScenePrototypeGrouping(
                feat_dim=self.gs_feat_dim,
                num_slots=int(num_slots),
                slot_dim=int(slot_dim),
                iterations=int(slot_iterations),
                temp=float(slot_temp),
            )
        else:
            self.prototype_grouping = None
        if self.scene_assignment:
            self.scene_assignment_head = SceneUnitAssignmentHead(
                feat_dim=self.gs_feat_dim
                + (int(unit_image_dim) if self.unit_image else 0)
                + (int(dino_unit_dim) if dino_unit else 0),
                num_slots=int(scene_slots),
                slot_dim=int(slot_dim),
                iterations=int(scene_slot_iters),
                temp=float(scene_slot_temp),
                pos_weight=float(scene_slot_pos_weight),
            )
        else:
            self.scene_assignment_head = None
        if self.unit_embedding and self.unit_encoder:
            self.identity_encoder = IdentityEncoder(
                feat_dim=self.gs_feat_dim, embed_dim=int(embed_dim)
            )
        else:
            self.identity_encoder = None
        if self.unit_image:
            dense_dim = int(self.opt.enc_embed_dim)
            if bool(getattr(self.opt, "instance_group_dense_multiscale", True)):
                dense_dim *= 2
            self.unit_image_net = UnitImageFeature(
                dense_dim=dense_dim, out_dim=int(unit_image_dim)
            )
        else:
            self.unit_image_net = None
        # Direction B0: scene-conditioned instance queries that condition the
        # unit-query initialization through a zero gate.  FPS seeds only
        # provide spatial coverage; the queries are NOT instance prototypes.
        self.sic_units = bool(sic_units)
        self.sic_usage = float(sic_usage)
        self.sic_module = None
        self.sic_gate = None
        self.sic_readout = None
        self.sic_pos_emb = None
        self.sic_h_proj = None
        self.sic_dense_proj = None
        self.sic_dino_proj = None
        self.sic_desc_norm = None
        self.sic_q_proj = None
        if self.sic_units:
            dense_dim_sic = int(self.opt.enc_embed_dim)
            if bool(
                getattr(self.opt, "instance_group_dense_multiscale", True)
            ):
                dense_dim_sic *= 2
            self.sic_module = SceneInstanceConditioning(
                desc_dim=int(sic_dim),
                num_queries=int(sic_queries),
                dim=int(sic_dim),
                num_heads=int(sic_heads),
                num_layers=int(sic_layers),
            )
            self.sic_gate = nn.Parameter(torch.zeros(()))
            self.sic_readout = nn.Sequential(
                nn.LayerNorm(int(sic_dim)),
                nn.Linear(int(sic_dim), self.gs_feat_dim),
            )
            self.sic_pos_emb = nn.Sequential(
                nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 64)
            )
            self.sic_h_proj = nn.Linear(int(token_dim), 64)
            self.sic_dense_proj = nn.Linear(dense_dim_sic, 64)
            self.sic_dino_proj = nn.Linear(768, 64)
            self.sic_desc_norm = nn.LayerNorm(int(sic_dim))
            self.sic_q_proj = nn.Linear(self.gs_feat_dim, int(sic_dim))
        self._gs_dino_cache = None
        self._gs_dense_cache = None
        self._sic_queries = None
        self._sic_seed_pos = None
        self._sic_att = None
        self._sic_usage_att = None
        self.last_instance_gaussians = None
        self.last_group_probs = None
        self.last_unit_embeddings = None
        self.last_unit_pu = None
        self.last_unit_bg_idx = -1

    def reset_parameters_fresh(self) -> None:
        """Re-run the instance-branch initialization with fresh RNG state.

        Used when a guarded-joint run resumes from a reconstruction-only
        checkpoint: only ``absolute_gs_head`` is imported and this method
        guarantees the unit/group-query branch never inherits the (frozen,
        untrained) parameter values saved by the full3 reconstruction run.
        Log-temperature and zero-gate constants are restored to their
        architectural defaults (they are not learned history).
        """
        with torch.no_grad():
            for module in self.modules():
                if module is self:
                    continue
                reset = getattr(module, "reset_parameters", None)
                if reset is None:
                    continue
                try:
                    reset()
                except (TypeError, RuntimeError):
                    pass
            # Raw learnable Parameters created directly in __init__.
            self.unit_queries.copy_(
                0.02 * torch.randn_like(self.unit_queries)
            )
            self.group_tokens.copy_(
                0.02 * torch.randn_like(self.group_tokens)
            )
            # Zero-gated heads must stay at zero (Linear.reset_parameters
            # would randomize them).
            nn.init.zeros_(self.void_head.weight)
            nn.init.zeros_(self.void_head.bias)
            if self.gs_refine_feat is not None:
                nn.init.zeros_(self.gs_refine_feat[-1].weight)
                nn.init.zeros_(self.gs_refine_feat[-1].bias)
            if self.sic_gate is not None:
                nn.init.zeros_(self.sic_gate)
        self._gs_dino_cache = None
        self._gs_dense_cache = None

    def _pseudo_gs_labels(
        self, means: torch.Tensor, data: dict, opt
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-GS pseudo GT instance + confidence (15-view majority vote).

        Returns (gs_gt, conf, n_views): gs_gt [B,N] instance ids (0=bg),
        conf [B,N] = majority-vote fraction among valid views, n_views [B,N]
        = number of valid views the GS projects into. Detached.
        """
        batch_size, n_gs, _ = means.shape
        cam_views_all = torch.cat(
            [data["cam_view_input"], data["cam_view"]], dim=1
        )
        intrinsics_all = torch.cat(
            [data["intrinsics_input"], data["intrinsics"]], dim=1
        )
        labels_all = torch.cat(
            [
                data["instance_label_input"],
                data["instance_label_output"],
            ],
            dim=1,
        )
        gs_gt = torch.zeros(
            (batch_size, n_gs), dtype=torch.long, device=means.device
        )
        gs_conf = torch.zeros(
            (batch_size, n_gs), dtype=torch.float32, device=means.device
        )
        gs_nviews = torch.zeros(
            (batch_size, n_gs), dtype=torch.long, device=means.device
        )
        with torch.no_grad():
            for b in range(batch_size):
                ids, valid = _project_gs_to_views(
                    means[b],
                    cam_views_all[b],
                    intrinsics_all[b],
                    labels_all[b],
                    tuple(opt.img_size),
                )
                gs_gt[b] = _gs_majority_target(ids, valid)
                votes = torch.where(
                    valid, ids, torch.full_like(ids, -1)
                )
                n_valid = valid.sum(dim=0).clamp_min(1)
                best = torch.zeros(n_gs, device=means.device)
                for iid in torch.unique(ids[valid]).tolist():
                    count = (votes == iid).sum(dim=0).float()
                    best = torch.where(count > best, count, best)
                gs_conf[b] = best / n_valid
                gs_nviews[b] = valid.sum(dim=0)
        return gs_gt, gs_conf, gs_nviews

    def _soft_info_nce(
        self, e: torch.Tensor, p_u: torch.Tensor, k: int = 16
    ) -> dict:
        """Distribution-level soft InfoNCE + monitors (permutation-invariant)."""
        B, U, _ = e.shape
        if self.embed_sample > 0 and U > self.embed_sample:
            # Sub-sample units for the pairwise objective when U is large
            # (K=64 -> 65536 units) so the UxU similarity matrix stays
            # feasible; the sampled InfoNCE remains an unbiased estimator.
            idx = torch.randperm(U, device=e.device)[: self.embed_sample]
            e = e[:, idx]
            p_u = p_u[:, idx]
            U = self.embed_sample
        diag = torch.eye(U, dtype=torch.bool, device=e.device)
        sim = torch.bmm(e, e.transpose(-1, -2)) / self.embed_temp  # [B,U,U]
        w = torch.bmm(p_u, p_u.transpose(-1, -2))  # [B,U,U] = <p_u,p_v>
        w = w * (~diag).unsqueeze(0)
        w_norm = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
        sim_nodiag = sim.masked_fill(diag.unsqueeze(0), -1e9)
        log_soft = sim - torch.logsumexp(sim, dim=-1, keepdim=True)
        per_u = -(w_norm * log_soft).sum(-1)  # [B,U]
        valid = (p_u.sum(-1) > 0.1) & (w.sum(-1) > 0)
        loss = (per_u * valid).sum() / valid.sum().clamp_min(1)
        with torch.no_grad():
            # Properly normalized weighted cosine averages (the raw sim is
            # already divided by embed_temp above, so recompute the cosine).
            sim_cos = torch.bmm(e, e.transpose(-1, -2)) * (~diag).unsqueeze(
                0
            )
            same = (w * sim_cos).sum() / w.sum().clamp_min(1)
            diff_mask = (~diag).unsqueeze(0) & (w < 0.1)
            diff = (diff_mask * sim_cos).sum() / diff_mask.sum().clamp_min(1)
            collapse = (
                sim_nodiag * (~diag).unsqueeze(0)
            ).sum() / (U * (U - 1))
            knn = sim_nodiag.topk(k, dim=-1).indices  # [B,U,k]
            knn_w = w.gather(-1, knn)
            knn_agree = (knn_w > 0.5).float().mean(dim=-1)  # [B,U]
            knn_agree = (knn_agree * valid).sum() / valid.sum().clamp_min(1)
        return {
            "loss": loss,
            "same_sim": same,
            "diff_sim": diff,
            "collapse_sim": collapse,
            "knn_agree": knn_agree,
        }

    def _soft_prototype_loss(
        self,
        e: torch.Tensor,
        p_u: torch.Tensor,
        bg_idx: int,
        proto_temp: float = 0.1,
        margin: float = 0.2,
        margin_weight: float = 1.0,
    ) -> dict:
        """Soft prototype/center loss + margin separation.

        Per-scene soft instance centers are built from the unit pseudo-label
        distributions (detached): c_i = sum_u p_u[u,i] e_u / sum_u p_u[u,i],
        which keeps the objective permutation-invariant and lets a unit that
        mixes several instances be pulled toward several centers.

        - Soft prototypical CE: pull each unit toward its instance-mass
          centers and away from other centers (soft labels p_norm; the
          label side is detached so gradients shape the embedding only).
        - Margin: for units whose dominant-instance share is high (purity
          weighting), require (score to own centers) - (best other center)
          >= margin, pushing different instances apart.

        Monitors (same keys as InfoNCE plus margin) are computed on the
        embedding so collapse stays visible.
        """
        B, U, _ = e.shape
        device = e.device
        fg_ids = (
            [i for i in range(p_u.shape[-1]) if i != bg_idx]
            if bg_idx >= 0
            else list(range(p_u.shape[-1]))
        )
        if not fg_ids:
            return self._soft_info_nce(e, p_u)
        n_fg = len(fg_ids)
        p_fg = p_u[..., fg_ids]  # [B,U,F]
        with torch.no_grad():
            w_u = (
                (1.0 - p_u[..., bg_idx]).clamp(0.0, 1.0)
                if bg_idx >= 0
                else torch.ones_like(p_u[..., 0])
            )
            p_norm = p_fg / p_fg.sum(-1, keepdim=True).clamp_min(1e-8)
            e_det = e.detach()
            num = torch.einsum(
                "bud,buf->bfd", e_det * w_u.unsqueeze(-1), p_norm
            )
            den = torch.einsum("bu,buf->bf", w_u, p_norm).clamp_min(1e-6)
            centers = F.normalize(num / den.unsqueeze(-1), dim=-1)
            valid = (w_u > 0.1) & (p_norm.sum(-1) > 0.05)
        cos = torch.bmm(e, centers.transpose(-1, -2))  # [B,U,F]
        logits = cos / float(proto_temp)
        log_soft = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        per_u = -(p_norm * log_soft).sum(-1)  # [B,U]
        denom = (w_u * valid).sum().clamp_min(1.0)
        loss_ce = (w_u * valid * per_u).sum() / denom
        # Margin: only for units with a clear dominant instance (purity 0.5-1
        # maps linearly to weight 0-1) so mixed units are not pushed away
        # from their secondary instance centers.
        top_p = p_norm.max(-1).values
        purity_w = ((top_p - 0.5) * 2.0).clamp(0.0, 1.0) * valid.float()
        other_mask = (p_norm < 0.5).float()
        other_cos = (cos * other_mask).max(-1).values.clamp_min(-1.0)
        own = (p_norm * cos).sum(-1)
        margin_gap = own - other_cos
        margin_loss = (
            purity_w * F.relu(margin - margin_gap)
        ).sum() / denom
        loss = loss_ce + margin_weight * margin_loss
        with torch.no_grad():
            diag = torch.eye(U, dtype=torch.bool, device=device)
            w = torch.bmm(p_u, p_u.transpose(-1, -2)) * (~diag).unsqueeze(0)
            sim = torch.bmm(e, e.transpose(-1, -2))
            same = (w * sim).sum() / w.sum().clamp_min(1)
            diff_mask = (~diag).unsqueeze(0) & (w < 0.1)
            diff = (diff_mask * sim).sum() / diff_mask.sum().clamp_min(1)
            collapse = (
                sim * (~diag).unsqueeze(0)
            ).sum() / (U * (U - 1))
            knn = sim.masked_fill(diag.unsqueeze(0), -1e9).topk(
                16, dim=-1
            ).indices
            knn_w = w.gather(-1, knn)
            knn_agree = (
                (knn_w > 0.5).float().mean(-1) * valid
            ).sum() / valid.sum().clamp_min(1)
            margin_avg = (margin_gap * valid).sum() / valid.sum().clamp_min(1)
        return {
            "loss": loss,
            "same_sim": same,
            "diff_sim": diff,
            "collapse_sim": collapse,
            "knn_agree": knn_agree,
            "margin": margin_avg,
        }

    def _soft_info_nce_center_push(
        self,
        e: torch.Tensor,
        p_u: torch.Tensor,
        bg_idx: int,
        margin: float,
        push_weight: float,
    ) -> dict:
        """Soft InfoNCE (intra-instance pull) + instance-center-level push.

        InstanceSplat-style push: soft instance centers are built from the
        DETACHED pseudo-instance distribution and pushed apart with a cosine
        hinge.  Unlike the failed per-unit margin, no gradient pushes an
        individual unit away from other instances; gradients only flow
        through each unit's own instance center, so intra-instance
        compactness is not directly attacked.

        Returned stats include the InfoNCE monitors plus center-level
        separation statistics (mean pairwise center cosine and the margin
        between a center and its nearest other center).
        """
        stats = self._soft_info_nce(e, p_u)
        B, U, _ = e.shape
        device = e.device
        fg_ids = (
            [i for i in range(p_u.shape[-1]) if i != bg_idx]
            if bg_idx >= 0
            else list(range(p_u.shape[-1]))
        )
        fg_ids = [i for i in fg_ids if p_u[..., i].sum() > 1e-3]
        if len(fg_ids) < 2:
            with torch.no_grad():
                stats["center_cos_mean"] = torch.zeros((), device=device)
                stats["center_margin"] = torch.zeros((), device=device)
            return stats

        p_fg = p_u[..., fg_ids]  # [B,U,F] (pseudo label side)
        with torch.no_grad():
            size_fg = p_fg.sum(dim=1)  # [B,F] unit mass per instance
            valid_c = size_fg > 1e-3
            p_norm = p_fg / p_fg.sum(dim=1, keepdim=True).clamp_min(1e-8)
        # Soft instance center: foreground-mass-weighted average of the unit
        # embeddings (pseudo labels detached, gradients flow into e only).
        centers = torch.einsum("bud,buf->bfd", e, p_norm)
        centers = F.normalize(centers, dim=-1)  # [B,F,D]
        center_sim = torch.bmm(centers, centers.transpose(-1, -2))
        # Zero-mass instances produce NaN centers; replace with a neutral
        # value and exclude them via the validity masks below.
        center_sim = torch.nan_to_num(center_sim, nan=0.0)
        diag = torch.eye(center_sim.shape[-1], dtype=torch.bool, device=device)
        off_sim = center_sim.masked_fill(diag.unsqueeze(0), 0.0)
        # Pairwise cosine hinge on centers.  Weight each pair by the smaller
        # instance mass so tiny/noisy instances do not dominate and large
        # instance pairs contribute proportionally.
        pair_w = torch.bmm(
            size_fg.unsqueeze(-1), size_fg.unsqueeze(1)
        ).masked_fill(diag.unsqueeze(0), 0.0)
        pair_w = pair_w * valid_c.unsqueeze(-1).float() * valid_c.unsqueeze(
            1
        ).float()
        hinge = F.relu(margin - off_sim)
        denom = pair_w.sum().clamp_min(1e-8)
        push = (pair_w * hinge).sum() / denom
        loss = stats["loss"] + push_weight * push
        with torch.no_grad():
            vf = valid_c.float()
            valid_sim = off_sim * vf.unsqueeze(-1) * vf.unsqueeze(1)
            n_pairs = (vf.unsqueeze(-1) * vf.unsqueeze(1) * (~diag)).sum(
                dim=(-1, -2)
            ).clamp_min(1)
            center_cos_mean = (
                valid_sim.sum(dim=(-1, -2)) / n_pairs
            ).mean()
            # Per-center margin: 1 - cos(own center, nearest other center).
            own = off_sim.max(dim=-1).values
            own = own.masked_fill(~valid_c, 1.0)
            if valid_c.any():
                center_margin = (1.0 - own[valid_c]).mean()
            else:
                center_margin = torch.zeros((), device=device)
            stats = dict(stats)
            stats["center_cos_mean"] = center_cos_mean.detach()
            stats["center_margin"] = center_margin.detach()
            stats["loss_center_push"] = push.detach()
        return stats

    def _embedding_stats(
        self, e: torch.Tensor, p_u: torch.Tensor, bg_idx: int
    ) -> dict:
        """Route the identity-embedding objective (InfoNCE or soft proto)."""
        if self.embed_loss_mode == "none":
            # Pure rendered-space grounding mode: no identity/clustering
            # objective on the unit embedding.
            device = e.device
            return {
                "loss": torch.zeros((), device=device),
                "same_sim": torch.zeros((), device=device),
                "diff_sim": torch.zeros((), device=device),
                "collapse_sim": torch.zeros((), device=device),
                "knn_agree": torch.zeros((), device=device),
                "margin": torch.zeros((), device=device),
                "center_cos_mean": torch.zeros((), device=device),
                "center_margin": torch.zeros((), device=device),
            }
        if self.embed_loss_mode == "softproto_margin":
            return self._soft_prototype_loss(
                e,
                p_u,
                bg_idx,
                self.embed_proto_temp,
                self.embed_margin,
                self.embed_margin_weight,
            )
        if self.embed_loss_mode == "info_nce_center_push":
            stats = self._soft_info_nce_center_push(
                e,
                p_u,
                bg_idx,
                self.embed_center_push_margin,
                self.embed_center_push_weight,
            )
            with torch.no_grad():
                stats = dict(stats)
                stats.setdefault("margin", torch.zeros((), device=e.device))
            return stats
        stats = self._soft_info_nce(e, p_u)
        with torch.no_grad():
            stats = dict(stats)
            stats["margin"] = torch.zeros((), device=e.device)
            stats["center_cos_mean"] = torch.zeros((), device=e.device)
            stats["center_margin"] = torch.zeros((), device=e.device)
        return stats

    def _project_gs_labels_zbuffer(
        self,
        means: torch.Tensor,
        gs_gt: torch.Tensor,
        cam_views: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """3D-consistent per-pixel instance map via z-buffer GS splatting.

        ``means`` [N,3], ``gs_gt`` [N] per-GS GT instance ids, ``cam_views``
        [V,4,4] provider convention, ``intrinsics`` [V,4].  Returns [V,H,W]
        long map (0 = background / no GS).  The id of the nearest (min-z) GS
        covering each pixel wins, so the same 3D instance keeps the same id
        across target views (enables cross-view prototype consistency).
        """
        height, width = int(image_size[0]), int(image_size[1])
        w2c = cam_views.transpose(-1, -2).float()
        homo = torch.cat(
            [means.float(), torch.ones_like(means[..., :1])], dim=-1
        )
        cam = torch.einsum("vij,nj->vni", w2c, homo)
        z = cam[..., 2]
        fx, fy, cx, cy = (
            intrinsics[:, 0],
            intrinsics[:, 1],
            intrinsics[:, 2],
            intrinsics[:, 3],
        )
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
        view_count = cam_views.shape[0]
        id_map = torch.zeros(
            (view_count, height, width),
            dtype=torch.long,
            device=means.device,
        )
        for v in range(view_count):
            idx = valid[v].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            x = px_l[v][idx]
            y = py_l[v][idx]
            zv = z[v][idx]
            ids = gs_gt[idx]
            flat = y * width + x
            # z descending -> later scatter writes the nearest (min-z) GS.
            order = torch.argsort(zv, descending=True)
            id_map[v].flatten()[flat[order]] = ids[order]
        return id_map

    def _grounding_3d_prototype_loss(
        self, e: torch.Tensor, p_u: torch.Tensor, bg_idx: int
    ) -> tuple[torch.Tensor, dict]:
        """InstanceSplat-style 3D instance-prototype pull/push.

        Per GT instance (soft distributions from the 15-view projected
        per-GS labels aggregated through the GS->unit assignment), a
        scene-level 3D prototype is the mass-weighted mean of the DETACHED
        unit features of that instance:
          - pull: each unit is pulled toward its instance prototype,
            weighted by its soft instance mass.  This tightens
            same-instance units directly in 3D (the rendered-space pull
            alone is diluted by alpha compositing);
          - push: different-instance prototypes are pushed apart on cosine
            (hinge), preventing all-prototype collapse.
        """
        B, U, D = e.shape
        m = p_u.shape[-1]
        fg_cols = [c for c in range(m) if c != bg_idx]
        pf = p_u[..., fg_cols]  # [B,U,F] foreground soft instance mass
        mass = pf.sum(-1, keepdim=True)
        valid = mass > 0.05
        v = valid.float()
        pf_n = pf / mass.clamp_min(1e-6)
        # Prototypes WITH gradients (the push must be able to move them).
        num = torch.einsum("buf,bud->bfd", pf_n * v, e)
        denom = v.sum(1).clamp_min(1e-6)
        proto = F.normalize(num / denom.unsqueeze(-1), dim=-1)  # [B,F,D]
        # Pull uses DETACHED prototypes (stable k-means-style targets).
        proto_t = proto.detach()
        cos = torch.einsum("bud,bfd->buf", e, proto_t)
        vv = v.squeeze(-1)  # [B,U]
        pull = -((pf_n * cos).sum(-1) * vv).sum() / vv.sum().clamp_min(1)
        # Push uses the non-detached prototypes so gradients flow.
        sim = proto @ proto.transpose(-1, -2)  # [B,F,F]
        n_inst = proto.shape[1]
        device = e.device
        if n_inst > 1:
            eye = torch.eye(n_inst, dtype=torch.bool, device=device)
            push = F.relu(
                self.grounding_3d_margin + sim.masked_fill(eye, -1e9)
            ).masked_select((~eye).unsqueeze(0)).mean()
        else:
            push = torch.zeros((), device=device)
        with torch.no_grad():
            same = (cos * pf_n).sum(-1)  # [B,U] cos to own prototype
            same_cos = (same * vv).sum() / vv.sum().clamp_min(1)
            if n_inst > 1:
                eye = torch.eye(n_inst, dtype=torch.bool, device=device)
                diff_cos = sim.masked_select((~eye).unsqueeze(0)).mean()
            else:
                diff_cos = torch.zeros((), device=device)
        stats = {
            "grounding_3d_pull": pull.detach(),
            "grounding_3d_push": push.detach(),
            "grounding_3d_same_cos": same_cos.detach(),
            "grounding_3d_diff_cos": diff_cos.detach(),
            "grounding_3d_gap": (same_cos - diff_cos).detach(),
            "grounding_3d_instances": torch.tensor(
                n_inst, device=device
            ),
        }
        return (
            self.grounding_3d_pull * pull
            + self.grounding_3d_push * push,
            stats,
        )

    def _rendered_space_embedding_loss(
        self,
        e: torch.Tensor,
        a: torch.Tensor,
        means: torch.Tensor,
        gaussians: torch.Tensor,
        gs_gt: torch.Tensor,
        data: dict,
        opt,
        model_input,
        batch_size: int,
        token_count: int,
        n_gs: int,
        p: int,
    ) -> tuple[torch.Tensor, dict]:
        """InstanceSplat-style rendered-space embedding supervision.

        The unit identity embedding is propagated to the GS through the
        frozen GS->unit assignment and rendered (alpha compositing) into the
        target views.  Per-view instance prototypes are built from the
        3D-consistent GT instance maps (z-buffer splat of the 15-view
        majority per-GS labels), then:
          - pull: rendered pixels pulled toward their instance prototype;
          - push: prototype-level hinge between different instances in the
            same view (no per-unit push);
          - cross: same instance's prototypes aligned across target views.
        """
        D = e.shape[-1]
        if e.ndim != 3:
            raise RuntimeError(
                f"rendered-space: expected e [B,U,D], got {tuple(e.shape)}"
            )
        e_t = e.reshape(
            batch_size, token_count, self.units_per_token, D
        )
        gs_emb = torch.einsum("btpk,btkd->btpd", a, e_t).reshape(
            batch_size, n_gs, D
        )
        render = self.renderer.render_feature_channels(
            gaussians.reshape(batch_size, n_gs, 14),
            gs_emb,
            model_input.decoder.cam_view,
            intrinsics=model_input.decoder.intrinsics,
        )
        rendered = render["images_pred"] / (
            render["alphas_pred"] + 1e-5
        )  # [B,V,D,H,W]
        rendered = F.normalize(rendered, dim=2)
        view_count = rendered.shape[1]
        if "instance_label_output" in data:
            # Direct per-view 2D GT instance masks (preferred): the rendered
            # feature map is supervised by exactly what the instance masks
            # are, without the z-buffer projection approximation.
            id_maps = data["instance_label_output"].long()
        else:
            id_maps = []
            for b in range(batch_size):
                id_maps.append(
                    self._project_gs_labels_zbuffer(
                        means[b].reshape(n_gs, 3),
                        gs_gt[b],
                        model_input.decoder.cam_view[b],
                        model_input.decoder.intrinsics[b],
                        tuple(opt.img_size),
                    )
                )
            id_maps = torch.stack(id_maps)  # [B,V,H,W]

        pull_losses = []
        push_losses = []
        cross_losses = []
        proto_cos = []
        for b in range(batch_size):
            for v in range(view_count):
                mask = id_maps[b, v]
                inst_ids = torch.unique(mask[mask > 0])
                if inst_ids.numel() < 1:
                    continue
                proto_by_id = {}
                for k in inst_ids.tolist():
                    px = rendered[b, v, :, mask == k]  # [D,npx]
                    if px.shape[1] < 2:
                        continue
                    proto = F.normalize(px.mean(dim=1), dim=0)
                    proto_by_id[k] = proto
                    sim = (px.T * proto).sum(-1)
                    pull_losses.append((1.0 - sim).mean())
                protos = list(proto_by_id.values())
                if len(protos) >= 2:
                    for i in range(len(protos)):
                        for j in range(i + 1, len(protos)):
                            d = (protos[i] - protos[j]).norm()
                            push_losses.append(
                                F.relu(self.render_space_margin_push - d)
                            )
                            proto_cos.append((protos[i] * protos[j]).sum())
                for k, proto_v in proto_by_id.items():
                    for v2 in range(view_count):
                        if v2 == v:
                            continue
                        mask2 = id_maps[b, v2]
                        px2 = rendered[b, v2, :, mask2 == k]
                        if px2.shape[1] < 2:
                            continue
                        proto2 = F.normalize(px2.mean(dim=1), dim=0)
                        d = (proto_v - proto2).norm()
                        cross_losses.append(
                            F.relu(d - self.render_space_margin_cross)
                        )
        L_pull = torch.stack(pull_losses).mean() if pull_losses else (
            torch.zeros((), device=e.device)
        )
        L_push = torch.stack(push_losses).mean() if push_losses else (
            torch.zeros((), device=e.device)
        )
        L_cross = torch.stack(cross_losses).mean() if cross_losses else (
            torch.zeros((), device=e.device)
        )
        # Pixel-level InfoNCE on the rendered unit embedding (same-instance
        # rendered pixels = soft positives, different-instance = negatives).
        # This directly supervises pixel-embedding separation, fixing the
        # prototype-mean pull/push failure mode (small mean offset only).
        if self.render_space_info_nce > 0 and "instance_label_output" in data:
            L_info, info_stats = self._rendered_pixel_info_nce(
                F.normalize(rendered, dim=2),
                data["instance_label_output"].long(),
                self.render_space_info_temp,
                self.render_space_info_samples,
            )
        else:
            L_info = torch.zeros((), device=e.device)
            info_stats = {}
        loss = (
            self.render_space_pull * L_pull
            + self.render_space_push * L_push
            + self.render_space_cross * L_cross
            + self.render_space_info_nce * L_info
        )
        with torch.no_grad():
            proto_cos_t = (
                torch.stack(proto_cos).mean() if proto_cos
                else torch.zeros((), device=e.device)
            )
        stats = {
            "render_space_pull": L_pull.detach(),
            "render_space_push": L_push.detach(),
            "render_space_cross": L_cross.detach(),
            "render_space_info_nce": L_info.detach(),
            "instance_proto_cos": proto_cos_t.detach(),
            "instance_proto_margin": (
                self.render_space_margin_push
                - torch.stack(push_losses).mean()
            ).detach() if push_losses else torch.zeros((), device=e.device),
        }
        stats.update(info_stats)
        return loss, stats

    def _rendered_pixel_info_nce(
        self,
        rendered_n: torch.Tensor,
        gt_maps: torch.Tensor,
        info_temp: float,
        samples_per_inst: int,
    ) -> tuple[torch.Tensor, dict]:
        """Pixel-level InfoNCE on the rendered instance embedding.

        For each target view, pixels of the same GT instance are soft
        positives and pixels of different instances are negatives.  This
        directly supervises the pixel-embedding separation (the failure mode
        of the prototype-mean pull/push: it only produced a small mean
        offset while pixel bodies stayed collinear).

        Returns (loss, stats) with pixel-level same/diff cosine monitors.
        """
        losses = []
        same_all = []
        diff_all = []
        B, V, D = rendered_n.shape[:3]
        for b in range(B):
            for v in range(V):
                mask = gt_maps[b, v]  # [H,W]
                emb = rendered_n[b, v].permute(1, 2, 0).reshape(-1, D)
                flat_mask = mask.reshape(-1)
                inst_ids = torch.unique(flat_mask[flat_mask > 0])
                sampled_emb = []
                sampled_lbl = []
                for k in inst_ids.tolist():
                    idx = (flat_mask == k).nonzero(as_tuple=False).squeeze(-1)
                    if idx.numel() < 2:
                        continue
                    if idx.numel() > samples_per_inst:
                        sel = idx[
                            torch.randperm(idx.numel(), device=idx.device)[
                                :samples_per_inst
                            ]
                        ]
                    else:
                        sel = idx
                    sampled_emb.append(emb[sel])
                    sampled_lbl.append(
                        torch.full((sel.numel(),), k, device=emb.device)
                    )
                if len(sampled_emb) < 2:
                    continue
                e = torch.cat(sampled_emb)  # [M,D]
                lbl = torch.cat(sampled_lbl)
                M = e.shape[0]
                if M < 4:
                    continue
                sim = (e @ e.T) / float(info_temp)
                eye = torch.eye(M, dtype=torch.bool, device=e.device)
                same_mask = (lbl[:, None] == lbl[None, :]) & (~eye)
                if not same_mask.any():
                    continue
                logsumexp_all = torch.logsumexp(sim, dim=-1)  # [M]
                pos_logsumexp = torch.logsumexp(
                    sim.masked_fill(~same_mask, -1e9), dim=-1
                )
                nce = logsumexp_all - pos_logsumexp  # [M]
                valid = same_mask.any(dim=-1)
                losses.append(nce[valid].mean())
                with torch.no_grad():
                    cos = e @ e.T
                    same_all.append(cos[same_mask].mean())
                    diff_all.append(
                        cos[(lbl[:, None] != lbl[None, :])].mean()
                    )
        if losses:
            L = torch.stack(losses).mean()
        else:
            L = torch.zeros((), device=rendered_n.device)
        with torch.no_grad():
            same_t = (
                torch.stack(same_all).mean() if same_all
                else torch.zeros((), device=rendered_n.device)
            )
            diff_t = (
                torch.stack(diff_all).mean() if diff_all
                else torch.zeros((), device=rendered_n.device)
            )
        stats = {
            "direct_gs_pixel_same": same_t.detach(),
            "direct_gs_pixel_diff": diff_t.detach(),
            "direct_gs_pixel_gap": (same_t - diff_t).detach(),
        }
        return L, stats

    def _forward_center_offset(
        self,
        a,
        unit_feat,
        unit_center,
        means,
        gaussians,
        data,
        opt,
        training,
        model_input,
        batch_size,
        token_count,
        n_gs,
        p,
        dense_features=None,
    ) -> dict:
        """PointGroup-style per-unit instance-center offset prediction.

        Predicts a 3D offset from each unit center toward its instance
        center.  At inference the offset-adjusted unit centers are tight per
        instance and grouped by a simple 3D center clustering (in the eval
        script), replacing the threshold-sensitive high-dim embedding
        clustering.  Training target: pseudo-GT instance centers from the
        15-view projected per-GS GT labels, aggregated per unit through the
        soft assignment (confidence-weighted, background excluded).
        """
        k = self.units_per_token
        u_count = token_count * k
        D = self.gs_feat_dim
        unit_feat_flat = unit_feat.reshape(batch_size, u_count, D)
        center = unit_center.reshape(batch_size, u_count, 3).float()
        scene_center = center.mean(dim=1, keepdim=True)
        scene_scale = (
            (center - scene_center).square().mean(dim=(1, 2), keepdim=True)
            .sqrt().clamp_min(1e-3)
        )
        pos_norm = (center - scene_center) / scene_scale
        unit_dino = self._dino_unit_features(
            a, means, data, opt, batch_size, token_count, n_gs, p
        ) if self.dino_unit else None
        dino_flat = (
            unit_dino.reshape(batch_size, u_count, -1)
            if unit_dino is not None
            else None
        )
        offset = self.center_offset_head(unit_feat_flat, dino_flat, pos_norm)
        center_pred = pos_norm + offset  # [B,U,3]

        self.last_pos_norm = pos_norm.detach()
        self.last_center_pred = center_pred.detach()
        self.last_unit_embeddings = unit_feat_flat.detach()
        self._aff_buffers = {
            "a": a.detach().float(),
            "pos_norm": pos_norm.detach().float(),
            "means": means.detach().float(),
            "gaussians": gaussians.detach().float(),
            "cam_view": model_input.decoder.cam_view.detach().float(),
            "intrinsics": model_input.decoder.intrinsics.detach().float(),
        }
        outputs = {
            "loss_instance_group": torch.zeros(
                (), device=center_pred.device
            ),
            "center_pred": center_pred.detach(),
            "center_offset": offset.detach(),
        }
        # pseudo-GT unit instance distributions (also used by the eval
        # script for the foreground mask and diagnostics)
        gs_gt = gs_conf = gs_nviews = None
        p_u = None
        bg_idx = -1
        if "instance_label_output" in data:
            gs_gt, gs_conf, gs_nviews = self._pseudo_gs_labels(
                means.reshape(batch_size, n_gs, 3), data, opt
            )
            high_conf = (
                (gs_conf >= self.pseudo_conf)
                & (gs_nviews >= self.pseudo_min_views)
            )
            ids_present = torch.unique(gs_gt)
            id_map = {
                int(iid): idx
                for idx, iid in enumerate(ids_present.tolist())
            }
            m = len(ids_present)
            onehot = torch.zeros(
                (batch_size, token_count, p, m),
                device=gs_gt.device,
                dtype=a.dtype,
            )
            for iid, idx in id_map.items():
                onehot[..., idx] = (
                    gs_gt.reshape(batch_size, token_count, p) == iid
                ).to(a.dtype) * high_conf.reshape(
                    batch_size, token_count, p
                ).to(a.dtype)
            h = torch.einsum("btpk,btpi->btki", a, onehot)
            mass_u = h.sum(-1, keepdim=True).clamp_min(1e-6)
            p_u = (h / mass_u).reshape(batch_size, u_count, m)
            bg_idx = id_map.get(0, -1)
        self.last_unit_pu = (
            p_u.detach() if p_u is not None else None
        )
        self.last_unit_bg_idx = bg_idx
        if training and "instance_label_output" in data:
            gs_pos = means.reshape(batch_size, n_gs, 3).float()
            gs_pos_norm = (gs_pos - scene_center) / scene_scale
            valid = (gs_gt > 0) & (
                gs_conf >= self.pseudo_conf
            )
            max_id = int(gs_gt.max()) + 1
            centers = torch.zeros(
                batch_size, max_id, 3, device=gs_pos.device
            )
            masses = torch.zeros(
                batch_size, max_id, device=gs_pos.device
            )
            for b in range(batch_size):
                fg = valid[b]
                if not fg.any():
                    continue
                idx = gs_gt[b][fg]
                wgt = gs_conf[b][fg].clamp_min(1e-6)
                centers[b].index_add_(
                    0, idx, gs_pos_norm[b][fg] * wgt.unsqueeze(-1)
                )
                masses[b].index_add_(0, idx, wgt)
            inst_center = centers / masses.clamp_min(1e-6).unsqueeze(-1)
            inst_center_mapped = torch.zeros(
                batch_size, m, 3, device=gs_pos.device
            )
            for iid, idx in id_map.items():
                if int(iid) < max_id:
                    inst_center_mapped[:, idx] = inst_center[:, int(iid)]
            p_fg = p_u.clone()
            if bg_idx >= 0:
                p_fg[:, :, bg_idx] = 0.0
            target = torch.einsum(
                "buk,bkd->bud", p_fg, inst_center_mapped
            )  # [B,U,3]
            fg_mass = p_fg.sum(-1)
            valid_u = fg_mass > 0.05
            w = valid_u.float()
            loss = (
                (center_pred - target).square().sum(-1) * w
            ).sum() / w.sum().clamp_min(1)
            center_err = (
                (center_pred - target).norm(dim=-1) * w
            ).sum() / w.sum().clamp_min(1)
            outputs["loss_instance_group"] = (
                float(
                    getattr(opt, "instance_center_loss_weight", 1.0)
                )
                * loss
            )
            outputs["center_err"] = center_err.detach()
            outputs["center_target"] = target.detach()
        return outputs

    def _forward_dynamic_queries(
        self,
        a,
        unit_feat,
        unit_center,
        means,
        gaussians,
        data,
        opt,
        training,
        model_input,
        batch_size,
        token_count,
        n_gs,
        p,
        dense_features=None,
    ) -> dict:
        """Dynamic scene instance queries -> direct mask prediction.

        One-shot feed-forward: scene-specific queries are generated from the
        current scene's unit features (PointGroup-style center offset prior
        + FPS + Mask3D-style cross-attention), predict query->unit mask
        logits, propagate to Gaussians through the frozen GS->unit
        assignment, render, and softmax over K+void.  Supervised by GT
        instance masks via the existing Hungarian + BCE/Dice +
        void/unmatched + usage entropy + 3D consistency.  No clustering at
        inference - the rendered probability is the instance mask output.
        """
        k = self.units_per_token
        u_count = token_count * k
        D = self.gs_feat_dim
        unit_feat_flat = unit_feat.reshape(batch_size, u_count, D)
        center = unit_center.reshape(batch_size, u_count, 3).float()
        scene_center = center.mean(dim=1, keepdim=True)
        scene_scale = (
            (center - scene_center).square().mean(dim=(1, 2), keepdim=True)
            .sqrt().clamp_min(1e-3)
        )
        pos_norm = (center - scene_center) / scene_scale
        self.last_pos_norm = pos_norm.detach()
        unit_dino = self._dino_unit_features(
            a, means, data, opt, batch_size, token_count, n_gs, p
        ) if self.dino_unit else None
        dino_flat = (
            unit_dino.reshape(batch_size, u_count, -1)
            if unit_dino is not None
            else None
        )
        unit_logits, q, seed_idx = self.dynamic_query_head(
            unit_feat_flat, pos_norm, dino_flat
        )
        K = int(unit_logits.shape[1])
        self.last_unit_embeddings = unit_feat_flat.detach()
        self.last_unit_pu = None
        self.last_unit_bg_idx = -1
        self.last_dynamic_query_pos = pos_norm.gather(
            1, seed_idx.unsqueeze(-1).expand(-1, -1, 3)
        ).detach()
        self.last_dynamic_query_seed = seed_idx.detach()

        unit_logits_t = unit_logits.reshape(
            batch_size, K, token_count, k
        )
        gs_logits = torch.einsum(
            "bktu,btpu->bktp", unit_logits_t, a
        ).reshape(batch_size, K, n_gs)
        # Mask3D-style per-query SIGMOID masks: every query independently
        # predicts foreground probability (no softmax competition over
        # queries, so no winner-take-all void absorption / starvation).
        void_logits = (
            torch.zeros(
                batch_size, 1, n_gs,
                device=gs_logits.device, dtype=gs_logits.dtype,
            )
            + self.dynamic_query_head.void_bias
        )
        gaussian_group_probs = torch.sigmoid(
            torch.cat([gs_logits, void_logits], dim=1)
        )  # [B,K+1,N]
        render = self.renderer.render_feature_channels(
            gaussians,
            gaussian_group_probs.transpose(1, 2),
            model_input.decoder.cam_view,
            intrinsics=model_input.decoder.intrinsics,
        )
        rendered_channels = render["images_pred"] / (
            render["alphas_pred"] + 1e-5
        )
        rendered_probs = rendered_channels.float()  # [B,V,K+1,H,W]
        rendered_probability = rendered_probs.permute(
            0, 2, 1, 3, 4
        ).unsqueeze(3)  # [B,K+1,V,1,H,W]
        outputs = {
            "gaussian_group_probs": gaussian_group_probs,
            "gaussian_group_probabilities": gaussian_group_probs,
            "rendered_instance_group_probability": rendered_probability,
            "rendered_instance_group_alpha": render["alphas_pred"],
            "num_clusters": torch.tensor(
                K, device=gs_logits.device
            ),
            "loss_instance_group": torch.zeros(
                (), device=gs_logits.device
            ),
        }
        if self.query_diversity > 0:
            outputs["loss_query_diversity"] = (
                self.query_diversity
                * self.dynamic_query_head.query_diversity_loss(
                    q, margin=self.query_diversity_margin
                )
            )
        else:
            outputs["loss_query_diversity"] = torch.zeros(
                (), device=gs_logits.device
            )
        with torch.no_grad():
            argmax = rendered_probs.argmax(dim=2)  # [B,V,H,W]
            active = argmax < K  # not void
            outputs["active_query_frac"] = (
                active.float().mean(dim=(2, 3)).mean(dim=1)
            )
            # number of distinct queries that win any non-void pixel
            onehot = F.one_hot(
                argmax.clamp(0, K), num_classes=K + 1
            )[..., :K]  # [B,V,H,W,K]
            outputs["active_query_count"] = onehot.any(
                dim=(1, 2, 3)
            ).float().sum(dim=-1)  # [B]
        if training and "instance_label_output" in data:
            loss, stats = hungarian_instance_group_loss(
                rendered_probability,
                data["instance_label_output"].long(),
                num_groups=K,
                min_instance_pixels=int(
                    getattr(opt, "instance_group_min_instance_pixels", 32)
                ),
                dice_weight=float(
                    getattr(opt, "lambda_instance_group_dice", 1.0)
                ),
                mask_weight=float(
                    getattr(opt, "lambda_instance_group_mask", 1.0)
                ),
                void_weight=float(
                    getattr(opt, "lambda_instance_group_void", 0.1)
                ),
                unmatched_weight=float(
                    getattr(opt, "lambda_instance_group_unmatched", 0.1)
                ),
                usage_entropy_weight=float(
                    getattr(opt, "instance_group_usage_entropy", 0.05)
                ),
            )
            outputs["loss_instance_group"] = loss
            outputs.update(stats)
            outputs["loss_instance_group"] = (
                outputs["loss_instance_group"]
                + outputs["loss_query_diversity"]
            )
            lambda_3d = float(getattr(opt, "lambda_instance_group_3d", 1.0))
            if lambda_3d > 0:
                cam_views = torch.cat(
                    [data["cam_view_input"], data["cam_view"]], dim=1
                )
                intrinsics_all = torch.cat(
                    [data["intrinsics_input"], data["intrinsics"]], dim=1
                )
                labels_all = torch.cat(
                    [
                        data["instance_label_input"],
                        data["instance_label_output"],
                    ],
                    dim=1,
                )
                loss_3d, stats_3d = instance_group_3d_loss(
                    gaussian_group_probs.transpose(1, 2),
                    gaussians,
                    cam_views,
                    intrinsics_all,
                    labels_all.long(),
                    num_groups=K,
                    image_size=tuple(opt.img_size),
                    min_instance_gs=int(
                        getattr(opt, "instance_group_3d_min_gs", 16)
                    ),
                    void_weight=float(
                        getattr(opt, "lambda_instance_group_void", 0.1)
                    ),
                    unmatched_weight=float(
                        getattr(opt, "lambda_instance_group_unmatched", 0.1)
                    ),
                )
                outputs["loss_instance_group"] = (
                    outputs["loss_instance_group"] + lambda_3d * loss_3d
                )
                outputs.update(stats_3d)
        return outputs

    def _forward_unit_embedding(
        self,
        a: torch.Tensor,
        unit_feat: torch.Tensor,
        unit_center: torch.Tensor,
        means: torch.Tensor,
        gaussians: torch.Tensor,
        data: dict,
        opt,
        training: bool,
        model_input,
        batch_size: int,
        token_count: int,
        n_gs: int,
        p: int,
        dense_features: torch.Tensor | None = None,
    ) -> dict:
        """Instance-aware unit embedding: soft InfoNCE (train) or
        deterministic agglomerative clustering + render (eval)."""
        k = self.units_per_token
        u_count = token_count * k
        unit_feat_flat = unit_feat.reshape(batch_size, u_count, self.gs_feat_dim)
        center = unit_center.reshape(batch_size, u_count, 3).float()
        scene_center = center.mean(dim=1, keepdim=True)
        scene_scale = (
            (center - scene_center).square().mean(dim=(1, 2), keepdim=True)
            .sqrt()
            .clamp_min(1e-3)
        )
        pos_norm = (center - scene_center) / scene_scale
        self.last_pos_norm = pos_norm.detach()
        unit_dino = self._dino_unit_features(
            a, means, data, opt, batch_size, token_count, n_gs, p
        ) if self.dino_unit else None
        if self.grounding:
            # InstanceSplat-style grounding: an explicit instance feature
            # per unit, generated from the unit feature + DINO + 3D
            # position, rendered and supervised by GT masks (no identity
            # InfoNCE / agglomerative training objective).
            g_in = [unit_feat_flat.float()]
            if unit_dino is not None:
                g_in.append(unit_dino)
            g_in.append(pos_norm)
            e = F.normalize(
                self.grounding_net(torch.cat(g_in, dim=-1)), dim=-1
            )
            self.last_instance_gaussians = gaussians.detach()
            parts = [e]
        else:
            if self.identity_encoder is not None:
                e = self.identity_encoder(unit_feat_flat, pos_norm)  # [B,U,D]
            else:
                # Feature shaping: the unit formation feature itself is the
                # instance-aware embedding (no extra projection).
                e = F.normalize(unit_feat_flat.float(), dim=-1)
                self.last_instance_gaussians = gaussians.detach()
            parts = [e]
            if self.unit_image and dense_features is not None:
                # Frozen multi-view encoder patch features projected onto GS
                # centers, aggregated per unit by the frozen assignment.
                fused = self._gs_dense_features_cached(
                    means.reshape(batch_size, n_gs, 3),
                    dense_features,
                    data,
                    opt,
                    batch_size,
                    n_gs,
                )  # [B,N,dense_dim]
                img_gs = self.unit_image_net(fused)  # [B,N,D]
                img_gs_t = img_gs.reshape(batch_size, token_count, p, -1)
                unit_img = torch.einsum("btpk,btpd->btkd", a, img_gs_t)
                unit_img = unit_img.reshape(batch_size, u_count, -1)
                parts.append(F.normalize(unit_img.float(), dim=-1))
            if unit_dino is not None:
                parts.append(unit_dino)
            if len(parts) > 1:
                e = F.normalize(torch.cat(parts, dim=-1), dim=-1)

        gs_gt, gs_conf, gs_nviews = self._pseudo_gs_labels(
            means.reshape(batch_size, n_gs, 3), data, opt
        )
        gt0 = gs_gt.view(batch_size, token_count, p)
        high_conf = (
            (gs_conf >= self.pseudo_conf)
            & (gs_nviews >= self.pseudo_min_views)
        ).view(batch_size, token_count, p)
        ids_present = torch.unique(gt0)
        id_map = {
            int(iid): idx for idx, iid in enumerate(ids_present.tolist())
        }
        m = len(ids_present)
        onehot = torch.zeros(
            (batch_size, token_count, p, m),
            device=means.device,
            dtype=a.dtype,
        )
        for iid, idx in id_map.items():
            onehot[..., idx] = (gt0 == iid).to(a.dtype) * high_conf.to(a.dtype)
        h = torch.einsum("btpk,btpi->btki", a, onehot)  # [B,T,K,m]
        mass = h.sum(-1, keepdim=True).clamp_min(1e-6)
        p_u = (h / mass).reshape(batch_size, u_count, m)  # [B,U,m]
        # Units whose high-confidence GS mass is too low are excluded from
        # the identity supervision (their p_u rows are zeroed).
        unit_mass_frac = torch.einsum(
            "btpk,btp->btk", a, high_conf.to(a.dtype)
        ) / a.sum(dim=2).clamp_min(1e-6)
        unit_valid = unit_mass_frac >= self.pseudo_unit_min_mass
        p_u = p_u * unit_valid.reshape(batch_size, u_count, 1).to(a.dtype)
        bg_idx = id_map.get(0, -1)
        self.last_unit_embeddings = e.detach()
        self.last_unit_pu = p_u.detach()
        self.last_unit_bg_idx = bg_idx

        if training:
            stats = self._embedding_stats(e, p_u, bg_idx)
            render_stats = {}
            if self.render_space and gs_gt.numel() > 0:
                rs_loss, render_stats = self._rendered_space_embedding_loss(
                    e,
                    a,
                    means,
                    gaussians,
                    gs_gt,
                    data,
                    opt,
                    model_input,
                    batch_size,
                    token_count,
                    n_gs,
                    p,
                )
            else:
                rs_loss = torch.zeros((), device=e.device)
            g3d_stats = {}
            if self.grounding and self.grounding_3d_pull > 0:
                g3d_loss, g3d_stats = self._grounding_3d_prototype_loss(
                    e, p_u, bg_idx
                )
            else:
                g3d_loss = torch.zeros((), device=e.device)
            # Keep units diverse and 3D-local while the extractor is shaped.
            a_flat = a.reshape(batch_size, token_count * p, k)
            usage = a_flat.mean(dim=1)  # [B,K]
            unit_entropy = -(
                usage * usage.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
            gs_center = means.mean(dim=(1, 2), keepdim=True)
            gs_scale = (
                (means - gs_center).square().mean(dim=(1, 2, 3), keepdim=True)
                .sqrt()
                .clamp_min(1e-3)
            )
            pos_norm_gs = (means - gs_center) / gs_scale
            unit_center_norm = torch.einsum("btpk,btpx->btkx", a, pos_norm_gs)
            diff = pos_norm_gs.unsqueeze(3) - unit_center_norm.unsqueeze(2)
            sq_dist = diff.square().sum(-1).clamp_max(25.0)
            compact = (
                a * sq_dist
            ).sum(dim=(2, 3)) / a.sum(dim=(2, 3)).clamp_min(1e-3)
            compact = compact.mean()
            sic_reg = torch.zeros((), device=e.device)
            sic_monitors = {}
            if self.sic_units and self._sic_usage_att is not None:
                usage = self._sic_usage_att
                ent = -(usage * usage.clamp_min(1e-8).log()).sum()
                sic_reg = -self.sic_usage * ent
                sic_monitors = {
                    "sic_usage_entropy": ent.detach(),
                    "sic_active_ratio": (
                        (usage.detach() > 1e-3).float().mean()
                    ),
                    "sic_gate_value": self.sic_gate.detach().reshape(()),
                }
            outputs = {
                "loss_instance_group": (
                    self.embed_loss * stats["loss"]
                    + self.unit_entropy_weight * unit_entropy
                    + self.unit_compactness_weight * compact
                    + rs_loss
                    + g3d_loss
                    + sic_reg
                ),
                "loss_unit_embedding": stats["loss"].detach(),
                "loss_unit_entropy": unit_entropy.detach(),
                "loss_unit_compactness": compact.detach(),
                "unit_embedding_same_sim": stats["same_sim"].detach(),
                "unit_embedding_diff_sim": stats["diff_sim"].detach(),
                "unit_embedding_collapse": stats["collapse_sim"].detach(),
                "unit_knn_agreement": stats["knn_agree"].detach(),
                "unit_embedding_margin": stats["margin"].detach(),
                "unit_embedding_center_cos_mean": stats[
                    "center_cos_mean"
                ].detach(),
                "unit_embedding_center_margin": stats[
                    "center_margin"
                ].detach(),
                "pseudo_gs_kept_ratio": high_conf.float().mean().detach(),
                "pseudo_unit_kept_ratio": unit_valid.float().mean().detach(),
            }
            for _k, _v in render_stats.items():
                outputs[_k] = _v.detach()
            for _k, _v in g3d_stats.items():
                outputs[_k] = _v.detach()
            outputs.update(sic_monitors)
            outputs["sic_reg"] = sic_reg.detach()
            outputs.update(
                self._cross_token_embedding_stats(e, p_u, bg_idx)
            )
            return outputs

        feat = torch.cat(
            [e, self.cluster_pos_weight * pos_norm], dim=-1
        ).detach().cpu().numpy()
        labels = np.zeros((batch_size, u_count), dtype=np.int64)
        if u_count > 20000:
            # Exact agglomerative is O(U^2) memory/time at 65536 units
            # (scipy materializes ~17 GB condensed distances, sklearn
            # takes ~6 min/scene).  Use density-based DBSCAN instead; noise
            # units become singleton clusters below so nothing is dropped.
            from sklearn.cluster import DBSCAN

            for b in range(batch_size):
                m = DBSCAN(
                    eps=float(self.cluster_eps),
                    min_samples=2,
                    metric="euclidean",
                    algorithm="kd_tree",
                    n_jobs=-1,
                )
                lab = m.fit_predict(feat[b])
                # noise (-1) -> own singleton cluster
                lab = np.where(lab < 0, np.arange(u_count) + 1000000, lab)
                uniq = {c: i for i, c in enumerate(np.unique(lab))}
                labels[b] = np.array([uniq[c] for c in lab], dtype=np.int64)
        else:
            from scipy.cluster.hierarchy import fcluster, linkage

            for b in range(batch_size):
                Z = linkage(feat[b], method="average")
                labels[b] = fcluster(
                    Z, t=self.cluster_eps, criterion="distance"
                ) - 1
        fg_share_u = (
            1.0 - p_u[..., bg_idx] if bg_idx >= 0 else torch.ones_like(
                p_u[..., 0]
            )
        )
        cluster_list = []
        used_per_batch = []
        for b in range(batch_size):
            used = {}
            for c in np.unique(labels[b]):
                sel = labels[b] == c
                fg_share = float(fg_share_u[b][sel].mean())
                if fg_share >= self.void_fg_share:
                    used[int(c)] = len(cluster_list)
                    cluster_list.append(int(c))
            used_per_batch.append(used)
        # Index-based membership: [B,U] cluster ids -> onehot [B,U,C+1]
        # (void channel = last), avoiding the O(U^2) one-hot matrix.
        num = len(cluster_list)
        cluster_ids = np.full((batch_size, u_count), num, dtype=np.int64)
        for b in range(batch_size):
            used = used_per_batch[b]
            for uu in range(u_count):
                c = int(labels[b, uu])
                if c in used:
                    cluster_ids[b, uu] = used[c]
        cluster_ids_t = torch.from_numpy(cluster_ids).to(e.device)
        cluster_onehot = torch.nn.functional.one_hot(
            cluster_ids_t, num_classes=num + 1
        ).to(e.dtype)
        num_clusters = len(cluster_list)
        cluster_probs = cluster_onehot[..., : num_clusters + 1]
        unit_probs_t = cluster_probs.reshape(
            batch_size, token_count, k, num_clusters + 1
        )
        gaussian_group_probs = torch.einsum(
            "btpk,btkl->btpl", a, unit_probs_t
        ).reshape(batch_size, n_gs, num_clusters + 1)
        render = self.renderer.render_feature_channels(
            gaussians.float(),
            gaussian_group_probs,
            model_input.decoder.cam_view,
            intrinsics=model_input.decoder.intrinsics,
            opacity_scale=float(getattr(opt, "instance_group_render_scale", 1.0)),
        )
        rendered_channels = render["images_pred"] / (render["alphas_pred"] + 1e-5)
        rendered_probs = rendered_channels.float() / rendered_channels.float().sum(
            dim=2, keepdim=True
        ).clamp_min(1e-6)
        rendered_probability = rendered_probs.permute(0, 2, 1, 3, 4).unsqueeze(3)
        stats = self._embedding_stats(e, p_u, bg_idx)
        outputs = {
            "instance_group_probabilities": gaussian_group_probs,
            "gaussian_group_probabilities": gaussian_group_probs,
            "rendered_instance_group_probability": rendered_probability,
            "rendered_instance_group_alpha": render["alphas_pred"],
            "loss_instance_group": torch.zeros(
                (), device=rendered_probability.device
            ),
            "num_clusters": torch.tensor(
                num_clusters, device=rendered_probability.device
            ),
            "loss_unit_embedding": stats["loss"].detach(),
            "unit_embedding_same_sim": stats["same_sim"].detach(),
            "unit_embedding_diff_sim": stats["diff_sim"].detach(),
            "unit_embedding_collapse": stats["collapse_sim"].detach(),
            "unit_knn_agreement": stats["knn_agree"].detach(),
            "unit_embedding_margin": stats["margin"].detach(),
            "unit_embedding_center_cos_mean": stats["center_cos_mean"].detach(),
            "unit_embedding_center_margin": stats["center_margin"].detach(),
            "pseudo_gs_kept_ratio": high_conf.float().mean().detach(),
            "pseudo_unit_kept_ratio": unit_valid.float().mean().detach(),
        }
        sic_monitors = {}
        if self.sic_units and self._sic_att is not None:
            usage = self._sic_att.mean(dim=(0, 1, 2))
            ent = -(usage * usage.clamp_min(1e-8).log()).sum()
            sic_monitors = {
                "sic_usage_entropy": ent.detach(),
                "sic_active_ratio": (
                    (usage.detach() > 1e-3).float().mean()
                ),
                "sic_gate_value": self.sic_gate.detach().reshape(()),
            }
        outputs.update(sic_monitors)
        outputs.update(self._cross_token_embedding_stats(e, p_u, bg_idx))
        return outputs

    def _ensure_dino_model(self):
        if self._dino_model is not None:
            return self._dino_model
        import torch.hub as hub

        hub.set_dir(str(Path.home() / ".cache/torch/hub"))
        model = hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            source="github",
            force_reload=False,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        # Keep DINOv2 OUT of nn.Module's state_dict: it is a frozen external
        # extractor and saving 86M of its weights bloats checkpoints and
        # breaks resume strict validation.
        self.__dict__["_dino_model"] = model
        return model

    def _dino_unit_features(
        self,
        a: torch.Tensor,
        means: torch.Tensor,
        data: dict,
        opt,
        batch_size: int,
        token_count: int,
        n_gs: int,
        p: int,
    ) -> torch.Tensor | None:
        """Frozen DINOv2 dense features -> unit-level [B,U,dino_dim].

        DINO patch tokens (252x252 frame, ViT-B/14) are projected onto the
        GS centers with the same camera/projection machinery as the patch
        branch, aggregated per unit by the frozen soft assignment, then
        mapped by the trainable ``dino_proj``.  The DINO extractor is
        frozen and detached; only the projection trains.
        """
        if not self.dino_unit:
            return None
        if "images_input" not in data or "cam_to_world_input" not in data:
            return None
        # One DINO forward per model forward: the raw per-GS features are
        # cached so the SIC token descriptor and the unit embedding share the
        # same (frozen) DINO evidence.
        gs_dino = self._dino_gs_raw(
            means.reshape(batch_size, n_gs, 3), data, batch_size, n_gs
        )  # [B,N,768]
        gs_dino_t = gs_dino.reshape(batch_size, token_count, p, -1)
        unit_dino = torch.einsum("btpk,btpd->btkd", a, gs_dino_t)
        mass = a.sum(dim=2).clamp_min(1e-6)  # [B,T,K]
        unit_dino = (unit_dino / mass.unsqueeze(-1)).reshape(
            batch_size, token_count * self.units_per_token, -1
        )
        unit_dino = F.normalize(unit_dino, dim=-1)
        # Trainable projection adapts DINO to the unit-embedding space.
        return F.normalize(self.dino_proj(unit_dino.detach()), dim=-1)

    def _dino_gs_raw(
        self,
        means: torch.Tensor,
        data: dict,
        batch_size: int,
        n_gs: int,
    ) -> torch.Tensor:
        """Per-GS DINOv2 dense features with per-forward caching."""
        if (
            self._gs_dino_cache is not None
            and self._gs_dino_cache.shape
            == (batch_size, n_gs, 768)
            and self._gs_dino_cache.device == means.device
        ):
            return self._gs_dino_cache
        feats = self._compute_dino_gs_raw(
            means, data, batch_size, n_gs
        )
        self._gs_dino_cache = feats
        return feats

    def _gs_dense_features_cached(
        self,
        means_flat: torch.Tensor,
        dense_features: torch.Tensor,
        data: dict,
        opt,
        batch_size: int,
        n_gs: int,
    ) -> torch.Tensor | None:
        """Frozen encoder patch features projected onto GS centers (cached).

        The SIC token descriptor and the unit-image embedding share one
        projection per forward; identical to the inline projection used by
        the 0.324 baseline so results are unchanged when SIC is off.
        """
        if dense_features is None:
            return None
        if (
            self._gs_dense_cache is not None
            and self._gs_dense_cache.shape[0] == batch_size
            and self._gs_dense_cache.shape[1] == n_gs
            and self._gs_dense_cache.device == means_flat.device
        ):
            return self._gs_dense_cache
        batch_size_, view_patch, channel = dense_features.shape
        view_count = int(getattr(opt, "num_input_views", 8))
        patches_per_view = view_patch // view_count
        hf = wf = int(round(patches_per_view**0.5))
        dense_img = (
            dense_features.reshape(
                batch_size_, view_count, patches_per_view, channel
            )
            .permute(0, 1, 3, 2)
            .reshape(batch_size_, view_count, channel, hf, wf)
        )
        fused, _ = _project_dense_features(
            means_flat,
            dense_img,
            data["cam_to_world_input"],
            data["intrinsics_input"],
            tuple(opt.img_size),
        )
        self._gs_dense_cache = fused
        return fused

    def _compute_dino_gs_raw(
        self,
        means: torch.Tensor,
        data: dict,
        batch_size: int,
        n_gs: int,
    ) -> torch.Tensor:
        """Per-GS DINOv2 dense features [B,N,768] (L2-normalized, detached).

        DINO patch tokens are projected onto the GS centers with the same
        camera/projection machinery as the patch branch.  The extractor is
        frozen; the result is detached (only the direct-GS embedding head
        trains against it).
        """
        if "images_input" not in data or "cam_to_world_input" not in data:
            return torch.zeros(
                (batch_size, n_gs, 768), device=means.device
            )
        images = data["images_input"]
        b, v, c, h, w = images.shape
        device = means.device
        model = self._ensure_dino_model().to(device)
        target = 252
        x = F.interpolate(
            images.reshape(b * v, c, h, w),
            size=(target, target),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=device
        ).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            out = model.get_intermediate_layers(x, n=1)
            tokens = out[0]
        dim = tokens.shape[-1]
        hf = wf = int(round(tokens.shape[1] ** 0.5))
        feats = tokens.reshape(b * v, hf, wf, dim).permute(0, 3, 1, 2)
        feats = F.normalize(feats, dim=1).reshape(b, v, dim, hf, wf)
        s = torch.tensor(
            [target / w, target / h, target / w, target / h],
            dtype=data["intrinsics_input"].dtype,
            device=device,
        )
        intrinsics = data["intrinsics_input"] * s
        with torch.no_grad():
            fused, _ = _project_dense_features(
                means.reshape(batch_size, n_gs, 3),
                feats,
                data["cam_to_world_input"],
                intrinsics,
                (target, target),
            )
        return F.normalize(fused, dim=-1).detach()

    def _forward_scene_assignment(
        self,
        a: torch.Tensor,
        unit_feat: torch.Tensor,
        unit_center: torch.Tensor,
        means: torch.Tensor,
        gaussians: torch.Tensor,
        data: dict,
        opt,
        training: bool,
        model_input,
        batch_size: int,
        token_count: int,
        n_gs: int,
        p: int,
        dense_features: torch.Tensor | None = None,
        lambda_eff: float = 1.0,
    ) -> dict:
        """Direct unit -> scene-specific instance assignment (no clustering).

        Frozen unit features (+ optional frozen multi-view image features)
        and 3D centers go into the scene assignment head: FPS-init slots,
        iterative refinement, then a unit->slot(+void) soft assignment.  GS
        inherits its unit's assignment and masks are rendered through the
        frozen Gaussian geometry.  Training supervises the assignment
        directly (unit-level Hungarian BCE/Dice on soft distributions, void
        BCE, usage entropy) plus the existing rendered-mask Hungarian
        BCE/Dice and 3D consistency.
        """
        k = self.units_per_token
        u_count = token_count * k
        unit_feat_flat = unit_feat.reshape(batch_size, u_count, self.gs_feat_dim)
        center = unit_center.reshape(batch_size, u_count, 3).float()
        scene_center = center.mean(dim=1, keepdim=True)
        scene_scale = (
            (center - scene_center).square().mean(dim=(1, 2), keepdim=True)
            .sqrt()
            .clamp_min(1e-3)
        )
        pos_norm = (center - scene_center) / scene_scale
        head_feat = F.normalize(unit_feat_flat.float(), dim=-1)
        if self.unit_image and dense_features is not None:
            B_, VP, C_ = dense_features.shape
            V_ = int(getattr(opt, "num_input_views", 8))
            P_ = VP // V_
            Hf = Wf = int(round(P_**0.5))
            dense_img = (
                dense_features.reshape(B_, V_, P_, C_)
                .permute(0, 1, 3, 2)
                .reshape(B_, V_, C_, Hf, Wf)
            )
            fused, _ = _project_dense_features(
                means.reshape(batch_size, n_gs, 3),
                dense_img,
                data["cam_to_world_input"],
                data["intrinsics_input"],
                tuple(opt.img_size),
            )
            img_gs = self.unit_image_net(fused)  # [B,N,D]
            img_gs_t = img_gs.reshape(batch_size, token_count, p, -1)
            unit_img = torch.einsum("btpk,btpd->btkd", a, img_gs_t)
            unit_img = unit_img.reshape(batch_size, u_count, -1)
            head_feat = F.normalize(
                torch.cat([head_feat, F.normalize(unit_img.float(), dim=-1)],
                          dim=-1),
                dim=-1,
            )
        if self.dino_unit:
            unit_dino = self._dino_unit_features(
                a, means, data, opt, batch_size, token_count, n_gs, p
            )
            if unit_dino is not None:
                head_feat = F.normalize(
                    torch.cat(
                        [head_feat, unit_dino.float()],
                        dim=-1,
                    ),
                    dim=-1,
                )

        pi_unit, logits = self.scene_assignment_head(
            head_feat, center
        )  # [B,U,M+1]
        M = self.scene_slots

        # Pseudo-label soft unit distributions (same convention as the
        # embedding path; conf-filtered).
        gs_gt, gs_conf, gs_nviews = self._pseudo_gs_labels(
            means.reshape(batch_size, n_gs, 3), data, opt
        )
        gt0 = gs_gt.view(batch_size, token_count, p)
        high_conf = (
            (gs_conf >= self.pseudo_conf)
            & (gs_nviews >= self.pseudo_min_views)
        ).view(batch_size, token_count, p)
        ids_present = torch.unique(gt0)
        id_map = {int(iid): idx for idx, iid in enumerate(ids_present.tolist())}
        m = len(ids_present)
        onehot = torch.zeros(
            (batch_size, token_count, p, m),
            device=means.device,
            dtype=a.dtype,
        )
        for iid, idx in id_map.items():
            onehot[..., idx] = (gt0 == iid).to(a.dtype) * high_conf.to(a.dtype)
        h = torch.einsum("btpk,btpi->btki", a, onehot)  # [B,T,K,m]
        mass = h.sum(-1, keepdim=True).clamp_min(1e-6)
        p_u = (h / mass).reshape(batch_size, u_count, m)
        unit_mass_frac = torch.einsum(
            "btpk,btp->btk", a, high_conf.to(a.dtype)
        ) / a.sum(dim=2).clamp_min(1e-6)
        unit_valid = unit_mass_frac >= self.pseudo_unit_min_mass
        p_u = p_u * unit_valid.reshape(batch_size, u_count, 1).to(a.dtype)
        bg_idx = id_map.get(0, -1)
        self.last_unit_embeddings = head_feat.detach()
        self.last_unit_pu = p_u.detach()
        self.last_unit_bg_idx = bg_idx
        self.last_instance_gaussians = gaussians.reshape(
            batch_size, n_gs, 14
        ).detach()

        # Eval: dynamically drop empty slots (no foreground unit mass) and
        # merge their probability into void, so the rendered channel count
        # adapts to the scene instead of always emitting all M slots.
        M_eff = M
        if not training and self.scene_slot_min_mass > 0:
            slot_mass = pi_unit[..., :M].sum(dim=1)  # [B,M]
            keep = slot_mass[0] > self.scene_slot_min_mass
            if keep.any() and not keep.all():
                keep_idx = keep.nonzero(as_tuple=False).squeeze(-1)
                dropped = (~keep)
                pi_keep = pi_unit[:, :, keep_idx]
                void_add = pi_unit[:, :, dropped].sum(
                    dim=-1, keepdim=True
                )
                pi_unit = torch.cat(
                    [pi_keep, pi_unit[:, :, M : M + 1] + void_add],
                    dim=-1,
                )
                M_eff = int(keep_idx.numel())
        # GS inherits its unit's assignment.
        pi_unit_t = pi_unit.reshape(batch_size, token_count, k, M_eff + 1)
        gaussian_group_probs = torch.einsum(
            "btpk,btkl->btpl", a, pi_unit_t
        ).reshape(batch_size, n_gs, M_eff + 1)
        render = self.renderer.render_feature_channels(
            gaussians.reshape(batch_size, n_gs, 14),
            gaussian_group_probs,
            model_input.decoder.cam_view,
            intrinsics=model_input.decoder.intrinsics,
            opacity_scale=float(getattr(opt, "instance_group_render_scale", 1.0)),
        )
        rendered_channels = render["images_pred"] / (
            render["alphas_pred"] + 1e-5
        )
        rendered_probs = rendered_channels.float() / rendered_channels.float().sum(
            dim=2, keepdim=True
        ).clamp_min(1e-6)
        rendered_probability = rendered_probs.permute(0, 2, 1, 3, 4).unsqueeze(3)
        outputs = {
            "instance_group_probabilities": gaussian_group_probs,
            "gaussian_group_probabilities": gaussian_group_probs,
            "rendered_instance_group_probability": rendered_probability,
            "rendered_instance_group_alpha": render["alphas_pred"],
            "loss_instance_group": torch.zeros(
                (), device=rendered_probability.device
            ),
            "num_clusters": torch.tensor(
                M_eff, device=rendered_probability.device
            ),
            "pseudo_gs_kept_ratio": high_conf.float().mean().detach(),
            "pseudo_unit_kept_ratio": unit_valid.float().mean().detach(),
        }
        if not training:
            return outputs

        unit_loss, stats = self.scene_assignment_head.unit_assignment_loss(
            pi_unit,
            p_u,
            bg_idx,
            void_weight=self.scene_slot_void,
            unmatched_weight=self.scene_slot_unmatched,
        )
        mask_outputs = _supervise_rendered_masks(
            gaussian_group_probs,
            gaussians.reshape(batch_size, n_gs, 14),
            rendered_probability,
            M_eff,
            data,
            opt,
            lambda_eff,
        )
        usage = pi_unit[..., :M_eff].mean(dim=1)
        entropy = -(
            usage * usage.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        loss = (
            self.scene_unit_loss_weight * unit_loss
            + lambda_eff * mask_outputs["loss_instance_group"]
            + self.scene_slot_entropy * entropy
        )
        outputs["loss_instance_group"] = loss
        outputs.update(stats)
        outputs.update(
            {
                "loss_unit_assignment": unit_loss.detach(),
                "slot_usage_entropy": entropy.detach(),
                "unit_assignment_agreement": stats[
                    "unit_assignment_agreement"
                ],
            }
        )
        for key, value in mask_outputs.items():
            if key not in outputs and torch.is_tensor(value):
                outputs[key] = value.detach()
        return outputs

    def _forward_dpg(
        self,
        a: torch.Tensor,
        unit_feat: torch.Tensor,
        unit_center: torch.Tensor,
        means: torch.Tensor,
        gaussians: torch.Tensor,
        data: dict,
        opt,
        training: bool,
        model_input,
        batch_size: int,
        token_count: int,
        n_gs: int,
        p: int,
        dense_features: torch.Tensor | None = None,
    ) -> dict:
        """DPG: learn scene-specific instance prototypes -> hard assignment.

        The unit identity embedding (unit_feat + unit_img + DINO) is fed to
        the ScenePrototypeLearner with K = GT instance count (train) or the
        GT-count oracle (eval).  The learned prototypes are supervised to
        match the GT instance prototypes (p_u-weighted unit-embedding means)
        via Hungarian-aligned cosine pull -- no hard-assignment
        bootstrapping, no fixed slots, no agglomerative.  Units are
        hard-assigned by cosine nearest prototype at inference and masks are
        rendered through the frozen geometry.
        """
        from scipy.optimize import linear_sum_assignment

        k = self.units_per_token
        u_count = token_count * k
        unit_feat_flat = unit_feat.reshape(
            batch_size, u_count, self.gs_feat_dim
        )
        center = unit_center.reshape(batch_size, u_count, 3).float()
        scene_center = center.mean(dim=1, keepdim=True)
        scene_scale = (
            (center - scene_center).square().mean(dim=(1, 2), keepdim=True)
            .sqrt()
            .clamp_min(1e-3)
        )
        pos_norm = (center - scene_center) / scene_scale
        e_u = F.normalize(unit_feat_flat.float(), dim=-1)
        if self.unit_image and dense_features is not None:
            B_, VP, C_ = dense_features.shape
            V_ = int(getattr(opt, "num_input_views", 8))
            P_ = VP // V_
            Hf = Wf = int(round(P_**0.5))
            dense_img = (
                dense_features.reshape(B_, V_, P_, C_)
                .permute(0, 1, 3, 2)
                .reshape(B_, V_, C_, Hf, Wf)
            )
            fused, _ = _project_dense_features(
                means.reshape(batch_size, n_gs, 3),
                dense_img,
                data["cam_to_world_input"],
                data["intrinsics_input"],
                tuple(opt.img_size),
            )
            img_gs = self.unit_image_net(fused)
            img_gs_t = img_gs.reshape(batch_size, token_count, p, -1)
            unit_img = torch.einsum("btpk,btpd->btkd", a, img_gs_t)
            unit_img = unit_img.reshape(batch_size, u_count, -1)
            e_u = F.normalize(
                torch.cat(
                    [e_u, F.normalize(unit_img.float(), dim=-1)],
                    dim=-1,
                ),
                dim=-1,
            )
        if self.dino_unit:
            unit_dino = self._dino_unit_features(
                a, means, data, opt, batch_size, token_count, n_gs, p
            )
            if unit_dino is not None:
                e_u = F.normalize(
                    torch.cat([e_u, unit_dino.float()], dim=-1),
                    dim=-1,
                )

        # GT soft instance distributions (15-view majority vote).
        gs_gt, gs_conf, gs_nviews = self._pseudo_gs_labels(
            means.reshape(batch_size, n_gs, 3), data, opt
        )
        gt0 = gs_gt.view(batch_size, token_count, p)
        high_conf = (
            (gs_conf >= self.pseudo_conf)
            & (gs_nviews >= self.pseudo_min_views)
        ).view(batch_size, token_count, p)
        ids_present = torch.unique(gt0)
        id_map = {
            int(iid): idx for idx, iid in enumerate(ids_present.tolist())
        }
        m = len(ids_present)
        onehot = torch.zeros(
            (batch_size, token_count, p, m),
            device=means.device,
            dtype=a.dtype,
        )
        for iid, idx in id_map.items():
            onehot[..., idx] = (gt0 == iid).to(a.dtype) * high_conf.to(a.dtype)
        h = torch.einsum("btpk,btpi->btki", a, onehot)
        mass = h.sum(-1, keepdim=True).clamp_min(1e-6)
        p_u = (h / mass).reshape(batch_size, u_count, m)
        unit_mass_frac = torch.einsum(
            "btpk,btp->btk", a, high_conf.to(a.dtype)
        ) / a.sum(dim=2).clamp_min(1e-6)
        unit_valid = unit_mass_frac >= self.pseudo_unit_min_mass
        p_u = p_u * unit_valid.reshape(batch_size, u_count, 1).to(a.dtype)
        bg_idx = id_map.get(0, -1)
        fg_ids = [
            i for i in range(m) if i != bg_idx
        ] if bg_idx >= 0 else list(range(m))
        fg_ids = [i for i in fg_ids if p_u[..., i].sum() > 1e-3]
        K = len(fg_ids)
        self.last_unit_embeddings = e_u.detach()
        self.last_unit_pu = p_u.detach()
        self.last_unit_bg_idx = bg_idx
        self.last_instance_gaussians = gaussians.reshape(
            batch_size, n_gs, 14
        ).detach()

        if K < 1:
            raise RuntimeError(f"no foreground instances in {data['scene_name']}")

        # Learn scene-specific prototypes (GT count as K).
        P_gen = self.dpg_learner(e_u, K)  # [B,K,D]
        # GT prototypes: p_u-weighted unit-embedding means (detached target).
        with torch.no_grad():
            p_fg = p_u[..., fg_ids]  # [B,U,F]
            denom = p_fg.sum(dim=1, keepdim=True).clamp_min(1e-6)
            P_gt = F.normalize(
                torch.einsum("bud,buf->bfd", e_u, p_fg)
                / denom.transpose(1, 2),
                dim=-1,
            )  # [B,F,D]
        # Hungarian-aligned cosine pull between learned and GT prototypes.
        cos_mat = torch.bmm(P_gen, P_gt.transpose(-1, -2))  # [B,K,F]
        cost = 1.0 - cos_mat
        matched_losses = []
        proto_sims = []
        for b in range(batch_size):
            row_ind, col_ind = linear_sum_assignment(
                cost[b].detach().float().cpu().numpy()
            )
            sims = cos_mat[b, row_ind, col_ind]
            matched_losses.append((1.0 - sims).mean())
            proto_sims.append(sims.mean())
        L_proto = torch.stack(matched_losses).mean()
        proto_sim = torch.stack(proto_sims).mean()

        # Hard cosine assignment (same as the 0.534 oracle).
        sim = torch.bmm(e_u, P_gen.transpose(-1, -2))  # [B,U,K]
        hard = sim.argmax(dim=-1)  # [B,U]
        # Assignment accuracy: unit hard prototype vs GT instance (via the
        # Hungarian mapping between prototypes and GT instances).
        fg_share_u = (
            1.0 - p_u[..., bg_idx] if bg_idx >= 0
            else torch.ones_like(p_u[..., 0])
        )
        fg_mask = (fg_share_u > 0.3) & (p_u[..., fg_ids].sum(-1) > 0)
        agree = torch.zeros((), device=e_u.device)
        agree_count = 0
        for b in range(batch_size):
            row_ind, col_ind = linear_sum_assignment(
                cost[b].detach().float().cpu().numpy()
            )
            map_arr = torch.full(
                (K,), -1, dtype=torch.long, device=e_u.device
            )
            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                map_arr[r] = fg_ids[c]
            hard_gt = map_arr[hard[b]]  # [U] GT id of the hard prototype
            unit_dom = p_u[b][:, fg_ids].argmax(dim=-1)  # [U] dominant GT
            valid = fg_mask[b]
            if valid.any():
                agree = agree + (
                    (hard_gt[valid] == unit_dom[valid]).float().mean()
                )
                agree_count += 1
        assignment_acc = agree / max(agree_count, 1)

        # Render: hard labels -> GS -> frozen geometry.
        void_labels = ~fg_mask
        labels_all = torch.where(
            void_labels,
            torch.full_like(hard, K),
            hard,
        )  # [B,U] (K = void channel)
        onehot_u = torch.zeros(
            (batch_size, u_count, K + 1),
            device=e_u.device,
            dtype=e_u.dtype,
        )
        onehot_u.scatter_(-1, labels_all.unsqueeze(-1), 1.0)
        unit_probs_t = onehot_u.reshape(batch_size, token_count, k, K + 1)
        gaussian_group_probs = torch.einsum(
            "btpk,btkl->btpl", a, unit_probs_t
        ).reshape(batch_size, n_gs, K + 1)
        render = self.renderer.render_feature_channels(
            gaussians.reshape(batch_size, n_gs, 14),
            gaussian_group_probs,
            model_input.decoder.cam_view,
            intrinsics=model_input.decoder.intrinsics,
            opacity_scale=float(getattr(opt, "instance_group_render_scale", 1.0)),
        )
        rendered_channels = render["images_pred"] / (
            render["alphas_pred"] + 1e-5
        )
        rendered_probs = rendered_channels.float() / rendered_channels.float().sum(
            dim=2, keepdim=True
        ).clamp_min(1e-6)
        rendered_probability = rendered_probs.permute(
            0, 2, 1, 3, 4
        ).unsqueeze(3)
        outputs = {
            "instance_group_probabilities": gaussian_group_probs,
            "gaussian_group_probabilities": gaussian_group_probs,
            "rendered_instance_group_probability": rendered_probability,
            "rendered_instance_group_alpha": render["alphas_pred"],
            "loss_instance_group": (
                self.dpg_proto_weight * L_proto if training
                else torch.zeros((), device=e_u.device)
            ),
            "loss_dpg_prototype": L_proto.detach(),
            "dpg_proto_similarity": proto_sim.detach(),
            "dpg_assignment_accuracy": assignment_acc.detach(),
            "num_clusters": torch.tensor(
                K, device=rendered_probability.device
            ),
            "pseudo_gs_kept_ratio": high_conf.float().mean().detach(),
            "pseudo_unit_kept_ratio": unit_valid.float().mean().detach(),
        }
        return outputs

    def _forward_direct_gs(
        self,
        token_hidden: torch.Tensor,
        gaussians: torch.Tensor,
        means: torch.Tensor,
        data: dict,
        opt,
        training: bool,
        model_input,
        batch_size: int,
        token_count: int,
        n_gs: int,
        p: int,
    ) -> dict:
        """InstanceSplat-style per-GS embedding supervised in rendered 2D.

        No unit clustering: a compact D-dim instance embedding is attached to
        each frozen Gaussian, rendered into the target views, and supervised
        directly by the 2D GT instance masks (prototype pull + prototype
        push + cross-view consistency).  At inference the rendered embedding
        is clustered per view (k-means with the GT-count oracle in this first
        version) into a per-pixel instance probability map.
        """
        from scipy.cluster.vq import kmeans2

        dino_gs = self._dino_gs_raw(
            means.reshape(batch_size, n_gs, 3), data, batch_size, n_gs
        ) if self.direct_gs_dino else None
        gs_emb = self.direct_gs_head(
            token_hidden, gaussians, means, dino_gs
        )  # [B,N,D]
        render = self.renderer.render_feature_channels(
            gaussians.reshape(batch_size, n_gs, 14),
            gs_emb,
            model_input.decoder.cam_view,
            intrinsics=model_input.decoder.intrinsics,
        )
        rendered = render["images_pred"] / (
            render["alphas_pred"] + 1e-5
        )  # [B,V,D,H,W]
        D = rendered.shape[2]
        # GT instance count (GT-count oracle at train and eval in v1).
        gs_gt, _, _ = self._pseudo_gs_labels(
            means.reshape(batch_size, n_gs, 3), data, opt
        )
        K = int(gs_gt[0].unique().numel() - 1)
        K = max(1, K)

        if not training:
            # Cluster the rendered per-view embedding (k-means, k=GT count).
            V = rendered.shape[1]
            rendered_n = F.normalize(rendered, dim=2)
            probs = torch.zeros(
                (batch_size, K + 1, V, 1, rendered.shape[3], rendered.shape[4]),
                device=rendered.device,
                dtype=rendered.dtype,
            )
            for b in range(batch_size):
                for v in range(V):
                    feat = rendered_n[b, v].permute(1, 2, 0).reshape(
                        -1, D
                    ).float().cpu().numpy()
                    centroids, labels = kmeans2(
                        feat, K, minit="++", iter=20, seed=0
                    )
                    labels = labels.reshape(
                        rendered.shape[3], rendered.shape[4]
                    )
                    for k in range(K):
                        probs[b, k, v, 0][labels == k] = 1.0
                    probs[b, K, v, 0][labels >= K] = 1.0
            outputs = {
                "instance_group_probabilities": probs,
                "gaussian_group_probabilities": torch.zeros(
                    (batch_size, n_gs, K + 1), device=rendered.device
                ),
                "rendered_instance_group_probability": probs,
                "rendered_instance_group_alpha": render["alphas_pred"],
                "loss_instance_group": torch.zeros(
                    (), device=rendered.device
                ),
                "num_clusters": torch.tensor(
                    K, device=rendered.device
                ),
            }
            return outputs

        # ---- training: rendered-space 2D GT-mask supervision ----
        rendered_n = F.normalize(rendered, dim=2)
        gt_maps = data["instance_label_output"].long()  # [B,V,H,W] per-frame
        # 3D-consistent maps for cross-view (per-GS GT labels -> views).
        cross_maps = []
        for b in range(batch_size):
            cross_maps.append(
                self._project_gs_labels_zbuffer(
                    means[b].reshape(n_gs, 3),
                    gs_gt[b],
                    model_input.decoder.cam_view[b],
                    model_input.decoder.intrinsics[b],
                    tuple(opt.img_size),
                )
            )
        cross_maps = torch.stack(cross_maps)
        pull_losses = []
        push_losses = []
        cross_losses = []
        proto_cos = []
        for b in range(batch_size):
            for v in range(rendered.shape[1]):
                mask = gt_maps[b, v]
                inst_ids = torch.unique(mask[mask > 0])
                if inst_ids.numel() < 1:
                    continue
                proto_by_id = {}
                for k in inst_ids.tolist():
                    px = rendered_n[b, v, :, mask == k]
                    if px.shape[1] < 2:
                        continue
                    proto = F.normalize(px.mean(dim=1), dim=0)
                    proto_by_id[k] = proto
                    pull_losses.append(
                        (1.0 - (px.T * proto).sum(-1)).mean()
                    )
                protos = list(proto_by_id.values())
                if len(protos) >= 2:
                    for i in range(len(protos)):
                        for j in range(i + 1, len(protos)):
                            d = (protos[i] - protos[j]).norm()
                            push_losses.append(
                                F.relu(self.direct_gs_margin_push - d)
                            )
                            proto_cos.append((protos[i] * protos[j]).sum())
                # cross-view via 3D-consistent instance ids.
                for k, proto_v in proto_by_id.items():
                    for v2 in range(rendered.shape[1]):
                        if v2 == v:
                            continue
                        px2 = rendered_n[b, v2, :, cross_maps[b, v2] == k]
                        if px2.shape[1] < 2:
                            continue
                        proto2 = F.normalize(px2.mean(dim=1), dim=0)
                        cross_losses.append(
                            F.relu(
                                (proto_v - proto2).norm()
                                - self.direct_gs_margin_cross
                            )
                        )
        L_pull = torch.stack(pull_losses).mean() if pull_losses else (
            torch.zeros((), device=rendered.device)
        )
        L_push = torch.stack(push_losses).mean() if push_losses else (
            torch.zeros((), device=rendered.device)
        )
        L_cross = torch.stack(cross_losses).mean() if cross_losses else (
            torch.zeros((), device=rendered.device)
        )
        L_info, info_stats = self._rendered_pixel_info_nce(
            rendered_n,
            gt_maps,
            self.direct_gs_info_temp,
            self.direct_gs_info_samples,
        )
        loss = (
            self.direct_gs_pull * L_pull
            + self.direct_gs_push * L_push
            + self.direct_gs_cross * L_cross
            + self.direct_gs_info_nce * L_info
        )
        with torch.no_grad():
            proto_cos_t = (
                torch.stack(proto_cos).mean() if proto_cos
                else torch.zeros((), device=rendered.device)
            )
        outputs = {
            "loss_instance_group": loss,
            "loss_direct_gs_pull": L_pull.detach(),
            "loss_direct_gs_push": L_push.detach(),
            "loss_direct_gs_cross": L_cross.detach(),
            "direct_gs_proto_cos": proto_cos_t.detach(),
            "direct_gs_proto_margin": (
                self.direct_gs_margin_push
                - (torch.stack(push_losses).mean() if push_losses else 0.0)
            ).detach(),
            "num_clusters": torch.tensor(K, device=rendered.device),
            "pseudo_gs_kept_ratio": torch.tensor(
                1.0, device=rendered.device
            ),
        }
        for _k, _v in info_stats.items():
            outputs[_k] = _v.detach()
        return outputs

    def forward_generative_gaussians(
        self,
        token_hidden: torch.Tensor,
        frozen_gaussians: torch.Tensor,
    ) -> torch.Tensor:
        """Token -> 8 local units -> unit-decoded Gaussians (residual).

        Re-runs the unit formation (same code path as ``forward``) and decodes
        each unit's Gaussians through ``unit_gaussian_decoder`` as a
        zero-initialized residual on the frozen per-GS parameters, so at
        initialization the decoded GS are exactly the frozen TokenGS GS
        (PSNR unchanged).  Reconstruction gradients then flow into the unit
        formation through this path, making the units the shared intermediate
        jointly optimized by reconstruction and instance supervision.

        Returns new Gaussians [B, N, 14] (detached base + trainable residual).
        """
        if self.unit_gaussian_decoder is None:
            return frozen_gaussians
        batch_size, token_count, _ = token_hidden.shape
        k = self.units_per_token
        p = self.gaussians_per_token
        n_gs = token_count * p
        th = (
            token_hidden.float()
            if self.backprop_token
            else token_hidden.detach().float()
        )
        fg = frozen_gaussians.detach().float()
        means = fg[..., :3].view(batch_size, token_count, p, 3)
        scene_center = means.mean(dim=(1, 2), keepdim=True)
        scene_scale = (
            (means - scene_center).square().mean(dim=(1, 2, 3), keepdim=True)
            .sqrt()
            .clamp_min(1e-3)
        )
        anchor = means.mean(dim=2, keepdim=True)
        rel_pos = (means - anchor) / scene_scale
        log_scale = fg[..., 4:7].log().view(batch_size, token_count, p, 3)
        opacity = fg[..., 3:4].view(batch_size, token_count, p, 1)
        h_expand = th.unsqueeze(2).expand(
            batch_size, token_count, p, self.token_dim
        )
        f_gs = self.gs_feature_mlp(
            torch.cat([h_expand, rel_pos, log_scale, opacity], dim=-1)
        )  # [B,T,P,Feat]
        q = self.unit_queries.unsqueeze(0).unsqueeze(0).expand(
            batch_size, token_count, k, self.gs_feat_dim
        )
        q_flat = q.reshape(batch_size * token_count, k, self.gs_feat_dim)
        f_flat = f_gs.reshape(batch_size * token_count, p, self.gs_feat_dim)
        for layer in self.unit_layers:
            q_flat = layer(q_flat, f_flat)
        q = q_flat.view(batch_size, token_count, k, self.gs_feat_dim)
        unit_temp = self.log_unit_temp.exp().clamp(0.5, 50.0)
        a_logits = unit_temp * torch.einsum(
            "btpf,btkf->btpk", f_gs, q
        )
        a = F.softmax(a_logits.float(), dim=-1)  # GS -> unit assignment
        unit_feat = torch.einsum("btpk,btpf->btkf", a, f_gs)
        unit_center = torch.einsum("btpk,btpx->btkx", a, means)
        # Per-GS unit context + relative position to the soft unit center.
        unit_ctx_gs = torch.einsum("btpk,btkf->btpf", a, unit_feat)
        unit_center_gs = torch.einsum("btpk,btkx->btpx", a, unit_center)
        rel_center = (means - unit_center_gs) / scene_scale
        fg_t = fg.reshape(batch_size, token_count, p, 14)
        gen_in = torch.cat([f_gs, unit_ctx_gs, rel_center, fg_t], dim=-1)
        delta = self.unit_gaussian_decoder(
            gen_in.reshape(batch_size * token_count * p, -1)
        ).view(batch_size, token_count, p, 14)
        # Keep the decoded parameters in the same valid ranges as the frozen
        # TokenGS activation head.  In particular the scale must stay
        # strictly positive: the unit formation takes ``log(scale)`` and a
        # non-positive scale would inject NaN into the unit features /
        # InfoNCE and poison the whole joint training.  Identity init is
        # preserved because the frozen values already satisfy these bounds.
        dec = fg_t + delta
        new_gs = torch.cat(
            [
                dec[..., :3],  # position (scene-dependent, unclamped)
                dec[..., 3:4].clamp(0.0, 1.0),  # opacity
                dec[..., 4:7].clamp_min(1e-6),  # scale > 0
                dec[..., 7:11],  # rotation
                dec[..., 11:14].clamp(0.0, 1.0),  # color
            ],
            dim=-1,
        ).reshape(batch_size, n_gs, 14)
        return new_gs

    def forward(
        self,
        token_hidden: torch.Tensor,
        frozen_gaussians: torch.Tensor,
        data: dict,
        model_input,
        opt,
        lambda_eff: float = 1.0,
        training: bool = True,
        dense_features: torch.Tensor | None = None,
    ) -> dict:
        batch_size, token_count, _ = token_hidden.shape
        k = self.units_per_token
        p = self.gaussians_per_token
        n_gs = token_count * p
        self._gs_dino_cache = None
        self._gs_dense_cache = None
        self._sic_queries = None
        self._sic_seed_pos = None
        self._sic_att = None
        self._sic_usage_att = None
        th = (
            token_hidden.float()
            if self.backprop_token
            else token_hidden.detach().float()
        )
        fg = frozen_gaussians.detach().float()
        means = fg[..., :3].view(batch_size, token_count, p, 3)
        scene_center = means.mean(dim=(1, 2), keepdim=True)
        scene_scale = (
            (means - scene_center).square().mean(dim=(1, 2, 3), keepdim=True)
            .sqrt()
            .clamp_min(1e-3)
        )
        anchor = means.mean(dim=2, keepdim=True)
        rel_pos = (means - anchor) / scene_scale
        log_scale = fg[..., 4:7].log().view(batch_size, token_count, p, 3)
        opacity = fg[..., 3:4].view(batch_size, token_count, p, 1)
        h_expand = th.unsqueeze(2).expand(
            batch_size, token_count, p, self.token_dim
        )
        f_gs = self.gs_feature_mlp(
            torch.cat([h_expand, rel_pos, log_scale, opacity], dim=-1)
        )  # [B,T,P,Feat]

        # --- learnable local units (soft k-means, geometry-conditioned) ---
        q = self.unit_queries.unsqueeze(0).unsqueeze(0).expand(
            batch_size, token_count, k, self.gs_feat_dim
        )
        if self.sic_units:
            # Direction B0: scene-conditioned instance queries enter the
            # unit-query initialization through a zero gate, so step 0 stays
            # identical to the non-conditioned baseline.
            q = self._sic_condition_unit_queries(
                q,
                th,
                means,
                batch_size,
                token_count,
                n_gs,
                dense_features,
                data,
                opt,
            )
        # Unit refinement is local to each token: [B*T, K, D] queries vs
        # [B*T, P, D] GS features (tiny 64x64 attention per token).
        q_flat = q.reshape(batch_size * token_count, k, self.gs_feat_dim)
        f_flat = f_gs.reshape(batch_size * token_count, p, self.gs_feat_dim)
        for layer in self.unit_layers:
            q_flat = layer(q_flat, f_flat)
        q = q_flat.view(batch_size, token_count, k, self.gs_feat_dim)
        unit_temp = self.log_unit_temp.exp().clamp(0.5, 50.0)
        a_logits = unit_temp * torch.einsum(
            "btpf,btkf->btpk", f_gs, q
        )  # [B,T,P,K]
        a = F.softmax(a_logits.float(), dim=-1)  # GS->unit assignment
        unit_feat = torch.einsum("btpk,btpf->btkf", a, f_gs)
        unit_center = torch.einsum("btpk,btpx->btkx", a, means)
        if self.center_offset:
            return self._forward_center_offset(
                a,
                unit_feat,
                unit_center,
                means,
                fg.reshape(batch_size, n_gs, 14),
                data,
                opt,
                training,
                model_input,
                batch_size,
                token_count,
                n_gs,
                p,
                dense_features,
            )
        if self.dynamic_queries:
            return self._forward_dynamic_queries(
                a,
                unit_feat,
                unit_center,
                means,
                fg.reshape(batch_size, n_gs, 14),
                data,
                opt,
                training,
                model_input,
                batch_size,
                token_count,
                n_gs,
                p,
                dense_features,
            )
        if self.direct_gs:
            return self._forward_direct_gs(
                th,
                fg.reshape(batch_size, n_gs, 14),
                means,
                data,
                opt,
                training,
                model_input,
                batch_size,
                token_count,
                n_gs,
                p,
            )
        if self.dpg:
            return self._forward_dpg(
                a,
                unit_feat,
                unit_center,
                means,
                fg.reshape(batch_size, n_gs, 14),
                data,
                opt,
                training,
                model_input,
                batch_size,
                token_count,
                n_gs,
                p,
                dense_features,
            )
        if self.scene_assignment:
            return self._forward_scene_assignment(
                a,
                unit_feat,
                unit_center,
                means,
                fg.reshape(batch_size, n_gs, 14),
                data,
                opt,
                training,
                model_input,
                batch_size,
                token_count,
                n_gs,
                p,
                dense_features,
                lambda_eff,
            )
        if self.unit_embedding:
            return self._forward_unit_embedding(
                a,
                unit_feat,
                unit_center,
                means,
                fg.reshape(batch_size, n_gs, 14),
                data,
                opt,
                training,
                model_input,
                batch_size,
                token_count,
                n_gs,
                p,
                dense_features,
            )
        unit_center_norm = (unit_center - scene_center) / scene_scale
        unit_ctx = self.unit_norm(
            self.unit_ctx_mlp(unit_feat)
            + self.unit_pos_mlp(unit_center_norm)
        )
        unit_ctx_flat = unit_ctx.reshape(
            batch_size, token_count * k, -1
        )

        # --- scene-specific dynamic prototypes (replace global Group Tokens) ---
        if self.scene_prototypes:
            unit_feat_flat = unit_feat.reshape(
                batch_size, token_count * k, self.gs_feat_dim
            )
            unit_center_flat = unit_center.reshape(
                batch_size, token_count * k, 3
            )
            pi_unit, unit_logits = self.prototype_grouping(
                unit_feat_flat, unit_center_flat
            )
        else:
            # --- global group tokens cross-attend to local units ---
            groups = self.group_tokens.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            for layer in self.group_layers:
                groups = layer(groups, unit_ctx_flat)
            groups = self.group_norm(groups)
            u = F.normalize(self.unit_assignment_proj(unit_ctx_flat), dim=-1)
            g = F.normalize(self.group_assignment_proj(groups), dim=-1)
            temperature = self.log_assignment_temperature.exp().clamp(
                1.0, 100.0
            )
            group_logits = temperature * torch.einsum(
                "bnd,bgd->bng", u, g
            )
            void_logits = self.void_head(unit_ctx_flat)
            unit_logits = torch.cat([group_logits, void_logits], dim=-1)
            pi_unit = F.softmax(unit_logits.float(), dim=-1)  # [B,T*K,G+1]

        # GS inherits its unit's group.
        pi_unit_t = pi_unit.view(
            batch_size, token_count, k, self.num_groups + 1
        )
        gaussian_group_probs = torch.einsum(
            "btpk,btkl->btpl", a, pi_unit_t
        ).reshape(batch_size, n_gs, self.num_groups + 1)  # [B,N,G+1]
        a_flat = a.reshape(batch_size, n_gs, k)
        if self.gs_refine:
            # Per-GS group-logit refinement on the unit prior (zero-init
            # residual; no Gaussian parameter changes).
            prior_logits_gs = torch.einsum(
                "btpk,btkl->btpl",
                a,
                unit_logits.view(
                    batch_size, token_count, k, self.num_groups + 1
                ),
            )  # [B,N,G+1]
            prior_logits_gs = prior_logits_gs.reshape(
                batch_size, n_gs, self.num_groups + 1
            )
            gs_unit_center = torch.einsum(
                "btpk,btkx->btpx", a, unit_center
            ).reshape(batch_size, n_gs, 3)
            pos_flat = means.reshape(batch_size, n_gs, 3)
            rel = (pos_flat - gs_unit_center) / scene_scale.view(
                batch_size, 1, 1
            )
            f_gs_flat = f_gs.reshape(batch_size, n_gs, self.gs_feat_dim)
            local = self.gs_refine_feat(
                torch.cat([f_gs_flat, rel], dim=-1)
            )
            queries = self.gs_refine_group_proj(groups)  # [B,L,R]
            residual = torch.einsum("bnd,bgd->bng", local, queries)
            scale = self.log_gs_refine_scale.exp().clamp(0.0, 1.0)
            refined = prior_logits_gs[..., : self.num_groups] + scale * residual
            final_logits = torch.cat(
                [refined, prior_logits_gs[..., self.num_groups :]], dim=-1
            )
            gaussian_group_probs = F.softmax(final_logits.float(), dim=-1)
        # v2 generative units: the mask-render path consumes the
        # (gradient-tracking) unit-decoded Gaussians so the rendered-mask
        # BCE/Dice and 3D-consistency losses flow into the unit decoder and
        # unit formation -- instance supervision directly shapes the
        # generative GS representation.  The unit formation itself still
        # uses the detached geometry (as in v1 / the 0.324 recipe).
        render_gs = (
            frozen_gaussians.float()
            if (self.generative_units or getattr(self, "abs_render_grad", False))
            else fg
        )
        self.last_instance_gaussians = render_gs.reshape(
            batch_size, n_gs, 14
        )
        self.last_group_probs = gaussian_group_probs.detach()
        self.last_unit_assignment = a.detach()
        self.last_unit_centers = unit_center.detach()

        # --- render instance masks (unit-decoded GS in generative mode) ---
        render = self.renderer.render_feature_channels(
            render_gs.reshape(batch_size, n_gs, 14),
            gaussian_group_probs,
            model_input.decoder.cam_view,
            intrinsics=model_input.decoder.intrinsics,
            opacity_scale=float(
                getattr(opt, "instance_group_render_scale", 1.0)
            ),
        )
        rendered_channels = render["images_pred"] / (
            render["alphas_pred"] + 1e-5
        )
        rendered_probs = rendered_channels.float() / rendered_channels.float().sum(
            dim=2, keepdim=True
        ).clamp_min(1e-6)
        rendered_probability = rendered_probs.permute(0, 2, 1, 3, 4).unsqueeze(
            3
        )
        outputs = {
            "instance_group_probabilities": gaussian_group_probs,
            "gaussian_group_probabilities": gaussian_group_probs,
            "rendered_instance_group_probability": rendered_probability,
            "rendered_instance_group_alpha": render["alphas_pred"],
            "loss_instance_group": torch.zeros(
                (), device=rendered_probability.device
            ),
        }
        if not training:
            return outputs
        outputs.update(
            _supervise_rendered_masks(
                gaussian_group_probs,
                render_gs.reshape(batch_size, n_gs, 14),
                rendered_probability,
                self.num_groups,
                data,
                opt,
                lambda_eff,
            )
        )
        # Unit-formation regularizer: encourage balanced soft units.
        usage = a_flat.mean(dim=1)  # [B,K]
        unit_entropy = -(
            usage * usage.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        outputs["loss_unit_entropy"] = unit_entropy.detach()
        # Spatial compactness (soft k-means objective): penalizes units whose
        # GS spread out, forcing 3D-local units and preventing collapse.
        pos_norm = (means - scene_center) / scene_scale  # [B,T,P,3]
        unit_center_norm = torch.einsum(
            "btpk,btpx->btkx", a, pos_norm
        )
        diff = pos_norm.unsqueeze(3) - unit_center_norm.unsqueeze(2)
        sq_dist = diff.square().sum(-1).clamp_max(25.0)  # robust to outliers
        weighted_var = (a * sq_dist).sum(dim=(2, 3)) / a.sum(
            dim=(2, 3)
        ).clamp_min(1e-3)  # [B,T]
        compact = weighted_var.mean()
        outputs["loss_unit_compactness"] = compact.detach()
        outputs["loss_instance_group"] = (
            outputs["loss_instance_group"]
            + self.unit_entropy_weight * unit_entropy
            + self.unit_compactness_weight * compact
        )
        if (
            self.unit_purity
            and not self.scene_prototypes
            and "instance_label_input" in data
        ):
            # Scheme A: Gaussian-level pseudo-GT -> unit purity loss. Per-GS
            # pseudo instance = 15-view projection + majority vote (detached).
            # Per unit, aggregate the pseudo-instance distribution by the soft
            # assignment a; minimize its entropy, foreground-weighted so
            # background does not dominate. Permutation-invariant by
            # construction (sum over units of per-unit entropies).
            with torch.no_grad():
                cam_views_all = torch.cat(
                    [data["cam_view_input"], data["cam_view"]], dim=1
                )
                intrinsics_all = torch.cat(
                    [data["intrinsics_input"], data["intrinsics"]], dim=1
                )
                labels_all = torch.cat(
                    [
                        data["instance_label_input"],
                        data["instance_label_output"],
                    ],
                    dim=1,
                )
                gs_gt = torch.zeros(
                    (batch_size, n_gs),
                    dtype=torch.long,
                    device=means.device,
                )
                for b in range(batch_size):
                    ids_b, valid_b = _project_gs_to_views(
                        means.reshape(batch_size, n_gs, 3)[b],
                        cam_views_all[b],
                        intrinsics_all[b],
                        labels_all[b],
                        tuple(opt.img_size),
                    )
                    gs_gt[b] = _gs_majority_target(ids_b, valid_b)
            gt0 = gs_gt.view(batch_size, token_count, p)
            ids_present = torch.unique(gt0)
            id_map = {
                int(iid): idx for idx, iid in enumerate(ids_present.tolist())
            }
            m = len(ids_present)
            onehot = torch.zeros(
                (batch_size, token_count, p, m),
                device=means.device,
                dtype=a.dtype,
            )
            for iid, idx in id_map.items():
                onehot[..., idx] = (gt0 == iid).to(a.dtype)
            h = torch.einsum("btpk,btpi->btki", a, onehot)  # [B,T,K,m]
            mass = h.sum(-1, keepdim=True).clamp_min(1e-6)
            dist = h / mass
            ent = -(dist * dist.clamp_min(1e-8).log()).sum(-1)  # [B,T,K]
            bg_idx = id_map.get(0, -1)
            if bg_idx >= 0:
                fg_mass = h.sum(-1) - h[..., bg_idx]
            else:
                fg_mass = h.sum(-1)
            fg_share = fg_mass / mass.squeeze(-1).clamp_min(1e-6)
            purity_loss = (ent * fg_share).sum() / fg_share.sum().clamp_min(
                1e-6
            )
            with torch.no_grad():
                monitor_purity = (
                    dist.max(-1).values * fg_share
                ).sum() / fg_share.sum().clamp_min(1e-6)
            outputs["loss_unit_purity"] = purity_loss.detach()
            outputs["unit_purity_monitor"] = monitor_purity.detach()
            outputs["loss_instance_group"] = (
                outputs["loss_instance_group"]
                + self.unit_purity_weight * purity_loss
            )
        return outputs


    def _sic_token_descriptor(
        self,
        th: torch.Tensor,
        means: torch.Tensor,
        batch_size: int,
        token_count: int,
        n_gs: int,
        dense_features: torch.Tensor | None,
        data: dict,
        opt,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token descriptors for the SIC queries.

        Inputs: token hidden, normalized token anchor 3D position, frozen
        encoder patch features pooled per token, and DINO dense features
        pooled per token.  Returns [B,T,sic_dim] descriptors and the
        normalized token anchor positions [B,T,3].
        """
        p = self.gaussians_per_token
        anchor = means.mean(dim=2)  # [B,T,3]
        flat = means.reshape(batch_size, -1, 3).float()
        scene_center = flat.mean(dim=1, keepdim=True)  # [B,1,3]
        scene_scale = (
            (flat - scene_center)
            .square()
            .mean(dim=(1, 2), keepdim=True)
            .sqrt()
            .clamp_min(1e-3)
        )  # [B,1,1]
        pos_norm = (anchor - scene_center) / scene_scale  # [B,T,3]
        parts = [self.sic_pos_emb(pos_norm.float())]  # 64
        parts.append(self.sic_h_proj(th.float()))  # 64
        if self.sic_dense_proj is not None:
            if dense_features is not None:
                fused = self._gs_dense_features_cached(
                    means.reshape(batch_size, n_gs, 3),
                    dense_features,
                    data,
                    opt,
                    batch_size,
                    n_gs,
                )  # [B,N,dense_dim]
                tok = fused.view(batch_size, token_count, p, -1).mean(dim=2)
                parts.append(self.sic_dense_proj(tok))
            else:
                parts.append(
                    torch.zeros(
                        batch_size, token_count, 64, device=means.device
                    )
                )
        dino = self._dino_gs_raw(
            means.reshape(batch_size, n_gs, 3), data, batch_size, n_gs
        )  # [B,N,768]
        tok_dino = dino.view(batch_size, token_count, p, 768).mean(dim=2)
        parts.append(self.sic_dino_proj(tok_dino))
        desc = self.sic_desc_norm(torch.cat(parts, dim=-1))
        return desc, pos_norm

    def _sic_condition_unit_queries(
        self,
        q: torch.Tensor,
        th: torch.Tensor,
        means: torch.Tensor,
        batch_size: int,
        token_count: int,
        n_gs: int,
        dense_features: torch.Tensor | None,
        data: dict,
        opt,
    ) -> torch.Tensor:
        """Zero-gated SIC readout added to the local unit queries."""
        desc, pos_norm = self._sic_token_descriptor(
            th,
            means,
            batch_size,
            token_count,
            n_gs,
            dense_features,
            data,
            opt,
        )
        sic, seed_pos, _ = self.sic_module(desc, pos_norm)  # [B,M,Dsic]
        query_att = self.sic_q_proj(q.float())  # [B,T,K,Dsic]
        scale = float(self.sic_module.dim) ** -0.5
        logits = (
            torch.einsum("btkd,bmd->btkm", query_att, sic) * scale
        )
        # Distance-aware locality prior in normalized scene coordinates:
        # each token's units prefer SIC queries seeded near their own 3D
        # anchor, giving cross-token-consistent conditioning for spatially
        # adjacent units (the same-instance binding hypothesis).
        dist = (
            pos_norm[:, :, None, :] - seed_pos[:, None, :, :]
        ).square().sum(-1).unsqueeze(2)  # [B,T,1,M]
        att = F.softmax((logits - dist).float(), dim=-1)
        ctx = torch.einsum("btkm,bmd->btkd", att, sic)
        q_out = q + self.sic_gate * self.sic_readout(ctx)
        self._sic_queries = sic.detach()
        self._sic_seed_pos = seed_pos.detach()
        self._sic_att = att.detach()
        self._sic_usage_att = att.mean(dim=(0, 1, 2))  # [M] differentiable
        return q_out

    def _cross_token_embedding_stats(
        self,
        e: torch.Tensor,
        p_u: torch.Tensor,
        bg_idx: int,
        sample: int = 2048,
    ) -> dict:
        """Cross-token binding diagnostics on the unit identity embedding.

        Uses the (detached) soft pseudo-instance distribution p_u to bucket
        unit pairs into:
          - same instance, same token      (unit_same_tok_sim)
          - same instance, different token (unit_cross_tok_sim)
          - different foreground instances (unit_diff_sim)
        B0's real claim is that same-instance / different-token consistency
        improves (cross_tok_sim rises and the same/cross gap shrinks), not
        just the overall same/diff gap.
        """
        batch_size, unit_count, _ = e.shape
        device = e.device
        k = max(1, int(self.units_per_token))
        same_tok_sum = torch.zeros((), device=device)
        same_tok_cnt = torch.zeros((), device=device)
        cross_tok_sum = torch.zeros((), device=device)
        cross_tok_cnt = torch.zeros((), device=device)
        diff_sum = torch.zeros((), device=device)
        diff_cnt = torch.zeros((), device=device)
        with torch.no_grad():
            for b in range(batch_size):
                if unit_count <= sample:
                    idx = torch.arange(unit_count, device=device)
                else:
                    idx = torch.randperm(unit_count, device=device)[
                        :sample
                    ]
                eb = F.normalize(e[b][idx].float(), dim=-1)
                sim = eb @ eb.T  # [S,S]
                w = p_u[b][idx] @ p_u[b][idx].T
                tok = idx // k
                if bg_idx is not None and bg_idx >= 0:
                    fg = (1.0 - p_u[b][idx][:, bg_idx]) > 0.5
                else:
                    fg = torch.ones(
                        idx.numel(), dtype=torch.bool, device=device
                    )
                diag = torch.eye(
                    sim.shape[0], dtype=torch.bool, device=device
                )
                same_inst = (
                    (w > 0.5)
                    & (~diag)
                    & fg[:, None]
                    & fg[None, :]
                )
                same_tok_mask = same_inst & (
                    tok[:, None] == tok[None, :]
                )
                cross_tok_mask = same_inst & (
                    tok[:, None] != tok[None, :]
                )
                diff_mask = (
                    (w < 0.1)
                    & (~diag)
                    & fg[:, None]
                    & fg[None, :]
                )
                same_tok_sum = same_tok_sum + sim[same_tok_mask].sum()
                same_tok_cnt = same_tok_cnt + same_tok_mask.sum().float()
                cross_tok_sum = cross_tok_sum + sim[cross_tok_mask].sum()
                cross_tok_cnt = cross_tok_cnt + cross_tok_mask.sum().float()
                diff_sum = diff_sum + sim[diff_mask].sum()
                diff_cnt = diff_cnt + diff_mask.sum().float()
        eps = 1e-6
        same_tok = (same_tok_sum / same_tok_cnt.clamp_min(eps)).clamp(
            -1.0, 1.0
        )
        cross_tok = (cross_tok_sum / cross_tok_cnt.clamp_min(eps)).clamp(
            -1.0, 1.0
        )
        diff = (diff_sum / diff_cnt.clamp_min(eps)).clamp(-1.0, 1.0)
        return {
            "unit_same_tok_sim": same_tok.detach(),
            "unit_cross_tok_sim": cross_tok.detach(),
            "unit_diff_sim": diff.detach(),
            "unit_cross_tok_gap": (same_tok - cross_tok).detach(),
            "unit_cross_tok_pairs": cross_tok_cnt.detach(),
        }


class InstanceGroupHead(nn.Module):
    """Map 1024 decoder tokens to soft instance-group assignments.

    Each of the ``num_groups`` groups competes for every token through a
    softmax over groups plus a zero-valued void channel (InstOk3D-style):
    tokens whose affinity to every group is low fall to void, which
    implicitly discourages unused groups from acquiring anchors. The
    per-token probabilities are then expanded to the 64 Gaussians produced
    by that token, so the whole scene is covered by ``num_groups`` instance
    channels plus void that can be rendered into per-view probability maps.
    """

    def __init__(
        self,
        token_dim: int = 1024,
        num_groups: int = 64,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        hidden_dim = int(hidden_dim or token_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_groups),
        )

    def forward(
        self, token_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (group_probs [B,T,G+1], group_logits [B,T,G+1]).

        The final channel is the zero-valued void logit kept constant.
        """
        if token_hidden.ndim != 3:
            raise ValueError(
                f"token_hidden must be [B,T,D], got {tuple(token_hidden.shape)}"
            )
        logits = self.net(token_hidden.float())
        void_logits = torch.zeros(
            (logits.shape[0], logits.shape[1], 1),
            dtype=logits.dtype,
            device=logits.device,
        )
        logits = torch.cat([logits, void_logits], dim=-1)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities, logits


class _GroupCrossAttnLayer(nn.Module):
    """One group-query -> token cross-attention layer (InstOk3D-style)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_hidden: int = 2048,
        zero_init_output: bool = False,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )
        if zero_init_output:
            # Identity at init (attention + MLP outputs are zero), so a
            # warm-started model reproduces the checkpoint exactly while the
            # layer can still learn from the very first gradient step.
            with torch.no_grad():
                self.attn.out_proj.weight.zero_()
                self.attn.out_proj.bias.zero_()
                self.mlp[2].weight.zero_()
                self.mlp[2].bias.zero_()

    def forward(
        self, groups: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        attended, _ = self.attn(
            self.norm_q(groups), self.norm_kv(tokens), self.norm_kv(tokens)
        )
        groups = groups + attended
        groups = groups + self.mlp(self.norm_out(groups))
        return groups


class InstanceCountHead(nn.Module):
    """Predict the number of active instance groups for the current scene.

    Mean-pools the anchor token hidden states (optionally concatenated with
    the per-token 3D anchor-position features) and regresses log(count).
    At inference the predicted count prunes the least-used groups, making
    the group budget scene-adaptive (InstOk3D uses per-scene object counts
    to size its group queries; the fixed 128-group budget mismatches scenes
    with 20~134 GT instances).
    """

    def __init__(
        self,
        input_dim: int = 1088,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        token_hidden: torch.Tensor,
        pos_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [B] log-count predictions (unbounded scalars)."""
        features = token_hidden.float()
        if pos_feat is not None:
            features = torch.cat([features, pos_feat.float()], dim=-1)
        pooled = features.mean(dim=1)  # [B,D]
        return self.net(pooled).squeeze(-1)  # [B]


class InstanceGroupDecoder(nn.Module):
    """Scene-adaptive anchor-to-group decoder (InstOk3D-style).

    Learnable group tokens cross-attend to the 1024 anchor/token hidden
    states, so each group query can adapt to the instances present in the
    current scene instead of collapsing onto a few global coarse groups.
    Assignment logits are dot-product similarities between refined group
    queries and token features, followed by a zero-valued void channel.

    Returns (group_probs [B,T,G+1], group_logits [B,T,G+1]) -- the same
    interface as ``InstanceGroupHead``.
    """

    def __init__(
        self,
        token_dim: int = 1024,
        num_groups: int = 64,
        num_layers: int = 2,
        num_heads: int = 16,
        mlp_hidden: int = 2048,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.group_tokens = nn.Parameter(
            torch.randn(self.num_groups, int(token_dim))
            * (int(token_dim) ** -0.5)
        )
        self.layers = nn.ModuleList(
            [
                _GroupCrossAttnLayer(
                    int(token_dim), num_heads=num_heads, mlp_hidden=mlp_hidden
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(int(token_dim))
        self.scale = float(token_dim) ** -0.5

    def forward(
        self, token_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_hidden.ndim != 3:
            raise ValueError(
                f"token_hidden must be [B,T,D], got {tuple(token_hidden.shape)}"
            )
        batch_size, token_count, dim = token_hidden.shape
        groups = self.group_tokens.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        for layer in self.layers:
            groups = layer(groups, token_hidden)
        groups = self.norm(groups)
        logits = (
            torch.einsum("bgd,btd->bgt", groups, token_hidden) * self.scale
        )  # [B,G,T]
        void_logits = torch.zeros(
            (logits.shape[0], 1, logits.shape[2]),
            dtype=logits.dtype,
            device=logits.device,
        )
        logits = torch.cat([logits, void_logits], dim=1).transpose(
            1, 2
        )  # [B,T,G+1]
        probabilities = F.softmax(logits.float(), dim=-1)
        return probabilities, logits


class GroupTokenPerGaussianHead(InstanceGroupDecoder):
    """InstOk3D-style group tokens plus a per-Gaussian refinement.

    The base ``InstanceGroupDecoder`` produces the token-level assignment
    (global instance proposal): L learnable group tokens cross-attend to the
    GS-token features and yield per-token logits over L+1 (L groups + a zero
    void channel). Every Gaussian then refines that proposal with its own
    local evidence -- the token-prior logits, its position relative to the
    token anchor, and its Gaussian parameters -- through a zero-initialized
    residual. At init the refined logits equal the token prior exactly, so
    the model starts at the token-level baseline and can sharpen instance
    boundaries per Gaussian during training.

    Returns (group_probs [B,N,G+1], group_logits [B,N,G+1]) -- Gaussian-level
    probabilities, consumed directly by the existing rendering path.
    """

    def __init__(
        self,
        token_dim: int = 1024,
        num_groups: int = 100,
        num_layers: int = 2,
        num_heads: int = 16,
        mlp_hidden: int = 2048,
        num_gaussians_per_token: int = 64,
        local_hidden: int = 256,
        refine_scale_init: float = 0.05,
        refine_scale_max: float = 0.3,
    ):
        super().__init__(
            token_dim=token_dim,
            num_groups=num_groups,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_hidden=mlp_hidden,
        )
        self.num_gaussians_per_token = int(num_gaussians_per_token)
        # Local evidence: token-prior logits (G+1) + relative position (3) +
        # Gaussian parameters (14: xyz, opacity, scaling, rotation, rgb).
        local_dim = int(num_groups) + 1 + 3 + 14
        self.local_in = nn.Sequential(
            nn.LayerNorm(local_dim),
            nn.Linear(local_dim, int(local_hidden)),
            nn.GELU(),
            nn.Linear(int(local_hidden), int(local_hidden)),
        )
        self.refine_proj = nn.Linear(int(local_hidden), int(num_groups) + 1)
        # Zero-initialized residual: init reproduces the token-level prior.
        nn.init.zeros_(self.refine_proj.weight)
        nn.init.zeros_(self.refine_proj.bias)
        # Small bounded refinement scale. The token-prior logits are already
        # tiny (dot product scaled by dim^-0.5), so an unbounded per-GS
        # residual re-ranks every Gaussian and fragments the rendered masks
        # (each group then wins scattered pixels -> ~8x predicted instances).
        # Keeping the residual small and zero-centered makes it a local
        # boundary correction around the coherent token proposal instead.
        self.refine_scale_max = float(refine_scale_max)
        self.log_refine_scale = nn.Parameter(
            torch.tensor(float(refine_scale_init)).log()
        )

    def forward(
        self,
        token_hidden: torch.Tensor,
        gaussians: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_hidden.ndim != 3:
            raise ValueError(
                f"token_hidden must be [B,T,D], got {tuple(token_hidden.shape)}"
            )
        batch_size, token_count, _ = token_hidden.shape
        expected_gaussians = token_count * self.num_gaussians_per_token
        if gaussians.shape[:2] != (batch_size, expected_gaussians):
            raise ValueError(
                "gaussians must be [B,T*P,14], got "
                f"{tuple(gaussians.shape)} for T={token_count}, "
                f"P={self.num_gaussians_per_token}"
            )
        # Token-level proposal from the base group-token decoder.
        _, token_logits = super().forward(token_hidden)  # [B,T,G+1]
        prior_logits = token_logits.repeat_interleave(
            self.num_gaussians_per_token, dim=1
        )  # [B,N,G+1]

        # Per-Gaussian local evidence.
        means = gaussians[..., :3].float().view(
            batch_size, token_count, self.num_gaussians_per_token, 3
        )
        anchor = means.mean(dim=2, keepdim=True)  # [B,T,1,3]
        scene_center = means.mean(dim=1, keepdim=True)  # [B,1,1,3]
        scene_scale = (
            (means - scene_center).square().sum(dim=-1)
            .mean(dim=(1, 2)).sqrt().clamp_min(1e-3)
        )  # [B]
        rel_pos = (
            (means - anchor)
            / scene_scale.view(batch_size, 1, 1, 1)
        ).reshape(batch_size, expected_gaussians, 3)
        local = torch.cat(
            [
                prior_logits.float(),
                rel_pos,
                gaussians[..., :14].float(),
            ],
            dim=-1,
        )
        local_features = self.local_in(local)
        refine_logits = self.refine_proj(local_features)
        # Zero-center the residual across groups so the prior's overall
        # distribution (and its confident argmax) is preserved; only the
        # relative ranking of near-tied groups can change.
        residual = refine_logits - refine_logits.mean(dim=-1, keepdim=True)
        refine_scale = self.log_refine_scale.exp().clamp(
            0.01, self.refine_scale_max
        )
        final_logits = prior_logits + refine_scale * residual
        probabilities = F.softmax(final_logits.float(), dim=-1)
        return probabilities, final_logits


class PerGaussianInstanceGroupHead(nn.Module):
    """Per-Gaussian instance grouping head.

    InstanceSplat-style granularity with InstOk3D-style assignment: the
    ``P*P`` Gaussians decoded by each anchor token get their own low-dim
    instance feature, so a token whose Gaussians straddle an object
    boundary can split them across groups instead of forcing the whole
    token into one instance. Scene-adaptive group queries are still formed
    by cross-attending learnable group tokens to the anchor tokens, then
    each Gaussian is soft-assigned over ``num_groups`` groups plus a
    zero-valued void channel.

    Returns (group_probs [B,N,G+1], group_logits [B,N,G+1]) where N is the
    total number of Gaussians (``num_tokens * num_gaussians_per_token``).
    """

    def __init__(
        self,
        token_dim: int = 1024,
        num_groups: int = 64,
        feature_dim: int = 16,
        num_layers: int = 2,
        num_heads: int = 16,
        mlp_hidden: int = 2048,
        use_anchor_pos: bool = True,
        num_gaussians_per_token: int = 64,
        pos_feat_dim: int = 64,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.num_gaussians_per_token = int(num_gaussians_per_token)
        self.use_anchor_pos = bool(use_anchor_pos)

        self.instance_feature_head = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), 512),
            nn.GELU(),
            nn.Linear(
                512,
                int(feature_dim) * self.num_gaussians_per_token,
            ),
        )
        self.pos_feat_dim = int(pos_feat_dim)
        if self.use_anchor_pos:
            self.anchor_pos_encoder = nn.Sequential(
                nn.Linear(3, int(pos_feat_dim)),
                nn.GELU(),
                nn.Linear(int(pos_feat_dim), int(pos_feat_dim)),
            )

        # Group queries live in the high-dim token space for cross-attention,
        # then are projected to the (smaller) assignment feature space.
        self.group_tokens = nn.Parameter(
            torch.randn(self.num_groups, int(token_dim))
            * (int(token_dim) ** -0.5)
        )
        self.layers = nn.ModuleList(
            [
                _GroupCrossAttnLayer(
                    int(token_dim), num_heads=num_heads, mlp_hidden=mlp_hidden
                )
                for _ in range(int(num_layers))
            ]
        )
        assign_dim = int(feature_dim) + (
            int(pos_feat_dim) if self.use_anchor_pos else 0
        )
        self.group_proj = nn.Linear(int(token_dim), assign_dim)
        self.scale = float(assign_dim) ** -0.5
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            # Small final projection: assignment starts near uniform so the
            # Hungarian matching sees a mild optimization landscape and the
            # void channel can win background pixels early.
            last = self.instance_feature_head[-1]
            last.weight.mul_(0.1)
            last.bias.zero_()
            self.group_proj.weight.mul_(0.1)
            self.group_proj.bias.zero_()

    def forward(
        self, token_hidden: torch.Tensor, gaussians: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_hidden.ndim != 3:
            raise ValueError(
                f"token_hidden must be [B,T,D], got {tuple(token_hidden.shape)}"
            )
        batch_size, token_count, _ = token_hidden.shape
        features = self.instance_feature_head(token_hidden.float())
        features = features.view(
            batch_size,
            token_count * self.num_gaussians_per_token,
            -1,
        )  # [B,N,F]
        if self.use_anchor_pos:
            pos = gaussians[..., :3].float()  # [B,N,3]
            features = torch.cat(
                [features, self.anchor_pos_encoder(pos)], dim=-1
            )

        groups = self.group_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            groups = layer(groups, token_hidden)
        groups = self.group_proj(groups)  # [B,L,assign_dim]

        logits = (
            torch.einsum("bnd,bld->bnl", features, groups) * self.scale
        )  # [B,N,G]
        void_logits = torch.zeros(
            (logits.shape[0], logits.shape[1], 1),
            dtype=logits.dtype,
            device=logits.device,
        )
        logits = torch.cat([logits, void_logits], dim=-1)  # [B,N,G+1]
        probabilities = F.softmax(logits.float(), dim=-1)
        return probabilities, logits


class PerGaussianResidualHead(nn.Module):
    """Per-Gaussian refinement on top of a warm-started token-level decoder.

    The token-level base path is *identical* to ``InstanceGroupDecoder``
    (same attribute names, same 1088-dim input formed by concatenating the
    token hidden states with the v4 anchor-position encoder output), so the
    ``instance_group_head.*`` weights of the wide7l checkpoints load
    directly and the model reproduces wide7l's assignment at init. A small
    per-Gaussian residual (zero-init-ish, scaled by ``residual_scale``) can
    then flip individual Gaussians across groups, giving the model the
    capacity to split a token that straddles an object boundary without
    letting the whole scene fragment (the failure mode of the standalone
    per-Gaussian head).

    Returns (group_probs [B,N,G+1], group_logits [B,N,G+1]).
    """

    def __init__(
        self,
        token_dim: int = 1024,
        num_groups: int = 64,
        feature_dim: int = 16,
        num_layers: int = 2,
        num_heads: int = 16,
        mlp_hidden: int = 2048,
        use_anchor_pos: bool = True,
        num_gaussians_per_token: int = 64,
        pos_feat_dim: int = 64,
        residual_scale: float = 0.3,
        pos_attn_layers: int = 0,
        pos_attn_scale: float = 1.0,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.num_gaussians_per_token = int(num_gaussians_per_token)
        self.use_anchor_pos = bool(use_anchor_pos)
        self.residual_scale = float(residual_scale)
        self.pos_attn_layers = int(pos_attn_layers)
        self.pos_attn_scale = float(pos_attn_scale)

        # --- Warm-started token-level base (InstanceGroupDecoder layout) ---
        base_dim = int(token_dim) + int(pos_feat_dim)
        self.group_tokens = nn.Parameter(
            torch.randn(self.num_groups, base_dim) * (base_dim ** -0.5)
        )
        self.layers = nn.ModuleList(
            [
                _GroupCrossAttnLayer(
                    base_dim, num_heads=num_heads, mlp_hidden=mlp_hidden
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(base_dim)
        self.scale = float(base_dim) ** -0.5

        # --- Per-Gaussian residual refinement (fresh, small init) ---
        self.refine_feature_head = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), 512),
            nn.GELU(),
            nn.Linear(
                512,
                int(feature_dim) * self.num_gaussians_per_token,
            ),
        )
        self.pos_feat_dim = int(pos_feat_dim)
        if self.use_anchor_pos:
            self.refine_pos_encoder = nn.Sequential(
                nn.Linear(3, int(pos_feat_dim)),
                nn.GELU(),
                nn.Linear(int(pos_feat_dim), int(pos_feat_dim)),
            )
        assign_dim = int(feature_dim) + (
            int(pos_feat_dim) if self.use_anchor_pos else 0
        )
        self.refine_group_proj = nn.Linear(base_dim, assign_dim)
        self.refine_scale = float(assign_dim) ** -0.5
        # --- InstOk3D-style position-aware group refinement (fresh) ---
        if self.pos_attn_layers > 0 and self.use_anchor_pos:
            self.pos_attn_encoder = nn.Sequential(
                nn.Linear(3, int(pos_feat_dim)),
                nn.GELU(),
                nn.Linear(int(pos_feat_dim), int(pos_feat_dim)),
            )
            self.pos_attn_proj = nn.Linear(int(pos_feat_dim), base_dim)
            self.pos_attn_layers_mod = nn.ModuleList(
                [
                    _GroupCrossAttnLayer(
                        base_dim,
                        num_heads=num_heads,
                        mlp_hidden=mlp_hidden,
                        zero_init_output=True,
                    )
                    for _ in range(int(pos_attn_layers))
                ]
            )
        else:
            self.pos_attn_layers = 0
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            # Small final projection: the residual starts near zero so the
            # first training steps reproduce the warm-started base decoder.
            last = self.refine_feature_head[-1]
            last.weight.mul_(0.01)
            last.bias.zero_()
            self.refine_group_proj.weight.mul_(0.01)
            self.refine_group_proj.bias.zero_()

    def forward(
        self,
        token_hidden: torch.Tensor,
        gaussians: torch.Tensor,
        base_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_hidden.ndim != 3:
            raise ValueError(
                f"token_hidden must be [B,T,D], got {tuple(token_hidden.shape)}"
            )
        if base_input is None:
            raise ValueError(
                "PerGaussianResidualHead requires base_input (token hidden "
                "concatenated with the v4 anchor-position features)"
            )
        batch_size, token_count, _ = token_hidden.shape
        num_gaussians = token_count * self.num_gaussians_per_token

        # Token-level base logits (same computation as InstanceGroupDecoder).
        groups = self.group_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            groups = layer(groups, base_input)
        groups = self.norm(groups)

        # Position-aware refinement: group queries cross-attend to the
        # per-token 3D anchor positions (mean of the token's Gaussian
        # centers), added as a gated residual so init stays exact.
        if self.pos_attn_layers > 0:
            means = gaussians[..., :3].float().view(
                batch_size,
                token_count,
                self.num_gaussians_per_token,
                3,
            ).mean(dim=2)  # [B,T,3]
            pos_feat_tok = self.pos_attn_encoder(means)  # [B,T,pos_dim]
            pos_kv = self.pos_attn_proj(pos_feat_tok)  # [B,T,base_dim]
            refined = groups
            for layer in self.pos_attn_layers_mod:
                refined = layer(refined, pos_kv)
            # Each pos-attn layer is identity at init (zero-init outputs),
            # so the warm-started base is preserved; pos_attn_scale controls
            # how strongly the learned position refinement displaces groups.
            groups = groups + self.pos_attn_scale * (refined - groups)

        base_logits = (
            torch.einsum("bgd,btd->bgt", groups, base_input) * self.scale
        )  # [B,G,T]
        base_logits = base_logits.transpose(1, 2)  # [B,T,G]

        # Per-Gaussian residual logits.
        features = self.refine_feature_head(token_hidden.float())
        features = features.view(batch_size, num_gaussians, -1)  # [B,N,F]
        if self.use_anchor_pos:
            pos = gaussians[..., :3].float()  # [B,N,3]
            features = torch.cat(
                [features, self.refine_pos_encoder(pos)], dim=-1
            )
        refine_queries = self.refine_group_proj(groups)  # [B,G,assign_dim]
        residual = (
            torch.einsum(
                "bnd,bgd->bng", features, refine_queries
            )
            * self.refine_scale
        )  # [B,N,G]

        base_rep = base_logits.repeat_interleave(
            self.num_gaussians_per_token, dim=1
        )  # [B,N,G]
        group_logits = base_rep + self.residual_scale * residual
        void_logits = torch.zeros(
            (batch_size, num_gaussians, 1),
            dtype=group_logits.dtype,
            device=group_logits.device,
        )
        group_logits = torch.cat([group_logits, void_logits], dim=-1)
        probabilities = F.softmax(group_logits.float(), dim=-1)
        return probabilities, group_logits


def _project_dense_features(
    xyz_world: torch.Tensor,
    features: torch.Tensor,
    source_c2w: torch.Tensor,
    source_intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
    chunk_size: int = 8192,
    znear: float = 0.025,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable projection of per-view dense features onto Gaussian
    centers (same geometry as ``SourceFeatureProjector`` but keeps gradients
    flowing into the dense instance decoder).

    Args:
        xyz_world: [B,N,3] Gaussian centers.
        features: [B,V,F,Hf,Wf] dense instance features.
        source_c2w: [B,V,4,4] camera-to-world of the source views.
        source_intrinsics: [B,V,4] (fx, fy, cx, cy) at ``image_hw``.
        image_hw: (H, W) of the source images (normalization frame).

    Returns:
        fused [B,N,F] (valid-view mean), has_source [B,N,1] bool.
    """
    batch, num_points, _ = xyz_world.shape
    num_views = features.shape[1]
    feat_dim = features.shape[2]
    feat_h, feat_w = features.shape[-2:]
    img_h, img_w = image_hw

    w2c = torch.linalg.inv(source_c2w.float())
    feature_flat = features.reshape(
        batch * num_views, feat_dim, feat_h, feat_w
    ).float()

    fused_chunks: list[torch.Tensor] = []
    valid_chunks: list[torch.Tensor] = []
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
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
        xyz_cam_h = torch.einsum("bvij,bnj->bvni", w2c, xyz_h)
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
            (z > znear)
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
            .reshape(batch, num_views, chunk_n, feat_dim)
        )
        weights = valid.float()
        denominator = weights.sum(dim=1, keepdim=False).unsqueeze(-1)
        fused = (
            sampled * weights[..., None]
        ).sum(dim=1) / denominator.clamp_min(1.0)
        fused_chunks.append(fused)
        valid_chunks.append(denominator > 0)

    return (
        torch.cat(fused_chunks, dim=1),
        torch.cat(valid_chunks, dim=1),
    )


class DenseInstanceResidualHead(PerGaussianResidualHead):
    """Per-Gaussian residual head plus a dense image-evidence residual.

    The base path is *identical* to ``PerGaussianResidualHead`` (same
    attribute names, same computation), so the pgr3df2 checkpoints load
    directly and the model reproduces AP50 0.231 at init. On top of it, a
    small trainable decoder maps the frozen TokenGS encoder's per-patch
    features (``[B, V*P, C]``, multi-scale optional) into per-location
    instance features which are projected onto the Gaussian centers through
    the source cameras and fused across views. Each Gaussian therefore gets
    its own *image evidence* (not just token-derived content + position),
    so a token straddling an object boundary can split its 64 Gaussians
    based on what the pixels actually say. The dense residual is
    small-initialized (``dense_scale``) so warm starts stay exact.

    Returns (group_probs [B,N,G+1], group_logits [B,N,G+1],
    dense_patch_features [B,V,F,Hf,Wf] or None).
    """

    def __init__(
        self,
        token_dim: int = 1024,
        num_groups: int = 64,
        feature_dim: int = 16,
        num_layers: int = 2,
        num_heads: int = 16,
        mlp_hidden: int = 2048,
        use_anchor_pos: bool = True,
        num_gaussians_per_token: int = 64,
        pos_feat_dim: int = 64,
        residual_scale: float = 0.3,
        pos_attn_layers: int = 0,
        pos_attn_scale: float = 1.0,
        dense_feature_dim: int = 16,
        dense_upsample: int = 2,
        dense_multiscale: bool = True,
        dense_scale: float = 0.3,
        dense_gate: bool = True,
    ):
        super().__init__(
            token_dim=token_dim,
            num_groups=num_groups,
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_hidden=mlp_hidden,
            use_anchor_pos=use_anchor_pos,
            num_gaussians_per_token=num_gaussians_per_token,
            pos_feat_dim=pos_feat_dim,
            residual_scale=residual_scale,
            pos_attn_layers=pos_attn_layers,
            pos_attn_scale=pos_attn_scale,
        )
        self.dense_feature_dim = int(dense_feature_dim)
        self.dense_upsample = int(dense_upsample)
        self.dense_multiscale = bool(dense_multiscale)
        self.dense_scale = float(dense_scale)
        self.dense_gate = bool(dense_gate)
        enc_dim = int(token_dim) * (2 if self.dense_multiscale else 1)
        self.dense_net = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, 512),
            nn.GELU(),
            nn.Linear(
                512,
                int(dense_feature_dim) * int(dense_upsample) ** 2,
            ),
        )
        if self.dense_gate:
            # Per-Gaussian gate: predicts how much the dense image evidence
            # may displace the warm-started base assignment for this
            # Gaussian. Initialized near zero (bias -5 -> sigmoid ~0.007) so
            # training starts exactly at the base checkpoint; the patch-level
            # aux loss trains dense_net meanwhile, and the gate only grows
            # where the evidence actually helps.
            self.dense_gate_net = nn.Linear(
                int(dense_feature_dim), 1
            )
        self._init_dense_weights()

    def _init_dense_weights(self):
        with torch.no_grad():
            # Small final projection: the dense residual starts near zero so
            # the first training steps reproduce the warm-started base.
            last = self.dense_net[-1]
            last.weight.mul_(0.01)
            last.bias.zero_()
            if self.dense_gate:
                gate = self.dense_gate_net
                gate.weight.mul_(0.01)
                gate.bias.fill_(-5.0)

    def dense_patch_features(
        self,
        dense_features: torch.Tensor,
        num_views: int,
    ) -> torch.Tensor:
        """Map cached encoder patch features to per-location instance
        features [B,V,Fd,Hf*U,Wf*U].

        ``dense_features`` is [B, V*P, C] (optionally multi-scale
        concatenated). Each patch (8x8 input pixels) decodes into a UxU
        block of instance feature locations, i.e. the dense field is at
        ``patch_size // dense_upsample`` input resolution.
        """
        batch, seq_len, _ = dense_features.shape
        patches_per_view = seq_len // int(num_views)
        spatial = int(round(patches_per_view ** 0.5))
        if spatial * spatial != patches_per_view:
            raise ValueError(
                f"dense patch features are not square per view: "
                f"{patches_per_view} patches/view"
            )
        feats = self.dense_net(dense_features.float())
        feats = feats.view(
            batch,
            int(num_views),
            spatial,
            spatial,
            self.dense_upsample,
            self.dense_upsample,
            self.dense_feature_dim,
        )
        feats = feats.permute(0, 1, 6, 2, 4, 3, 5)
        return feats.reshape(
            batch,
            int(num_views),
            self.dense_feature_dim,
            spatial * self.dense_upsample,
            spatial * self.dense_upsample,
        )

    def forward(
        self,
        token_hidden: torch.Tensor,
        gaussians: torch.Tensor,
        base_input: torch.Tensor | None = None,
        dense_features: torch.Tensor | None = None,
        source_c2w: torch.Tensor | None = None,
        source_intrinsics: torch.Tensor | None = None,
        image_hw: tuple[int, int] | None = None,
        num_views: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if dense_features is None:
            probabilities, logits = super().forward(
                token_hidden, gaussians, base_input=base_input
            )
            return probabilities, logits, None
        if (
            base_input is None
            or source_c2w is None
            or source_intrinsics is None
            or image_hw is None
            or num_views is None
        ):
            raise ValueError(
                "DenseInstanceResidualHead requires base_input, "
                "source_c2w, source_intrinsics, image_hw and num_views"
            )
        batch_size, token_count, _ = token_hidden.shape
        num_gaussians = token_count * self.num_gaussians_per_token

        # --- Token-level base logits (same as PerGaussianResidualHead) ---
        groups = self.group_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            groups = layer(groups, base_input)
        groups = self.norm(groups)
        base_logits = (
            torch.einsum("bgd,btd->bgt", groups, base_input) * self.scale
        )  # [B,G,T]
        base_logits = base_logits.transpose(1, 2)  # [B,T,G]

        # --- Token + position residual (same as PerGaussianResidualHead) ---
        features = self.refine_feature_head(token_hidden.float())
        features = features.view(batch_size, num_gaussians, -1)  # [B,N,F]
        pos = gaussians[..., :3].float()
        if self.use_anchor_pos:
            features = torch.cat(
                [features, self.refine_pos_encoder(pos)], dim=-1
            )
        refine_queries = self.refine_group_proj(groups)  # [B,G,assign_dim]
        residual = (
            torch.einsum("bnd,bgd->bng", features, refine_queries)
            * self.refine_scale
        )  # [B,N,G]

        # --- Dense image-evidence residual ---
        patch_feats = self.dense_patch_features(dense_features, num_views)
        fused, has_source = _project_dense_features(
            pos,
            patch_feats,
            source_c2w,
            source_intrinsics,
            image_hw,
        )
        dense_gs = fused
        if self.use_anchor_pos:
            dense_gs = torch.cat([dense_gs, self.refine_pos_encoder(pos)], dim=-1)
        residual_dense = (
            torch.einsum("bnd,bgd->bng", dense_gs, refine_queries)
            * self.refine_scale
        )  # [B,N,G]
        if self.dense_gate:
            gate_logit = self.dense_gate_net(fused)  # [B,N,1]
            gate = torch.sigmoid(gate_logit) * has_source.float()
        else:
            gate = has_source.float()
        residual_dense = residual_dense * gate

        base_rep = base_logits.repeat_interleave(
            self.num_gaussians_per_token, dim=1
        )  # [B,N,G]
        group_logits = (
            base_rep
            + self.residual_scale * residual
            + self.dense_scale * residual_dense
        )
        void_logits = torch.zeros(
            (batch_size, num_gaussians, 1),
            dtype=group_logits.dtype,
            device=group_logits.device,
        )
        group_logits = torch.cat([group_logits, void_logits], dim=-1)
        probabilities = F.softmax(group_logits.float(), dim=-1)
        return probabilities, group_logits, patch_feats
