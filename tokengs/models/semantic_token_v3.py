"""Token-compressed open-vocabulary semantic field (V3, token_decoder_lowrank).

Ported from the tokengs_c3g experiment that reached ~0.536 ScanNet C3G8 mIoU
(``re10k_semantic_lseg_v3_token_r16``). The module and parameter names are
kept identical so the pretrained ``semantic_head.*`` weights load directly
into this class.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticTokenDecoderLayer(nn.Module):
    """TokenGS-style semantic decoder layer.

    The semantic tokens first cross-attend to source LSeg patch tokens,
    then communicate through token self-attention, followed by an FFN.
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        self.query_cross_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        hidden_dim = int(dim * mlp_ratio)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        semantic_tokens: torch.Tensor,
        source_memory: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_cross_norm(semantic_tokens)
        memory = self.memory_norm(source_memory)

        cross_output, _ = self.cross_attn(
            query=query,
            key=memory,
            value=memory,
            need_weights=False,
        )
        semantic_tokens = semantic_tokens + cross_output

        self_query = self.self_norm(semantic_tokens)
        self_output, _ = self.self_attn(
            query=self_query,
            key=self_query,
            value=self_query,
            need_weights=False,
        )
        semantic_tokens = semantic_tokens + self_output

        semantic_tokens = semantic_tokens + self.ffn(
            self.ffn_norm(semantic_tokens)
        )
        return semantic_tokens


class TokenSemanticFieldV3(nn.Module):
    """Hierarchical token-compressed open-vocabulary semantic field.

    Each geometry token is paired with one semantic token. The semantic token
    owns one 512-D base feature shared by its local Gaussians. Each local
    Gaussian stores only a low-rank residual code.
    """

    def __init__(
        self,
        num_tokens: int = 1024,
        geometry_token_dim: int = 1024,
        feature_dim: int = 512,
        num_gaussians_per_token: int = 64,
        semantic_patch_size: int = 8,
        num_decoder_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        local_rank: int = 16,
        local_hidden_dim: int = 128,
        max_views: int = 8,
        token_residual_scale: float = 0.1,
        local_residual_scale: float = 0.1,
        pool_use_opacity: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_tokens = int(num_tokens)
        self.geometry_token_dim = int(geometry_token_dim)
        self.feature_dim = int(feature_dim)
        self.num_gaussians_per_token = int(
            num_gaussians_per_token
        )
        self.semantic_patch_size = int(semantic_patch_size)
        self.local_rank = int(local_rank)
        self.local_hidden_dim = int(local_hidden_dim)
        self.max_views = int(max_views)
        self.token_residual_scale = float(token_residual_scale)
        self.local_residual_scale = float(local_residual_scale)
        self.pool_use_opacity = bool(pool_use_opacity)

        if self.local_rank <= 0:
            raise ValueError(
                f"local_rank must be positive, got {self.local_rank}"
            )

        self.semantic_tokens = nn.Parameter(
            torch.empty(self.num_tokens, self.feature_dim)
        )
        nn.init.normal_(self.semantic_tokens, std=0.02)

        self.geometry_to_query = nn.Sequential(
            nn.LayerNorm(self.geometry_token_dim),
            nn.Linear(self.geometry_token_dim, self.feature_dim),
        )

        self.geometry_fallback = nn.Sequential(
            nn.LayerNorm(self.geometry_token_dim),
            nn.Linear(self.geometry_token_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        self.evidence_to_query = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        # LSeg features are already semantic embeddings. Use average pooling
        # for patch reduction, followed by a lightweight 1x1 projection.
        self.semantic_patch_embed = nn.Conv2d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            kernel_size=1,
            stride=1,
        )
        self.view_embedding = nn.Embedding(
            self.max_views,
            self.feature_dim,
        )
        nn.init.normal_(self.view_embedding.weight, std=0.02)

        self.decoder_layers = nn.ModuleList(
            [
                SemanticTokenDecoderLayer(
                    dim=self.feature_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(int(num_decoder_layers))
            ]
        )
        self.decoder_norm = nn.LayerNorm(self.feature_dim)

        self.token_residual_head = nn.Linear(
            self.feature_dim,
            self.feature_dim,
        )
        nn.init.normal_(
            self.token_residual_head.weight,
            std=1e-3,
        )
        nn.init.zeros_(self.token_residual_head.bias)

        self.semantic_local = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.local_hidden_dim),
        )
        self.gaussian_local = nn.Sequential(
            nn.Linear(14, self.local_hidden_dim),
            nn.LayerNorm(self.local_hidden_dim),
            nn.GELU(),
        )
        self.projected_delta_local = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.local_hidden_dim),
            nn.GELU(),
        )

        self.local_code_head = nn.Sequential(
            nn.Linear(
                self.local_hidden_dim * 3 + 1,
                self.local_hidden_dim * 2,
            ),
            nn.GELU(),
            nn.Linear(
                self.local_hidden_dim * 2,
                self.local_rank,
            ),
        )
        nn.init.normal_(
            self.local_code_head[-1].weight,
            std=1e-3,
        )
        nn.init.zeros_(self.local_code_head[-1].bias)

        self.local_basis = nn.Parameter(
            torch.empty(self.local_rank, self.feature_dim)
        )
        nn.init.normal_(self.local_basis, std=0.02)

    @staticmethod
    def _sincos_2d_position(
        height: int,
        width: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dim % 4 != 0:
            raise ValueError(
                f"2D sin-cos position requires dim % 4 == 0, got {dim}"
            )

        quarter_dim = dim // 4
        omega = torch.arange(
            quarter_dim,
            device=device,
            dtype=torch.float32,
        )
        omega = 1.0 / (
            10000.0 ** (omega / max(quarter_dim - 1, 1))
        )

        y = torch.linspace(
            0.0,
            2.0 * math.pi,
            height,
            device=device,
            dtype=torch.float32,
        )
        x = torch.linspace(
            0.0,
            2.0 * math.pi,
            width,
            device=device,
            dtype=torch.float32,
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        x_phase = xx.reshape(-1, 1) * omega.reshape(1, -1)
        y_phase = yy.reshape(-1, 1) * omega.reshape(1, -1)

        position = torch.cat(
            [
                x_phase.sin(),
                x_phase.cos(),
                y_phase.sin(),
                y_phase.cos(),
            ],
            dim=-1,
        )
        return position.unsqueeze(0).to(dtype=dtype)

    def _build_source_memory(
        self,
        source_features: torch.Tensor,
    ) -> torch.Tensor:
        if source_features.ndim != 5:
            raise ValueError(
                "source_features must be [B,V,C,H,W], "
                f"got {tuple(source_features.shape)}"
            )

        B, V, C, H, W = source_features.shape
        if C != self.feature_dim:
            raise ValueError(
                f"Expected source feature dim {self.feature_dim}, got {C}"
            )
        if V > self.max_views:
            raise ValueError(
                f"V={V} exceeds max_views={self.max_views}"
            )
        if H % self.semantic_patch_size != 0:
            raise ValueError(
                f"LSeg feature height {H} must be divisible by "
                f"semantic_patch_size={self.semantic_patch_size}"
            )
        if W % self.semantic_patch_size != 0:
            raise ValueError(
                f"LSeg feature width {W} must be divisible by "
                f"semantic_patch_size={self.semantic_patch_size}"
            )

        source_flat = source_features.reshape(B * V, C, H, W)
        memory_2d = F.avg_pool2d(
            source_flat,
            kernel_size=self.semantic_patch_size,
            stride=self.semantic_patch_size,
        )
        memory_2d = self.semantic_patch_embed(memory_2d)
        Hp, Wp = memory_2d.shape[-2:]

        memory = memory_2d.flatten(2).transpose(1, 2)
        memory = memory.reshape(B, V, Hp * Wp, self.feature_dim)

        position = self._sincos_2d_position(
            height=Hp,
            width=Wp,
            dim=self.feature_dim,
            device=memory.device,
            dtype=memory.dtype,
        ).reshape(1, 1, Hp * Wp, self.feature_dim)

        view_ids = torch.arange(V, device=memory.device)
        view_bias = self.view_embedding(view_ids).reshape(
            1,
            V,
            1,
            self.feature_dim,
        )

        memory = memory + position + view_bias
        return memory.reshape(B, V * Hp * Wp, self.feature_dim)

    def _pool_projected_features(
        self,
        projected_features: torch.Tensor,
        has_source: torch.Tensor,
        confidence: torch.Tensor,
        gaussians: torch.Tensor,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N_gaussian, C = projected_features.shape
        if C != self.feature_dim:
            raise ValueError(
                f"projected feature dim mismatch: {C}"
            )
        if N_gaussian % num_tokens != 0:
            raise ValueError(
                f"N_gaussian={N_gaussian} is not divisible by "
                f"num_tokens={num_tokens}"
            )

        gs_per_token = N_gaussian // num_tokens
        if gs_per_token != self.num_gaussians_per_token:
            raise ValueError(
                f"Expected {self.num_gaussians_per_token} Gaussians/token, "
                f"got {gs_per_token}"
            )

        projected = projected_features.reshape(
            B,
            num_tokens,
            gs_per_token,
            C,
        )
        valid = has_source.reshape(
            B,
            num_tokens,
            gs_per_token,
            1,
        ).float()
        confidence_grouped = confidence.reshape(
            B,
            num_tokens,
            gs_per_token,
            1,
        ).float()

        weights = valid * confidence_grouped
        if self.pool_use_opacity:
            opacity = gaussians[..., 3:4].detach().reshape(
                B,
                num_tokens,
                gs_per_token,
                1,
            )
            weights = weights * opacity.clamp_min(0.05)

        denominator = weights.sum(dim=2)
        pooled = (
            projected * weights
        ).sum(dim=2) / denominator.clamp_min(1e-6)

        token_valid = denominator > 0
        token_confidence = confidence_grouped.mean(dim=2)
        return pooled, token_valid, token_confidence

    def materialize_gaussian_features(
        self,
        token_features: torch.Tensor,
        local_codes: torch.Tensor,
    ) -> torch.Tensor:
        local_residual = torch.einsum(
            "btgr,rc->btgc",
            local_codes,
            self.local_basis,
        )
        gaussian_features = (
            token_features[:, :, None, :]
            + self.local_residual_scale * local_residual
        )
        return gaussian_features.reshape(
            gaussian_features.shape[0],
            -1,
            self.feature_dim,
        )

    def forward(
        self,
        decoded_tokens: torch.Tensor,
        gaussians: torch.Tensor,
        source_features: torch.Tensor,
        projected_features: torch.Tensor,
        has_source: torch.Tensor,
        confidence: torch.Tensor,
        materialize_features: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        B, num_tokens, geometry_dim = decoded_tokens.shape
        if geometry_dim != self.geometry_token_dim:
            raise ValueError(
                f"Expected geometry token dim {self.geometry_token_dim}, "
                f"got {geometry_dim}"
            )
        if num_tokens > self.num_tokens:
            raise ValueError(
                f"num_tokens={num_tokens} exceeds configured "
                f"num_tokens={self.num_tokens}"
            )

        pooled, token_valid, token_confidence = (
            self._pool_projected_features(
                projected_features=projected_features,
                has_source=has_source,
                confidence=confidence,
                gaussians=gaussians,
                num_tokens=num_tokens,
            )
        )

        geometry_fallback = self.geometry_fallback(decoded_tokens)
        token_initial_feature = torch.where(
            token_valid,
            pooled,
            geometry_fallback,
        )

        learned_tokens = self.semantic_tokens[
            :num_tokens
        ].unsqueeze(0).expand(B, -1, -1)

        semantic_tokens = (
            learned_tokens
            + self.geometry_to_query(decoded_tokens)
            + self.evidence_to_query(token_initial_feature)
        )

        source_memory = self._build_source_memory(source_features)
        for layer in self.decoder_layers:
            semantic_tokens = layer(
                semantic_tokens=semantic_tokens,
                source_memory=source_memory,
            )
        semantic_tokens = self.decoder_norm(semantic_tokens)

        token_features = (
            token_initial_feature
            + self.token_residual_scale
            * self.token_residual_head(semantic_tokens)
        )

        gs_per_token = gaussians.shape[1] // num_tokens
        gaussian_grouped = gaussians.reshape(
            B,
            num_tokens,
            gs_per_token,
            14,
        )
        projected_grouped = projected_features.reshape(
            B,
            num_tokens,
            gs_per_token,
            self.feature_dim,
        )
        has_source_grouped = has_source.reshape(
            B,
            num_tokens,
            gs_per_token,
            1,
        )
        confidence_grouped = confidence.reshape(
            B,
            num_tokens,
            gs_per_token,
            1,
        )

        token_expanded = token_features[:, :, None, :].expand(
            -1,
            -1,
            gs_per_token,
            -1,
        )
        projected_or_token = torch.where(
            has_source_grouped,
            projected_grouped,
            token_expanded,
        )
        projected_delta = projected_or_token - token_expanded

        semantic_local = self.semantic_local(
            semantic_tokens
        )[:, :, None, :].expand(
            -1,
            -1,
            gs_per_token,
            -1,
        )
        gaussian_local = self.gaussian_local(gaussian_grouped)
        projected_local = self.projected_delta_local(projected_delta)

        local_input = torch.cat(
            [
                semantic_local,
                gaussian_local,
                projected_local,
                confidence_grouped,
            ],
            dim=-1,
        )
        local_codes = torch.tanh(
            self.local_code_head(local_input)
        )

        gaussian_features = None
        if materialize_features:
            gaussian_features = self.materialize_gaussian_features(
                token_features=token_features,
                local_codes=local_codes,
            )

        return {
            "semantic_token_features": token_features,
            "semantic_local_codes": local_codes,
            "semantic_local_basis": self.local_basis,
            "semantic_token_valid": token_valid,
            "semantic_token_confidence": token_confidence,
            "gaussian_semantic_features": gaussian_features,
        }
