"""TA-RIU-v3 dual-stream token-aligned reconstruction/instance path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokengs.models.ta_riu_v2 import (
    DINO_EXPECTED_SHA256,
    FrozenDINOv2Extractor,
    sha256_file,
)


V3_CONTEXT_VIEWS = 8
V3_UNITS_PER_VIEW = 1024
V3_DINO_TOKENS_PER_VIEW = 324
V3_DINO_DIM = 768


@dataclass
class TARIUV3Output:
    reconstruction_units: torch.Tensor
    instance_units: torch.Tensor
    joint_reconstruction_units: torch.Tensor
    joint_instance_units: torch.Tensor
    dino_memory: torch.Tensor
    reconstruction_delta: torch.Tensor
    instance_delta: torch.Tensor


def restore_encoder_memory(
    encoder_values: torch.Tensor,
    num_views: int = V3_CONTEXT_VIEWS,
    patches_per_view: int = V3_UNITS_PER_VIEW,
) -> torch.Tensor:
    """Restore [B,H,S,Dh] encoder values to view-major unit memory."""
    if encoder_values.ndim != 4:
        raise ValueError(
            "restore_encoder_memory expects [B,H,S,Dh], got "
            f"{tuple(encoder_values.shape)}"
        )
    batch, heads, sequence, head_dim = encoder_values.shape
    if heads * head_dim != 1024:
        raise ValueError(
            f"encoder H*Dh must equal 1024, got H={heads}, Dh={head_dim}"
        )
    expected = int(num_views) * int(patches_per_view)
    if sequence != expected:
        raise ValueError(
            f"encoder sequence must equal {expected}, got {sequence}"
        )
    if not torch.isfinite(encoder_values).all():
        raise ValueError("encoder values contain non-finite values")
    return (
        encoder_values.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(batch, expected, heads * head_dim)
    )


def unflatten_reconstruction_units(
    units: torch.Tensor,
    context_views: int = V3_CONTEXT_VIEWS,
    units_per_view: int = V3_UNITS_PER_VIEW,
) -> torch.Tensor:
    """Restore view-major [B,V,U,F] units to the absolute-head layout."""
    if units.ndim != 3:
        raise ValueError(
            f"unflatten_reconstruction_units expects [B,N,F], got {tuple(units.shape)}"
        )
    expected = int(context_views) * int(units_per_view)
    if units.shape[1] != expected:
        raise ValueError(f"expected N={expected}, got {units.shape[1]}")
    return units.reshape(units.shape[0], int(context_views), int(units_per_view), -1).permute(
        0, 2, 1, 3
    ).contiguous()


class ContextDINOEncoder(nn.Module):
    """Frozen local DINO backbone plus trainable patch projection."""

    def __init__(
        self,
        repo_path: str,
        weight_path: str,
        output_dim: int = 256,
        expected_context_views: int = V3_CONTEXT_VIEWS,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.repo_path = str(repo_path)
        self.weight_path = str(weight_path)
        self.expected_context_views = int(expected_context_views)
        self.output_dim = int(output_dim)
        self.dino_extractor = FrozenDINOv2Extractor(
            self.repo_path, self.weight_path, model_name="dinov2_vitb14"
        )
        if not freeze:
            raise ValueError("TA-RIU-v3 requires a frozen DINO backbone")
        self.dino_norm = nn.LayerNorm(V3_DINO_DIM)
        self.dino_proj = nn.Linear(V3_DINO_DIM, self.output_dim)
        self.last_context_shape: Optional[tuple[int, ...]] = None

    @torch.no_grad()
    def encode_backbone(self, context_images: torch.Tensor) -> torch.Tensor:
        if context_images.ndim != 5:
            raise ValueError(
                "ContextDINOEncoder expects context_images [B,V,3,H,W], got "
                f"{tuple(context_images.shape)}"
            )
        if context_images.shape[1] != self.expected_context_views:
            raise ValueError(
                f"ContextDINOEncoder expects {self.expected_context_views} context "
                f"views, got shape {tuple(context_images.shape)}"
            )
        patches = self.dino_extractor(context_images)
        if patches.ndim != 5 or patches.shape[2:] != (V3_DINO_DIM, 18, 18):
            raise ValueError(
                "DINO patch output must be [B,V,768,18,18], got "
                f"{tuple(patches.shape)}"
            )
        self.last_context_shape = tuple(int(x) for x in context_images.shape)
        return patches.permute(0, 1, 3, 4, 2).reshape(
            patches.shape[0], patches.shape[1], V3_DINO_TOKENS_PER_VIEW, V3_DINO_DIM
        )

    def forward(self, context_images: torch.Tensor) -> torch.Tensor:
        # FrozenDINOv2Extractor intentionally returns an inference tensor.
        # A trainable projection follows it, and autograd cannot save an
        # inference tensor for that backward graph.  Clone exactly once at
        # this boundary; the DINO backbone remains inference-only and the
        # clone carries no gradient edge into DINO.
        patches = self.encode_backbone(context_images).float().clone()
        projected = self.dino_proj(self.dino_norm(patches))
        if not torch.isfinite(projected).all():
            raise RuntimeError("TA-RIU-v3 DINO projection produced non-finite values")
        return projected

    @staticmethod
    def verify_weight_hash(weight_path: str) -> str:
        actual = sha256_file(weight_path)
        if actual != DINO_EXPECTED_SHA256:
            raise RuntimeError(
                f"TA-RIU-v3 DINO SHA256 mismatch for {weight_path}: "
                f"{actual} != {DINO_EXPECTED_SHA256}"
            )
        return actual


class ReconstructionUnitAdapter(nn.Module):
    """Convert the absolute decoder's [B,T,K,F] layout to view-major units."""

    def __init__(
        self,
        feature_dim: int = 256,
        context_views: int = V3_CONTEXT_VIEWS,
        units_per_view: int = V3_UNITS_PER_VIEW,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.context_views = int(context_views)
        self.units_per_view = int(units_per_view)

    def forward(self, q_abs: torch.Tensor) -> torch.Tensor:
        if q_abs.ndim != 4 or q_abs.shape[-1] != self.feature_dim:
            raise ValueError(
                "ReconstructionUnitAdapter expects [B,T,K,"
                f"{self.feature_dim}], got {tuple(q_abs.shape)}"
            )
        if q_abs.shape[1] != self.units_per_view or q_abs.shape[2] != self.context_views:
            raise ValueError(
                f"ReconstructionUnitAdapter expects T={self.units_per_view}, "
                f"K={self.context_views}, got {tuple(q_abs.shape)}"
            )
        return q_abs.permute(0, 2, 1, 3).contiguous().reshape(
            q_abs.shape[0], self.context_views * self.units_per_view, self.feature_dim
        )


class AlignedInstanceQueryEmbedding(nn.Module):
    def __init__(
        self,
        context_views: int = V3_CONTEXT_VIEWS,
        units_per_view: int = V3_UNITS_PER_VIEW,
        dim: int = 256,
    ) -> None:
        super().__init__()
        self.context_views = int(context_views)
        self.units_per_view = int(units_per_view)
        self.dim = int(dim)
        self.spatial_query = nn.Parameter(torch.empty(1, self.units_per_view, self.dim))
        self.view_embedding = nn.Parameter(
            torch.zeros(1, self.context_views, 1, self.dim)
        )
        nn.init.trunc_normal_(self.spatial_query, std=0.02)

    def forward(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        queries = self.spatial_query.to(device=device, dtype=dtype)[:, None, :, :]
        views = self.view_embedding.to(device=device, dtype=dtype)
        queries = (queries + views).expand(
            int(batch_size), self.context_views, self.units_per_view, self.dim
        )
        return queries.reshape(
            int(batch_size), self.context_views * self.units_per_view, self.dim
        )


class _InstanceUnitBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.dino_query_norm = nn.LayerNorm(dim)
        self.dino_memory_norm = nn.LayerNorm(dim)
        self.dino_cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )
        self.dino_mlp_norm = nn.LayerNorm(dim)
        self.dino_mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        self.aligned_x_norm = nn.LayerNorm(dim)
        self.aligned_r_norm = nn.LayerNorm(dim)
        self.aligned_conditioning = nn.Linear(dim * 3, dim)
        self.aligned_gate = nn.Linear(dim * 2, dim)
        self.aligned_mlp_norm = nn.LayerNorm(dim)
        self.aligned_mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        reconstruction_units: torch.Tensor,
        dino_memory: torch.Tensor,
        context_views: int,
        units_per_view: int,
    ) -> torch.Tensor:
        batch = x.shape[0]
        x_view = x.reshape(batch * context_views, units_per_view, -1)
        dino_view = dino_memory.reshape(batch * context_views, -1, x.shape[-1])
        dino_out, _ = self.dino_cross_attn(
            self.dino_query_norm(x_view),
            self.dino_memory_norm(dino_view),
            self.dino_memory_norm(dino_view),
        )
        x_view = x_view + dino_out
        x_view = x_view + self.dino_mlp(self.dino_mlp_norm(x_view))
        x = x_view.reshape(batch, context_views * units_per_view, -1)

        x_norm = self.aligned_x_norm(x)
        r_norm = self.aligned_r_norm(reconstruction_units)
        cond = self.aligned_conditioning(
            torch.cat([x_norm, r_norm, x_norm * r_norm], dim=-1)
        )
        gate = torch.sigmoid(self.aligned_gate(torch.cat([x_norm, r_norm], dim=-1)))
        x = x + gate * cond
        return x + self.aligned_mlp(self.aligned_mlp_norm(x))


class AlignedInstanceUnitFormer(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        context_views: int = V3_CONTEXT_VIEWS,
        units_per_view: int = V3_UNITS_PER_VIEW,
        dino_tokens_per_view: int = V3_DINO_TOKENS_PER_VIEW,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.context_views = int(context_views)
        self.units_per_view = int(units_per_view)
        self.dino_tokens_per_view = int(dino_tokens_per_view)
        self.blocks = nn.ModuleList(
            [_InstanceUnitBlock(self.dim, num_heads, mlp_ratio, dropout) for _ in range(int(depth))]
        )
        self.final_norm = nn.LayerNorm(self.dim)

    def forward(
        self,
        instance_queries: torch.Tensor,
        reconstruction_units: torch.Tensor,
        dino_memory: torch.Tensor,
    ) -> torch.Tensor:
        expected_units = self.context_views * self.units_per_view
        if instance_queries.shape != reconstruction_units.shape:
            raise ValueError(
                "AlignedInstanceUnitFormer query/reconstruction shape mismatch: "
                f"{tuple(instance_queries.shape)} vs {tuple(reconstruction_units.shape)}"
            )
        if (
            instance_queries.ndim != 3
            or instance_queries.shape[1] != expected_units
            or instance_queries.shape[-1] != self.dim
        ):
            raise ValueError(
                f"AlignedInstanceUnitFormer expects [B,{expected_units},{self.dim}], "
                f"got {tuple(instance_queries.shape)}"
            )
        expected_dino = (
            instance_queries.shape[0],
            self.context_views,
            self.dino_tokens_per_view,
            self.dim,
        )
        if tuple(dino_memory.shape) != expected_dino:
            raise ValueError(
                f"AlignedInstanceUnitFormer expects DINO {expected_dino}, "
                f"got {tuple(dino_memory.shape)}"
            )
        x = instance_queries
        for block in self.blocks:
            x = block(
                x,
                reconstruction_units,
                dino_memory,
                self.context_views,
                self.units_per_view,
            )
            if not torch.isfinite(x).all():
                raise RuntimeError("TA-RIU-v3 instance former produced non-finite values")
        return self.final_norm(x)


class PairUnitMixer(nn.Module):
    def __init__(self, dim: int = 256, hidden_dim: int = 512) -> None:
        super().__init__()
        self.dim = int(dim)
        self.rec_norm = nn.LayerNorm(self.dim)
        self.ins_norm = nn.LayerNorm(self.dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.dim * 4, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.dim),
        )
        self.rec_out = nn.Linear(self.dim, self.dim)
        self.ins_out = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.rec_out.weight)
        nn.init.zeros_(self.rec_out.bias)
        nn.init.zeros_(self.ins_out.weight)
        nn.init.zeros_(self.ins_out.bias)

    def forward(
        self,
        reconstruction_units: torch.Tensor,
        instance_units: torch.Tensor,
        gate: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0.0 <= float(gate) <= 1.0:
            raise ValueError(f"PairUnitMixer gate must be in [0,1], got {gate}")
        if reconstruction_units.shape != instance_units.shape:
            raise ValueError(
                "PairUnitMixer input shape mismatch: "
                f"{tuple(reconstruction_units.shape)} vs {tuple(instance_units.shape)}"
            )
        r = self.rec_norm(reconstruction_units)
        i = self.ins_norm(instance_units)
        p = torch.cat([r, i, r * i, r - i], dim=-1)
        h = self.trunk(p)
        delta_rec = self.rec_out(h)
        delta_ins = self.ins_out(h)
        joint_rec = reconstruction_units + float(gate) * delta_rec
        joint_ins = instance_units + float(gate) * delta_ins
        return joint_rec, joint_ins, delta_rec, delta_ins


class TARIUV3DualStream(nn.Module):
    def __init__(
        self,
        dino_repo_path: str,
        dino_weight_path: str,
        dim: int = 256,
        context_views: int = V3_CONTEXT_VIEWS,
        units_per_view: int = V3_UNITS_PER_VIEW,
        instance_depth: int = 2,
        num_heads: int = 8,
        mixer_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.context_views = int(context_views)
        self.units_per_view = int(units_per_view)
        self.reconstruction_adapter = ReconstructionUnitAdapter(
            feature_dim=self.dim,
            context_views=self.context_views,
            units_per_view=self.units_per_view,
        )
        self.context_dino = ContextDINOEncoder(
            dino_repo_path,
            dino_weight_path,
            output_dim=self.dim,
            expected_context_views=self.context_views,
        )
        self.instance_query_embedding = AlignedInstanceQueryEmbedding(
            context_views=self.context_views,
            units_per_view=self.units_per_view,
            dim=self.dim,
        )
        self.instance_unit_former = AlignedInstanceUnitFormer(
            dim=self.dim,
            num_heads=int(num_heads),
            depth=int(instance_depth),
            context_views=self.context_views,
            units_per_view=self.units_per_view,
        )
        self.pair_mixer = PairUnitMixer(self.dim, int(mixer_hidden_dim))
        # Diagnostic-only switches.  The normal path is "normal" and the
        # formal trainer never changes these attributes.
        self.diagnostic_dino_mode = "normal"
        self.diagnostic_instance_mode = "normal"
        self.diagnostic_index_mode = "normal"

    def forward(
        self,
        q_abs: torch.Tensor,
        context_images: torch.Tensor,
        gate: float,
    ) -> TARIUV3Output:
        if q_abs.ndim != 4 or q_abs.shape[-1] != self.dim:
            raise ValueError(
                f"TARIUV3DualStream expects q_abs [B,T,K,{self.dim}], got "
                f"{tuple(q_abs.shape)}"
            )
        reconstruction_units = self.reconstruction_adapter(q_abs)
        dino_memory = self.context_dino(context_images)
        if self.diagnostic_dino_mode == "zero":
            dino_memory = torch.zeros_like(dino_memory)
        elif self.diagnostic_dino_mode == "view_shift":
            dino_memory = torch.roll(dino_memory, shifts=1, dims=1)
        elif self.diagnostic_dino_mode != "normal":
            raise ValueError(f"unknown diagnostic_dino_mode={self.diagnostic_dino_mode}")
        instance_queries = self.instance_query_embedding(
            q_abs.shape[0], q_abs.device, q_abs.dtype
        )
        conditioning_units = reconstruction_units.detach()
        if self.diagnostic_index_mode == "view_shift":
            conditioning_units = conditioning_units.reshape(
                q_abs.shape[0], self.context_views, self.units_per_view, self.dim
            ).roll(1, dims=1).reshape_as(conditioning_units)
        elif self.diagnostic_index_mode != "normal":
            raise ValueError(f"unknown diagnostic_index_mode={self.diagnostic_index_mode}")
        instance_units = self.instance_unit_former(
            instance_queries,
            conditioning_units,
            dino_memory,
        )
        if self.diagnostic_instance_mode == "zero":
            instance_units = torch.zeros_like(instance_units)
        elif self.diagnostic_instance_mode != "normal":
            raise ValueError(
                f"unknown diagnostic_instance_mode={self.diagnostic_instance_mode}"
            )
        joint_rec, joint_ins, delta_rec, delta_ins = self.pair_mixer(
            reconstruction_units, instance_units, gate
        )
        return TARIUV3Output(
            reconstruction_units=reconstruction_units,
            instance_units=instance_units,
            joint_reconstruction_units=joint_rec,
            joint_instance_units=joint_ins,
            dino_memory=dino_memory,
            reconstruction_delta=delta_rec,
            instance_delta=delta_ins,
        )
