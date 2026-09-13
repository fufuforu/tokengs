"""Context-only historical DINO evidence and metric heads for TokenGS-ERU.

The alignment in this file deliberately follows the already recovered
historical implementation: DINO patch features are projected onto the
context-view Gaussian centers and then reduced in the existing eight-child
local-unit order.  The DINO backbone is supplied by the offline extractor in
``ta_riu_v2``; its external model is kept out of the parent state dict there.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from tokengs.models.instance_group_head import _project_dense_features
from tokengs.models.ta_riu_v2 import FrozenDINOv2Extractor


@dataclass(frozen=True)
class DINOUnitEvidence:
    unit_features: torch.Tensor
    patch_features: torch.Tensor
    alignment_weights: torch.Tensor | None
    context_views: int


class HistoricalDINOUnitEncoder(nn.Module):
    """Frozen context DINO plus trainable historical patch-to-unit projector."""

    def __init__(
        self,
        dino_repo_path: str,
        dino_weight_path: str,
        output_dim: int = 256,
        num_context_views: int = 8,
        num_tokens: int = 1024,
        units_per_token: int = 8,
    ) -> None:
        super().__init__()
        if int(num_context_views) != 8:
            raise ValueError("TokenGS-ERU-DINO requires exactly 8 context views")
        if int(num_tokens) != 1024 or int(units_per_token) != 8:
            raise ValueError("historical DINO alignment requires 1024x8 units")
        self.output_dim = int(output_dim)
        self.num_context_views = int(num_context_views)
        self.num_tokens = int(num_tokens)
        self.units_per_token = int(units_per_token)
        self.dino_extractor = FrozenDINOv2Extractor(
            dino_repo_path, dino_weight_path
        )
        # This is the only trainable projection in the alignment encoder.
        self.unit_projector = nn.Linear(768, self.output_dim)
        nn.init.xavier_uniform_(self.unit_projector.weight)
        nn.init.zeros_(self.unit_projector.bias)

    @torch.no_grad()
    def extract_dino_patches(self, context_images: torch.Tensor) -> torch.Tensor:
        if (
            not torch.is_tensor(context_images)
            or context_images.ndim != 5
            or context_images.shape[1] != self.num_context_views
            or context_images.shape[2] != 3
        ):
            raise ValueError(
                "context_images must have shape [B,8,3,H,W], got "
                f"{getattr(context_images, 'shape', None)}"
            )
        return self.dino_extractor(context_images)

    def forward(
        self,
        context_images: torch.Tensor,
        historical_alignment_inputs: dict[str, torch.Tensor],
    ) -> DINOUnitEvidence:
        patches = self.extract_dino_patches(context_images)
        if patches.ndim != 5 or tuple(patches.shape[1:3]) != (8, 768):
            raise RuntimeError(
                "historical DINO patch shape must be [B,8,768,Hf,Wf], got "
                f"{tuple(patches.shape)}"
            )
        required = (
            "base_gaussians",
            "cam_to_world_input",
            "intrinsics_input",
            "image_hw",
        )
        missing = [key for key in required if key not in historical_alignment_inputs]
        if missing:
            raise RuntimeError(f"missing DINO alignment inputs: {missing}")
        gaussians = historical_alignment_inputs["base_gaussians"]
        if gaussians.ndim != 3 or gaussians.shape[1] != 65536:
            raise RuntimeError(
                "historical DINO alignment expects [B,65536,14] Gaussians, got "
                f"{tuple(gaussians.shape)}"
            )
        b, v, dino_dim, hf, wf = patches.shape
        if v != self.num_context_views or dino_dim != 768:
            raise RuntimeError("DINO context or feature dimension mismatch")
        image_hw = tuple(int(x) for x in historical_alignment_inputs["image_hw"])
        target_hw = (252, 252)
        scale = patches.new_tensor(
            [252 / image_hw[1], 252 / image_hw[0], 252 / image_hw[1], 252 / image_hw[0]]
        )
        intrinsics = historical_alignment_inputs["intrinsics_input"] * scale
        projected, weights = _project_dense_features(
            gaussians[..., :3].float(),
            patches.float(),
            historical_alignment_inputs["cam_to_world_input"],
            intrinsics,
            target_hw,
        )
        if projected.shape != (b, 65536, 768):
            raise RuntimeError(
                "DINO-to-Gaussian projection returned unexpected shape: "
                f"{tuple(projected.shape)}"
            )
        child = projected.reshape(b, self.num_tokens, self.units_per_token, 8, 768)
        unit_raw = child.mean(dim=3)
        unit_raw = F.normalize(unit_raw.float(), dim=-1)
        unit_features = self.unit_projector(unit_raw)
        unit_features = F.normalize(unit_features, dim=-1)
        if unit_features.shape != (b, 1024, 8, self.output_dim):
            raise RuntimeError(
                "DINO unit evidence is not aligned to [B,1024,8,D]: "
                f"{tuple(unit_features.shape)}"
            )
        alignment_weights = None
        if weights is not None:
            # The recovered projector returns the valid-view count per child
            # Gaussian, not a per-view visibility tensor.  Reduce its eight
            # children in the same unit order used above.
            alignment_weights = weights.reshape(
                b, self.num_tokens, self.units_per_token, 8, 1
            ).float().mean(dim=3)
        return DINOUnitEvidence(
            unit_features=unit_features,
            patch_features=patches.detach(),
            alignment_weights=alignment_weights,
            context_views=self.num_context_views,
        )


class DINOUnitFusion(nn.Module):
    """Fuse projected DINO evidence while preserving exact gate-zero identity."""

    def __init__(self, unit_dim: int = 256, dino_unit_dim: int = 256) -> None:
        super().__init__()
        self.unit_dim = int(unit_dim)
        self.dino_unit_dim = int(dino_unit_dim)
        self.dino_projection = nn.Linear(self.dino_unit_dim, self.unit_dim)
        self.unit_norm = nn.LayerNorm(self.unit_dim)

    def forward(
        self,
        understanding_units: torch.Tensor,
        dino_unit_features: torch.Tensor,
        gate: float,
    ) -> torch.Tensor:
        if understanding_units.ndim != 4 or dino_unit_features.ndim != 4:
            raise ValueError("DINO fusion inputs must be [B,1024,8,D]")
        if understanding_units.shape[:3] != dino_unit_features.shape[:3]:
            raise ValueError("DINO and ERU unit layouts do not match")
        if understanding_units.shape[-1] != self.unit_dim:
            raise ValueError("unexpected ERU unit dimension")
        if dino_unit_features.shape[-1] != self.dino_unit_dim:
            raise ValueError("unexpected DINO unit dimension")
        if not torch.isfinite(understanding_units).all() or not torch.isfinite(
            dino_unit_features
        ).all():
            raise FloatingPointError("non-finite DINO fusion input")
        gate = float(gate)
        if gate == 0.0:
            return understanding_units
        delta = self.dino_projection(dino_unit_features)
        # This algebraic form is required: gate=0 returns the original tensor,
        # without a LayerNorm-induced value change.
        return understanding_units + gate * (
            self.unit_norm(understanding_units + delta) - understanding_units
        )


class MetricEmbeddingHead(nn.Module):
    """Historical-style 128D normalized local-unit metric embedding."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.embedding_dim),
        )

    def forward(self, fused_units: torch.Tensor) -> torch.Tensor:
        if fused_units.ndim != 4 or fused_units.shape[-1] != self.input_dim:
            raise ValueError("fused_units must be [B,1024,8,256]")
        if not torch.isfinite(fused_units).all():
            raise FloatingPointError("metric head input contains NaN/Inf")
        result = F.normalize(self.net(fused_units), dim=-1, eps=1e-6)
        if not torch.isfinite(result).all():
            raise FloatingPointError("metric head output contains NaN/Inf")
        return result
