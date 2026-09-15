"""Explicit reconstruction-derived 3D conditioning for ERU unit queries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch
from torch import nn


@dataclass(frozen=True)
class Unit3DAnchorOutput:
    unit_centers_world: torch.Tensor
    unit_centers_normalized: torch.Tensor
    opacity_mass: torch.Tensor
    fallback_mask: torch.Tensor
    position_features: torch.Tensor
    anchor_delta: torch.Tensor
    anchored_units: torch.Tensor


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def reshape_child_gaussian_attributes(
    means: torch.Tensor,
    opacities: torch.Tensor,
    *,
    token_count: int = 1024,
    units_per_token: int = 8,
    children_per_unit: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the decoder's contiguous token/unit/child Gaussian layout."""
    if means.ndim != 3 or means.shape[-1] != 3:
        raise ValueError(f"means must be [B,N,3], got {tuple(means.shape)}")
    if opacities.ndim == 2:
        opacities = opacities.unsqueeze(-1)
    elif opacities.ndim != 3 or opacities.shape[-1] != 1:
        raise ValueError(
            "opacities must be [B,N] or [B,N,1], "
            f"got {tuple(opacities.shape)}"
        )
    if means.shape[:2] != opacities.shape[:2]:
        raise ValueError("means and opacities must share batch and Gaussian count")
    expected = int(token_count) * int(units_per_token) * int(children_per_unit)
    if means.shape[1] != expected:
        raise ValueError(f"expected {expected} Gaussians, got {means.shape[1]}")
    _finite("means", means)
    _finite("opacities", opacities)
    if bool((opacities < -1e-6).any()) or bool((opacities > 1.0 + 1e-6).any()):
        raise ValueError("opacities must be activated values in [0,1]")
    opacities = opacities.clamp(0.0, 1.0)
    shape = (means.shape[0], int(token_count), int(units_per_token), int(children_per_unit))
    return means.reshape(*shape, 3), opacities.reshape(*shape, 1)


def compute_opacity_weighted_unit_centers(
    child_means: torch.Tensor,
    child_opacities: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute opacity-weighted unit centers with a mean-child fallback."""
    if child_means.ndim != 5 or child_means.shape[-1] != 3:
        raise ValueError("child_means must be [B,T,K,C,3]")
    if child_opacities.shape != child_means.shape[:-1] + (1,):
        raise ValueError("child_opacities must be [B,T,K,C,1]")
    _finite("child_means", child_means)
    _finite("child_opacities", child_opacities)
    alpha = child_opacities.clamp(0.0, 1.0)
    mass = alpha.sum(dim=-2)
    weighted = (alpha * child_means).sum(dim=-2) / mass.clamp_min(float(eps))
    fallback = child_means.mean(dim=-2)
    fallback_mask = (mass <= float(eps))
    centers = torch.where(fallback_mask, fallback, weighted)
    _finite("unit_centers", centers)
    _finite("opacity_mass", mass)
    return centers, mass, fallback_mask


def normalize_unit_centers(
    unit_centers_world: torch.Tensor,
    *,
    min_scale: float = 1e-3,
    clamp_value: float = 10.0,
    detach_statistics: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize centers by detached per-scene center and RMS scale."""
    if unit_centers_world.ndim != 4 or unit_centers_world.shape[-1] != 3:
        raise ValueError("unit_centers_world must be [B,T,K,3]")
    _finite("unit_centers_world", unit_centers_world)
    scene_center = unit_centers_world.mean(dim=(1, 2), keepdim=True)
    centered = unit_centers_world - scene_center
    scene_scale = torch.sqrt(centered.square().sum(dim=-1).mean(dim=(1, 2), keepdim=True))
    scene_scale = scene_scale.clamp_min(float(min_scale))
    if detach_statistics:
        scene_center = scene_center.detach()
        scene_scale = scene_scale.detach()
    normalized = (centered / scene_scale).clamp(-float(clamp_value), float(clamp_value))
    _finite("normalized_unit_centers", normalized)
    _finite("scene_center", scene_center)
    _finite("scene_scale", scene_scale)
    return normalized, scene_center, scene_scale


class FixedFourierPositionEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 6, include_input: bool = True) -> None:
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        if self.num_frequencies < 0:
            raise ValueError("num_frequencies must be non-negative")
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32),
            persistent=False,
        )

    @property
    def output_dim(self) -> int:
        return (3 if self.include_input else 0) + 3 * 2 * self.num_frequencies

    def forward(self, xyz_normalized: torch.Tensor) -> torch.Tensor:
        if xyz_normalized.shape[-1] != 3:
            raise ValueError("xyz_normalized last dimension must be 3")
        values = [xyz_normalized] if self.include_input else []
        for frequency in self.frequencies.to(xyz_normalized):
            phase = torch.pi * frequency * xyz_normalized
            values.extend((phase.sin(), phase.cos()))
        output = torch.cat(values, dim=-1)
        _finite("position_features", output)
        return output


class Unit3DAnchor(nn.Module):
    def __init__(
        self,
        unit_dim: int = 256,
        hidden_dim: int = 256,
        num_frequencies: int = 6,
        eps: float = 1e-6,
        min_scale: float = 1e-3,
        clamp_value: float = 10.0,
        injection_scale: float = 1.0,
        detach_statistics: bool = True,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.min_scale = float(min_scale)
        self.clamp_value = float(clamp_value)
        self.injection_scale = float(injection_scale)
        self.detach_statistics = bool(detach_statistics)
        self.position_encoding = FixedFourierPositionEncoding(num_frequencies, True)
        self.position_mlp = nn.Sequential(
            nn.Linear(self.position_encoding.output_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(unit_dim)),
        )
        nn.init.zeros_(self.position_mlp[-1].weight)
        nn.init.zeros_(self.position_mlp[-1].bias)
        # Evaluation-only override.  This is deliberately not a Parameter or
        # a buffer, so it cannot enter state_dict/checkpoints.
        self._eval_ablation_mode = "full"
        self._eval_ablation_permutation_seed = 42
        self._last_eval_permutation_sha256 = None

    def set_eval_ablation(
        self,
        mode: str = "full",
        *,
        permutation_seed: int = 42,
    ) -> None:
        """Set a non-persistent causal evaluation override.

        Non-full modes are intentionally forbidden during training.  The
        caller supplies a deterministic per-window seed for ``shuffle``.
        """
        mode = str(mode)
        if mode not in {"full", "off", "shuffle", "zero"}:
            raise ValueError(f"unknown anchor evaluation ablation: {mode}")
        if self.training and mode != "full":
            raise RuntimeError(
                "anchor evaluation ablation is only valid when model.eval() "
                "is active"
            )
        self._eval_ablation_mode = mode
        self._eval_ablation_permutation_seed = int(permutation_seed)
        self._last_eval_permutation_sha256 = None

    @property
    def eval_ablation_mode(self) -> str:
        return self._eval_ablation_mode

    @property
    def last_eval_permutation_sha256(self) -> str | None:
        return self._last_eval_permutation_sha256

    @staticmethod
    def _permutation_sha256(permutation: torch.Tensor) -> str:
        value = permutation.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def forward(
        self,
        understanding_units: torch.Tensor,
        gaussian_means: torch.Tensor,
        gaussian_opacities: torch.Tensor,
    ) -> Unit3DAnchorOutput:
        if understanding_units.ndim != 4 or understanding_units.shape[-1] != 256:
            raise ValueError("understanding_units must be [B,1024,8,256]")
        child_means, child_opacities = reshape_child_gaussian_attributes(
            gaussian_means, gaussian_opacities,
            token_count=understanding_units.shape[1],
            units_per_token=understanding_units.shape[2],
        )
        centers, mass, fallback_mask = compute_opacity_weighted_unit_centers(
            child_means, child_opacities, eps=self.eps
        )
        normalized, _, _ = normalize_unit_centers(
            centers,
            min_scale=self.min_scale,
            clamp_value=self.clamp_value,
            detach_statistics=self.detach_statistics,
        )
        mode = self._eval_ablation_mode
        if self.training and mode != "full":
            raise RuntimeError(
                "anchor evaluation ablation is only valid when model.eval() "
                "is active"
            )
        self._last_eval_permutation_sha256 = None
        if mode == "zero":
            normalized = torch.zeros_like(normalized)
        elif mode == "shuffle":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(self._eval_ablation_permutation_seed))
            flat = normalized.reshape(normalized.shape[0], -1, 3)
            permutation = torch.randperm(flat.shape[1], generator=generator)
            permutation = permutation.to(device=normalized.device)
            normalized = flat[:, permutation, :].reshape_as(normalized)
            self._last_eval_permutation_sha256 = self._permutation_sha256(permutation)
        position_features = self.position_encoding(normalized)
        anchor_delta = self.position_mlp(position_features)
        if mode == "off":
            anchored_units = understanding_units
        else:
            anchored_units = understanding_units + self.injection_scale * anchor_delta
        _finite("anchor_delta", anchor_delta)
        _finite("anchored_units", anchored_units)
        return Unit3DAnchorOutput(
            unit_centers_world=centers,
            unit_centers_normalized=normalized,
            opacity_mass=mass,
            fallback_mask=fallback_mask,
            position_features=position_features,
            anchor_delta=anchor_delta,
            anchored_units=anchored_units,
        )
