"""Metric-embedding residual coupling for the native ERU query head.

This module is deliberately limited to logits-space coupling.  It does not
alter the renderer, the metric embedding head, or the existing query head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class QueryMetricCouplingOutput:
    final_unit_logits: torch.Tensor
    metric_group_logits: torch.Tensor
    centered_metric_group_logits: torch.Tensor
    residual_logits: torch.Tensor
    query_metric_embeddings: torch.Tensor
    gate: torch.Tensor
    temperature: torch.Tensor


class QueryMetricCoupling(nn.Module):
    """Add centered metric-to-query logits to the native 100-group logits."""

    def __init__(
        self,
        *,
        unit_embedding_dim: int = 128,
        query_dim: int = 256,
        num_groups: int = 100,
        initial_temperature: float = 10.0,
        max_gate: float = 0.25,
    ) -> None:
        super().__init__()
        if int(unit_embedding_dim) <= 0 or int(query_dim) <= 0:
            raise ValueError("embedding and query dimensions must be positive")
        if int(num_groups) != 100:
            raise ValueError("QMC requires exactly 100 object-query groups")
        if not torch.isfinite(torch.tensor(float(initial_temperature))):
            raise ValueError("initial_temperature must be finite")
        if float(initial_temperature) <= 0.0:
            raise ValueError("initial_temperature must be positive")
        if not torch.isfinite(torch.tensor(float(max_gate))):
            raise ValueError("max_gate must be finite")
        if float(max_gate) < 0.0:
            raise ValueError("max_gate must be non-negative")
        self.unit_embedding_dim = int(unit_embedding_dim)
        self.query_dim = int(query_dim)
        self.num_groups = int(num_groups)
        self.max_gate = float(max_gate)
        self.query_norm = nn.LayerNorm(self.query_dim)
        self.query_projection = nn.Linear(
            self.query_dim, self.unit_embedding_dim, bias=False
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(float(initial_temperature)).log()
        )
        # The optimizer's existing ndim/no-decay splitter recognizes this
        # marker; the scalar temperature must not receive weight decay.
        self.log_temperature._no_weight_decay = True

    @staticmethod
    def _check_finite(name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"QMC {name} contains NaN or Inf")

    def forward(
        self,
        unit_metric_embeddings: torch.Tensor,
        query_features: torch.Tensor,
        base_unit_logits: torch.Tensor,
        *,
        gate: float,
    ) -> QueryMetricCouplingOutput:
        if not isinstance(gate, (float, int)):
            raise ValueError("QMC gate must be a Python float")
        gate_value = float(gate)
        if not torch.isfinite(torch.tensor(gate_value)):
            raise ValueError("QMC gate must be finite")
        if gate_value < 0.0 or gate_value > self.max_gate:
            raise ValueError(
                f"QMC gate must be in [0, {self.max_gate}], got {gate_value}"
            )
        if unit_metric_embeddings.ndim != 4:
            raise ValueError(
                "unit_metric_embeddings must be [B,1024,8,D], got "
                f"{tuple(unit_metric_embeddings.shape)}"
            )
        if unit_metric_embeddings.shape[1:3] != (1024, 8):
            raise ValueError(
                "unit_metric_embeddings must have 1024 tokens and 8 units/token"
            )
        if unit_metric_embeddings.shape[-1] != self.unit_embedding_dim:
            raise ValueError(
                f"unit metric dim must be {self.unit_embedding_dim}, got "
                f"{unit_metric_embeddings.shape[-1]}"
            )
        if query_features.ndim != 3 or query_features.shape[1:] != (
            self.num_groups,
            self.query_dim,
        ):
            raise ValueError(
                f"query_features must be [B,{self.num_groups},{self.query_dim}], got "
                f"{tuple(query_features.shape)}"
            )
        if base_unit_logits.ndim != 4 or base_unit_logits.shape[1:3] != (1024, 8):
            raise ValueError(
                "base_unit_logits must be [B,1024,8,101], got "
                f"{tuple(base_unit_logits.shape)}"
            )
        if base_unit_logits.shape[-1] != self.num_groups + 1:
            raise ValueError("base_unit_logits must have 100 groups plus void")
        if unit_metric_embeddings.shape[0] != query_features.shape[0] or (
            unit_metric_embeddings.shape[0] != base_unit_logits.shape[0]
        ):
            raise ValueError("QMC batch dimensions do not match")
        self._check_finite("unit_metric_embeddings", unit_metric_embeddings)
        self._check_finite("query_features", query_features)
        self._check_finite("base_unit_logits", base_unit_logits)
        self._check_finite("log_temperature", self.log_temperature)

        e = F.normalize(unit_metric_embeddings.float(), dim=-1, eps=1e-6)
        q = F.normalize(
            self.query_projection(self.query_norm(query_features.float())),
            dim=-1,
            eps=1e-6,
        )
        temperature = self.log_temperature.exp().clamp(1.0, 100.0)
        metric_group_logits = temperature * torch.einsum(
            "btkd,bgd->btkg", e, q
        )
        centered_metric_group_logits = metric_group_logits - metric_group_logits.mean(
            dim=-1, keepdim=True
        )
        void_residual = torch.zeros(
            *centered_metric_group_logits.shape[:-1],
            1,
            device=centered_metric_group_logits.device,
            dtype=centered_metric_group_logits.dtype,
        )
        residual_logits = torch.cat(
            [centered_metric_group_logits, void_residual], dim=-1
        )
        if gate_value == 0.0:
            final_unit_logits = base_unit_logits
        else:
            final_unit_logits = base_unit_logits + (
                gate_value * residual_logits
            ).to(dtype=base_unit_logits.dtype)
        self._check_finite("final_unit_logits", final_unit_logits)
        self._check_finite("metric_group_logits", metric_group_logits)
        return QueryMetricCouplingOutput(
            final_unit_logits=final_unit_logits,
            metric_group_logits=metric_group_logits,
            centered_metric_group_logits=centered_metric_group_logits,
            residual_logits=residual_logits,
            query_metric_embeddings=q,
            gate=torch.as_tensor(
                gate_value, device=base_unit_logits.device, dtype=torch.float32
            ),
            temperature=temperature.to(device=base_unit_logits.device),
        )
