from __future__ import annotations

import torch
from torch import nn


class SemanticQueryHead(nn.Module):
    """A single fixed-width classifier over the existing 100 object queries."""

    def __init__(
        self,
        query_dim: int,
        num_semantic_classes: int,
        *,
        init_std: float = 0.02,
        init_seed: int = 3407,
    ) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.num_semantic_classes = int(num_semantic_classes)
        if self.query_dim <= 0 or self.num_semantic_classes <= 0:
            raise ValueError("query_dim and num_semantic_classes must be positive")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(init_seed))
            self.linear = nn.Linear(self.query_dim, self.num_semantic_classes + 1)
            nn.init.trunc_normal_(self.linear.weight, std=float(init_std))
            nn.init.zeros_(self.linear.bias)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        if query_features.ndim != 3:
            raise ValueError(
                f"query_features must be [B,100,D], got {tuple(query_features.shape)}"
            )
        if query_features.shape[1] != 100:
            raise ValueError(f"expected 100 object queries, got {query_features.shape[1]}")
        if query_features.shape[2] != self.query_dim:
            raise ValueError(
                f"expected query dim {self.query_dim}, got {query_features.shape[2]}"
            )
        if not torch.isfinite(query_features).all():
            raise FloatingPointError("query_features contains NaN or Inf")
        logits = self.linear(query_features)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("semantic query logits contain NaN or Inf")
        return logits
