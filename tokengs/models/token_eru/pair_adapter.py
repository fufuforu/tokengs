from __future__ import annotations

import torch
from torch import nn


class ZeroInitPairAdapter(nn.Module):
    """A zero-at-initialization residual adapter between decoder streams."""

    def __init__(self, dim: int, bottleneck_dim: int = 128) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.down = nn.Linear(int(dim), int(bottleneck_dim))
        self.act = nn.GELU()
        self.up = nn.Linear(int(bottleneck_dim), int(dim))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(self.norm(source))))
