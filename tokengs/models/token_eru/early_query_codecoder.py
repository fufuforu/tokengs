"""Early query/U-stream co-decoder for TokenGS-ERU EQC-v1.

The module contains no DINO, geometry, clustering, or target-dependent
inputs.  Its output projections are zero initialized so a local stage step-0
forward is an identity when the early-query gate is zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EarlyQueryCoDecoderOutput:
    understanding_hidden: torch.Tensor
    query_state: torch.Tensor


class EarlyObjectQueryAdapter(nn.Module):
    """Bidirectional attention between 100 object queries and U tokens."""

    def __init__(
        self,
        understanding_dim: int = 1024,
        query_dim: int = 256,
        attention_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        understanding_write_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if int(understanding_dim) != 1024:
            raise ValueError("EQC requires understanding_dim=1024")
        if int(query_dim) != 256 or int(attention_dim) != 256:
            raise ValueError("EQC requires query_dim=attention_dim=256")
        if int(num_heads) != 8:
            raise ValueError("EQC requires num_heads=8")
        if float(dropout) != 0.0:
            raise ValueError("EQC requires dropout=0")
        if float(mlp_ratio) <= 0.0:
            raise ValueError("EQC mlp_ratio must be positive")
        self.understanding_dim = int(understanding_dim)
        self.query_dim = int(query_dim)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.understanding_write_scale = float(understanding_write_scale)

        self.understanding_norm = nn.LayerNorm(self.understanding_dim)
        self.understanding_projection = nn.Linear(
            self.understanding_dim, self.attention_dim
        )
        self.query_norm = nn.LayerNorm(self.query_dim)
        self.query_attention = nn.MultiheadAttention(
            self.attention_dim,
            self.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.query_output = nn.Linear(self.attention_dim, self.query_dim)
        self.query_ffn_norm = nn.LayerNorm(self.query_dim)
        self.query_ffn = nn.Sequential(
            nn.Linear(self.query_dim, int(self.query_dim * float(mlp_ratio))),
            nn.GELU(),
            nn.Linear(int(self.query_dim * float(mlp_ratio)), self.query_dim),
        )

        self.query_to_understanding_norm = nn.LayerNorm(self.query_dim)
        self.understanding_attention = nn.MultiheadAttention(
            self.attention_dim,
            self.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.understanding_output = nn.Linear(
            self.attention_dim, self.understanding_dim
        )
        self.understanding_ffn_norm = nn.LayerNorm(self.understanding_dim)
        self.understanding_ffn = nn.Sequential(
            nn.Linear(
                self.understanding_dim,
                int(self.understanding_dim * float(mlp_ratio)),
            ),
            nn.GELU(),
            nn.Linear(
                int(self.understanding_dim * float(mlp_ratio)),
                self.understanding_dim,
            ),
        )

        # The four output projections are the only zero-initialized layers.
        # This makes gate=0 an exact identity without suppressing fresh
        # gradients once the stage gate opens.
        nn.init.zeros_(self.query_output.weight)
        nn.init.zeros_(self.query_output.bias)
        nn.init.zeros_(self.query_ffn[-1].weight)
        nn.init.zeros_(self.query_ffn[-1].bias)
        nn.init.zeros_(self.understanding_output.weight)
        nn.init.zeros_(self.understanding_output.bias)
        nn.init.zeros_(self.understanding_ffn[-1].weight)
        nn.init.zeros_(self.understanding_ffn[-1].bias)

    @staticmethod
    def _check_shape(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
        if not torch.is_tensor(value) or value.ndim != len(shape):
            raise ValueError(f"{name} must have shape {shape}, got {getattr(value, 'shape', None)}")
        for actual, expected in zip(value.shape, shape):
            if expected >= 0 and int(actual) != expected:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")

    def forward(
        self,
        understanding_hidden: torch.Tensor,
        query_state: torch.Tensor,
        gate: float | torch.Tensor,
        *,
        query_update: bool = True,
        u_write: bool = True,
    ) -> EarlyQueryCoDecoderOutput:
        self._check_shape("understanding_hidden", understanding_hidden, (-1, 1024, 1024))
        self._check_shape("query_state", query_state, (-1, 100, 256))
        if understanding_hidden.shape[0] != query_state.shape[0]:
            raise ValueError("EQC batch dimensions do not match")
        gate_value = float(gate.detach().item() if torch.is_tensor(gate) else gate)
        if not torch.isfinite(torch.tensor(gate_value)) or gate_value < 0.0 or gate_value > 1.0:
            raise ValueError("EQC gate must be finite and in [0,1]")
        if gate_value == 0.0:
            return EarlyQueryCoDecoderOutput(understanding_hidden, query_state)

        if not isinstance(query_update, bool) or not isinstance(u_write, bool):
            raise ValueError("query_update and u_write must be bool")

        u = understanding_hidden
        q = query_state
        u_attn = self.understanding_projection(self.understanding_norm(u))
        q_msg, _ = self.query_attention(
            self.query_norm(q), u_attn, u_attn, need_weights=False
        )
        q_new = q
        if query_update:
            q_new = q_new + gate_value * self.query_output(q_msg)
            q_new = q_new + gate_value * self.query_ffn(
                self.query_ffn_norm(q_new)
            )

        q_memory = self.query_to_understanding_norm(q_new)
        u_msg, _ = self.understanding_attention(
            u_attn, q_memory, q_memory, need_weights=False
        )
        u_new = u
        if u_write:
            write_scale = self.understanding_write_scale * gate_value
            u_new = u_new + write_scale * self.understanding_output(u_msg)
            u_new = u_new + write_scale * self.understanding_ffn(
                self.understanding_ffn_norm(u_new)
            )
        return EarlyQueryCoDecoderOutput(u_new, q_new)


__all__ = ["EarlyObjectQueryAdapter", "EarlyQueryCoDecoderOutput"]
