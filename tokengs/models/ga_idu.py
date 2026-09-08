"""GA-IDU-1: memory-backed, BaseTSH-channel-aligned instance residuals.

This module deliberately has no geometry, appearance, matching, or score
head.  Its only input anchor is the batch-specific frozen BaseTSH state.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MemoryResampler(nn.Module):
    def __init__(self, input_dim: int, dim: int = 256, memories: int = 256, heads: int = 8):
        super().__init__()
        self.proj = nn.Linear(input_dim, dim)
        self.latents = nn.Parameter(torch.randn(memories, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 4:
            # EncoderLatent.values is [B, heads, sequence, head_dim].
            # Reconstruct the projected full-channel sequence; flattening
            # heads into sequence would mix attention heads with tokens.
            b, heads, sequence, head_dim = values.shape
            values = values.permute(0, 2, 1, 3).contiguous().reshape(
                b, sequence, heads * head_dim
            )
        if values.ndim != 3:
            raise ValueError(f"encoder values must be [B,M,C] or [B,V,M,C], got {tuple(values.shape)}")
        memory = self.proj(values)
        q = self.latents.unsqueeze(0).expand(values.shape[0], -1, -1)
        delta, _ = self.attn(self.norm_q(q), self.norm_kv(memory), self.norm_kv(memory))
        q = q + delta
        return q + self.ffn(self.norm_ffn(q))


class InstanceDecoder(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, layers: int = 1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.ModuleList(
            [nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0) for _ in range(layers)]
        )
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.ffn = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)) for _ in range(layers)]
        )

    def forward(self, units: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = units.reshape(units.shape[0], -1, units.shape[-1])
        for attn, norm, ffn_norm, ffn in zip(self.attn, [self.norm] * len(self.attn), self.ffn_norm, self.ffn):
            y, _ = attn(norm(x), memory, memory)
            x = x + y
            x = x + ffn(ffn_norm(x))
        return x.reshape_as(units)


class GroupResidualDecoder(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.m_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, groups: torch.Tensor, units: torch.Tensor) -> torch.Tensor:
        delta, _ = self.attn(self.q_norm(groups), self.m_norm(units), self.m_norm(units))
        return delta + self.ffn(self.ffn_norm(groups + delta))


class AssignmentResidual(nn.Module):
    def __init__(self, dim: int = 256, groups: int = 100):
        super().__init__()
        self.unit = nn.Linear(dim, dim, bias=False)
        self.group = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, units: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        u = F.normalize(self.unit(units), dim=-1)
        g = F.normalize(self.group(groups), dim=-1)
        # Pairwise unit/group projection is zero-initialized, so the residual
        # is exactly zero at step 0 while its output projection can learn.
        pair = torch.einsum("bnd,bgd->bng", u, g).unsqueeze(-1)
        return self.out(pair).squeeze(-1)


class GAIDU1(nn.Module):
    """GA-IDU-1 forward; GA-IDU-2/3 components are intentionally absent."""

    def __init__(self, input_dim: int, dim: int = 256, groups: int = 100, heads: int = 8):
        super().__init__()
        self.memory_resampler = MemoryResampler(input_dim, dim, 256, heads)
        self.instance_decoder = InstanceDecoder(dim, heads, 1)
        self.group_residual_decoder = GroupResidualDecoder(dim, heads)
        self.assignment_residual = AssignmentResidual(dim, groups)
        self.groups = groups

    def forward(self, q_abs, encoder_values, base_groups, base_logits, gate: float = 0.0):
        memory = self.memory_resampler(encoder_values)
        z_inst = self.instance_decoder(q_abs, memory)
        anchored_groups = base_groups.detach()
        new_groups = anchored_groups + self.group_residual_decoder(
            anchored_groups, z_inst.reshape(z_inst.shape[0], -1, z_inst.shape[-1])
        )
        delta = self.assignment_residual(z_inst.reshape(z_inst.shape[0], -1, z_inst.shape[-1]), new_groups)
        final_logits = torch.cat((base_logits[..., : self.groups].detach() + float(gate) * delta, base_logits[..., self.groups:self.groups + 1].detach()), dim=-1)
        pi_flat = F.softmax(final_logits.float(), dim=-1)
        pi_unit = pi_flat.reshape(q_abs.shape[0], q_abs.shape[1], q_abs.shape[2], self.groups + 1)
        pi_gs = pi_unit.unsqueeze(-2).expand(*pi_unit.shape[:3], 8, pi_unit.shape[-1]).reshape(pi_unit.shape[0], -1, pi_unit.shape[-1])
        return {"memory": memory, "z_inst": z_inst, "groups": new_groups, "delta_group_logits": delta, "unit_logits": final_logits, "pi_unit": pi_unit, "pi_gs": pi_gs}
