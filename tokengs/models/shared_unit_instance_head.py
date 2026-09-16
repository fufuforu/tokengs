"""True-Shared instance head for the guarded joint baseline.

The only unit representation in the model is ``q_abs`` produced by
``AbsoluteUnitDecoder`` (token-aligned Shared Local Unit formation).  This
head consumes that same tensor:

    q_abs [B, T, K, F]  (T=1024 tokens, K=8 units, F=256)
      |  optional LayerNorm + residual MLP adapter (per unit, same tensor)
      v
    flattened units [B, T*K=8192, F]
      |  100 learnable group queries cross-attend over ALL units
      v
    unit assignment [B, 8192, G+1]  ->  pi_unit [B, T, K, G+1]
      |  pi_gs = expand over the 8 fixed GS slots of each unit
      v
    [B, T*K*8=65536, G+1]

The head never rebuilds units from decoder tokens or Student GS geometry and
contains no ``gs_feature_mlp`` / ``unit_queries`` / ``unit_layers`` /
GS->unit soft-grouping parameters.  Geometry detachment for the instance
render is applied by the caller, so instance gradients can reach the unit
formation (q_abs producer) but never the GS geometry decoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TSHCrossBlock(nn.Module):
    """Group-query self-attention + cross-attention over all shared units."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        mlp_hidden: int = 1024,
    ):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=0.0
        )
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=0.0
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        units: torch.Tensor,
    ) -> torch.Tensor:
        # Group queries refine among themselves.
        q = self.norm0(queries)
        attn, _ = self.self_attn(q, q, q)
        queries = queries + attn
        # Then every group query reads all T*K units.
        q = self.norm1(queries)
        attn, _ = self.cross_attn(q, units, units)
        queries = queries + attn
        q = self.norm2(queries)
        return queries + self.mlp(q)


class GroupQueryMemoryRefiner(nn.Module):
    """Assignment-conditioned refinement of the existing group queries."""

    def __init__(self, dim: int, num_groups: int, num_heads: int = 8, rounds: int = 2):
        super().__init__()
        self.dim = int(dim)
        self.num_groups = int(num_groups)
        self.rounds = int(rounds)
        self.memory_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.rounds)])
        self.query_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.rounds)])
        self.memory_attn = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.0)
            for _ in range(self.rounds)
        ])
        self.self_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.rounds)])
        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.0)
            for _ in range(self.rounds)
        ])
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.rounds)])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
            for _ in range(self.rounds)
        ])
        self.unit_residual_proj = nn.Linear(dim, dim, bias=False)
        self.group_residual_proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.group_residual_proj.weight)

    def forward(self, groups, units, base_logits):
        assignment = F.softmax(base_logits[..., : self.num_groups].float(), dim=-1)
        unit_memory = torch.einsum("bng,bnd->bgd", assignment, units)
        unit_mass = assignment.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        unit_memory = unit_memory / unit_mass
        refined = groups
        for memory_norm, query_norm, memory_attn, self_norm, self_attn, ffn_norm, ffn in zip(
            self.memory_norm, self.query_norm, self.memory_attn,
            self.self_norm, self.self_attn, self.ffn_norm, self.ffn,
        ):
            memory = memory_norm(unit_memory)
            delta, _ = memory_attn(query_norm(refined), memory, memory)
            refined = refined + delta
            query = self_norm(refined)
            delta, _ = self_attn(query, query, query)
            refined = refined + delta
            refined = refined + ffn(ffn_norm(refined))
        unit_proj = F.normalize(self.unit_residual_proj(units), dim=-1)
        group_proj = F.normalize(self.group_residual_proj(refined), dim=-1)
        residual = torch.einsum("bnd,bgd->bng", unit_proj, group_proj)
        residual = torch.cat(
            [residual, residual.new_zeros(*residual.shape[:-1], 1)], dim=-1
        )
        return refined, residual


class SharedUnitInstanceHead(nn.Module):
    """Instance query/assignment head on the single shared unit tensor."""

    def __init__(
        self,
        unit_dim: int = 256,
        units_per_token: int = 8,
        gaussians_per_unit: int = 8,
        num_tokens: int = 1024,
        num_groups: int = 100,
        num_heads: int = 8,
        num_layers: int = 2,
        assignment_temperature: float = 10.0,
        query_memory_refine: bool = False,
        query_memory_refine_rounds: int = 2,
    ):
        super().__init__()
        self.unit_dim = int(unit_dim)
        self.units_per_token = int(units_per_token)
        self.gaussians_per_unit = int(gaussians_per_unit)
        self.num_tokens = int(num_tokens)
        self.num_groups = int(num_groups)
        self.query_memory_refine = bool(query_memory_refine)
        # Optional per-unit adapter: z_u = q_u + MLP(LN(q_u)).
        self.q_norm = nn.LayerNorm(self.unit_dim)
        self.q_adapter = nn.Sequential(
            nn.Linear(self.unit_dim, self.unit_dim),
            nn.GELU(),
            nn.Linear(self.unit_dim, self.unit_dim),
        )
        self.group_tokens = nn.Parameter(
            0.02
            * torch.randn(self.num_groups, self.unit_dim)
        )
        self.layers = nn.ModuleList(
            [
                _TSHCrossBlock(
                    dim=self.unit_dim,
                    num_heads=int(num_heads),
                    mlp_hidden=self.unit_dim * 4,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.group_norm = nn.LayerNorm(self.unit_dim)
        self.unit_assignment_proj = nn.Linear(
            self.unit_dim, self.unit_dim, bias=False
        )
        self.group_assignment_proj = nn.Linear(
            self.unit_dim, self.unit_dim, bias=False
        )
        self.log_assignment_temperature = nn.Parameter(
            torch.tensor(float(assignment_temperature)).log()
        )
        self.void_head = nn.Linear(self.unit_dim, 1)
        nn.init.zeros_(self.void_head.weight)
        nn.init.zeros_(self.void_head.bias)
        self.query_memory_refiner = None
        if self.query_memory_refine:
            self.query_memory_refiner = GroupQueryMemoryRefiner(
                dim=self.unit_dim,
                num_groups=self.num_groups,
                num_heads=int(num_heads),
                rounds=int(query_memory_refine_rounds),
            )
        self._last_q_in = None
        self._last_pi_unit = None
        self._last_pi_gs = None

    def reset_parameters_fresh(self) -> None:
        """Fresh random re-init (used when starting from full3 abs weights)."""
        with torch.no_grad():
            # nn.MultiheadAttention only exposes the private
            # ``_reset_parameters`` (xavier in_proj_weight / uniform bias),
            # which a public ``reset_parameters`` walk would miss.
            for module in self.modules():
                if isinstance(module, nn.MultiheadAttention):
                    reset = getattr(module, "_reset_parameters", None)
                    if reset is not None:
                        reset()
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
            self.group_tokens.copy_(
                0.02 * torch.randn_like(self.group_tokens)
            )
            nn.init.zeros_(self.void_head.weight)
            nn.init.zeros_(self.void_head.bias)

    def forward(
        self,
        q_abs: torch.Tensor,
        refine_gate: float = 0.0,
        return_base_states: bool = False,
        query_state_override: torch.Tensor | None = None,
    ) -> dict:
        """q_abs: [B, T, K, F]. Returns pi_unit and pi_gs plus stats."""
        batch_size, token_count, units_per_token, feat_dim = q_abs.shape
        if token_count != self.num_tokens:
            raise RuntimeError(
                f"SharedUnitInstanceHead expects {self.num_tokens} tokens, "
                f"got {token_count}"
            )
        if units_per_token != self.units_per_token:
            raise RuntimeError(
                f"SharedUnitInstanceHead expects {self.units_per_token} "
                f"units/token, got {units_per_token}"
            )
        self._last_q_in = q_abs
        n_units = token_count * units_per_token
        q = q_abs.reshape(batch_size, n_units, feat_dim)
        # Per-unit adapter: same tensor, no new clustering/unit formation.
        z = q + self.q_adapter(self.q_norm(q))

        if query_state_override is None:
            groups = self.get_object_query_seed(batch_size)
        else:
            if tuple(query_state_override.shape) != (batch_size, self.num_groups, self.unit_dim):
                raise ValueError(
                    "query_state_override must be "
                    f"[B,{self.num_groups},{self.unit_dim}], got "
                    f"{tuple(query_state_override.shape)}"
                )
            groups = query_state_override
        for layer in self.layers:
            groups = layer(groups, z)
        groups = self.group_norm(groups)

        u = F.normalize(self.unit_assignment_proj(z), dim=-1)
        g = F.normalize(self.group_assignment_proj(groups), dim=-1)
        temperature = self.log_assignment_temperature.exp().clamp(1.0, 100.0)
        group_logits = temperature * torch.einsum(
            "bnd,bgd->bng", u, g
        )  # [B,N,G]
        void_logits = self.void_head(z)  # [B,N,1]
        base_unit_logits = torch.cat([group_logits, void_logits], dim=-1)
        unit_logits = base_unit_logits
        residual_logits = torch.zeros_like(base_unit_logits)
        refined_groups = groups
        if self.query_memory_refiner is not None and float(refine_gate) > 0.0:
            refined_groups, residual_logits = self.query_memory_refiner(
                groups, z, base_unit_logits
            )
            unit_logits = base_unit_logits + float(refine_gate) * residual_logits
        pi_unit_flat = F.softmax(unit_logits.float(), dim=-1)
        pi_unit = pi_unit_flat.view(
            batch_size, token_count, units_per_token, self.num_groups + 1
        )  # [B,T,K,G+1]

        # Strict mapping: each of the 8 fixed GS slots of unit u receives
        # exactly pi_unit[..., u, :].
        gs = self.gaussians_per_unit
        pi_gs = (
            pi_unit.unsqueeze(-2)
            .expand(
                batch_size,
                token_count,
                units_per_token,
                gs,
                self.num_groups + 1,
            )
            .reshape(
                batch_size,
                token_count * units_per_token * gs,
                self.num_groups + 1,
            )
        )
        pi_unit_expanded = pi_unit.unsqueeze(-2).expand(
            batch_size,
            token_count,
            units_per_token,
            gs,
            self.num_groups + 1,
        )
        assert torch.equal(
            pi_gs.reshape(
                batch_size,
                token_count,
                units_per_token,
                gs,
                self.num_groups + 1,
            ),
            pi_unit_expanded,
        ), "unit -> GS slot expansion mismatch"
        # Runtime blockwise assertion: GS of one unit share probabilities.
        per_block = pi_gs.reshape(
            batch_size, token_count, units_per_token, gs, -1
        )
        if not bool(
            (per_block - per_block[..., :1, :]).abs().max() < 1e-6
        ):
            raise RuntimeError(
                "pi_gs is not constant within a unit's 8 GS slots"
            )
        self._last_pi_unit = pi_unit
        self._last_pi_gs = pi_gs
        stats = {
            "unit_logits_max": unit_logits.max().detach(),
            "unit_logits_min": unit_logits.min().detach(),
            "base_unit_logits_max_diff": (unit_logits - base_unit_logits).abs().max().detach(),
            "query_memory_refine_gate": torch.as_tensor(
                float(refine_gate), device=unit_logits.device
            ),
            "assignment_temperature": temperature.detach(),
            "void_share": pi_unit[..., -1].mean().detach(),
            "max_group_prob_share": pi_unit[..., :-1].max(dim=-1).values.mean().detach(),
        }
        return {
            "pi_unit": pi_unit,
            "pi_gs": pi_gs,
            "unit_logits": unit_logits,
            "adapter_z": z,
            "groups": groups,
            "refined_groups": refined_groups,
            "base_unit_logits": base_unit_logits,
            "residual_logits": residual_logits,
            **{key: value for key, value in stats.items()},
        }

    def get_object_query_seed(self, batch_size: int) -> torch.Tensor:
        """Expand the native 100-query parameter without registering a copy."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return self.group_tokens.unsqueeze(0).expand(batch_size, -1, -1)
