"""Lightweight per-GS slot refinement on top of the True-Shared units.

The only instance identity source remains the unit-level assignment computed
by ``SharedUnitInstanceHead`` from the single shared ``q_abs``:

    unit_logits [B,T,K,G+1]
        |  broadcast to the 8 fixed GS slots of each unit
        v
    coarse per-GS logits [B,T*K*8,G+1]
        +
    alpha * residual_gs_logits     (alpha zero-init, gate warmed from 0)
        v
    final per-GS logits [B,T*K*8,G+1]
        |  softmax
        v
    pi_gs (rendered instance masks)

The residual branch is zero-initialized (final Linear bias/weight = 0) and
its output is passed through tanh, so at gate=0 the unit's 8 GS share
identical assignment and the forward matches the previous checkpoint
exactly.  All Student-GS attributes are consumed detached, so instance
BCE/Dice never reach ``center_mlp`` / ``gs_decoder`` / the reconstruction
slot embedding; the only new embedding lives inside this head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerGSSlotRefineHead(nn.Module):
    def __init__(
        self,
        unit_dim: int = 256,
        num_groups: int = 100,
        units_per_token: int = 8,
        gaussians_per_unit: int = 8,
        num_tokens: int = 1024,
        hidden: int = 128,
        slot_emb_dim: int = 32,
        gs_attr_dim: int = 7,
        use_group_context: bool = True,
    ):
        super().__init__()
        self.unit_dim = int(unit_dim)
        self.num_groups = int(num_groups)
        self.units_per_token = int(units_per_token)
        self.gaussians_per_unit = int(gaussians_per_unit)
        self.num_tokens = int(num_tokens)
        self.hidden = int(hidden)
        self.use_group_context = bool(use_group_context)
        # Per-slot embedding inside this head (never touches the
        # reconstruction-side slot_emb of the absolute GS decoder).
        self.slot_emb = nn.Parameter(
            0.02
            * torch.randn(
                int(gaussians_per_unit), int(slot_emb_dim)
            )
        )
        self.attr_norm = nn.LayerNorm(int(gs_attr_dim))
        self.attr_proj = nn.Sequential(
            nn.Linear(int(gs_attr_dim), int(hidden) // 4),
            nn.GELU(),
        )
        self.q_proj = nn.Sequential(
            nn.Linear(int(unit_dim), int(hidden) // 2),
            nn.GELU(),
        )
        self.slot_proj = nn.Sequential(
            nn.Linear(int(slot_emb_dim), int(hidden) // 8),
            nn.GELU(),
        )
        if self.use_group_context:
            self.group_proj = nn.Sequential(
                nn.Linear(int(unit_dim), int(hidden) // 8),
                nn.GELU(),
            )
            in_dim = (
                int(hidden) // 2
                + int(hidden) // 4
                + int(hidden) // 8
                + int(hidden) // 8
            )
        else:
            in_dim = int(hidden) // 2 + int(hidden) // 4 + int(hidden) // 8
        self.head = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.GELU(),
        )
        self.logit_proj = nn.Linear(int(hidden), int(num_groups) + 1)
        # Zero init: residual is exactly 0 at step 0.
        nn.init.zeros_(self.logit_proj.weight)
        nn.init.zeros_(self.logit_proj.bias)
        # Gate: the trainer ramps gate_eff 0 -> 1 over the first
        # tsh_per_gs_ramp_steps optimizer steps, which alone guarantees the
        # step-0 identity.  alpha_param is a learnable logit-scale
        # calibration factor (init 1) so the residual branch receives a
        # non-zero gradient as soon as the gate opens (zero * zero would
        # otherwise dead-lock the residual branch).
        self.alpha_param = nn.Parameter(torch.ones(1))
        self._last_unit_logits = None
        self._last_residual = None

    def reset_parameters_fresh(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.logit_proj.weight)
            nn.init.zeros_(self.logit_proj.bias)
            self.alpha_param.fill_(1.0)

    def forward(
        self,
        q_abs: torch.Tensor,
        unit_logits: torch.Tensor,
        group_features: torch.Tensor | None,
        student_gaussians: torch.Tensor,
        gate_eff: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Return (final_gs_logits, residual_gs_logits, alpha_eff)."""
        batch_size, token_count, units_per_token, _ = q_abs.shape
        gs = self.gaussians_per_unit
        n_units = token_count * units_per_token
        n_gs = n_units * gs
        if q_abs.shape[0] != batch_size or n_gs != student_gaussians.shape[1]:
            raise RuntimeError(
                f"PerGSSlotRefineHead shape mismatch: q={tuple(q_abs.shape)} "
                f"gs={tuple(student_gaussians.shape)}"
            )
        # q feature per GS: unit u repeated over its gs slots (order t,u,s).
        q_unit = q_abs.reshape(batch_size, n_units, self.unit_dim)
        q_gs = q_unit.repeat_interleave(gs, dim=1)  # [B,N,F]
        q_feat = self.q_proj(q_gs).float()

        slot_idx = (
            torch.arange(n_gs, device=q_abs.device) % gs
        )
        slot_feat = self.slot_proj(
            self.slot_emb[slot_idx].unsqueeze(0).expand(
                batch_size, -1, -1
            )
        ).float()

        gs_attr = student_gaussians.detach()
        attr = torch.cat(
            [
                gs_attr[..., :3],  # means
                gs_attr[..., 4:7],  # scales
                gs_attr[..., 3:4],  # opacity
            ],
            dim=-1,
        ).float()
        attr_feat = self.attr_proj(self.attr_norm(attr)).float()

        feats = [q_feat, attr_feat, slot_feat]
        if self.use_group_context:
            # Per-unit group-query context: soft-pool the G group features
            # with the (detached) coarse unit assignment, then repeat over
            # the unit's GS slots.
            if group_features is None:
                raise RuntimeError(
                    "PerGSSlotRefineHead requires group_features when "
                    "use_group_context=True"
                )
            unit_logits_4d = unit_logits.reshape(
                batch_size, token_count, units_per_token, self.num_groups + 1
            )
            p = F.softmax(
                unit_logits_4d[..., :-1].float(), dim=-1
            ).detach()
            ctx = torch.einsum(
                "btkg,bgf->btkf", p, group_features.float()
            )
            ctx_gs = (
                ctx.reshape(batch_size, n_units, self.unit_dim)
                .repeat_interleave(gs, dim=1)
            )
            feats.append(self.group_proj(ctx_gs).float())
        fused = self.head(torch.cat(feats, dim=-1)).float()
        residual = torch.tanh(self.logit_proj(fused)).float()
        alpha_val = self.alpha_param.clamp(-5.0, 5.0)
        alpha_eff_scalar = float(alpha_val.item()) * float(gate_eff)
        coarse = unit_logits.reshape(
            batch_size, n_units, self.num_groups + 1
        ).repeat_interleave(gs, dim=1)
        final_logits = coarse + alpha_val * float(gate_eff) * residual
        self._last_unit_logits = unit_logits.detach()
        self._last_residual = residual.detach()
        return final_logits, residual, alpha_eff_scalar
