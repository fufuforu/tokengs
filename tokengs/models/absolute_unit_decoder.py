"""Absolute unit-based Gaussian student (token-aligned structure).

Token -> 8 shared local units -> 8 GS per unit -> COMPLETE student GS.

The student predicts every Gaussian parameter (position/opacity/scale/
rotation/color) from the shared decoder tokens only.  It never adds or
depends on the frozen old GS head output at runtime; the old head is used
exclusively as a no_grad teacher during the bootstrap stage (rendered-RGB +
low-weight per-GS distillation through a per-token Hungarian pairing).

The instance side (unit-level cross-token instance queries -> masks rendered
through the student GS) is handled by the existing ``TokenLocalUnitGrouping``
path in ``semantic_tokengs_v6``; this module only owns GS generation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AbsoluteUnitDecoder(nn.Module):
    """Complete per-token GS decoder structured through shared local units.

    Each of the 1024 tokens produces K=8 unit queries (shared query prior +
    token content readout).  A shared unit decoder maps every
    (unit, slot) pair to one complete 14-dim Gaussian, so a token always
    yields K*G = 64 Gaussians.  No old-head geometry is read: inputs are the
    decoder tokens (``gs_token_hidden``) only.
    """

    def __init__(
        self,
        token_dim: int = 1024,
        units_per_token: int = 8,
        gaussians_per_unit: int = 8,
        feat_dim: int = 256,
        slot_emb_dim: int = 32,
    ):
        super().__init__()
        self.units_per_token = int(units_per_token)
        self.gaussians_per_unit = int(gaussians_per_unit)
        self.feat_dim = int(feat_dim)
        self.token_dim = int(token_dim)

        self.tok_norm = nn.LayerNorm(int(token_dim))
        self.tok_proj = nn.Linear(int(token_dim), int(feat_dim))
        # Shared local-unit queries (same K units for every token; content
        # readout below makes them token-specific/local).
        self.unit_queries = nn.Parameter(
            0.02 * torch.randn(self.units_per_token, int(feat_dim))
        )
        self.unit_readout = nn.Sequential(
            nn.Linear(int(feat_dim) * 2, int(feat_dim)),
            nn.GELU(),
            nn.Linear(int(feat_dim), int(feat_dim)),
        )
        # Per-slot embedding shared across all units/tokens.
        self.slot_emb = nn.Parameter(
            0.02
            * torch.randn(
                self.gaussians_per_unit, int(slot_emb_dim)
            )
        )
        dec_in = (
            int(feat_dim)  # unit feature
            + 3  # unit center (predicted below from the unit feature)
            + int(slot_emb_dim)
        )
        self.gs_decoder = nn.Sequential(
            nn.LayerNorm(dec_in),
            nn.Linear(dec_in, int(feat_dim)),
            nn.GELU(),
            nn.Linear(int(feat_dim), int(feat_dim)),
            nn.GELU(),
            nn.Linear(int(feat_dim), 14),
        )
        # Unit center is a small MLP on the unit feature.
        self.center_mlp = nn.Sequential(
            nn.LayerNorm(int(feat_dim)),
            nn.Linear(int(feat_dim), int(feat_dim)),
            nn.GELU(),
            nn.Linear(int(feat_dim), 3),
        )

    @property
    def gaussians_per_token(self) -> int:
        return self.units_per_token * self.gaussians_per_unit

    def form_units(self, gs_token_hidden: torch.Tensor) -> torch.Tensor:
        """Form token-local units without running the Gaussian decoder."""
        if gs_token_hidden.ndim != 3 or gs_token_hidden.shape[-1] != self.token_dim:
            raise ValueError(
                "AbsoluteUnitDecoder.form_units expects [B,T,"
                f"{self.token_dim}], got {tuple(gs_token_hidden.shape)}"
            )
        batch_size, token_count, _ = gs_token_hidden.shape
        hidden = gs_token_hidden.float()
        hp = self.tok_proj(self.tok_norm(hidden))
        k = self.units_per_token
        f = self.feat_dim
        q0 = self.unit_queries.unsqueeze(0).unsqueeze(0).expand(
            batch_size, token_count, k, f
        )
        hp_exp = hp.unsqueeze(2).expand(batch_size, token_count, k, f)
        return self.unit_readout(torch.cat([q0, hp_exp], dim=-1))

    def decode_units(
        self, unit_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode [B,T,K,F] units into complete Gaussians and centers."""
        if unit_features.ndim != 4 or unit_features.shape[-1] != self.feat_dim:
            raise ValueError(
                "AbsoluteUnitDecoder.decode_units expects [B,T,K,"
                f"{self.feat_dim}], got {tuple(unit_features.shape)}"
            )
        batch_size, token_count, units_per_token, feat_dim = unit_features.shape
        if units_per_token != self.units_per_token:
            raise ValueError(
                f"decode_units expects K={self.units_per_token}, got "
                f"{units_per_token}"
            )
        q = unit_features
        center = self.center_mlp(q)
        g = self.gaussians_per_unit
        slot = self.slot_emb.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
            batch_size, token_count, units_per_token, g, -1
        )
        q_exp = q.unsqueeze(3).expand(
            batch_size, token_count, units_per_token, g, feat_dim
        )
        c_exp = center.unsqueeze(3).expand(
            batch_size, token_count, units_per_token, g, 3
        )
        raw = self.gs_decoder(torch.cat([q_exp, c_exp, slot], dim=-1))
        pos = raw[..., :3]
        opacity = torch.sigmoid(raw[..., 3:4])
        log_scale = raw[..., 4:7].clamp(-8.0, 8.0)
        scale = log_scale.exp().clamp_min(1e-6)
        quat = F.normalize(raw[..., 7:11], dim=-1, eps=1e-6)
        color = torch.sigmoid(raw[..., 11:14])
        new_gs = torch.cat([pos, opacity, scale, quat, color], dim=-1)
        return (
            new_gs.reshape(batch_size, token_count * self.gaussians_per_token, 14),
            center,
        )

    def forward(
        self, gs_token_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (student_gaussians [B,N,14], unit_feat [B,T,K,F], unit_center
        [B,T,K,3])."""
        q = self.form_units(gs_token_hidden)
        gaussians, center = self.decode_units(q)
        return gaussians, q, center


def pair_teacher_student(teacher_gs: torch.Tensor, student_gs: torch.Tensor) -> torch.Tensor:
    """Hungarian-pair teacher GS to student slots per token by position.

    teacher_gs/student_gs: [B, T*P, 14].  Returns teacher_gs reordered to
    match the student slot order ([B, T*P, 14], detached).  Non-differentiable
    matching is fine: the loss flows through the student only.
    """
    from scipy.optimize import linear_sum_assignment

    batch_size, n_gs, _ = teacher_gs.shape
    p = 64
    token_count = n_gs // p
    t_pos = teacher_gs[..., :3].detach().float().cpu().numpy()
    s_pos = student_gs[..., :3].detach().float().cpu().numpy()
    out = teacher_gs.detach().clone()
    for b in range(batch_size):
        for t in range(token_count):
            t_seg = t_pos[b, t * p : (t + 1) * p]
            s_seg = s_pos[b, t * p : (t + 1) * p]
            cost = np.sqrt(
                ((t_seg[:, None, :] - s_seg[None, :, :]) ** 2).sum(-1)
            )
            t_idx, _ = linear_sum_assignment(cost)
            out[b, t * p : (t + 1) * p] = teacher_gs[b][
                t * p + t_idx
            ]
    return out


def teacher_gs_distill_loss(
    teacher_gs: torch.Tensor,
    student_gs: torch.Tensor,
    teacher_rgb: torch.Tensor | None,
    student_rgb: torch.Tensor | None,
    rgb_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Low-weight GS distillation + optional rendered-RGB bootstrap."""
    paired = pair_teacher_student(teacher_gs, student_gs)
    w = torch.ones(
        student_gs.shape[-1], device=student_gs.device, dtype=student_gs.dtype
    )
    w[4:7] = 10.0  # scale drift is the most PSNR-sensitive
    w[7:11] = 0.2
    w[11:] = 0.0  # color is covered by the RGB term
    loss_gs = (
        (student_gs - paired).square() * w.view(1, 1, -1)
    ).mean()
    loss_rgb = torch.zeros((), device=student_gs.device)
    if (
        rgb_weight > 0
        and teacher_rgb is not None
        and student_rgb is not None
    ):
        loss_rgb = (
            student_rgb.clamp(0, 1) - teacher_rgb.clamp(0, 1)
        ).square().mean()
    return loss_gs, loss_rgb, paired
