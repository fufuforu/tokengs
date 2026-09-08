"""Token-Aligned Reconstruction--Instance Joint Units (TA-RIU-v1)."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from tokengs.models.ga_idu import MemoryResampler


class SharedUnitMixer(nn.Module):
    """Memory-conditioned mixer preserving the [token, local-unit] axes."""

    def __init__(self, input_dim=1024, dim=256, memories=256, heads=8):
        super().__init__()
        self.memory_resampler = MemoryResampler(input_dim, dim, memories, heads)
        self.q_norm = nn.LayerNorm(dim)
        self.m_norm = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.out_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, q_abs: torch.Tensor, encoder_values: torch.Tensor, gate: float):
        memory = self.memory_resampler(encoder_values)
        b, t, k, d = q_abs.shape
        units = q_abs.reshape(b, t * k, d)
        delta, _ = self.cross(self.q_norm(units), self.m_norm(memory), self.m_norm(memory))
        delta = delta + self.ffn(self.ffn_norm(units + delta))
        delta = self.out(self.out_norm(delta)).reshape_as(q_abs)
        z = q_abs + float(gate) * delta
        return z, memory, delta


class GeometryResidualHead(nn.Module):
    """Bounded xyz/log-scale/rotation-tangent/opacity residual per GS."""

    def __init__(self, dim=256, hidden=256, gaussians_per_unit=8):
        super().__init__()
        self.gaussians_per_unit = int(gaussians_per_unit)
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, self.gaussians_per_unit * 10))
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        b, t, k, d = z.shape
        return self.net(z).reshape(b, t, k, self.gaussians_per_unit, 10)


class AppearanceResidualHead(nn.Module):
    """Bounded RGB residual per GS."""

    def __init__(self, dim=256, hidden=256, gaussians_per_unit=8):
        super().__init__()
        self.gaussians_per_unit = int(gaussians_per_unit)
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, self.gaussians_per_unit * 3))
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        b, t, k, d = z.shape
        return self.net(z).reshape(b, t, k, self.gaussians_per_unit, 3)


def quat_mul(q, r):
    qx, qy, qz, qw = q.unbind(-1)
    rx, ry, rz, rw = r.unbind(-1)
    return torch.stack((
        qw * rx + qx * rw + qy * rz - qz * ry,
        qw * ry - qx * rz + qy * rw + qz * rx,
        qw * rz + qx * ry - qy * rx + qz * rw,
        qw * rw - qx * rx - qy * ry - qz * rz,
    ), dim=-1)


def apply_joint_residual(base_gaussians, geo, app, geo_gate, app_gate, xyz_scale=0.02, log_scale_scale=0.05, rot_scale=0.05, opacity_scale=0.05, color_scale=0.05):
    b, n, _ = base_gaussians.shape
    p = geo.shape[3]
    base = base_gaussians.reshape(b, -1, p, 14)
    # The residual heads return [B,T,K,G,*], while the absolute GS decoder
    # returns [B,T*K*G,14].  Flatten only the token/unit axes so every local
    # unit remains aligned with its fixed G=8 Gaussian slots.
    geo = torch.tanh(geo).reshape(b, -1, p, 10)
    app = torch.tanh(app).reshape(b, -1, p, 3)
    out = base.clone()
    out[..., :3] = base[..., :3] + float(geo_gate) * xyz_scale * geo[..., :3]
    out[..., 4:7] = base[..., 4:7] * torch.exp(float(geo_gate) * log_scale_scale * geo[..., 3:6])
    tangent = float(geo_gate) * rot_scale * geo[..., 6:9]
    zero = torch.zeros_like(tangent[..., :1])
    dq = F.normalize(torch.cat((0.5 * tangent, torch.ones_like(zero)), dim=-1), dim=-1)
    out[..., 7:11] = F.normalize(quat_mul(base[..., 7:11], dq), dim=-1)
    out[..., 3:4] = (base[..., 3:4] + float(geo_gate) * opacity_scale * geo[..., 9:10]).clamp(0.0, 1.0)
    out[..., 11:14] = (base[..., 11:14] + float(app_gate) * color_scale * app).clamp(0.0, 1.0)
    return out.reshape(b, n, 14)
