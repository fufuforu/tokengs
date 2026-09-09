from __future__ import annotations

import torch
import torch.nn.functional as F

from .types import CandidateLayout


def _weights(logits: torch.Tensor, groups: int, tau: float) -> torch.Tensor:
    b, p, m, one = logits.shape
    if one != 1 or m % groups:
        raise ValueError(f"invalid gate shape/groups: {tuple(logits.shape)}, {groups}")
    return F.softmax(logits.view(b, p, groups, m // groups, 1) / float(tau), dim=3)


def capture_candidate_layout(gaussian_decoder: torch.nn.Module, scene_geo: torch.Tensor) -> CandidateLayout:
    if scene_geo.ndim != 3:
        raise ValueError("scene_geo must be [B,2048,512]")
    stage = int(gaussian_decoder.stage)
    mix = float(gaussian_decoder.mix)
    max_candidates = int(gaussian_decoder.M_max)
    logits = gaussian_decoder.gate_readout(scene_geo).view(scene_geo.shape[0], scene_geo.shape[1], max_candidates, 1)
    groups = 1 << stage
    current = _weights(logits, groups, float(gaussian_decoder.gate_tau))
    previous = _weights(logits, groups >> 1, float(gaussian_decoder.gate_tau)) if stage > 0 else None
    return CandidateLayout(stage, mix, groups, logits, current, previous)


def _reduce(x: torch.Tensor, groups: int, weights: torch.Tensor) -> torch.Tensor:
    b, p, m, d = x.shape
    if m == groups:
        return x
    return (weights * x.view(b, p, groups, m // groups, d)).sum(dim=3)


def reduce_aligned_candidates(x_full: torch.Tensor, layout: CandidateLayout, *,
                             gate_gradient_scale: float, normalize: bool = False) -> torch.Tensor:
    if x_full.ndim != 4 or x_full.shape[:3] != layout.gate_logits_full.shape[:3]:
        raise ValueError(f"x_full must be [B,2048,16,D], got {tuple(x_full.shape)}")
    def scaled(w: torch.Tensor) -> torch.Tensor:
        return w.detach() + float(gate_gradient_scale) * (w - w.detach())
    current = _reduce(x_full, layout.active_per_slot, scaled(layout.weights_current))
    if layout.stage > 0:
        if layout.weights_previous is None:
            raise ValueError("stage>0 requires previous weights")
        previous = _reduce(x_full, layout.active_per_slot // 2, scaled(layout.weights_previous))
        previous = previous.repeat_interleave(2, dim=2)
        result = (1.0 - layout.mix) * previous + layout.mix * current
    else:
        result = current
    result = result.reshape(result.shape[0], -1, result.shape[-1])
    return F.normalize(result, dim=-1, eps=1e-6) if normalize else result
