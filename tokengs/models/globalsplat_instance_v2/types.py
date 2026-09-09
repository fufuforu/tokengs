from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CandidateLayout:
    stage: int
    mix: float
    active_per_slot: int
    gate_logits_full: torch.Tensor
    weights_current: torch.Tensor
    weights_previous: torch.Tensor | None


@dataclass(frozen=True)
class TriStreamOutput:
    scene_geo: torch.Tensor
    scene_tex: torch.Tensor
    scene_ins: torch.Tensor
    aux: torch.Tensor
    scale_token: torch.Tensor


@dataclass(frozen=True)
class InstanceDecode:
    gaussian_embeddings: torch.Tensor
    gaussian_objectness_logits: torch.Tensor
    object_queries: torch.Tensor
    assignment_logits: torch.Tensor
    assignment_probabilities: torch.Tensor


@dataclass(frozen=True)
class SceneAssignment:
    batch_index: int
    query_to_instance_id: dict[int, int]
    instance_visible_views: dict[int, tuple[int, ...]]
    dropped_instance_ids: tuple[int, ...]


@dataclass(frozen=True)
class GSIModelOutput:
    gaussians: Any
    rendered_rgb: torch.Tensor
    rendered_alpha: torch.Tensor
    rendered_depth: torch.Tensor | None
    layout: CandidateLayout
    instance: InstanceDecode | None
    rendered_assignment: torch.Tensor | None
