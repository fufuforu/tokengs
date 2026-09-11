"""Lazy, isolated GlobalSplat-Instance v2 integration."""

from .model import GlobalSplatInstanceV2
from .semantic_query_head import SemanticQueryHead
from .types import CandidateLayout, GSIModelOutput, InstanceDecode, SceneAssignment, TriStreamOutput

__all__ = [
    "GlobalSplatInstanceV2",
    "SemanticQueryHead",
    "CandidateLayout",
    "GSIModelOutput",
    "InstanceDecode",
    "SceneAssignment",
    "TriStreamOutput",
]
