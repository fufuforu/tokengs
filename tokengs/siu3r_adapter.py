"""TokenGS -> SIU3R input/output boundary.

The boundary is intentionally strict: it accepts only official 2-context / 6-
target camera contracts and never receives target RGB, target depth or target
labels.  The current J2 checkpoint fails the input gate before model execution;
this prevents accidental 8-view substitution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .siu3r_protocol import CONTEXT_VIEWS, IMAGE_SIZE, TARGET_VIEWS


STRICT_INPUT_FAILURE = "STRICT_SIU3R_INPUT_VIEW_PARITY: NO"
RETRAINING_FAILURE = "RETRAINING_OR_VARIABLE_VIEW_SUPPORT_REQUIRED: YES"


@dataclass(frozen=True)
class Siu3rModelInput:
    context_images: Any
    context_intrinsics: Any
    target_cam_to_world: Any
    target_intrinsics: Any
    context_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


def validate_model_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a capability report without constructing or running TokenGS."""
    input_views = int(config.get("num_input_views", -1))
    output_views = int(config.get("num_views", -1))
    image_size = tuple(config.get("img_size", ()))
    semantic_classes = int(config.get("semantic_class_count", config.get("semantic_v2_num_classes", -1)))
    return {
        "two_context_forward_supported": input_views == CONTEXT_VIEWS and output_views >= CONTEXT_VIEWS + TARGET_VIEWS,
        "configured_input_views": input_views,
        "configured_total_views": output_views,
        "image_size": list(image_size),
        "semantic_20_class_supported": semantic_classes == 20,
        "strict_input_failure": input_views != CONTEXT_VIEWS,
        "failure_messages": [STRICT_INPUT_FAILURE, RETRAINING_FAILURE] if input_views != CONTEXT_VIEWS else [],
    }


def build_input(
    *,
    context_images: Any,
    context_intrinsics: Any,
    target_cam_to_world: Any,
    target_intrinsics: Any,
    context_ids: list[int] | tuple[int, ...],
    target_ids: list[int] | tuple[int, ...],
) -> Siu3rModelInput:
    """Build the model boundary; no target appearance or target labels accepted."""
    context_ids = tuple(int(value) for value in context_ids)
    target_ids = tuple(int(value) for value in target_ids)
    if len(context_ids) != CONTEXT_VIEWS or len(target_ids) != TARGET_VIEWS:
        raise ValueError("SIU3R adapter requires exactly 2 context and 6 target IDs")
    context_images = np.asarray(context_images)
    context_intrinsics = np.asarray(context_intrinsics)
    target_cam_to_world = np.asarray(target_cam_to_world)
    target_intrinsics = np.asarray(target_intrinsics)
    if context_images.shape[:2] != (CONTEXT_VIEWS, 3) or tuple(context_images.shape[-2:]) != IMAGE_SIZE:
        raise ValueError(f"context_images must be [2,3,256,256], got {context_images.shape}")
    if context_intrinsics.shape != (CONTEXT_VIEWS, 3, 3):
        raise ValueError(f"context_intrinsics must be [2,3,3], got {context_intrinsics.shape}")
    if target_cam_to_world.shape != (TARGET_VIEWS, 4, 4):
        raise ValueError(f"target_cam_to_world must be [6,4,4], got {target_cam_to_world.shape}")
    if target_intrinsics.shape != (TARGET_VIEWS, 3, 3):
        raise ValueError(f"target_intrinsics must be [6,3,3], got {target_intrinsics.shape}")
    if not all(np.isfinite(value).all() for value in (context_images, context_intrinsics, target_cam_to_world, target_intrinsics)):
        raise ValueError("SIU3R adapter input contains non-finite values")
    return Siu3rModelInput(
        context_images=context_images,
        context_intrinsics=context_intrinsics,
        target_cam_to_world=target_cam_to_world,
        target_intrinsics=target_intrinsics,
        context_ids=context_ids,
        target_ids=target_ids,
    )


def assert_runtime_contract(config: Mapping[str, Any]) -> None:
    report = validate_model_contract(config)
    if not report["two_context_forward_supported"]:
        raise RuntimeError(f"{STRICT_INPUT_FAILURE}\n{RETRAINING_FAILURE}")


def render_six_target_cameras(
    model: Any,
    reconstruction: Any,
    model_input: Siu3rModelInput,
    config: Mapping[str, Any],
) -> Any:
    """Render only the six supplied target cameras; no GT is available here."""
    assert_runtime_contract(config)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("TokenGS render requires the project PyTorch environment") from exc
    cam_to_world = torch.as_tensor(model_input.target_cam_to_world)
    target_intrinsics = torch.as_tensor(model_input.target_intrinsics)
    cam_view = torch.linalg.inv(cam_to_world).transpose(-1, -2).unsqueeze(0)
    target_intrinsics = target_intrinsics.unsqueeze(0)
    return model.render_gaussians(reconstruction, cam_view, intrinsics=target_intrinsics)


def validate_class_aware_prediction(
    semantic_logits: Any,
    instance_scores: Any,
    *,
    num_classes: int = 20,
) -> None:
    """Require native class logits/scores; never infer classes from GT."""
    logits = np.asarray(semantic_logits)
    scores = np.asarray(instance_scores)
    if logits.shape[-1] != num_classes:
        raise ValueError(f"SIU3R requires native {num_classes}-class logits, got {logits.shape}")
    if not np.isfinite(logits).all() or not np.isfinite(scores).all():
        raise ValueError("prediction logits/scores contain non-finite values")
