"""Shared cosine semantic space for frozen TokenGS tokens and CLIP text."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizerFast

from tokengs.models.prompt_matching import DEFAULT_CLIP_MODEL_PATH


C3G8_CLASS_NAMES = (
    "wall",
    "floor",
    "ceiling",
    "chair",
    "table",
    "sofa",
    "bed",
    "other",
)


def semantic_token_probabilities(
    token_logits: torch.Tensor, score_mode: str
) -> torch.Tensor:
    """Normalize per-token class scores before Gaussian expansion."""
    if score_mode == "sigmoid":
        return token_logits.float().sigmoid()
    if score_mode == "softmax":
        return token_logits.float().softmax(dim=1)
    raise ValueError(f"Unsupported semantic score mode: {score_mode}")
class FrozenCLIPTextEncoder(nn.Module):
    """Local, frozen CLIP text encoder used only to build eight prototypes."""

    def __init__(self, model_path: str | Path = DEFAULT_CLIP_MODEL_PATH):
        super().__init__()
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Local CLIP checkpoint is unavailable: {path}")
        self.clip_model = CLIPModel.from_pretrained(
            str(path), local_files_only=True
        )
        self.tokenizer = CLIPTokenizerFast.from_pretrained(
            str(path), local_files_only=True
        )
        self.output_dim = int(self.clip_model.config.projection_dim)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenCLIPTextEncoder":
        del mode
        super().train(False)
        self.clip_model.eval()
        return self

    def encode(self, prompts: Sequence[str]) -> torch.Tensor:
        device = next(self.clip_model.parameters()).device
        tokenized = self.tokenizer(
            list(prompts), padding=True, truncation=True, return_tensors="pt"
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        with torch.no_grad():
            embeddings = self.clip_model.get_text_features(**tokenized)
        return F.normalize(embeddings.float(), dim=-1)


class SemanticTokenAdapter(nn.Module):
    """Residual two-layer MLP from TokenGS hidden features to semantic space."""

    def __init__(self, input_dim: int = 1024, semantic_dim: int = 256):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, semantic_dim)
        self.residual_mlp = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, semantic_dim),
        )
        self.output_norm = nn.LayerNorm(semantic_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(self.input_norm(hidden))
        adapted = self.output_norm(projected + self.residual_mlp(projected))
        return F.normalize(adapted.float(), dim=-1)


class PromptSemanticAdapter(nn.Module):
    """Light CLIP-text projection into the shared semantic space."""

    def __init__(self, input_dim: int = 512, semantic_dim: int = 256):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, semantic_dim),
        )
        self.output_norm = nn.LayerNorm(semantic_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        adapted = self.output_norm(self.projection(self.input_norm(embeddings)))
        return F.normalize(adapted.float(), dim=-1)


class SemanticMatcherV2(nn.Module):
    """Eight-way cosine matcher without token self-attention or image queries."""

    def __init__(
        self,
        clip_model_path: str | Path = DEFAULT_CLIP_MODEL_PATH,
        token_dim: int = 1024,
        semantic_dim: int = 256,
        temperature_init: float = 14.285714,
        class_names: Sequence[str] = C3G8_CLASS_NAMES,
    ):
        super().__init__()
        if tuple(class_names) != C3G8_CLASS_NAMES:
            raise ValueError("Semantic Adapter V2 requires the fixed C3G8 class order")
        if temperature_init <= 0:
            raise ValueError("temperature_init must be positive")

        self.class_names = tuple(class_names)
        self.text_encoder = FrozenCLIPTextEncoder(clip_model_path)
        with torch.no_grad():
            clip_prototypes = self.text_encoder.encode(self.class_names)
        self.register_buffer(
            "clip_text_prototypes", clip_prototypes, persistent=False
        )
        self.semantic_token_adapter = SemanticTokenAdapter(token_dim, semantic_dim)
        self.prompt_semantic_adapter = PromptSemanticAdapter(
            self.text_encoder.output_dim, semantic_dim
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(float(temperature_init)).log()
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(max=100.0)

    def train(self, mode: bool = True) -> "SemanticMatcherV2":
        super().train(mode)
        self.text_encoder.eval()
        return self

    def forward(self, token_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        if token_hidden.ndim != 3:
            raise ValueError("token_hidden must have shape [B,T,1024]")
        semantic_tokens = self.semantic_token_adapter(token_hidden)
        semantic_prompts = self.prompt_semantic_adapter(
            self.clip_text_prototypes
        ).unsqueeze(0).expand(token_hidden.shape[0], -1, -1)
        token_prompt_logits = torch.einsum(
            "bqd,btd->bqt", semantic_prompts, semantic_tokens
        ) * self.temperature
        return {
            "semantic_tokens": semantic_tokens,
            "semantic_prompts": semantic_prompts,
            "token_logits": token_prompt_logits,
            "temperature": self.temperature,
        }

    def trainable_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state = OrderedDict()
        for prefix, module in (
            ("semantic_token_adapter", self.semantic_token_adapter),
            ("prompt_semantic_adapter", self.prompt_semantic_adapter),
        ):
            for key, value in module.state_dict().items():
                state[f"{prefix}.{key}"] = value
        state["log_temperature"] = self.log_temperature
        return state

    def load_trainable_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> nn.modules.module._IncompatibleKeys:
        current = self.trainable_state_dict()
        missing = sorted(set(current) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(current))
        mismatched = sorted(
            key
            for key in set(current) & set(state_dict)
            if current[key].shape != state_dict[key].shape
        )
        if strict and (missing or unexpected or mismatched):
            raise RuntimeError(
                "Semantic Adapter V2 checkpoint mismatch: "
                f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
            )
        with torch.no_grad():
            for key in set(current) & set(state_dict):
                if current[key].shape == state_dict[key].shape:
                    current[key].copy_(state_dict[key])
        return nn.modules.module._IncompatibleKeys(missing, unexpected)


def compute_semantic_v2_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """FP32 per-class sufficient statistics for macro IoU and mAcc."""
    if probability.ndim != 6 or probability.shape[1] != len(C3G8_CLASS_NAMES):
        raise ValueError("probability must have shape [B,8,V,1,H,W]")
    with torch.autocast(device_type=probability.device.type, enabled=False):
        probability = probability.float()
        target_bool = target.float() >= 0.5
        valid_bool = valid.bool()
        predicted = probability >= float(threshold)
        reduce_dims = (0, 2, 3, 4, 5)
        intersection = (predicted & target_bool & valid_bool).sum(reduce_dims).float()
        union = ((predicted | target_bool) & valid_bool).sum(reduce_dims).float()
        target_count = (target_bool & valid_bool).sum(reduce_dims).float()
        predicted_count = (predicted & valid_bool).sum(reduce_dims).float()
        valid_count = valid_bool.sum(reduce_dims).float()
        foreground_probability_sum = (
            probability * target_bool.float() * valid_bool.float()
        ).sum(reduce_dims)
        background_count = ((~target_bool) & valid_bool).sum(reduce_dims).float()
        background_rejection_sum = (
            (1.0 - probability) * (~target_bool).float() * valid_bool.float()
        ).sum(reduce_dims)
        per_class_iou = intersection / union.clamp_min(eps)
        per_class_recall = intersection / target_count.clamp_min(eps)
        per_class_predicted_ratio = predicted_count / valid_count.clamp_min(eps)
        per_class_gt_ratio = target_count / valid_count.clamp_min(eps)
        per_class_foreground_probability = (
            foreground_probability_sum / target_count.clamp_min(eps)
        )
        per_class_background_probability = (
            background_rejection_sum / background_count.clamp_min(eps)
        )
        pixel_valid = valid_bool[:, 0]
        pixel_probability_sum = probability.sum(dim=1)
        top2 = probability.topk(k=2, dim=1).values
        multiclass_margin = top2[:, 0] - top2[:, 1]
        simultaneous_high = (probability >= float(threshold)).sum(dim=1) > 1
        gt_class = target_bool.float().argmax(dim=1)
        predicted_class = probability.argmax(dim=1)
        valid_gt = pixel_valid & (target_bool.sum(dim=1) == 1)
        confusion_indices = (
            gt_class[valid_gt].long() * len(C3G8_CLASS_NAMES)
            + predicted_class[valid_gt].long()
        )
        confusion = torch.bincount(
            confusion_indices,
            minlength=len(C3G8_CLASS_NAMES) ** 2,
        ).reshape(len(C3G8_CLASS_NAMES), len(C3G8_CLASS_NAMES)).float()
        pixel_count = pixel_valid.sum().float()
        argmax_correct = (predicted_class == gt_class) & valid_gt
    return {
        "mask_iou": per_class_iou.mean(),
        "macro_miou": per_class_iou.mean(),
        "macc": per_class_recall.mean(),
        "foreground_probability": per_class_foreground_probability.mean(),
        "background_probability": per_class_background_probability.mean(),
        "predicted_foreground_ratio": per_class_predicted_ratio.mean(),
        "gt_foreground_ratio": per_class_gt_ratio.mean(),
        "per_class_iou": per_class_iou,
        "per_class_recall": per_class_recall,
        "per_class_predicted_ratio": per_class_predicted_ratio,
        "per_class_gt_ratio": per_class_gt_ratio,
        "class_intersection": intersection,
        "class_union": union,
        "class_target_count": target_count,
        "class_predicted_count": predicted_count,
        "class_valid_count": valid_count,
        "class_foreground_probability_sum": foreground_probability_sum,
        "class_background_rejection_sum": background_rejection_sum,
        "class_background_count": background_count,
        "probability_sum": (
            pixel_probability_sum * pixel_valid.float()
        ).sum() / pixel_count.clamp_min(eps),
        "top1_top2_margin": (
            multiclass_margin * pixel_valid.float()
        ).sum() / pixel_count.clamp_min(eps),
        "simultaneous_high_ratio": (
            simultaneous_high & pixel_valid
        ).sum().float() / pixel_count.clamp_min(eps),
        "argmax_accuracy": argmax_correct.sum().float()
        / valid_gt.sum().float().clamp_min(eps),
        "argmax_confusion": confusion,
    }
