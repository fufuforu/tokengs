# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen CLIP prompt encoding and lightweight prompt-to-GS-token matching."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast


PromptMode = Literal["text_only", "image_only", "text_image_mixed"]
ImagePoolingMode = Literal["masked_patch", "masked_input_cls"]
DEFAULT_CLIP_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "checkpoints" / "clip-vit-base-patch32"
)


class PromptEncoder(nn.Module):
    """Encode text or masked image queries with a frozen local CLIP model."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_CLIP_MODEL_PATH,
        image_pooling: ImagePoolingMode = "masked_patch",
    ):
        super().__init__()
        self.model_path = Path(model_path)
        if image_pooling not in ("masked_patch", "masked_input_cls"):
            raise ValueError(f"Unsupported CLIP image pooling: {image_pooling}")
        self.image_pooling = image_pooling
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Local CLIP checkpoint is unavailable: {self.model_path}")

        self.clip_model = CLIPModel.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        self.tokenizer = CLIPTokenizerFast.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        image_processor = CLIPImageProcessor.from_pretrained(
            str(self.model_path), local_files_only=True
        )

        image_size = self.clip_model.config.vision_config.image_size
        if not isinstance(image_size, int):
            raise ValueError(f"Expected a scalar CLIP image size, got {image_size!r}")
        self.image_size = int(image_size)
        self.output_dim = int(self.clip_model.config.projection_dim)
        self.register_buffer(
            "image_mean",
            torch.tensor(image_processor.image_mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_processor.image_std).view(1, 3, 1, 1),
            persistent=False,
        )

        self.clip_model.requires_grad_(False)
        self.clip_model.eval()

    def train(self, mode: bool = True) -> "PromptEncoder":
        # A parent matcher can train while CLIP must remain deterministic and frozen.
        super().train(False)
        self.clip_model.eval()
        return self

    @staticmethod
    def _flatten_text_queries(
        text_query: str | Sequence[str] | Sequence[Sequence[str]],
    ) -> tuple[list[str], int, int]:
        if isinstance(text_query, str):
            return [text_query], 1, 1
        rows = list(text_query)
        if not rows:
            raise ValueError("text_query must not be empty")
        if all(isinstance(item, str) for item in rows):
            return [str(item) for item in rows], len(rows), 1
        nested = [list(row) for row in rows]
        if not nested or not nested[0]:
            raise ValueError("text_query rows must not be empty")
        query_count = len(nested[0])
        if any(len(row) != query_count for row in nested):
            raise ValueError("Every text-query batch row must have the same query count")
        if any(not isinstance(item, str) for row in nested for item in row):
            raise TypeError("All text queries must be strings")
        return [item for row in nested for item in row], len(nested), query_count

    def encode_text(
        self, text_query: str | Sequence[str] | Sequence[Sequence[str]]
    ) -> torch.Tensor:
        texts, batch_size, query_count = self._flatten_text_queries(text_query)
        device = next(self.clip_model.parameters()).device
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {name: value.to(device) for name, value in tokenized.items()}
        with torch.no_grad():
            embedding = self.clip_model.get_text_features(**tokenized)
        embedding = F.normalize(embedding.float(), dim=-1)
        return embedding.reshape(batch_size, query_count, self.output_dim)

    @staticmethod
    def _flatten_images(query_image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if query_image.ndim == 4:
            query_image = query_image[:, None]
        if query_image.ndim != 5 or query_image.shape[2] != 3:
            raise ValueError(
                "query_image must have shape [B,3,H,W] or [B,Q,3,H,W]"
            )
        batch_size, query_count = query_image.shape[:2]
        return query_image.flatten(0, 1), batch_size, query_count

    @staticmethod
    def _flatten_masks(
        query_mask: torch.Tensor, batch_size: int, query_count: int
    ) -> torch.Tensor:
        if query_mask.ndim == 4:
            query_mask = query_mask[:, None]
        if query_mask.ndim != 5 or query_mask.shape[2] != 1:
            raise ValueError(
                "query_mask must have shape [B,1,H,W] or [B,Q,1,H,W]"
            )
        if query_mask.shape[:2] != (batch_size, query_count):
            raise ValueError("query_mask batch/query dimensions must match query_image")
        return query_mask.flatten(0, 1)

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float()
        if images.numel() and (images.min() < 0 or images.max() > 1):
            raise ValueError("query_image must contain raw RGB values in [0, 1]")
        images = F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0, 1)
        return (images - self.image_mean) / self.image_std

    def encode_image(
        self, query_image: torch.Tensor, query_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        images, batch_size, query_count = self._flatten_images(query_image)
        device = next(self.clip_model.parameters()).device
        images = images.to(device)
        masks = None
        if query_mask is not None:
            masks = self._flatten_masks(query_mask, batch_size, query_count)
            masks = masks.to(device=device, dtype=images.dtype).clamp(0, 1)
            if masks.shape[-2:] != images.shape[-2:]:
                raise ValueError("query_mask spatial dimensions must match query_image")
            if torch.any(masks.flatten(1).sum(dim=1) <= 0):
                raise ValueError("query_mask must contain foreground pixels")
            if self.image_pooling == "masked_input_cls":
                neutral = self.image_mean.to(device=device, dtype=images.dtype)
                images = images * masks + neutral * (1.0 - masks)
        images = self._preprocess_images(images)

        with torch.no_grad():
            vision_output = self.clip_model.vision_model(pixel_values=images)
            if masks is None or self.image_pooling == "masked_input_cls":
                pooled = vision_output.pooler_output
            else:
                patch_tokens = vision_output.last_hidden_state[:, 1:]
                patch_count = patch_tokens.shape[1]
                grid_size = int(round(patch_count**0.5))
                if grid_size * grid_size != patch_count:
                    raise ValueError(f"CLIP patch layout is not square: {patch_count} tokens")
                patch_weights = F.interpolate(
                    masks, size=(grid_size, grid_size), mode="area"
                ).flatten(1)
                weight_sum = patch_weights.sum(dim=1, keepdim=True)
                pooled = torch.sum(
                    patch_tokens * patch_weights.unsqueeze(-1), dim=1
                ) / weight_sum
                pooled = self.clip_model.vision_model.post_layernorm(pooled)
            embedding = self.clip_model.visual_projection(pooled)

        embedding = F.normalize(embedding.float(), dim=-1)
        return embedding.reshape(batch_size, query_count, self.output_dim)

    def forward(
        self,
        mode: PromptMode,
        text_query: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
        query_image: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        mixed_text_weight: float = 0.5,
    ) -> torch.Tensor:
        if mode == "text_only":
            if text_query is None:
                raise ValueError("text_only mode requires text_query")
            return self.encode_text(text_query)
        if mode == "image_only":
            if query_image is None:
                raise ValueError("image_only mode requires query_image")
            return self.encode_image(query_image, query_mask)
        if mode != "text_image_mixed":
            raise ValueError(f"Unsupported prompt mode: {mode}")
        if text_query is None or query_image is None:
            raise ValueError("text_image_mixed mode requires both text and image queries")
        if not 0.0 <= mixed_text_weight <= 1.0:
            raise ValueError("mixed_text_weight must be in [0, 1]")
        text_embedding = self.encode_text(text_query)
        image_embedding = self.encode_image(query_image, query_mask)
        if text_embedding.shape != image_embedding.shape:
            raise ValueError(
                "Text and image prompt shapes must match for mixed fusion: "
                f"{tuple(text_embedding.shape)} vs {tuple(image_embedding.shape)}"
            )
        return F.normalize(
            mixed_text_weight * text_embedding
            + (1.0 - mixed_text_weight) * image_embedding,
            dim=-1,
        )


class PromptGaussianDecoder(nn.Module):
    """One-layer prompt-to-token cross-attention and binary matching head."""

    def __init__(
        self,
        token_dim: int = 1024,
        prompt_dim: int = 512,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.token_dim = int(token_dim)
        self.prompt_dim = int(prompt_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_projection = nn.Linear(token_dim, hidden_dim)
        self.prompt_projection = nn.Linear(prompt_dim, hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.prompt_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_attention_norm = nn.LayerNorm(hidden_dim)
        self.matching_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, gs_token_hidden: torch.Tensor, prompt_embedding: torch.Tensor
    ) -> torch.Tensor:
        if gs_token_hidden.ndim != 3 or gs_token_hidden.shape[-1] != self.token_dim:
            raise ValueError(
                f"gs_token_hidden must have shape [B,T,{self.token_dim}]"
            )
        if prompt_embedding.ndim == 2:
            prompt_embedding = prompt_embedding[:, None]
        if prompt_embedding.ndim != 3 or prompt_embedding.shape[-1] != self.prompt_dim:
            raise ValueError(
                f"prompt_embedding must have shape [B,Q,{self.prompt_dim}]"
            )
        if gs_token_hidden.shape[0] != prompt_embedding.shape[0]:
            raise ValueError("GS tokens and prompts must have the same batch size")

        tokens = self.token_norm(self.token_projection(gs_token_hidden))
        prompts = self.prompt_norm(self.prompt_projection(prompt_embedding))
        prompt_context, _ = self.cross_attention(
            query=prompts,
            key=tokens,
            value=tokens,
            need_weights=False,
        )
        prompts = self.cross_attention_norm(prompts + prompt_context)

        token_pairs = tokens[:, None].expand(-1, prompts.shape[1], -1, -1)
        prompt_pairs = prompts[:, :, None].expand(-1, -1, tokens.shape[1], -1)
        pair_features = torch.cat(
            (
                token_pairs,
                prompt_pairs,
                token_pairs * prompt_pairs,
                torch.abs(token_pairs - prompt_pairs),
            ),
            dim=-1,
        )
        return self.matching_head(pair_features).squeeze(-1)


class ConditionalQueryDecoder(nn.Module):
    """OV-DETR-style prompt addition before a semantic TokenGS decoder block."""

    def __init__(
        self,
        decoder_block: nn.Module,
        token_dim: int = 1024,
        prompt_dim: int = 512,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.prompt_dim = int(prompt_dim)
        self.semantic_last_decoder = decoder_block
        self.prompt_projection = nn.Linear(prompt_dim, token_dim)
        self.matching_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 1),
        )

    def set_trainable(self, tune_last_cross_attention: bool = True) -> None:
        """Train the heads and optionally cross-attention in the copied block."""
        self.requires_grad_(False)
        self.prompt_projection.requires_grad_(True)
        self.matching_head.requires_grad_(True)
        if tune_last_cross_attention:
            self.semantic_last_decoder.gs_cross_attn.requires_grad_(True)
            self.semantic_last_decoder.gs_cross_attn_scale.requires_grad_(True)

    def forward(
        self,
        gs_tokens: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        prompt_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gs_tokens.ndim != 3 or gs_tokens.shape[-1] != self.token_dim:
            raise ValueError(f"gs_tokens must have shape [B,T,{self.token_dim}]")
        if prompt_embedding.ndim == 2:
            prompt_embedding = prompt_embedding[:, None]
        if prompt_embedding.ndim != 3 or prompt_embedding.shape[-1] != self.prompt_dim:
            raise ValueError(
                f"prompt_embedding must have shape [B,Q,{self.prompt_dim}]"
            )
        batch_size, query_count = prompt_embedding.shape[:2]
        if gs_tokens.shape[0] != batch_size:
            raise ValueError("GS tokens and prompts must have the same batch size")

        condition = self.prompt_projection(prompt_embedding.float()).to(gs_tokens.dtype)
        conditioned = gs_tokens[:, None] + condition[:, :, None]
        conditioned = conditioned.flatten(0, 1)
        repeated_keys = keys.repeat_interleave(query_count, dim=0)
        repeated_values = values.repeat_interleave(query_count, dim=0)
        hidden = self.semantic_last_decoder(
            gs_tokens=conditioned,
            keys=repeated_keys,
            values=repeated_values,
        )
        token_logits = self.matching_head(hidden.float()).squeeze(-1)
        token_logits = token_logits.reshape(
            batch_size, query_count, gs_tokens.shape[1]
        )
        return token_logits, hidden.reshape(
            batch_size, query_count, gs_tokens.shape[1], self.token_dim
        )


class ConditionalPromptMatcher(nn.Module):
    """Frozen CLIP plus a shared text/image conditional query decoder."""

    def __init__(
        self,
        decoder_block: nn.Module,
        clip_model_path: str | Path = DEFAULT_CLIP_MODEL_PATH,
        token_dim: int = 1024,
        mixed_text_weight: float = 0.5,
        image_pooling: ImagePoolingMode = "masked_patch",
        tune_last_cross_attention: bool = True,
    ):
        super().__init__()
        self.prompt_encoder = PromptEncoder(
            clip_model_path, image_pooling=image_pooling
        )
        self.matching_decoder = ConditionalQueryDecoder(
            decoder_block=decoder_block,
            token_dim=token_dim,
            prompt_dim=self.prompt_encoder.output_dim,
        )
        if not 0.0 <= mixed_text_weight <= 1.0:
            raise ValueError("mixed_text_weight must be in [0, 1]")
        self.mixed_text_weight = float(mixed_text_weight)
        self.tune_last_cross_attention = bool(tune_last_cross_attention)
        self.prompt_encoder.requires_grad_(False)
        self.matching_decoder.set_trainable(self.tune_last_cross_attention)

    def train(self, mode: bool = True) -> "ConditionalPromptMatcher":
        super().train(mode)
        self.prompt_encoder.eval()
        self.prompt_encoder.requires_grad_(False)
        self.matching_decoder.semantic_last_decoder.train(
            mode and self.tune_last_cross_attention
        )
        return self

    def trainable_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        trainable_names = {
            name
            for name, parameter in self.matching_decoder.named_parameters()
            if parameter.requires_grad
        }
        return OrderedDict(
            (f"matching_decoder.{name}", value)
            for name, value in self.matching_decoder.state_dict().items()
            if name in trainable_names
        )

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
                "Conditional prompt checkpoint mismatch: "
                f"missing={missing}, unexpected={unexpected}, "
                f"shape_mismatched={mismatched}"
            )
        with torch.no_grad():
            for key in set(current) & set(state_dict):
                if current[key].shape == state_dict[key].shape:
                    current[key].copy_(state_dict[key])
        return nn.modules.module._IncompatibleKeys(
            missing, unexpected + mismatched
        )


class PromptConditionedTokenMatcher(nn.Module):
    """Compose frozen prompt encoding with the trainable token matcher."""

    def __init__(
        self,
        clip_model_path: str | Path = DEFAULT_CLIP_MODEL_PATH,
        token_dim: int = 1024,
        hidden_dim: int = 128,
        num_heads: int = 4,
        mixed_text_weight: float = 0.5,
        image_pooling: ImagePoolingMode = "masked_patch",
    ):
        super().__init__()
        self.prompt_encoder = PromptEncoder(
            clip_model_path, image_pooling=image_pooling
        )
        self.matching_decoder = PromptGaussianDecoder(
            token_dim=token_dim,
            prompt_dim=self.prompt_encoder.output_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        if not 0.0 <= mixed_text_weight <= 1.0:
            raise ValueError("mixed_text_weight must be in [0, 1]")
        self.mixed_text_weight = float(mixed_text_weight)

    def train(self, mode: bool = True) -> "PromptConditionedTokenMatcher":
        super().train(mode)
        self.prompt_encoder.eval()
        self.prompt_encoder.requires_grad_(False)
        return self

    def freeze_backbones(self, tokengs: nn.Module) -> None:
        tokengs.requires_grad_(False)
        tokengs.eval()
        self.prompt_encoder.requires_grad_(False)
        self.prompt_encoder.eval()
        self.matching_decoder.requires_grad_(True)

    def trainable_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Return only matching weights; frozen CLIP weights are loaded locally."""
        return OrderedDict(
            (f"matching_decoder.{name}", value)
            for name, value in self.matching_decoder.state_dict().items()
        )

    def state_dict(self, *args, **kwargs) -> OrderedDict[str, torch.Tensor]:
        """Keep local frozen CLIP weights out of standard training checkpoints."""
        state = super().state_dict(*args, **kwargs)
        for key in list(state):
            if "prompt_encoder.clip_model." in key:
                del state[key]
        return state

    def load_trainable_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> nn.modules.module._IncompatibleKeys:
        prefix = "matching_decoder."
        matching_state = {
            name[len(prefix) :] if name.startswith(prefix) else name: value
            for name, value in state_dict.items()
        }
        return self.matching_decoder.load_state_dict(matching_state, strict=strict)

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> nn.modules.module._IncompatibleKeys:
        del assign
        return self.load_trainable_state_dict(state_dict, strict=strict)

    def forward(
        self,
        gs_token_hidden: torch.Tensor,
        mode: PromptMode,
        text_query: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
        query_image: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        mixed_text_weight: float | None = None,
    ) -> dict[str, torch.Tensor]:
        prompt_embedding = self.prompt_encoder(
            mode=mode,
            text_query=text_query,
            query_image=query_image,
            query_mask=query_mask,
            mixed_text_weight=(
                self.mixed_text_weight
                if mixed_text_weight is None
                else float(mixed_text_weight)
            ),
        )
        token_logits = self.matching_decoder(
            gs_token_hidden.detach(), prompt_embedding.detach()
        )
        return {
            "prompt_embedding": prompt_embedding,
            "token_logits": token_logits,
        }
