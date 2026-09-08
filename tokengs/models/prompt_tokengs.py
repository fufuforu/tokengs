"""Frozen TokenGS reconstruction with trainable prompt-to-token mask matching."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from tokengs.models.input_types import split_data
from tokengs.models.prompt_matching import PromptConditionedTokenMatcher
from tokengs.models.prompt_training import (
    compute_prompt_mask_loss,
    compute_prompt_mask_metrics,
    token_scores_to_gaussians,
)
from tokengs.models.tokengs import TokenGS
from tokengs.utils.metrics import MetricsCalculator


class PromptTokenGS(TokenGS):
    """Frozen reconstruction plus optional semantic last-layer adaptation."""

    def __init__(self, opt):
        if not getattr(opt, "prompt_training", False):
            raise ValueError("PromptTokenGS requires prompt_training=True")
        if int(opt.num_gs_tokens) <= 0:
            raise ValueError("num_gs_tokens must be positive")
        super().__init__(opt)

        self.num_gaussians_per_token = int(self.opt.dec_patch_size) ** 2
        if self.num_gaussians_per_token != 64:
            raise ValueError(
                "PromptTokenGS requires dec_patch_size=8 (64 Gaussians per token)"
            )
        self._load_pretrained_tokengs(self.opt.prompt_tokengs_checkpoint)
        self.prompt_matcher = PromptConditionedTokenMatcher(
            clip_model_path=self.opt.prompt_clip_model_path,
            token_dim=self.opt.token_dim,
            hidden_dim=self.opt.prompt_hidden_dim,
            num_heads=self.opt.prompt_attention_heads,
            mixed_text_weight=self.opt.prompt_mixed_text_weight,
            image_pooling=self.opt.prompt_image_pooling,
            text_adapter=bool(self.opt.prompt_text_adapter),
        )
        self.semantic_last_decoder = (
            deepcopy(self.enc_dec_backbone.decoder_blocks[-1])
            if self.opt.prompt_tune_last_cross_attention
            else None
        )
        self._quality_metrics = MetricsCalculator(device="cpu")
        self._freeze_for_prompt_training()
        groups = self.prompt_trainable_groups()
        print(
            "[PromptTokenGS] trainable parameter groups: "
            + ", ".join(
                f"{name}={sum(parameter.numel() for parameter in parameters):,}"
                for name, parameters in groups.items()
            )
        )
        print(
            "[PromptTokenGS] reconstruction TokenGS remains frozen: "
            f"{not any(parameter.requires_grad for parameter in self.enc_dec_backbone.parameters())}"
        )

    def _load_pretrained_tokengs(self, checkpoint_path: str) -> None:
        if not checkpoint_path:
            print(
                "[PromptTokenGS] training TokenGS from scratch: no "
                "pretrained checkpoint (random encoder/decoder/GS-token "
                "initialization)"
            )
            return
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained TokenGS checkpoint is unavailable: {path}")
        checkpoint = load_file(str(path), device="cpu")
        current = TokenGS.state_dict(self)
        missing = sorted(set(current) - set(checkpoint))
        unexpected = sorted(set(checkpoint) - set(current))
        mismatched = sorted(
            (key, tuple(checkpoint[key].shape), tuple(current[key].shape))
            for key in set(current) & set(checkpoint)
            if checkpoint[key].shape != current[key].shape
        )
        token_mismatch = (
            "gs_tokens" in checkpoint
            and "gs_tokens" in current
            and checkpoint["gs_tokens"].shape != current["gs_tokens"].shape
        )
        mismatched = [
            item for item in mismatched if item[0] != "gs_tokens"
        ]
        checkpoint_token_shape = tuple(checkpoint.get("gs_tokens", torch.empty(0)).shape)
        current_token_shape = tuple(current["gs_tokens"].shape)
        print(f"[PromptTokenGS] pretrained checkpoint: {path}")
        print(f"[PromptTokenGS] checkpoint GS token shape: {checkpoint_token_shape}")
        print(f"[PromptTokenGS] configured GS token shape: {current_token_shape}")
        print(f"[PromptTokenGS] missing keys: {missing}")
        print(f"[PromptTokenGS] unexpected keys: {unexpected}")
        print(f"[PromptTokenGS] shape-mismatched keys: {mismatched}")
        if checkpoint_token_shape[1:] != (1024,):
            raise RuntimeError(
                f"Expected checkpoint gs_tokens [*,1024], got {checkpoint_token_shape}"
            )
        if token_mismatch:
            old_count = checkpoint_token_shape[0]
            new_count = current_token_shape[0]
            resized = current["gs_tokens"].clone()
            if new_count >= old_count:
                resized[:old_count].copy_(checkpoint["gs_tokens"])
                extra = new_count - old_count
                repeat_factor = (extra + old_count - 1) // old_count
                expanded = checkpoint["gs_tokens"].repeat(repeat_factor, 1)[:extra]
                resized[old_count:] = expanded + 0.01 * torch.randn_like(
                    expanded
                )
            else:
                indices = torch.linspace(0, old_count - 1, new_count).long()
                resized.copy_(checkpoint["gs_tokens"][indices])
            checkpoint["gs_tokens"] = resized
            print(
                f"[PromptTokenGS] resized gs_tokens: {old_count} -> {new_count}"
            )
        if missing or unexpected or mismatched:
            raise RuntimeError("Pretrained TokenGS checkpoint failed strict validation")
        with torch.no_grad():
            for key, value in checkpoint.items():
                if key in current and current[key].shape == value.shape:
                    current[key].copy_(value)

    def _freeze_for_prompt_training(self) -> None:
        self.requires_grad_(False)
        self.prompt_matcher.prompt_encoder.requires_grad_(False)
        self.prompt_matcher.matching_decoder.requires_grad_(True)
        if self.prompt_matcher.prompt_adapter is not None:
            self.prompt_matcher.prompt_adapter.requires_grad_(True)
        if self.semantic_last_decoder is not None:
            self.semantic_last_decoder.gs_cross_attn.requires_grad_(True)
            self.semantic_last_decoder.gs_cross_attn_scale.requires_grad_(True)
        self.train(True)

    def train(self, mode: bool = True) -> "PromptTokenGS":
        nn.Module.train(self, mode)
        for name, module in self.named_children():
            if name != "prompt_matcher":
                module.eval()
        if hasattr(self, "prompt_matcher"):
            self.prompt_matcher.train(mode)
        if self.semantic_last_decoder is not None:
            # The copied block has no train-time state; keeping it in eval mode
            # prevents stochastic changes outside the selected cross-attention.
            self.semantic_last_decoder.eval()
        return self

    def prompt_trainable_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = {
            "matching_decoder": list(
                self.prompt_matcher.matching_decoder.parameters()
            )
        }
        if self.prompt_matcher.prompt_adapter is not None:
            groups["prompt_adapter"] = list(
                self.prompt_matcher.prompt_adapter.parameters()
            )
        if self.semantic_last_decoder is not None:
            groups["last_cross_attention"] = [
                parameter
                for parameter in self.semantic_last_decoder.parameters()
                if parameter.requires_grad
            ]
        return groups

    def state_dict(self, *args, **kwargs) -> OrderedDict[str, torch.Tensor]:
        """Prompt checkpoints contain no frozen TokenGS or CLIP parameters."""
        state = self.prompt_matcher.trainable_state_dict()
        if self.semantic_last_decoder is not None:
            for key, value in self.semantic_last_decoder.state_dict().items():
                if key.startswith("gs_cross_attn.") or key.startswith(
                    "gs_cross_attn_scale."
                ):
                    state[f"semantic_last_decoder.{key}"] = value
        return state

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> nn.modules.module._IncompatibleKeys:
        # DINOv2 is a frozen external extractor loaded lazily; if an older
        # checkpoint saved its weights (instance_branch._dino_model.*), they
        # are not part of the trainable model and are ignored.  Delegate to
        # the base implementation (which copies into the real parameters,
        # unlike the previous manual copy from a state_dict() snapshot that
        # silently no-op'd).
        filtered = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("instance_branch._dino_model.")
        }
        return super().load_state_dict(filtered, strict=strict, assign=assign)

    def _forward_prompt_reconstruction(self, model_input):
        """Keep geometry frozen and optionally adapt a copied semantic last layer."""
        if self.semantic_last_decoder is None:
            with torch.no_grad():
                reconstruction, hidden = super().forward_reconstruction(
                    model_input, return_gs_token_hidden=True
                )
                rgb_results = super().render_reconstruction(
                    reconstruction, model_input.decoder
                )
            return reconstruction, hidden, rgb_results

        with torch.no_grad():
            encoder_latent = super().forward_encoder(model_input.encoder)
            hidden = super().get_gs_tokens(
                encoder_latent.keys.shape[0],
                encoder_latent=encoder_latent,
                decoder_input=model_input.decoder,
            )
            hidden = super()._apply_time_embedding_to_gs_tokens(
                hidden, model_input.decoder
            )
            for layer in self.enc_dec_backbone.decoder_blocks[:-1]:
                hidden = layer(
                    gs_tokens=hidden,
                    keys=encoder_latent.keys,
                    values=encoder_latent.values,
                )
            geometry_hidden = self.enc_dec_backbone.decoder_blocks[-1](
                gs_tokens=hidden,
                keys=encoder_latent.keys,
                values=encoder_latent.values,
            )
            gaussians = self.activation_head(geometry_hidden)
            gaussians[..., 2] = gaussians[..., 2] + self.opt.gaussian_z_offset
            reconstruction = self._reconstruction_from_gaussians(gaussians)
            rgb_results = super().render_reconstruction(
                reconstruction, model_input.decoder
            )

        semantic_hidden = self.semantic_last_decoder(
            gs_tokens=hidden.detach(),
            keys=encoder_latent.keys.detach(),
            values=encoder_latent.values.detach(),
        )
        return reconstruction, semantic_hidden, rgb_results

    def _matching_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Detach the baseline hidden but retain gradients from the semantic fork."""
        return hidden if self.semantic_last_decoder is not None else hidden.detach()

    def _encode_prompt_batch(self, data: dict) -> torch.Tensor:
        mode = self.opt.prompt_mode
        text_query = data.get("positive_text_prompt")
        query_image = data.get("query_image")
        query_mask = data.get("query_mask")
        encoder = self.prompt_matcher.prompt_encoder
        if mode == "manifest":
            modes = list(data["prompt_mode"])
            allowed = {"text_only", "image_only", "text_image_mixed"}
            if any(value not in allowed for value in modes):
                raise ValueError(f"Unsupported manifest prompt modes: {modes}")
            batch_size = len(modes)
            output = torch.empty(
                (batch_size, 1, encoder.output_dim),
                device=query_image.device,
                dtype=torch.float32,
            )
            text_indices = torch.tensor(
                [index for index, value in enumerate(modes) if value != "image_only"],
                device=query_image.device,
                dtype=torch.long,
            )
            image_indices = torch.tensor(
                [index for index, value in enumerate(modes) if value != "text_only"],
                device=query_image.device,
                dtype=torch.long,
            )
            text_embeddings = None
            if text_indices.numel():
                selected_text = [text_query[index] for index in text_indices.tolist()]
                text_embeddings = encoder.encode_text(selected_text)
                output[text_indices] = text_embeddings
            image_embeddings = None
            if image_indices.numel():
                if not bool(data["has_image_query"][image_indices].all()):
                    raise ValueError("Manifest image/mixed prompt is missing its image query")
                image_embeddings = encoder.encode_image(
                    query_image[image_indices], query_mask[image_indices]
                )
                output[image_indices] = image_embeddings
            mixed_indices = [
                index for index, value in enumerate(modes)
                if value == "text_image_mixed"
            ]
            if mixed_indices:
                weight = float(self.opt.prompt_mixed_text_weight)
                for batch_index in mixed_indices:
                    text_row = (text_indices == batch_index).nonzero(as_tuple=False).item()
                    image_row = (image_indices == batch_index).nonzero(as_tuple=False).item()
                    output[batch_index] = F.normalize(
                        weight * text_embeddings[text_row]
                        + (1.0 - weight) * image_embeddings[image_row],
                        dim=-1,
                    )
            return output
        if mode == "text_only":
            return encoder.encode_text(text_query)
        if mode == "image_only":
            if query_image is None or not bool(data["has_image_query"].all()):
                raise ValueError("image_only prompt batch contains a missing image query")
            return encoder.encode_image(query_image, query_mask)
        if mode != "text_image_mixed":
            raise ValueError(f"Unsupported prompt mode: {mode}")

        text_embedding = encoder.encode_text(text_query)
        has_image = data["has_image_query"].bool()
        if not has_image.any():
            return text_embedding
        image_indices = torch.nonzero(has_image, as_tuple=False).flatten()
        image_embedding = encoder.encode_image(
            query_image[image_indices], query_mask[image_indices]
        )
        output = text_embedding.clone()
        weight = float(self.opt.prompt_mixed_text_weight)
        output[image_indices] = F.normalize(
            weight * text_embedding[image_indices] + (1.0 - weight) * image_embedding,
            dim=-1,
        )
        return output

    @staticmethod
    def _target_and_label_valid(data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        if "binary_mask_output" not in data:
            raise KeyError("Prompt training requires binary_mask_output")
        target = data["binary_mask_output"].float()
        semantic = data.get("semantic_label_output")
        label_valid = torch.ones_like(target, dtype=torch.bool)
        if semantic is not None:
            label_valid = semantic != 0
        return target, label_valid

    def forward(
        self,
        data: dict,
        skip_loss: bool = False,
        compute_quality_metrics: bool = False,
        **_kwargs,
    ) -> dict:
        del skip_loss
        model_input, supervision = split_data(data, self.opt)
        reconstruction, gs_token_hidden, rgb_results = (
            self._forward_prompt_reconstruction(model_input)
        )

        if gs_token_hidden.ndim != 3 or gs_token_hidden.shape[-1] != 1024:
            raise RuntimeError(
                f"Expected gs_token_hidden [B,T,1024], got {tuple(gs_token_hidden.shape)}"
            )
        expected_gaussians = (
            int(self.opt.num_gs_tokens) * int(self.num_gaussians_per_token)
        )
        if reconstruction.gaussians.shape[1] != expected_gaussians:
            raise RuntimeError(
                f"Expected {expected_gaussians} Gaussians, got "
                f"{reconstruction.gaussians.shape[1]}"
            )

        prompt_embedding = self.prompt_matcher.apply_prompt_adapter(
            self._encode_prompt_batch(data).detach()
        )
        token_logits = self.prompt_matcher.matching_decoder(
            self._matching_hidden(gs_token_hidden), prompt_embedding
        )
        if token_logits.shape[-1] != int(self.opt.num_gs_tokens):
            raise RuntimeError(
                f"Expected token_logits [B,Q,{self.opt.num_gs_tokens}], "
                f"got {token_logits.shape}"
            )
        gaussian_scores = token_scores_to_gaussians(
            token_logits, self.num_gaussians_per_token
        )
        prompt_render = self.gs.render_prompt_scores(
            reconstruction.gaussians,
            gaussian_scores,
            model_input.decoder.cam_view,
            intrinsics=model_input.decoder.intrinsics,
        )
        rendered_probability = prompt_render["rendered_prompt_probability"]
        rendered_alpha = prompt_render["rendered_alpha"]
        target_mask, label_valid = self._target_and_label_valid(data)
        alpha_valid = rendered_alpha >= float(self.opt.prompt_valid_alpha_threshold)
        if label_valid.ndim == 4:
            label_valid = label_valid[:, None, :, None]
        elif label_valid.ndim == 5:
            label_valid = label_valid[:, None]
        valid_mask = alpha_valid & label_valid.expand_as(alpha_valid)

        losses = compute_prompt_mask_loss(
            rendered_probability,
            target_mask,
            valid_mask,
            lambda_bce=self.opt.prompt_lambda_bce,
            lambda_dice=self.opt.prompt_lambda_dice,
            balance_classes=bool(self.opt.prompt_balanced_bce),
            pos_weight=self.opt.prompt_balanced_bce_pos_weight,
        )
        mask_metrics = compute_prompt_mask_metrics(
            rendered_probability,
            target_mask,
            valid_mask,
            threshold=self.opt.prompt_threshold,
        )
        pred_rgb = rgb_results["images_pred"]
        gt_rgb = supervision.images_output
        with torch.no_grad():
            mse = (pred_rgb.clamp(0, 1) - gt_rgb.clamp(0, 1)).square().mean()
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
            if compute_quality_metrics:
                self._quality_metrics.device = str(pred_rgb.device)
                ssim = self._quality_metrics.calculate_ssim(pred_rgb, gt_rgb)
                lpips = self._quality_metrics.calculate_lpips(pred_rgb, gt_rgb)
            else:
                ssim = torch.full((), float("nan"), device=pred_rgb.device)
                lpips = torch.full((), float("nan"), device=pred_rgb.device)

        if target_mask.ndim == 4:
            target_prompt_mask = target_mask[:, None, :, None]
        else:
            target_prompt_mask = target_mask[:, None]
        return {
            **losses,
            **mask_metrics,
            **rgb_results,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "gaussians": reconstruction.gaussians,
            "gs_token_hidden": gs_token_hidden,
            "prompt_embedding": prompt_embedding,
            "token_logits": token_logits,
            "gaussian_scores": gaussian_scores,
            "rendered_prompt_probability": rendered_probability,
            "rendered_alpha": rendered_alpha,
            "valid_mask": valid_mask,
            "target_prompt_mask": target_prompt_mask.expand_as(rendered_probability),
        }
