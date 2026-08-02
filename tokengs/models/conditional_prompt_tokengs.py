"""OV-DETR-style conditional binary matching over TokenGS Gaussian queries."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from tokengs.models.input_types import split_data
from tokengs.models.prompt_matching import ConditionalPromptMatcher
from tokengs.models.prompt_tokengs import PromptTokenGS
from tokengs.models.prompt_training import (
    compute_prompt_mask_loss,
    compute_prompt_mask_metrics,
    token_scores_to_gaussians,
)
from tokengs.models.tokengs import TokenGS
from tokengs.utils.metrics import MetricsCalculator


class ConditionalPromptTokenGS(PromptTokenGS):
    """Condition a copied final TokenGS decoder block on text or image prompts."""

    def __init__(self, opt):
        if not getattr(opt, "prompt_training", False):
            raise ValueError("ConditionalPromptTokenGS requires prompt_training=True")
        if int(opt.num_gs_tokens) != 1024:
            raise ValueError("ConditionalPromptTokenGS requires num_gs_tokens=1024")
        TokenGS.__init__(self, opt)
        self.num_gaussians_per_token = int(self.opt.dec_patch_size) ** 2
        if self.num_gaussians_per_token != 64:
            raise ValueError("ConditionalPromptTokenGS requires 64 Gaussians per token")

        self._load_pretrained_tokengs(self.opt.prompt_tokengs_checkpoint)
        semantic_decoder = deepcopy(self.enc_dec_backbone.decoder_blocks[-1])
        self.prompt_matcher = ConditionalPromptMatcher(
            decoder_block=semantic_decoder,
            clip_model_path=self.opt.prompt_clip_model_path,
            token_dim=self.opt.token_dim,
            mixed_text_weight=self.opt.prompt_mixed_text_weight,
            image_pooling=self.opt.prompt_image_pooling,
            tune_last_cross_attention=(
                self.opt.conditional_v3_tune_last_cross_attention
            ),
        )
        self._quality_metrics = MetricsCalculator(device="cpu")
        self._freeze_for_prompt_training()
        groups = self.conditional_trainable_groups()
        print(
            "[ConditionalPromptTokenGS] trainable parameter groups: "
            + ", ".join(
                f"{name}={sum(parameter.numel() for parameter in parameters):,}"
                for name, parameters in groups.items()
            )
        )
        print(
            "[ConditionalPromptTokenGS] reconstruction decoder remains frozen: "
            f"{not any(parameter.requires_grad for parameter in self.enc_dec_backbone.parameters())}"
        )

    def _freeze_for_prompt_training(self) -> None:
        self.requires_grad_(False)
        self.prompt_matcher.prompt_encoder.requires_grad_(False)
        self.prompt_matcher.matching_decoder.set_trainable(
            self.opt.conditional_v3_tune_last_cross_attention
        )
        self.train(True)

    def conditional_trainable_groups(self) -> dict[str, list[nn.Parameter]]:
        decoder = self.prompt_matcher.matching_decoder
        return {
            "condition_projection": list(decoder.prompt_projection.parameters()),
            "matching_head": list(decoder.matching_head.parameters()),
            "last_cross_attention": [
                parameter
                for parameter in decoder.semantic_last_decoder.parameters()
                if parameter.requires_grad
            ],
        }

    def _frozen_reconstruction_context(self, model_input):
        with torch.no_grad():
            encoder_latent = TokenGS.forward_encoder(self, model_input.encoder)
            hidden = TokenGS.get_gs_tokens(self, encoder_latent.keys.shape[0])
            hidden = TokenGS._apply_time_embedding_to_gs_tokens(
                self, hidden, model_input.decoder
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
            rgb_results = TokenGS.render_reconstruction(
                self, reconstruction, model_input.decoder
            )
        return reconstruction, hidden, encoder_latent, rgb_results

    def forward(
        self,
        data: dict,
        skip_loss: bool = False,
        compute_quality_metrics: bool = False,
        **_kwargs,
    ) -> dict:
        del skip_loss
        model_input, supervision = split_data(data, self.opt)
        (
            reconstruction,
            decoder_input_tokens,
            encoder_latent,
            rgb_results,
        ) = self._frozen_reconstruction_context(model_input)
        if decoder_input_tokens.shape[1:] != (1024, 1024):
            raise RuntimeError(
                "Expected decoder input tokens [B,1024,1024], got "
                f"{tuple(decoder_input_tokens.shape)}"
            )
        if reconstruction.gaussians.shape[1] != 65_536:
            raise RuntimeError("ConditionalPromptTokenGS requires 65,536 Gaussians")

        prompt_embedding = self._encode_prompt_batch(data).detach()
        token_logits, conditioned_hidden = self.prompt_matcher.matching_decoder(
            decoder_input_tokens.detach(),
            encoder_latent.keys.detach(),
            encoder_latent.values.detach(),
            prompt_embedding,
        )
        if token_logits.shape[-1] != 1024:
            raise RuntimeError(f"Expected token_logits [B,Q,1024], got {token_logits.shape}")
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
            "gs_token_hidden": conditioned_hidden,
            "prompt_embedding": prompt_embedding,
            "token_logits": token_logits,
            "gaussian_scores": gaussian_scores,
            "rendered_prompt_probability": rendered_probability,
            "rendered_alpha": rendered_alpha,
            "valid_mask": valid_mask,
            "target_prompt_mask": target_prompt_mask.expand_as(
                rendered_probability
            ),
        }
