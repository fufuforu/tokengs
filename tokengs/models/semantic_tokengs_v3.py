"""Open-vocabulary semantic TokenGS with prompt-conditioned prototypes."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokengs.models.input_types import split_data
from tokengs.models.prompt_tokengs import PromptTokenGS
from tokengs.models.prompt_training import compute_prompt_mask_loss
from tokengs.models.semantic_adapter_v2 import (
    C3G8_CLASS_NAMES,
    OpenVocabSemanticMatcher,
    compute_semantic_v2_metrics,
    semantic_token_probabilities,
)
from tokengs.models.tokengs import TokenGS
from tokengs.utils.metrics import MetricsCalculator


class SemanticTokenGSv3(PromptTokenGS):
    """Eight-class prototype space with open-vocabulary text/image prompts.

    The frozen TokenGS reconstruction is unchanged. Token hidden states are
    projected into a shared 256-dim semantic space (semantic_token_adapter).
    Eight class prototypes come from adapted CLIP class texts; during training
    the prompted class's prototype is replaced by the sample's adapted prompt
    embedding (text, masked image, or a CLIP-space mix), so arbitrary text and
    image queries learn to align with the same token semantics while the joint
    eight-class supervision is preserved. At inference any text/image query can
    serve as the prototype for a binary mask.
    """

    def __init__(self, opt):
        if not getattr(opt, "prompt_training", False):
            raise ValueError("SemanticTokenGSv3 requires prompt_training=True")
        if int(opt.num_gs_tokens) != 1024:
            raise ValueError("SemanticTokenGSv3 requires num_gs_tokens=1024")
        TokenGS.__init__(self, opt)
        self.num_gaussians_per_token = int(self.opt.dec_patch_size) ** 2
        if self.num_gaussians_per_token != 64:
            raise ValueError("SemanticTokenGSv3 requires dec_patch_size=8")
        self._load_pretrained_tokengs(self.opt.prompt_tokengs_checkpoint)
        self.semantic_last_decoder = (
            deepcopy(self.enc_dec_backbone.decoder_blocks[-1])
            if self.opt.semantic_v2_tune_last_cross_attention
            else None
        )
        self.prompt_matcher = OpenVocabSemanticMatcher(
            clip_model_path=self.opt.prompt_clip_model_path,
            token_dim=self.opt.token_dim,
            semantic_dim=self.opt.semantic_v2_dim,
            temperature_init=self.opt.semantic_v2_temperature_init,
            image_pooling=self.opt.prompt_image_pooling,
        )
        # Kept as an alias so the shared trainer/metadata code paths that look
        # up semantic_matcher work unchanged for v3.
        self.semantic_matcher = self.prompt_matcher
        self._quality_metrics = MetricsCalculator(device="cpu")
        self._freeze_for_prompt_training()
        groups = self.prompt_trainable_groups()
        print(
            "[SemanticTokenGSv3] trainable parameter groups: "
            + ", ".join(
                f"{name}={sum(parameter.numel() for parameter in parameters):,}"
                for name, parameters in groups.items()
            )
        )
        print(
            "[SemanticTokenGSv3] reconstruction TokenGS remains frozen: "
            f"{not any(parameter.requires_grad for parameter in self.enc_dec_backbone.parameters())}"
        )

    def _freeze_for_prompt_training(self) -> None:
        self.requires_grad_(False)
        self.prompt_matcher.text_encoder.requires_grad_(False)
        self.prompt_matcher.image_encoder.requires_grad_(False)
        self.prompt_matcher.semantic_token_adapter.requires_grad_(True)
        self.prompt_matcher.prompt_adapter.requires_grad_(True)
        self.prompt_matcher.log_temperature.requires_grad_(True)
        if self.semantic_last_decoder is not None:
            self.semantic_last_decoder.gs_cross_attn.requires_grad_(True)
            self.semantic_last_decoder.gs_cross_attn_scale.requires_grad_(True)
        if self.opt.prompt_unfreeze_tokengs:
            for name, parameter in self.named_parameters():
                if not name.startswith("prompt_matcher."):
                    parameter.requires_grad_(True)
        self.train(True)

    def prompt_trainable_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = {
            "token_adapter": list(
                self.prompt_matcher.semantic_token_adapter.parameters()
            ),
            "prompt_adapter": list(
                self.prompt_matcher.prompt_adapter.parameters()
            ),
            "temperature": [self.prompt_matcher.log_temperature],
        }
        if self.semantic_last_decoder is not None:
            groups["last_cross_attention"] = [
                parameter
                for parameter in self.semantic_last_decoder.parameters()
                if parameter.requires_grad
            ]
        if self.opt.prompt_unfreeze_tokengs:
            groups["tokengs"] = [
                parameter
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
                and not name.startswith("prompt_matcher.")
            ]
        return groups

    semantic_trainable_groups = prompt_trainable_groups

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if self.opt.prompt_unfreeze_tokengs:
            for name, parameter in self.named_parameters():
                if parameter.requires_grad and not name.startswith(
                    "prompt_matcher."
                ):
                    state[name] = parameter.detach().cpu()
        return state

    def _forward_prompt_reconstruction(self, model_input):
        if not self.opt.prompt_unfreeze_tokengs:
            return super()._forward_prompt_reconstruction(model_input)
        # Joint semantic+geometry training: run the encoder/decoder with
        # gradients so the RGB loss and the mask loss both shape the geometry.
        reconstruction, hidden = super().forward_reconstruction(
            model_input, return_gs_token_hidden=True
        )
        rgb_results = super().render_reconstruction(
            reconstruction, model_input.decoder
        )
        return reconstruction, hidden, rgb_results

    @staticmethod
    def _build_targets(
        semantic_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if semantic_labels.ndim != 4:
            raise ValueError("semantic_label_output must have shape [B,V,H,W]")
        class_ids = torch.arange(
            1, 9, device=semantic_labels.device, dtype=semantic_labels.dtype
        ).view(1, 8, 1, 1, 1)
        targets = (semantic_labels[:, None] == class_ids).float().unsqueeze(3)
        label_valid = (semantic_labels != 0)[:, None, :, None]
        return targets, label_valid.expand_as(targets)

    @staticmethod
    def _embedding_diagnostics(
        semantic_tokens: torch.Tensor,
        semantic_prompts: torch.Tensor,
        token_probabilities: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            token_float = semantic_tokens.float()
            token_variance = token_float.var(dim=1, unbiased=False).mean()
            adjacent_cosine = (
                token_float * torch.roll(token_float, shifts=1, dims=1)
            ).sum(dim=-1).mean()
            prototype_cosine = torch.einsum(
                "bqd,bkd->bqk", semantic_prompts.float(), semantic_prompts.float()
            ).mean(dim=0)
            offdiag = ~torch.eye(
                prototype_cosine.shape[0],
                dtype=torch.bool,
                device=prototype_cosine.device,
            )
            token_scores = token_probabilities.float()
            weights = token_scores / token_scores.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            class_region_embeddings = F.normalize(
                torch.einsum("bqt,btd->bqd", weights, token_float), dim=-1
            )
        return {
            "semantic_token_variance": token_variance,
            "semantic_token_adjacent_cosine": adjacent_cosine,
            "prototype_cosine_matrix": prototype_cosine,
            "prototype_offdiag_cosine_mean": prototype_cosine[offdiag].mean(),
            "prototype_offdiag_cosine_max": prototype_cosine[offdiag].max(),
            "class_token_score_mean": token_scores.mean(dim=(0, 2)),
            "class_token_score_std": token_scores.std(dim=(0, 2), unbiased=False),
            "class_token_score_min": token_scores.amin(dim=(0, 2)),
            "class_token_score_max": token_scores.amax(dim=(0, 2)),
            "class_region_embeddings": class_region_embeddings,
        }

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
        if gs_token_hidden.shape[1:] != (1024, 1024):
            raise RuntimeError(
                f"Expected gs_token_hidden [B,1024,1024], got "
                f"{tuple(gs_token_hidden.shape)}"
            )
        if reconstruction.gaussians.shape[1] != 65_536:
            raise RuntimeError("SemanticTokenGSv3 requires exactly 65,536 Gaussians")

        has_prompts = (
            "prompt_mode" in data and "prompt_class_id" in data
        )
        prompt_embedding = None
        positive_class_ids = None
        if has_prompts:
            prompt_embedding = self._encode_prompt_batch(data)
            # Manifest class ids are 1..8; prototype indices are 0..7.
            positive_class_ids = data["prompt_class_id"].long() - 1

        semantic_output = self.prompt_matcher(
            gs_token_hidden,
            prompt_embeddings=prompt_embedding,
            positive_class_ids=positive_class_ids,
        )
        token_logits = semantic_output["token_logits"]
        if token_logits.shape[1:] != (8, 1024):
            raise RuntimeError(
                f"Expected token logits [B,8,1024], got {token_logits.shape}"
            )
        token_probabilities = semantic_token_probabilities(
            token_logits, self.opt.semantic_v2_score_mode
        )
        gaussian_scores = token_probabilities.repeat_interleave(
            self.num_gaussians_per_token, dim=-1
        )
        semantic_render = self.gs.render_prompt_scores(
            reconstruction.gaussians,
            gaussian_scores,
            model_input.decoder.cam_view,
            intrinsics=model_input.decoder.intrinsics,
        )
        rendered_probability = semantic_render["rendered_prompt_probability"]
        rendered_alpha = semantic_render["rendered_alpha"]
        target_mask, label_valid = self._build_targets(
            data["semantic_label_output"].long()
        )
        valid_mask = (
            rendered_alpha >= float(self.opt.prompt_valid_alpha_threshold)
        ) & label_valid

        losses = compute_prompt_mask_loss(
            rendered_probability,
            target_mask,
            valid_mask,
            lambda_bce=self.opt.prompt_lambda_bce,
            lambda_dice=self.opt.prompt_lambda_dice,
            balance_classes=self.opt.semantic_v2_balanced_bce,
        )
        semantic_metrics = compute_semantic_v2_metrics(
            rendered_probability,
            target_mask,
            valid_mask,
            threshold=self.opt.prompt_threshold,
        )
        diagnostics = self._embedding_diagnostics(
            semantic_output["semantic_tokens"],
            semantic_output["semantic_prompts"],
            token_probabilities,
        )
        diagnostics["class_gt_present"] = semantic_metrics[
            "class_target_count"
        ] > 0

        pred_rgb = rgb_results["images_pred"]
        gt_rgb = supervision.images_output
        loss_rgb = torch.zeros((), device=pred_rgb.device)
        if float(self.opt.lambda_rgb) > 0:
            loss_rgb = float(self.opt.lambda_rgb) * (
                pred_rgb.clamp(0, 1) - gt_rgb.clamp(0, 1)
            ).square().mean()
        joint_loss = losses["loss"] + loss_rgb
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

        return {
            **losses,
            "loss": joint_loss,
            "loss_rgb": loss_rgb,
            **semantic_metrics,
            **diagnostics,
            **rgb_results,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "gaussians": reconstruction.gaussians,
            "gs_token_hidden": gs_token_hidden,
            "semantic_tokens": semantic_output["semantic_tokens"],
            "semantic_prompts": semantic_output["semantic_prompts"],
            "temperature": semantic_output["temperature"],
            "token_logits": token_logits,
            "token_probabilities": token_probabilities,
            "gaussian_scores": gaussian_scores,
            "rendered_prompt_probability": rendered_probability,
            "rendered_alpha": rendered_alpha,
            "valid_mask": valid_mask,
            "target_prompt_mask": target_mask,
            "semantic_class_names": C3G8_CLASS_NAMES,
        }
