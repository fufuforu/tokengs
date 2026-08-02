"""Frozen TokenGS with an eight-class shared cosine semantic adapter."""

from __future__ import annotations

from copy import deepcopy
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from tokengs.models.input_types import split_data
from tokengs.models.prompt_training import (
    compute_prompt_mask_loss,
)
from tokengs.models.semantic_adapter_v2 import (
    C3G8_CLASS_NAMES,
    SemanticMatcherV2,
    compute_semantic_v2_metrics,
    semantic_token_probabilities,
)
from tokengs.models.tokengs import TokenGS
from tokengs.utils.metrics import MetricsCalculator


class SemanticTokenGSv2(TokenGS):
    """Text-only eight-class semantic diagnostic over frozen TokenGS tokens."""

    def __init__(self, opt):
        if not getattr(opt, "prompt_training", False):
            raise ValueError("SemanticTokenGSv2 requires prompt_training=True")
        if int(opt.num_gs_tokens) != 1024:
            raise ValueError("SemanticTokenGSv2 requires num_gs_tokens=1024")
        super().__init__(opt)

        self.num_gaussians_per_token = int(self.opt.dec_patch_size) ** 2
        if self.num_gaussians_per_token != 64:
            raise ValueError("SemanticTokenGSv2 requires 64 Gaussians per token")
        self._load_pretrained_tokengs(self.opt.prompt_tokengs_checkpoint)
        self.semantic_last_decoder = None
        if self.opt.semantic_v2_tune_last_cross_attention:
            # The reconstruction path keeps the original frozen block. This copy is
            # initialized identically and may specialize without moving RGB geometry.
            self.semantic_last_decoder = deepcopy(
                self.enc_dec_backbone.decoder_blocks[-1]
            )
        self.semantic_matcher = SemanticMatcherV2(
            clip_model_path=self.opt.prompt_clip_model_path,
            token_dim=self.opt.token_dim,
            semantic_dim=self.opt.semantic_v2_dim,
            temperature_init=self.opt.semantic_v2_temperature_init,
        )
        self._quality_metrics = MetricsCalculator(device="cpu")
        self._freeze_for_semantic_training()
        trainable_groups = self.semantic_trainable_groups()
        print(
            "[SemanticTokenGSv2] trainable parameter groups: "
            + ", ".join(
                f"{name}={sum(parameter.numel() for parameter in parameters):,}"
                for name, parameters in trainable_groups.items()
            )
        )
        print(
            "[SemanticTokenGSv2] reconstruction decoder remains frozen: "
            f"{not any(parameter.requires_grad for parameter in self.enc_dec_backbone.parameters())}"
        )

    def _load_pretrained_tokengs(self, checkpoint_path: str) -> None:
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
        checkpoint_token_shape = tuple(checkpoint.get("gs_tokens", torch.empty(0)).shape)
        current_token_shape = tuple(current["gs_tokens"].shape)
        print(f"[SemanticTokenGSv2] pretrained checkpoint: {path.resolve()}")
        print(f"[SemanticTokenGSv2] checkpoint GS token shape: {checkpoint_token_shape}")
        print(f"[SemanticTokenGSv2] configured GS token shape: {current_token_shape}")
        print(f"[SemanticTokenGSv2] missing keys: {missing}")
        print(f"[SemanticTokenGSv2] unexpected keys: {unexpected}")
        print(f"[SemanticTokenGSv2] shape-mismatched keys: {mismatched}")
        if checkpoint_token_shape != (1024, 1024):
            raise RuntimeError(
                f"Expected checkpoint gs_tokens [1024,1024], got {checkpoint_token_shape}"
            )
        if current_token_shape != checkpoint_token_shape:
            raise RuntimeError("Configured GS tokens do not match the RE10K checkpoint")
        if missing or unexpected or mismatched:
            raise RuntimeError("Pretrained TokenGS checkpoint failed strict validation")
        with torch.no_grad():
            for key, value in checkpoint.items():
                current[key].copy_(value)

    def _freeze_for_semantic_training(self) -> None:
        self.requires_grad_(False)
        self.semantic_matcher.semantic_token_adapter.requires_grad_(True)
        self.semantic_matcher.prompt_semantic_adapter.requires_grad_(True)
        self.semantic_matcher.log_temperature.requires_grad_(True)
        if self.semantic_last_decoder is not None:
            self.semantic_last_decoder.gs_cross_attn.requires_grad_(True)
            self.semantic_last_decoder.gs_cross_attn_scale.requires_grad_(True)
        self.train(True)

    def train(self, mode: bool = True) -> "SemanticTokenGSv2":
        nn.Module.train(self, mode)
        for name, module in self.named_children():
            if name != "semantic_matcher":
                module.eval()
        if hasattr(self, "semantic_matcher"):
            self.semantic_matcher.train(mode)
        if self.semantic_last_decoder is not None:
            self.semantic_last_decoder.train(mode)
        return self

    def state_dict(self, *args, **kwargs) -> OrderedDict[str, torch.Tensor]:
        del args, kwargs
        state = self.semantic_matcher.trainable_state_dict()
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
        del assign
        current = self.state_dict()
        missing = sorted(set(current) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(current))
        mismatched = sorted(
            key
            for key in set(current) & set(state_dict)
            if current[key].shape != state_dict[key].shape
        )
        if strict and (missing or unexpected or mismatched):
            raise RuntimeError(
                "SemanticTokenGSv2 checkpoint mismatch: "
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

    def semantic_trainable_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = {
            "token_adapter": list(
                self.semantic_matcher.semantic_token_adapter.parameters()
            ),
            "prompt_adapter": list(
                self.semantic_matcher.prompt_semantic_adapter.parameters()
            ),
            "temperature": [self.semantic_matcher.log_temperature],
        }
        if self.semantic_last_decoder is not None:
            groups["last_cross_attention"] = [
                parameter
                for parameter in self.semantic_last_decoder.parameters()
                if parameter.requires_grad
            ]
        return groups

    def _forward_semantic_reconstruction(self, model_input):
        """Run frozen geometry and an optional trainable semantic last-layer fork."""
        if self.semantic_last_decoder is None:
            with torch.no_grad():
                reconstruction, hidden = super().forward_reconstruction(
                    model_input, return_gs_token_hidden=True
                )
                rgb_results = super().render_reconstruction(
                    reconstruction, model_input.decoder
                )
            return reconstruction, hidden, hidden, rgb_results

        with torch.no_grad():
            encoder_latent = super().forward_encoder(model_input.encoder)
            hidden = super().get_gs_tokens(encoder_latent.keys.shape[0])
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
        return reconstruction, geometry_hidden, semantic_hidden, rgb_results

    @staticmethod
    def _build_targets(semantic_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
            weights = token_scores / token_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
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
        (
            reconstruction,
            geometry_gs_token_hidden,
            gs_token_hidden,
            rgb_results,
        ) = self._forward_semantic_reconstruction(model_input)
        if gs_token_hidden.shape[1:] != (1024, 1024):
            raise RuntimeError(
                f"Expected gs_token_hidden [B,1024,1024], got {gs_token_hidden.shape}"
            )
        if reconstruction.gaussians.shape[1] != 65_536:
            raise RuntimeError("SemanticTokenGSv2 requires exactly 65,536 Gaussians")

        semantic_output = self.semantic_matcher(gs_token_hidden)
        token_logits = semantic_output["token_logits"]
        if token_logits.shape[1:] != (8, 1024):
            raise RuntimeError(f"Expected token logits [B,8,1024], got {token_logits.shape}")
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
            **semantic_metrics,
            **diagnostics,
            **rgb_results,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "gaussians": reconstruction.gaussians,
            "gs_token_hidden": gs_token_hidden,
            "geometry_gs_token_hidden": geometry_gs_token_hidden,
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
