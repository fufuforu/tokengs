"""Per-Gaussian feature field with an LSeg teacher and a pixel decode head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokengs.models.input_types import split_data
from tokengs.models.lseg_teacher import LSegTeacher
from tokengs.models.semantic_adapter_v2 import (
    C3G8_CLASS_NAMES,
    semantic_token_probabilities,
)
from tokengs.models.semantic_tokengs_v4 import (
    SemanticTokenGSv4,
)


class RenderedSemanticClassifier(nn.Module):
    """Small pixel-level CNN decoding rendered per-Gaussian features.

    Input  : rendered feature map [B*V, D, H, W]
    Output : per-pixel class logits [B*V, num_classes, H, W]
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 8,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(self, rendered_features: torch.Tensor) -> torch.Tensor:
        batch, views = rendered_features.shape[:2]
        logits = self.net(
            rendered_features.reshape(
                batch * views,
                *rendered_features.shape[2:],
            )
        )
        return logits.reshape(batch, views, *logits.shape[1:])


class SemanticTokenGSv5(SemanticTokenGSv4):
    """Open-vocabulary prompt model with LSeg lifting and pixel decoding.

    Builds on the v4 per-Gaussian feature field (still driven by arbitrary
    CLIP text/image prompts) and adds:
      * a frozen LSeg teacher whose dense 512-dim features supervise the
        field through a fixed random projection (feature lifting),
      * a pixel-level classifier head on rendered features, trained with the
        ScanNet eight-class ground truth (per-pixel decoding),
      * optional joint geometry training (lambda_rgb), same as v4.
    The C3G protocol evaluation uses the pixel decoder logits; arbitrary
    text/image prompts still use the cosine field.
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.pixel_classifier = RenderedSemanticClassifier(
            feature_dim=getattr(opt, "semantic_v4_feature_dim", 64),
            hidden_dim=int(getattr(opt, "semantic_v5_classifier_hidden", 128)),
            num_classes=8,
        )
        for parameter in self.pixel_classifier.parameters():
            parameter.requires_grad_(True)
        # Frozen random orthogonal projection from LSeg 512-dim to field dim.
        generator = torch.Generator().manual_seed(1)
        raw = torch.randn(
            512,
            self.pixel_classifier.feature_dim,
            generator=generator,
        )
        q, _ = torch.linalg.qr(raw)
        self.register_buffer(
            "lseg_projection",
            q[:, : self.pixel_classifier.feature_dim],
            persistent=False,
        )
        self._lseg_teacher = None
        self.lseg_checkpoint = str(
            getattr(
                opt,
                "lseg_checkpoint_path",
                "/space0/mawb/tokengs/checkpoints/demo_e200.ckpt",
            )
        )
        self._lseg_initialized = False
        print(
            "[SemanticTokenGSv5] pixel classifier params: "
            f"{sum(parameter.numel() for parameter in self.pixel_classifier.parameters()):,}"
        )

    def _ensure_lseg(self) -> LSegTeacher:
        if self._lseg_teacher is None:
            self._lseg_teacher = LSegTeacher(self.lseg_checkpoint)
        return self._lseg_teacher

    def prompt_trainable_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = super().prompt_trainable_groups()
        if hasattr(self, "pixel_classifier"):
            groups["pixel_classifier"] = list(
                self.pixel_classifier.parameters()
            )
        return groups

    semantic_trainable_groups = prompt_trainable_groups

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for key, value in self.pixel_classifier.state_dict().items():
            state[f"pixel_classifier.{key}"] = value
        return state

    @staticmethod
    def _pixel_ce_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-entropy on valid (alpha + labeled) pixels; background ignored."""
        batch, views = logits.shape[:2]
        logits_flat = logits.permute(0, 1, 3, 4, 2).reshape(-1, 8)
        labels_flat = torch.where(
            labels.reshape(-1) > 0,
            labels.reshape(-1) - 1,
            torch.tensor(-100, device=labels.device),
        )
        valid_flat = valid.reshape(-1)
        ce_map = F.cross_entropy(
            logits_flat.float(),
            labels_flat,
            ignore_index=-100,
            reduction="none",
        )
        masked = (ce_map * valid_flat.float()).sum()
        denom = valid_flat.float().sum().clamp_min(1e-6)
        return masked / denom, (valid_flat.float().sum() / max(valid_flat.numel(), 1))

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
        gaussians = reconstruction.gaussians

        has_prompts = (
            "prompt_mode" in data and "prompt_class_id" in data
        )
        prompt_embedding = None
        positive_class_ids = None
        if has_prompts:
            prompt_embedding = self._encode_prompt_batch(data)
            positive_class_ids = data["prompt_class_id"].long() - 1

        semantic_output = self.prompt_matcher(
            gs_token_hidden,
            gaussians,
            prompt_embeddings=prompt_embedding,
            positive_class_ids=positive_class_ids,
        )
        gaussian_features = semantic_output["gaussian_features"]
        gaussian_logits = semantic_output["token_logits"]
        token_probabilities = semantic_token_probabilities(
            gaussian_logits, self.opt.semantic_v2_score_mode
        )
        feature_dim = gaussian_features.shape[-1]
        lambda_ce_cosine = float(getattr(self.opt, "lambda_ce_cosine", 0.0))
        use_ce_cosine = self.training and lambda_ce_cosine > 0
        render_parts = [
            gaussian_features,
            token_probabilities.transpose(1, 2),
        ]
        if use_ce_cosine:
            render_parts.append(gaussian_logits.transpose(1, 2))
        render_channels = torch.cat(render_parts, dim=-1)
        feature_render = self.gs.render_feature_channels(
            gaussians,
            render_channels,
            model_input.decoder.cam_view,
            intrinsics=model_input.decoder.intrinsics,
        )
        rendered_all = feature_render["images_pred"]
        rendered_alpha = feature_render["alphas_pred"]
        rendered_features = rendered_all[:, :, :feature_dim]
        prob_end = feature_dim + 8
        rendered_probability = (
            rendered_all[:, :, feature_dim:prob_end]
            .permute(0, 2, 1, 3, 4)
            .unsqueeze(3)
        )
        pixel_logits = self.pixel_classifier(rendered_features)

        target_mask, label_valid = self._build_targets(
            data["semantic_label_output"].long()
        )
        valid_mask = (
            rendered_alpha >= float(self.opt.prompt_valid_alpha_threshold)
        ) & label_valid
        losses = self._mask_losses(
            rendered_probability, target_mask, valid_mask
        )
        semantic_metrics = self._semantic_metrics(
            rendered_probability, target_mask, valid_mask
        )

        gt_labels = data["semantic_label_output"].long()
        alpha_valid = (
            rendered_alpha >= float(self.opt.prompt_valid_alpha_threshold)
        )
        loss_ce, ce_valid_ratio = self._pixel_ce_loss(
            pixel_logits,
            gt_labels,
            alpha_valid & (gt_labels > 0),
        )
        lambda_ce = float(getattr(self.opt, "lambda_ce", 1.0))
        loss_ce = lambda_ce * loss_ce

        loss_feat = torch.zeros((), device=rendered_probability.device)
        if self.training and float(self.opt.lambda_feat) > 0:
            loss_feat = self._lseg_feature_loss(
                supervision.images_output,
                rendered_features,
                alpha_valid,
            )

        loss_ce_cosine = torch.zeros((), device=rendered_probability.device)
        if use_ce_cosine:
            rendered_logits = rendered_all[:, :, prob_end:]  # [B,V,8,H,W]
            gt_labels = data["semantic_label_output"].long()
            valid_ce = alpha_valid[:, :, 0] & (gt_labels > 0)
            loss_ce_cosine = lambda_ce_cosine * self._cosine_ce_loss(
                rendered_logits,
                gt_labels,
                valid_ce,
                getattr(self.opt, "semantic_v2_class_weights", None),
            )

        pred_rgb = rgb_results["images_pred"]
        gt_rgb = supervision.images_output
        loss_rgb = torch.zeros((), device=pred_rgb.device)
        if float(self.opt.lambda_rgb) > 0:
            loss_rgb = float(self.opt.lambda_rgb) * (
                pred_rgb.clamp(0, 1) - gt_rgb.clamp(0, 1)
            ).square().mean()
        joint_loss = (
            losses["loss"] + loss_rgb + loss_feat + loss_ce + loss_ce_cosine
        )

        token_probabilities_token = token_probabilities.view(
            token_probabilities.shape[0], 8, 1024, 64
        ).mean(dim=-1)
        diagnostics = self._embedding_diagnostics(
            semantic_output["semantic_tokens"],
            semantic_output["semantic_prompts"],
            token_probabilities_token,
        )
        diagnostics["class_gt_present"] = semantic_metrics[
            "class_target_count"
        ] > 0
        diagnostics["ce_valid_ratio"] = ce_valid_ratio

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
            "loss_feat": loss_feat,
            "loss_ce": loss_ce,
            "loss_ce_cosine": loss_ce_cosine,
            **semantic_metrics,
            **diagnostics,
            **rgb_results,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "gaussians": gaussians,
            "gs_token_hidden": gs_token_hidden,
            "semantic_tokens": semantic_output["semantic_tokens"],
            "semantic_prompts": semantic_output["semantic_prompts"],
            "gaussian_features": gaussian_features,
            "temperature": semantic_output["temperature"],
            "token_logits": gaussian_logits,
            "token_probabilities": token_probabilities,
            "gaussian_scores": token_probabilities,
            "rendered_prompt_probability": rendered_probability,
            "rendered_pixel_logits": pixel_logits,
            "rendered_alpha": rendered_alpha,
            "valid_mask": valid_mask,
            "target_prompt_mask": target_mask,
            "semantic_class_names": C3G8_CLASS_NAMES,
        }

    def _mask_losses(
        self,
        rendered_probability: torch.Tensor,
        target_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        from tokengs.models.prompt_training import compute_prompt_mask_loss

        return compute_prompt_mask_loss(
            rendered_probability,
            target_mask,
            valid_mask,
            lambda_bce=self.opt.prompt_lambda_bce,
            lambda_dice=self.opt.prompt_lambda_dice,
            balance_classes=self.opt.semantic_v2_balanced_bce,
            class_weights=getattr(
                self.opt, "semantic_v2_class_weights", None
            ),
        )

    def _semantic_metrics(
        self,
        rendered_probability: torch.Tensor,
        target_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        from tokengs.models.semantic_adapter_v2 import (
            compute_semantic_v2_metrics,
        )

        return compute_semantic_v2_metrics(
            rendered_probability,
            target_mask,
            valid_mask,
            threshold=self.opt.prompt_threshold,
        )

    def _lseg_feature_loss(
        self,
        target_rgb: torch.Tensor,
        rendered_features: torch.Tensor,
        alpha_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine distillation of rendered features against LSeg target view."""
        teacher = self._ensure_lseg()
        batch, views = target_rgb.shape[:2]
        lseg_feat = teacher.extract(
            target_rgb.reshape(batch * views, 3, *target_rgb.shape[3:])
        )  # [B*V,512,128,128]
        lseg_feat = lseg_feat.reshape(
            batch, views, -1, *lseg_feat.shape[2:]
        )
        teacher_map = torch.einsum(
            "bvchw,cd->bvdhw", lseg_feat, self.lseg_projection
        )
        teacher_map = F.normalize(teacher_map.float(), dim=2)
        teacher_map = F.interpolate(
            teacher_map.reshape(batch * views, *teacher_map.shape[2:]),
            size=self.opt.img_size,
            mode="bilinear",
            align_corners=False,
        ).reshape(
            batch, views, teacher_map.shape[2], *self.opt.img_size
        )
        teacher_map = F.normalize(teacher_map, dim=2)
        rendered = F.normalize(rendered_features.float(), dim=2)
        cosine = (rendered * teacher_map).sum(dim=2)
        valid = alpha_valid[:, :, 0]
        loss = (1.0 - cosine)[valid].mean()
        return float(self.opt.lambda_feat) * loss
