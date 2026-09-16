"""Per-Gaussian open-vocabulary feature field with dense CLIP distillation."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from tokengs.models.input_types import split_data
from tokengs.models.prompt_matching import PromptEncoder
from tokengs.models.prompt_tokengs import PromptTokenGS
from tokengs.models.prompt_training import (
    compute_prompt_mask_loss,
    instance_contrastive_loss,
)
from tokengs.models.lifting_semantic import (
    GaussianSemanticHeadV2,
    RenderedSemanticClassifier,
    SourceFeatureProjector,
    lseg_features_to_pseudo_labels,
    semantic_feature_loss,
)
from tokengs.models.semantic_token_v3 import TokenSemanticFieldV3
from tokengs.models.lseg_teacher import LSegTeacher
from tokengs.models.semantic_adapter_v2 import (
    C3G8_CLASS_NAMES,
    PromptSemanticAdapter,
    SemanticTokenAdapter,
    compute_semantic_v2_metrics,
    semantic_token_probabilities,
)
from tokengs.models.instance_group_head import InstanceGroupHead
from tokengs.models.absolute_unit_decoder import teacher_gs_distill_loss
from tokengs.models.instance_group_loss import (
    count_instance_masks,
    hungarian_instance_group_loss,
    instance_group_3d_loss,
)
from tokengs.models.ta_riu_v2 import (
    build_unit_soft_instance_targets,
    soft_unit_info_nce,
)
from tokengs.models.ta_riu_v3 import (
    TARIUV3DualStream,
    unflatten_reconstruction_units,
)
from tokengs.models.tokengs import TokenGS
from tokengs.utils.metrics import MetricsCalculator


class _GradScale(torch.autograd.Function):
    """Forward-identity / backward-scaled gradient gate.

    Used to attach the semantic / instance branches to the shared
    ``gs_token_hidden`` latent while keeping their gradients small relative
    to the reconstruction path: forward is the identity, backward
    multiplies the incoming gradient by ``alpha``.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.alpha, None


def grad_scale(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Identity in the forward pass, ``alpha``-scaled gradient backward."""
    if float(alpha) == 1.0:
        return x
    return _GradScale.apply(x, float(alpha))


class GaussianFeatureFieldMatcher(nn.Module):
    """Per-Gaussian semantic feature field with CLIP text/image matching.

    Token hidden states are projected into a shared semantic space and expanded
    to per-Gaussian features conditioned on each Gaussian's geometry and
    appearance. Arbitrary CLIP text/image prompts become dynamic prototypes
    matched against those features by cosine similarity, so the model stays
    open-vocabulary while gaining per-Gaussian granularity. During training the
    field is additionally supervised by dense CLIP patch features of the target
    views (feature lifting), either through a frozen random projection or a
    trainable alignment layer.
    """

    def __init__(
        self,
        clip_model_path: str,
        token_dim: int = 1024,
        semantic_dim: int = 256,
        feature_dim: int = 32,
        temperature_init: float = 14.285714,
        class_names=C3G8_CLASS_NAMES,
        use_geometry: bool = True,
        teacher_projection: str = "frozen_random",
        image_pooling: str = "masked_input_cls",
    ):
        super().__init__()
        if tuple(class_names) != C3G8_CLASS_NAMES:
            raise ValueError(
                "GaussianFeatureFieldMatcher requires the C3G8 class order"
            )
        if temperature_init <= 0:
            raise ValueError("temperature_init must be positive")
        if teacher_projection not in ("frozen_random", "trainable"):
            raise ValueError(
                f"Unknown teacher_projection: {teacher_projection}"
            )
        self.class_names = tuple(class_names)
        self.feature_dim = int(feature_dim)
        self.use_geometry = bool(use_geometry)

        self.prompt_encoder = PromptEncoder(
            clip_model_path, image_pooling=image_pooling
        )
        self.prompt_encoder.requires_grad_(False)
        self.prompt_encoder.eval()

        self.semantic_token_adapter = SemanticTokenAdapter(
            token_dim, semantic_dim
        )
        geometry_dim = 14 if self.use_geometry else 0
        hidden_dim = max(64, self.feature_dim * 4)
        self.gaussian_feature_head = nn.Sequential(
            nn.LayerNorm(semantic_dim + geometry_dim),
            nn.Linear(semantic_dim + geometry_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.feature_dim),
        )
        self.prompt_semantic_adapter = PromptSemanticAdapter(
            self.prompt_encoder.output_dim, self.feature_dim
        )
        # Alias kept for shared freeze/trainable-group code paths.
        self.prompt_adapter = self.prompt_semantic_adapter
        self.log_temperature = nn.Parameter(
            torch.tensor(float(temperature_init)).log()
        )

        with torch.no_grad():
            clip_prototypes = self.prompt_encoder.encode_text(
                [list(self.class_names)]
            )[0]
        if tuple(clip_prototypes.shape) != (
            len(self.class_names),
            self.prompt_encoder.output_dim,
        ):
            raise RuntimeError(
                "Expected one CLIP prototype per class, got "
                f"{tuple(clip_prototypes.shape)}"
            )
        self.register_buffer(
            "clip_text_prototypes", clip_prototypes, persistent=False
        )

        if teacher_projection == "frozen_random":
            generator = torch.Generator().manual_seed(0)
            raw = torch.randn(
                self.prompt_encoder.output_dim,
                self.feature_dim,
                generator=generator,
            )
            q, _ = torch.linalg.qr(raw)
            # Orthonormal columns preserve CLIP geometry without collapse.
            self.register_buffer(
                "teacher_projection",
                q[:, : self.feature_dim],
                persistent=False,
            )
            self.feature_align = None
        else:
            self.teacher_projection = None
            self.feature_align = nn.Linear(
                self.prompt_encoder.output_dim, self.feature_dim
            )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(max=100.0)

    def train(self, mode: bool = True) -> "GaussianFeatureFieldMatcher":
        super().train(mode)
        self.prompt_encoder.eval()
        self.prompt_encoder.requires_grad_(False)
        return self

    def trainable_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state: OrderedDict[str, torch.Tensor] = OrderedDict()
        for prefix, module in (
            ("semantic_token_adapter", self.semantic_token_adapter),
            ("gaussian_feature_head", self.gaussian_feature_head),
            ("prompt_semantic_adapter", self.prompt_semantic_adapter),
        ):
            for key, value in module.state_dict().items():
                state[f"{prefix}.{key}"] = value
        if self.feature_align is not None:
            for key, value in self.feature_align.state_dict().items():
                state[f"feature_align.{key}"] = value
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
                "GaussianFeatureFieldMatcher checkpoint mismatch: "
                f"missing={missing}, unexpected={unexpected}, "
                f"mismatched={mismatched}"
            )
        with torch.no_grad():
            for key in set(current) & set(state_dict):
                if current[key].shape == state_dict[key].shape:
                    current[key].copy_(state_dict[key])
        return nn.modules.module._IncompatibleKeys(missing, unexpected + mismatched)

    def build_prototypes(
        self,
        batch_size: int,
        prompt_embeddings: torch.Tensor | None = None,
        positive_class_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_prototypes = (
            self.prompt_semantic_adapter(self.clip_text_prototypes)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )
        if prompt_embeddings is None:
            return text_prototypes
        if positive_class_ids is None:
            raise ValueError(
                "positive_class_ids is required with prompt embeddings"
            )
        adapted_prompts = self.prompt_semantic_adapter(prompt_embeddings)
        if adapted_prompts.shape[1] != 1:
            raise ValueError(
                "prompt_embeddings must have shape [B,1,dim] for prototype "
                f"replacement, got {tuple(adapted_prompts.shape)}"
            )
        adapted_prompts = adapted_prompts[:, 0]
        if positive_class_ids.min() < 0 or positive_class_ids.max() >= 8:
            raise ValueError(
                f"positive_class_ids out of range: {positive_class_ids.tolist()}"
            )
        # One-hot replacement avoids CUDA advanced-indexing assignment quirks.
        one_hot = F.one_hot(
            positive_class_ids, num_classes=len(self.class_names)
        ).float().unsqueeze(-1)  # [B,8,1]
        text_prototypes = (
            text_prototypes * (1.0 - one_hot)
            + adapted_prompts[:, None, :] * one_hot
        )
        return text_prototypes

    def forward(
        self,
        token_hidden: torch.Tensor,
        gaussians: torch.Tensor,
        prompt_embeddings: torch.Tensor | None = None,
        positive_class_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if token_hidden.ndim != 3:
            raise ValueError("token_hidden must have shape [B,T,1024]")
        if gaussians.ndim != 3 or gaussians.shape[1] != token_hidden.shape[1] * 64:
            raise ValueError(
                "gaussians must have shape [B,T*64,14] with 64 Gaussians "
                f"per token, got {tuple(gaussians.shape)}"
            )
        batch_size = token_hidden.shape[0]
        token_semantic = self.semantic_token_adapter(token_hidden)
        token_features = token_semantic.repeat_interleave(64, dim=1)
        inputs = [token_features]
        if self.use_geometry:
            inputs.append(gaussians[..., :14].detach().float())
        gaussian_features = F.normalize(
            self.gaussian_feature_head(torch.cat(inputs, dim=-1)).float(),
            dim=-1,
        )
        semantic_prompts = self.build_prototypes(
            batch_size, prompt_embeddings, positive_class_ids
        )
        gaussian_logits = (
            torch.einsum("bqd,bnd->bqn", semantic_prompts, gaussian_features)
            * self.temperature
        )
        return {
            "gaussian_features": gaussian_features,
            "semantic_tokens": token_semantic,
            "semantic_prompts": semantic_prompts,
            "token_logits": gaussian_logits,
            "temperature": self.temperature,
        }


class SemanticTokenGSv4(PromptTokenGS):
    """Open-vocabulary prompt model with a per-Gaussian feature field.

    Keeps the TokenGS reconstruction backbone, adds a per-Gaussian feature
    field and dense CLIP feature distillation (feature lifting), and matches
    arbitrary text/image prompts against the field. The eight-class C3G8
    supervision and the open-vocabulary prompt interface are unchanged.
    """

    def __init__(self, opt):
        if not getattr(opt, "prompt_training", False):
            raise ValueError("SemanticTokenGSv4 requires prompt_training=True")
        if int(opt.num_gs_tokens) <= 0:
            raise ValueError("num_gs_tokens must be positive")
        TokenGS.__init__(self, opt)
        self.num_gaussians_per_token = int(self.opt.dec_patch_size) ** 2
        if self.num_gaussians_per_token != 64:
            raise ValueError("SemanticTokenGSv4 requires dec_patch_size=8")
        self._load_pretrained_tokengs(self.opt.prompt_tokengs_checkpoint)
        self.semantic_last_decoder = None
        self.prompt_matcher = GaussianFeatureFieldMatcher(
            clip_model_path=self.opt.prompt_clip_model_path,
            token_dim=self.opt.token_dim,
            semantic_dim=self.opt.semantic_v2_dim,
            feature_dim=getattr(self.opt, "semantic_v4_feature_dim", 32),
            temperature_init=self.opt.semantic_v2_temperature_init,
            use_geometry=getattr(self.opt, "semantic_v4_use_geometry", True),
            teacher_projection=getattr(
                self.opt, "semantic_v4_teacher_projection", "frozen_random"
            ),
            image_pooling=self.opt.prompt_image_pooling,
        )
        self.semantic_matcher = self.prompt_matcher
        self.instance_group_head = None
        self.semantic_lifting_head = None
        self.semantic_head = None
        self.semantic_projector = None
        self.semantic_classifier = None
        self.lseg_teacher = None
        branch_version = str(
            getattr(self.opt, "semantic_branch_version", "source_projected")
        )
        if float(getattr(self.opt, "lambda_semantic_feature", 0.0)) > 0:
            if branch_version == "token_decoder_lowrank":
                # V3 token-compressed semantic field (reproduces the old
                # ~0.536 ScanNet C3G8 result). Kept under ``semantic_head``
                # so the pretrained ``semantic_head.*`` weights load directly.
                self.semantic_head = TokenSemanticFieldV3(
                    num_tokens=int(self.opt.num_gs_tokens),
                    geometry_token_dim=int(self.opt.token_dim),
                    feature_dim=512,
                    num_gaussians_per_token=self.num_gaussians_per_token,
                    semantic_patch_size=int(
                        getattr(self.opt, "semantic_token_patch_size", 8)
                    ),
                    num_decoder_layers=int(
                        getattr(self.opt, "semantic_token_decoder_layers", 2)
                    ),
                    num_heads=int(
                        getattr(self.opt, "semantic_token_decoder_heads", 8)
                    ),
                    mlp_ratio=float(
                        getattr(self.opt, "semantic_token_mlp_ratio", 4.0)
                    ),
                    local_rank=int(
                        getattr(self.opt, "semantic_local_rank", 16)
                    ),
                    local_hidden_dim=int(
                        getattr(self.opt, "semantic_local_hidden_dim", 128)
                    ),
                    max_views=int(
                        getattr(self.opt, "semantic_token_max_views", 8)
                    ),
                    token_residual_scale=float(
                        getattr(self.opt, "semantic_token_residual_scale", 0.1)
                    ),
                    local_residual_scale=float(
                        getattr(self.opt, "semantic_local_residual_scale", 0.1)
                    ),
                    pool_use_opacity=bool(
                        getattr(self.opt, "semantic_token_pool_use_opacity", True)
                    ),
                    dropout=float(
                        getattr(self.opt, "semantic_token_dropout", 0.0)
                    ),
                )
            else:
                self.semantic_lifting_head = GaussianSemanticHeadV2(
                    token_dim=int(self.opt.token_dim),
                    feature_dim=512,
                    local_dim=128,
                    residual_scale=float(
                        getattr(self.opt, "semantic_residual_scale", 0.1)
                    ),
                )
            self.semantic_projector = SourceFeatureProjector(
                image_hw=tuple(self.opt.img_size)
            )
            self.lseg_teacher = LSegTeacher(
                getattr(self.opt, "lseg_checkpoint_path", None)
                or "/space0/mawb/tokengs/checkpoints/demo_e200.ckpt"
            )
        if branch_version == "token_decoder_lowrank" and self.semantic_head is None:
            raise ValueError(
                "token_decoder_lowrank requires lambda_semantic_feature > 0"
            )
        if float(getattr(self.opt, "lambda_semantic_ce", 0.0)) > 0:
            self.semantic_classifier = RenderedSemanticClassifier(
                feature_dim=512,
                hidden_dim=int(
                    getattr(self.opt, "semantic_classifier_hidden_dim", 256)
                ),
                num_classes=int(getattr(self.opt, "semantic_num_classes", 8)),
            )
        self.anchor_pos_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.instance_group_lambda_eff = 1.0
        self.teacher_lambda_eff = 0.0
        self.instance_stage_eff = 1.0
        # Guarded-joint baseline schedule state (set per step by the trainer
        # or directly by smoke harnesses).
        self.guarded_instance_loss_weight_eff = 0.0
        self.guarded_instance_unit_grad_eff = 0.0
        # True-Shared baseline schedule state.
        self.tsh_instance_loss_weight_eff = 0.0
        self.tsh_unit_grad_eff = 0.0
        # SIU3R-style U->R (mask-guided depth smoothness) effective weight,
        # set by the trainer each step (model.forward reads it).
        self.tsh_mbm_u2r_eff = 0.0
        # Per-GS slot refinement state (built by v6 when enabled).
        self.tsh_slot_refine_head = None
        self.tsh_per_gs_gate_eff = 0.0
        # Undetached weighted U->R scalar of the latest training forward
        # (used by gradient audits; zero/None when disabled).
        self._tsh_last_mbm_u2r_loss = None
        self.teacher_called = False
        self._teacher_gaussians = None
        self._teacher_rgb = None
        self._quality_metrics = MetricsCalculator(device="cpu")
        self._freeze_for_prompt_training()
        groups = self.prompt_trainable_groups()
        print(
            "[SemanticTokenGSv4] trainable parameter groups: "
            + ", ".join(
                f"{name}={sum(parameter.numel() for parameter in parameters):,}"
                for name, parameters in groups.items()
            )
        )
        print(
            "[SemanticTokenGSv4] reconstruction TokenGS remains frozen: "
            f"{not any(parameter.requires_grad for parameter in self.enc_dec_backbone.parameters())}"
        )

    def _freeze_for_prompt_training(self) -> None:
        self.requires_grad_(False)
        self.prompt_matcher.prompt_encoder.requires_grad_(False)
        self.prompt_matcher.semantic_token_adapter.requires_grad_(True)
        self.prompt_matcher.gaussian_feature_head.requires_grad_(True)
        self.prompt_matcher.prompt_semantic_adapter.requires_grad_(True)
        self.prompt_matcher.log_temperature.requires_grad_(True)
        if self.prompt_matcher.feature_align is not None:
            self.prompt_matcher.feature_align.requires_grad_(True)
        if self.opt.prompt_unfreeze_tokengs:
            unfreeze_mode = getattr(
                self.opt, "prompt_unfreeze_tokengs_mode", "all"
            )
            for name, parameter in self.named_parameters():
                if not name.startswith("prompt_matcher."):
                    if unfreeze_mode == "all":
                        parameter.requires_grad_(True)
                    elif unfreeze_mode == "decoder":
                        if (
                            name.startswith(
                                "enc_dec_backbone.decoder_blocks."
                            )
                            or name.startswith("activation_head.")
                            or name in ("gs_tokens", "gs_tokens_dynamic")
                        ):
                            parameter.requires_grad_(True)
                    else:
                        raise ValueError(
                            f"Unknown prompt_unfreeze_tokengs_mode: "
                            f"{unfreeze_mode}"
                        )
        self.train(True)

    def prompt_trainable_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = {
            "token_adapter": list(
                self.prompt_matcher.semantic_token_adapter.parameters()
            ),
            "prompt_adapter": list(
                self.prompt_matcher.prompt_semantic_adapter.parameters()
            ),
            "gaussian_feature_head": list(
                self.prompt_matcher.gaussian_feature_head.parameters()
            ),
            "temperature": [self.prompt_matcher.log_temperature],
        }
        if self.prompt_matcher.feature_align is not None:
            groups["feature_align"] = list(
                self.prompt_matcher.feature_align.parameters()
            )
        if self.opt.prompt_unfreeze_tokengs:
            groups["tokengs"] = [
                parameter
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
                and not name.startswith("prompt_matcher.")
            ]
        return groups

    semantic_trainable_groups = prompt_trainable_groups

    @staticmethod
    def compute_instance_group_lambda_eff(
        global_step: int, opt
    ) -> float:
        """Linear warm-up of the instance-group supervision weight."""
        warmup = int(getattr(opt, "instance_group_lambda_warmup_steps", 0))
        if warmup <= 0:
            return 1.0
        return min(1.0, max(0.0, float(global_step)) / warmup)

    @staticmethod
    def compute_teacher_lambda_eff(
        global_step: int, opt
    ) -> float:
        """Teacher weight for the frozen GS-head bootstrap.

        Absolute student mode: 1.0 during the reconstruction bootstrap, then
        a linear decay to 0 (after which the teacher forward is skipped).
        Residual generative mode: linear decay from step 0 (unchanged).
        """
        if bool(getattr(opt, "instance_branch_abs_units", False)):
            bootstrap = int(getattr(opt, "abs_bootstrap_steps", 600))
            decay = int(getattr(opt, "abs_teacher_decay_steps", 400))
            if global_step < bootstrap:
                return 1.0
            if decay <= 0:
                return 0.0
            return max(
                0.0, 1.0 - float(global_step - bootstrap) / decay
            )
        if not bool(getattr(opt, "gen_teacher_distill", False)):
            return 0.0
        decay = int(getattr(opt, "gen_teacher_decay_steps", 1500))
        if decay <= 0:
            return 1.0
        return max(0.0, 1.0 - float(max(0, global_step)) / decay)

    @staticmethod
    def compute_instance_stage_eff(
        global_step: int, opt
    ) -> float:
        """Instance-loss stage weight.

        Absolute student mode: 0 during the reconstruction bootstrap and the
        teacher-decay window, then linear warm-up to 1.  All other recipes
        keep 1.0 (no behavior change).
        """
        if not bool(getattr(opt, "instance_branch_abs_units", False)):
            return 1.0
        bootstrap = int(getattr(opt, "abs_bootstrap_steps", 600))
        decay = int(getattr(opt, "abs_teacher_decay_steps", 400))
        warmup = int(getattr(opt, "abs_instance_warmup_steps", 800))
        start = bootstrap + decay
        if global_step < start:
            return 0.0
        if warmup <= 0:
            return 1.0
        return min(1.0, float(global_step - start) / warmup)

    @staticmethod
    def compute_guarded_instance_effs(
        global_step: int, opt
    ) -> tuple[float, float]:
        """Guarded-joint schedule: (loss_weight_eff, instance_unit_grad_eff).

        ``loss_weight_eff`` ramps 0 -> 1 over the whole first joint window
        (0 .. guarded_instance_full_joint_steps), so the effective instance
        loss weight is lambda_max * loss_weight_eff and instance supervision
        is always present but small early.  ``instance_unit_grad_eff`` is 0
        during the warm-up (student GS detached from instance loss), then
        ramps 0 -> 1 between the warm-up end and the full-joint step.
        """
        warmup = max(
            1, int(getattr(opt, "guarded_instance_warmup_steps", 1000))
        )
        full_joint = max(
            warmup,
            int(
                getattr(
                    opt, "guarded_instance_full_joint_steps", 5680
                )
            ),
        )
        step = max(0, int(global_step))
        weight_eff = min(1.0, step / float(full_joint))
        if step <= warmup:
            unit_eff = 0.0
        elif step >= full_joint:
            unit_eff = 1.0
        else:
            unit_eff = (step - warmup) / float(full_joint - warmup)
        return float(weight_eff), float(unit_eff)

    @staticmethod
    def compute_tsh_effs(
        global_step: int, opt
    ) -> tuple[float, float]:
        """True-Shared schedules: (head_loss_weight_eff, unit_grad_eff).

        head_loss_weight_eff ramps 0 -> 1 over the head warm-up window so
        the new instance head is trained alone (q_abs detached).  The
        unit_grad_eff is 0 during warm-up and ramps to
        ``tsh_unit_gradient_multiplier_max`` across the second window.  The
        two controls are applied independently: the former scales the
        instance loss for the whole head, the latter only re-blends q_abs
        so instance gradients enter the unit formation.
        """
        warmup = max(
            1, int(getattr(opt, "tsh_instance_warmup_steps", 1000))
        )
        ramp_end = max(
            warmup,
            int(getattr(opt, "tsh_instance_ramp_end_steps", 5680)),
        )
        step = max(0, int(global_step))
        head_eff = min(1.0, step / float(warmup))
        if step <= warmup:
            unit_eff = 0.0
        elif step >= ramp_end:
            unit_eff = float(
                getattr(opt, "tsh_unit_gradient_multiplier_max", 1.0)
            )
        else:
            unit_eff = float(
                getattr(opt, "tsh_unit_gradient_multiplier_max", 1.0)
            ) * ((step - warmup) / float(ramp_end - warmup))
        return float(head_eff), float(unit_eff)

    @staticmethod
    def compute_tsh_mbm_u2r_eff(global_step: int, opt) -> float:
        """Effective weight of the SIU3R mask-guided depth smoothness.

        Official SIU3R config: ``weight_depth_smoothness = 0.05``
        (configs/main.yaml / main_multi.yaml line 56); the loss is computed
        in ``src/pipeline.py:249-265`` (and the identical multi-view copy in
        ``src/pipeline_multi.py:254-265``) on every training step.  TokenGS
        adds a from-0 warm-up window (start step + ramp length, optimizer
        steps) so that unreliable early predicted masks never disturb the
        geometry.
        """
        mode = str(getattr(opt, "tsh_mbm_mode", "off"))
        if mode not in ("u2r", "both"):
            return 0.0
        weight = float(getattr(opt, "tsh_mbm_u2r_weight", 0.0))
        if weight <= 0.0:
            return 0.0
        start = max(0, int(getattr(opt, "tsh_mbm_u2r_warmup_start_step", 0)))
        ramp = max(1, int(getattr(opt, "tsh_mbm_u2r_warmup_steps", 0)))
        step = max(0, int(global_step))
        if step < start:
            return 0.0
        return float(weight * min(1.0, (step - start) / float(ramp)))

    @staticmethod
    def compute_tsh_per_gs_gate_eff(global_step: int, opt) -> float:
        """Linear 0 -> 1 ramp of the per-GS residual gate."""
        ramp = max(
            1, int(getattr(opt, "tsh_per_gs_ramp_steps", 125))
        )
        return float(
            min(1.0, max(0, int(global_step)) / float(ramp))
        )

    @staticmethod
    def compute_tsh_query_memory_refine_eff(global_step: int, opt) -> float:
        """Linear 0 -> 1 gate for the assignment-conditioned refiner."""
        if not bool(getattr(opt, "tsh_query_memory_refine", False)):
            return 0.0
        ramp = max(1, int(getattr(opt, "tsh_query_memory_refine_gate_steps", 50)))
        return float(min(1.0, max(0, int(global_step)) / float(ramp)))

    @staticmethod
    def compute_ga_idu_gate_eff(global_step: int, opt) -> float:
        """GA-IDU-1 external 0.2, ..., 1.0 schedule (step 0 is audit)."""
        if str(getattr(opt, "ga_idu_mode", "off")) != "1":
            return 0.0
        return float(min(1.0, max(0, int(global_step) + 1) / float(max(1, getattr(opt, "ga_idu_gate_steps", 5)))))

    @staticmethod
    def compute_ta_riu_gate_eff(global_step: int, opt) -> float:
        """TA-RIU gate: step 0 is identity, then ramp to one by step 25."""
        if not bool(getattr(opt, "ta_riu_enabled", False)):
            return 0.0
        ramp = max(1, int(getattr(opt, "ta_riu_gate_steps", 25)))
        return float(min(1.0, max(0, int(global_step) + 1) / float(ramp)))

    @staticmethod
    def compute_ta_riu_v2_gate_eff(global_step: int, opt) -> float:
        """TA-RIU-v2 evidence gate; the pre-step-0 audit is exact identity."""
        if not bool(getattr(opt, "ta_riu_v2_enabled", False)):
            return 0.0
        ramp = max(1, int(getattr(opt, "ta_riu_v2_gate_steps", 25)))
        return float(min(1.0, max(0, int(global_step) + 1) / float(ramp)))

    @staticmethod
    def compute_ta_riu_v3_gate_eff(global_step: int, opt) -> float:
        """TA-RIU-v3 gate; the pre-step-0 diagnostic is exact identity."""
        if not bool(getattr(opt, "ta_riu_v3_enabled", False)):
            return 0.0
        ramp = max(1, int(getattr(opt, "ta_riu_v3_gate_ramp_steps", 25)))
        return float(min(1.0, max(0, int(global_step)) / float(ramp)))

    def _instance_group_head_forward(
        self,
        gs_token_hidden: torch.Tensor,
        gaussians: torch.Tensor,
        data: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Group head with optional per-token 3D anchor-position conditioning.

        The anchor position is the mean of the 64 Gaussian centers decoded by
        each token -- a stable spatial prior (InstOk3D-style anchors) that
        gives the global head cross-scene consistency, which is exactly what
        the feed-forward grouping is missing (per-scene TTT proves the token
        content alone already contains the instance information).
        """
        if getattr(self.opt, "instance_group_conditioned_gaussians", False):
            cache = getattr(self, "_instance_conditioning_cache", None)
            if not cache:
                raise RuntimeError(
                    "GC2 instance assignment is unavailable: Gaussian "
                    "conditioning must run before the instance branch"
                )
            probabilities = cache["probabilities"]
            logits = cache["logits"]
            valid_shapes = {
                tuple(gs_token_hidden.shape[:2]),
                tuple(gaussians.shape[:2]),
            }
            if tuple(probabilities.shape[:2]) not in valid_shapes:
                raise RuntimeError(
                    "GC2/GC3 assignment shape mismatch: "
                    f"assignment={tuple(probabilities.shape)}, "
                    f"tokens={tuple(gs_token_hidden.shape)}, "
                    f"gaussians={tuple(gaussians.shape)}"
                )
            return probabilities, logits, None
        if getattr(self.opt, "instance_group_per_gaussian", False):
            # Per-Gaussian head returns [B,N,G+1] directly.
            if getattr(self.opt, "instance_group_residual_head", False):
                # The residual head reuses the v4 anchor-position encoder
                # for its warm-started token-level base, so the base
                # reproduces wide7l's exact input (token hidden + pos feat).
                base_input = gs_token_hidden
                if getattr(self.opt, "instance_group_use_anchor_pos", False):
                    batch_size, token_count, _ = gs_token_hidden.shape
                    means = gaussians[..., :3].view(
                        batch_size,
                        token_count,
                        self.num_gaussians_per_token,
                        3,
                    )
                    anchor_pos = means.mean(dim=2)  # [B,T,3]
                    pos_feat = self.anchor_pos_encoder(anchor_pos.float())
                    base_input = torch.cat([gs_token_hidden, pos_feat], dim=-1)
                if getattr(self.opt, "instance_group_dense_decoder", False):
                    dense_features = self._gather_dense_features()
                    if dense_features is not None and data is not None:
                        return self.instance_group_head(
                            gs_token_hidden,
                            gaussians,
                            base_input=base_input,
                            dense_features=dense_features,
                            source_c2w=data.get("cam_to_world_input"),
                            source_intrinsics=data.get("intrinsics_input"),
                            image_hw=tuple(self.opt.img_size),
                            num_views=int(self.opt.num_input_views),
                        )
                probabilities, logits = self.instance_group_head(
                    gs_token_hidden, gaussians, base_input=base_input
                )
                return probabilities, logits, None
            probabilities, logits = self.instance_group_head(
                gs_token_hidden, gaussians
            )
            return probabilities, logits, None
        head_input = gs_token_hidden
        if getattr(self.opt, "instance_group_use_anchor_pos", False):
            batch_size, token_count, _ = gs_token_hidden.shape
            means = gaussians[..., :3].view(
                batch_size,
                token_count,
                self.num_gaussians_per_token,
                3,
            )
            anchor_pos = means.mean(dim=2)  # [B,T,3]
            pos_feat = self.anchor_pos_encoder(anchor_pos.float())  # [B,T,64]
            head_input = torch.cat([gs_token_hidden, pos_feat], dim=-1)
        # Both the plain InstanceGroupHead and the InstOk3D-style
        # InstanceGroupDecoder (group tokens) return (probabilities, logits),
        # so unpack here before forwarding the tuple upstream.
        probabilities, logits = self.instance_group_head(head_input)
        return probabilities, logits, None

    def _gather_dense_features(self) -> torch.Tensor | None:
        """Concatenate the cached frozen-encoder patch features (multi-scale
        optional) into a single [B, V*P, C] tensor, or None if the encoder
        has not run in this forward (e.g. pure TTT path)."""
        cache = getattr(self, "_dense_feature_cache", None)
        if not cache:
            return None
        scales = []
        if "last" in cache and cache["last"] is not None:
            scales.append(cache["last"])
        if (
            getattr(self.opt, "instance_group_dense_multiscale", True)
            and "mid" in cache
            and cache["mid"] is not None
        ):
            scales.append(cache["mid"])
        if not scales:
            return None
        if len(scales) == 1:
            return scales[0]
        return torch.cat(scales, dim=-1)

    @staticmethod
    def _instance_boundary_mask(
        labels: torch.Tensor,
        dilate: int = 1,
        include_background: bool = True,
    ) -> torch.Tensor:
        """GT instance ids [B,V,H,W] -> boundary mask [B,V,H,W] bool.

        A pixel is on a boundary when any 4-neighbour has a different
        instance id (background counts as a neighbour when
        ``include_background`` is True). Optionally dilated so the RGB
        gradient band around edges is wider.
        """
        left = labels[..., 1:, :] != labels[..., :-1, :]  # [B,V,H-1,W]
        up = labels[..., :, 1:] != labels[..., :, :-1]  # [B,V,H,W-1]
        boundary = F.pad(left, (0, 0, 0, 1)) | F.pad(up, (0, 1, 0, 0))
        if not include_background:
            boundary = boundary & (labels > 0)
        if dilate > 0:
            kernel = 2 * int(dilate) + 1
            boundary = (
                F.max_pool2d(
                    boundary.float(),
                    kernel_size=kernel,
                    stride=1,
                    padding=int(dilate),
                )
                > 0
            )
        return boundary

    def compute_semantic_lifting_loss(
        self,
        data: dict,
        gaussians: torch.Tensor,
        gs_token_hidden: torch.Tensor,
        model_input,
        bg_color: torch.Tensor,
    ) -> torch.Tensor:
        """Fit per-Gaussian semantic features to LSeg features (target+source).

        C3G-style lifting + the learned fusion head (source_projected
        branch from the tokengs_c3g experiment): project context-view LSeg
        features onto the Gaussian centers, refine with token/geometry via
        ``GaussianSemanticHeadV2``, render to target/source views, and
        supervise with cosine + smooth-L1 against the LSeg features of the
        target and source images.
        """
        opt = self.opt
        input_rgb = data["images_input"]  # [B,Vsrc,3,H,W]
        output_rgb = data["images_output"]  # [B,Vtgt,3,H,W]
        batch, views_src, _, height, width = input_rgb.shape
        _, views_tgt, _, _, _ = output_rgb.shape

        feat_in = self.lseg_teacher.extract(
            input_rgb.reshape(batch * views_src, 3, height, width)
        )
        feat_out = self.lseg_teacher.extract(
            output_rgb.reshape(batch * views_tgt, 3, height, width)
        )
        lseg_in = feat_in.reshape(
            batch, views_src, feat_in.shape[1], feat_in.shape[2], feat_in.shape[3]
        )
        lseg_out = feat_out.reshape(
            batch, views_tgt, feat_out.shape[1], feat_out.shape[2], feat_out.shape[3]
        )

        source_depth = None
        if getattr(opt, "semantic_use_depth_filter", False):
            with torch.no_grad():
                source_depth = self.gs.render(
                    gaussians.detach(),
                    data["cam_view_input"],
                    bg_color=bg_color,
                    intrinsics=data["intrinsics_input"],
                )["depths_pred"]

        fused, has_source, confidence = self.semantic_projector(
            xyz_world=gaussians[..., 0:3],
            source_features=lseg_in,
            source_c2w=data["cam_to_world_input"],
            source_intrinsics=data["intrinsics_input"],
            source_depth=source_depth,
        )
        gaussian_feat = self.semantic_lifting_head(
            decoded_tokens=grad_scale(
                gs_token_hidden,
                float(getattr(self.opt, "grad_scale_sem", 0.3)),
            ),
            gaussians=gaussians,
            projected_features=fused,
            has_source=has_source,
            confidence=confidence,
        )
        target_render = self.gs.render_feature_channels(
            gaussians,
            gaussian_feat,
            model_input.decoder.cam_view,
            intrinsics=model_input.decoder.intrinsics,
        )
        source_render = self.gs.render_feature_channels(
            gaussians,
            gaussian_feat,
            data["cam_view_input"],
            intrinsics=data["intrinsics_input"],
        )
        loss_target = semantic_feature_loss(
            target_render["images_pred"],
            lseg_out,
            target_render["alphas_pred"],
            lambda_cosine=float(getattr(opt, "lambda_semantic_cosine", 1.0)),
            lambda_l1=float(getattr(opt, "lambda_semantic_l1", 0.05)),
            alpha_threshold=float(getattr(opt, "semantic_alpha_threshold", 0.05)),
        )
        loss_source = semantic_feature_loss(
            source_render["images_pred"],
            lseg_in,
            source_render["alphas_pred"],
            lambda_cosine=float(getattr(opt, "lambda_semantic_cosine", 1.0)),
            lambda_l1=float(getattr(opt, "lambda_semantic_l1", 0.05)),
            alpha_threshold=float(getattr(opt, "semantic_alpha_threshold", 0.05)),
        )
        lambda_source = float(getattr(opt, "lambda_semantic_source", 0.5))
        loss_feature = loss_target + lambda_source * loss_source
        loss = (
            float(getattr(opt, "lambda_semantic_feature", 1.0))
            * loss_feature
        )
        if self.semantic_classifier is not None and float(
            getattr(opt, "lambda_semantic_ce", 0.0)
        ) > 0:
            # Pseudo-label CE (tokengs_c3g recipe): decode the LSeg teacher
            # features of the source/target images into 8-class pseudo labels
            # with confidence, then classify the rendered features with the
            # learned 1x1 classifier. This is what makes the lifting head
            # learn a meaningful correction (the cosine/L1 loss alone leaves
            # the residual near zero).
            from tokengs.models.semantic_adapter_v2 import C3G8_CLASS_NAMES

            target_logits = self.classify_rendered_semantic_features(
                target_render["images_pred"]
            )
            source_logits = self.classify_rendered_semantic_features(
                source_render["images_pred"]
            )
            target_ce, _, _ = self.semantic_pseudo_ce_loss(
                logits=target_logits,
                features=lseg_out,
                alpha=target_render["alphas_pred"],
                labelset=C3G8_CLASS_NAMES,
            )
            source_ce, _, _ = self.semantic_pseudo_ce_loss(
                logits=source_logits,
                features=lseg_in,
                alpha=source_render["alphas_pred"],
                labelset=C3G8_CLASS_NAMES,
            )
            loss = loss + float(
                getattr(opt, "lambda_semantic_ce", 0.0)
            ) * (target_ce + lambda_source * source_ce)
        return loss

    def classify_rendered_semantic_features(
        self, rendered_features: torch.Tensor
    ) -> torch.Tensor:
        """Rendered [B,V,512,H,W] -> classifier logits [B,V,8,H,W]."""
        if self.semantic_classifier is None:
            raise RuntimeError("semantic_classifier is not initialized")
        batch, views, channels, height, width = rendered_features.shape
        normalized = F.normalize(rendered_features.float(), dim=2, eps=1e-6)
        logits = self.semantic_classifier(
            normalized.reshape(batch * views, channels, height, width)
        )
        return logits.reshape(
            batch, views, logits.shape[1], height, width
        )

    def semantic_pseudo_ce_loss(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        alpha: torch.Tensor,
        labelset: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Confidence-weighted CE against LSeg text-decoded pseudo labels."""
        opt = self.opt
        batch, views, num_classes, height, width = logits.shape
        pseudo_labels, pseudo_confidence = lseg_features_to_pseudo_labels(
            self.lseg_teacher, features, labelset
        )
        pseudo_labels = F.interpolate(
            pseudo_labels.reshape(batch * views, 1, *pseudo_labels.shape[-2:]),
            size=(height, width),
            mode="nearest",
        ).reshape(batch, views, height, width).long()
        pseudo_confidence = F.interpolate(
            pseudo_confidence.reshape(
                batch * views, 1, *pseudo_confidence.shape[-2:]
            ),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, views, height, width)
        alpha_valid = (
            alpha.detach()[:, :, 0]
            > float(getattr(opt, "semantic_alpha_threshold", 0.05))
        )
        teacher_valid = (
            pseudo_confidence
            > float(getattr(opt, "semantic_pseudo_conf_threshold", 0.40))
        )
        valid = alpha_valid & teacher_valid
        ce_map = F.cross_entropy(
            logits.reshape(batch * views, num_classes, height, width),
            pseudo_labels.reshape(batch * views, height, width),
            reduction="none",
        ).reshape(batch, views, height, width)
        valid_float = valid.float()
        confidence_weight = pseudo_confidence.detach() * valid_float
        loss = (ce_map * confidence_weight).sum() / confidence_weight.sum().clamp_min(
            1.0
        )
        valid_ratio = valid_float.mean().detach()
        prediction = logits.argmax(dim=2)
        accuracy = (
            (prediction == pseudo_labels).float() * valid_float
        ).sum() / valid_float.sum().clamp_min(1.0)
        return loss, valid_ratio, accuracy.detach()

    def build_semantic_gaussian_features(
        self,
        gaussians: torch.Tensor,
        decoded_tokens: torch.Tensor,
        source_lseg_features: torch.Tensor,
        source_c2w: torch.Tensor,
        source_intrinsics: torch.Tensor,
        source_depth: torch.Tensor | None = None,
        materialize_features: bool = False,
    ) -> dict:
        """Project source LSeg features onto Gaussians and run the branch head.

        Mirrors the tokengs_c3g ``build_semantic_gaussian_features`` dispatch:
        ``token_decoder_lowrank`` uses the token-compressed field, everything
        else uses the per-Gaussian residual head.
        """
        projected_features, has_source, confidence = (
            self.semantic_projector(
                xyz_world=gaussians[..., 0:3].detach(),
                source_features=source_lseg_features.detach(),
                source_c2w=source_c2w,
                source_intrinsics=source_intrinsics,
                source_depth=source_depth,
            )
        )

        branch_version = str(
            getattr(self.opt, "semantic_branch_version", "source_projected")
        )
        tokens_for_semantic = (
            decoded_tokens.detach()
            if getattr(self.opt, "semantic_detach_tokens", True)
            else decoded_tokens
        )
        gaussians_for_semantic = (
            gaussians.detach()
            if getattr(self.opt, "semantic_detach_geometry", True)
            else gaussians
        )

        if branch_version == "token_decoder_lowrank":
            token_semantic = self.semantic_head(
                decoded_tokens=tokens_for_semantic,
                gaussians=gaussians_for_semantic,
                source_features=source_lseg_features.detach(),
                projected_features=projected_features,
                has_source=has_source,
                confidence=confidence,
                materialize_features=materialize_features,
            )
            token_semantic.update(
                {
                    "semantic_representation": "token_lowrank",
                    "semantic_projected_features": projected_features,
                    "semantic_projection_valid": has_source,
                    "semantic_projection_confidence": confidence,
                }
            )
            return token_semantic

        gaussian_features = self.semantic_lifting_head(
            decoded_tokens=tokens_for_semantic,
            gaussians=gaussians_for_semantic,
            projected_features=projected_features,
            has_source=has_source,
            confidence=confidence,
        )
        return {
            "semantic_representation": "per_gaussian",
            "gaussian_semantic_features": gaussian_features,
            "semantic_projected_features": projected_features,
            "semantic_projection_valid": has_source,
            "semantic_projection_confidence": confidence,
        }

    def render_semantic_3d_features(
        self,
        gaussians: torch.Tensor,
        semantic_3d: dict,
        cam_view: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> dict:
        """Render the semantic field to a view (streaming for token_lowrank)."""
        representation = semantic_3d.get("semantic_representation", "per_gaussian")

        if representation == "token_lowrank":
            if bool(
                getattr(self.opt, "semantic_stream_compressed_features", True)
            ):
                return self.gs.render_token_features(
                    gaussians=gaussians,
                    token_features=semantic_3d["semantic_token_features"],
                    local_codes=semantic_3d["semantic_local_codes"],
                    local_basis=semantic_3d["semantic_local_basis"],
                    cam_view=cam_view,
                    intrinsics=intrinsics,
                    local_residual_scale=float(
                        getattr(self.opt, "semantic_local_residual_scale", 0.1)
                    ),
                    feature_chunk_size=int(
                        getattr(self.opt, "semantic_render_chunk", 32)
                    ),
                    render_scale=float(
                        getattr(self.opt, "semantic_render_scale", 0.5)
                    ),
                    detach_geometry=bool(
                        getattr(self.opt, "semantic_detach_geometry", True)
                    ),
                )
            gaussian_features = semantic_3d["gaussian_semantic_features"]
            if gaussian_features is None:
                gaussian_features = self.semantic_head.materialize_gaussian_features(
                    token_features=semantic_3d["semantic_token_features"],
                    local_codes=semantic_3d["semantic_local_codes"],
                )
            return self.gs.render_feature_channels(
                gaussians, gaussian_features, cam_view, intrinsics=intrinsics
            )

        gaussian_features = semantic_3d["gaussian_semantic_features"]
        return self.gs.render_feature_channels(
            gaussians, gaussian_features, cam_view, intrinsics=intrinsics
        )

    def semantic_feature_loss_v3(
        self,
        prediction: torch.Tensor,
        teacher: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Cosine + smooth-L1 feature loss with optional alpha masking."""
        opt = self.opt
        if teacher.shape[-2:] != prediction.shape[-2:]:
            B, V, C, H, W = teacher.shape
            teacher = F.interpolate(
                teacher.reshape(B * V, C, H, W),
                size=prediction.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape(B, V, C, prediction.shape[-2], prediction.shape[-1])

        pred_normalized = F.normalize(prediction.float(), dim=2, eps=1e-6)
        teacher_normalized = F.normalize(teacher.float(), dim=2, eps=1e-6)

        if bool(getattr(opt, "semantic_feature_use_alpha_mask", True)):
            valid = (
                alpha.detach() > float(getattr(opt, "semantic_alpha_threshold", 0.05))
            ).float()
        else:
            valid = torch.ones_like(alpha, dtype=torch.float32)

        denominator = valid.sum().clamp_min(1.0)
        cosine_map = 1.0 - (pred_normalized * teacher_normalized).sum(
            dim=2, keepdim=True
        )
        loss_cosine = (cosine_map * valid).sum() / denominator

        l1_map = F.smooth_l1_loss(
            prediction.float(), teacher.float(), reduction="none"
        ).mean(dim=2, keepdim=True)
        loss_l1 = (l1_map * valid).sum() / denominator

        loss = (
            float(getattr(opt, "lambda_semantic_cosine", 1.0)) * loss_cosine
            + float(getattr(opt, "lambda_semantic_l1", 0.0)) * loss_l1
        )
        return loss, {
            "cosine": loss_cosine.detach(),
            "l1": loss_l1.detach(),
        }

    def compute_semantic_v3_loss(
        self,
        data: dict,
        gaussians: torch.Tensor,
        gs_token_hidden: torch.Tensor,
        model_input,
        bg_color: torch.Tensor,
    ) -> torch.Tensor:
        """V3 token_decoder_lowrank training loss (target + source views)."""
        opt = self.opt
        input_rgb = data["images_input"]
        output_rgb = data["images_output"]
        batch, views_src, _, height, width = input_rgb.shape
        _, views_tgt, _, _, _ = output_rgb.shape

        feat_in = self.lseg_teacher.extract(
            input_rgb.reshape(batch * views_src, 3, height, width)
        )
        feat_out = self.lseg_teacher.extract(
            output_rgb.reshape(batch * views_tgt, 3, height, width)
        )
        lseg_in = feat_in.reshape(batch, views_src, *feat_in.shape[1:])
        lseg_out = feat_out.reshape(batch, views_tgt, *feat_out.shape[1:])

        source_depth = None
        if getattr(opt, "semantic_use_depth_filter", False):
            with torch.no_grad():
                source_depth = self.gs.render(
                    gaussians.detach(),
                    data["cam_view_input"],
                    bg_color=bg_color,
                    intrinsics=data["intrinsics_input"],
                )["depths_pred"]

        semantic_3d = self.build_semantic_gaussian_features(
            gaussians=gaussians,
            decoded_tokens=grad_scale(
                gs_token_hidden,
                float(getattr(self.opt, "grad_scale_sem", 0.3)),
            ),
            source_lseg_features=lseg_in,
            source_c2w=data["cam_to_world_input"],
            source_intrinsics=data["intrinsics_input"],
            source_depth=source_depth,
            materialize_features=not bool(
                getattr(opt, "semantic_stream_compressed_features", True)
            ),
        )

        target_render = self.render_semantic_3d_features(
            gaussians,
            semantic_3d,
            model_input.decoder.cam_view,
            model_input.decoder.intrinsics,
        )
        source_render = self.render_semantic_3d_features(
            gaussians,
            semantic_3d,
            data["cam_view_input"],
            data["intrinsics_input"],
        )

        target_loss, _ = self.semantic_feature_loss_v3(
            prediction=target_render["semantic_features_pred"],
            teacher=lseg_out,
            alpha=target_render["semantic_alphas_pred"],
        )
        source_loss, _ = self.semantic_feature_loss_v3(
            prediction=source_render["semantic_features_pred"],
            teacher=lseg_in,
            alpha=source_render["semantic_alphas_pred"],
        )

        loss = float(getattr(opt, "lambda_semantic_feature", 1.0)) * (
            target_loss + views_src * source_loss
        ) / (views_src + 1)
        return loss

    def forward_instance_group_branch(
        self,
        data: dict,
        gaussians: torch.Tensor,
        gs_token_hidden: torch.Tensor,
        gaussian_features: torch.Tensor,
        semantic_prompts: torch.Tensor,
        model_input,
    ) -> dict:
        """Render per-group instance probabilities and compute group losses.

        The head assigns the 1024 decoder tokens to ``num_groups`` instances
        through a per-token softmax; probabilities are expanded to the 64
        Gaussians of each token and alpha-composited on the output views. A
        void channel absorbs pixels with no GT instance. Returns rendered
        probabilities, group-level semantic embeddings for retrieval, and
        (when training with instance labels) the Hungarian-matched loss.
        """
        if (
            getattr(self.opt, "instance_branch_independent", False)
            or getattr(self.opt, "instance_branch_token_units", False)
        ):
            branch = getattr(self, "instance_branch", None)
            if branch is None:
                raise RuntimeError(
                    "independent/token-units branch requires self.instance_branch"
                )
            return branch(
                token_hidden=grad_scale(
                    gs_token_hidden,
                    float(getattr(self.opt, "grad_scale_ins", 0.1)),
                ),
                frozen_gaussians=gaussians,
                data=data,
                model_input=model_input,
                opt=self.opt,
                lambda_eff=float(
                    getattr(self, "instance_group_lambda_eff", 1.0)
                ),
                training=self.training,
                dense_features=self._gather_dense_features(),
            )
        if self.instance_group_head is None:
            return {}
        num_groups = self.instance_group_head.num_groups
        lambda_eff = float(getattr(self, "instance_group_lambda_eff", 1.0))
        group_probs, group_logits, dense_patch_features = (
            self._instance_group_head_forward(
                gs_token_hidden, gaussians, data=data
            )
        )
        if group_probs.shape[1] == gaussians.shape[1]:
            gaussian_group_probs = group_probs  # already [B,N,G+1]
        else:
            gaussian_group_probs = group_probs.repeat_interleave(
                self.num_gaussians_per_token, dim=1
            )  # [B,N,G+1] (last channel is void)
        render = self.gs.render_feature_channels(
            gaussians,
            gaussian_group_probs,
            model_input.decoder.cam_view,
            intrinsics=model_input.decoder.intrinsics,
            opacity_scale=float(
                getattr(self.opt, "instance_group_render_scale", 1.0)
            ),
        )
        rendered_groups = render["images_pred"]  # [B,V,G,H,W]
        rendered_alpha = render["alphas_pred"]  # [B,V,1,H,W]
        rendered_channels = rendered_groups / (rendered_alpha + 1e-5)
        # ``rendered_channels`` is already a per-pixel probability vector
        # (alpha-weighted average of per-Gaussian softmax probabilities,
        # summing to ~1 over the G+1 channels). Applying softmax again would
        # treat probabilities as logits and flatten a sharp distribution
        # (max ~0.9 -> ~0.01), destroying the rendered confidence and the
        # training signal. Normalize with L1 instead.
        rendered_probs = rendered_channels.float() / rendered_channels.float().sum(
            dim=2, keepdim=True
        ).clamp_min(1e-6)
        rendered_probability = rendered_probs.permute(0, 2, 1, 3, 4).unsqueeze(
            3
        )  # [B,G+1,V,1,H,W]

        # Group-level semantic embeddings for instance retrieval.
        instance_group_probs = gaussian_group_probs[..., :num_groups]
        group_embeddings = F.normalize(
            torch.einsum(
                "bng,bnd->bgd",
                instance_group_probs.detach().float(),
                gaussian_features.float(),
            ),
            dim=-1,
        )
        group_semantic_logits = (
            torch.einsum("bqd,bgd->bqg", semantic_prompts, group_embeddings)
            * self.prompt_matcher.temperature
        )

        outputs = {
            "instance_group_probabilities": group_probs,
            "instance_group_logits": group_logits,
            "gaussian_group_probabilities": gaussian_group_probs,
            "rendered_instance_group_probability": rendered_probability,
            "rendered_instance_group_alpha": rendered_alpha,
            "group_embeddings": group_embeddings,
            "group_semantic_logits": group_semantic_logits,
            "loss_instance_group": torch.zeros(
                (), device=rendered_probability.device
            ),
        }
        if getattr(self.opt, "instance_group_conditioned_gaussians", False):
            cache = self._instance_conditioning_cache
            with torch.no_grad():
                token_distribution = cache.get(
                    "token_probabilities", group_probs
                ).float()
                entropy = -(
                    token_distribution
                    * token_distribution.clamp_min(1e-8).log()
                ).sum(dim=-1).mean()
                nonvoid = token_distribution[..., :num_groups]
                hard_groups = nonvoid.argmax(dim=-1)
                active_counts = torch.tensor(
                    [
                        torch.unique(scene_groups).numel()
                        for scene_groups in hard_groups
                    ],
                    device=hard_groups.device,
                    dtype=torch.float32,
                )
                base_hidden = cache["base_hidden"].float()
                conditioned_hidden = cache["conditioned_hidden"].float()
                residual_ratio = (
                    (conditioned_hidden - base_hidden).norm(dim=-1).mean()
                    / base_hidden.norm(dim=-1).mean().clamp_min(1e-6)
                )
                proposal_gaussians = cache["proposal_gaussians"].float()
                final_gaussians = cache["final_gaussians"].float()
                gaussian_delta = final_gaussians - proposal_gaussians
                gaussian_distribution = group_probs.float()
                outputs.update(
                    {
                        "gc2_assignment_entropy": entropy,
                        "gc2_assignment_max_probability": token_distribution.max(
                            dim=-1
                        ).values.mean(),
                        "gc2_assignment_void_share": token_distribution[
                            ..., num_groups
                        ].mean(),
                        "gc2_active_group_count": active_counts.mean(),
                        "gc2_conditioning_residual_ratio": residual_ratio,
                        "gc2_gaussian_abs_delta": gaussian_delta.abs().mean(),
                        "gc2_xyz_abs_delta": gaussian_delta[
                            ..., :3
                        ].abs().mean(),
                        "gc3_gaussian_assignment_entropy": (
                            -(
                                gaussian_distribution
                                * gaussian_distribution.clamp_min(1e-8).log()
                            )
                            .sum(dim=-1)
                            .mean()
                        ),
                        "gc3_gaussian_assignment_void_share": gaussian_distribution[
                            ..., num_groups
                        ].mean(),
                    }
                )
                if cache.get("image_anchor_valid_share") is not None:
                    outputs["gc4_image_anchor_valid_share"] = cache[
                        "image_anchor_valid_share"
                    ]
                if cache.get("image_anchor_gate") is not None:
                    outputs["gc4_image_anchor_gate"] = cache[
                        "image_anchor_gate"
                    ]
        # Dense image-evidence decoder gets direct 2D supervision on the
        # source-view patch features (InfoNCE over instance prototypes), so
        # it learns pixel-level instance boundaries without depending on the
        # sparse Gaussian projection path for its training signal.
        lambda_dense_aux = float(
            getattr(self.opt, "lambda_instance_dense_aux", 0.0)
        )
        if (
            self.training
            and lambda_dense_aux > 0
            and dense_patch_features is not None
            and "instance_label_input" in data
        ):
            labels = data["instance_label_input"].long()
            batch_size, views, feat_dim, feat_h, feat_w = (
                dense_patch_features.shape
            )
            label_h, label_w = labels.shape[-2:]
            feats_img = F.interpolate(
                dense_patch_features.reshape(
                    batch_size * views, feat_dim, feat_h, feat_w
                ),
                size=(label_h, label_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch_size, views, feat_dim, label_h, label_w)
            valid = torch.ones_like(labels, dtype=torch.bool)
            dense_aux = instance_contrastive_loss(
                feats_img,
                labels,
                valid,
                temperature=0.07,
                min_pixels=int(
                    getattr(
                        self.opt, "instance_group_min_instance_pixels", 32
                    )
                ),
            )
            outputs["loss_instance_group"] = (
                outputs["loss_instance_group"]
                + lambda_dense_aux * lambda_eff * dense_aux
            )
            with torch.no_grad():
                outputs["loss_instance_dense_aux"] = dense_aux.detach()
        if self.training and "instance_label_output" in data:
            use_adaptive = bool(
                getattr(self.opt, "instance_group_adaptive_count", False)
            )
            supervision_probability = rendered_probability
            supervision_labels = data["instance_label_output"].long()
            if bool(
                getattr(
                    self.opt,
                    "instance_group_supervise_input_views",
                    False,
                )
            ):
                required = (
                    "cam_view_input",
                    "intrinsics_input",
                    "instance_label_input",
                )
                missing = [key for key in required if key not in data]
                if missing:
                    raise KeyError(
                        "Input-view instance supervision requires data keys: "
                        + ", ".join(missing)
                    )
                source_render = self.gs.render_feature_channels(
                    gaussians,
                    gaussian_group_probs,
                    data["cam_view_input"],
                    intrinsics=data["intrinsics_input"],
                    opacity_scale=float(
                        getattr(self.opt, "instance_group_render_scale", 1.0)
                    ),
                )
                source_channels = source_render["images_pred"] / (
                    source_render["alphas_pred"] + 1e-5
                )
                source_probs = source_channels.float() / source_channels.float().sum(
                    dim=2, keepdim=True
                ).clamp_min(1e-6)
                source_probability = source_probs.permute(
                    0, 2, 1, 3, 4
                ).unsqueeze(3)
                supervision_probability = torch.cat(
                    [source_probability, rendered_probability], dim=2
                )
                supervision_labels = torch.cat(
                    [
                        data["instance_label_input"].long(),
                        data["instance_label_output"].long(),
                    ],
                    dim=1,
                )
                outputs["instance_group_supervised_view_count"] = torch.tensor(
                    supervision_labels.shape[1],
                    device=rendered_probability.device,
                    dtype=torch.float32,
                )
            loss, stats = hungarian_instance_group_loss(
                supervision_probability,
                supervision_labels,
                num_groups=num_groups,
                min_instance_pixels=int(
                    getattr(self.opt, "instance_group_min_instance_pixels", 64)
                ),
                dice_weight=float(
                    getattr(self.opt, "lambda_instance_group_dice", 1.0)
                ),
                mask_weight=float(
                    getattr(self.opt, "lambda_instance_group_mask", 1.0)
                ),
                void_weight=float(
                    getattr(self.opt, "lambda_instance_group_void", 0.1)
                ),
                unmatched_weight=float(
                    getattr(self.opt, "lambda_instance_group_unmatched", 0.1)
                ),
                lambda_eff=lambda_eff,
                area_alpha=float(
                    getattr(self.opt, "instance_group_area_alpha", 0.0)
                ),
                match_area_norm=bool(
                    getattr(self.opt, "instance_group_match_area_norm", False)
                ),
                ce_weight=float(
                    getattr(self.opt, "lambda_instance_group_ce", 0.0)
                ),
                match_topk=int(
                    getattr(self.opt, "instance_group_match_topk", 1)
                ),
                secondary_pair_weight=float(
                    getattr(
                        self.opt,
                        "instance_group_secondary_pair_weight",
                        0.3,
                    )
                ),
                usage_entropy_weight=float(
                    getattr(self.opt, "instance_group_usage_entropy", 0.0)
                ),
                use_adaptive_groups=use_adaptive,
                scene_level_matching=bool(
                    getattr(
                        self.opt,
                        "instance_group_scene_level_matching",
                        False,
                    )
                ),
            )
            outputs["loss_instance_group"] = loss
            outputs.update(stats)
            lambda_3d = float(
                getattr(self.opt, "lambda_instance_group_3d", 0.0)
            )
            if lambda_3d > 0.0 and "instance_label_input" in data:
                cam_views = torch.cat(
                    [data["cam_view_input"], data["cam_view"]], dim=1
                )
                intrinsics_all = torch.cat(
                    [data["intrinsics_input"], data["intrinsics"]], dim=1
                )
                labels_all = torch.cat(
                    [
                        data["instance_label_input"],
                        data["instance_label_output"],
                    ],
                    dim=1,
                )
                loss_3d, stats_3d = instance_group_3d_loss(
                    gaussian_group_probs,
                    gaussians,
                    cam_views,
                    intrinsics_all,
                    labels_all.long(),
                    num_groups=num_groups,
                    image_size=tuple(self.opt.img_size),
                    min_instance_gs=int(
                        getattr(self.opt, "instance_group_3d_min_gs", 16)
                    ),
                    ce_weight=float(
                        getattr(self.opt, "lambda_instance_group_3d_ce", 1.0)
                    ),
                    dice_weight=float(
                        getattr(self.opt, "lambda_instance_group_dice", 1.0)
                    ),
                    mask_weight=float(
                        getattr(self.opt, "lambda_instance_group_mask", 1.0)
                    ),
                    void_weight=float(
                        getattr(self.opt, "lambda_instance_group_void", 0.1)
                    ),
                    unmatched_weight=float(
                        getattr(
                            self.opt,
                            "lambda_instance_group_unmatched",
                            0.1,
                        )
                    ),
                    match_topk=int(
                        getattr(self.opt, "instance_group_3d_match_topk", 1)
                    ),
                    secondary_pair_weight=float(
                        getattr(
                            self.opt,
                            "instance_group_secondary_pair_weight",
                            0.3,
                        )
                    ),
                    use_adaptive_groups=use_adaptive,
                )
                outputs["loss_instance_group"] = (
                    outputs["loss_instance_group"]
                    + lambda_3d * lambda_eff * loss_3d
                )
                outputs.update(stats_3d)
        count_head = getattr(self, "instance_count_head", None)
        if count_head is not None:
            batch_size, token_count, _ = gs_token_hidden.shape
            pos_feat = None
            if getattr(self.opt, "instance_group_use_anchor_pos", False):
                means = gaussians[..., :3].view(
                    batch_size,
                    token_count,
                    self.num_gaussians_per_token,
                    3,
                )
                anchor_pos = means.mean(dim=2)  # [B,T,3]
                pos_feat = self.anchor_pos_encoder(anchor_pos.float())
            pred_log = count_head(gs_token_hidden, pos_feat)  # [B]
            predicted_count = torch.clamp(
                torch.round(torch.exp(pred_log.detach())).long(),
                1,
                num_groups,
            )
            outputs["predicted_instance_count"] = predicted_count
            lambda_count = float(
                getattr(self.opt, "lambda_instance_group_count", 0.0)
            )
            if (
                self.training
                and lambda_count > 0.0
                and "instance_label_output" in data
            ):
                labels_all = torch.cat(
                    [
                        data["instance_label_input"],
                        data["instance_label_output"],
                    ],
                    dim=1,
                )
                counts = count_instance_masks(
                    labels_all.long(),
                    min_pixels=int(
                        getattr(
                            self.opt, "instance_group_min_instance_pixels", 64
                        )
                    ),
                )  # [B,V]
                gt_count = counts.max(dim=1).values.clamp(1, num_groups)
                target_log = torch.log(gt_count.float())
                count_loss = torch.nn.functional.smooth_l1_loss(
                    pred_log.float(), target_log, reduction="mean"
                )
                outputs["loss_instance_group"] = (
                    outputs["loss_instance_group"]
                    + lambda_count * lambda_eff * count_loss
                )
                with torch.no_grad():
                    outputs["instance_group_gt_count_scene"] = (
                        gt_count.detach()
                    )
                    outputs["loss_instance_group_count"] = (
                        count_loss.detach()
                    )
        return outputs

    def instance_group_loss_on_views(
        self,
        gs_token_hidden: torch.Tensor,
        gaussians: torch.Tensor,
        cam_view: torch.Tensor,
        intrinsics: torch.Tensor,
        instance_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Render group probabilities on arbitrary views and compute the
        instance-group loss (used for test-time refinement on the context
        views, which have GT instance masks but are not the eval views).
        """
        num_groups = self.instance_group_head.num_groups
        group_probs, _, _ = self._instance_group_head_forward(
            gs_token_hidden, gaussians, data=None
        )
        if group_probs.shape[1] == gaussians.shape[1]:
            gaussian_group_probs = group_probs  # already [B,N,G+1]
        else:
            gaussian_group_probs = group_probs.repeat_interleave(
                self.num_gaussians_per_token, dim=1
            )
        render = self.gs.render_feature_channels(
            gaussians,
            gaussian_group_probs,
            cam_view.float(),
            intrinsics=intrinsics,
            opacity_scale=float(
                getattr(self.opt, "instance_group_render_scale", 1.0)
            ),
        )
        rendered_groups = render["images_pred"]
        rendered_alpha = render["alphas_pred"]
        rendered_channels = rendered_groups / (rendered_alpha + 1e-5)
        rendered_probs = rendered_channels.float() / rendered_channels.float().sum(
            dim=2, keepdim=True
        ).clamp_min(1e-6)
        rendered_probability = rendered_probs.permute(0, 2, 1, 3, 4).unsqueeze(
            3
        )  # [B,G+1,V,1,H,W]
        loss, stats = hungarian_instance_group_loss(
            rendered_probability,
            instance_labels.long(),
            num_groups=num_groups,
            min_instance_pixels=int(
                getattr(self.opt, "instance_group_min_instance_pixels", 64)
            ),
            dice_weight=float(
                getattr(self.opt, "lambda_instance_group_dice", 1.0)
            ),
            mask_weight=float(
                getattr(self.opt, "lambda_instance_group_mask", 1.0)
            ),
            void_weight=float(
                getattr(self.opt, "lambda_instance_group_void", 0.1)
            ),
            unmatched_weight=float(
                getattr(self.opt, "lambda_instance_group_unmatched", 0.1)
            ),
            lambda_eff=1.0,
            area_alpha=float(
                getattr(self.opt, "instance_group_area_alpha", 0.0)
            ),
            match_area_norm=bool(
                getattr(self.opt, "instance_group_match_area_norm", False)
            ),
            ce_weight=float(
                getattr(self.opt, "lambda_instance_group_ce", 0.0)
            ),
            match_topk=int(
                getattr(self.opt, "instance_group_match_topk", 1)
            ),
            secondary_pair_weight=float(
                getattr(
                    self.opt,
                    "instance_group_secondary_pair_weight",
                    0.3,
                )
            ),
            usage_entropy_weight=float(
                getattr(self.opt, "instance_group_usage_entropy", 0.0)
            ),
            scene_level_matching=bool(
                getattr(
                    self.opt,
                    "instance_group_scene_level_matching",
                    False,
                )
            ),
        )
        return loss, stats

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        # Any trainable parameter outside the prompt matcher must be
        # checkpointed, otherwise a head that was unfrozen (e.g. the
        # semantic-lifting head) trains in memory but is silently dropped
        # from every saved checkpoint. Previously this loop only ran when
        # ``prompt_unfreeze_tokengs`` was set, which lost the lifting head
        # for frozen-backbone recipes (RE10K semantic lifting).
        for name, parameter in self.named_parameters():
            keep = parameter.requires_grad or name.startswith(
                "instance_branch."
            )
            if bool(getattr(self.opt, "tsh_query_memory_refine_probe", False)) or bool(
                getattr(self.opt, "ta_riu_enabled", False)
            ) or str(
                getattr(self.opt, "ga_idu_mode", "off")
            ) in ("0", "1") or bool(
                getattr(self.opt, "token_eru_enabled", False)
            ):
                keep = keep or name.startswith(
                    (
                        "absolute_gs_head.",
                        "tsh_instance_head.",
                        "enc_dec_backbone.decoder_blocks.",
                    )
                )
            if keep and not name.startswith("prompt_matcher."):
                state[name] = parameter.detach().cpu()
        return state

    def _forward_prompt_reconstruction(self, model_input):
        if getattr(self.opt, "instance_group_conditioned_gaussians", False):
            # The encoder stays frozen, but decoder-only fine-tuning must keep
            # the decoder path in autograd. The previous implementation put
            # both encoder and decoder under no_grad, making lambda_rgb and
            # joint instance supervision unable to change Gaussian generation.
            with torch.no_grad():
                encoder_latent = self.forward_encoder(model_input.encoder)
                geometry_hidden = self.get_gs_tokens(
                    encoder_latent.keys.shape[0],
                    encoder_latent=encoder_latent,
                    decoder_input=model_input.decoder,
                )
                geometry_hidden = self._apply_time_embedding_to_gs_tokens(
                    geometry_hidden, model_input.decoder
                )
            if not bool(getattr(self.opt, "prompt_unfreeze_tokengs", False)):
                with torch.no_grad():
                    for layer in self.enc_dec_backbone.decoder_blocks:
                        geometry_hidden = layer(
                            gs_tokens=geometry_hidden,
                            keys=encoder_latent.keys,
                            values=encoder_latent.values,
                        )
                geometry_hidden = geometry_hidden.detach()
            else:
                use_checkpoint = bool(
                    getattr(
                        self.opt,
                        "instance_group_condition_decoder_checkpoint",
                        False,
                    )
                )
                for layer in self.enc_dec_backbone.decoder_blocks:
                    if use_checkpoint and self.training:
                        geometry_hidden = torch_checkpoint(
                            lambda hidden, keys, values, block=layer: block(
                                gs_tokens=hidden, keys=keys, values=values
                            ),
                            geometry_hidden,
                            encoder_latent.keys,
                            encoder_latent.values,
                            use_reentrant=False,
                        )
                    else:
                        geometry_hidden = layer(
                            gs_tokens=geometry_hidden,
                            keys=encoder_latent.keys,
                            values=encoder_latent.values,
                        )
            conditioned_hidden = self._condition_geometry_hidden(
                geometry_hidden,
                model_input.decoder,
                model_input.encoder,
            )
            gaussians = self.activation_head(conditioned_hidden)
            gaussians = gaussians.clone()
            gaussians[..., 2] = (
                gaussians[..., 2] + self.opt.gaussian_z_offset
            )
            gaussians = self._finalize_conditioned_gaussians(gaussians)
            reconstruction = self._reconstruction_from_gaussians(gaussians)
            if self.training and bool(
                getattr(self.opt, "prompt_unfreeze_tokengs", False)
            ):
                rgb_results = super().render_reconstruction(
                    reconstruction, model_input.decoder
                )
            else:
                with torch.no_grad():
                    rgb_reconstruction = self._reconstruction_from_gaussians(
                        gaussians.detach()
                    )
                    rgb_results = super().render_reconstruction(
                        rgb_reconstruction, model_input.decoder
                    )
            return reconstruction, conditioned_hidden, rgb_results
        if not self.opt.prompt_unfreeze_tokengs:
            return super()._forward_prompt_reconstruction(model_input)
        reconstruction, hidden = super().forward_reconstruction(
            model_input, return_gs_token_hidden=True
        )
        if getattr(
            self.opt, "prompt_detach_reconstruction_tokens", False
        ):
            # Protected-decoder joint: the RGB reconstruction runs from the
            # DETACHED decoder hidden (monitor only, so the reconstruction
            # loss cannot overfit the training windows), while ``hidden``
            # keeps gradients for the instance branch / decoder tail.  The
            # frozen teacher distillation pins the live hidden.
            with torch.no_grad():
                det = hidden.detach()
                gaussians = self.activation_head(det)
                gaussians = gaussians.clone()
                gaussians[..., 2] = (
                    gaussians[..., 2] + self.opt.gaussian_z_offset
                )
                reconstruction = self._reconstruction_from_gaussians(
                    gaussians
                )
                rgb_results = super().render_reconstruction(
                    reconstruction, model_input.decoder
                )
        else:
            rgb_results = super().render_reconstruction(
                reconstruction, model_input.decoder
            )
        return reconstruction, hidden, rgb_results

    def _forward_abs_hidden(self, model_input) -> torch.Tensor:
        """Decoder tokens only (no old activation head / no teacher call).

        Runs encoder + time embedding + the shared token transformer exactly
        like the teacher path, but stops before the Gaussian activation
        head.  The absolute student decodes its complete GS from these
        tokens.
        """
        encoder_latent = self.forward_encoder(model_input.encoder)
        # GA-IDU consumes the real encoder memory, not a fabricated
        # token-to-patch alignment.  Keep a detached audit reference here;
        # the active branch receives the live tensor from this forward.
        self._last_encoder_values = encoder_latent.values
        if encoder_latent.values.ndim == 4:
            self._last_encoder_memory_meta = {
                "shape": list(encoder_latent.values.shape),
                "layout": "[B, num_heads, sequence_length, head_dim]",
                "num_heads": int(encoder_latent.values.shape[1]),
                "sequence_length": int(encoder_latent.values.shape[2]),
                "head_dim": int(encoder_latent.values.shape[3]),
                "reconstructed_shape": [
                    int(encoder_latent.values.shape[0]),
                    int(encoder_latent.values.shape[2]),
                    int(encoder_latent.values.shape[1] * encoder_latent.values.shape[3]),
                ],
            }
        else:
            self._last_encoder_memory_meta = {
                "shape": list(encoder_latent.values.shape),
                "layout": "[B, sequence_length, channels]",
            }
        gs_tokens = self.get_gs_tokens(
            encoder_latent.keys.shape[0],
            encoder_latent=encoder_latent,
            decoder_input=model_input.decoder,
        )
        gs_tokens = self._apply_time_embedding_to_gs_tokens(
            gs_tokens, model_input.decoder
        )
        for layer in self.enc_dec_backbone.decoder_blocks:
            gs_tokens = layer(
                gs_tokens=gs_tokens,
                keys=encoder_latent.keys,
                values=encoder_latent.values,
            )
        condition_geometry = getattr(self, "_condition_geometry_hidden", None)
        if condition_geometry is not None:
            gs_tokens = condition_geometry(gs_tokens, model_input.decoder)
        return gs_tokens

    def _condition_geometry_hidden(
        self,
        geometry_hidden: torch.Tensor,
        decoder_input=None,
        encoder_input=None,
    ) -> torch.Tensor:
        """Inject object context before Gaussian activation when enabled."""
        del decoder_input
        if not bool(
            getattr(self.opt, "instance_group_conditioned_gaussians", False)
        ):
            return geometry_hidden
        adapter = getattr(self, "instance_group_head", None)
        if adapter is None:
            raise RuntimeError("GC2 requires a shared instance_group_head")
        with torch.no_grad():
            proposal_gaussians = self.activation_head(geometry_hidden.detach())
            proposal_gaussians = proposal_gaussians.clone()
            proposal_gaussians[..., 2] = (
                proposal_gaussians[..., 2] + self.opt.gaussian_z_offset
            )
        dense_features = None
        source_c2w = None
        source_intrinsics = None
        if bool(
            getattr(self.opt, "instance_group_condition_image_anchors", False)
        ):
            dense_features = self._gather_dense_features()
            if encoder_input is None or dense_features is None:
                raise RuntimeError(
                    "GC4 image-aligned anchors require encoder dense features"
                )
            source_c2w = encoder_input.cam_to_world_input
            source_intrinsics = encoder_input.intrinsics_input
        conditioned, probabilities, logits, queries, anchor_positions = adapter(
            geometry_hidden,
            proposal_gaussians,
            dense_features=dense_features,
            source_c2w=source_c2w,
            source_intrinsics=source_intrinsics,
            image_hw=tuple(self.opt.img_size),
            num_views=int(self.opt.num_input_views),
        )
        gaussian_probabilities = getattr(
            adapter, "last_gaussian_probabilities", None
        )
        gaussian_opacity_delta = getattr(
            adapter, "last_gaussian_opacity_delta", None
        )
        if gaussian_opacity_delta is None:
            gaussian_opacity_delta = getattr(adapter, "last_opacity_delta", None)
        geometry_offset = getattr(adapter, "last_geometry_offset", None)
        self._instance_conditioning_cache = {
            "probabilities": probabilities,
            "token_probabilities": getattr(
                adapter, "last_token_probabilities", probabilities
            ),
            "logits": logits,
            "queries": queries,
            "anchor_positions": anchor_positions,
            "proposal_gaussians": proposal_gaussians,
            "base_hidden": geometry_hidden,
            "conditioned_hidden": conditioned,
            "gaussian_probabilities": gaussian_probabilities,
            "gaussian_logits": getattr(adapter, "last_gaussian_logits", None),
            "gaussian_opacity_delta": gaussian_opacity_delta,
            "image_anchor_valid_share": getattr(
                adapter, "last_image_anchor_valid_share", None
            ),
            "image_anchor_gate": getattr(
                adapter, "last_image_anchor_gate", None
            ),
            "geometry_offset": geometry_offset,
            "gaussian_opacity_delta": gaussian_opacity_delta,
        }
        if gaussian_probabilities is not None:
            self._instance_conditioning_cache["probabilities"] = (
                gaussian_probabilities
            )
            self._instance_conditioning_cache["logits"] = self._instance_conditioning_cache[
                "gaussian_logits"
            ]
        return conditioned

    def _finalize_conditioned_gaussians(
        self, conditioned_gaussians: torch.Tensor
    ) -> torch.Tensor:
        """Bound GC2 geometry drift in Gaussian space.

        Token hidden perturbations can be amplified by the exponential
        position activation. Blending against the frozen proposal directly
        constrains the final reconstruction while preserving a differentiable
        object-query-to-Gaussian path.
        """
        if not bool(
            getattr(self.opt, "instance_group_conditioned_gaussians", False)
        ):
            return conditioned_gaussians
        cache = getattr(self, "_instance_conditioning_cache", None)
        if not cache or "proposal_gaussians" not in cache:
            raise RuntimeError("GC2 proposal Gaussians are unavailable")
        proposal = cache["proposal_gaussians"].to(conditioned_gaussians.dtype)
        blend = float(
            getattr(self.opt, "instance_group_condition_gaussian_blend", 0.1)
        )
        if not 0.0 < blend <= 1.0:
            raise ValueError(
                "instance_group_condition_gaussian_blend must be in (0, 1]"
            )
        gaussians = proposal + blend * (conditioned_gaussians - proposal)
        geometry_offset = cache.get("geometry_offset")
        if geometry_offset is not None:
            # Experiment B: explicit bounded per-anchor group-driven geometry
            # offsets (broadcast to the token's Gaussians).
            offset = geometry_offset.to(gaussians.dtype).repeat_interleave(
                self.num_gaussians_per_token, dim=1
            )
            gaussians = gaussians.clone()
            gaussians[..., :3] = gaussians[..., :3] + offset
        opacity_delta = cache.get("gaussian_opacity_delta")
        if opacity_delta is not None:
            if (
                opacity_delta.shape[1] != gaussians.shape[1]
                and opacity_delta.shape[1] * self.num_gaussians_per_token
                == gaussians.shape[1]
            ):
                opacity_delta = opacity_delta.repeat_interleave(
                    self.num_gaussians_per_token, dim=1
                )
            gaussians = gaussians.clone()
            gaussians[..., 3:4] = (
                gaussians[..., 3:4]
                + opacity_delta.to(gaussians.dtype)
            ).clamp(0.0, 1.0)
        rotation = F.normalize(gaussians[..., 7:11].float(), dim=-1).to(
            gaussians.dtype
        )
        gaussians = torch.cat(
            [gaussians[..., :7], rotation, gaussians[..., 11:]], dim=-1
        )
        cache["final_gaussians"] = gaussians
        return gaussians

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

    @staticmethod
    def _cosine_ce_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid: torch.Tensor,
        class_weights: tuple[float, ...] | None = None,
    ) -> torch.Tensor:
        """Softmax cross-entropy over the eight rendered cosine logits.

        Classes compete explicitly (positive query vs the other seven
        prototypes), background is ignored -- exactly the decision the C3G
        protocol evaluates. Loss is weighted per GT class (inverse-frequency
        weights when provided).
        """
        logits_flat = logits.permute(0, 1, 3, 4, 2).reshape(-1, 8)
        labels_flat = labels.reshape(-1)
        valid_flat = valid.reshape(-1)
        ce_labels = torch.where(
            labels_flat > 0,
            labels_flat - 1,
            torch.tensor(-100, device=labels.device),
        )
        ce_map = F.cross_entropy(
            logits_flat.float(),
            ce_labels,
            ignore_index=-100,
            reduction="none",
        )
        if class_weights is not None:
            weight_vector = torch.tensor(
                [0.0, *class_weights],
                device=logits.device,
                dtype=torch.float32,
            )
            weight_flat = weight_vector[labels_flat.clamp(min=0)]
        else:
            weight_flat = torch.ones_like(labels_flat.float())
        masked = ce_map * weight_flat * valid_flat.float()
        denom = (weight_flat * valid_flat.float()).sum().clamp_min(1e-6)
        return masked.sum() / denom

    def _tsh_mbm_u2r_loss(
        self,
        depth: torch.Tensor,
        rendered_probability: torch.Tensor,
        rendered_alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Official SIU3R Mask-Guided Geometry Refinement (U->R).

        Official reference (src/pipeline.py:249-265, identical in
        pipeline_multi.py):

            d_dx = D[..., :-1] - D[..., 1:]     (horizontal neighbour diff)
            d_dy = D[..., :-1, :] - D[..., 1:, :]
            inside = adjacent pixels with the SAME predicted segment id
            (hard, detached); boundaries / void / -1 pixels excluded
            L = weight * ( mean(|d_dx| * inside_x) + mean(|d_dy| * inside_y) )

        TokenGS adaptation: D is the differentiable Student-GS rendered
        depth (output views, [B,V,1,H,W]) of the same Gaussian batch that
        renders RGB and instance masks; the segment id map is the per-pixel
        argmax over the G+1 rendered group channels (void excluded), and
        low-confidence / low-alpha pixels never participate.  Masks are
        detached, so this loss updates ONLY geometry (Student GS decoder /
        unit formation / optional decoder tail) -- never the instance head.
        """
        b, view, _, h, w = depth.shape
        g_plus_1 = rendered_probability.shape[1]
        p = rendered_probability.detach().float()  # [B,G+1,V,1,H,W]
        alpha = rendered_alpha.detach().float()  # [B,V,1,H,W]
        ids = p.argmax(dim=1)  # [B,V,1,H,W]
        conf = p.max(dim=1).values  # [B,V,1,H,W]
        min_conf = float(getattr(self.opt, "tsh_mbm_u2r_min_conf", 0.3))
        min_alpha = float(getattr(self.opt, "tsh_mbm_u2r_min_alpha", 0.05))
        valid = (
            (conf >= min_conf)
            & (ids != g_plus_1 - 1)
            & (alpha >= min_alpha)
            & depth.isfinite()
            & (depth > 0.0)
        )  # [B,V,1,H,W]
        depth_f = depth.float()
        d = depth_f[:, :, 0]  # [B,V,H,W]
        v = valid[:, :, 0]  # [B,V,H,W]
        idm = ids[:, :, 0]  # [B,V,H,W]

        same_x = idm[..., 1:] == idm[..., :-1]
        inside_x = v[..., 1:] & v[..., :-1] & same_x
        d_dx = d[..., 1:] - d[..., :-1]
        loss_x = (d_dx.abs() * inside_x.float()).mean()

        same_y = idm[:, :, 1:, :] == idm[:, :, :-1, :]
        inside_y = v[:, :, 1:, :] & v[:, :, :-1, :] & same_y
        d_dy = d[:, :, 1:, :] - d[:, :, :-1, :]
        loss_y = (d_dy.abs() * inside_y.float()).mean()

        loss = loss_x + loss_y
        share_x = inside_x.float().mean()
        share_y = inside_y.float().mean()
        conf_mean = conf.float().mean()
        valid_share = v.float().mean()
        depth_nan = bool((~depth_f.isfinite()).any().item())
        stats = {
            "mbm_u2r_loss_x": loss_x.detach(),
            "mbm_u2r_loss_y": loss_y.detach(),
            "mbm_u2r_interior_share_x": share_x.detach(),
            "mbm_u2r_interior_share_y": share_y.detach(),
            "mbm_u2r_valid_share": valid_share.detach(),
            "mbm_u2r_conf_mean": conf_mean.detach(),
            "mbm_u2r_depth_has_nan": torch.as_tensor(
                float(depth_nan), device=loss.device
            ),
        }
        return loss, stats

    def _forward_tsh_instance_branch(
        self,
        data: dict,
        q_abs: torch.Tensor,
        student_gaussians: torch.Tensor,
        model_input,
        training: bool,
        unit_metric_embeddings: torch.Tensor | None = None,
    ) -> dict:
        """True-Shared instance head: consume q_abs directly, no dual units.

        GS geometry is detached for this path (instance->geometry is OFF in
        this baseline); the pi_gs tensor keeps gradients so instance loss
        reaches SharedUnitInstanceHead and, through q_abs, the single
        absolute unit formation.
        """
        from tokengs.models.instance_group_loss import (
            hungarian_instance_group_loss,
        )

        head = getattr(self, "tsh_instance_head", None)
        if head is None:
            raise RuntimeError("tsh_instance_head is not built")
        ga_idu = getattr(self, "ga_idu1_head", None)
        ga_mode = str(getattr(self.opt, "ga_idu_mode", "off"))
        if ga_idu is not None and ga_mode in ("0", "1"):
            # GA-IDU uses frozen, batch-specific BaseTSH states as its only
            # group anchor.  The old 50-key head is called with gate zero;
            # no independent query table or old refiner is involved.
            base_out = head(
                q_abs,
                refine_gate=0.0,
                return_base_states=True,
                query_state_override=getattr(
                    self, "_token_eru_last_early_query_state", None
                ),
            )
            if self.training:
                ga_gate = float(getattr(self, "ga_idu_gate_eff", 0.0))
            else:
                ga_gate = float(getattr(self, "ga_idu_eval_gate_override", 1.0))
            if ga_mode == "0":
                ga_gate = 0.0
            ga_out = ga_idu(
                q_abs,
                getattr(self, "_last_encoder_values", None),
                base_out["groups"],
                base_out["base_unit_logits"],
                gate=ga_gate,
            )
            head_out = dict(base_out)
            head_out.update(ga_out)
            head_out["base_unit_logits"] = base_out["base_unit_logits"]
        else:
            head_out = None
        unit_eff = float(
            getattr(self, "tsh_unit_grad_eff", 1.0)
            if self.training
            else 1.0
        )
        v2_instance_path = bool(
            getattr(self.opt, "ta_riu_v2_enabled", False)
        ) and getattr(self, "ta_riu_v2_unit_encoder", None) is not None
        v3_instance_path = bool(
            getattr(self.opt, "ta_riu_v3_enabled", False)
        ) and getattr(self, "ta_riu_v3_dual_stream", None) is not None
        max_mult = float(
            getattr(self.opt, "tsh_unit_gradient_multiplier_max", 1.0)
        )
        if v2_instance_path or v3_instance_path:
            # v2 consumes a detached q_abs and produces z_inst through its
            # own trainable encoder.  v3 supplies its own joint instance
            # units.  In both cases the historical unit-producer multiplier
            # must not detach the downstream semantic edge.
            q_in = q_abs
        elif self.training:
            # ``unit_eff`` already includes max_mult*gate; separate the
            # value-domain detach gate (<=1) from the producer-path gradient
            # multiplier (max_mult).  grad_scale creates a separate graph
            # edge from q_abs to the instance head, so RGB/teacher backward
            # through the shared q tensor is never multiplied.
            gate = (
                min(1.0, unit_eff / max_mult)
                if max_mult > 0
                else unit_eff
            )
            if gate < 1.0:
                q_gated = gate * q_abs + (1.0 - gate) * q_abs.detach()
            else:
                q_gated = q_abs
            if max_mult != 1.0 and unit_eff > 0.0:
                q_in = grad_scale(q_gated, max_mult)
            else:
                q_in = q_gated
        else:
            q_in = q_abs
        if self.training:
            query_memory_gate = float(
                getattr(self, "tsh_query_memory_refine_gate_eff", 0.0)
            )
        else:
            # Evaluation uses the fully enabled refiner by default.  A
            # diagnostic may override this for a deterministic gate-0
            # identity comparison without changing the normal eval path.
            query_memory_gate = float(
                getattr(self, "tsh_query_memory_refine_eval_gate_override", 1.0)
            )
        if head_out is None:
            head_out = head(
                q_in,
                refine_gate=query_memory_gate,
                query_state_override=getattr(
                    self, "_token_eru_last_early_query_state", None
                ),
            )
        pi_unit = head_out["pi_unit"]  # [B,T,K,G+1]
        unit_logits = head_out["unit_logits"]  # [B,T,K,G+1]
        qmc_output = None
        if bool(getattr(self.opt, "token_eru_query_metric_enabled", False)):
            if unit_metric_embeddings is None:
                raise RuntimeError(
                    "QMC is enabled but unit_metric_embeddings are missing"
                )
            coupling = getattr(self, "token_eru_query_metric_coupling", None)
            if coupling is None:
                raise RuntimeError("QMC is enabled but its coupling module is missing")
            qmc_gate = float(
                getattr(
                    self,
                    "_token_eru_query_metric_eval_gate_override",
                    self.token_eru_query_metric_gate(
                        int(getattr(self, "_token_eru_query_metric_local_step", 0)),
                        self.opt,
                    ),
                )
            )
            base_unit_logits_4d = unit_logits.view(
                unit_logits.shape[0],
                int(q_abs.shape[1]),
                int(q_abs.shape[2]),
                unit_logits.shape[-1],
            )
            qmc_output = coupling(
                unit_metric_embeddings,
                head_out["refined_groups"],
                base_unit_logits_4d,
                gate=qmc_gate,
            )
            unit_logits = qmc_output.final_unit_logits.reshape_as(unit_logits)
            pi_unit = F.softmax(
                qmc_output.final_unit_logits.float(), dim=-1
            ).view_as(pi_unit)
        refine_head = getattr(self, "tsh_slot_refine_head", None)
        if self.training:
            per_gs_gate_eff = float(
                getattr(self, "tsh_per_gs_gate_eff", 0.0)
            )
        else:
            per_gs_gate_eff = 1.0
        if refine_head is not None:
            final_logits, residual_logits, alpha_eff = refine_head(
                q_in,
                unit_logits,
                head_out.get("groups"),
                student_gaussians,
                per_gs_gate_eff,
            )
        else:
            final_logits = unit_logits.reshape(
                unit_logits.shape[0],
                int(q_abs.shape[1]) * int(q_abs.shape[2]),
                int(head.num_groups) + 1,
            ).repeat_interleave(int(head.gaussians_per_unit), dim=1)
            residual_logits = None
            alpha_eff = 0.0
        pi_gs = F.softmax(final_logits.float(), dim=-1)  # [B,T*K*8,G+1]
        self._tsh_last_pi_unit = pi_unit.detach()
        self._tsh_last_pi_gs = pi_gs.detach()

        gs = int(getattr(head, "gaussians_per_unit", 8))
        block_view = pi_gs.reshape(
            pi_gs.shape[0],
            int(q_abs.shape[1]),
            int(q_abs.shape[2]),
            gs,
            pi_gs.shape[-1],
        )
        block_logit_view = final_logits.reshape(
            pi_gs.shape[0],
            int(q_abs.shape[1]),
            int(q_abs.shape[2]),
            gs,
            pi_gs.shape[-1],
        )
        if refine_head is None or per_gs_gate_eff <= 0.0:
            # Gate closed: identity must reproduce the checkpoint behavior.
            if not bool(
                (block_view - block_view[..., :1, :]).abs().max() < 1e-6
            ):
                raise RuntimeError(
                    "per-GS gate=0 must keep the 8 GS of a unit identical"
                )
        within_unit_logit_diff = (
            (block_logit_view - block_logit_view[..., :1, :])
            .abs()
            .max(dim=-1)
            .values
        )
        within_unit_prob_diff = (
            (block_view - block_view[..., :1, :])
            .abs()
            .max(dim=-1)
            .values
        )
        per_gs_stats = {
            "tsh_per_gs_alpha": torch.tensor(
                float(alpha_eff), device=pi_gs.device
            ),
            "tsh_per_gs_unit_logit_diff_mean": (
                within_unit_logit_diff.mean().detach()
                if within_unit_logit_diff.numel()
                else torch.zeros((), device=pi_gs.device)
            ),
            "tsh_per_gs_unit_logit_diff_max": (
                within_unit_logit_diff.max().detach()
                if within_unit_logit_diff.numel()
                else torch.zeros((), device=pi_gs.device)
            ),
            "tsh_per_gs_unit_prob_diff_mean": (
                within_unit_prob_diff.mean().detach()
                if within_unit_prob_diff.numel()
                else torch.zeros((), device=pi_gs.device)
            ),
        }

        # Historical TSH keeps the RGB geometry detached from instance loss.
        # TA-RIU is the explicit exception: its same G_joint tensor is the
        # reconstruction and mask carrier, so the residual readout remains in
        # this graph.  The base q_abs/absolute GS modules are still frozen by
        # the TA-RIU model boundary.
        if bool(getattr(self.opt, "ta_riu_enabled", False)):
            # TA-RIU keeps the same G_joint support for masks, but the first
            # version deliberately blocks instance->opacity updates. RGB
            # still renders the full joint Gaussian tensor above, so opacity
            # and appearance residuals remain RGB-trainable.
            render_gs = student_gaussians.clone()
            render_gs[..., 3:4] = student_gaussians[..., 3:4].detach()
        elif bool(
            getattr(self.opt, "token_eru_dino_metric_joint_formation", False)
        ) or getattr(self, "token_eru_3d_anchor", None) is not None:
            # JointFormation deliberately keeps the native instance mask
            # renderer on the live Student-GS tensor.  The legacy ERU and
            # Stage-M paths retain their historical detached geometry path.
            # render_feature_channels consumes assignment channels, so SH/
            # color do not acquire an instance-loss edge.
            render_gs = student_gaussians
        else:
            render_gs = student_gaussians.detach()
        self._tsh_last_render_gs = render_gs
        render = self.gs.render_feature_channels(
            render_gs,
            pi_gs,
            model_input.decoder.cam_view,
            intrinsics=model_input.decoder.intrinsics,
            opacity_scale=float(
                getattr(self.opt, "instance_group_render_scale", 1.0)
            ),
        )
        rendered_groups = render["images_pred"]
        rendered_alpha = render["alphas_pred"]
        rendered_channels = rendered_groups / (rendered_alpha + 1e-5)
        rendered_probs = (
            rendered_channels.float()
            / rendered_channels.float().sum(dim=2, keepdim=True).clamp_min(1e-6)
        )
        rendered_probability = rendered_probs.permute(
            0, 2, 1, 3, 4
        ).unsqueeze(3)  # [B,G+1,V,1,H,W]

        outputs = {
            "pi_unit": pi_unit,
            "gaussian_group_probabilities": pi_gs,
            "instance_group_probabilities": pi_unit,
            "pi_gs_refined": pi_gs,
            "unit_logits": unit_logits.detach(),
            **per_gs_stats,
            "rendered_instance_group_probability": rendered_probability,
            "rendered_instance_group_alpha": rendered_alpha,
            "tsh_adapter_z": head_out["adapter_z"],
            "loss_instance_group": torch.zeros(
                (), device=rendered_probability.device
            ),
        }
        if qmc_output is not None:
            outputs.update(
                {
                    "query_metric_enabled": True,
                    "query_metric_gate": qmc_output.gate.detach(),
                    "query_metric_temperature": qmc_output.temperature.detach(),
                    "query_metric_group_logits": qmc_output.metric_group_logits.detach(),
                    "query_metric_centered_logits": qmc_output.centered_metric_group_logits.detach(),
                    "query_metric_residual_logits": qmc_output.residual_logits.detach(),
                    "query_metric_query_embeddings": qmc_output.query_metric_embeddings.detach(),
                    "query_metric_residual_abs_mean": qmc_output.residual_logits.abs().mean().detach(),
                    "query_metric_residual_abs_max": qmc_output.residual_logits.abs().max().detach(),
                    "query_metric_final_vs_base_max_diff": (
                        qmc_output.final_unit_logits
                        - head_out["unit_logits"].view_as(qmc_output.final_unit_logits)
                    ).abs().max().detach(),
                }
            )
        else:
            outputs["query_metric_enabled"] = False
        for key, value in head_out.items():
            if key.startswith(("unit_logits", "base_unit_logits", "residual_logits",
                               "assignment_temperature", "query_memory_refine_gate",
                               "void_share", "max_group_prob_share")):
                # When QMC is enabled, ``unit_logits`` above is the final
                # logits tensor used for pi_unit, child-Gaussian inheritance,
                # and rendering.  Do not overwrite that diagnostic with the
                # SharedUnitInstanceHead base logits here.
                if qmc_output is not None and key == "unit_logits":
                    continue
                outputs[key] = value.detach()
        if training and "instance_label_output" in data:
            matching_mode = str(
                getattr(self.opt, "token_eru_matching_mode", "per_view")
            )
            if matching_mode == "per_view":
                loss, stats = hungarian_instance_group_loss(
                    rendered_probability,
                    data["instance_label_output"].long(),
                    num_groups=int(head.num_groups),
                    min_instance_pixels=int(
                        getattr(self.opt, "instance_group_min_instance_pixels", 32)
                    ),
                    dice_weight=float(
                        getattr(self.opt, "lambda_instance_group_dice", 1.0)
                    ),
                    mask_weight=float(
                        getattr(self.opt, "lambda_instance_group_mask", 1.0)
                    ),
                    void_weight=float(
                        getattr(self.opt, "lambda_instance_group_void", 0.1)
                    ),
                    unmatched_weight=float(
                        getattr(self.opt, "lambda_instance_group_unmatched", 0.1)
                    ),
                    usage_entropy_weight=float(
                        getattr(self.opt, "instance_group_usage_entropy", 0.05)
                    ),
                    scene_level_matching=False,
                )
            elif matching_mode == "scene":
                if rendered_probability.shape[2] != 7:
                    raise AssertionError(
                        "TokenGS-ERU scene matching requires 7 target views"
                    )
                from tokengs.models.token_eru.scene_hungarian_loss import (
                    scene_hungarian_instance_group_loss,
                )

                scene_config = dict(vars(self.opt))
                scene_config["_scene_void_probability"] = rendered_probability[
                    :, int(head.num_groups), :, 0
                ]
                loss, stats = scene_hungarian_instance_group_loss(
                    rendered_probability[:, : int(head.num_groups), :, 0].permute(
                        0, 2, 1, 3, 4
                    ),
                    data["instance_label_output"].long(),
                    existing_loss_config=scene_config,
                )
                if int(stats["instance_group_hungarian_calls"]) != rendered_probability.shape[0]:
                    raise AssertionError(
                        "scene matching must call Hungarian once per batch scene"
                    )
            else:
                raise ValueError(
                    "token_eru_matching_mode must be 'per_view' or 'scene', "
                    f"got {matching_mode!r}"
                )
            outputs["loss_instance_group"] = loss
            for key, value in stats.items():
                outputs[f"tsh_{key}"] = (
                    value.detach() if torch.is_tensor(value) else value
                )
        return outputs

    def forward(
        self,
        data: dict,
        skip_loss: bool = False,
        compute_quality_metrics: bool = False,
        **_kwargs,
    ) -> dict:
        del skip_loss
        model_input, supervision = split_data(data, self.opt)
        expected_gaussians = (
            int(self.opt.num_gs_tokens) * int(self.num_gaussians_per_token)
        )
        abs_mode = (
            bool(getattr(self.opt, "instance_branch_abs_units", False))
            and getattr(self, "absolute_gs_head", None) is not None
        )
        generative_units = False  # overwritten below on the non-abs path
        ta_riu_enabled = bool(
            getattr(self.opt, "ta_riu_enabled", False)
        ) and getattr(self, "ta_riu_shared_mixer", None) is not None
        ta_riu_q_abs = None
        ta_riu_z_shared = None
        ta_riu_memory = None
        ta_riu_delta = None
        ta_riu_geo_raw = None
        ta_riu_app_raw = None
        ta_riu_base_gaussians = None
        ta_riu_joint_gaussians = None
        ta_riu_v2_enabled = bool(
            getattr(self.opt, "ta_riu_v2_enabled", False)
        ) and getattr(self, "ta_riu_v2_unit_encoder", None) is not None
        ta_riu_v2_outputs = None
        ta_riu_v2_loss = torch.zeros((), device=next(self.parameters()).device)
        ta_riu_v3_enabled = bool(
            getattr(self.opt, "ta_riu_v3_enabled", False)
        ) and getattr(self, "ta_riu_v3_dual_stream", None) is not None
        ta_riu_v3_outputs = None
        ta_riu_v3_align_loss = torch.zeros(
            (), device=next(self.parameters()).device
        )
        ta_riu_v3_align_raw = torch.zeros(
            (), device=next(self.parameters()).device
        )
        token_eru_active = getattr(self, "token_eru_decoder", None) is not None
        token_eru_dino_metric_outputs = {}
        token_eru_3d_anchor_outputs = {}
        token_eru_dino_metric_loss = torch.zeros(
            (), device=next(self.parameters()).device
        )
        teacher_on = abs_mode and self.training and (
            float(getattr(self, "teacher_lambda_eff", 0.0)) > 0.0
        )
        if abs_mode:
            if teacher_on:
                # Bootstrap stage: the frozen old GS head runs under no_grad
                # and is used ONLY to provide teacher GS + teacher RGB.
                with torch.no_grad():
                    teacher_recon, gs_token_hidden, teacher_rgb_res = (
                        self._forward_prompt_reconstruction(model_input)
                    )
                if float(
                    getattr(self.opt, "tsh_mbm_decoder_tail_lr", 0.0)
                ) > 0.0:
                    # MBM family: the shared decoder tail keeps autograd even
                    # during the early teacher window, otherwise DDP treats
                    # the trainable decoder blocks as unused in the first
                    # optimizer step.  The old-head teacher path itself stays
                    # under no_grad (teacher artifacts above are detached).
                    gs_token_hidden = self._forward_abs_hidden(model_input)
                teacher_gaussians = teacher_recon.gaussians.detach()
                teacher_rgb = teacher_rgb_res["images_pred"].detach()
                self._teacher_gaussians = teacher_gaussians
                self.teacher_called = True
            else:
                # Teacher off: skip the old GS head entirely.  The student
                # forward/backward goes Token->Unit->GS only.
                if token_eru_active:
                    # ERU must retain autograd through both new streams even
                    # though the original reconstruction modules are frozen:
                    # u2r adapters need the live frozen-path Jacobian.
                    gs_token_hidden = self._forward_abs_hidden(model_input)
                elif bool(
                    getattr(self.opt, "prompt_unfreeze_tokengs", False)
                ) and float(
                    getattr(self.opt, "tsh_mbm_decoder_tail_lr", 0.0)
                ) > 0.0:
                    # MBM family: the shared token-transformer tail stays in
                    # autograd so its low-LR group can be updated by RGB /
                    # instance / U->R gradients once the teacher is off.
                    gs_token_hidden = self._forward_abs_hidden(model_input)
                else:
                    with torch.no_grad():
                        gs_token_hidden = self._forward_abs_hidden(
                            model_input
                        )
                teacher_gaussians = None
                teacher_rgb = None
                self._teacher_gaussians = None
                self.teacher_called = False
            if gs_token_hidden.ndim != 3 or gs_token_hidden.shape[-1] != 1024:
                raise RuntimeError(
                    f"Expected gs_token_hidden [B,T,1024], got "
                    f"{tuple(gs_token_hidden.shape)}"
                )
            new_gaussians, q_abs, _unit_centers = self.absolute_gs_head(
                gs_token_hidden
            )
            # Runtime-only references for the JointFormation gradient audit;
            # these are not parameters, buffers, or checkpoint state.
            self._tsh_last_q_abs_live = q_abs
            self._tsh_last_student_gaussians_live = new_gaussians
            self._tsh_last_q_abs = q_abs.detach().clone()
            self._tsh_last_student_gaussians = new_gaussians.detach().clone()
            if new_gaussians.shape[1] != expected_gaussians:
                raise RuntimeError(
                    f"Absolute student expects {expected_gaussians} "
                    f"Gaussians, got {new_gaussians.shape[1]}"
                )
            if ta_riu_v3_enabled:
                if "images_input" not in data:
                    raise RuntimeError("TA-RIU-v3 requires context images_input")
                v3_gate = (
                    float(getattr(self, "ta_riu_v3_gate_eff", 0.0))
                    if self.training
                    else float(
                        getattr(self, "ta_riu_v3_eval_gate_override", 1.0)
                    )
                )
                ta_riu_v3_outputs = self.ta_riu_v3_dual_stream(
                    q_abs,
                    data["images_input"],
                    v3_gate,
                )
                joint_q = unflatten_reconstruction_units(
                    ta_riu_v3_outputs.joint_reconstruction_units,
                    context_views=int(
                        getattr(self.opt, "ta_riu_v3_context_views", 8)
                    ),
                    units_per_view=int(
                        getattr(self.opt, "ta_riu_v3_units_per_view", 1024)
                    ),
                )
                new_gaussians, _ = self.absolute_gs_head.decode_units(joint_q)
                if v3_gate == 0.0:
                    q_abs_for_instance = q_abs
                else:
                    q_abs_for_instance = unflatten_reconstruction_units(
                        ta_riu_v3_outputs.joint_instance_units,
                        context_views=int(
                            getattr(self.opt, "ta_riu_v3_context_views", 8)
                        ),
                        units_per_view=int(
                            getattr(self.opt, "ta_riu_v3_units_per_view", 1024)
                        ),
                    )
                if v3_gate > 0.0 and self.training:
                    rec_norm = F.layer_norm(
                        ta_riu_v3_outputs.joint_reconstruction_units,
                        (ta_riu_v3_outputs.joint_reconstruction_units.shape[-1],),
                    )
                    ins_norm = F.layer_norm(
                        ta_riu_v3_outputs.joint_instance_units,
                        (ta_riu_v3_outputs.joint_instance_units.shape[-1],),
                    )
                    ta_riu_v3_align_raw = (
                        1.0
                        - F.cosine_similarity(
                            rec_norm.detach(), ins_norm, dim=-1, eps=1e-6
                        )
                    ).mean()
                    ta_riu_v3_align_loss = ta_riu_v3_align_raw * v3_gate
                self._tsh_last_student_gaussians = new_gaussians.detach().clone()
            if ta_riu_enabled:
                from tokengs.models.ta_riu import apply_joint_residual

                ta_riu_q_abs = q_abs
                encoder_values = getattr(self, "_last_encoder_values", None)
                if encoder_values is None:
                    raise RuntimeError("TA-RIU requires encoder memory from the same forward")
                memory_override = getattr(self, "ta_riu_memory_override", None)
                if memory_override is not None:
                    encoder_values = memory_override.to(
                        device=encoder_values.device,
                        dtype=encoder_values.dtype,
                    )
                memory_mode = str(getattr(self, "ta_riu_memory_mode", "normal"))
                if memory_mode == "zero":
                    encoder_values = torch.zeros_like(encoder_values)
                elif memory_mode == "shuffle":
                    # Diagnostic-only memory ablation.  It is deliberately
                    # deterministic and never selected by the trainer.
                    if encoder_values.shape[0] > 1:
                        encoder_values = torch.roll(encoder_values, shifts=1, dims=0)
                    else:
                        encoder_values = torch.flip(encoder_values, dims=(-2,))
                ta_gate = (
                    float(getattr(self, "ta_riu_gate_eff", 1.0))
                    if self.training
                    else float(getattr(self, "ta_riu_eval_gate_override", 1.0))
                )
                geo_gate = (
                    float(getattr(self, "ta_riu_geo_gate_eff", ta_gate))
                    if self.training
                    else float(getattr(self, "ta_riu_eval_geo_gate_override", ta_gate))
                )
                app_gate = (
                    float(getattr(self, "ta_riu_app_gate_eff", ta_gate))
                    if self.training
                    else float(getattr(self, "ta_riu_eval_app_gate_override", ta_gate))
                )
                if ta_gate == 0.0 and geo_gate == 0.0 and app_gate == 0.0:
                    # Strict step-0 identity: do not introduce a multiply by
                    # zero / clone-and-assign round trip into the renderer.
                    ta_riu_z_shared = q_abs
                    ta_riu_memory = torch.zeros(
                        q_abs.shape[0],
                        int(getattr(self.opt, "ta_riu_memory_latents", 256)),
                        int(getattr(self.opt, "ta_riu_dim", 256)),
                        device=q_abs.device,
                        dtype=q_abs.dtype,
                    )
                    ta_riu_delta = torch.zeros_like(q_abs)
                else:
                    ta_riu_z_shared, ta_riu_memory, ta_riu_delta = (
                        self.ta_riu_shared_mixer(q_abs, encoder_values, ta_gate)
                    )
                ta_riu_geo_raw = self.ta_riu_geometry_head(ta_riu_z_shared)
                ta_riu_app_raw = self.ta_riu_appearance_head(ta_riu_z_shared)
                ta_riu_base_gaussians = new_gaussians.detach().clone()
                if ta_gate == 0.0 and geo_gate == 0.0 and app_gate == 0.0:
                    ta_riu_joint_gaussians = ta_riu_base_gaussians
                else:
                    ta_riu_joint_gaussians = apply_joint_residual(
                        ta_riu_base_gaussians,
                        ta_riu_geo_raw,
                        ta_riu_app_raw,
                        geo_gate,
                        app_gate,
                        xyz_scale=float(getattr(self.opt, "ta_riu_xyz_scale", 0.02)),
                        log_scale_scale=float(getattr(self.opt, "ta_riu_log_scale_scale", 0.05)),
                        rot_scale=float(getattr(self.opt, "ta_riu_rot_scale", 0.05)),
                        opacity_scale=float(getattr(self.opt, "ta_riu_opacity_scale", 0.05)),
                        color_scale=float(getattr(self.opt, "ta_riu_color_scale", 0.05)),
                    )
                new_gaussians = ta_riu_joint_gaussians
                self._tsh_last_student_gaussians = new_gaussians.detach().clone()
                q_abs_for_instance = ta_riu_z_shared
            elif not ta_riu_v3_enabled:
                q_abs_for_instance = (
                    getattr(self, "_token_eru_understanding_units", q_abs)
                    if token_eru_active
                    else q_abs
                )
            reconstruction = self._reconstruction_from_gaussians(
                new_gaussians
            )
            rgb_results = super().render_reconstruction(
                reconstruction, model_input.decoder
            )
            self._tsh_last_rgb = rgb_results["images_pred"].detach().clone()
            gaussians = new_gaussians.detach().clone()
            instance_gaussians = new_gaussians
            self._last_abs_student_gaussians = (
                new_gaussians.detach().clone()
            )
            if ta_riu_v2_enabled:
                v2_gate = (
                    float(getattr(self, "ta_riu_v2_gate_eff", 1.0))
                    if self.training
                    else float(getattr(self, "ta_riu_v2_eval_gate_override", 1.0))
                )
                if "images_input" not in data:
                    raise RuntimeError("TA-RIU-v2 requires context images_input")
                image_hw = tuple(int(x) for x in data["images_input"].shape[-2:])
                ta_riu_v2_outputs = self.ta_riu_v2_unit_encoder(
                    q_abs.detach(),
                    new_gaussians.detach(),
                    data["images_input"],
                    data["cam_view_input"],
                    data["intrinsics_input"],
                    image_hw=image_hw,
                    gate=v2_gate,
                )
                q_abs_for_instance = ta_riu_v2_outputs["z_inst"]
                if self.training and "instance_label_output" in data:
                    targets, valid_units, target_stats = build_unit_soft_instance_targets(
                        new_gaussians.detach(),
                        data,
                        tuple(int(x) for x in data["instance_label_output"].shape[-2:]),
                        num_tokens=int(self.opt.num_gs_tokens),
                        units_per_token=int(self.absolute_gs_head.units_per_token),
                        gaussians_per_unit=int(self.absolute_gs_head.gaussians_per_unit),
                        min_valid_votes=int(getattr(self.opt, "ta_riu_v2_min_valid_votes", 8)),
                        min_foreground_fraction=float(getattr(self.opt, "ta_riu_v2_min_foreground_fraction", .25)),
                    )
                    ta_riu_v2_loss, nce_stats = soft_unit_info_nce(
                        ta_riu_v2_outputs["unit_embedding"],
                        targets,
                        valid_units,
                        temperature=float(getattr(self.opt, "ta_riu_v2_embedding_temperature", .1)),
                        max_units=int(getattr(self.opt, "ta_riu_v2_embedding_max_units", 2048)),
                    )
                    ta_riu_v2_outputs.update(target_stats)
                    ta_riu_v2_outputs.update(nce_stats)
                q_abs_for_instance = ta_riu_v2_outputs["z_inst"]
            if getattr(self, "token_eru_dino_encoder", None) is not None:
                token_eru_dino_metric_outputs = self._forward_token_eru_dino_metric(
                    data,
                    q_abs_for_instance,
                    new_gaussians,
                    model_input,
                    training=self.training,
                )
                q_abs_for_instance = token_eru_dino_metric_outputs[
                    "fused_understanding_units"
                ]
                token_eru_dino_metric_loss = token_eru_dino_metric_outputs[
                    "loss_instance_metric_weighted"
                ]
            anchor = getattr(self, "token_eru_3d_anchor", None)
            if anchor is not None:
                anchor_output = anchor(
                    q_abs_for_instance,
                    new_gaussians[..., :3],
                    new_gaussians[..., 3:4],
                )
                q_abs_for_instance = anchor_output.anchored_units
                token_eru_3d_anchor_outputs = {
                    "unit_3d_anchor_centers_world": anchor_output.unit_centers_world,
                    "unit_3d_anchor_centers_normalized": anchor_output.unit_centers_normalized,
                    "unit_3d_anchor_opacity_mass": anchor_output.opacity_mass,
                    "unit_3d_anchor_fallback_mask": anchor_output.fallback_mask,
                    "unit_3d_anchor_position_features": anchor_output.position_features,
                    "unit_3d_anchor_delta": anchor_output.anchor_delta,
                    "unit_3d_anchor_anchored_units": anchor_output.anchored_units,
                }
                self._token_eru_3d_anchor_output = anchor_output
        else:
            reconstruction, gs_token_hidden, rgb_results = (
                self._forward_prompt_reconstruction(model_input)
            )
            if gs_token_hidden.ndim != 3 or gs_token_hidden.shape[-1] != 1024:
                raise RuntimeError(
                    f"Expected gs_token_hidden [B,T,1024], got "
                    f"{tuple(gs_token_hidden.shape)}"
                )
            if reconstruction.gaussians.shape[1] != expected_gaussians:
                raise RuntimeError(
                    f"SemanticTokenGSv4 requires exactly {expected_gaussians} "
                    f"Gaussians"
                )
            gaussians = reconstruction.gaussians
            # Frozen-teacher artifacts for the generative-unit bootstrap
            # (residual v3): the original activation-head Gaussians + RGB are
            # distillation targets only.
            teacher_gaussians = gaussians.detach()
            teacher_rgb = rgb_results["images_pred"].detach()
            self._teacher_gaussians = teacher_gaussians
            self.teacher_called = True

            generative_units = (
                bool(getattr(self.opt, "generative_units", False))
                and getattr(self, "instance_branch", None) is not None
            )
            if generative_units:
                # Token -> K local units -> unit-decoded Gaussians (zero-init
                # residual; keeps the 0.324-style geometry as bootstrap).
                teacher_rgb = rgb_results["images_pred"].detach()
                new_gaussians = (
                    self.instance_branch.forward_generative_gaussians(
                        gs_token_hidden, gaussians
                    )
                )
                reconstruction = self._reconstruction_from_gaussians(
                    new_gaussians
                )
                rgb_results = super().render_reconstruction(
                    reconstruction, model_input.decoder
                )
                gaussians = new_gaussians.detach().clone()
                instance_gaussians = new_gaussians
                self._last_abs_student_gaussians = (
                    new_gaussians.detach().clone()
                )
            else:
                instance_gaussians = gaussians

        has_prompts = (
            "prompt_mode" in data and "prompt_class_id" in data
        )
        prompt_embedding = None
        positive_class_ids = None
        if has_prompts:
            prompt_embedding = self._encode_prompt_batch(data)
            # Manifest class ids are 1..8; prototype indices are 0..7.
            positive_class_ids = data["prompt_class_id"].long() - 1

        if abs_mode:
            # Absolute-student phase has semantic OFF: run the prompt/semantic
            # field under no_grad (keeps the render outputs for metrics) so
            # its adapters/heads receive no gradients and their losses never
            # enter the total loss.
            with torch.no_grad():
                semantic_output = self.prompt_matcher(
                    gs_token_hidden,
                    gaussians,
                    prompt_embeddings=prompt_embedding,
                    positive_class_ids=positive_class_ids,
                )
        else:
            semantic_output = self.prompt_matcher(
                gs_token_hidden,
                gaussians,
                prompt_embeddings=prompt_embedding,
                positive_class_ids=positive_class_ids,
            )
        gaussian_features = semantic_output["gaussian_features"]  # [B,N,D]
        gaussian_logits = semantic_output["token_logits"]  # [B,8,N]
        if gaussian_logits.shape[1:] != (8, expected_gaussians):
            raise RuntimeError(
                f"Expected gaussian logits [B,8,{expected_gaussians}], got "
                f"{tuple(gaussian_logits.shape)}"
            )
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
        rendered_all = feature_render["images_pred"]  # [B,V,D+8,H,W]
        rendered_alpha = feature_render["alphas_pred"]  # [B,V,1,H,W]
        rendered_features = rendered_all[:, :, :feature_dim]
        prob_end = feature_dim + 8
        rendered_probability = (
            rendered_all[:, :, feature_dim:prob_end]
            .permute(0, 2, 1, 3, 4)
            .unsqueeze(3)
        )  # [B,8,V,1,H,W]

        has_sem = "semantic_label_output" in data
        if has_sem and not abs_mode:
            target_mask, label_valid = self._build_targets(
                data["semantic_label_output"].long()
            )
            valid_mask = (
                rendered_alpha
                >= float(self.opt.prompt_valid_alpha_threshold)
            ) & label_valid
            losses = compute_prompt_mask_loss(
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
            semantic_metrics = compute_semantic_v2_metrics(
                rendered_probability,
                target_mask,
                valid_mask,
                threshold=self.opt.prompt_threshold,
            )
        else:
            losses = {
                "loss": torch.zeros((), device=rendered_probability.device)
            }
            semantic_metrics = {}
            if has_sem:
                # Absolute-student phase keeps semantic losses OFF; provide
                # the placeholder keys consumed by diagnostics only.
                semantic_metrics["class_target_count"] = torch.zeros(
                    (), device=rendered_probability.device
                )
            target_mask = torch.zeros_like(rendered_probability)
            valid_mask = torch.zeros(
                rendered_alpha.shape[0],
                rendered_alpha.shape[1],
                rendered_alpha.shape[3],
                rendered_alpha.shape[4],
                dtype=torch.bool,
                device=rendered_probability.device,
            ).unsqueeze(2)

        loss_feat = torch.zeros((), device=rendered_probability.device)
        if (
            self.training
            and not abs_mode
            and float(self.opt.lambda_feat) > 0
        ):
            gt_rgb = supervision.images_output
            dense = self.prompt_matcher.prompt_encoder.encode_dense_features(
                gt_rgb
            )  # [B,V,7,7,512]
            if self.prompt_matcher.teacher_projection is not None:
                teacher = torch.einsum(
                    "bvhwc,cd->bvhwd",
                    dense,
                    self.prompt_matcher.teacher_projection,
                )
            else:
                teacher = self.prompt_matcher.feature_align(dense)
            teacher = F.normalize(teacher.float(), dim=-1)
            batch_size, view_count = teacher.shape[:2]
            teacher_flat = teacher.permute(0, 1, 4, 2, 3).flatten(0, 1)
            teacher_flat = F.interpolate(
                teacher_flat,
                size=self.opt.img_size,
                mode="bilinear",
                align_corners=False,
            )
            teacher_map = F.normalize(teacher_flat, dim=1).view(
                batch_size, view_count, feature_dim, *self.opt.img_size
            )
            rendered_feat = F.normalize(rendered_features, dim=2)
            valid_feat = (
                rendered_alpha >= float(self.opt.prompt_valid_alpha_threshold)
            )
            cosine = (rendered_feat * teacher_map).sum(dim=2)  # [B,V,H,W]
            loss_feat = float(self.opt.lambda_feat) * (
                1.0 - cosine
            )[valid_feat[:, :, 0]].mean()

        loss_ce_cosine = torch.zeros((), device=rendered_probability.device)
        if use_ce_cosine and has_sem and not abs_mode:
            rendered_logits = rendered_all[:, :, prob_end:]  # [B,V,8,H,W]
            gt_labels = data["semantic_label_output"].long()
            alpha_valid = (
                rendered_alpha
                >= float(self.opt.prompt_valid_alpha_threshold)
            )
            valid_ce = alpha_valid[:, :, 0] & (gt_labels > 0)
            loss_ce_cosine = lambda_ce_cosine * self._cosine_ce_loss(
                rendered_logits,
                gt_labels,
                valid_ce,
                getattr(self.opt, "semantic_v2_class_weights", None),
            )

        loss_instance_contrastive = torch.zeros(
            (), device=rendered_probability.device
        )
        lambda_contrastive = float(
            getattr(self.opt, "lambda_instance_contrastive", 0.0)
        )
        if (
            self.training
            and not abs_mode
            and lambda_contrastive > 0
            and "instance_label_output" in data
        ):
            alpha_valid = (
                rendered_alpha
                >= float(self.opt.prompt_valid_alpha_threshold)
            )
            loss_instance_contrastive = (
                lambda_contrastive
                * instance_contrastive_loss(
                    rendered_features,
                    data["instance_label_output"].long(),
                    alpha_valid,
                    temperature=0.07,
                    min_pixels=int(
                        getattr(
                            self.opt,
                            "instance_group_min_instance_pixels",
                            32,
                        )
                    ),
                )
            )

        token_probabilities_token = token_probabilities.view(
            token_probabilities.shape[0],
            8,
            int(self.opt.num_gs_tokens),
            64,
        ).mean(dim=-1)
        diagnostics = self._embedding_diagnostics(
            semantic_output["semantic_tokens"],
            semantic_output["semantic_prompts"],
            token_probabilities_token,
        )
        diagnostics["class_gt_present"] = (
            semantic_metrics["class_target_count"] > 0
            if has_sem
            else torch.tensor(False, device=rendered_probability.device)
        )
        diagnostics["feature_teacher_cosine"] = (
            1.0 - loss_feat / max(float(self.opt.lambda_feat), 1e-6)
            if float(self.opt.lambda_feat) > 0
            else torch.zeros((), device=rendered_probability.device)
        )

        pred_rgb = rgb_results["images_pred"]
        gt_rgb = supervision.images_output
        loss_rgb = torch.zeros((), device=pred_rgb.device)
        if float(self.opt.lambda_rgb) > 0:
            loss_rgb = float(self.opt.lambda_rgb) * (
                pred_rgb.clamp(0, 1) - gt_rgb.clamp(0, 1)
            ).square().mean()
        loss_boundary_rgb = torch.zeros((), device=pred_rgb.device)
        lambda_boundary_rgb = float(
            getattr(self.opt, "lambda_boundary_rgb", 0.0)
        )
        if (
            self.training
            and lambda_boundary_rgb > 0
            and "instance_label_output" in data
        ):
            boundary = self._instance_boundary_mask(
                data["instance_label_output"].long(),
                dilate=int(getattr(self.opt, "boundary_rgb_dilate", 1)),
                include_background=bool(
                    getattr(
                        self.opt,
                        "boundary_rgb_include_background",
                        True,
                    )
                ),
            )
            mse_map = (
                pred_rgb.clamp(0, 1) - gt_rgb.clamp(0, 1)
            ).square().mean(dim=2)  # [B,V,H,W]
            boundary_float = boundary.float()
            loss_boundary_rgb = (
                lambda_boundary_rgb
                * (mse_map * boundary_float).sum()
                / boundary_float.sum().clamp_min(1.0)
            )
        joint_loss = (
            losses["loss"]
            + loss_rgb
            + loss_feat
            + loss_ce_cosine
            + loss_instance_contrastive
            + loss_boundary_rgb
        )
        if self.training and token_eru_dino_metric_outputs:
            joint_loss = joint_loss + token_eru_dino_metric_loss
        if self.training and ta_riu_v2_enabled:
            joint_loss = joint_loss + ta_riu_v2_loss * float(
                getattr(self.opt, "ta_riu_v2_embedding_weight", .1)
            )
        if self.training and ta_riu_v3_enabled:
            joint_loss = joint_loss + ta_riu_v3_align_loss * float(
                getattr(self.opt, "ta_riu_v3_align_weight", 0.01)
            )
        loss_teacher_gs = torch.zeros((), device=joint_loss.device)
        loss_teacher_rgb = torch.zeros((), device=joint_loss.device)
        teacher_eff = float(getattr(self, "teacher_lambda_eff", 0.0))
        if (
            self.training
            and generative_units
            and bool(getattr(self.opt, "gen_teacher_distill", False))
            and teacher_eff > 0.0
        ):
            # Frozen-teacher bootstrap: pin the unit-decoded Gaussians and
            # their rendered RGB to the original (frozen) activation-head
            # output while the units learn to generate; the teacher weight
            # decays linearly to zero (gen_teacher_decay_steps) and the old
            # head path can then be removed.
            w = torch.ones(
                new_gaussians.shape[-1], device=new_gaussians.device
            )
            w[4:7] = 10.0  # scale drift is the most PSNR-sensitive
            w[11:] = 0.0  # color is covered by the RGB distill term
            loss_teacher_gs = (
                (new_gaussians - teacher_gaussians).square()
                * w.view(1, 1, -1)
            ).mean()
            if float(getattr(self.opt, "gen_teacher_rgb_weight", 1.0)) > 0:
                loss_teacher_rgb = (
                    rgb_results["images_pred"].clamp(0, 1)
                    - teacher_rgb.clamp(0, 1)
                ).square().mean()
            joint_loss = joint_loss + teacher_eff * (
                float(getattr(self.opt, "gen_teacher_gs_weight", 1.0))
                * loss_teacher_gs
                + float(getattr(self.opt, "gen_teacher_rgb_weight", 1.0))
                * loss_teacher_rgb
            )
        tsh = bool(
            getattr(self.opt, "abs_true_shared_units", False)
        ) and getattr(self, "tsh_instance_head", None) is not None
        if tsh:
            # True-Shared baseline: the head consumes the SAME q_abs that
            # generated Student GS.  Geometry is detached inside the mask
            # renderer, so instance loss only reaches the unit formation.
            instance_outputs = self._forward_tsh_instance_branch(
                data,
                q_abs_for_instance,
                new_gaussians,
                model_input,
                training=self.training,
                unit_metric_embeddings=(
                    token_eru_dino_metric_outputs.get("unit_metric_embeddings")
                    if token_eru_dino_metric_outputs
                    else None
                ),
            )
            if (
                not self.training
                and token_eru_dino_metric_outputs
                and str(getattr(self.opt, "token_eru_dino_eval_mode", "query"))
                == "metric_cluster"
            ):
                instance_outputs["metric_cluster_output"] = (
                    self.build_token_eru_dino_metric_cluster(
                        token_eru_dino_metric_outputs["unit_metric_embeddings"],
                        instance_outputs["unit_logits"],
                        new_gaussians,
                        model_input,
                    )
                )
            if self.training and instance_outputs:
                head_eff = float(
                    getattr(
                        self, "tsh_instance_loss_weight_eff", 0.0
                    )
                )
                tsh_lambda = float(
                    getattr(self.opt, "tsh_lambda_instance", 0.05)
                )
                joint_loss = joint_loss + instance_outputs[
                    "loss_instance_group"
                ] * (tsh_lambda * head_eff)
                # U->R: official SIU3R mask-guided rendered-depth smoothness
                # (pipeline.py:249-265).  Only the instance-mode switches
                # that include "u2r" enable it; the weight is warmed from 0
                # by compute_tsh_mbm_u2r_eff (trainer sets
                # self.tsh_mbm_u2r_eff each step).
                mbm_mode = str(getattr(self.opt, "tsh_mbm_mode", "off"))
                u2r_eff = float(getattr(self, "tsh_mbm_u2r_eff", 0.0))
                depths_pred = rgb_results.get("depths_pred")
                if (
                    mbm_mode in ("u2r", "both")
                    and u2r_eff > 0.0
                    and depths_pred is not None
                    and "instance_label_output" in data
                ):
                    loss_u2r, u2r_stats = self._tsh_mbm_u2r_loss(
                        depths_pred,
                        instance_outputs[
                            "rendered_instance_group_probability"
                        ],
                        instance_outputs["rendered_instance_group_alpha"],
                    )
                    instance_outputs["loss_mbm_u2r"] = loss_u2r.detach()
                    instance_outputs["tsh_mbm_u2r_eff"] = torch.tensor(
                        u2r_eff, device=joint_loss.device
                    )
                    for key, value in u2r_stats.items():
                        instance_outputs[key] = (
                            value.detach()
                            if torch.is_tensor(value)
                            else value
                        )
                    joint_loss = joint_loss + loss_u2r * u2r_eff
                    self._tsh_last_mbm_u2r_loss = loss_u2r * u2r_eff
                else:
                    instance_outputs["loss_mbm_u2r"] = joint_loss.new_zeros(())
                    instance_outputs["tsh_mbm_u2r_eff"] = torch.tensor(
                        0.0, device=joint_loss.device
                    )
                    self._tsh_last_mbm_u2r_loss = None
        elif abs_mode and self.training:
            guarded = bool(
                getattr(self.opt, "abs_joint_guarded", False)
            )
            instance_stage_eff_now = float(
                getattr(self, "instance_stage_eff", 1.0)
            )
            if guarded:
                # Legacy dual-unit guarded baseline (kept as historical
                # control): student GS detached during warm-up.
                weight_eff = float(
                    getattr(
                        self, "guarded_instance_loss_weight_eff", 0.0
                    )
                )
                unit_eff = float(
                    getattr(self, "guarded_instance_unit_grad_eff", 0.0)
                )
                if unit_eff >= 1.0:
                    ins_gs = new_gaussians
                elif unit_eff <= 0.0:
                    ins_gs = new_gaussians.detach()
                else:
                    ins_gs = (
                        unit_eff * new_gaussians
                        + (1.0 - unit_eff) * new_gaussians.detach()
                    )
                instance_outputs = self.forward_instance_group_branch(
                    data,
                    ins_gs,
                    gs_token_hidden,
                    gaussian_features,
                    semantic_output["semantic_prompts"],
                    model_input,
                )
                lambda_max = float(
                    getattr(
                        self.opt, "guarded_lambda_instance_max", 0.05
                    )
                )
                if instance_outputs:
                    joint_loss = joint_loss + instance_outputs[
                        "loss_instance_group"
                    ] * (lambda_max * weight_eff)
            elif instance_stage_eff_now <= 0.0:
                # Bootstrap / teacher-decay window: instance loss is off and
                # the branch forward is skipped entirely.
                instance_outputs = {}
            else:
                instance_outputs = self.forward_instance_group_branch(
                    data,
                    instance_gaussians,
                    gs_token_hidden,
                    gaussian_features,
                    semantic_output["semantic_prompts"],
                    model_input,
                )
                if instance_outputs:
                    joint_loss = joint_loss + instance_outputs[
                        "loss_instance_group"
                    ] * instance_stage_eff_now
        else:
            # Eval / non-training forward: instance masks are rendered from
            # the absolute student GS (inference only, no grads).
            instance_outputs = self.forward_instance_group_branch(
                data,
                instance_gaussians,
                gs_token_hidden,
                gaussian_features,
                semantic_output["semantic_prompts"],
                model_input,
            )

        if abs_mode and teacher_on:
            # Absolute-student teacher distill: per-token Hungarian-paired
            # GS distillation (low weight) + rendered-RGB bootstrap.
            abs_gs_w = float(
                getattr(self.opt, "abs_teacher_gs_weight", 1.0)
            )
            abs_rgb_w = float(
                getattr(self.opt, "abs_teacher_rgb_weight", 1.0)
            )
            loss_teacher_gs, loss_teacher_rgb, _ = (
                teacher_gs_distill_loss(
                    teacher_gaussians,
                    new_gaussians,
                    teacher_rgb,
                    rgb_results["images_pred"],
                    rgb_weight=abs_rgb_w,
                )
            )
            joint_loss = joint_loss + float(
                getattr(self, "teacher_lambda_eff", 0.0)
            ) * (abs_gs_w * loss_teacher_gs + abs_rgb_w * loss_teacher_rgb)

        loss_semantic_lifting = torch.zeros(
            (), device=joint_loss.device
        )
        if (
            self.training
            and self.semantic_lifting_head is not None
            and float(getattr(self.opt, "lambda_semantic_feature", 0.0)) > 0
        ):
            loss_semantic_lifting = self.compute_semantic_lifting_loss(
                data,
                gaussians,
                gs_token_hidden,
                model_input,
                bg_color=reconstruction.background_color,
            )
            joint_loss = joint_loss + loss_semantic_lifting
        loss_semantic_v3 = torch.zeros((), device=joint_loss.device)
        if (
            self.training
            and self.semantic_head is not None
            and float(getattr(self.opt, "lambda_semantic_feature", 0.0)) > 0
        ):
            loss_semantic_v3 = self.compute_semantic_v3_loss(
                data,
                gaussians,
                gs_token_hidden,
                model_input,
                bg_color=reconstruction.background_color,
            )
            joint_loss = joint_loss + loss_semantic_v3
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
            "loss_boundary_rgb": loss_boundary_rgb,
            "loss_feat": loss_feat,
            "loss_ce_cosine": loss_ce_cosine,
            "loss_instance_contrastive": loss_instance_contrastive,
            "loss_semantic_lifting": loss_semantic_lifting,
            "loss_semantic_v3": loss_semantic_v3,
            "loss_ta_riu_v2_unit_embedding": ta_riu_v2_loss,
            "loss_ta_riu_v3_align_raw": ta_riu_v3_align_raw,
            "loss_ta_riu_v3_align": ta_riu_v3_align_loss * float(
                getattr(self.opt, "ta_riu_v3_align_weight", 0.01)
            ),
            "loss_teacher_gs": loss_teacher_gs,
            "loss_teacher_rgb": loss_teacher_rgb,
            "teacher_lambda_eff": torch.tensor(
                teacher_eff, device=joint_loss.device
            ),
            "guarded_instance_loss_weight": torch.tensor(
                (
                    float(getattr(self.opt, "guarded_lambda_instance_max", 0.05))
                    * float(
                        getattr(
                            self, "guarded_instance_loss_weight_eff", 0.0
                        )
                    )
                    if bool(
                        getattr(self.opt, "abs_joint_guarded", False)
                    )
                    else 0.0
                ),
                device=joint_loss.device,
            ),
            "guarded_instance_unit_grad_eff": torch.tensor(
                float(
                    getattr(self, "guarded_instance_unit_grad_eff", 0.0)
                ),
                device=joint_loss.device,
            ),
            "tsh_instance_loss_weight": torch.tensor(
                (
                    float(getattr(self.opt, "tsh_lambda_instance", 0.05))
                    * float(
                        getattr(
                            self, "tsh_instance_loss_weight_eff", 0.0
                        )
                    )
                    if bool(
                        getattr(self.opt, "abs_true_shared_units", False)
                    )
                    else 0.0
                ),
                device=joint_loss.device,
            ),
            "tsh_unit_grad_eff": torch.tensor(
                float(getattr(self, "tsh_unit_grad_eff", 0.0)),
                device=joint_loss.device,
            ),
            **semantic_metrics,
            **diagnostics,
            **rgb_results,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "images_pred": rgb_results["images_pred"],
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
            "rendered_alpha": rendered_alpha,
            "valid_mask": valid_mask,
            "target_prompt_mask": target_mask,
            "semantic_class_names": C3G8_CLASS_NAMES,
            **(
                {
                    "ta_riu_q_abs_base": ta_riu_q_abs.detach(),
                    "ta_riu_z_shared": ta_riu_z_shared.detach(),
                    "ta_riu_memory": ta_riu_memory.detach(),
                    "ta_riu_delta": ta_riu_delta.detach(),
                    "ta_riu_geo_raw": ta_riu_geo_raw.detach(),
                    "ta_riu_app_raw": ta_riu_app_raw.detach(),
                    "ta_riu_base_gaussians": ta_riu_base_gaussians.detach(),
                    "ta_riu_joint_gaussians": ta_riu_joint_gaussians.detach(),
                    "ta_riu_gate": torch.tensor(
                        float(
                            getattr(
                                self,
                                "ta_riu_gate_eff",
                                getattr(self, "ta_riu_eval_gate_override", 1.0),
                            )
                        ),
                        device=joint_loss.device,
                    ),
                    "ta_riu_geo_gate": torch.tensor(
                        float(
                            getattr(
                                self,
                                "ta_riu_geo_gate_eff",
                                getattr(self, "ta_riu_eval_geo_gate_override", 1.0),
                            )
                        ),
                        device=joint_loss.device,
                    ),
                    "ta_riu_app_gate": torch.tensor(
                        float(
                            getattr(
                                self,
                                "ta_riu_app_gate_eff",
                                getattr(self, "ta_riu_eval_app_gate_override", 1.0),
                            )
                        ),
                        device=joint_loss.device,
                    ),
                }
                if ta_riu_enabled
                else {}
            ),
            **token_eru_dino_metric_outputs,
            **token_eru_3d_anchor_outputs,
            # The eval-only GT-free cluster object is added to
            # instance_outputs after the metric branch returns.  Keep this
            # merge order so the metric branch's placeholder None cannot
            # overwrite the actual rendered cluster output.
            **instance_outputs,
            **(
                {
                    "ta_riu_v3_reconstruction_units": ta_riu_v3_outputs.reconstruction_units.detach(),
                    "ta_riu_v3_instance_units": ta_riu_v3_outputs.instance_units.detach(),
                    "ta_riu_v3_joint_reconstruction_units": ta_riu_v3_outputs.joint_reconstruction_units.detach(),
                    "ta_riu_v3_joint_instance_units": ta_riu_v3_outputs.joint_instance_units.detach(),
                    "ta_riu_v3_dino_memory": ta_riu_v3_outputs.dino_memory.detach(),
                    "ta_riu_v3_reconstruction_delta": ta_riu_v3_outputs.reconstruction_delta.detach(),
                    "ta_riu_v3_instance_delta": ta_riu_v3_outputs.instance_delta.detach(),
                    "ta_riu_v3_gate": torch.tensor(
                        float(
                            getattr(
                                self,
                                "ta_riu_v3_gate_eff",
                                getattr(self, "ta_riu_v3_eval_gate_override", 1.0),
                            )
                        ),
                        device=joint_loss.device,
                    ),
                }
                if ta_riu_v3_outputs is not None
                and bool(getattr(self, "ta_riu_v3_return_debug", False))
                else {}
            ),
            **(
                {
                    f"ta_riu_v2_{key}": value.detach()
                    if torch.is_tensor(value) else value
                    for key, value in ta_riu_v2_outputs.items()
                    if key != "z_inst"
                    and key != "delta"
                }
                if ta_riu_v2_outputs is not None
                and bool(getattr(self, "ta_riu_v2_return_debug", False))
                else {}
            ),
        }
