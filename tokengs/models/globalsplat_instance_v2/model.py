from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokengs.models.input_types import split_data

from .camera_adapter import (make_official_context_frustum_meta,
                              make_official_context_input,
                              make_official_target_meta,
                              split_eight_context_anchor_alternating)
from .candidate_layout import capture_candidate_layout
from .dependency import load_globalsplat_symbols, load_official_state_strict
from .instance_head import SceneGlobalInstanceHead
from .renderer import normalize_rendered_assignment, render_feature_channels_sh_geometry, render_rgb_sh
from .reconstruction_loss import GSIReconstructionLoss
from .scene_instance_loss import scene_global_hungarian_instance_loss
from .tri_stream import build_tri_stream_slot_encoder
from .types import GSIModelOutput


class GlobalSplatInstanceV2(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        if getattr(opt, "model_type", None) != "globalsplat_instance_v2":
            raise ValueError("GlobalSplatInstanceV2 requires model_type=globalsplat_instance_v2")
        self.opt = opt
        self.phase = str(opt.globalsplat_instance_v2_phase)
        if self.phase not in ("reconstruction", "joint"):
            raise ValueError(f"invalid GSI phase {self.phase}")
        self.symbols = load_globalsplat_symbols(opt.gsi_v2_globalsplat_repo, opt.gsi_v2_globalsplat_commit)
        self.reconstruction = self.symbols.GlobalSplat(
            sh_degree=3, static_only=True, use_camera_diff_as_input=False,
            patch_size=8, latent_rep_token_amount=2048, dim_latents=512,
            dim_rays=256, dim_rgb_feat=512, rounds=4,
            slot_calib_layers_per_round=2, num_heads=8, M_max=16,
        )
        self._official_checkpoint_report = None
        mode = str(getattr(opt, "gsi_v2_resume_mode", "official"))
        if mode == "official":
            self._official_checkpoint_report = load_official_state_strict(
                self.reconstruction, opt.gsi_v2_official_checkpoint, opt.gsi_v2_official_checkpoint_sha256
            )
        self.reconstruction.set_stage(3, mix=1.0)
        self.instance_head = None
        if self.phase == "joint":
            dual_state = dict(self.reconstruction.slot_encoder.state_dict())
            tri = build_tri_stream_slot_encoder(
                self.symbols,
                dual_state,
                initialize_instance_from_geometry=(
                    str(getattr(opt, "gsi_v2_joint_instance_stream_init", "copy_geometry"))
                    == "copy_geometry"
                ),
            )
            self.reconstruction.slot_encoder = tri
            self.instance_head = SceneGlobalInstanceHead(
                slot_dim=512, embedding_dim=64, num_queries=100, max_candidates=16,
                query_layers=2, query_heads=8, temperature_init=10.0,
            )
        self.reconstruction_loss = GSIReconstructionLoss(
            symbols=self.symbols, mode=opt.gsi_v2_recon_loss_mode,
            vgg_weight_path=opt.gsi_v2_vgg_weight_path,
        )
        self._train_step = 0
        self._gate = 0.0
        self._gate_gradient_scale = 0.0
        self._instance_weight = 0.0
        self._set_requires_grad()

    @property
    def instance_enabled(self) -> bool:
        return self.instance_head is not None

    def _set_requires_grad(self) -> None:
        for parameter in self.reconstruction.parameters():
            parameter.requires_grad_(True)
        if self.instance_head is not None:
            for parameter in self.instance_head.parameters():
                parameter.requires_grad_(True)
        if self.reconstruction_loss.perceptual_loss is not None:
            for parameter in self.reconstruction_loss.perceptual_loss.parameters():
                parameter.requires_grad_(False)

    def load_initial_state(self) -> dict[str, object]:
        report = load_official_state_strict(self.reconstruction, self.opt.gsi_v2_official_checkpoint, self.opt.gsi_v2_official_checkpoint_sha256)
        self._official_checkpoint_report = report
        return {"path": report.path, "sha256": report.sha256, "keys": f"{report.loaded_tensor_count}/{report.checkpoint_tensor_count}", "numel": f"{report.loaded_state_numel}/{report.checkpoint_state_numel}"}

    def load_phase_r_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, object]:
        if self.phase != "joint":
            raise RuntimeError("Phase R -> J loading requires phase=joint")
        current = nn.Module.state_dict(self)
        tri_only = ("reconstruction.slot_encoder.slot_to_ins.", "reconstruction.slot_encoder.ins_rounds.", "reconstruction.slot_encoder.tri_adapters.")
        recon_keys = [key for key in current if key.startswith("reconstruction.") and not key.startswith(tri_only)]
        missing = [key for key in recon_keys if key not in state_dict]
        unexpected = [key for key in state_dict if key.startswith("reconstruction.") and key not in current]
        mismatch = [key for key in recon_keys if key in state_dict and current[key].shape != state_dict[key].shape]
        if missing or unexpected or mismatch:
            raise RuntimeError(f"Phase R restore failed: missing={missing[:8]} unexpected={unexpected[:8]} mismatch={mismatch[:8]}")
        nn.Module.load_state_dict(self, {key: value for key, value in state_dict.items() if key in current and not key.startswith(tri_only)}, strict=False)
        if str(getattr(self.opt, "gsi_v2_joint_instance_stream_init", "copy_geometry")) == "copy_geometry":
            self.reconstruction.slot_encoder.slot_to_ins = copy.deepcopy(self.reconstruction.slot_encoder.slot_to_geo)
            self.reconstruction.slot_encoder.ins_rounds = copy.deepcopy(self.reconstruction.slot_encoder.geo_rounds)
        return {"reconstruction_keys": len(recon_keys), "loaded_reconstruction_keys": len(recon_keys), "missing": (), "unexpected": (), "shape_mismatch": ()}

    def set_train_step(self, optimizer_step: int) -> None:
        self._train_step = int(optimizer_step)
        if self.phase == "reconstruction":
            self._gate = 0.0
            self._gate_gradient_scale = 0.0
            self._instance_weight = 0.0
        else:
            instance_warmup = max(1, int(getattr(self.opt, "gsi_v2_instance_loss_warmup_steps", 500)))
            injection_start = int(getattr(self.opt, "gsi_v2_instance_to_reconstruction_start_step", 500))
            injection_ramp = max(1, int(getattr(self.opt, "gsi_v2_instance_to_reconstruction_ramp_steps", 500)))
            injection_max = float(getattr(self.opt, "gsi_v2_instance_to_reconstruction_max", 0.1))
            self._gate = (
                0.0 if optimizer_step < injection_start
                else min(1.0, (optimizer_step - injection_start) / injection_ramp)
            )
            self._gate_gradient_scale = injection_max * self._gate
            self._instance_weight = min(1.0, max(0.0, optimizer_step / instance_warmup))

    def set_eval_stage(self) -> None:
        self.reconstruction.set_stage(3, mix=1.0)

    def reconstruction_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(f"reconstruction.{name}", parameter) for name, parameter in self.reconstruction.named_parameters()]

    def instance_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        if self.instance_head is None:
            return []
        return [(f"reconstruction.slot_encoder.{name}", parameter) for name, parameter in self.reconstruction.slot_encoder.named_parameters() if name.startswith(("slot_to_ins.", "ins_rounds.", "tri_adapters."))] + [(f"instance_head.{name}", parameter) for name, parameter in self.instance_head.named_parameters()]

    def _official_gaussians(self, model_input, context):
        tokens = self.reconstruction._tokenize(context["images"], context["intrinsic"], context["c2w"])
        state = self.reconstruction.scene_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        if self.phase == "joint":
            tri = self.reconstruction.slot_encoder.forward_joint(
                tokens, state, recon_to_instance_grad_scale=float(getattr(self.opt, "gsi_v2_recon_to_instance_grad_scale", 0.1)),
                instance_to_reconstruction_gate=self._gate,
                camera_motion_tokens=None, mem_drop_p=0.0,
            )
            scene_tex, scene_geo, scene_ins = tri.scene_tex, tri.scene_geo, tri.scene_ins
        else:
            _aux, encoded, _scale = self.reconstruction.slot_encoder(tokens, state=state)
            scene_tex, scene_geo = encoded
            scene_ins = None
        gaussians_tuple = self.reconstruction.gaussian_decoder((scene_tex, scene_geo))
        gaussians = self.symbols.Gaussians(
            means=gaussians_tuple[0], rotations=gaussians_tuple[1], scales=gaussians_tuple[2],
            sh=gaussians_tuple[3], opacities=gaussians_tuple[4], reg=gaussians_tuple[5],
        )
        return gaussians, scene_geo, scene_ins

    def forward_core(self, data: dict, *, compute_instance: bool,
                     context_override=None, target_meta_override=None) -> GSIModelOutput:
        model_input, _supervision = split_data(data, self.opt)
        context = make_official_context_input(model_input) if context_override is None else context_override
        gaussians, scene_geo, scene_ins = self._official_gaussians(model_input, context)
        layout = capture_candidate_layout(self.reconstruction.gaussian_decoder, scene_geo)
        target_meta = (
            make_official_target_meta(model_input, (int(self.opt.img_size[0]), int(self.opt.img_size[1])))
            if target_meta_override is None else target_meta_override
        )
        rgb = render_rgb_sh(self.symbols, gaussians, target_meta, render_depth=True)
        instance = None
        rendered_assignment = None
        if compute_instance:
            if self.instance_head is None or scene_ins is None:
                raise RuntimeError("instance forward requested in reconstruction phase")
            instance = self.instance_head(scene_ins, layout, gate_gradient_scale=self._gate_gradient_scale)
            # Render all 101 assignment channels in one pass: 100 foreground
            # queries plus the head's channel-100 void probability.  The
            # normalizer preserves that void channel and only overrides it at
            # pixels with no accumulated Gaussian alpha.
            # The instance rasterizer has a direct derivative through Gaussian
            # projection/alpha, in addition to the explicit tri-stream return
            # path.  Keep that path fully detached during the instance-only
            # warm-up (gate=0), while preserving the exact forward values; once
            # the configured return gate opens, the original geometry gradient
            # path is enabled again.
            instance_gaussians = gaussians
            if float(self._gate) == 0.0:
                instance_gaussians = replace(
                    gaussians,
                    means=gaussians.means.detach(),
                    scales=gaussians.scales.detach(),
                    rotations=gaussians.rotations.detach(),
                    opacities=gaussians.opacities.detach(),
                )
            composed, alpha = render_feature_channels_sh_geometry(
                self.symbols, instance_gaussians, instance.assignment_probabilities, target_meta, packed=False
            )
            rendered_assignment = normalize_rendered_assignment(composed, alpha)
        return GSIModelOutput(gaussians, rgb["images_pred"], rgb["alphas_pred"], rgb["depths_pred"], layout, instance, rendered_assignment)

    def forward(self, data: dict, skip_loss: bool = False) -> dict[str, torch.Tensor]:
        compute_instance = self.instance_enabled
        model_input, supervision = split_data(data, self.opt)
        full_context = make_official_context_input(model_input)
        target_meta = make_official_target_meta(
            model_input, (int(self.opt.img_size[0]), int(self.opt.img_size[1]))
        )
        use_subset = bool(
            self.training and self.phase == "reconstruction"
            and getattr(self.opt, "gsi_v2_subset_consistency", False)
        )
        subset_outputs = None
        if use_subset:
            context_a, context_b = split_eight_context_anchor_alternating(full_context, False)
            output_a = self.forward_core(
                data, compute_instance=False, context_override=context_a,
                target_meta_override=target_meta,
            )
            output_b = self.forward_core(
                data, compute_instance=False, context_override=context_b,
                target_meta_override=target_meta,
            )
            output = output_a
            subset_outputs = (output_a, output_b)
        else:
            output = self.forward_core(
                data, compute_instance=compute_instance,
                context_override=full_context, target_meta_override=target_meta,
            )
        pred = output.rendered_rgb
        zero = pred.sum() * 0.0
        if skip_loss:
            loss = zero
            rgb_loss = zero
            instance_loss = zero
            stats = {}
        else:
            context_K, context_w2c = make_official_context_frustum_meta(model_input)
            target_rgb = supervision.images_output.reshape_as(pred)
            if subset_outputs is None:
                reconstruction_total, loss_stats, _cache = self.reconstruction_loss(
                    output.gaussians, pred, target_rgb, context_K, context_w2c
                )
            else:
                output_a, output_b = subset_outputs
                context_a, context_b = split_eight_context_anchor_alternating(full_context, False)
                loss_a, stats_a, target_cache = self.reconstruction_loss(
                    output_a.gaussians, output_a.rendered_rgb, target_rgb,
                    context_a["intrinsic"], torch.linalg.inv(context_a["c2w"]),
                )
                loss_b, stats_b, _ = self.reconstruction_loss(
                    output_b.gaussians, output_b.rendered_rgb, target_rgb,
                    context_b["intrinsic"], torch.linalg.inv(context_b["c2w"]),
                    target_cache=target_cache,
                )
                alpha_a, alpha_b = output_a.rendered_alpha, output_b.rendered_alpha
                depth_a, depth_b = output_a.rendered_depth, output_b.rendered_depth
                subset_alpha = 0.5 * (
                    (alpha_a - alpha_b.detach()).abs().mean()
                    + (alpha_b - alpha_a.detach()).abs().mean()
                )
                valid = (alpha_a > 1e-2) & (alpha_b > 1e-2)
                if depth_a is None or depth_b is None or not valid.any():
                    subset_depth = pred.new_zeros(())
                else:
                    count = valid.float().sum().clamp_min(1.0)
                    subset_depth = (
                        ((depth_a - depth_b.detach()).abs() * valid.float()).sum()
                        + ((depth_b - depth_a.detach()).abs() * valid.float()).sum()
                    ) / (2.0 * count)
                reconstruction_total = (
                    0.5 * (loss_a + loss_b) + 1e-3 * subset_alpha
                    + 1e-2 * subset_depth
                )
                loss_stats = {
                    "loss_rgb": 0.5 * (stats_a["loss_rgb"] + stats_b["loss_rgb"]),
                    "loss_mse": 0.5 * (stats_a["loss_mse"] + stats_b["loss_mse"]),
                    "loss_perceptual": 0.5 * (stats_a["loss_perceptual"] + stats_b["loss_perceptual"]),
                    "loss_inview": 0.5 * (stats_a["loss_inview"] + stats_b["loss_inview"]),
                    "loss_decoder_reg": 0.5 * (stats_a["loss_decoder_reg"] + stats_b["loss_decoder_reg"]),
                    "gsi_v2_subset_alpha": subset_alpha.detach(),
                    "gsi_v2_subset_depth": subset_depth.detach(),
                    "gsi_v2_subset_alpha_finite": torch.isfinite(subset_alpha.detach()).float(),
                    "gsi_v2_subset_depth_finite": torch.isfinite(subset_depth.detach()).float(),
                }
            loss = reconstruction_total
            rgb_loss = loss_stats["loss_rgb"]
            instance_loss = zero
            stats = loss_stats
            if compute_instance and output.rendered_assignment is not None and "instance_label_output" in data:
                instance_loss, instance_stats, _ = scene_global_hungarian_instance_loss(
                    output.rendered_assignment, data["instance_label_output"].long(),
                    num_queries=100, min_visible_pixels=int(self.opt.gsi_v2_min_visible_pixels),
                    bce_weight=float(self.opt.gsi_v2_match_bce_weight), dice_weight=float(self.opt.gsi_v2_match_dice_weight),
                    void_weight=float(self.opt.gsi_v2_void_weight), unmatched_weight=float(self.opt.gsi_v2_unmatched_weight),
                    absent_view_weight=float(self.opt.gsi_v2_absent_view_weight),
                )
                instance_total = float(getattr(self, "_instance_weight", 1.0)) * float(self.opt.gsi_v2_instance_loss_weight) * instance_loss
                loss = loss + instance_total
                stats.update(instance_stats)
            else:
                instance_total = zero
        result = {
            "loss": loss, "loss_rgb": rgb_loss, "loss_instance_group": instance_loss,
            "loss_reconstruction": reconstruction_total if not skip_loss else zero,
            "psnr": (-10.0 * torch.log10(torch.clamp(F.mse_loss(pred.float(), supervision.images_output.float()), min=1e-8))) if not skip_loss else zero,
            "images_pred": pred, "alphas_pred": output.rendered_alpha, "depths_pred": output.rendered_depth,
            "means": output.gaussians.means, "scales": output.gaussians.scales,
            "rotations": output.gaussians.rotations, "sh": output.gaussians.sh,
            "opacities": output.gaussians.opacities,
        }
        if compute_instance and output.instance is not None and output.rendered_assignment is not None:
            result.update({
                "gaussian_instance_embeddings": output.instance.gaussian_embeddings,
                "gaussian_objectness_logits": output.instance.gaussian_objectness_logits,
                "gaussian_group_probabilities": output.instance.assignment_probabilities,
                "rendered_instance_group_probability": output.rendered_assignment,
                "rendered_instance_group_alpha": output.rendered_assignment[:, :1].sum(dim=1),
                "loss_instance_group": instance_loss,
                "tsh_instance_loss_weight": torch.as_tensor(getattr(self, "_instance_weight", 1.0), device=pred.device),
            })
            # Diagnostics may request the pre-render tensors needed to audit
            # visible-unit/assignment changes.  Keep these opt-in so the
            # normal forward/checkpoint interface and memory footprint are
            # unchanged for formal training and evaluation.
            if bool(getattr(self.opt, "gsi_v2_return_debug_tensors", False)):
                result.update({
                    "gsi_v2_instance_assignment_logits": output.instance.assignment_logits,
                    "gsi_v2_instance_assignment_probabilities": output.instance.assignment_probabilities,
                    "gsi_v2_candidate_gate_logits": output.layout.gate_logits_full,
                })
        result.update(stats)
        return result

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {key: value for key, value in state.items() if not key.startswith("reconstruction_loss.perceptual_loss.")}

    def load_state_dict(self, state_dict, strict: bool = True, *args, **kwargs):
        incoming = dict(state_dict)
        prefix = "reconstruction_loss.perceptual_loss."
        if self.reconstruction_loss.perceptual_loss is not None:
            own = nn.Module.state_dict(self)
            for key, value in own.items():
                if key.startswith(prefix) and key not in incoming:
                    incoming[key] = value
        return super().load_state_dict(incoming, strict=strict, *args, **kwargs)

    def lineage_metadata(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "official_globalsplat_commit": self.opt.gsi_v2_globalsplat_commit,
            "official_checkpoint_sha256": self.opt.gsi_v2_official_checkpoint_sha256,
            "sh_degree": 3, "scene_tokens": 2048, "candidate_count": 16384,
            "model_type": "globalsplat_instance_v2",
        }
