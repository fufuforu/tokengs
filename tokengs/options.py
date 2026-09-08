# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tyro CLI options and named presets (`AllConfigs` subcommands)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

import tyro


@dataclass
class Options:
    # --- general
    evaluating: bool = False
    workspace: str = "./workspace"
    resume: str | None = None
    model_type: str = "tokengs"
    seed: int = 42

    # --- wandb / logging
    use_wandb: bool = False
    experiment_name: str = "tokengs"
    out_dir: str = "outputs"
    project_name: str = "TokenGS"

    # --- model architecture
    img_size: tuple[int, int] = (256, 256)
    patch_size: int = 8
    dec_patch_size: int | None = None
    enc_depth: int = 3
    dec_depth: int = 12
    enc_embed_dim: int = 1024
    enc_num_heads: int = 16
    mlp_ratio: int = 4
    clip_head_readout_std: float = 0.002
    clip_head_z_init: float | None = None
    dec_init_values: float | None = None

    # --- gaussian splatting
    bg_color: Literal["white", "black", "grey"] = "grey"
    gaussian_scale_cap: float = 0.075
    opacity_bias: float = 2.0
    gaussian_z_offset: float = 1.0
    num_gs_tokens: int = 1024
    token_dim: int = 1024
    gs_token_std: float = 1e-2
    # --- GS token initialization (experiment: scene-adaptive 3D anchoring)
    gs_token_init: Literal["learnable", "anchor_3d"] = "learnable"
    anchor_3d_query_source: Literal["hidden", "global"] = "hidden"
    anchor_3d_pos_scale: float = 0.1
    anchor_3d_pos_freqs: int = 4
    anchor_3d_pass_decoder: bool = False
    # --- Experiment B: group-conditioned Gaussian generation ---
    # GroupConditionedGaussianGenerator: group tokens participate in decoding
    # BEFORE the Gaussian activation head (token rewrite) and produce explicit
    # bounded per-anchor geometry/opacity offsets, so instance identity can
    # change Gaussian parameters. Distinct from the old post-hoc mask heads.
    instance_group_condition_generator: bool = False
    instance_group_condition_geometry_scale: float = 0.05
    # --- Experiment C: fully independent instance-structured branch ---
    # Frozen TokenGS RGB reconstruction (encoder + decoder + activation head)
    # stays untouched. A NEW scene-adaptive anchor + 100 group tokens +
    # independent Gaussian decoder generates its OWN Gaussians for instance
    # masks; instance loss never backpropagates into the RGB branch.
    instance_branch_independent: bool = False
    instance_branch_num_groups: int = 100
    instance_branch_anchor_dim: int = 256
    instance_branch_num_heads: int = 8
    instance_branch_num_layers: int = 2
    instance_branch_gaussians_per_anchor: int = 16
    instance_branch_pos_offset_scale: float = 0.2
    instance_branch_scale_delta_amp: float = 0.5
    instance_branch_opacity_delta_amp: float = 0.3
    instance_branch_rgb_delta_amp: float = 0.5
    # --- Experiment D: Token -> local spatial units -> Group ---
    # Learn K=8 3D-local units inside each 64-GS token (soft k-means over GS
    # features + geometry), then 100 group tokens assign units; GS inherit
    # their unit's group. Frozen geometry renders the masks; only unit
    # formation + group tokens + assignment train.
    instance_branch_token_units: bool = False
    instance_branch_units_per_token: int = 8
    instance_branch_unit_feat_dim: int = 128
    instance_branch_unit_layers: int = 2
    instance_branch_unit_temp: float = 5.0
    instance_branch_unit_entropy: float = 0.05
    instance_branch_unit_compactness: float = 0.05
    instance_branch_unit_purity: bool = False
    instance_branch_unit_purity_weight: float = 0.1
    instance_branch_gs_refine: bool = False
    instance_branch_gs_refine_scale: float = 0.1
    instance_branch_gs_refine_dim: int = 64
    # --- Scene-specific dynamic prototypes / slots (replaces global Group
    # Tokens for final instance identity). Unit formation stays frozen; the
    # prototypes are formed per scene from the 8192 local units (unit
    # feature + 3D center) via iterative soft k-means-style slot grouping.
    instance_branch_scene_prototypes: bool = False
    instance_branch_num_slots: int = 100
    instance_branch_slot_dim: int = 128
    instance_branch_slot_iterations: int = 3
    instance_branch_slot_temp: float = 5.0
    instance_branch_unit_resume: str = ""
    # --- Instance-aware Local Unit Embedding + deterministic clustering ---
    # Learn a 128-D normalized unit identity embedding with a soft InfoNCE
    # (distribution-level, permutation-invariant). Inference groups scene
    # units by agglomerative clustering on [embedding, pos] with a fixed
    # distance threshold (no global queries, no iterative slots).
    instance_branch_unit_embedding: bool = False
    instance_branch_unit_encoder: bool = False
    instance_branch_embed_dim: int = 128
    instance_branch_embed_temp: float = 0.1
    instance_branch_embed_loss: float = 1.0
    instance_branch_embed_loss_mode: str = "info_nce"
    # Sub-sample units for the pairwise identity losses (InfoNCE etc.) when
    # units_per_token is large (e.g. K=64 -> 65536 units/scene), where the
    # full UxU similarity matrix does not fit.  0 = no sampling (K=8 path).
    instance_branch_embed_sample: int = 0
    instance_branch_embed_proto_temp: float = 0.1
    instance_branch_embed_margin: float = 0.2
    instance_branch_embed_margin_weight: float = 1.0
    # InstanceSplat-inspired prototype-level push: separates soft instance
    # centers (built from the detached pseudo-instance distribution) with a
    # cosine hinge, WITHOUT pushing individual units away from other
    # instances.  Only active when embed_loss_mode == "info_nce_center_push".
    instance_branch_embed_center_push_margin: float = 0.2
    instance_branch_embed_center_push_weight: float = 0.1
    instance_branch_unit_image: bool = False
    instance_branch_unit_image_dim: int = 64
    # DINOv2 dense features projected onto GS centers and aggregated per
    # local unit (frozen extractor + small trainable projection).  When
    # enabled, the unit identity embedding becomes
    # normalize(concat(unit_feat, unit_img, proj(dino_unit_feat))).
    instance_branch_unit_dino: bool = False
    instance_branch_unit_dino_dim: int = 64
    # GradScale gates on the shared gs_token_hidden latent: semantic and
    # instance branches attach through a forward-identity / backward-scaled
    # gate so their gradients into the shared representation are muted
    # relative to the reconstruction path (alpha=1).  Prevents PSNR collapse
    # during multi-task joint fine-tuning while still letting instance /
    # semantic supervision shape the shared latent.
    grad_scale_sem: float = 0.3
    grad_scale_ins: float = 0.1
    # Generative local units: re-route Token -> 64 GS through
    # Token -> K local units -> unit-decoded GS (zero-init residual on the
    # frozen per-GS parameters).  The units then become the intermediate
    # representation jointly optimized by reconstruction (RGB through the
    # decoded GS) and instance supervision (InfoNCE on unit features), which
    # is the structural fix for the joint-training conflict.
    generative_units: bool = False
    # Teacher bootstrap for generative units (v3): the OLD frozen TokenGS
    # Gaussian head acts only as a frozen teacher during the early
    # reconstruction bootstrap.  A weighted Gaussian-parameter + rendered-RGB
    # distillation loss anchors the unit-decoded GS to the teacher while
    # the units learn to generate, then the teacher weight decays to zero
    # (after ``gen_teacher_decay_steps``) so the old head path can be
    # removed entirely.
    gen_teacher_distill: bool = False
    gen_teacher_decay_steps: int = 1500
    gen_teacher_gs_weight: float = 1.0
    gen_teacher_rgb_weight: float = 1.0
    # Absolute token-aligned student (no residual / no old-head runtime
    # dependence).  Stage schedule over the training run:
    #   0 .. abs_bootstrap_steps          : reconstruction bootstrap only
    #                                       (teacher GS/RGB distill, instance
    #                                       loss off)
    #   abs_bootstrap_steps .. +abs_teacher_decay_steps
    #                                       : teacher weight decays to 0 and
    #                                         the teacher forward is skipped
    #   then instance loss warms in over abs_instance_warmup_steps
    instance_branch_abs_units: bool = False
    abs_bootstrap_steps: int = 600
    abs_teacher_decay_steps: int = 400
    abs_instance_warmup_steps: int = 800
    abs_teacher_gs_weight: float = 1.0
    abs_teacher_rgb_weight: float = 1.0
    # Reconstruction-only full-data run: also hard-freeze the instance branch
    # (instance is never enabled) and save head checkpoints every
    # ``abs_ckpt_every`` steps inside each epoch (0 = epoch cadence only).
    abs_freeze_instance: bool = False
    abs_ckpt_every: int = 0
    # Guarded joint baseline (independent experiment, no SIU3R / semantic /
    # DINO / LSeg additions): instance supervision enters the absolute
    # Shared Local Units while the reconstruction guardrails stay on.
    #  * stage 1 [0, guarded_instance_warmup_steps): instance head trains on
    #    detached student GS (no instance grads into absolute units);
    #  * stage 2 [warmup, guarded_instance_full_joint_steps): instance->unit
    #    gradients and the effective instance loss weight ramp linearly;
    #  * stage 3 [full_joint_steps, end): full joint training.
    # ``guarded_instance_head_reinit`` makes resume load ONLY absolute_gs_head
    # from the previous reconstruction checkpoint (instance branch starts
    # from its fresh initialization).
    abs_joint_guarded: bool = False
    guarded_instance_head_reinit: bool = False
    guarded_instance_warmup_steps: int = 1000
    guarded_instance_full_joint_steps: int = 5680
    guarded_lambda_instance_max: float = 0.05
    guarded_abs_lr: float = 1.0e-5
    guarded_instance_lr: float = 1.0e-4
    # True-Shared guarded joint baseline: the ONLY unit formation is
    # AbsoluteUnitDecoder's q_abs, consumed by both the GS decoder and the
    # instance assignment head.  The legacy dual-unit instance branch is not
    # instantiated here.
    abs_true_shared_units: bool = False
    tsh_instance_warmup_steps: int = 1000
    tsh_instance_ramp_end_steps: int = 5680
    tsh_lambda_instance: float = 0.05
    tsh_unit_gradient_multiplier_max: float = 32.0
    tsh_abs_lr: float = 1.0e-5
    tsh_instance_lr: float = 1.0e-4
    tsh_num_groups: int = 100
    tsh_num_heads: int = 8
    tsh_num_layers: int = 2
    # True-Shared DDP8 variant marker: stage thresholds are expressed in
    # DDP optimizer steps (125 / 710 / ...) and extra metadata is written.
    tsh_ddp8: bool = False
    # Causal-fork continuation: when resuming a common warm-up checkpoint at
    # this optimizer step, continue inside epoch 0 from the given step
    # (same optimizer/scheduler state, no re-init, no schedule recompute).
    tsh_fork_continue_step: int = 0
    # SIU3R-style Mutual Benefit extension on top of the True-Shared
    # baseline.  The official SIU3R repository implements the R->U
    # "Multi-View Mask Aggregation" as an inference-time 2D-mask -> per-GS
    # logit lift -> Gaussian splat re-render (no trainable parameters, no
    # training loss).  In TokenGS the same aggregation is already structural:
    # one unit-level assignment is expanded to its fixed 8 GS and
    # alpha-blended into every view by the shared Student GS, so novel-view
    # masks are view-consistent by construction and no render->lift->render
    # loop is added.  "r2u" therefore only gates the existing unit-level
    # evidence path / unit gradient multiplier, while "u2r" adds the one
    # official mechanism that is missing from the baseline: the
    # mask-guided rendered-depth smoothness loss
    #   L = weight * ( mean(|dD/dw| * inside_w) + mean(|dD/dh| * inside_h) )
    # with ``inside`` derived from the detached predicted instance ids of the
    # same Student-GS render (identical to SIU3R pipeline.py lines 249-265).
    # Mode choices: "off" | "r2u" | "u2r" | "both".
    tsh_mbm_mode: str = "off"
    # Official weight_depth_smoothness = 0.05 (configs/main.yaml line 56).
    tsh_mbm_u2r_weight: float = 0.0
    # Warm-up in optimizer steps: effective weight is 0 below start and
    # linearly ramps 0 -> 1 over ``tsh_mbm_u2r_warmup_steps``.
    tsh_mbm_u2r_warmup_start_step: int = 0
    tsh_mbm_u2r_warmup_steps: int = 0
    # Pixels whose predicted group confidence (max softmax over the G+1
    # channels) is below min_conf, whose instance-render alpha is below
    # min_alpha, or whose argmax is the void channel never participate.
    tsh_mbm_u2r_min_conf: float = 0.3
    tsh_mbm_u2r_min_alpha: float = 0.05
    # When > 0 the TokenGS shared token transformer tail (decoder_blocks)
    # becomes trainable at this LR (optimizer geometry group).  The image
    # encoder, activation-head teacher and gs_tokens stay frozen.
    tsh_mbm_decoder_tail_lr: float = 0.0
    # Per-GS slot refinement on the single True-Shared unit assignment:
    # final_gs_logits = broadcast(unit_logits) + alpha * residual_gs_logits.
    # Residual branch and gate alpha are zero-initialized; gate_eff ramps
    # 0 -> 1 over ``tsh_per_gs_ramp_steps`` optimizer steps.
    tsh_per_gs_refine: bool = False
    tsh_per_gs_ramp_steps: int = 125
    tsh_per_gs_hidden: int = 128
    tsh_per_gs_use_group_context: bool = True
    # Assignment-conditioned group-query <-> q_abs unit-memory refinement.
    tsh_query_memory_refine: bool = False
    tsh_query_memory_refine_rounds: int = 2
    tsh_query_memory_refine_gate_steps: int = 50
    tsh_query_memory_refine_probe: bool = False
    tsh_query_memory_refine_head_joint_probe: bool = False
    tsh_query_memory_refine_lr: float = 1.0e-4
    # GA-IDU v3 public framework / GA-IDU-0/1 only.  Geometry adaptation,
    # preservation, cross-view consistency and dustbin matching are separate
    # future phases and remain disabled here.
    ga_idu_mode: str = "off"
    ga_idu_dim: int = 256
    ga_idu_input_dim: int = 1024
    ga_idu_memory_latents: int = 256
    ga_idu_heads: int = 8
    ga_idu_instance_lr: float = 1.0e-4
    ga_idu_gate_steps: int = 5
    # TA-RIU v1: token-aligned reconstruction/instance joint units.  The
    # branch is opt-in and remains completely absent from historical recipes.
    ta_riu_enabled: bool = False
    ta_riu_dim: int = 256
    ta_riu_memory_latents: int = 256
    ta_riu_heads: int = 8
    ta_riu_shared_lr: float = 1.0e-5
    ta_riu_instance_lr: float = 3.0e-5
    ta_riu_geometry_lr: float = 1.0e-5
    ta_riu_appearance_lr: float = 1.0e-5
    ta_riu_gate_steps: int = 25
    ta_riu_xyz_scale: float = 0.02
    ta_riu_log_scale_scale: float = 0.05
    ta_riu_rot_scale: float = 0.05
    ta_riu_opacity_scale: float = 0.05
    ta_riu_color_scale: float = 0.05
    # Extra intra-epoch optimizer steps that also save head checkpoints
    # (ddp8 metadata included).  Empty by default.
    abs_ckpt_steps_extra: tuple[int, ...] = ()
    # Save optimizer/scheduler/per-rank RNG and provenance sidecars for
    # intra-epoch snapshots when an independently resumable probe needs them.
    abs_ckpt_full_state: bool = False
    # Dedicated rendered-space instance feature (InstanceSplat-style
    # grounding): an explicit instance head on the 8 local units, propagated
    # to GS, rendered to 2D feature maps and supervised by per-view GT
    # instance masks via prototype pull/push/cross-view.  When enabled the
    # identity InfoNCE / Agglomerative objective is not used for training;
    # the grounding feature itself is what eval clusters.
    instance_branch_grounding: bool = False
    instance_branch_grounding_dim: int = 64
    # InstanceSplat-style 3D instance-prototype pull/push on the unit
    # grounding features (in addition to the rendered-space loss): per-GT
    # instance 3D prototypes built from the unit features (detached), each
    # unit pulled toward its instance prototype (tightening same-instance
    # units), and different-instance prototypes pushed apart (anti-collapse).
    instance_branch_grounding_3d_pull: float = 0.0
    instance_branch_grounding_3d_push: float = 0.0
    instance_branch_grounding_3d_margin: float = 0.2
    # Dynamic Scene Instance Query -> direct mask prediction (one-shot
    # feed-forward).  Scene-specific queries are generated from the current
    # scene's unit features (PointGroup-style 3D center prior + FPS +
    # Mask3D-style cross-attention refinement) and directly predict
    # query->unit mask logits, propagated to GS and rendered.  No
    # Agglomerative / clustering / TTA.
    instance_branch_dynamic_queries: bool = False
    instance_branch_num_queries: int = 128
    instance_branch_query_dim: int = 128
    instance_branch_query_layers: int = 2
    # Anti query-collapse: push the K scene-specific query features apart
    # (DETR-style set-diversity), so the Hungarian matching does not keep
    # re-selecting the same few queries and voiding the rest.
    instance_branch_query_diversity: float = 0.0
    instance_branch_query_diversity_margin: float = 0.1
    # PointGroup-style per-unit instance-center offset prediction: a small
    # head regresses the 3D offset from each 8-local-unit center to its
    # instance center; inference groups the offset-adjusted (tight) centers
    # with a simple 3D center clustering instead of high-dim embedding
    # clustering.
    instance_branch_center_offset: bool = False
    instance_branch_center_offset_hidden: int = 256
    instance_center_loss_weight: float = 1.0
    instance_branch_pseudo_conf: float = 0.0
    instance_branch_pseudo_min_views: int = 0
    instance_branch_pseudo_unit_min_mass: float = 0.0
    instance_branch_scene_assignment: bool = False
    instance_branch_scene_slots: int = 100
    instance_branch_scene_slot_iters: int = 3
    instance_branch_scene_slot_temp: float = 5.0
    instance_branch_scene_slot_pos_weight: float = 1.0
    instance_branch_scene_slot_entropy: float = 0.05
    instance_branch_scene_slot_void: float = 0.1
    # Penalty on the mean foreground mass of slots that no GT instance
    # matched (pushes surplus slots toward empty), and the minimum per-slot
    # unit mass for a slot to survive the eval-time empty-slot filter.
    instance_branch_scene_slot_unmatched: float = 0.1
    instance_branch_scene_slot_min_mass: float = 1.0
    # DPG: learn scene-specific instance prototypes (FPS init + cross-attn
    # refinement) supervised by GT instance prototypes (p_u-weighted unit
    # embedding means) via Hungarian cosine pull; hard cosine assignment.
    # K = GT instance count (train) / GT-count oracle (eval) in this first
    # version -- no count head yet.
    instance_branch_dpg: bool = False
    instance_branch_dpg_proto_dim: int = 256
    instance_branch_dpg_heads: int = 4
    instance_branch_dpg_layers: int = 2
    instance_branch_dpg_proto_weight: float = 1.0
    # InstanceSplat-style rendered-space embedding supervision: render the
    # unit identity embedding through the frozen Gaussians into the target
    # views, then apply prototype pull (pixels -> instance prototype),
    # prototype-level push (different instances), and cross-view prototype
    # consistency.  Applied on top of the existing InfoNCE; Agglomerative
    # inference unchanged.
    instance_branch_render_space: bool = False
    instance_branch_render_space_pull: float = 1.0
    instance_branch_render_space_push: float = 0.5
    instance_branch_render_space_cross: float = 1.0
    instance_branch_render_space_margin_push: float = 0.5
    instance_branch_render_space_margin_cross: float = 0.2
    # Pixel-level InfoNCE on the rendered unit embedding (same-instance
    # rendered pixels = soft positives, different-instance = negatives).
    # Fixes the prototype-mean pull/push failure mode (small mean offset
    # only) by supervising pixel-embedding separation directly.
    instance_branch_render_space_info_nce: float = 0.0
    instance_branch_render_space_info_temp: float = 0.2
    instance_branch_render_space_info_samples: int = 32
    # InstanceSplat-style direct per-GS instance embedding (no unit
    # clustering): a compact D-dim embedding attached to each frozen
    # Gaussian, rendered into target views and supervised directly by the
    # 2D GT instance masks (prototype pull / prototype push / cross-view).
    # Inference clusters the rendered embedding per view (k = GT-count
    # oracle in v1).
    instance_branch_direct_gs: bool = False
    instance_branch_direct_gs_embed_dim: int = 8
    instance_branch_direct_gs_hidden: int = 128
    instance_branch_direct_gs_dino: bool = True
    instance_branch_direct_gs_pull: float = 1.0
    instance_branch_direct_gs_push: float = 2.0
    instance_branch_direct_gs_cross: float = 1.0
    instance_branch_direct_gs_margin_push: float = 1.0
    instance_branch_direct_gs_margin_cross: float = 0.2
    # Rendered-space pixel-level InfoNCE: same-instance rendered pixels are
    # soft positives, different-instance pixels negatives.  Directly
    # supervises pixel-embedding separation (prototype-mean pull/push alone
    # only produced a small mean offset while pixel bodies stayed collinear).
    instance_branch_direct_gs_info_nce: float = 1.0
    instance_branch_direct_gs_info_temp: float = 0.1
    instance_branch_direct_gs_info_samples: int = 32
    # Joint training: allow the instance loss to backprop into the TokenGS
    # token hidden / encoder (instead of detaching it), so the reconstruction
    # features can become instance-aware.  Must be combined with a
    # reconstruction-dominant loss and an instance-loss warm-up to avoid
    # breaking RGB reconstruction.
    instance_branch_backprop_token: bool = False
    lambda_scene_assignment_unit: float = 1.0
    instance_branch_cluster_pos_weight: float = 1.0
    instance_branch_cluster_eps: float = 1.0
    instance_branch_void_fg_share: float = 0.5
    # --- Direction B (B0): Scene-Instance-Conditioned Local Unit Formation
    # ---
    # Scene-conditioned instance queries (SIC): 3D-FPS-anchored queries over
    # per-token descriptors (token hidden + anchor 3D pos + encoder patch +
    # DINO) refined by cross-attention.  The per-token unit queries are
    # initialized with a ZERO-gated readout of the SIC queries, so instance
    # context enters unit formation while step 0 stays exactly equal to the
    # 0.324 baseline (gate=0).  SIC queries are NOT instance prototypes:
    # FPS only gives spatial coverage for the scene-level conditioning.
    instance_branch_sic_units: bool = False
    instance_branch_sic_queries: int = 128
    instance_branch_sic_dim: int = 256
    instance_branch_sic_heads: int = 4
    instance_branch_sic_layers: int = 2
    instance_branch_sic_usage: float = 0.02
    num_dynamic_gs_tokens: int = 0
    init_dynamic_tokens_from_static: bool = False
    init_tokens_from_existing: bool = False
    init_latents_from_existing: bool = False

    # --- dataset
    data_mode: tuple[tuple[str, int], ...] = (("dl3dv_scaled_0.15", 6),)
    num_views: int = 8
    num_input_views: int = 4
    znear: float = 0.025
    zfar: float = 125.0
    camera_normalization_method: Literal["mean_cam", "first_cam"] = "first_cam"
    camera_scale_method: Literal["constant", "distance", "bound", "pointmap"] = "constant"
    pointmap_trim_lo: float = 0.0
    pointmap_trim_hi: float = 1.0
    num_workers: int = 16
    dataset_kwargs: dict[str, Any] | None = None
    prompt_mode: Literal[
        "text_only", "image_only", "text_image_mixed", "manifest"
    ] = "text_image_mixed"
    query_image_size: tuple[int, int] = (224, 224)
    prompt_image_probability: float = 0.5
    prompt_min_target_pixels: int = 64
    prompt_same_scene_query_min_gap: int = 90
    prompt_same_scene_query_ratio: float = 0.5

    # --- prompt-conditioned token matching
    prompt_training: bool = False
    prompt_tokengs_checkpoint: str = "/space0/mawb/tokengs/workspace/tokengs_re10k/model.safetensors"
    prompt_semantic_adapter_resume: str = ""
    prompt_clip_model_path: str = "/space0/mawb/tokengs/checkpoints/clip-vit-base-patch32"
    prompt_hidden_dim: int = 128
    prompt_attention_heads: int = 4
    prompt_mixed_text_weight: float = 0.5
    prompt_image_pooling: Literal["masked_patch", "masked_input_cls"] = "masked_input_cls"
    prompt_lambda_bce: float = 1.0
    prompt_lambda_dice: float = 1.0
    prompt_balanced_bce: bool = False
    prompt_balanced_bce_pos_weight: float = 0.5
    prompt_text_adapter: bool = False
    prompt_unfreeze_tokengs: bool = False
    prompt_unfreeze_tokengs_lr: float = 0.0
    # Protected-decoder joint training: when unfreezing a decoder tail, run
    # the RGB reconstruction from the DETACHED decoder hidden (so the
    # reconstruction loss becomes a monitor and cannot overfit the training
    # windows), while the instance branch still consumes the grad-carrying
    # hidden.  The frozen teacher distillation then pins PSNR.
    prompt_detach_reconstruction_tokens: bool = False
    # Geometry unfreeze granularity when prompt_unfreeze_tokengs=True:
    #  * "all"      - unfreeze every non-matcher parameter (old behaviour).
    #  * "decoder"  - only decoder_blocks + activation_head + gs_tokens are
    #    trainable; the encoder and patch embeddings stay frozen so the
    #    cross-view features remain stable while the token->Gaussian
    #    geometry can still adapt to instance boundaries.
    prompt_unfreeze_tokengs_mode: str = "all"
    # Boundary-aware reconstruction: an extra RGB term that only weights
    # pixels on GT instance boundaries (optionally dilated), so the
    # reconstruction is pushed to resolve sharp edges where instances meet.
    # This is the instance-to-geometry coupling (InstanceSplat-style) that
    # makes unfrozen joint training refine boundaries instead of drifting.
    lambda_boundary_rgb: float = 0.0
    boundary_rgb_dilate: int = 1
    boundary_rgb_include_background: bool = True
    prompt_threshold: float = 0.5
    prompt_valid_alpha_threshold: float = 1e-3
    prompt_overfit_single_batch: bool = False
    prompt_overfit_sample_index: int = 0
    prompt_visualization_steps: tuple[int, ...] = (0, 10, 50, 100, 200, 500)
    prompt_save_validation_checkpoints: bool = False
    prompt_tune_last_cross_attention: bool = False
    semantic_v2_dim: int = 256
    semantic_v2_temperature_init: float = 14.285714
    semantic_v2_balanced_bce: bool = False
    semantic_v2_score_mode: Literal["sigmoid", "softmax"] = "sigmoid"
    semantic_v2_tune_last_cross_attention: bool = False
    conditional_v3_tune_last_cross_attention: bool = True
    semantic_v4_feature_dim: int = 32
    semantic_v4_use_geometry: bool = True
    semantic_v4_teacher_projection: Literal[
        "frozen_random", "trainable"
    ] = "frozen_random"
    semantic_v2_class_weights: tuple[float, ...] = (
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    )
    lseg_checkpoint_path: str = (
        "/space0/mawb/tokengs/checkpoints/demo_e200.ckpt"
    )
    semantic_v5_classifier_hidden: int = 128
    lambda_ce: float = 1.0
    lambda_ce_cosine: float = 0.0
    use_instance_labels: bool = False
    instance_group_num_groups: int = 64
    instance_group_alpha_threshold: float = 0.05
    instance_group_min_instance_pixels: int = 64
    lambda_instance_group_dice: float = 1.0
    lambda_instance_group_mask: float = 1.0
    lambda_instance_group_void: float = 0.1
    lambda_instance_group_unmatched: float = 0.1
    instance_group_lambda_warmup_steps: int = 1500
    instance_group_render_scale: float = 1.0
    instance_group_area_alpha: float = 0.0
    instance_group_match_area_norm: bool = False
    lambda_instance_group_ce: float = 0.0
    instance_group_decoder: bool = False
    instance_group_decoder_layers: int = 2
    instance_group_group_token_refine: bool = False
    instance_group_group_token_refine_scale_init: float = 0.05
    instance_group_group_token_refine_scale_max: float = 0.3
    instance_group_match_topk: int = 1
    instance_group_secondary_pair_weight: float = 0.3
    instance_group_usage_entropy: float = 0.0
    instance_group_scene_level_matching: bool = False
    instance_group_supervise_input_views: bool = False
    lambda_instance_contrastive: float = 0.0
    instance_group_use_anchor_pos: bool = False
    instance_group_per_gaussian: bool = False
    instance_group_feature_dim: int = 16
    instance_group_residual_head: bool = False
    instance_group_residual_scale: float = 0.3
    lambda_instance_group_3d: float = 0.0
    lambda_instance_group_3d_ce: float = 1.0
    instance_group_3d_min_gs: int = 16
    instance_group_3d_match_topk: int = 1
    # Scene-adaptive group budget: a lightweight count head predicts the
    # number of active instance groups per scene; both the 2D and 3D losses
    # only supervise the top-``G`` most-used groups (the rest are pushed to
    # void), and eval prunes inactive groups before mask extraction. This
    # attacks the fixed 128-groups vs 20~134 GT instances mismatch.
    instance_group_adaptive_count: bool = False
    instance_group_count_head: bool = False
    instance_group_count_hidden: int = 128
    lambda_instance_group_count: float = 0.1
    # InstOk3D-style position-aware group refinement: extra cross-attention
    # layers whose keys are the per-token 3D anchor positions, added to the
    # group queries behind a zero-initialized gate so the warm-started base
    # decoder reproduces the checkpoint exactly at init.
    instance_group_pos_attn_layers: int = 0
    instance_group_pos_attn_scale: float = 1.0
    # Object-conditioned Gaussian generation: group queries attend to the
    # TokenGS decoder anchors and inject an identity-preserving residual before
    # activation. Disabled by default to keep existing recipes unchanged.
    instance_group_conditioned_gaussians: bool = False
    instance_group_condition_dim: int = 256
    instance_group_condition_heads: int = 8
    instance_group_condition_layers: int = 2
    instance_group_condition_residual_scale: float = 1.0
    instance_group_condition_assignment_temperature: float = 10.0
    instance_group_condition_gaussian_blend: float = 0.1
    # GC3: shared scene queries plus independent local Gaussian assignments.
    # The local branch is zero-initialized and only applies a bounded opacity
    # residual; xyz/scale/rotation remain on the frozen TokenGS proposal.
    instance_group_condition_per_gaussian: bool = False
    instance_group_condition_per_gaussian_opacity_scale: float = 0.05
    # GC4: project frozen encoder patch features onto the provisional 3D
    # anchor centers before group-query decoding. This makes the group path
    # consume pixel-aligned multi-view evidence rather than only global GS
    # token content and post-activation Gaussian geometry.
    instance_group_condition_image_anchors: bool = False
    instance_group_condition_image_feature_dim: int = 64
    instance_group_condition_image_upsample: int = 2
    instance_group_condition_image_multiscale: bool = True
    instance_group_condition_image_scale: float = 1.0
    instance_group_condition_decoder_checkpoint: bool = False
    # Dense image-evidence instance decoder: a small trainable head maps the
    # frozen TokenGS encoder's per-patch features (multi-scale optional) into
    # per-location instance features, projects them onto the Gaussian centers
    # through the source cameras, and adds a small residual to the warm-started
    # token-level assignment. This gives every Gaussian its own pixel evidence
    # instead of only token-derived content + 3D position.
    instance_group_dense_decoder: bool = False
    instance_group_dense_feature_dim: int = 16
    instance_group_dense_upsample: int = 2
    instance_group_dense_multiscale: bool = True
    instance_group_dense_scale: float = 0.3
    instance_group_dense_gate: bool = True
    lambda_instance_dense_aux: float = 0.0
    backbone_resume: str = ""
    lambda_semantic_feature: float = 0.0
    lambda_semantic_cosine: float = 1.0
    lambda_semantic_l1: float = 0.05
    semantic_alpha_threshold: float = 0.05
    semantic_use_depth_filter: bool = False
    semantic_residual_scale: float = 0.1
    lambda_semantic_ce: float = 0.0
    lambda_semantic_source: float = 0.5
    semantic_pseudo_conf_threshold: float = 0.40
    semantic_classifier_hidden_dim: int = 256
    # V3 token_decoder_lowrank semantic field (reproduces ~0.536 C3G8).
    semantic_branch_version: str = "source_projected"
    semantic_token_patch_size: int = 8
    semantic_token_decoder_layers: int = 2
    semantic_token_decoder_heads: int = 8
    semantic_token_mlp_ratio: float = 4.0
    semantic_token_dropout: float = 0.0
    semantic_token_max_views: int = 8
    semantic_local_rank: int = 16
    semantic_local_hidden_dim: int = 128
    semantic_token_residual_scale: float = 0.1
    semantic_local_residual_scale: float = 0.1
    semantic_token_pool_use_opacity: bool = True
    semantic_stream_compressed_features: bool = True
    semantic_feature_use_alpha_mask: bool = True
    semantic_detach_tokens: bool = True
    semantic_detach_geometry: bool = True
    semantic_render_chunk: int = 32
    semantic_render_scale: float = 0.5
    lambda_feat: float = 0.0

    # --- training
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_epochs: int = 30
    max_iters_per_epoch: int = 1_000_000
    lr: float = 4e-4
    lr_scheduler: Literal["onecycle", "constant"] = "onecycle"
    weight_decay: float = 0.05
    pct_start_steps: int = 1000
    final_div_factor: float = 1000.0
    gradient_clip: float = 1.0
    mixed_precision: str = "bf16"
    deferred_bp: bool = False
    use_input_supervision: bool = False
    mean_of_grads: Literal["none", "per-scene", "per-view"] = "none"
    mean_of_grads_scene_chunk_size: int = 1
    mean_of_grads_view_chunk_size: int | None = None

    # --- loss weights
    rgb_loss_type: Literal["l1", "l2"] = "l2"
    lambda_rgb: float = 1.0
    lambda_lpips: float = 0.0
    lambda_mask: float = 0.0
    lambda_ssim: float = 0.2
    lambda_visibility: float = 1.0
    visibility_distance_threshold: float = 1.0
    lambda_opacity: float = 0.0
    lambda_dyn_aux: float = 0.0
    lambda_dyn_aux_warmup_steps: int = 0
    lambda_dyn_aux_decay_steps: int = 0
    lambda_dyn_aux_min: float = 0.0

    # --- logging frequency
    print_freq: int = 10
    log_image_freq: int = 100

    # --- evaluation
    eval_n_media_dumps: int = 0
    max_eval_iters: int = 0
    eval_before_training: bool = True
    strict_checkpoint_loading: bool = True

    # --- test-time training (eval)
    use_ttt_for_eval: bool = False
    ttt_mode: Literal["token-tuning", "scene-latent-tuning", "tokens", "latents"] = "token-tuning"
    ttt_n_steps: int = 50
    ttt_lr: float = 1e-4

    # --- dynamic scenes
    time_embedding: bool = False
    time_embedding_dim: int = 2
    use_interp_target: bool = False

    # --- latent bottleneck release architecture
    use_multiscale_encoder: bool = False
    multiscale_encoder_layers: tuple[int, ...] = (5, 7, 9, 11)
    use_latent_bottleneck: bool = False
    num_latents: int = 4096
    latent_cross_attn_depth: int = 12

    # --- augmentation
    random_reflect: bool = True

    def __post_init__(self) -> None:
        if self.dec_patch_size is None:
            self.dec_patch_size = self.patch_size
        self.validate()

    def validate(self) -> None:
        if self.evaluating:
            assert not self.use_input_supervision, "use_input_supervision must be False when evaluating"
        if self.mean_of_grads not in ("none", "per-scene", "per-view"):
            raise ValueError("mean_of_grads must be one of: none, per-scene, per-view")
        if self.deferred_bp and self.mean_of_grads != "none":
            raise ValueError("deferred_bp and mean_of_grads are mutually exclusive backprop strategies")
        if self.mean_of_grads_scene_chunk_size <= 0:
            raise ValueError("mean_of_grads_scene_chunk_size must be positive")
        if self.mean_of_grads_view_chunk_size is not None and self.mean_of_grads_view_chunk_size <= 0:
            raise ValueError("mean_of_grads_view_chunk_size must be positive")
        if self.prompt_training:
            if self.model_type not in (
                "prompt_tokengs",
                "semantic_tokengs_v2",
                "semantic_tokengs_v3",
                "semantic_tokengs_v4",
                "semantic_tokengs_v5",
                "semantic_tokengs_v6",
                "conditional_prompt_tokengs",
            ):
                raise ValueError(
                    "prompt_training=True requires a prompt or semantic model"
                )
            if self.num_gs_tokens <= 0:
                raise ValueError("num_gs_tokens must be positive")
            if self.dec_patch_size != 8:
                raise ValueError("Prompt training requires dec_patch_size=8")
            if self.deferred_bp:
                raise ValueError("Prompt training requires deferred_bp=False")
            if self.use_ttt_for_eval:
                raise ValueError("Prompt training does not support TTT")
            if not 0.0 <= self.prompt_threshold <= 1.0:
                raise ValueError("prompt_threshold must be in [0, 1]")
            if not 0.0 <= self.prompt_mixed_text_weight <= 1.0:
                raise ValueError("prompt_mixed_text_weight must be in [0, 1]")
            if self.semantic_v2_dim <= 0:
                raise ValueError("semantic_v2_dim must be positive")
            if self.semantic_v2_temperature_init <= 0:
                raise ValueError("semantic_v2_temperature_init must be positive")
            if self.semantic_v2_score_mode not in ("sigmoid", "softmax"):
                raise ValueError("semantic_v2_score_mode must be sigmoid or softmax")
        if self.prompt_overfit_sample_index < 0:
            raise ValueError("prompt_overfit_sample_index must be non-negative")
        if (
            self.prompt_tune_last_cross_attention
            and self.model_type != "prompt_tokengs"
        ):
            raise ValueError(
                "prompt_tune_last_cross_attention requires model_type=prompt_tokengs"
            )
        if any(step < 0 for step in self.prompt_visualization_steps):
            raise ValueError("prompt_visualization_steps must be non-negative")

    def evolve(self, **changes: Any) -> Options:
        """Return a deep copy with the given fields replaced."""
        new_instance = copy.deepcopy(self)
        for key, value in changes.items():
            if not hasattr(new_instance, key):
                raise AttributeError(f"Options has no attribute '{key}'")
            setattr(new_instance, key, value)
        new_instance.validate()
        return new_instance


config_defaults: dict[str, Options] = {}
config_doc: dict[str, str] = {}

config_doc["train_dl3dv_base"] = "DL3DV training defaults (long schedule, capped iters/epoch)."
config_defaults["train_dl3dv_base"] = Options(
    num_epochs=300,
    max_iters_per_epoch=500,
    pct_start_steps=2000,
)

config_doc["finetune_dl3dv_2view"] = "Short finetune from existing tokens, 2 input views, wide images."
config_defaults["finetune_dl3dv_2view"] = config_defaults["train_dl3dv_base"].evolve(
    num_epochs=20,
    pct_start_steps=400,
    lr=4e-5,
    num_gs_tokens=4096,
    init_tokens_from_existing=True,
    num_input_views=2,
    img_size=(256, 448),
)

config_doc["finetune_dl3dv_4view"] = "Like finetune_dl3dv_2view with 4 input views."
config_defaults["finetune_dl3dv_4view"] = config_defaults["finetune_dl3dv_2view"].evolve(
    num_input_views=4,
)

config_doc["finetune_dl3dv_6view"] = "Like finetune_dl3dv_2view with 6 input views and 10 total views."
config_defaults["finetune_dl3dv_6view"] = config_defaults["finetune_dl3dv_2view"].evolve(
    num_input_views=6,
    num_views=10,
)

config_doc["eval_dl3dv_2view"] = "DL3DV eval preset: 2 views, eval JSON, single batch."
config_defaults["eval_dl3dv_2view"] = Options(
    data_mode=(("dl3dv_eval_scaled_0.15", 1),),
    dataset_kwargs={"evaluation_json": "assets/evaluation_idx_dl3dv_depthsplat_2v.json"},
    num_input_views=2,
    img_size=(256, 448),
    evaluating=True,
    num_gs_tokens=4096,
    use_input_supervision=False,
    batch_size=1,
)

config_doc["eval_dl3dv_4view"] = "DL3DV eval preset: 4 input views."
config_defaults["eval_dl3dv_4view"] = config_defaults["eval_dl3dv_2view"].evolve(
    num_input_views=4,
    dataset_kwargs={"evaluation_json": "assets/evaluation_idx_dl3dv_depthsplat_4v.json"},
)
config_doc["eval_dl3dv_6view"] = "DL3DV eval preset: 6 input views."
config_defaults["eval_dl3dv_6view"] = config_defaults["eval_dl3dv_2view"].evolve(
    num_input_views=6,
    dataset_kwargs={"evaluation_json": "assets/evaluation_idx_dl3dv_depthsplat_6v.json"},
)


# ----- DL3DV latent-bottleneck release presets -----
_LATENT_D12_ARCH = {
    "enc_depth": 12,
    "dec_depth": 1,
    "dec_patch_size": 8,
    "clip_head_readout_std": 0.002,
    "clip_head_z_init": 0.1,
    "dec_init_values": 0.01,
    "gaussian_z_offset": 0.0,
    "opacity_bias": 2.0,
    "gs_token_std": 0.02,
    "use_multiscale_encoder": True,
    "multiscale_encoder_layers": (5, 7, 9, 11),
    "use_latent_bottleneck": True,
    "num_latents": 4096,
    "latent_cross_attn_depth": 12,
    "camera_normalization_method": "mean_cam",
    "camera_scale_method": "constant",
}

_SSIM_LOSS = {
    "rgb_loss_type": "l1",
    "lambda_rgb": 0.8,
    "lambda_ssim": 0.2,
    "lambda_lpips": 0.0,
}

_LPIPS_LOSS = {
    "rgb_loss_type": "l2",
    "lambda_rgb": 1.0,
    "lambda_ssim": 0.0,
    "lambda_lpips": 0.5,
}


def _latent_dl3dv_train_preset(num_input_views: int, num_views: int) -> Options:
    return config_defaults["train_dl3dv_base"].evolve(
        data_mode=(("dl3dv_scaled_0.15", 6),),
        num_epochs=20,
        pct_start_steps=400,
        lr=4e-5,
        num_gs_tokens=4096,
        init_tokens_from_existing=True,
        init_latents_from_existing=True,
        num_input_views=num_input_views,
        num_views=num_views,
        img_size=(256, 448),
        **_LATENT_D12_ARCH,
    )


config_doc["train_dl3dv_latent_base"] = (
    "Scratch DL3DV latent-bottleneck training base. Uses pointmap scene "
    "rescaling, the 12-layer encoder latent architecture, and no checkpoint "
    "initialization."
)
config_defaults["train_dl3dv_latent_base"] = Options(
    data_mode=(("dl3dv_scaled_1.0", 6),),
    num_epochs=178,
    pct_start_steps=2000,
    use_input_supervision=True,
    rgb_loss_type="l1",
    lambda_rgb=0.8,
    **{**_LATENT_D12_ARCH, "camera_scale_method": "pointmap"},
)


def _latent_dl3dv_eval_preset(num_input_views: int, evaluation_json: str) -> Options:
    return Options(
        data_mode=(("dl3dv_eval_scaled_0.15", 1),),
        dataset_kwargs={"evaluation_json": evaluation_json},
        num_input_views=num_input_views,
        img_size=(256, 448),
        evaluating=True,
        num_gs_tokens=4096,
        use_input_supervision=False,
        ttt_mode="scene-latent-tuning",
        ttt_lr=1e-2,
        batch_size=1,
        **_LATENT_D12_ARCH,
    )


for _num_input_views, _num_views in ((2, 8), (4, 8), (6, 10)):
    _name = f"finetune_dl3dv_latent_{_num_input_views}view_ssim"
    config_doc[_name] = (
        f"Finetune the released SA d12 latent-bottleneck DL3DV {_num_input_views}-view SSIM model."
    )
    config_defaults[_name] = _latent_dl3dv_train_preset(
        _num_input_views, _num_views
    ).evolve(**_SSIM_LOSS)

    _name = f"finetune_dl3dv_latent_{_num_input_views}view_lpips"
    config_doc[_name] = (
        f"Finetune the released SA d12 latent-bottleneck DL3DV {_num_input_views}-view LPIPS model."
    )
    config_defaults[_name] = _latent_dl3dv_train_preset(
        _num_input_views, _num_views
    ).evolve(**_LPIPS_LOSS)

for _num_input_views in (2, 4, 6):
    _eval_json = f"assets/evaluation_idx_dl3dv_depthsplat_{_num_input_views}v.json"

    _name = f"eval_dl3dv_latent_{_num_input_views}view_ssim"
    config_doc[_name] = (
        f"Evaluate the released SA d12 latent-bottleneck DL3DV {_num_input_views}-view SSIM checkpoint."
    )
    config_defaults[_name] = _latent_dl3dv_eval_preset(
        _num_input_views, _eval_json
    ).evolve(**_SSIM_LOSS)

    _name = f"eval_dl3dv_latent_{_num_input_views}view_lpips"
    config_doc[_name] = (
        f"Evaluate the released SA d12 latent-bottleneck DL3DV {_num_input_views}-view LPIPS checkpoint."
    )
    config_defaults[_name] = _latent_dl3dv_eval_preset(
        _num_input_views, _eval_json
    ).evolve(**_LPIPS_LOSS)


# ----- Kubric 4D dynamic finetune -----
config_doc["finetune_dl3dv_kubric_dyn"] = (
    "Dynamic finetune of the DL3DV base on Kubric4D. Adds dynamic GS tokens "
    "warm-started from the static tokens, sinusoidal time embeddings, "
    "interpolated target-frame sampling, and pointmap camera scaling."
)
config_defaults["finetune_dl3dv_kubric_dyn"] = config_defaults["train_dl3dv_base"].evolve(
    use_input_supervision=False,
    data_mode=(("kubric_scaled_1.0", 100),),
    dataset_kwargs={},
    num_input_views=4,
    num_views=8,
    img_size=(256, 256),
    num_epochs=100,
    max_iters_per_epoch=500,
    pct_start_steps=2500,
    lr=4e-5,
    init_tokens_from_existing=True,
    num_gs_tokens=1024,
    num_dynamic_gs_tokens=256,
    init_dynamic_tokens_from_static=True,
    time_embedding=True,
    time_embedding_dim=2,
    use_interp_target=True,
    camera_normalization_method="mean_cam",
    camera_scale_method="pointmap",
    enc_depth=12,
    dec_depth=1,
    dec_patch_size=8,
    clip_head_readout_std=0.002,
    clip_head_z_init=0.1,
    dec_init_values=0.01,
    gaussian_z_offset=0.0,
    opacity_bias=2.0,
    gs_token_std=0.02,
    use_multiscale_encoder=True,
    multiscale_encoder_layers=(5, 7, 9, 11),
    use_latent_bottleneck=True,
    num_latents=4096,
    latent_cross_attn_depth=12,
    rgb_loss_type="l1",
    lambda_rgb=0.8,
    lambda_ssim=0.2,
    lambda_visibility=1.0,
    lambda_opacity=0.0,
    project_name="TokenGS-Kubric",
)

config_doc["finetune_dl3dv_kubric_static"] = (
    "Stage-1 Kubric domain finetune without dynamic tokens or time embeddings. "
    "Use this when reproducing the two-stage warm-start variant."
)
config_defaults["finetune_dl3dv_kubric_static"] = config_defaults["finetune_dl3dv_kubric_dyn"].evolve(
    num_dynamic_gs_tokens=0,
    init_dynamic_tokens_from_static=False,
    time_embedding=False,
)

config_doc["finetune_dl3dv_kubric_dyn_v2"] = (
    "Kubric dynamic finetune with an auxiliary dynamic-only render loss."
)
config_defaults["finetune_dl3dv_kubric_dyn_v2"] = config_defaults["finetune_dl3dv_kubric_dyn"].evolve(
    lambda_dyn_aux=0.3,
)

_kubric_dyn_release = config_defaults["finetune_dl3dv_kubric_dyn_v2"].evolve(
    lambda_dyn_aux=0.3,
    lambda_dyn_aux_warmup_steps=5000,
    lambda_dyn_aux_decay_steps=10000,
    lambda_dyn_aux_min=0.0,
)
config_doc["finetune_dl3dv_kubric_dyn_release"] = (
    "Released Kubric dynamic finetune schedule: hold lambda_dyn_aux=0.3 for "
    "5K steps, then linearly decay it to 0 over 10K steps."
)
config_defaults["finetune_dl3dv_kubric_dyn_release"] = _kubric_dyn_release
config_doc["finetune_dl3dv_kubric_dyn_v3"] = (
    "Backward-compatible alias for finetune_dl3dv_kubric_dyn_release."
)
config_defaults["finetune_dl3dv_kubric_dyn_v3"] = _kubric_dyn_release

# ----- ScanNet data-pipeline debug -----
config_doc["debug_scannet_dataset"] = (
    "Load one labeled ScanNet scene for data-pipeline and visualization checks."
)
config_defaults["debug_scannet_dataset"] = Options(
    data_mode=(("scannet_scaled_0.15", 1),),
    dataset_kwargs={
        "subset": "scene0286_01",
        "frame_stride": "10",
        "label_mapping": "raw",
    },
    img_size=(256, 256),
    num_input_views=2,
    num_views=3,
    batch_size=1,
    num_workers=0,
    evaluating=True,
    random_reflect=False,
    workspace="workspace/scannet_debug",
)

config_doc["eval_scannet_c3g8"] = (
    "C3G/LSM manifest-driven ScanNet eight-class evaluation dataset."
)
config_defaults["eval_scannet_c3g8"] = Options(
    data_mode=(("scannet_c3g8_eval", 1),),
    img_size=(256, 256),
    num_input_views=2,
    num_views=3,
    batch_size=1,
    num_workers=0,
    evaluating=True,
    random_reflect=False,
    workspace="workspace/scannet_c3g8_eval",
)

config_doc["eval_scannet_c3g8_prompt_text"] = (
    "Final eight-class text-only prompt evaluation over the held-out C3G "
    "ScanNet split (per-class binary masks, mIoU/mAcc + PSNR/SSIM/LPIPS)."
)
config_defaults["eval_scannet_c3g8_prompt_text"] = Options(
    data_mode=(("scannet_c3g8_prompt_eval", 1),),
    model_type="prompt_tokengs",
    prompt_training=True,
    prompt_mode="text_only",
    prompt_tokengs_checkpoint="/space0/mawb/tokengs/workspace/tokengs_re10k/model.safetensors",
    prompt_clip_model_path="/space0/mawb/tokengs/checkpoints/clip-vit-base-patch32",
    prompt_image_pooling="masked_input_cls",
    img_size=(256, 256),
    num_input_views=2,
    num_views=3,
    batch_size=1,
    num_workers=0,
    evaluating=True,
    random_reflect=False,
    max_eval_iters=0,
    eval_n_media_dumps=8,
    mixed_precision="bf16",
    workspace="workspace/scannet_c3g8_prompt_eval",
    experiment_name="scannet_c3g8_prompt_eval",
)

config_doc["eval_scannet_c3g8_semantic_v2"] = (
    "Final eight-class semantic_v2 evaluation over the held-out C3G ScanNet "
    "split using the joint eight-class prototype adapter (no CLIP text prompts)."
)
config_defaults["eval_scannet_c3g8_semantic_v2"] = Options(
    data_mode=(("scannet_c3g8_eval", 1),),
    model_type="semantic_tokengs_v2",
    prompt_training=True,
    prompt_tokengs_checkpoint="/space0/mawb/tokengs/workspace/tokengs_re10k/model.safetensors",
    prompt_clip_model_path="/space0/mawb/tokengs/checkpoints/clip-vit-base-patch32",
    semantic_v2_dim=256,
    semantic_v2_temperature_init=14.285714,
    semantic_v2_balanced_bce=True,
    img_size=(256, 256),
    num_input_views=2,
    num_views=3,
    batch_size=1,
    num_workers=0,
    evaluating=True,
    random_reflect=False,
    max_eval_iters=0,
    eval_n_media_dumps=8,
    mixed_precision="bf16",
    workspace="workspace/scannet_c3g8_semantic_v2_eval",
    experiment_name="scannet_c3g8_semantic_v2_eval",
)

config_doc["debug_scannet_prompt"] = (
    "Prompt-training ScanNet sample with a forced cross-scene image query."
)
config_defaults["eval_scannet_lsm_instance"] = Options(
    data_mode=(("scannet_lsm_instance_eval", 1),),
    model_type="semantic_tokengs_v6",
    prompt_training=True,
    prompt_tokengs_checkpoint="/space0/mawb/tokengs/workspace/tokengs_re10k/model.safetensors",
    prompt_clip_model_path="/space0/mawb/tokengs/checkpoints/clip-vit-large-patch14",
    semantic_v2_dim=256,
    semantic_v2_temperature_init=14.285714,
    semantic_v2_balanced_bce=True,
    img_size=(256, 256),
    num_input_views=8,
    num_views=15,
    batch_size=1,
    num_workers=0,
    evaluating=True,
    random_reflect=False,
    use_instance_labels=True,
    instance_group_num_groups=64,
    max_eval_iters=0,
    eval_n_media_dumps=0,
    mixed_precision="bf16",
    workspace="workspace/scannet_lsm_instance_eval",
    experiment_name="scannet_lsm_instance_eval",
)
config_defaults["debug_scannet_prompt"] = Options(
    data_mode=(("scannet_prompt_train", 1),),
    prompt_mode="image_only",
    query_image_size=(224, 224),
    img_size=(256, 256),
    num_input_views=2,
    num_views=3,
    batch_size=1,
    num_workers=0,
    evaluating=True,
    random_reflect=False,
    workspace="workspace/scannet_prompt_debug",
)


_PROMPT_TRAINING_COMMON = {
    "model_type": "prompt_tokengs",
    "prompt_training": True,
    "prompt_tokengs_checkpoint": "/space0/mawb/tokengs/workspace/tokengs_re10k/model.safetensors",
    "prompt_clip_model_path": "/space0/mawb/tokengs/checkpoints/clip-vit-base-patch32",
    "data_mode": (("scannet_prompt_train", 1),),
    "prompt_mode": "text_image_mixed",
    "prompt_image_probability": 0.5,
    "prompt_hidden_dim": 128,
    "prompt_attention_heads": 4,
    "prompt_mixed_text_weight": 0.5,
    "prompt_lambda_bce": 1.0,
    "prompt_lambda_dice": 1.0,
    "prompt_threshold": 0.5,
    "num_gs_tokens": 1024,
    "dec_patch_size": 8,
    "num_input_views": 2,
    "num_views": 3,
    "img_size": (256, 256),
    "batch_size": 1,
    "lr": 1e-4,
    "weight_decay": 0.05,
    "lambda_rgb": 0.0,
    "lambda_ssim": 0.0,
    "lambda_lpips": 0.0,
    "lambda_visibility": 0.0,
    "lambda_opacity": 0.0,
    "random_reflect": False,
    "deferred_bp": False,
    "use_wandb": False,
}

config_doc["prompt_scannet_smoke"] = "One-batch prompt TokenGS training smoke test."
config_defaults["prompt_scannet_smoke"] = Options(
    **{**_PROMPT_TRAINING_COMMON, "prompt_image_probability": 1.0},
    num_workers=0,
    num_epochs=1,
    max_iters_per_epoch=1,
    max_eval_iters=1,
    print_freq=1,
    log_image_freq=1,
    mixed_precision="no",
    workspace="workspace/prompt_scannet_smoke",
    experiment_name="prompt_scannet_smoke",
)

config_doc["prompt_scannet_overfit"] = "Repeat one fixed ScanNet batch for prompt-mask overfitting."
config_defaults["prompt_scannet_overfit"] = Options(
    **{**_PROMPT_TRAINING_COMMON, "prompt_image_probability": 1.0},
    prompt_overfit_single_batch=True,
    prompt_overfit_sample_index=12,
    prompt_visualization_steps=(0, 10, 50, 100, 200, 500),
    num_workers=0,
    num_epochs=500,
    max_iters_per_epoch=1,
    max_eval_iters=1,
    print_freq=10,
    log_image_freq=10,
    lr_scheduler="constant",
    mixed_precision="no",
    workspace="workspace/prompt_scannet_overfit_500",
    experiment_name="prompt_scannet_overfit_500",
)

config_doc["prompt_scannet_train"] = "First full ScanNet prompt-conditioned TokenGS training preset."
config_defaults["prompt_scannet_train"] = Options(
    **_PROMPT_TRAINING_COMMON,
    num_workers=4,
    num_epochs=30,
    max_iters_per_epoch=500,
    max_eval_iters=16,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_train",
    experiment_name="prompt_scannet_train",
)


_PROMPT_SMALL_COMMON = {
    **_PROMPT_TRAINING_COMMON,
    "data_mode": (("scannet_prompt_small", 1),),
    "prompt_mode": "manifest",
    "batch_size": 1,
    "num_workers": 0,
    "lr": 1e-4,
    "lr_scheduler": "constant",
    "max_eval_iters": 24,
    "eval_n_media_dumps": 6,
    "eval_before_training": False,
}

config_doc["prompt_scannet_small_smoke"] = (
    "Twenty-step balanced 64/8-scene prompt training smoke test."
)
config_defaults["prompt_scannet_small_smoke"] = Options(
    **_PROMPT_SMALL_COMMON,
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/prompt_scannet_small_smoke_20",
    experiment_name="prompt_scannet_small_smoke_20",
)

config_doc["prompt_scannet_small_train"] = (
    "Two-thousand-step balanced multi-scene prompt training with fixed validation."
)
config_defaults["prompt_scannet_small_train"] = Options(
    **_PROMPT_SMALL_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_small_train_2000",
    experiment_name="prompt_scannet_small_train_2000",
)

config_doc["prompt_scannet_small_eval"] = (
    "Evaluate a prompt matching checkpoint on the fixed 8-scene validation split."
)
config_defaults["prompt_scannet_small_eval"] = Options(
    **_PROMPT_SMALL_COMMON,
    evaluating=True,
    resume="workspace/prompt_scannet_small_train_2000/model.safetensors",
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_small_validation",
    experiment_name="prompt_scannet_small_validation",
)


_PROMPT_QUERY_DIVERSE_MANIFEST = (
    "/space0/mawb/tokengs/data/scannet_prompt/"
    "scannet_prompt_small_64_8_query_diverse.json"
)
_PROMPT_QUERY_DIVERSE_COMMON = {
    **_PROMPT_SMALL_COMMON,
    "dataset_kwargs": {"small_manifest_path": _PROMPT_QUERY_DIVERSE_MANIFEST},
}

_PROMPT_TARGET_DIVERSE_MANIFEST = (
    "/space0/mawb/tokengs/data/scannet_prompt/"
    "scannet_prompt_target_diverse_64_8.json"
)
_PROMPT_TARGET_DIVERSE_COMMON = {
    **_PROMPT_SMALL_COMMON,
    "dataset_kwargs": {"small_manifest_path": _PROMPT_TARGET_DIVERSE_MANIFEST},
    "prompt_save_validation_checkpoints": True,
    "eval_n_media_dumps": 16,
    "max_eval_iters": 24,
}

_PROMPT_TARGET_DIVERSE_LAST_CROSS_COMMON = {
    **_PROMPT_TARGET_DIVERSE_COMMON,
    "prompt_tune_last_cross_attention": True,
    "eval_n_media_dumps": 24,
}

config_doc["prompt_scannet_target_diverse_smoke"] = (
    "Twenty-step smoke test using independent dense target-frame sampling."
)
config_defaults["prompt_scannet_target_diverse_smoke"] = Options(
    **{**_PROMPT_TARGET_DIVERSE_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/prompt_scannet_target_diverse_smoke_20",
    experiment_name="prompt_scannet_target_diverse_smoke_20",
)

config_doc["prompt_scannet_target_diverse_train"] = (
    "Two-thousand-step baseline matcher training with independent target-frame diversity."
)
config_defaults["prompt_scannet_target_diverse_train"] = Options(
    **_PROMPT_TARGET_DIVERSE_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_target_diverse_train_2000",
    experiment_name="prompt_scannet_target_diverse_train_2000",
)

config_doc["prompt_scannet_target_diverse_eval"] = (
    "Evaluate the best independent target-frame diversity checkpoint."
)
config_defaults["prompt_scannet_target_diverse_eval"] = Options(
    **_PROMPT_TARGET_DIVERSE_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "prompt_scannet_target_diverse_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_target_diverse_validation_best",
    experiment_name="prompt_scannet_target_diverse_validation_best",
)

config_doc["prompt_scannet_target_diverse_last_cross_smoke"] = (
    "Twenty-step target-diverse smoke test with a semantic final cross-attention fork."
)
config_defaults["prompt_scannet_target_diverse_last_cross_smoke"] = Options(
    **{**_PROMPT_TARGET_DIVERSE_LAST_CROSS_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/prompt_scannet_target_diverse_last_cross_smoke_20",
    experiment_name="prompt_scannet_target_diverse_last_cross_smoke_20",
)

config_doc["prompt_scannet_target_diverse_last_cross_train"] = (
    "Two-thousand-step target-diverse semantic final cross-attention adaptation."
)
config_defaults["prompt_scannet_target_diverse_last_cross_train"] = Options(
    **_PROMPT_TARGET_DIVERSE_LAST_CROSS_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_target_diverse_last_cross_train_2000",
    experiment_name="prompt_scannet_target_diverse_last_cross_train_2000",
)

config_doc["prompt_scannet_target_diverse_last_cross_eval"] = (
    "Evaluate the best target-diverse semantic final cross-attention checkpoint."
)
config_defaults["prompt_scannet_target_diverse_last_cross_eval"] = Options(
    **_PROMPT_TARGET_DIVERSE_LAST_CROSS_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "prompt_scannet_target_diverse_last_cross_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_target_diverse_last_cross_validation_best",
    experiment_name="prompt_scannet_target_diverse_last_cross_validation_best",
)

_PROMPT_TARGET_DIVERSE_BALANCED_CLS_COMMON = {
    **_PROMPT_TARGET_DIVERSE_COMMON,
    "prompt_image_pooling": "masked_input_cls",
    "prompt_balanced_bce": True,
    "eval_n_media_dumps": 24,
}

_PROMPT_TARGET_DIVERSE_BALANCED_BCE_COMMON = {
    **_PROMPT_TARGET_DIVERSE_COMMON,
    "prompt_image_pooling": "masked_patch",
    "prompt_balanced_bce": True,
    "eval_n_media_dumps": 24,
}

config_doc["prompt_scannet_target_diverse_balanced_cls_smoke"] = (
    "Twenty-step target-diverse smoke test with mask-neutralized CLIP CLS "
    "image queries and per-class balanced BCE."
)
config_defaults["prompt_scannet_target_diverse_balanced_cls_smoke"] = Options(
    **{**_PROMPT_TARGET_DIVERSE_BALANCED_CLS_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/prompt_scannet_target_diverse_balanced_cls_smoke_20",
    experiment_name="prompt_scannet_target_diverse_balanced_cls_smoke_20",
)

config_doc["prompt_scannet_target_diverse_balanced_cls_train"] = (
    "Two-thousand-step target-diverse baseline with mask-neutralized CLIP CLS "
    "image queries and per-class balanced BCE (no last-cross fork)."
)
config_defaults["prompt_scannet_target_diverse_balanced_cls_train"] = Options(
    **_PROMPT_TARGET_DIVERSE_BALANCED_CLS_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_target_diverse_balanced_cls_train_2000",
    experiment_name="prompt_scannet_target_diverse_balanced_cls_train_2000",
)

config_doc["prompt_scannet_target_diverse_balanced_cls_eval"] = (
    "Evaluate the best target-diverse masked-CLS + balanced-BCE checkpoint."
)
config_defaults["prompt_scannet_target_diverse_balanced_cls_eval"] = Options(
    **_PROMPT_TARGET_DIVERSE_BALANCED_CLS_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "prompt_scannet_target_diverse_balanced_cls_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_target_diverse_balanced_cls_validation_best",
    experiment_name="prompt_scannet_target_diverse_balanced_cls_validation_best",
)

config_doc["prompt_scannet_target_diverse_balanced_bce_smoke"] = (
    "Twenty-step target-diverse smoke test with per-class balanced BCE "
    "and the original masked-patch pooling (isolates the BCE change)."
)
config_defaults["prompt_scannet_target_diverse_balanced_bce_smoke"] = Options(
    **{**_PROMPT_TARGET_DIVERSE_BALANCED_BCE_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/prompt_scannet_target_diverse_balanced_bce_smoke_20",
    experiment_name="prompt_scannet_target_diverse_balanced_bce_smoke_20",
)

config_doc["prompt_scannet_target_diverse_balanced_bce_train"] = (
    "Two-thousand-step target-diverse baseline with per-class balanced BCE "
    "and the original masked-patch pooling (isolates the BCE change)."
)
config_defaults["prompt_scannet_target_diverse_balanced_bce_train"] = Options(
    **_PROMPT_TARGET_DIVERSE_BALANCED_BCE_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_target_diverse_balanced_bce_train_2000",
    experiment_name="prompt_scannet_target_diverse_balanced_bce_train_2000",
)

config_doc["prompt_scannet_target_diverse_balanced_bce_eval"] = (
    "Evaluate the best target-diverse balanced-BCE (masked-patch) checkpoint."
)
config_defaults["prompt_scannet_target_diverse_balanced_bce_eval"] = Options(
    **_PROMPT_TARGET_DIVERSE_BALANCED_BCE_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "prompt_scannet_target_diverse_balanced_bce_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_target_diverse_balanced_bce_validation_best",
    experiment_name="prompt_scannet_target_diverse_balanced_bce_validation_best",
)

_PROMPT_TARGET_DIVERSE_BALANCED_BCE_CALIB_COMMON = {
    **_PROMPT_TARGET_DIVERSE_BALANCED_BCE_COMMON,
    "prompt_balanced_bce_pos_weight": 0.3,
}

config_doc["prompt_scannet_target_diverse_balanced_bce_calib_smoke"] = (
    "Twenty-step smoke test for calibrated balanced BCE (pos weight 0.3) "
    "with masked-patch pooling."
)
config_defaults["prompt_scannet_target_diverse_balanced_bce_calib_smoke"] = Options(
    **{**_PROMPT_TARGET_DIVERSE_BALANCED_BCE_CALIB_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/prompt_scannet_target_diverse_balanced_bce_calib_smoke_20",
    experiment_name="prompt_scannet_target_diverse_balanced_bce_calib_smoke_20",
)

config_doc["prompt_scannet_target_diverse_balanced_bce_calib_train"] = (
    "Two-thousand-step calibrated balanced-BCE (pos weight 0.3) training "
    "with masked-patch pooling."
)
config_defaults["prompt_scannet_target_diverse_balanced_bce_calib_train"] = Options(
    **_PROMPT_TARGET_DIVERSE_BALANCED_BCE_CALIB_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_target_diverse_balanced_bce_calib_train_2000",
    experiment_name="prompt_scannet_target_diverse_balanced_bce_calib_train_2000",
)

config_doc["prompt_scannet_target_diverse_balanced_bce_calib_eval"] = (
    "Evaluate the best calibrated balanced-BCE (masked-patch) checkpoint."
)
config_defaults["prompt_scannet_target_diverse_balanced_bce_calib_eval"] = Options(
    **_PROMPT_TARGET_DIVERSE_BALANCED_BCE_CALIB_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "prompt_scannet_target_diverse_balanced_bce_calib_train_2000/"
        "model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_target_diverse_balanced_bce_calib_validation_best",
    experiment_name="prompt_scannet_target_diverse_balanced_bce_calib_validation_best",
)

_PROMPT_TARGET_DIVERSE_BALANCED_CLS_CALIB_COMMON = {
    **_PROMPT_TARGET_DIVERSE_BALANCED_CLS_COMMON,
    "prompt_balanced_bce_pos_weight": 0.3,
}

config_doc["prompt_scannet_target_diverse_balanced_cls_calib_smoke"] = (
    "Twenty-step smoke test for calibrated balanced BCE (pos weight 0.3) "
    "with masked-input-CLS pooling."
)
config_defaults["prompt_scannet_target_diverse_balanced_cls_calib_smoke"] = Options(
    **{**_PROMPT_TARGET_DIVERSE_BALANCED_CLS_CALIB_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/prompt_scannet_target_diverse_balanced_cls_calib_smoke_20",
    experiment_name="prompt_scannet_target_diverse_balanced_cls_calib_smoke_20",
)

config_doc["prompt_scannet_target_diverse_balanced_cls_calib_train"] = (
    "Two-thousand-step calibrated balanced-BCE (pos weight 0.3) training "
    "with masked-input-CLS pooling."
)
config_defaults["prompt_scannet_target_diverse_balanced_cls_calib_train"] = Options(
    **_PROMPT_TARGET_DIVERSE_BALANCED_CLS_CALIB_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_target_diverse_balanced_cls_calib_train_2000",
    experiment_name="prompt_scannet_target_diverse_balanced_cls_calib_train_2000",
)

config_doc["prompt_scannet_target_diverse_balanced_cls_calib_eval"] = (
    "Evaluate the best calibrated balanced-BCE (masked-input-CLS) checkpoint."
)
config_defaults["prompt_scannet_target_diverse_balanced_cls_calib_eval"] = Options(
    **_PROMPT_TARGET_DIVERSE_BALANCED_CLS_CALIB_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "prompt_scannet_target_diverse_balanced_cls_calib_train_2000/"
        "model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_target_diverse_balanced_cls_calib_validation_best",
    experiment_name="prompt_scannet_target_diverse_balanced_cls_calib_validation_best",
)

_PROMPT_TARGET_DIVERSE_BIGDATA_MANIFEST = (
    "/space0/mawb/tokengs/data/scannet_prompt/"
    "scannet_prompt_target_diverse_128_192.json"
)
_PROMPT_TARGET_DIVERSE_BIGDATA_COMMON = {
    **_PROMPT_TARGET_DIVERSE_COMMON,
    "dataset_kwargs": {"small_manifest_path": _PROMPT_TARGET_DIVERSE_BIGDATA_MANIFEST},
    "prompt_image_pooling": "masked_input_cls",
    "prompt_balanced_bce": True,
    "eval_n_media_dumps": 24,
}

config_doc["prompt_scannet_target_diverse_bigdata_smoke"] = (
    "Twenty-step smoke test on the 128-scene / 192-per-class manifest."
)
config_defaults["prompt_scannet_target_diverse_bigdata_smoke"] = Options(
    **{**_PROMPT_TARGET_DIVERSE_BIGDATA_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/prompt_scannet_target_diverse_bigdata_smoke_20",
    experiment_name="prompt_scannet_target_diverse_bigdata_smoke_20",
)

config_doc["prompt_scannet_target_diverse_bigdata_train"] = (
    "Four-thousand-step training on the 128-scene / 192-per-class manifest "
    "with masked-input-CLS pooling and balanced BCE (no last-cross fork)."
)
config_defaults["prompt_scannet_target_diverse_bigdata_train"] = Options(
    **_PROMPT_TARGET_DIVERSE_BIGDATA_COMMON,
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_target_diverse_bigdata_train_4000",
    experiment_name="prompt_scannet_target_diverse_bigdata_train_4000",
)

config_doc["prompt_scannet_target_diverse_bigdata_eval"] = (
    "Evaluate the best big-data checkpoint on the fixed 24-sample proxy split."
)
config_defaults["prompt_scannet_target_diverse_bigdata_eval"] = Options(
    **_PROMPT_TARGET_DIVERSE_BIGDATA_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "prompt_scannet_target_diverse_bigdata_train_4000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_target_diverse_bigdata_validation_best",
    experiment_name="prompt_scannet_target_diverse_bigdata_validation_best",
)

_PROMPT_TARGET_DIVERSE_BIGDATA_ADAPTER_COMMON = {
    **_PROMPT_TARGET_DIVERSE_BIGDATA_COMMON,
    "prompt_text_adapter": True,
}

config_doc["prompt_scannet_target_diverse_bigdata_adapter_smoke"] = (
    "Twenty-step smoke test for the trainable CLIP text adapter on the "
    "128-scene manifest."
)
config_defaults["prompt_scannet_target_diverse_bigdata_adapter_smoke"] = Options(
    **{**_PROMPT_TARGET_DIVERSE_BIGDATA_ADAPTER_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/prompt_scannet_target_diverse_bigdata_adapter_smoke_20",
    experiment_name="prompt_scannet_target_diverse_bigdata_adapter_smoke_20",
)

config_doc["prompt_scannet_target_diverse_bigdata_adapter_train"] = (
    "Four-thousand-step training on the 128-scene manifest with masked-input-CLS "
    "pooling, balanced BCE, and a trainable CLIP text adapter."
)
config_defaults["prompt_scannet_target_diverse_bigdata_adapter_train"] = Options(
    **_PROMPT_TARGET_DIVERSE_BIGDATA_ADAPTER_COMMON,
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_target_diverse_bigdata_adapter_train_4000",
    experiment_name="prompt_scannet_target_diverse_bigdata_adapter_train_4000",
)

config_doc["prompt_scannet_target_diverse_bigdata_adapter_eval"] = (
    "Evaluate the best big-data text-adapter checkpoint on the fixed "
    "24-sample proxy split."
)
config_defaults["prompt_scannet_target_diverse_bigdata_adapter_eval"] = Options(
    **_PROMPT_TARGET_DIVERSE_BIGDATA_ADAPTER_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "prompt_scannet_target_diverse_bigdata_adapter_train_4000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_target_diverse_bigdata_adapter_validation_best",
    experiment_name="prompt_scannet_target_diverse_bigdata_adapter_validation_best",
)

_SEMANTIC_V3_BIGDATA_COMMON = {
    **_PROMPT_TARGET_DIVERSE_BIGDATA_COMMON,
    "model_type": "semantic_tokengs_v3",
    "prompt_mode": "manifest",
    "semantic_v2_dim": 256,
    "semantic_v2_temperature_init": 14.285714,
    "semantic_v2_balanced_bce": True,
    "semantic_v2_score_mode": "sigmoid",
}

config_doc["semantic_v3_open_vocab_bigdata_smoke"] = (
    "Twenty-step open-vocabulary semantic_v3 smoke test on the 128-scene manifest."
)
config_defaults["semantic_v3_open_vocab_bigdata_smoke"] = Options(
    **{**_SEMANTIC_V3_BIGDATA_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v3_open_vocab_bigdata_smoke_20",
    experiment_name="semantic_v3_open_vocab_bigdata_smoke_20",
)

config_doc["semantic_v3_open_vocab_bigdata_train"] = (
    "Two-thousand-step open-vocabulary semantic_v3 training on the 128-scene "
    "manifest with prompt-conditioned prototypes and joint eight-class "
    "balanced-BCE supervision."
)
config_defaults["semantic_v3_open_vocab_bigdata_train"] = Options(
    **_SEMANTIC_V3_BIGDATA_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_bigdata_train_2000",
    experiment_name="semantic_v3_open_vocab_bigdata_train_2000",
)

config_doc["semantic_v3_open_vocab_bigdata_eval"] = (
    "Evaluate the best open-vocabulary semantic_v3 checkpoint on the fixed "
    "24-sample proxy split."
)
config_defaults["semantic_v3_open_vocab_bigdata_eval"] = Options(
    **_SEMANTIC_V3_BIGDATA_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v3_open_vocab_bigdata_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_bigdata_validation_best",
    experiment_name="semantic_v3_open_vocab_bigdata_validation_best",
)

_SEMANTIC_V3_FG02_COMMON = {
    **_SEMANTIC_V3_BIGDATA_COMMON,
    "dataset_kwargs": {
        "small_manifest_path": (
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "scannet_prompt_target_diverse_128_192_fg02.json"
        )
    },
}

config_doc["semantic_v3_open_vocab_fg02_smoke"] = (
    "Twenty-step v3 smoke test on the min-foreground-ratio-0.02 manifest."
)
config_defaults["semantic_v3_open_vocab_fg02_smoke"] = Options(
    **{**_SEMANTIC_V3_FG02_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v3_open_vocab_fg02_smoke_20",
    experiment_name="semantic_v3_open_vocab_fg02_smoke_20",
)

config_doc["semantic_v3_open_vocab_fg02_train"] = (
    "Two-thousand-step v3 training on the 128-scene manifest with target "
    "frames filtered to foreground ratio >= 0.02."
)
config_defaults["semantic_v3_open_vocab_fg02_train"] = Options(
    **_SEMANTIC_V3_FG02_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_fg02_train_2000",
    experiment_name="semantic_v3_open_vocab_fg02_train_2000",
)

config_doc["semantic_v3_open_vocab_fg02_eval"] = (
    "Evaluate the best fg02 v3 checkpoint on the fixed 24-sample proxy split."
)
config_defaults["semantic_v3_open_vocab_fg02_eval"] = Options(
    **_SEMANTIC_V3_FG02_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v3_open_vocab_fg02_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_fg02_validation_best",
    experiment_name="semantic_v3_open_vocab_fg02_validation_best",
)

_SEMANTIC_V3_FG03_COMMON = {
    **_SEMANTIC_V3_BIGDATA_COMMON,
    "dataset_kwargs": {
        "small_manifest_path": (
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "scannet_prompt_target_diverse_128_192_fg03.json"
        )
    },
}

config_doc["semantic_v3_open_vocab_fg03_smoke"] = (
    "Twenty-step v3 smoke test on the min-foreground-ratio-0.03 manifest."
)
config_defaults["semantic_v3_open_vocab_fg03_smoke"] = Options(
    **{**_SEMANTIC_V3_FG03_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v3_open_vocab_fg03_smoke_20",
    experiment_name="semantic_v3_open_vocab_fg03_smoke_20",
)

config_doc["semantic_v3_open_vocab_fg03_train"] = (
    "Two-thousand-step v3 training on the 128-scene manifest with target "
    "frames filtered to foreground ratio >= 0.03."
)
config_defaults["semantic_v3_open_vocab_fg03_train"] = Options(
    **_SEMANTIC_V3_FG03_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_fg03_train_2000",
    experiment_name="semantic_v3_open_vocab_fg03_train_2000",
)

config_doc["semantic_v3_open_vocab_fg03_eval"] = (
    "Evaluate the best fg03 v3 checkpoint on the fixed 24-sample proxy split."
)
config_defaults["semantic_v3_open_vocab_fg03_eval"] = Options(
    **_SEMANTIC_V3_FG03_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v3_open_vocab_fg03_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_fg03_validation_best",
    experiment_name="semantic_v3_open_vocab_fg03_validation_best",
)

_SEMANTIC_V3_FG02_TUNE_COMMON = {
    **_SEMANTIC_V3_FG02_COMMON,
    "semantic_v2_tune_last_cross_attention": True,
}

config_doc["semantic_v3_open_vocab_fg02_tune_smoke"] = (
    "Twenty-step v3 smoke test with a tunable semantic last cross-attention "
    "fork on the fg02 manifest."
)
config_defaults["semantic_v3_open_vocab_fg02_tune_smoke"] = Options(
    **{**_SEMANTIC_V3_FG02_TUNE_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v3_open_vocab_fg02_tune_smoke_20",
    experiment_name="semantic_v3_open_vocab_fg02_tune_smoke_20",
)

config_doc["semantic_v3_open_vocab_fg02_tune_train"] = (
    "Two-thousand-step v3 training on the fg02 manifest with a tunable "
    "semantic last cross-attention fork."
)
config_defaults["semantic_v3_open_vocab_fg02_tune_train"] = Options(
    **_SEMANTIC_V3_FG02_TUNE_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_fg02_tune_train_2000",
    experiment_name="semantic_v3_open_vocab_fg02_tune_train_2000",
)

config_doc["semantic_v3_open_vocab_fg02_tune_eval"] = (
    "Evaluate the best fg02-tune v3 checkpoint on the fixed 24-sample proxy."
)
config_defaults["semantic_v3_open_vocab_fg02_tune_eval"] = Options(
    **_SEMANTIC_V3_FG02_TUNE_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v3_open_vocab_fg02_tune_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_fg02_tune_validation_best",
    experiment_name="semantic_v3_open_vocab_fg02_tune_validation_best",
)

_SEMANTIC_V3_FG02_JOINT_COMMON = {
    **_SEMANTIC_V3_FG02_COMMON,
    "prompt_unfreeze_tokengs": True,
    "lambda_rgb": 50.0,
}

config_doc["semantic_v3_open_vocab_fg02_joint_smoke"] = (
    "Twenty-step v3 joint semantic+geometry training smoke test on fg02 data."
)
config_defaults["semantic_v3_open_vocab_fg02_joint_smoke"] = Options(
    **{**_SEMANTIC_V3_FG02_JOINT_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v3_open_vocab_fg02_joint_smoke_20",
    experiment_name="semantic_v3_open_vocab_fg02_joint_smoke_20",
)

config_doc["semantic_v3_open_vocab_fg02_joint_train"] = (
    "Two-thousand-step v3 joint training on the fg02 manifest with an "
    "unfrozen TokenGS (semantic mask loss + lambda_rgb reconstruction loss)."
)
config_defaults["semantic_v3_open_vocab_fg02_joint_train"] = Options(
    **_SEMANTIC_V3_FG02_JOINT_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_fg02_joint_train_2000",
    experiment_name="semantic_v3_open_vocab_fg02_joint_train_2000",
)

config_doc["semantic_v3_open_vocab_fg02_joint_eval"] = (
    "Evaluate the best fg02 joint v3 checkpoint on the fixed 24-sample proxy."
)
config_defaults["semantic_v3_open_vocab_fg02_joint_eval"] = Options(
    **_SEMANTIC_V3_FG02_JOINT_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v3_open_vocab_fg02_joint_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_fg02_joint_validation_best",
    experiment_name="semantic_v3_open_vocab_fg02_joint_validation_best",
)

_SEMANTIC_V3_FG02_JOINT2_COMMON = {
    **_SEMANTIC_V3_FG02_COMMON,
    "prompt_unfreeze_tokengs": True,
    "lambda_rgb": 200.0,
}

config_doc["semantic_v3_open_vocab_fg02_joint2_smoke"] = (
    "Twenty-step v3 joint training smoke test with a stronger RGB anchor."
)
config_defaults["semantic_v3_open_vocab_fg02_joint2_smoke"] = Options(
    **{**_SEMANTIC_V3_FG02_JOINT2_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v3_open_vocab_fg02_joint2_smoke_20",
    experiment_name="semantic_v3_open_vocab_fg02_joint2_smoke_20",
)

config_doc["semantic_v3_open_vocab_fg02_joint2_train"] = (
    "Four-thousand-step v3 joint training on the fg02 manifest with "
    "unfrozen TokenGS and lambda_rgb=200 (stronger reconstruction anchor)."
)
config_defaults["semantic_v3_open_vocab_fg02_joint2_train"] = Options(
    **_SEMANTIC_V3_FG02_JOINT2_COMMON,
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_fg02_joint2_train_4000",
    experiment_name="semantic_v3_open_vocab_fg02_joint2_train_4000",
)

config_doc["semantic_v3_open_vocab_fg02_joint2_eval"] = (
    "Evaluate the best fg02 joint2 v3 checkpoint on the fixed 24-sample proxy."
)
config_defaults["semantic_v3_open_vocab_fg02_joint2_eval"] = Options(
    **_SEMANTIC_V3_FG02_JOINT2_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v3_open_vocab_fg02_joint2_train_4000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_fg02_joint2_validation_best",
    experiment_name="semantic_v3_open_vocab_fg02_joint2_validation_best",
)

_SEMANTIC_V3_FG02_JOINT3_COMMON = {
    **_SEMANTIC_V3_FG02_COMMON,
    "prompt_unfreeze_tokengs": True,
    "prompt_unfreeze_tokengs_lr": 1e-5,
    "lambda_rgb": 200.0,
}

config_doc["semantic_v3_open_vocab_fg02_joint3_smoke"] = (
    "Twenty-step v3 joint training smoke test with a low geometry LR."
)
config_defaults["semantic_v3_open_vocab_fg02_joint3_smoke"] = Options(
    **{**_SEMANTIC_V3_FG02_JOINT3_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v3_open_vocab_fg02_joint3_smoke_20",
    experiment_name="semantic_v3_open_vocab_fg02_joint3_smoke_20",
)

config_doc["semantic_v3_open_vocab_fg02_joint3_train"] = (
    "Four-thousand-step v3 joint training on the fg02 manifest with "
    "unfrozen TokenGS, lambda_rgb=200, and a 10x lower geometry learning rate."
)
config_defaults["semantic_v3_open_vocab_fg02_joint3_train"] = Options(
    **_SEMANTIC_V3_FG02_JOINT3_COMMON,
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_fg02_joint3_train_4000",
    experiment_name="semantic_v3_open_vocab_fg02_joint3_train_4000",
)

config_doc["semantic_v3_open_vocab_fg02_joint3_eval"] = (
    "Evaluate the best fg02 joint3 v3 checkpoint on the fixed 24-sample proxy."
)
config_defaults["semantic_v3_open_vocab_fg02_joint3_eval"] = Options(
    **_SEMANTIC_V3_FG02_JOINT3_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v3_open_vocab_fg02_joint3_train_4000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_fg02_joint3_validation_best",
    experiment_name="semantic_v3_open_vocab_fg02_joint3_validation_best",
)

_SEMANTIC_V3_FG02_JOINT4_COMMON = {
    **_SEMANTIC_V3_FG02_COMMON,
    "prompt_unfreeze_tokengs": True,
    "prompt_unfreeze_tokengs_lr": 3e-5,
    "lambda_rgb": 200.0,
}

config_doc["semantic_v3_open_vocab_fg02_joint4_smoke"] = (
    "Twenty-step v3 joint training smoke test with geometry LR 3e-5."
)
config_defaults["semantic_v3_open_vocab_fg02_joint4_smoke"] = Options(
    **{**_SEMANTIC_V3_FG02_JOINT4_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v3_open_vocab_fg02_joint4_smoke_20",
    experiment_name="semantic_v3_open_vocab_fg02_joint4_smoke_20",
)

config_doc["semantic_v3_open_vocab_fg02_joint4_train"] = (
    "Four-thousand-step v3 joint training on the fg02 manifest with "
    "unfrozen TokenGS, lambda_rgb=200, and geometry LR 3e-5."
)
config_defaults["semantic_v3_open_vocab_fg02_joint4_train"] = Options(
    **_SEMANTIC_V3_FG02_JOINT4_COMMON,
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_fg02_joint4_train_4000",
    experiment_name="semantic_v3_open_vocab_fg02_joint4_train_4000",
)

config_doc["semantic_v3_open_vocab_fg02_joint4_eval"] = (
    "Evaluate the best fg02 joint4 v3 checkpoint on the fixed 24-sample proxy."
)
config_defaults["semantic_v3_open_vocab_fg02_joint4_eval"] = Options(
    **_SEMANTIC_V3_FG02_JOINT4_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v3_open_vocab_fg02_joint4_train_4000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_fg02_joint4_validation_best",
    experiment_name="semantic_v3_open_vocab_fg02_joint4_validation_best",
)

_SEMANTIC_V4_FG02_COMMON = {
    **_SEMANTIC_V3_FG02_COMMON,
    "model_type": "semantic_tokengs_v4",
    "semantic_v4_feature_dim": 32,
    "semantic_v4_use_geometry": True,
    "semantic_v4_teacher_projection": "frozen_random",
    "lambda_feat": 1.0,
}

_SEMANTIC_V4_FG02_JOINT4_COMMON = {
    **_SEMANTIC_V4_FG02_COMMON,
    "prompt_unfreeze_tokengs": True,
    "prompt_unfreeze_tokengs_lr": 3e-5,
    "lambda_rgb": 200.0,
}

config_doc["semantic_v4_open_vocab_fg02_joint4_smoke"] = (
    "Twenty-step v4 smoke test: per-Gaussian feature field with dense CLIP "
    "distillation and unfrozen TokenGS (geometry LR 3e-5)."
)
config_defaults["semantic_v4_open_vocab_fg02_joint4_smoke"] = Options(
    **{**_SEMANTIC_V4_FG02_JOINT4_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v4_open_vocab_fg02_joint4_smoke_20",
    experiment_name="semantic_v4_open_vocab_fg02_joint4_smoke_20",
)

config_doc["semantic_v4_open_vocab_fg02_joint4_train"] = (
    "Four-thousand-step v4 joint training on the fg02 manifest: per-Gaussian "
    "feature field + dense CLIP feature distillation + unfrozen TokenGS "
    "(geometry LR 3e-5), lambda_feat=1.0, lambda_rgb=200."
)
config_defaults["semantic_v4_open_vocab_fg02_joint4_train"] = Options(
    **_SEMANTIC_V4_FG02_JOINT4_COMMON,
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v4_open_vocab_fg02_joint4_train_4000",
    experiment_name="semantic_v4_open_vocab_fg02_joint4_train_4000",
)

config_doc["semantic_v4_open_vocab_fg02_joint4_eval"] = (
    "Evaluate the best fg02 joint4 v4 checkpoint on the fixed 24-sample proxy."
)
config_defaults["semantic_v4_open_vocab_fg02_joint4_eval"] = Options(
    **_SEMANTIC_V4_FG02_JOINT4_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v4_open_vocab_fg02_joint4_train_4000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v4_open_vocab_fg02_joint4_validation_best",
    experiment_name="semantic_v4_open_vocab_fg02_joint4_validation_best",
)

_SEMANTIC_V4_FG02_JOINT4_VITL_COMMON = {
    **_SEMANTIC_V4_FG02_JOINT4_COMMON,
    "prompt_clip_model_path": (
        "/space0/mawb/tokengs/checkpoints/clip-vit-large-patch14"
    ),
    "semantic_v4_feature_dim": 64,
    # sqrt inverse foreground-pixel frequency over the fg02 training manifest
    # (wall, floor, ceiling, chair, table, sofa, bed, other), mean-normalized.
    "semantic_v2_class_weights": (
        0.956, 0.907, 1.057, 1.325, 1.256, 0.868, 0.896, 0.736,
    ),
}

config_doc["semantic_v4_open_vocab_fg02_joint4_vitl_smoke"] = (
    "Twenty-step v4 smoke with CLIP ViT-L/14 dense teacher, feature_dim=64 "
    "and frequency-based class weights."
)
config_defaults["semantic_v4_open_vocab_fg02_joint4_vitl_smoke"] = Options(
    **{**_SEMANTIC_V4_FG02_JOINT4_VITL_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v4_open_vocab_fg02_joint4_vitl_smoke_20",
    experiment_name="semantic_v4_open_vocab_fg02_joint4_vitl_smoke_20",
)

config_doc["semantic_v4_open_vocab_fg02_joint4_vitl_train"] = (
    "Four-thousand-step v4 training with a CLIP ViT-L/14 dense teacher "
    "(16x16 patch grid), feature_dim=64, frequency-based class weights, "
    "unfrozen TokenGS (geometry LR 3e-5)."
)
config_defaults["semantic_v4_open_vocab_fg02_joint4_vitl_train"] = Options(
    **_SEMANTIC_V4_FG02_JOINT4_VITL_COMMON,
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v4_open_vocab_fg02_joint4_vitl_train_4000",
    experiment_name="semantic_v4_open_vocab_fg02_joint4_vitl_train_4000",
)

config_doc["semantic_v4_open_vocab_fg02_joint4_vitl_eval"] = (
    "Evaluate the best fg02 joint4 v4-ViT-L checkpoint on the 24-sample proxy."
)
config_defaults["semantic_v4_open_vocab_fg02_joint4_vitl_eval"] = Options(
    **_SEMANTIC_V4_FG02_JOINT4_VITL_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v4_open_vocab_fg02_joint4_vitl_train_4000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v4_open_vocab_fg02_joint4_vitl_validation_best",
    experiment_name="semantic_v4_open_vocab_fg02_joint4_vitl_validation_best",
)

_SEMANTIC_V5_FG02_JOINT4_LSEG_COMMON = {
    **_SEMANTIC_V4_FG02_JOINT4_VITL_COMMON,
    "model_type": "semantic_tokengs_v5",
    "lseg_checkpoint_path": (
        "/space0/mawb/tokengs/checkpoints/demo_e200.ckpt"
    ),
    "lambda_ce": 1.0,
}

config_doc["semantic_v5_open_vocab_fg02_joint4_lseg_smoke"] = (
    "Twenty-step v5 smoke: per-Gaussian field distilled from frozen LSeg "
    "features plus a pixel decode head supervised by ScanNet GT labels."
)
config_defaults["semantic_v5_open_vocab_fg02_joint4_lseg_smoke"] = Options(
    **{**_SEMANTIC_V5_FG02_JOINT4_LSEG_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v5_open_vocab_fg02_joint4_lseg_smoke_20",
    experiment_name="semantic_v5_open_vocab_fg02_joint4_lseg_smoke_20",
)

config_doc["semantic_v5_open_vocab_fg02_joint4_lseg_train"] = (
    "Four-thousand-step v5 training: frozen LSeg teacher feature lifting, "
    "pixel decode head with GT eight-class CE, ViT-L prompt encoder, "
    "unfrozen TokenGS (geometry LR 3e-5)."
)
config_defaults["semantic_v5_open_vocab_fg02_joint4_lseg_train"] = Options(
    **_SEMANTIC_V5_FG02_JOINT4_LSEG_COMMON,
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v5_open_vocab_fg02_joint4_lseg_train_4000",
    experiment_name="semantic_v5_open_vocab_fg02_joint4_lseg_train_4000",
)

config_doc["semantic_v5_open_vocab_fg02_joint4_lseg_eval"] = (
    "Evaluate the best fg02 joint4 v5-LSeg checkpoint on the 24-sample proxy."
)
config_defaults["semantic_v5_open_vocab_fg02_joint4_lseg_eval"] = Options(
    **_SEMANTIC_V5_FG02_JOINT4_LSEG_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v5_open_vocab_fg02_joint4_lseg_train_4000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v5_open_vocab_fg02_joint4_lseg_validation_best",
    experiment_name="semantic_v5_open_vocab_fg02_joint4_lseg_validation_best",
)

_SEMANTIC_V4_FG02_JOINT4_VITL_CE_COMMON = {
    **_SEMANTIC_V4_FG02_JOINT4_VITL_COMMON,
    "lambda_ce_cosine": 1.0,
}

config_doc["semantic_v4_open_vocab_fg02_joint4_vitl_ce_smoke"] = (
    "Twenty-step v4-ViT-L smoke with a softmax-CE term on the cosine path "
    "(explicit eight-class competition, background ignored)."
)
config_defaults["semantic_v4_open_vocab_fg02_joint4_vitl_ce_smoke"] = Options(
    **{**_SEMANTIC_V4_FG02_JOINT4_VITL_CE_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v4_open_vocab_fg02_joint4_vitl_ce_smoke_20",
    experiment_name="semantic_v4_open_vocab_fg02_joint4_vitl_ce_smoke_20",
)

config_doc["semantic_v4_open_vocab_fg02_joint4_vitl_ce_train"] = (
    "Four-thousand-step v4-ViT-L training with cosine-path softmax-CE: the "
    "positive prompt must beat the other seven prototypes per pixel "
    "(background ignored), matching the C3G argmax protocol."
)
config_defaults["semantic_v4_open_vocab_fg02_joint4_vitl_ce_train"] = Options(
    **_SEMANTIC_V4_FG02_JOINT4_VITL_CE_COMMON,
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v4_open_vocab_fg02_joint4_vitl_ce_train_4000",
    experiment_name="semantic_v4_open_vocab_fg02_joint4_vitl_ce_train_4000",
)

_SEMANTIC_V5_FG02_JOINT4_LSEG_CE_COMMON = {
    **_SEMANTIC_V5_FG02_JOINT4_LSEG_COMMON,
    "lambda_ce_cosine": 1.0,
}

config_doc["semantic_v5_open_vocab_fg02_joint4_lseg_ce_smoke"] = (
    "Twenty-step v5-LSeg smoke with an additional softmax-CE term on the "
    "cosine path (shared feature field becomes class-competitive)."
)
config_defaults["semantic_v5_open_vocab_fg02_joint4_lseg_ce_smoke"] = Options(
    **{**_SEMANTIC_V5_FG02_JOINT4_LSEG_CE_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v5_open_vocab_fg02_joint4_lseg_ce_smoke_20",
    experiment_name="semantic_v5_open_vocab_fg02_joint4_lseg_ce_smoke_20",
)

config_doc["semantic_v5_open_vocab_fg02_joint4_lseg_ce_train"] = (
    "Four-thousand-step v5-LSeg training with cosine-path softmax-CE: LSeg "
    "feature lifting + pixel decode head + explicit eight-class competition "
    "on the shared feature field."
)
config_defaults["semantic_v5_open_vocab_fg02_joint4_lseg_ce_train"] = Options(
    **_SEMANTIC_V5_FG02_JOINT4_LSEG_CE_COMMON,
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v5_open_vocab_fg02_joint4_lseg_ce_train_4000",
    experiment_name="semantic_v5_open_vocab_fg02_joint4_lseg_ce_train_4000",
)

_PROMPT_FULL_MANIFEST = (
    "/space0/mawb/tokengs/data/scannet_prompt/scannet_prompt_full_ratio02_16f.json"
)

_SEMANTIC_V4_FULL_CE_COMMON = {
    **_SEMANTIC_V4_FG02_JOINT4_VITL_CE_COMMON,
    "dataset_kwargs": {"small_manifest_path": _PROMPT_FULL_MANIFEST},
    # sqrt inverse foreground-pixel frequency computed over the full
    # 1425-scene manifest (scannet_prompt_full_ratio02_16f.json), mean-normalized.
    "semantic_v2_class_weights": (
        0.8869, 0.9536, 1.2475, 1.2257, 1.1961, 0.9201, 0.8012, 0.7689,
    ),
}

config_doc["semantic_v4_open_vocab_full_ce_smoke"] = (
    "Twenty-step v4-ViT-L smoke on the full 1425-scene manifest, warm "
    "started from the best fg02 v4-CE checkpoint (step 3200)."
)
config_defaults["semantic_v4_open_vocab_full_ce_smoke"] = Options(
    **{**_SEMANTIC_V4_FULL_CE_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v4_open_vocab_fg02_joint4_vitl_ce_train_4000/"
        "checkpoints/model_step_003200.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v4_open_vocab_full_ce_smoke_20",
    experiment_name="semantic_v4_open_vocab_full_ce_smoke_20",
)

config_doc["semantic_v4_open_vocab_full_ce_train"] = (
    "Four-thousand-step v4-ViT-L cosine-CE training on the full 1425-scene "
    "manifest (5680 samples, 710/class), warm started from the best fg02 "
    "v4-CE checkpoint (step 3200), with class weights recomputed over the "
    "full manifest."
)
config_defaults["semantic_v4_open_vocab_full_ce_train"] = Options(
    **_SEMANTIC_V4_FULL_CE_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v4_open_vocab_fg02_joint4_vitl_ce_train_4000/"
        "checkpoints/model_step_003200.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v4_open_vocab_full_ce_train_4000",
    experiment_name="semantic_v4_open_vocab_full_ce_train_4000",
)

config_doc["semantic_v4_open_vocab_full_ce_eval"] = (
    "Evaluate the full-data v4-CE checkpoint on the fixed 24-sample proxy split."
)
config_defaults["semantic_v4_open_vocab_full_ce_eval"] = Options(
    **_SEMANTIC_V4_FULL_CE_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v4_open_vocab_full_ce_train_4000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v4_open_vocab_full_ce_validation_best",
    experiment_name="semantic_v4_open_vocab_full_ce_validation_best",
)

_SEMANTIC_V5_FULL_CE_COMMON = {
    **_SEMANTIC_V5_FG02_JOINT4_LSEG_CE_COMMON,
    "dataset_kwargs": {"small_manifest_path": _PROMPT_FULL_MANIFEST},
    # sqrt inverse foreground-pixel frequency computed over the full
    # 1425-scene manifest (scannet_prompt_full_ratio02_16f.json), mean-normalized.
    "semantic_v2_class_weights": (
        0.8869, 0.9536, 1.2475, 1.2257, 1.1961, 0.9201, 0.8012, 0.7689,
    ),
}

config_doc["semantic_v5_open_vocab_full_ce_smoke"] = (
    "Twenty-step v5-LSeg smoke on the full 1425-scene manifest, warm "
    "started from the best fg02 v5-CE checkpoint (step 3200)."
)
config_defaults["semantic_v5_open_vocab_full_ce_smoke"] = Options(
    **{**_SEMANTIC_V5_FULL_CE_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v5_open_vocab_fg02_joint4_lseg_ce_train_4000/"
        "checkpoints/model_step_003200.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v5_open_vocab_full_ce_smoke_20",
    experiment_name="semantic_v5_open_vocab_full_ce_smoke_20",
)

config_doc["semantic_v5_open_vocab_full_ce_train"] = (
    "Four-thousand-step v5-LSeg cosine-CE training on the full 1425-scene "
    "manifest (5680 samples, 710/class), warm started from the best fg02 "
    "v5-CE checkpoint (step 3200), with class weights recomputed over the "
    "full manifest."
)
config_defaults["semantic_v5_open_vocab_full_ce_train"] = Options(
    **_SEMANTIC_V5_FULL_CE_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v5_open_vocab_fg02_joint4_lseg_ce_train_4000/"
        "checkpoints/model_step_003200.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v5_open_vocab_full_ce_train_4000",
    experiment_name="semantic_v5_open_vocab_full_ce_train_4000",
)

config_doc["semantic_v5_open_vocab_full_ce_eval"] = (
    "Evaluate the full-data v5-CE checkpoint on the fixed 24-sample proxy split."
)
config_defaults["semantic_v5_open_vocab_full_ce_eval"] = Options(
    **_SEMANTIC_V5_FULL_CE_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v5_open_vocab_full_ce_train_4000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v5_open_vocab_full_ce_validation_best",
    experiment_name="semantic_v5_open_vocab_full_ce_validation_best",
)

_SEMANTIC_V6_FULL_CE_COMMON = {
    **_SEMANTIC_V4_FULL_CE_COMMON,
    "model_type": "semantic_tokengs_v6",
    "use_instance_labels": True,
    "instance_group_num_groups": 64,
    "instance_group_min_instance_pixels": 64,
    "lambda_instance_group_dice": 1.0,
    "lambda_instance_group_mask": 1.0,
    "lambda_instance_group_void": 0.1,
    "lambda_instance_group_unmatched": 0.1,
}

config_doc["semantic_v6_open_vocab_full_ce_smoke"] = (
    "Twenty-step v6 smoke on the full 1425-scene manifest with instance "
    "group supervision, warm started from the best full-data v4-CE "
    "checkpoint (step 2000)."
)
config_defaults["semantic_v6_open_vocab_full_ce_smoke"] = Options(
    **{**_SEMANTIC_V6_FULL_CE_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v4_open_vocab_full_ce_train_4000/"
        "checkpoints/model_step_002000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_full_ce_smoke_20",
    experiment_name="semantic_v6_open_vocab_full_ce_smoke_20",
)

config_doc["semantic_v6_open_vocab_full_ce_train"] = (
    "Four-thousand-step v6 training on the full 1425-scene manifest: v4-CE "
    "open-vocabulary semantics plus a 64-group instance assignment head "
    "supervised by ScanNet 2D instance masks (Hungarian BCE+Dice+void). "
    "Warm started from the best full-data v4-CE checkpoint (step 2000)."
)
config_defaults["semantic_v6_open_vocab_full_ce_train"] = Options(
    **_SEMANTIC_V6_FULL_CE_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v4_open_vocab_full_ce_train_4000/"
        "checkpoints/model_step_002000.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce_train_4000",
    experiment_name="semantic_v6_open_vocab_full_ce_train_4000",
)

config_doc["semantic_v6_open_vocab_full_ce_eval"] = (
    "Evaluate the full-data v6-CE checkpoint on the fixed 24-sample proxy split."
)
config_defaults["semantic_v6_open_vocab_full_ce_eval"] = Options(
    **_SEMANTIC_V6_FULL_CE_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce_train_4000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce_validation_best",
    experiment_name="semantic_v6_open_vocab_full_ce_validation_best",
)

_SEMANTIC_V6_FULL_CE2_COMMON = {
    **_SEMANTIC_V6_FULL_CE_COMMON,
    "instance_group_num_groups": 128,
    "instance_group_min_instance_pixels": 32,
    "lambda_instance_group_dice": 1.0,
    "lambda_instance_group_mask": 1.0,
    "lambda_instance_group_void": 0.1,
    "lambda_instance_group_unmatched": 0.1,
    "instance_group_lambda_warmup_steps": 1500,
}

config_doc["semantic_v6_open_vocab_full_ce2_smoke"] = (
    "Twenty-step smoke of the InstOk3D-style grouping loss: full Hungarian "
    "assignment (every GT instance covered), void in the assignment softmax, "
    "and lambda warm-up over 1500 steps. Warm started from v6@4000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_smoke"] = Options(
    **{**_SEMANTIC_V6_FULL_CE2_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce_train_4000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_smoke_20",
    experiment_name="semantic_v6_open_vocab_full_ce2_smoke_20",
)

config_doc["semantic_v6_open_vocab_full_ce2_train"] = (
    "Four-thousand-step v6 training with the InstOk3D-style grouping loss: "
    "full Hungarian assignment so every GT instance is supervised, void "
    "channel inside the assignment softmax, 128 groups, and a 1500-step "
    "lambda warm-up. Continues from v6@4000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_train"] = Options(
    **_SEMANTIC_V6_FULL_CE2_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce_train_4000/model.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_train_4000",
    experiment_name="semantic_v6_open_vocab_full_ce2_train_4000",
)

config_doc["semantic_v6_open_vocab_full_ce2_eval"] = (
    "Evaluate the ce2 v6 checkpoint on the fixed 24-sample proxy split."
)
config_defaults["semantic_v6_open_vocab_full_ce2_eval"] = Options(
    **_SEMANTIC_V6_FULL_CE2_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_train_4000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_validation_best",
    experiment_name="semantic_v6_open_vocab_full_ce2_validation_best",
)

_WIDE_MANIFEST = (
    "/space0/mawb/tokengs/data/scannet_prompt/"
    "scannet_prompt_full_wide_8x7.json"
)
_SEMANTIC_V6_FULL_CE2_WIDE_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_COMMON,
    "dataset_kwargs": {
        "small_manifest_path": _WIDE_MANIFEST,
        "wide_target_subsample": 2,
    },
    "num_input_views": 8,
    "num_views": 15,
}

config_doc["semantic_v6_open_vocab_full_ce2_wide_smoke"] = (
    "Twenty-step smoke of wide-baseline 8-context / 7-test training with "
    "2 supervised targets per step, warm started from v6-ce2@4000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide_smoke"] = Options(
    **{**_SEMANTIC_V6_FULL_CE2_WIDE_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_train_4000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide_smoke_20",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide_smoke_20",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide_train"] = (
    "Wide-baseline v6-ce2 training aligned with the LSM instance protocol: "
    "8 context views + 7 interleaved test views per sample (2 targets "
    "supervised per step), so reconstruction and grouping are learned on the "
    "same view distribution as evaluation. Continues from v6-ce2@4000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide_train"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_train_4000/model.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide_train_4000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide_train_4000",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide_eval"] = (
    "Evaluate the wide-baseline v6-ce2 checkpoint on the 24-sample proxy split."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide_eval"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide_train_4000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide_validation_best",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide_validation_best",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide_train_8000"] = (
    "Continuation of wide-baseline v6-ce2 training to 8000 total steps "
    "(4000 more steps), resuming from v6-wide@4000, to push instance AP "
    "further on the LSM-aligned view distribution."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide_train_8000"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide_train_4000/model.safetensors"
    ),
    num_epochs=40,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide_train_8000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide_train_8000",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide_eval_8000"] = (
    "Evaluate the 8000-step wide v6-ce2 checkpoint on the 24-sample proxy."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide_eval_8000"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide_train_8000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide_validation_best_8000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide_validation_best_8000",
)

_SEMANTIC_V6_FULL_CE2_WIDE7_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_WIDE_COMMON,
    "dataset_kwargs": {
        "small_manifest_path": _WIDE_MANIFEST,
        "wide_target_subsample": 0,  # supervise all 7 target views per step
    },
    "instance_group_lambda_warmup_steps": 0,  # instance head already trained
    "instance_group_area_alpha": 0.5,  # up-weight small GT instances
    "instance_group_match_area_norm": True,  # small-instance friendly matching
}

config_doc["semantic_v6_open_vocab_full_ce2_wide7_smoke"] = (
    "Twenty-step smoke of the wide7 variant: all 7 target views supervised "
    "per step (wide_target_subsample=0), no lambda warmup, area-aware pair "
    "weighting and area-normalized matching cost. Warm started from "
    "v6-wide@8000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7_smoke"] = Options(
    **{**_SEMANTIC_V6_FULL_CE2_WIDE7_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide_train_8000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7_smoke_20",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7_smoke_20",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide7_train"] = (
    "Wide7 training: same wide-baseline 8-context / 7-test samples as "
    "v6-wide, but every target view is supervised per step (7 targets), "
    "with area-aware instance pair weighting and area-normalized matching "
    "cost to raise small-instance coverage. No lambda warmup; warm starts "
    "from v6-wide@8000 and runs 4000 more steps."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7_train"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE7_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide_train_8000/model.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7_train_4000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7_train_4000",
)

_SEMANTIC_V6_FULL_CE2_WIDE7B_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_WIDE7_COMMON,
    # 10x weaker unmatched-group suppression: tests whether the group
    # collapse (only ~4 active groups per scene vs ~60 GT instances) is
    # driven by the unmatched pressure pushing non-matched groups to void.
    "lambda_instance_group_unmatched": 0.01,
}

config_doc["semantic_v6_open_vocab_full_ce2_wide7b_smoke"] = (
    "Twenty-step smoke of the wide7b variant (unmatched pressure 0.1 -> "
    "0.01) warm started from v6-wide7@4000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7b_smoke"] = Options(
    **{**_SEMANTIC_V6_FULL_CE2_WIDE7B_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7_train_4000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7b_smoke_20",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7b_smoke_20",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide7b_train"] = (
    "Wide7b: wide7 (7 targets/step, area-aware weighting, area-normalized "
    "matching) plus 10x weaker unmatched-group suppression. Goal: stop "
    "non-matched groups from being driven to void so that more instance "
    "groups stay active and small/medium instances gain pixels. Warm starts "
    "from v6-wide7@4000 and runs 4000 steps."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7b_train"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE7B_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7_train_4000/model.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7b_train_4000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7b_train_4000",
)

_SEMANTIC_V6_FULL_CE2_WIDE7C_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_WIDE7_COMMON,
    # Per-pixel cross-entropy over the G+1 group channels: makes groups
    # compete for pixels (BCE+Dice alone lets coarse groups swallow other
    # groups' instances). This is the anti-"4-group collapse" term.
    "lambda_instance_group_ce": 1.0,
}

config_doc["semantic_v6_open_vocab_full_ce2_wide7c_smoke"] = (
    "Twenty-step smoke of the wide7c variant (per-pixel group CE added, "
    "unmatched pressure back to 0.1) warm started from v6-wide7@4000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7c_smoke"] = Options(
    **{**_SEMANTIC_V6_FULL_CE2_WIDE7C_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7_train_4000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7c_smoke_20",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7c_smoke_20",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide7c_train"] = (
    "Wide7c: wide7 (7 targets/step, area-aware weighting, area-normalized "
    "matching, unmatched 0.1) plus a per-pixel cross-entropy term over the "
    "rendered group channels so groups compete for pixels. Target: break "
    "the coarse 4-group collapse (walls/floors swallow small instances). "
    "Warm starts from v6-wide7@4000 and runs 4000 steps."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7c_train"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE7C_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7_train_4000/model.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7c_train_4000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7c_train_4000",
)

_SEMANTIC_V6_FULL_CE2_WIDE7D_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_WIDE7_COMMON,
    # InstOk3D-style scene-adaptive group decoder: learnable group queries
    # cross-attend to the token hidden states, so groups specialize to the
    # instances of each scene instead of collapsing onto 4 global coarse
    # groups (walls/floors/ceilings). Short lambda warmup for the fresh head.
    "instance_group_decoder": True,
    "instance_group_decoder_layers": 2,
    "instance_group_lambda_warmup_steps": 500,
}

config_doc["semantic_v6_open_vocab_full_ce2_wide7d_smoke"] = (
    "Twenty-step smoke of the wide7d variant (scene-adaptive group decoder) "
    "warm started from v6-wide7@4000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7d_smoke"] = Options(
    **{**_SEMANTIC_V6_FULL_CE2_WIDE7D_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7_train_4000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7d_smoke_20",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7d_smoke_20",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide7d_train"] = (
    "Wide7d: InstOk3D-style scene-adaptive group decoder on top of the "
    "wide7 setup (7 targets/step, area-aware weighting, area-normalized "
    "matching). The old linear group head collapsed onto 4 global coarse "
    "groups; learnable group queries with cross-attention should let groups "
    "specialize per scene and cover small/medium instances. Warm starts "
    "from v6-wide7@4000 (group head reinitialized) and runs 4000 steps."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7d_train"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE7D_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7_train_4000/model.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7d_train_4000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7d_train_4000",
)

_SEMANTIC_V6_FULL_CE2_WIDE7E_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_WIDE7_COMMON,
    # Anti-collapse mechanisms: (1) top-k Hungarian matching so ~3x more
    # groups receive positive gradient per view; (2) weak group-usage
    # entropy (KL from uniform) so a few groups cannot hoard all tokens.
    "instance_group_match_topk": 3,
    "instance_group_secondary_pair_weight": 0.3,
    "instance_group_usage_entropy": 0.05,
}

config_doc["semantic_v6_open_vocab_full_ce2_wide7e_smoke"] = (
    "Twenty-step smoke of the wide7e variant (top-k=3 matching + usage "
    "entropy 0.05) warm started from v6-wide7@4000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7e_smoke"] = Options(
    **{**_SEMANTIC_V6_FULL_CE2_WIDE7E_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7_train_4000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7e_smoke_20",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7e_smoke_20",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide7e_train"] = (
    "Wide7e: wide7 plus anti-collapse mechanisms -- top-k=3 Hungarian "
    "matching (each GT instance claims up to 3 groups, secondary weight "
    "0.3) and weak group-usage entropy regularization (0.05). Goal: force "
    "more than the ~4 coarse groups to stay active so small/medium "
    "instances gain pixels. Warm starts from v6-wide7@4000 and runs 4000 "
    "steps."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7e_train"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE7E_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7_train_4000/model.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7e_train_4000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7e_train_4000",
)

_SEMANTIC_V6_FULL_CE2_WIDE7F_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_WIDE7E_COMMON,
    # Per-pixel CE with the primary group as target: top-k=3 created good
    # coverage (AP25 0.37 -> 0.48) but fragmented each instance across
    # ~2 groups (AP50 dropped). CE makes groups compete per pixel with the
    # primary winner as target, consolidating fragments while keeping the
    # anti-collapse coverage.
    "lambda_instance_group_ce": 1.0,
}

config_doc["semantic_v6_open_vocab_full_ce2_wide7f_smoke"] = (
    "Twenty-step smoke of the wide7f variant (wide7e + per-pixel group CE) "
    "warm started from v6-wide7e@4000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7f_smoke"] = Options(
    **{**_SEMANTIC_V6_FULL_CE2_WIDE7F_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7e_train_4000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7f_smoke_20",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7f_smoke_20",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide7f_train"] = (
    "Wide7f: wide7e (top-k=3 + usage entropy, L1-rendered probs) plus a "
    "per-pixel cross-entropy term targeting the primary matched group, to "
    "consolidate the fragmented instance masks produced by top-k while "
    "keeping the improved coverage. Warm starts from v6-wide7e@4000 and "
    "runs 4000 steps."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7f_train"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE7F_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7e_train_4000/model.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7f_train_4000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7f_train_4000",
)

_SEMANTIC_V6_FULL_CE2_WIDE7G_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_WIDE7F_COMMON,
    # wide7f diagnostics: still ~1.3x over-segmentation and the void channel
    # wins ~20% of instance pixels (token-level void prob ~0.22). Reduce
    # top-k to 2 (fewer secondary fragments) and raise the void weight to
    # push void probability off instance pixels.
    "instance_group_match_topk": 2,
    "lambda_instance_group_void": 0.3,
}

config_doc["semantic_v6_open_vocab_full_ce2_wide7g_smoke"] = (
    "Twenty-step smoke of the wide7g variant (topk=2, void weight 0.3) "
    "warm started from v6-wide7f@4000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7g_smoke"] = Options(
    **{**_SEMANTIC_V6_FULL_CE2_WIDE7G_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7f_train_4000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7g_smoke_20",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7g_smoke_20",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide7g_train"] = (
    "Wide7g: wide7f (topk=3 + entropy + CE) with topk reduced to 2 and the "
    "void weight raised 0.1 -> 0.3, targeting the remaining "
    "over-segmentation (~1.3x pred/GT) and the void channel winning ~20% "
    "of instance pixels. Warm starts from v6-wide7f@4000 and runs 4000 "
    "steps."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7g_train"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE7G_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7f_train_4000/model.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7g_train_4000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7g_train_4000",
)

_SEMANTIC_V6_FULL_CE2_WIDE7H_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_WIDE7F_COMMON,
    # Retry the scene-adaptive group decoder on top of the fixed (L1)
    # rendering pipeline and the full anti-collapse loss stack (topk=3,
    # usage entropy, per-pixel CE). The earlier decoder run (wide7d) used
    # the double-softmax-flattened gradients and collapsed; with sharp
    # gradients the scene-adaptive queries should specialize per scene and
    # tighten instance masks. Short lambda warmup for the fresh head.
    "instance_group_decoder": True,
    "instance_group_decoder_layers": 2,
    "instance_group_lambda_warmup_steps": 500,
}

config_doc["semantic_v6_open_vocab_full_ce2_wide7h_smoke"] = (
    "Twenty-step smoke of the wide7h variant (wide7f + group decoder) warm "
    "started from v6-wide7f@4000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7h_smoke"] = Options(
    **{**_SEMANTIC_V6_FULL_CE2_WIDE7H_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7f_train_4000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7h_smoke_20",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7h_smoke_20",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide7h_train"] = (
    "Wide7h: wide7f (topk=3 + usage entropy + CE, L1-rendered probs) with "
    "the InstOk3D-style scene-adaptive group decoder (group queries + "
    "cross-attention). Retrying the decoder now that gradients are sharp; "
    "goal: scene-adaptive groups tighten mask boundaries and recover "
    "instance pixels from void. Warm starts from v6-wide7f@4000 (group "
    "head reinitialized, 500-step lambda warmup) and runs 4000 steps."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7h_train"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE7H_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7f_train_4000/model.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7h_train_4000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7h_train_4000",
)

_SEMANTIC_V6_FULL_CE2_WIDE7I_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_WIDE7F_COMMON,
    # Instance-contrastive loss on the rendered token features: pulls
    # same-instance pixels together and pushes different instances apart, so
    # the shared token features (and thus the group head input) become
    # instance-discriminative in a single feed-forward pass -- the direct
    # fix for the gap between feed-forward grouping (~0.19 AP50) and
    # per-scene TTT grouping (~0.39+).
    "lambda_instance_contrastive": 0.1,
}

config_doc["semantic_v6_open_vocab_full_ce2_wide7i_smoke"] = (
    "Twenty-step smoke of the wide7i variant (wide7f + instance-contrastive "
    "loss) warm started from v6-wide7f@4000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7i_smoke"] = Options(
    **{**_SEMANTIC_V6_FULL_CE2_WIDE7I_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7f_train_4000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7i_smoke_20",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7i_smoke_20",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide7i_train"] = (
    "Wide7i: wide7f (topk=3 + usage entropy + CE, L1-rendered probs) plus "
    "an instance-contrastive loss on the rendered token features, making "
    "the feed-forward grouping head's input instance-discriminative. Goal: "
    "raise feed-forward AP50 (0.19) toward the per-scene TTT level without "
    "any test-time optimization. Warm starts from v6-wide7f@4000 and runs "
    "4000 steps."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7i_train"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE7I_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7f_train_4000/model.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7i_train_4000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7i_train_4000",
)

_SEMANTIC_V6_4096_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_WIDE7F_COMMON,
    "num_gs_tokens": 4096,
}

config_doc["semantic_v6_4096_recon_warmup_smoke"] = (
    "Twenty-step smoke of the 4096-token reconstruction upgrade, warm "
    "started from v6-wide7f@4000 (gs_tokens tiled 1024 -> 4096)."
)
config_defaults["semantic_v6_4096_recon_warmup_smoke"] = Options(
    **{**_SEMANTIC_V6_4096_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7f_train_4000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_4096_recon_warmup_smoke_20",
    experiment_name="semantic_v6_4096_recon_warmup_smoke_20",
)

config_doc["semantic_v6_4096_recon_warmup"] = (
    "4096-token reconstruction upgrade: 4x the Gaussian tokens (65,536 -> "
    "262,144 Gaussians) to close the reconstruction gap to InstOk3D "
    "(PSNR ~19.9 -> target 22+). Warm starts from v6-wide7f@4000 with "
    "gs_tokens tiled 1024 -> 4096 (+noise), keeps the wide7f loss stack, "
    "and runs 8000 steps so the new tokens adapt and reconstruction "
    "recovers/improves before instance evaluation."
)
config_defaults["semantic_v6_4096_recon_warmup"] = Options(
    **_SEMANTIC_V6_4096_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7f_train_4000/model.safetensors"
    ),
    num_epochs=40,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_4096_recon_warmup_8000",
    experiment_name="semantic_v6_4096_recon_warmup_8000",
)

_SEMANTIC_V6_FULL_CE2_WIDE7J_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_WIDE7F_COMMON,
    # Cross-view instance contrastive loss: ScanNet instance ids are
    # consistent across the frames of a scene, so pixels of the same
    # instance are aggregated across all 7 target views into one prototype
    # (the wide7i version used per-view prototypes and did not help). This
    # directly teaches cross-view feature consistency so the feed-forward
    # group head can separate instances without test-time optimization.
    "lambda_instance_contrastive": 0.1,
}

config_doc["semantic_v6_open_vocab_full_ce2_wide7j_smoke"] = (
    "Twenty-step smoke of the wide7j variant (wide7f + cross-view "
    "instance contrastive) warm started from v6-wide7f@4000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7j_smoke"] = Options(
    **{**_SEMANTIC_V6_FULL_CE2_WIDE7J_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7f_train_4000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7j_smoke_20",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7j_smoke_20",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide7j_train"] = (
    "Wide7j: wide7f (topk=3 + usage entropy + CE, L1-rendered probs) plus "
    "the cross-view instance-contrastive loss (same-instance pixels "
    "aggregated across the 7 target views). Goal: teach cross-view "
    "consistent, instance-discriminative token features so feed-forward "
    "AP50 rises without test-time optimization. Warm starts from "
    "v6-wide7f@4000 and runs 4000 steps."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7j_train"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE7J_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7f_train_4000/model.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7j_train_4000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7j_train_4000",
)

_SEMANTIC_V6_FULL_CE2_WIDE7K_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_WIDE7F_COMMON,
    # Per-token 3D anchor-position conditioning for the group head: each
    # token's anchor is the mean of its 64 decoded Gaussian centers. This
    # gives the global head a stable spatial prior (InstOk3D-style anchors)
    # so instance grouping generalizes across scenes in a single forward
    # pass, instead of only working after per-scene adaptation (TTT).
    "instance_group_use_anchor_pos": True,
}

config_doc["semantic_v6_open_vocab_full_ce2_wide7k_smoke"] = (
    "Twenty-step smoke of the wide7k variant (wide7f + anchor-position "
    "conditioned group head) warm started from v6-wide7f@4000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7k_smoke"] = Options(
    **{**_SEMANTIC_V6_FULL_CE2_WIDE7K_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7f_train_4000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7k_smoke_20",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7k_smoke_20",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide7k_train"] = (
    "Wide7k: wide7f (topk=3 + usage entropy + CE, L1-rendered probs) with "
    "per-token 3D anchor-position conditioning for the group head. Goal: "
    "give the feed-forward grouping a stable spatial prior so AP50 rises "
    "without per-scene optimization. Warm starts from v6-wide7f@4000 "
    "(group head reinitialized) and runs 4000 steps."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7k_train"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE7K_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7f_train_4000/model.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7k_train_4000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7k_train_4000",
)

_SEMANTIC_V6_FULL_CE2_WIDE7L_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_WIDE7F_COMMON,
    # Combine the two structural levers that each worked partially:
    # wide7h's scene-adaptive group decoder (best AP50) and wide7k's
    # per-token anchor-position conditioning (best coverage/AP). The
    # decoder cross-attends to anchor-conditioned token features --
    # the closest we can get to InstOk3D's anchor + group-decoder design
    # without changing the backbone. Trained 8000 steps (the decoder only
    # ever got 4000 after reinit before).
    "instance_group_decoder": True,
    "instance_group_decoder_layers": 2,
    "instance_group_use_anchor_pos": True,
    "instance_group_lambda_warmup_steps": 500,
}

config_doc["semantic_v6_open_vocab_full_ce2_wide7l_smoke"] = (
    "Twenty-step smoke of the wide7l variant (anchor-position conditioned "
    "group decoder) warm started from v6-wide7f@4000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7l_smoke"] = Options(
    **{**_SEMANTIC_V6_FULL_CE2_WIDE7L_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7f_train_4000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7l_smoke_20",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7l_smoke_20",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide7l_train"] = (
    "Wide7l: wide7f plus the scene-adaptive group decoder AND per-token "
    "anchor-position conditioning, trained for 8000 steps. This is the "
    "closest architecture to InstOk3D's anchor + group-decoder design "
    "within the TokenGS backbone. Goal: break the feed-forward AP50 "
    "plateau (~0.19) toward beating GG/IGGT (0.27+). Warm starts from "
    "v6-wide7f@4000 (group head reinitialized, 500-step lambda warmup)."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7l_train"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE7L_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7f_train_4000/model.safetensors"
    ),
    num_epochs=40,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7l_train_8000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7l_train_8000",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide7l_train_12000"] = (
    "Continuation of wide7l (anchor + decoder) from step 8000 to 12000. "
    "The AP50 curve was still climbing at 8000 (0.192 -> 0.194 -> 0.205), "
    "so keep training to see how far the feed-forward grouping improves "
    "before adding the cross-view contrastive loss."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7l_train_12000"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE7L_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/model.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7l_train_12000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7l_train_12000",
)

_SEMANTIC_V6_PG_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_WIDE7L_COMMON,
    # Per-Gaussian instance grouping: each of the 64 Gaussians decoded by a
    # token carries its own low-dim instance feature (InstanceSplat-level
    # granularity) while the InstOk3D-style group decoder still forms
    # scene-adaptive group queries. This lets a token straddling an object
    # boundary split its Gaussians across instances.
    "instance_group_per_gaussian": True,
    "instance_group_feature_dim": 16,
}

config_doc["semantic_v6_open_vocab_pg_smoke"] = (
    "Twenty-step smoke of the per-Gaussian instance group head (wide7l "
    "recipe + per-Gaussian features), warm started from wide7l@8000."
)
config_defaults["semantic_v6_open_vocab_pg_smoke"] = Options(
    **{**_SEMANTIC_V6_PG_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_pg_smoke_20",
    experiment_name="semantic_v6_open_vocab_pg_smoke_20",
)

config_doc["semantic_v6_open_vocab_pg_train"] = (
    "Per-Gaussian instance grouping trained with the wide7l recipe (8+7 "
    "views, unfrozen geometry at LR 3e-5 with RGB loss) for 8000 steps. "
    "Warm starts the backbone from wide7l@8000; the new per-Gaussian head "
    "is reinitialized (architecture changed)."
)
config_defaults["semantic_v6_open_vocab_pg_train"] = Options(
    **_SEMANTIC_V6_PG_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=40,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_pg_train_8000",
    experiment_name="semantic_v6_open_vocab_pg_train_8000",
)

_SEMANTIC_V6_PGR_COMMON = {
    **_SEMANTIC_V6_PG_COMMON,
    # Per-Gaussian *residual* head: warm-starts the InstOk3D-style token
    # decoder from wide7l@8000 (exact same architecture at init) and adds a
    # small per-Gaussian offset (residual_scale) so individual Gaussians can
    # split across groups only near object boundaries. The standalone
    # per-Gaussian head (pg_train) fragmented every instance into ~1.4x
    # masks and dropped AP50 0.205 -> 0.152; the residual keeps the base
    # assignment intact unless the GT loss rewards a split.
    "instance_group_residual_head": True,
    "instance_group_residual_scale": 0.3,
}

config_doc["semantic_v6_open_vocab_pgr_smoke"] = (
    "Twenty-step smoke of the per-Gaussian residual head warm started from "
    "wide7l@8000 (base decoder weights loaded, residual fresh)."
)
config_defaults["semantic_v6_open_vocab_pgr_smoke"] = Options(
    **{**_SEMANTIC_V6_PGR_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_pgr_smoke_20",
    experiment_name="semantic_v6_open_vocab_pgr_smoke_20",
)

config_doc["semantic_v6_open_vocab_pgr_train"] = (
    "Per-Gaussian residual grouping trained with the wide7l recipe (8+7 "
    "views, unfrozen geometry at LR 3e-5 with RGB loss) for 8000 steps. "
    "The token-level base decoder is warm-started from wide7l@8000 so the "
    "model starts at AP50 0.205; the per-Gaussian residual can only split "
    "Gaussians near boundaries (residual_scale 0.3)."
)
config_defaults["semantic_v6_open_vocab_pgr_train"] = Options(
    **_SEMANTIC_V6_PGR_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=40,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_pgr_train_8000",
    experiment_name="semantic_v6_open_vocab_pgr_train_8000",
)

_SEMANTIC_V6_PGR2_COMMON = {
    **_SEMANTIC_V6_PGR_COMMON,
    # pgr (residual head) recovered AP25 (0.460 > 0.446) and cut the
    # over-segmentation from 1.39x to 1.19x, but AP50 (0.172) still trails
    # the wide7l base (0.205): the residual splits each instance into
    # partial fragments that pass AP25 yet fail AP50. Two loss-level levers
    # target the remaining fragments directly: topk=1 keeps one group per
    # GT instance (removing the 3-way split pressure of match_topk=3) and
    # the higher void weight keeps instance pixels off the void channel.
    "instance_group_match_topk": 1,
    "lambda_instance_group_void": 0.3,
}

config_doc["semantic_v6_open_vocab_pgr2_smoke"] = (
    "Twenty-step smoke of the pgr2 recipe (residual head + topk=1 + void "
    "0.3) warm started from wide7l@8000."
)
config_defaults["semantic_v6_open_vocab_pgr2_smoke"] = Options(
    **{**_SEMANTIC_V6_PGR2_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_pgr2_smoke_20",
    experiment_name="semantic_v6_open_vocab_pgr2_smoke_20",
)

config_doc["semantic_v6_open_vocab_pgr2_train"] = (
    "Per-Gaussian residual grouping (pgr) with topk=1 and void weight 0.3: "
    "one group per GT instance removes the split pressure that still "
    "fragments instances at AP50, while the higher void weight keeps "
    "instance pixels on the matched group. Warm starts from wide7l@8000 "
    "and runs 8000 steps."
)
config_defaults["semantic_v6_open_vocab_pgr2_train"] = Options(
    **_SEMANTIC_V6_PGR2_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=40,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_pgr2_train_8000",
    experiment_name="semantic_v6_open_vocab_pgr2_train_8000",
)

_SEMANTIC_V6_PGR3D_COMMON = {
    **_SEMANTIC_V6_PGR2_COMMON,
    # InstOk3D-style 3D anchor-level supervision on top of the residual
    # head: project every Gaussian into all 15 labeled views, majority-vote
    # a per-Gaussian instance id, Hungarian-match groups to instances in 3D
    # and supervise the per-Gaussian assignment directly (CE + BCE + Dice +
    # void + unmatched). The feed-forward head plateaus at ~0.20 AP50 across
    # every 2D-rendered loss variant while per-scene TTT reaches 0.72 --
    # the 3D loss teaches cross-view-consistent grouping directly instead
    # of view-dependent rendered pixels, which is what InstOk3D does.
    "lambda_instance_group_3d": 1.0,
    "lambda_instance_group_3d_ce": 1.0,
    "instance_group_3d_min_gs": 16,
    "instance_group_3d_match_topk": 1,
}

config_doc["semantic_v6_open_vocab_pgr3d_smoke"] = (
    "Twenty-step smoke of the residual head + 3D anchor-level instance loss "
    "warm started from wide7l@8000."
)
config_defaults["semantic_v6_open_vocab_pgr3d_smoke"] = Options(
    **{**_SEMANTIC_V6_PGR3D_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_pgr3d_smoke_20",
    experiment_name="semantic_v6_open_vocab_pgr3d_smoke_20",
)

config_doc["semantic_v6_open_vocab_pgr3d_train"] = (
    "Residual per-Gaussian grouping (pgr2: topk=1, void 0.3) plus the 3D "
    "anchor-level instance loss (lambda 1.0): every Gaussian gets a "
    "cross-view majority-vote instance target and the assignment is "
    "supervised directly in 3D. Warm starts from wide7l@8000 and runs "
    "8000 steps."
)
config_defaults["semantic_v6_open_vocab_pgr3d_train"] = Options(
    **_SEMANTIC_V6_PGR3D_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=40,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_pgr3d_train_8000",
    experiment_name="semantic_v6_open_vocab_pgr3d_train_8000",
)

_SEMANTIC_V6_PGR3DF_COMMON = {
    **_SEMANTIC_V6_PGR3D_COMMON,
    # Freeze the TokenGS backbone: (1) protects reconstruction -- every
    # unfrozen head run (pgr/pgr2/pgr3d) drifted geometry and lost ~0.45
    # PSNR because the instance losses fine-tune the shared encoder-decoder;
    # (2) gives the group head a stable input distribution instead of a
    # moving target, which is exactly the setting where per-scene TTT proved
    # the token features contain the instance information (AP50 0.72). The
    # head + adapters still train (3D + 2D rendered losses).
    "prompt_unfreeze_tokengs": False,
}

config_doc["semantic_v6_open_vocab_pgr3df_smoke"] = (
    "Twenty-step smoke of the frozen-backbone pgr3d recipe (3D + 2D "
    "instance losses, residual head, backbone frozen)."
)
config_defaults["semantic_v6_open_vocab_pgr3df_smoke"] = Options(
    **{**_SEMANTIC_V6_PGR3DF_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_pgr3df_smoke_20",
    experiment_name="semantic_v6_open_vocab_pgr3df_smoke_20",
)

config_doc["semantic_v6_open_vocab_pgr3df_train"] = (
    "Frozen-backbone per-Gaussian residual grouping with the 3D "
    "anchor-level loss: same recipe as pgr3d (topk=1, void 0.3, 3D "
    "supervision, residual head warm-started from wide7l@8000) but the "
    "TokenGS encoder-decoder is frozen so PSNR stays at the wide7l level "
    "and the head trains on stable features. Runs 8000 steps."
)
config_defaults["semantic_v6_open_vocab_pgr3df_train"] = Options(
    **_SEMANTIC_V6_PGR3DF_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=40,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_pgr3df_train_8000",
    experiment_name="semantic_v6_open_vocab_pgr3df_train_8000",
)

_SEMANTIC_V6_PGR4_COMMON = {
    **_SEMANTIC_V6_PGR3DF_COMMON,
    # Scene-adaptive group count + InstOk3D-style position-aware group
    # refinement on top of the frozen-backbone pgr3df recipe:
    #  * adaptive count: a count head predicts the per-scene number of
    #    instances; training supervises only the top-G most-used groups and
    #    eval prunes inactive groups (fixed 128 groups vs 20~134 GT
    #    instances is the biggest remaining mismatch -- the 40-scene eval
    #    shows the worst scenes are exactly the 90+ instance ones).
    #  * pos attn: one extra cross-attention layer whose keys are the
    #    per-token anchor 3D positions (InstOk3D anchors carry 3D
    #    positions); zero-init gate keeps the warm-start exact.
    "instance_group_adaptive_count": True,
    "instance_group_count_head": True,
    "instance_group_count_hidden": 128,
    "lambda_instance_group_count": 0.1,
    "instance_group_pos_attn_layers": 1,
    "instance_group_pos_attn_scale": 1.0,
}

config_doc["semantic_v6_open_vocab_pgr4_smoke"] = (
    "Twenty-step smoke of pgr4: frozen-backbone residual head warm started "
    "from pgr3df@8000, plus scene-adaptive group count (count head + "
    "top-G supervision + eval pruning) and position-aware group refinement "
    "cross-attention over anchor 3D positions."
)
config_defaults["semantic_v6_open_vocab_pgr4_smoke"] = Options(
    **{**_SEMANTIC_V6_PGR4_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_pgr4_smoke_20",
    experiment_name="semantic_v6_open_vocab_pgr4_smoke_20",
)

config_doc["semantic_v6_open_vocab_pgr4_train"] = (
    "Frozen-backbone pgr4 recipe for 8000 steps: scene-adaptive group "
    "count + position-aware refinement, warm started from pgr3df@8000 "
    "(best frozen head, AP50 0.218) with the wide7l@8000 backbone frozen."
)
config_defaults["semantic_v6_open_vocab_pgr4_train"] = Options(
    **_SEMANTIC_V6_PGR4_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=40,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_pgr4_train_8000",
    experiment_name="semantic_v6_open_vocab_pgr4_train_8000",
)

_SEMANTIC_V6_PGR5_COMMON = {
    **_SEMANTIC_V6_PGR3DF_COMMON,
    # pgr5 = pgr4 minus the two things that failed, keeping only the
    # position-aware group refinement with a learnable form:
    #  * adaptive top-G masking and the count head are OFF (pgr4 showed the
    #    hard top-G supervision starves small-instance groups and drops
    #    AP50 0.218 -> 0.205; per-view GT is only ~9 instances so the count
    #    budget was never the bottleneck).
    #  * the pos-attn refinement now uses zero-init output projections
    #    instead of a zero-init gate (pgr4's gate stayed ~0.0004 after 8000
    #    steps, i.e. the layer never engaged). Identity at init -> exact
    #    warm-start from pgr3df@8000, learnable from step one.
    "instance_group_adaptive_count": False,
    "instance_group_count_head": False,
    "instance_group_pos_attn_layers": 1,
    "instance_group_pos_attn_scale": 1.0,
}

config_doc["semantic_v6_open_vocab_pgr5_smoke"] = (
    "Twenty-step smoke of pgr5: frozen-backbone residual head warm started "
    "from pgr3df@8000 with the fixed position-aware refinement (zero-init "
    "output projections) and no adaptive count."
)
config_defaults["semantic_v6_open_vocab_pgr5_smoke"] = Options(
    **{**_SEMANTIC_V6_PGR5_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_pgr5_smoke_20",
    experiment_name="semantic_v6_open_vocab_pgr5_smoke_20",
)

config_doc["semantic_v6_open_vocab_pgr5_train"] = (
    "Frozen-backbone pgr5 recipe for 12000 steps: position-aware group "
    "refinement only (fixed zero-init-output form), warm started from "
    "pgr3df@8000 with the wide7l@8000 backbone frozen. Same duration as "
    "the pgr3df continuation (A) so the two lines are directly comparable."
)
config_defaults["semantic_v6_open_vocab_pgr5_train"] = Options(
    **_SEMANTIC_V6_PGR5_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=60,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_pgr5_train_12000",
    experiment_name="semantic_v6_open_vocab_pgr5_train_12000",
)

config_doc["semantic_v6_open_vocab_pgr6_smoke"] = (
    "Twenty-step smoke of pgr6: stack the position-aware refinement on top "
    "of the best base head (A = pgr3df continued to 12000 steps, AP50 "
    "0.231). The pos-attn layers are fresh zero-init-output so the run "
    "starts exactly at A@12000."
)
config_defaults["semantic_v6_open_vocab_pgr6_smoke"] = Options(
    **{**_SEMANTIC_V6_PGR5_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_pgr6_smoke_20",
    experiment_name="semantic_v6_open_vocab_pgr6_smoke_20",
)

config_doc["semantic_v6_open_vocab_pgr6_train"] = (
    "Combined run: position-aware refinement (B, pgr5) on top of the best "
    "base head (A@12000, AP50 0.231). Frozen backbone, warm start from "
    "A@12000 with fresh zero-init-output pos-attn layers; 8000 steps with "
    "early-stop evals. B overfits after ~5000 steps so the combined run is "
    "evaluated at 2000/4000/6000/8000."
)
config_defaults["semantic_v6_open_vocab_pgr6_train"] = Options(
    **_SEMANTIC_V6_PGR5_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=40,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_pgr6_train_8000",
    experiment_name="semantic_v6_open_vocab_pgr6_train_8000",
)

_SEMANTIC_V6_DENSE_COMMON = {
    **_SEMANTIC_V6_PGR3DF_COMMON,
    # pgr3df2 settings (best head, AP50 0.231): frozen backbone, per-GS
    # residual head, 128 groups, 3D + 2D instance losses.
    "instance_group_num_groups": 128,
    "instance_group_lambda_warmup_steps": 500,
    "instance_group_area_alpha": 0.5,
    "instance_group_match_area_norm": True,
    "instance_group_min_instance_pixels": 32,
    "instance_group_usage_entropy": 0.05,
    "instance_group_match_topk": 1,
    "lambda_instance_group_void": 0.3,
    "instance_group_pos_attn_layers": 0,
    # Dense image-evidence decoder: per-Gaussian residual features now come
    # from the frozen encoder's per-patch features projected onto the
    # Gaussian centers (multi-scale, 4x4 resolution), plus a patch-level
    # InfoNCE auxiliary loss so the dense decoder learns boundaries directly
    # from 2D instance supervision.
    "instance_group_dense_decoder": True,
    "instance_group_dense_feature_dim": 16,
    "instance_group_dense_upsample": 2,
    "instance_group_dense_multiscale": True,
    "instance_group_dense_scale": 0.3,
    "lambda_instance_dense_aux": 0.1,
}

config_doc["semantic_v6_open_vocab_dense_smoke"] = (
    "Twenty-step smoke of the dense image-evidence instance decoder: "
    "per-Gaussian residual head warm-started from pgr3df2@12000 (AP50 "
    "0.231) plus a fresh dense residual from frozen encoder patch features "
    "projected onto Gaussian centers, and a patch-level InfoNCE aux loss."
)
config_defaults["semantic_v6_open_vocab_dense_smoke"] = Options(
    **{**_SEMANTIC_V6_DENSE_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_dense_smoke_20",
    experiment_name="semantic_v6_open_vocab_dense_smoke_20",
)

config_doc["semantic_v6_open_vocab_dense_train"] = (
    "Dense image-evidence instance decoder trained for 12000 steps from "
    "pgr3df2@12000 (AP50 0.231): frozen backbone, warm-started token-level "
    "base + per-GS residual + new dense pixel-evidence residual, patch-level "
    "InfoNCE aux loss (lambda 0.1). The dense path is small-initialized so "
    "the run starts exactly at the best checkpoint."
)
config_defaults["semantic_v6_open_vocab_dense_train"] = Options(
    **_SEMANTIC_V6_DENSE_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=60,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_dense_train_12000",
    experiment_name="semantic_v6_open_vocab_dense_train_12000",
)

_SEMANTIC_V6_DENSE_GATE_COMMON = {
    **_SEMANTIC_V6_DENSE_COMMON,
    # Per-Gaussian gate on the dense residual: gate starts at ~0 so the run
    # begins exactly at pgr3df2@12000 (AP50 0.231) and the dense evidence can
    # only displace assignments where the learned gate opens. The patch-level
    # InfoNCE aux loss trains dense_net meanwhile, so the gate grows on top
    # of learned features instead of washing out the base (the failure mode
    # of the ungated dense run: AP50 0.231 -> 0.205).
    "instance_group_dense_gate": True,
}

config_doc["semantic_v6_open_vocab_dense_gate_smoke"] = (
    "Twenty-step smoke of the gated dense image-evidence instance decoder: "
    "same warm start as dense (pgr3df2@12000) but the dense residual is "
    "multiplied by a per-Gaussian gate initialized at ~0."
)
config_defaults["semantic_v6_open_vocab_dense_gate_smoke"] = Options(
    **{**_SEMANTIC_V6_DENSE_GATE_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_dense_gate_smoke_20",
    experiment_name="semantic_v6_open_vocab_dense_gate_smoke_20",
)

config_doc["semantic_v6_open_vocab_dense_gate_train"] = (
    "Gated dense image-evidence instance decoder trained for 6000 steps "
    "from pgr3df2@12000. Short run: the recent head variants peak at "
    "2000~6000 steps (dense val peak at 2400, pgr5 at 5000, pgr6 at 4000) "
    "and overfit after; checkpoints are saved every 200 steps so the peak "
    "can be picked up from LSM evals at 2000/4000/6000."
)
config_defaults["semantic_v6_open_vocab_dense_gate_train"] = Options(
    **_SEMANTIC_V6_DENSE_GATE_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_dense_gate_train_6000",
    experiment_name="semantic_v6_open_vocab_dense_gate_train_6000",
)

_SEMANTIC_V6_JOINT_BD_COMMON = {
    **_SEMANTIC_V6_DENSE_COMMON,
    # Joint geometry + instance training from the best frozen head
    # (pgr3df2@12000, AP50 0.231):
    #  * unfreeze ONLY the decoder stack (decoder_blocks + activation_head +
    #    gs_tokens) at LR 3e-5; the encoder stays frozen so cross-view
    #    features are stable (the previous unfreeze-all runs drifted PSNR
    #    because the instance losses corrupted the shared encoder).
    #  * boundary-aware RGB loss: extra reconstruction weight on GT instance
    #    boundary pixels so geometry is pushed to sharpen exactly where
    #    instances meet (InstanceSplat-style instance-to-geometry coupling).
    # Dense decoder is OFF so the run isolates the joint-training lever.
    "prompt_unfreeze_tokengs": True,
    "prompt_unfreeze_tokengs_lr": 3e-5,
    "prompt_unfreeze_tokengs_mode": "decoder",
    "lambda_boundary_rgb": 1.0,
    "boundary_rgb_dilate": 1,
    "boundary_rgb_include_background": True,
    "instance_group_dense_decoder": False,
}

config_doc["semantic_v6_open_vocab_joint_bd_smoke"] = (
    "Twenty-step smoke of joint geometry+instance training: decoder-only "
    "unfreeze at LR 3e-5 plus boundary-aware RGB loss, warm started from "
    "pgr3df2@12000."
)
config_defaults["semantic_v6_open_vocab_joint_bd_smoke"] = Options(
    **{**_SEMANTIC_V6_JOINT_BD_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_joint_bd_smoke_20",
    experiment_name="semantic_v6_open_vocab_joint_bd_smoke_20",
)

config_doc["semantic_v6_open_vocab_joint_bd_train"] = (
    "Joint geometry+instance training for 6000 steps from pgr3df2@12000: "
    "decoder-only unfreeze at LR 3e-5 (encoder frozen), boundary-aware RGB "
    "loss (lambda 1.0, dilated boundary band), all pgr3df2 instance losses "
    "kept. Short run because recent head variants peak at 2000~6000 steps; "
    "checkpoints saved every 200 steps, LSM evals at 2000/4000/6000."
)
config_defaults["semantic_v6_open_vocab_joint_bd_train"] = Options(
    **_SEMANTIC_V6_JOINT_BD_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_joint_bd_train_6000",
    experiment_name="semantic_v6_open_vocab_joint_bd_train_6000",
)

_SEMANTIC_V6_PGR3DF2_JOINT_RGB_COMMON = {
    **_SEMANTIC_V6_PGR3DF_COMMON,
    # Joint geometry + instance training from the best frozen head
    # (pgr3df2@12000, AP50 0.232): unfreeze ONLY the decoder stack
    # (decoder_blocks + activation_head + gs_tokens) so the instance losses
    # can actually shape the token/GS features instead of training a head on
    # fixed features. The RGB reconstruction loss (lambda_rgb=200, full MSE)
    # anchors the geometry, following the gc4-validated safe combination
    # (decoder-only + LR 1e-5 + strong RGB). The failed joint_bd run used a
    # weaker boundary-only RGB band (lambda 1.0) at LR 3e-5 and crashed PSNR
    # to 15.5; this run deliberately keeps full-image RGB and a 3x lower
    # geometry LR.
    "prompt_unfreeze_tokengs": True,
    "prompt_unfreeze_tokengs_lr": 1e-5,
    "prompt_unfreeze_tokengs_mode": "decoder",
    "lambda_rgb": 200.0,
    "lambda_boundary_rgb": 0.0,
}

config_doc["semantic_v6_open_vocab_pgr3df2_joint_rgb_smoke"] = (
    "Two-step smoke of decoder-unfrozen joint instance training from "
    "pgr3df2@12000 with the full-image RGB anchor."
)
config_defaults["semantic_v6_open_vocab_pgr3df2_joint_rgb_smoke"] = Options(
    **{**_SEMANTIC_V6_PGR3DF2_JOINT_RGB_COMMON, "max_eval_iters": 1},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_pgr3df2_joint_rgb_smoke_2",
    experiment_name="semantic_v6_open_vocab_pgr3df2_joint_rgb_smoke_2",
)

config_doc["semantic_v6_open_vocab_pgr3df2_joint_rgb_train"] = (
    "Joint geometry+instance training for 6000 steps from pgr3df2@12000: "
    "decoder-only unfreeze at LR 1e-5 (encoder frozen), full-image RGB MSE "
    "anchor at lambda 200, all pgr3df2 instance losses kept. This lets the "
    "instance objective backprop into the decoder so token/GS features are "
    "shaped by grouping, not only reconstruction. Checkpoints every 200 "
    "steps; LSM evals at 1000/2000/3000/4000/5000/6000."
)
config_defaults["semantic_v6_open_vocab_pgr3df2_joint_rgb_train"] = Options(
    **_SEMANTIC_V6_PGR3DF2_JOINT_RGB_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_pgr3df2_joint_rgb_train_6000",
    experiment_name="semantic_v6_open_vocab_pgr3df2_joint_rgb_train_6000",
)

_SEMANTIC_V6_GROUP_TOKEN_INSTOK3D_COMMON = {
    **_SEMANTIC_V6_PGR3DF_COMMON,
    # Clean InstOk3D-style Group Token baseline. Only the explicit group
    # tokens are tested: InstanceGroupDecoder has L learnable group tokens
    # that cross-attend to the GS-token features, producing per-token
    # assignment logits over L+1 (L groups + a zero void channel, softmax).
    # Every Gaussian inherits its token's assignment and is rendered into L
    # soft instance masks by the existing render_feature_channels path.
    # No per-Gaussian / residual / conditioned / dense / count heads.
    "instance_group_conditioned_gaussians": False,
    "instance_group_per_gaussian": False,
    "instance_group_residual_head": False,
    "instance_group_dense_decoder": False,
    "instance_group_count_head": False,
    "instance_group_adaptive_count": False,
    "instance_group_decoder": True,
    "instance_group_num_groups": 100,  # L = 100 (InstOk3D-style budget)
    "instance_group_decoder_layers": 2,
    "instance_group_use_anchor_pos": False,
    "instance_group_pos_attn_layers": 0,
    # InstOk3D protocol: one per-scene Hungarian matching shared across all
    # 15 views, matching cost = BCE + Dice; context and target views both
    # supervised. Pure 2D objective, no extra losses.
    "instance_group_scene_level_matching": True,
    "instance_group_supervise_input_views": True,
    "instance_group_lambda_warmup_steps": 0,
    "instance_group_match_topk": 1,
    "instance_group_match_area_norm": False,
    "instance_group_area_alpha": 0.0,
    "lambda_instance_group_dice": 1.0,
    "lambda_instance_group_mask": 1.0,  # lambda_bce
    "lambda_instance_group_void": 0.1,
    "lambda_instance_group_unmatched": 0.1,
    "lambda_instance_group_ce": 0.0,
    "instance_group_usage_entropy": 0.0,
    "lambda_instance_group_3d": 0.0,
    "lambda_instance_group_3d_ce": 0.0,
    "lambda_instance_contrastive": 0.0,
    # Pure instance objective: all semantic / prompt / teacher losses off.
    "lambda_ce_cosine": 0.0,
    "lambda_feat": 0.0,
    "prompt_lambda_bce": 0.0,
    "prompt_lambda_dice": 0.0,
    # Reconstruction is structurally protected: the backbone is frozen and
    # this head only emits assignment logits (it cannot alter Gaussian
    # geometry/appearance), so segmentation training cannot degrade RGB.
    # lambda_rgb stays 0 to avoid a constant loss term.
    "lambda_rgb": 0.0,
    "lambda_boundary_rgb": 0.0,
    "prompt_unfreeze_tokengs": False,
}

config_doc["semantic_v6_group_token_instok3d_smoke"] = (
    "Minimal smoke of the clean InstOk3D-style Group Token baseline: "
    "InstanceGroupDecoder (L=100) on the frozen wide7l backbone, per-scene "
    "Hungarian + BCE/Dice, pure instance objective. Verifies forward, "
    "shapes, loss and backward of the group-token head."
)
config_defaults["semantic_v6_group_token_instok3d_smoke"] = Options(
    **{**_SEMANTIC_V6_GROUP_TOKEN_INSTOK3D_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_group_token_instok3d_smoke_2",
    experiment_name="semantic_v6_group_token_instok3d_smoke_2",
)

config_doc["semantic_v6_group_token_instok3d_train"] = (
    "Clean InstOk3D-style Group Token baseline, 6000 steps on the frozen "
    "wide7l backbone (same data as pgr3df2 for a fair comparison). "
    "Explicit group tokens (L=100) cross-attend to GS-token features, "
    "per-token assignment logits over L+1 with a zero void channel, "
    "Gaussians inherit the assignment and render L soft instance masks. "
    "One per-scene Hungarian matching over all 15 views, BCE+Dice matching "
    "cost and loss. Pure instance objective (semantic losses off); "
    "reconstruction is structurally protected (frozen backbone, head does "
    "not modify Gaussians). Checkpoints every 200 steps; LSM evals at "
    "1000/2000/3000/4000/5000/6000."
)
config_defaults["semantic_v6_group_token_instok3d_train"] = Options(
    **_SEMANTIC_V6_GROUP_TOKEN_INSTOK3D_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_group_token_instok3d_train_6000",
    experiment_name="semantic_v6_group_token_instok3d_train_6000",
)

_SEMANTIC_V6_GROUP_TOKEN_PGR_COMMON = {
    **_SEMANTIC_V6_GROUP_TOKEN_INSTOK3D_COMMON,
    # Group tokens (global instance proposal, L=100) + per-Gaussian
    # refinement: the GroupTokenPerGaussianHead keeps the token-level
    # assignment as a prior and adds a zero-initialized per-GS residual
    # (local evidence: token-prior logits + relative position + Gaussian
    # parameters). Warm-started from the token-level baseline
    # (group_token_instok3d@5000, AP50 0.219) so the run starts exactly at
    # that checkpoint and only the refinement module learns at first.
    # v2: the per-GS residual is zero-centered across groups and bounded to
    # [0.01, 0.3] (init 0.05) so it corrects boundaries locally instead of
    # re-ranking every Gaussian (the v1 unbounded residual fragmented the
    # rendered masks and exploded predicted instances ~8x).
    "instance_group_per_gaussian": True,
    "instance_group_group_token_refine": True,
    "instance_group_residual_head": False,
    "instance_group_dense_decoder": False,
    "instance_group_group_token_refine_scale_init": 0.05,
    "instance_group_group_token_refine_scale_max": 0.3,
}

config_doc["semantic_v6_group_token_pgr_smoke"] = (
    "Minimal smoke of group tokens + per-Gaussian refinement: warm-started "
    "from group_token_instok3d@5000 with a fresh zero-init refinement "
    "module; verifies forward, shapes, loss, backward and the exact "
    "warm-start of the token prior."
)
config_defaults["semantic_v6_group_token_pgr_smoke"] = Options(
    **{**_SEMANTIC_V6_GROUP_TOKEN_PGR_COMMON, "max_eval_iters": 1},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_group_token_instok3d_train_6000/"
        "checkpoints/model_step_005000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_group_token_pgr_v2_smoke",
    experiment_name="semantic_v6_group_token_pgr_v2_smoke",
)

config_doc["semantic_v6_group_token_pgr_train"] = (
    "Group tokens + per-Gaussian refinement, 6000 steps, warm-started from "
    "group_token_instok3d@5000 (AP50 0.219). The token-level proposal is "
    "kept as a prior; a zero-init per-GS residual (local position + "
    "Gaussian parameters + token prior) sharpens instance boundaries. Same "
    "frozen wide7l backbone, data and per-scene Hungarian + BCE/Dice "
    "objective as the group-token baseline. v2 keeps the per-GS residual "
    "small (zero-centered, scale 0.05 init / 0.3 max) so the token prior "
    "stays dominant and only boundaries are sharpened. Checkpoints every "
    "200 steps; LSM evals at 1000/2000/3000/4000/5000/6000."
)
config_defaults["semantic_v6_group_token_pgr_train"] = Options(
    **_SEMANTIC_V6_GROUP_TOKEN_PGR_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_group_token_instok3d_train_6000/"
        "checkpoints/model_step_005000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_group_token_pgr_v2_train_6000",
    experiment_name="semantic_v6_group_token_pgr_v2_train_6000",
)

_SEMANTIC_V6_GROUP_TOKEN_JOINT_COMMON = {
    **_SEMANTIC_V6_GROUP_TOKEN_INSTOK3D_COMMON,
    # InstOk3D-style joint training from the start. The frozen encoder plays
    # the role of the frozen VGGT feature extractor; the decoder (which
    # produces the GS-token hidden states and the Gaussian geometry) and the
    # 100 group tokens are trained together on ScanNet with reconstruction
    # (lambda_rgb MSE) and instance grouping (per-scene Hungarian BCE+Dice)
    # from step 0. lambda_seg ramps in linearly over
    # instance_group_lambda_warmup_steps so early training is
    # reconstruction-dominated and the pretrained geometry is not destroyed
    # (the earlier decoder-unfreeze runs had full-strength instance loss
    # from the start and collapsed PSNR to ~15.4).
    "prompt_unfreeze_tokengs": True,
    "prompt_unfreeze_tokengs_lr": 1e-5,
    "prompt_unfreeze_tokengs_mode": "decoder",
    "lambda_rgb": 200.0,
    "lambda_boundary_rgb": 0.0,
    "instance_group_lambda_warmup_steps": 2000,
}

config_doc["semantic_v6_group_token_joint_smoke"] = (
    "Minimal smoke of InstOk3D-style joint training: frozen encoder, "
    "decoder + 100 group tokens trained from scratch-ish on ScanNet with "
    "reconstruction and instance losses, lambda_seg linear warm-up over "
    "2000 steps. Verifies RGB reconstruction, grouping, Hungarian, "
    "gradients into the decoder, and that warm-up keeps PSNR stable."
)
config_defaults["semantic_v6_group_token_joint_smoke"] = Options(
    **{**_SEMANTIC_V6_GROUP_TOKEN_JOINT_COMMON, "max_eval_iters": 1},
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_group_token_joint_smoke_2",
    experiment_name="semantic_v6_group_token_joint_smoke_2",
)

config_doc["semantic_v6_group_token_joint_train"] = (
    "InstOk3D-style joint training, 6000 steps on ScanNet "
    "(scannet_prompt_small, same split as the frozen-backbone baselines). "
    "Frozen encoder (VGGT analogue); decoder + 100 group tokens optimized "
    "jointly from step 0 with reconstruction (lambda_rgb=200 MSE) and "
    "per-scene Hungarian BCE+Dice, lambda_seg warm-up over 2000 steps. "
    "Head is fully random-init (no warm start). This tests whether training "
    "the reconstruction and instance structure together from the start "
    "beats the frozen-backbone ceiling (0.21-0.23) instead of adding "
    "supervision only after reconstruction pretraining. Checkpoints every "
    "200 steps; LSM evals at 1000/2000/3000/4000/5000/6000."
)
config_defaults["semantic_v6_group_token_joint_train"] = Options(
    **_SEMANTIC_V6_GROUP_TOKEN_JOINT_COMMON,
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_group_token_joint_train_6000",
    experiment_name="semantic_v6_group_token_joint_train_6000",
)

_SEMANTIC_V6_GROUP_TOKEN_SCRATCH_COMMON = {
    **_SEMANTIC_V6_GROUP_TOKEN_INSTOK3D_COMMON,
    # Pure from-scratch joint training on ScanNet: no pretrained TokenGS
    # checkpoint is loaded (random encoder/decoder/GS-token init), and
    # reconstruction + instance grouping are optimized together from step 0
    # with lambda_seg warm-up. This is the cleanest test of whether the
    # "reconstruction-pretrain-then-instance" paradigm is the bottleneck:
    # the model never sees a reconstruction-only phase.
    "prompt_tokengs_checkpoint": "",
    "prompt_unfreeze_tokengs": True,
    "prompt_unfreeze_tokengs_lr": 1e-4,  # from scratch: geometry = main LR
    "prompt_unfreeze_tokengs_mode": "all",
    "lambda_rgb": 200.0,
    "lambda_boundary_rgb": 0.0,
    "instance_group_lambda_warmup_steps": 2000,
    "lr": 1e-4,
    "lr_scheduler": "onecycle",
    "pct_start_steps": 2000,
}

config_doc["semantic_v6_group_token_scratch_smoke"] = (
    "Minimal smoke of from-scratch joint training: no pretrained TokenGS "
    "checkpoint, random init, reconstruction + instance from step 0 with "
    "lambda_seg warm-up. Verifies forward (no NaN), shapes, loss and "
    "backward through the whole model."
)
config_defaults["semantic_v6_group_token_scratch_smoke"] = Options(
    **{**_SEMANTIC_V6_GROUP_TOKEN_SCRATCH_COMMON, "max_eval_iters": 1},
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_group_token_scratch_smoke_2",
    experiment_name="semantic_v6_group_token_scratch_smoke_2",
)

config_doc["semantic_v6_group_token_scratch_train"] = (
    "From-scratch joint training of TokenGS + 100 Group Tokens on ScanNet "
    "(scannet_prompt_small): no pretrained reconstruction checkpoint, "
    "random encoder/decoder/GS tokens, reconstruction (lambda_rgb=200 MSE) "
    "and per-scene Hungarian BCE+Dice optimized together from step 0 with "
    "lambda_seg warm-up over 2000 steps. Onecycle LR 1e-4, 12000 steps. "
    "This directly tests whether the reconstruction-pretrain-then-instance "
    "paradigm (rather than the head) is the bottleneck vs the frozen "
    "0.21-0.23 ceiling. Checkpoints every 200 steps; LSM evals at "
    "2000/4000/6000/8000/10000/12000."
)
config_defaults["semantic_v6_group_token_scratch_train"] = Options(
    **_SEMANTIC_V6_GROUP_TOKEN_SCRATCH_COMMON,
    num_epochs=60,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_group_token_scratch_train_12000",
    experiment_name="semantic_v6_group_token_scratch_train_12000",
)

_SEMANTIC_V6_GROUP_TOKEN_WIDE7L_JOINT_COMMON = {
    **_SEMANTIC_V6_GROUP_TOKEN_INSTOK3D_COMMON,
    # The missing 2x2 cell: instance-shaped initialization (wide7l@8000
    # backbone, produced by ~36K steps of unfrozen semantic+instance+RGB
    # training) + continued joint training. The old instance head is NOT
    # loaded (backbone_resume only loads encoder/decoder/GS-token weights),
    # so the 100 group tokens start fully random. The encoder stays frozen;
    # the decoder uses a small LR; lambda_seg ramps in over 2000 steps and
    # the strong RGB anchor (200) protects reconstruction. This tests
    # whether continuing to shape already-instance-friendly features beats
    # the frozen-head ceiling (0.219/0.232).
    "prompt_unfreeze_tokengs": True,
    "prompt_unfreeze_tokengs_lr": 1e-5,
    "prompt_unfreeze_tokengs_mode": "decoder",
    "lambda_rgb": 200.0,
    "lambda_boundary_rgb": 0.0,
    "instance_group_lambda_warmup_steps": 2000,
}

config_doc["semantic_v6_group_token_wide7l_joint_smoke"] = (
    "Minimal smoke of wide7l-backbone + fresh 100 Group Tokens + "
    "decoder-only joint training: verifies the backbone loads from "
    "wide7l@8000 (head stays fresh/random), the decoder is trainable at a "
    "small LR, reconstruction stays at the wide7l level, and warm-up "
    "scales the instance loss."
)
config_defaults["semantic_v6_group_token_wide7l_joint_smoke"] = Options(
    **{**_SEMANTIC_V6_GROUP_TOKEN_WIDE7L_JOINT_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_group_token_wide7l_joint_smoke_2",
    experiment_name="semantic_v6_group_token_wide7l_joint_smoke_2",
)

config_doc["semantic_v6_group_token_wide7l_joint_train"] = (
    "Wide7l-backbone + fresh 100 Group Tokens + joint training, 6000 steps "
    "on ScanNet (scannet_prompt_small). Backbone = wide7l@8000 (instance-"
    "shaped, loaded via backbone_resume with the old head discarded), "
    "encoder frozen, decoder trainable at LR 1e-5, reconstruction "
    "(lambda_rgb=200 MSE) + per-scene Hungarian BCE+Dice with lambda_seg "
    "warm-up over 2000 steps. Tests whether continued joint training on "
    "already-instance-friendly features exceeds the frozen-head ceiling "
    "(group_token 0.219 / pgr3df2 0.232). Checkpoints every 200 steps; LSM "
    "evals at 1000/2000/3000/4000/5000/6000."
)
config_defaults["semantic_v6_group_token_wide7l_joint_train"] = Options(
    **_SEMANTIC_V6_GROUP_TOKEN_WIDE7L_JOINT_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_group_token_wide7l_joint_train_6000",
    experiment_name="semantic_v6_group_token_wide7l_joint_train_6000",
)

_SEMANTIC_V6_GROUP_TOKEN_WIDE7L_SEM_COMMON = {
    **_SEMANTIC_V6_GROUP_TOKEN_WIDE7L_JOINT_COMMON,
    # Semantic-stabilizer test: identical to wide7l_joint (instance-shaped
    # backbone, fresh 100 Group Tokens, decoder-only unfreeze at LR 1e-5,
    # lambda_seg warm-up, strong RGB anchor) plus the wide7-chain semantic
    # losses (cosine CE, CLIP feature distillation, prompt BCE+Dice). The
    # wide7 chain is the only unfrozen formulation that historically kept
    # PSNR (~19.8), and it always trained with these semantic losses on;
    # every instance-only unfreeze run collapsed PSNR to ~15.5. The semantic
    # adapters are warm-started from wide7l@8000 so the losses are
    # meaningful from step 0.
    "lambda_ce_cosine": 1.0,
    "lambda_feat": 1.0,
    "prompt_lambda_bce": 1.0,
    "prompt_lambda_dice": 1.0,
    "prompt_semantic_adapter_resume": (
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
}

config_doc["semantic_v6_group_token_wide7l_sem_smoke"] = (
    "Minimal smoke of the semantic-stabilizer test: wide7l_joint plus the "
    "wide7 semantic losses (warm-started adapters). Verifies the semantic "
    "losses are active, the adapters load, and the first steps are stable."
)
config_defaults["semantic_v6_group_token_wide7l_sem_smoke"] = Options(
    **{**_SEMANTIC_V6_GROUP_TOKEN_WIDE7L_SEM_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_group_token_wide7l_sem_smoke_2",
    experiment_name="semantic_v6_group_token_wide7l_sem_smoke_2",
)

config_doc["semantic_v6_group_token_wide7l_sem_train"] = (
    "Semantic-stabilizer test, 6000 steps: wide7l_joint plus the wide7 "
    "semantic losses (cosine CE 1.0, CLIP feat 1.0, prompt BCE+Dice 1.0) "
    "with adapters warm-started from wide7l@8000. If PSNR stays near the "
    "wide7l level (~19-20) instead of collapsing to ~15.5, semantic "
    "supervision is what makes unfrozen training stable, giving a viable "
    "joint formulation to continue from. Checkpoints every 200 steps; LSM "
    "evals at 1000/2000/3000/4000/5000/6000."
)
config_defaults["semantic_v6_group_token_wide7l_sem_train"] = Options(
    **_SEMANTIC_V6_GROUP_TOKEN_WIDE7L_SEM_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_group_token_wide7l_sem_train_6000",
    experiment_name="semantic_v6_group_token_wide7l_sem_train_6000",
)

_SEMANTIC_V6_PGR3DF2_BIGDATA_COMMON = {
    **_SEMANTIC_V6_PGR3DF_COMMON,
    # Data expansion on the best recipe (pgr3df2, AP50 0.232): same per-GS
    # residual head on the frozen wide7l backbone and the same loss stack,
    # but the training manifest is expanded from 5680 windows to ~32000
    # (--train-samples-per-class 4000 over 1425 scenes, same-scene
    # on-demand stride 25). The head warm-starts from pgr3df2@12000.
    "dataset_kwargs": {
        "small_manifest_path": (
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "scannet_prompt_full_wide_8x7_exp4k.json"
        ),
        "wide_target_subsample": 0,
    },
}

config_doc["semantic_v6_pgr3df2_bigdata_smoke"] = (
    "Two-step smoke of the expanded-data pgr3df2 run: verifies the new "
    "manifest loads and the warm-started head trains normally."
)
config_defaults["semantic_v6_pgr3df2_bigdata_smoke"] = Options(
    **{**_SEMANTIC_V6_PGR3DF2_BIGDATA_COMMON, "max_eval_iters": 1},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_pgr3df2_bigdata_smoke_2",
    experiment_name="semantic_v6_pgr3df2_bigdata_smoke_2",
)

config_doc["semantic_v6_pgr3df2_bigdata_train"] = (
    "Data expansion on the best recipe (pgr3df2, AP50 0.232): expanded "
    "manifest (~32000 windows vs 5680), frozen wide7l backbone, per-GS "
    "residual head warm-started from pgr3df2@12000, same loss stack. "
    "12000 steps with checkpoints every 200; LSM evals at "
    "2000/4000/6000/8000/10000/12000."
)
config_defaults["semantic_v6_pgr3df2_bigdata_train"] = Options(
    **_SEMANTIC_V6_PGR3DF2_BIGDATA_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=60,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_pgr3df2_bigdata_train_12000",
    experiment_name="semantic_v6_pgr3df2_bigdata_train_12000",
)

_SEMANTIC_V6_PGR3DF2_ANCHOR3D_COMMON = {
    **_SEMANTIC_V6_PGR3DF_COMMON,
    # Scene-adaptive 3D GS-token initialization (Experiment 3): the frozen
    # decoder, head, losses, backbone, data and evaluation are untouched;
    # only the GS token queries fed into the decoder change. A frozen probe
    # pass decodes the default tokens to get per-token 3D anchor positions
    # (mean of the token's Gaussian centers); the queries are then the probe
    # token features plus a fixed, scale-controlled Fourier position
    # embedding of the normalized anchors. With the default
    # anchor_3d_pass_decoder=False the geometry stays the frozen baseline
    # exactly (PSNR identical) and the instance head reads the
    # position-augmented token features; the pass_decoder=True variant feeds
    # the anchored queries through the frozen decoder (requires a small
    # pos_scale, e.g. 0.003, to keep geometry stable). No learned parameters,
    # no gradients through the init path -- initialization only.
    "gs_token_init": "anchor_3d",
    "anchor_3d_query_source": "global",
    "anchor_3d_pos_scale": 0.1,
    "anchor_3d_pos_freqs": 4,
    "anchor_3d_pass_decoder": False,
}

config_doc["semantic_v6_pgr3df2_anchor3d_smoke"] = (
    "Smoke of Experiment 3 (scene-adaptive 3D GS-token initialization): "
    "pgr3df2 recipe (frozen wide7l backbone, per-GS residual head, 2D+3D "
    "losses, 5680-window manifest) with gs_token_init=anchor_3d. Verifies "
    "the probe/init path, shapes, loss, backward and PSNR sanity."
)
config_defaults["semantic_v6_pgr3df2_anchor3d_smoke"] = Options(
    **{**_SEMANTIC_V6_PGR3DF2_ANCHOR3D_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_pgr3df2_anchor3d_smoke",
    experiment_name="semantic_v6_pgr3df2_anchor3d_smoke",
)

config_doc["semantic_v6_pgr3df2_anchor3d_train"] = (
    "Experiment 3 (12000 steps): the exact pgr3df2 recipe (frozen wide7l "
    "backbone, PerGaussianResidualHead, 128 groups, 2D BCE/Dice/CE + 3D "
    "anchor-level loss + count loss, 5680-window manifest) with the ONLY "
    "change being gs_token_init=anchor_3d: per-scene 3D-anchored GS token "
    "queries instead of the global learnable tokens. Head starts fresh "
    "(resume=null) so the comparison to a fresh learnable-init control is "
    "causal. LSM evals at 1000/2000/.../12000."
)
config_defaults["semantic_v6_pgr3df2_anchor3d_train"] = Options(
    **_SEMANTIC_V6_PGR3DF2_ANCHOR3D_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=60,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_pgr3df2_anchor3d_train_12000",
    experiment_name="semantic_v6_pgr3df2_anchor3d_train_12000",
)

config_doc["semantic_v6_pgr3df2_fresh_train"] = (
    "Control for Experiment 3 (12000 steps): identical to the anchor3d run "
    "except gs_token_init stays 'learnable' (the standard global GS token "
    "queries). Also starts fresh (resume=null). Together with the anchor3d "
    "run this isolates the effect of 3D-anchored token initialization on "
    "the same head/loss/backbone/data/eval."
)
config_defaults["semantic_v6_pgr3df2_fresh_train"] = Options(
    **_SEMANTIC_V6_PGR3DF_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=60,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_pgr3df2_fresh_train_12000",
    experiment_name="semantic_v6_pgr3df2_fresh_train_12000",
)

_SEMANTIC_V6_GROUPGEN_COMMON = {
    **_SEMANTIC_V6_PGR3DF_COMMON,
    # Experiment B: GroupToken participates in Gaussian *generation*, not
    # just mask assignment. Group tokens cross-attend to the 3D-anchored
    # token hidden, produce the slot-competition assignment, and then (a)
    # rewrite the token hidden before the Gaussian activation head
    # (residual_scale=0.3) and (b) emit explicit bounded per-anchor
    # geometry/opacity deltas (geometry_scale=0.05, opacity_scale=0.05,
    # blend=0.5 so the conditioned Gaussians are not diluted to 10%). The
    # decoder is unfrozen at the wide7l-stable LR with strong RGB anchoring,
    # and the 3D anchor-level loss is ON (the never-combined recipe: group-
    # conditioned generation + 3D supervision + decoder unfreeze).
    "instance_group_conditioned_gaussians": True,
    "instance_group_condition_generator": True,
    "instance_group_condition_residual_scale": 0.3,
    "instance_group_condition_gaussian_blend": 0.5,
    "instance_group_condition_geometry_scale": 0.05,
    "instance_group_condition_per_gaussian_opacity_scale": 0.05,
    "instance_group_num_groups": 128,
    "instance_group_scene_level_matching": False,
    "prompt_unfreeze_tokengs": True,
    "prompt_unfreeze_tokengs_lr": 1.0e-05,
    "prompt_unfreeze_tokengs_mode": "decoder",
    "lambda_rgb": 200.0,
    "lambda_instance_group_3d": 1.0,
    "lambda_instance_group_3d_ce": 1.0,
    "instance_group_lambda_warmup_steps": 500,
}

config_doc["semantic_v6_groupgen_smoke"] = (
    "Smoke of Experiment B (group-conditioned Gaussian generation): verifies "
    "the GroupToken path runs BEFORE the activation head, changes Gaussian "
    "parameters (xyz/opacity), gradients flow from the instance loss into the "
    "group tokens and the unfrozen decoder, and RGB/PSNR stay anchored."
)
config_defaults["semantic_v6_groupgen_smoke"] = Options(
    **{**_SEMANTIC_V6_GROUPGEN_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_groupgen_smoke",
    experiment_name="semantic_v6_groupgen_smoke",
)

config_doc["semantic_v6_groupgen_train"] = (
    "Experiment B (12000 steps): GroupToken-conditioned Gaussian generation "
    "on the wide7l@8000 backbone. Group tokens enter the decoder before the "
    "Gaussian activation head (token rewrite + explicit bounded xyz/opacity "
    "deltas), so instance identity changes Gaussian parameters. Decoder "
    "unfrozen (LR 1e-5), RGB lambda 200, instance BCE/Dice/CE + 3D anchor "
    "loss (lambda 1.0), warm-up 500 steps -- the never-combined recipe. "
    "Resume=null; head and group tokens start fresh. LSM evals at "
    "1000/2000/.../12000."
)
config_defaults["semantic_v6_groupgen_train"] = Options(
    **_SEMANTIC_V6_GROUPGEN_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=60,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_groupgen_train_12000",
    experiment_name="semantic_v6_groupgen_train_12000",
)

_SEMANTIC_V6_INSTANCE_BRANCH_COMMON = {
    **_SEMANTIC_V6_PGR3DF_COMMON,
    # Experiment C: fully independent instance-structured branch. The frozen
    # TokenGS RGB reconstruction (encoder + decoder + activation head) is
    # untouched; instance loss never flows back. The branch builds its own
    # scene-adaptive 3D anchors (frozen token hidden + normalized 3D anchor
    # positions from the frozen Gaussians), 100 learnable group tokens
    # (slot-competition assignment), an instance-aware latent, and its OWN
    # Gaussian decoder that generates the Gaussians used to render instance
    # masks. Supervision: existing Hungarian BCE/Dice + 3D anchor-level loss,
    # both on the branch's own Gaussians.
    "instance_branch_independent": True,
    "instance_branch_num_groups": 100,
    "instance_branch_anchor_dim": 256,
    "instance_branch_num_heads": 8,
    "instance_branch_num_layers": 2,
    "instance_branch_gaussians_per_anchor": 64,
    "instance_branch_pos_offset_scale": 0.2,
    "instance_branch_scale_delta_amp": 0.5,
    "instance_branch_opacity_delta_amp": 0.3,
    "instance_branch_rgb_delta_amp": 0.5,
    "instance_group_num_groups": 100,
    "instance_group_scene_level_matching": False,
    "lambda_instance_group_3d": 1.0,
    "lambda_instance_group_3d_ce": 1.0,
    "instance_group_lambda_warmup_steps": 500,
    "prompt_unfreeze_tokengs": False,
    "lambda_rgb": 200.0,
}

config_doc["semantic_v6_instance_branch_smoke"] = (
    "Smoke of Experiment C (independent instance branch): verifies the "
    "branch constructs, renders its own instance masks, the instance loss "
    "trains only branch parameters (backbone grads all zero), and the RGB "
    "reconstruction stays byte-identical (PSNR at the frozen 19.81 level)."
)
config_defaults["semantic_v6_instance_branch_smoke"] = Options(
    **{**_SEMANTIC_V6_INSTANCE_BRANCH_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_instance_branch_smoke",
    experiment_name="semantic_v6_instance_branch_smoke",
)

config_doc["semantic_v6_instance_branch_train"] = (
    "Experiment C v2 (12000 steps): frozen TokenGS RGB branch untouched; a new "
    "scene-adaptive anchor + 100 group tokens + independent Gaussian decoder "
    "(64 GS/anchor, warm-started at the frozen geometry) generates the "
    "instance-mask Gaussians AND a branch-only RGB reconstruction. "
    "Hungarian BCE/Dice + 3D anchor-level loss + lambda_rgb*MSE on the "
    "branch's own Gaussians; instance/RGB loss never backpropagates into the "
    "reconstruction branch. Head/branch starts fresh (resume=null); "
    "backbone = wide7l@8000. LSM evals at "
    "1000/2000/.../12000."
)
config_defaults["semantic_v6_instance_branch_train"] = Options(
    **_SEMANTIC_V6_INSTANCE_BRANCH_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=60,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_instance_branch_v2_train_12000",
    experiment_name="semantic_v6_instance_branch_v2_train_12000",
)

_SEMANTIC_V6_TOKEN_UNITS_COMMON = {
    **_SEMANTIC_V6_PGR3DF_COMMON,
    # Experiment D: Token -> local spatial units -> Group. Each 64-GS token
    # learns K=8 3D-local units (soft k-means over GS features + geometry);
    # 100 group tokens assign units; GS inherit their unit's group; masks
    # render through the ORIGINAL frozen geometry. Only unit formation +
    # group tokens + assignment train. Hungarian BCE/Dice + 3D loss.
    "instance_branch_token_units": True,
    "instance_branch_units_per_token": 8,
    "instance_branch_unit_feat_dim": 128,
    "instance_branch_unit_layers": 2,
    "instance_branch_num_groups": 100,
    "instance_branch_anchor_dim": 256,
    "instance_branch_num_heads": 8,
    "instance_branch_num_layers": 2,
    "instance_branch_unit_temp": 5.0,
    "instance_branch_unit_entropy": 0.05,
    "instance_branch_unit_compactness": 0.05,
    "lambda_instance_group_void": 0.1,
    "instance_group_num_groups": 100,
    "instance_group_scene_level_matching": False,
    "lambda_instance_group_3d": 1.0,
    "lambda_instance_group_3d_ce": 1.0,
    "instance_group_lambda_warmup_steps": 500,
    "prompt_unfreeze_tokengs": False,
    "lambda_rgb": 200.0,
}

config_doc["semantic_v6_token_units_smoke"] = (
    "Smoke of Experiment D (Token -> local spatial units -> Group): verifies "
    "the 8 units/token form dynamically (soft k-means over GS features + "
    "geometry), group tokens assign units, GS inherit unit groups, masks "
    "render through the frozen geometry, instance loss trains only the new "
    "parameters (backbone grads all zero), and the RGB reconstruction stays "
    "byte-identical."
)
config_defaults["semantic_v6_token_units_smoke"] = Options(
    **{**_SEMANTIC_V6_TOKEN_UNITS_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_token_units_smoke",
    experiment_name="semantic_v6_token_units_smoke",
)

config_doc["semantic_v6_token_units_train"] = (
    "Experiment D (12000 steps): frozen TokenGS geometry; each 64-GS token "
    "learns 8 dynamic 3D-local units; 100 group tokens assign units; GS "
    "inherit unit groups; masks rendered through the frozen geometry. "
    "Hungarian BCE/Dice + 3D anchor loss + unit-entropy regularizer. Only "
    "unit formation + group tokens + assignment train. Backbone = "
    "wide7l@8000, resume=null. LSM evals at 1000/2000/.../12000."
)
config_defaults["semantic_v6_token_units_train"] = Options(
    **_SEMANTIC_V6_TOKEN_UNITS_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=60,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_token_units_train_12000",
    experiment_name="semantic_v6_token_units_train_12000",
)

_SEMANTIC_V6_TOKEN_UNITS_REFINE_COMMON = {
    **_SEMANTIC_V6_TOKEN_UNITS_COMMON,
    # Direction 1: per-GS group-logit refinement on top of the unit proposal.
    # The unit->group prior is refined by a zero-init residual per Gaussian
    # (GS feature + position relative to its soft unit center), so the last
    # ~1.5 mixed instances per unit can be split at boundaries. No Gaussian
    # parameter changes; frozen geometry still renders the masks.
    "instance_branch_gs_refine": True,
    "instance_branch_gs_refine_scale": 0.1,
    "instance_branch_gs_refine_dim": 64,
}

config_doc["semantic_v6_token_units_refine_smoke"] = (
    "Smoke of Direction 1 (unit proposal + per-GS logit refinement): "
    "verifies the zero-init residual is identity at init, gradients flow "
    "only into the new branch, masks render through the frozen geometry, and "
    "RGB stays byte-identical."
)
config_defaults["semantic_v6_token_units_refine_smoke"] = Options(
    **{**_SEMANTIC_V6_TOKEN_UNITS_REFINE_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_token_units_refine_smoke",
    experiment_name="semantic_v6_token_units_refine_smoke",
)

config_doc["semantic_v6_token_units_refine_train"] = (
    "Direction 1 (4000 steps): token->8 units->100 groups + zero-init "
    "per-GS logit refinement. Shorter schedule with checkpoints every 200 "
    "so the curve can be read early (evals at 1000/2000/3000/4000). Frozen "
    "backbone wide7l@8000; only unit/group/refine params train."
)
config_defaults["semantic_v6_token_units_refine_train"] = Options(
    **_SEMANTIC_V6_TOKEN_UNITS_REFINE_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_token_units_refine_train_4000",
    experiment_name="semantic_v6_token_units_refine_train_4000",
)

_SEMANTIC_V6_TOKEN_UNITS_PURITY_COMMON = {
    **_SEMANTIC_V6_TOKEN_UNITS_COMMON,
    # Scheme A: Gaussian-level pseudo-GT -> Unit Purity Loss. Per-GS pseudo
    # instance (15-view majority vote, detached) aggregates per unit through
    # the soft assignment a; per-unit instance entropy is minimized,
    # foreground-weighted so background does not dominate. Permutation-
    # invariant. Everything else (units, GroupToken, Hungarian BCE/Dice, 3D
    # consistency, usage entropy, compactness) unchanged; no per-GS
    # refinement, no contrastive, no pseudo-unit matching.
    "instance_branch_unit_purity": True,
    "instance_branch_unit_purity_weight": 0.1,
}

config_doc["semantic_v6_token_units_purity_smoke"] = (
    "Smoke of Scheme A (unit purity loss): verifies the pseudo-GT majority "
    "vote, the per-unit instance distribution, the foreground-weighted "
    "entropy loss, gradients flow only into the new branch, and RGB stays "
    "byte-identical."
)
config_defaults["semantic_v6_token_units_purity_smoke"] = Options(
    **{**_SEMANTIC_V6_TOKEN_UNITS_PURITY_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_token_units_purity_smoke",
    experiment_name="semantic_v6_token_units_purity_smoke",
)

config_doc["semantic_v6_token_units_purity_train"] = (
    "Scheme A (4000 steps): Token->8 units->Group + unit purity loss "
    "(pseudo-GT instance entropy per unit, foreground-weighted). Short "
    "schedule with checkpoints every 200; LSM evals at "
    "1000/2000/3000/4000. Frozen wide7l@8000 backbone; only unit/group "
    "params train."
)
config_defaults["semantic_v6_token_units_purity_train"] = Options(
    **_SEMANTIC_V6_TOKEN_UNITS_PURITY_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_token_units_purity_train_4000",
    experiment_name="semantic_v6_token_units_purity_train_4000",
)

_SEMANTIC_V6_SCENE_PROTO_COMMON = {
    **_SEMANTIC_V6_TOKEN_UNITS_COMMON,
    # Scene-specific dynamic instance prototypes (replaces global Group
    # Tokens). Unit formation (gs_feature_mlp / unit_queries / unit_layers /
    # log_unit_temp) is LOADED from the trained token-units checkpoint and
    # FROZEN; only the prototype module (unit embedding MLP + void head +
    # temperatures) trains. Prototypes are formed per scene from the 8192
    # local units (feature + 3D center) by FPS init + iterative soft
    # k-means-style slot grouping (no global instance queries). Hungarian
    # BCE/Dice + 3D consistency supervision unchanged.
    "instance_branch_scene_prototypes": True,
    "instance_branch_num_slots": 100,
    "instance_branch_slot_dim": 128,
    "instance_branch_slot_iterations": 3,
    "instance_branch_slot_temp": 5.0,
    "instance_branch_unit_resume": (
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_token_units_train_12000/"
        "checkpoints/model_step_006000.safetensors"
    ),
}

config_doc["semantic_v6_scene_proto_smoke"] = (
    "Smoke of scene-specific dynamic prototypes: verifies unit formation "
    "loads frozen from the token-units checkpoint (no grads), the prototype "
    "module is the only trainable part, masks render through the frozen "
    "geometry, and RGB stays byte-identical."
)
config_defaults["semantic_v6_scene_proto_smoke"] = Options(
    **{**_SEMANTIC_V6_SCENE_PROTO_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_scene_proto_smoke",
    experiment_name="semantic_v6_scene_proto_smoke",
)

config_doc["semantic_v6_scene_proto_train"] = (
    "Scene-specific dynamic prototypes (6000 steps): frozen unit formation "
    "(loaded from token-units@6000) + per-scene FPS-init iterative slot "
    "grouping (100 slots) replacing global Group Tokens. Only the prototype "
    "module trains. Hungarian BCE/Dice + 3D consistency. Checkpoints every "
    "200; LSM evals at 1000/2000/.../6000."
)
config_defaults["semantic_v6_scene_proto_train"] = Options(
    **_SEMANTIC_V6_SCENE_PROTO_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_scene_proto_train_6000",
    experiment_name="semantic_v6_scene_proto_train_6000",
)

_SEMANTIC_V6_UNIT_EMBEDDING_COMMON = {
    **_SEMANTIC_V6_TOKEN_UNITS_COMMON,
    # Instance-aware Local Unit Embedding: freeze the token-units formation,
    # train only a 128-D IdentityEncoder with a distribution-level soft
    # InfoNCE (per-unit pseudo-instance distributions as soft positives).
    # Inference: deterministic agglomerative clustering on
    # [embedding, pos] with a fixed distance threshold; no global queries,
    # no iterative slots, no GroupToken/prototype/purity/refine.
    "instance_branch_unit_embedding": True,
    "instance_branch_embed_dim": 128,
    "instance_branch_embed_temp": 0.1,
    "instance_branch_embed_loss": 1.0,
    "instance_branch_cluster_pos_weight": 1.0,
    "instance_branch_cluster_eps": 1.0,
    "instance_branch_void_fg_share": 0.5,
    "instance_branch_unit_resume": (
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_token_units_train_12000/"
        "checkpoints/model_step_006000.safetensors"
    ),
}

config_doc["semantic_v6_unit_embedding_smoke"] = (
    "Smoke of instance-aware unit embedding: verifies unit formation loads "
    "frozen, IdentityEncoder is the only trainable part, the soft InfoNCE "
    "trains (embedding gradients, no collapse), and the eval path runs the "
    "deterministic agglomerative clustering + render through the frozen "
    "geometry."
)
config_defaults["semantic_v6_unit_embedding_smoke"] = Options(
    **{**_SEMANTIC_V6_UNIT_EMBEDDING_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_unit_embedding_smoke",
    experiment_name="semantic_v6_unit_embedding_smoke",
)

config_doc["semantic_v6_unit_embedding_train"] = (
    "Instance-aware Local Unit Embedding (6000 steps): frozen unit "
    "formation + trainable IdentityEncoder (soft InfoNCE over unit "
    "pseudo-instance distributions). Inference = agglomerative clustering on "
    "[embedding, pos] (fixed eps), render with frozen geometry. Checkpoints "
    "every 200; LSM evals at 1000/2000/.../6000."
)
config_defaults["semantic_v6_unit_embedding_train"] = Options(
    **_SEMANTIC_V6_UNIT_EMBEDDING_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_unit_embedding_train_6000",
    experiment_name="semantic_v6_unit_embedding_train_6000",
)

_SEMANTIC_V6_UNIT_SHAPING_COMMON = {
    **_SEMANTIC_V6_UNIT_EMBEDDING_COMMON,
    # Instance-aware Unit Feature Shaping: the unit formation feature
    # extractor itself (gs_feature_mlp / unit_queries / unit_layers /
    # log_unit_temp) is WARM-STARTED from token-units@6000 and TRAINED with
    # the distribution-level soft InfoNCE + unit usage entropy + 3D
    # compactness. No IdentityEncoder (the unit feature is the embedding);
    # inference = agglomerative clustering on [unit feature, pos] (eps 0.5).
    # The TokenGS reconstruction path stays frozen (no gradients, PSNR
    # unchanged).
    "instance_branch_unit_encoder": False,
    "instance_branch_cluster_eps": 0.5,
}

config_doc["semantic_v6_unit_shaping_smoke"] = (
    "Smoke of instance-aware unit feature shaping: verifies unit formation "
    "loads warm (trainable), gradients flow only into the extractor (backbone "
    "zero), soft InfoNCE + usage entropy + compactness train, and the eval "
    "clustering path renders through the frozen geometry."
)
config_defaults["semantic_v6_unit_shaping_smoke"] = Options(
    **{**_SEMANTIC_V6_UNIT_SHAPING_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_unit_shaping_smoke",
    experiment_name="semantic_v6_unit_shaping_smoke",
)

config_doc["semantic_v6_unit_shaping_train"] = (
    "Instance-aware Unit Feature Shaping (6000 steps): trainable unit "
    "formation extractor (warm-started from token-units@6000), soft InfoNCE "
    "+ usage entropy + 3D compactness; no IdentityEncoder. Inference: "
    "agglomerative clustering on [unit feature, pos] with eps=0.5, render "
    "with frozen geometry. Checkpoints every 200; LSM evals at "
    "1000/2000/.../6000."
)
config_defaults["semantic_v6_unit_shaping_train"] = Options(
    **_SEMANTIC_V6_UNIT_SHAPING_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_unit_shaping_train_6000",
    experiment_name="semantic_v6_unit_shaping_train_6000",
)

_SEMANTIC_V6_UNIT_SHAPING_IMG_COMMON = {
    **_SEMANTIC_V6_UNIT_SHAPING_COMMON,
    # Step 2: add frozen multi-view encoder patch features. The unit
    # embedding becomes [normalized shaped unit feature, normalized per-unit
    # image feature (projected GS patch features -> trainable MLP)].
    # Enables the frozen encoder dense-feature cache (last block only).
    "instance_branch_unit_image": True,
    "instance_branch_unit_image_dim": 64,
    "instance_group_dense_decoder": True,
    "instance_group_dense_multiscale": False,
}

config_doc["semantic_v6_unit_shaping_img_smoke"] = (
    "Smoke of unit shaping + frozen encoder patch features: verifies the "
    "dense cache populates, the per-GS patch projection + unit image branch "
    "run, unit formation + image branch train (backbone zero), and the eval "
    "clustering path renders."
)
config_defaults["semantic_v6_unit_shaping_img_smoke"] = Options(
    **{**_SEMANTIC_V6_UNIT_SHAPING_IMG_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_unit_shaping_img_smoke",
    experiment_name="semantic_v6_unit_shaping_img_smoke",
)

config_doc["semantic_v6_unit_shaping_img_train"] = (
    "Unit shaping + frozen encoder patch features (3000 steps, per the fast "
    "convergence finding): trainable unit formation extractor + unit image "
    "branch; soft InfoNCE + usage entropy + compactness. Inference: "
    "agglomerative clustering on [embedding, pos] eps=0.5. Checkpoints every "
    "200; LSM evals at 1000/1500/2000/2500/3000."
)
config_defaults["semantic_v6_unit_shaping_img_train"] = Options(
    **_SEMANTIC_V6_UNIT_SHAPING_IMG_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_unit_shaping_img_train_3000",
    experiment_name="semantic_v6_unit_shaping_img_train_3000",
)

config_doc["semantic_v6_unit_shaping_img_center_push_smoke"] = (
    "Smoke of InstanceSplat-inspired InfoNCE + instance-center-level push: "
    "keeps the unit-shaping-img pipeline and soft InfoNCE, adds a cosine "
    "hinge between soft instance centers (built from the detached pseudo "
    "instance distribution; margin=0.2, weight=0.1). No per-unit margin. "
    "Verifies loss, gradients (backbone zero) and new center monitors "
    "(center_cos_mean / center_margin) with intra compactness intact."
)
config_defaults["semantic_v6_unit_shaping_img_center_push_smoke"] = Options(
    **{
        **_SEMANTIC_V6_UNIT_SHAPING_IMG_COMMON,
        "instance_branch_embed_loss_mode": "info_nce_center_push",
        "instance_branch_embed_center_push_margin": 0.2,
        "instance_branch_embed_center_push_weight": 0.1,
        "max_eval_iters": 1,
    },
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_unit_shaping_img_center_push_smoke",
    experiment_name="semantic_v6_unit_shaping_img_center_push_smoke",
)

config_doc["semantic_v6_unit_shaping_img_center_push_train"] = (
    "InstanceSplat-inspired InfoNCE + center-level push (3000 steps, fixed "
    "margin=0.2, weight=0.1): identical to semantic_v6_unit_shaping_img_train "
    "except the embedding objective is info_nce_center_push. Checkpoints "
    "every 200; LSM evals at 1000/1500/2000/2500/3000."
)
config_defaults["semantic_v6_unit_shaping_img_center_push_train"] = Options(
    **{
        **_SEMANTIC_V6_UNIT_SHAPING_IMG_COMMON,
        "instance_branch_embed_loss_mode": "info_nce_center_push",
        "instance_branch_embed_center_push_margin": 0.2,
        "instance_branch_embed_center_push_weight": 0.1,
    },
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_unit_shaping_img_center_push_train_3000",
    experiment_name="semantic_v6_unit_shaping_img_center_push_train_3000",
)

config_doc["semantic_v6_unit_shaping_img_dino_smoke"] = (
    "Smoke of DINOv2-augmented unit representation: identical to "
    "semantic_v6_unit_shaping_img_train (unit shaping + patch features + "
    "soft InfoNCE + Agglomerative) plus frozen DINOv2 ViT-B/14 dense "
    "features projected onto GS centers and aggregated per unit, injected "
    "into the identity embedding through a small trainable projection. "
    "Verifies DINO extraction, embedding dims, gradients (backbone zero, "
    "dino_proj trainable) and the eval clustering path."
)
config_defaults["semantic_v6_unit_shaping_img_dino_smoke"] = Options(
    **{
        **_SEMANTIC_V6_UNIT_SHAPING_IMG_COMMON,
        "instance_branch_unit_dino": True,
        "instance_branch_unit_dino_dim": 64,
        "max_eval_iters": 1,
    },
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_unit_shaping_img_dino_smoke",
    experiment_name="semantic_v6_unit_shaping_img_dino_smoke",
)

config_doc["semantic_v6_unit_shaping_img_dino_train"] = (
    "DINOv2-augmented unit representation (3000 steps): unit shaping + "
    "patch features + frozen DINOv2 dense features in the unit identity "
    "embedding, soft InfoNCE + usage entropy + compactness. Inference: "
    "Agglomerative on [embedding, pos] eps=0.5. Checkpoints every 200; "
    "LSM evals at 1000/1500/2000/2500/3000."
)
config_defaults["semantic_v6_unit_shaping_img_dino_train"] = Options(
    **{
        **_SEMANTIC_V6_UNIT_SHAPING_IMG_COMMON,
        "instance_branch_unit_dino": True,
        "instance_branch_unit_dino_dim": 64,
    },
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_unit_shaping_img_dino_train_3000",
    experiment_name="semantic_v6_unit_shaping_img_dino_train_3000",
)

config_doc["semantic_v6_unit_shaping_img_dino_k64_train"] = (
    "Same 0.324 recipe (DINO + patch + InfoNCE, frozen backbone) but with "
    "64 3D-local units per 64-GS token instead of 8.  Identity losses "
    "sub-sample 8192 units/step (full UxU similarity is infeasible at "
    "65536 units).  Eval clustering auto-switches to a scalable DBSCAN "
    "path for the larger unit count (exact agglomerative is O(U^2) there)."
)
config_defaults["semantic_v6_unit_shaping_img_dino_k64_train"] = Options(
    **{
        **_SEMANTIC_V6_UNIT_SHAPING_IMG_COMMON,
        "instance_branch_unit_dino": True,
        "instance_branch_unit_dino_dim": 64,
        "instance_branch_units_per_token": 64,
        "instance_branch_embed_sample": 8192,
    },
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_unit_shaping_img_dino_k64_train_3000",
    experiment_name="semantic_v6_unit_shaping_img_dino_k64_train_3000",
)

_SEMANTIC_V6_UNIT_SHAPING_IMG_DINO_SIC = {
    **_SEMANTIC_V6_UNIT_SHAPING_IMG_COMMON,
    # Direction B0: identical 0.324 recipe (DINO + patch + 8 local units +
    # soft InfoNCE + Agglomerative eps=0.5, frozen backbone) except the
    # per-token unit queries are initialized with a zero-gated readout of
    # scene-conditioned instance queries (SIC).  SIC queries are produced by
    # deterministic 3D-FPS anchoring over token centers + cross-attention
    # over token descriptors; FPS is spatial coverage only (NOT instance
    # prototypes).  gate=0 makes step 0 exactly equal to the 0.324
    # checkpoint.
    "instance_branch_unit_dino": True,
    "instance_branch_unit_dino_dim": 64,
    "instance_branch_sic_units": True,
    "instance_branch_sic_queries": 128,
    "instance_branch_sic_dim": 256,
    "instance_branch_sic_heads": 4,
    "instance_branch_sic_layers": 2,
    "instance_branch_sic_usage": 0.02,
    # Warm start the whole existing branch from the best 0.324 checkpoint
    # (unit extractor / image branch / dino_proj).  The new SIC params are
    # absent from the checkpoint and stay at their random/zero init.
    "instance_branch_unit_resume": (
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_unit_shaping_img_dino_train_3000/"
        "checkpoints/model_step_003000.safetensors"
    ),
}

config_doc["semantic_v6_unit_shaping_img_dino_sic_smoke"] = (
    "Smoke of Direction B0 (SIC-conditioned unit formation): verifies "
    "gate=0 gives byte-level baseline parity (SIC on/off identical), "
    "reconstruction stays frozen, only the new SIC conditioning branch "
    "adds gradients, the gate learns away from 0, and the cross-token "
    "binding diagnostics (same-instance same-token / cross-token / "
    "different-instance similarity) run in both train and eval."
)
config_defaults["semantic_v6_unit_shaping_img_dino_sic_smoke"] = Options(
    **{**_SEMANTIC_V6_UNIT_SHAPING_IMG_DINO_SIC, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_unit_shaping_img_dino_sic_smoke",
    experiment_name="semantic_v6_unit_shaping_img_dino_sic_smoke",
)

config_doc["semantic_v6_unit_shaping_img_dino_sic_train"] = (
    "Direction B0 (3000 steps): 0.324 recipe + SIC-conditioned unit "
    "formation.  Warm-started from the 0.324 checkpoint (existing params) "
    "with a zero gate for the new scene-conditioning readout; same soft "
    "InfoNCE + usage entropy + compactness (+ small SIC usage-entropy "
    "anti-collapse term).  Inference stays Agglomerative eps=0.5 so this is "
    "a pure structural ablation.  Checkpoints every 500; LSM evals at "
    "500/1000/.../3000."
)
config_defaults["semantic_v6_unit_shaping_img_dino_sic_train"] = Options(
    **_SEMANTIC_V6_UNIT_SHAPING_IMG_DINO_SIC,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_unit_shaping_img_dino_sic_train_3000",
    experiment_name="semantic_v6_unit_shaping_img_dino_sic_train_3000",
)

config_doc["semantic_v6_unit_shaping_img_dino_base8k_train"] = (
    "Decisive step-2 experiment: rerun the exact 0.324 recipe "
    "(unit shaping + patch + DINO + soft InfoNCE + Agglomerative eps=0.5, "
    "3000 steps, scannet_prompt_full_wide_8x7 data, warm start from "
    "token-units@6000) with the ONLY change being the frozen backbone: "
    "tokengs_re10k -> ScanNet pure-reconstruction fine-tune "
    "(scannet_recon_finetune_base_8k@8000, PSNR 20.12 on LSM-40).  Both "
    "prompt_tokengs_checkpoint and backbone_resume point to the new base "
    "so training features and LSM eval features are consistent (the "
    "historical 0.324 run had train on re10k / eval on wide7l).  Compare "
    "AP50 against lsm_dino_003000 (0.324)."
)
config_defaults["semantic_v6_unit_shaping_img_dino_base8k_train"] = Options(
    **{
        **_SEMANTIC_V6_UNIT_SHAPING_IMG_COMMON,
        "instance_branch_unit_dino": True,
        "instance_branch_unit_dino_dim": 64,
        "prompt_tokengs_checkpoint": (
            "/space0/mawb/tokengs/workspace/"
            "scannet_recon_finetune_base_8k/"
            "tokengs_backbone_step_008000.safetensors"
        ),
        "instance_branch_unit_resume": (
            "/space0/mawb/tokengs/workspace/"
            "semantic_v6_token_units_train_12000/"
            "checkpoints/model_step_006000.safetensors"
        ),
    },
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "scannet_recon_finetune_base_8k/checkpoints/"
        "model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_unit_shaping_img_dino_base8k_train_3000",
    experiment_name="semantic_v6_unit_shaping_img_dino_base8k_train_3000",
)

_SEMANTIC_V6_UNIT_SHAPING_IMG_CONF_COMMON = {
    **_SEMANTIC_V6_UNIT_SHAPING_IMG_COMMON,
    # Pseudo-label denoising: only GS whose 15-view majority vote is
    # confident (vote fraction >= 0.6 across >= 3 valid views) contribute to
    # the unit soft-instance distribution p_u used by the InfoNCE. Units
    # whose high-confidence GS mass fraction < 0.25 are excluded from the
    # supervision. Loss structure, clustering and rendering unchanged.
    "instance_branch_pseudo_conf": 0.6,
    "instance_branch_pseudo_min_views": 3,
    "instance_branch_pseudo_unit_min_mass": 0.25,
}

config_doc["semantic_v6_unit_shaping_img_conf_smoke"] = (
    "Smoke of pseudo-label denoising: verifies the confidence filtering "
    "keeps only high-conf GS in p_u, records kept GS/unit ratios, and the "
    "InfoNCE + clustering still run."
)
config_defaults["semantic_v6_unit_shaping_img_conf_smoke"] = Options(
    **{**_SEMANTIC_V6_UNIT_SHAPING_IMG_CONF_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_unit_shaping_img_conf_smoke",
    experiment_name="semantic_v6_unit_shaping_img_conf_smoke",
)

config_doc["semantic_v6_unit_shaping_img_conf_train"] = (
    "Pseudo-label denoising on top of unit-shaping+patch (3000 steps): "
    "conf>=0.6 & >=3 views for GS, unit high-conf mass >=0.25; everything "
    "else identical to the 0.279 run. Checkpoints every 200; LSM evals at "
    "1000/1500/2000/2500/3000 with eps sweep."
)
config_defaults["semantic_v6_unit_shaping_img_conf_train"] = Options(
    **_SEMANTIC_V6_UNIT_SHAPING_IMG_CONF_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_unit_shaping_img_conf_train_3000",
    experiment_name="semantic_v6_unit_shaping_img_conf_train_3000",
)

_SEMANTIC_V6_SCENE_ASSIGNMENT_COMMON = {
    **_SEMANTIC_V6_UNIT_SHAPING_IMG_COMMON,
    # Directly learn unit -> scene-specific instance assignment, replacing
    # both global GroupTokens and the 'embedding -> clustering' path.  Unit
    # formation (gs_feature_mlp / unit_queries / unit_layers / log_unit_temp
    # / unit_image_net) is LOADED from the best unit-shaping+patch checkpoint
    # and FROZEN.  The only trainable module is the scene assignment head:
    # FPS-init scene slots -> iterative refinement -> unit->slot(+void)
    # assignment.  Supervision acts directly on the assignment:
    # unit-level Hungarian BCE/Dice on soft pseudo-label distributions,
    # void BCE, slot-usage entropy, plus the existing rendered-mask
    # Hungarian BCE/Dice and 3D consistency.  No clustering at inference.
    "instance_branch_unit_embedding": False,
    "instance_branch_scene_assignment": True,
    "instance_branch_scene_slots": 100,
    "instance_branch_scene_slot_iters": 3,
    "instance_branch_scene_slot_temp": 5.0,
    "instance_branch_scene_slot_pos_weight": 1.0,
    "instance_branch_scene_slot_entropy": 0.05,
    "instance_branch_scene_slot_void": 0.1,
    "lambda_scene_assignment_unit": 1.0,
    "instance_branch_pseudo_conf": 0.6,
    "instance_branch_pseudo_min_views": 3,
    "instance_branch_pseudo_unit_min_mass": 0.25,
    "instance_branch_unit_resume": (
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_unit_shaping_img_train_3000/"
        "checkpoints/model_step_002000.safetensors"
    ),
}

config_doc["semantic_v6_scene_assignment_smoke"] = (
    "Smoke of direct unit->scene instance assignment: verifies unit "
    "formation + unit image branch load frozen from the best unit-shaping "
    "checkpoint, the scene assignment head is the only trainable part, the "
    "unit-level Hungarian BCE/Dice + void BCE + usage entropy train, the "
    "rendered-mask Hungarian and 3D loss still run, and the eval path "
    "renders M+1 channels (no clustering). RGB stays byte-identical."
)
config_defaults["semantic_v6_scene_assignment_smoke"] = Options(
    **{**_SEMANTIC_V6_SCENE_ASSIGNMENT_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_scene_assignment_smoke",
    experiment_name="semantic_v6_scene_assignment_smoke",
)

config_doc["semantic_v6_scene_assignment_train"] = (
    "Direct unit->scene instance assignment (3000 steps): frozen unit "
    "formation from unit-shaping+patch@2000 + trainable scene assignment "
    "head (FPS-init slots, iterative refinement, unit->slot+void). "
    "Supervision: unit-level Hungarian BCE/Dice on soft pseudo distributions "
    "+ void BCE + usage entropy + rendered-mask Hungarian BCE/Dice + 3D "
    "consistency. No clustering at inference. Checkpoints every 500; LSM "
    "evals at 1000/2000/3000 (or 500/1000/1500/2000/2500/3000)."
)
config_defaults["semantic_v6_scene_assignment_train"] = Options(
    **_SEMANTIC_V6_SCENE_ASSIGNMENT_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=6,
    max_iters_per_epoch=500,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_scene_assignment_train_3000",
    experiment_name="semantic_v6_scene_assignment_train_3000",
)

_SEMANTIC_V6_SCENE_ASSIGNMENT_DINO_COMMON = {
    **_SEMANTIC_V6_SCENE_ASSIGNMENT_COMMON,
    # DINOv2-augmented scene assignment: the unit features entering the
    # assignment head are normalize(concat(unit_feat, unit_img,
    # dino_proj(dino_unit))).  Unit formation, unit image branch and the
    # DINO projection are loaded from the DINO-trained checkpoint
    # (unit-shaping+patch+DINO @3000) and frozen; only the scene assignment
    # head trains.
    "instance_branch_unit_dino": True,
    "instance_branch_unit_dino_dim": 64,
    "instance_branch_unit_resume": (
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_unit_shaping_img_dino_train_3000/"
        "model.safetensors"
    ),
}

config_doc["semantic_v6_scene_assignment_dino_smoke"] = (
    "Smoke of DINO-augmented direct unit->scene assignment: verifies the "
    "DINO projection loads frozen from the DINO@3000 checkpoint, the scene "
    "assignment head (feat_dim includes DINO) trains, unit-level Hungarian "
    "+ void BCE + usage entropy run, rendered-mask Hungarian + 3D loss run, "
    "and eval renders M+1 channels without clustering."
)
config_defaults["semantic_v6_scene_assignment_dino_smoke"] = Options(
    **{**_SEMANTIC_V6_SCENE_ASSIGNMENT_DINO_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_scene_assignment_dino_smoke",
    experiment_name="semantic_v6_scene_assignment_dino_smoke",
)

config_doc["semantic_v6_scene_assignment_dino_train"] = (
    "DINO-augmented direct unit->scene assignment (3000 steps): frozen "
    "DINO-enriched unit features from unit-shaping+patch+DINO@3000 + "
    "trainable scene assignment head (FPS-init slots, iterative refinement, "
    "unit->slot+void).  Supervision: unit-level Hungarian BCE/Dice on soft "
    "pseudo distributions + void BCE + usage entropy + rendered-mask "
    "Hungarian BCE/Dice + 3D consistency.  No clustering at inference. "
    "Checkpoints every 200; LSM evals at 500/1000/1500/2000/2500/3000."
)
config_defaults["semantic_v6_scene_assignment_dino_train"] = Options(
    **_SEMANTIC_V6_SCENE_ASSIGNMENT_DINO_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=6,
    max_iters_per_epoch=500,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_scene_assignment_dino_train_3000",
    experiment_name="semantic_v6_scene_assignment_dino_train_3000",
)

_SEMANTIC_V6_GT_ASSIGNMENT_COMMON = {
    **_SEMANTIC_V6_SCENE_ASSIGNMENT_DINO_COMMON,
    # Minimal GT-supervised scene-specific assignment (no count head, no
    # prototype regression): 100 scene-specific slots are supervised
    # directly with the 15-view majority-voted GT soft instance
    # distributions via Hungarian-aligned BCE/Dice + void BCE + an
    # unmatched-slot penalty (empty slots are pushed toward void).  Slot
    # usage entropy is DISABLED (it fragmented slots in the earlier
    # experiment), and empty slots are dynamically filtered at inference.
    "instance_branch_scene_slot_entropy": 0.0,
    "instance_branch_scene_slot_unmatched": 0.1,
    "instance_branch_scene_slot_min_mass": 1.0,
}

config_doc["semantic_v6_gt_assignment_smoke"] = (
    "Smoke of GT-supervised scene-specific assignment: DINO-enriched unit "
    "features frozen, 100 scene slots directly supervised by GT soft "
    "instance distributions (Hungarian BCE/Dice + void + unmatched), no "
    "usage entropy, eval filters empty slots. Verifies forward/loss/"
    "backward and the adaptive eval channel count."
)
config_defaults["semantic_v6_gt_assignment_smoke"] = Options(
    **{**_SEMANTIC_V6_GT_ASSIGNMENT_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_gt_assignment_smoke",
    experiment_name="semantic_v6_gt_assignment_smoke",
)

config_doc["semantic_v6_gt_assignment_train"] = (
    "GT-supervised scene-specific assignment (3000 steps): frozen "
    "DINO-enriched unit features, 100 scene slots directly supervised by "
    "GT soft instance distributions (Hungarian BCE/Dice + void BCE + "
    "unmatched-slot penalty), no usage entropy. Eval dynamically filters "
    "empty slots. Checkpoints every 200; LSM evals at "
    "500/1000/1500/2000/2500/3000."
)
config_defaults["semantic_v6_gt_assignment_train"] = Options(
    **_SEMANTIC_V6_GT_ASSIGNMENT_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=6,
    max_iters_per_epoch=500,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_gt_assignment_train_3000",
    experiment_name="semantic_v6_gt_assignment_train_3000",
)

_SEMANTIC_V6_DPG_COMMON = {
    **_SEMANTIC_V6_SCENE_ASSIGNMENT_DINO_COMMON,
    # DPG: learn scene-specific instance prototypes (no fixed slots, no
    # agglomerative).  Unit formation / unit image / dino_proj load frozen
    # from the DINO@3000 checkpoint.  Prototypes are FPS-init on the unit
    # embeddings and refined by cross-attention, supervised to match the GT
    # instance prototypes (p_u-weighted unit-embedding means) via Hungarian
    # cosine pull.  Units are hard-assigned by cosine nearest prototype at
    # inference.  K = GT instance count (train / GT-count oracle eval); no
    # count head in this version.
    "instance_branch_dpg": True,
    "instance_branch_dpg_proto_dim": 256,
    "instance_branch_dpg_heads": 4,
    "instance_branch_dpg_layers": 2,
    "instance_branch_dpg_proto_weight": 1.0,
    "instance_branch_scene_assignment": False,
}

config_doc["semantic_v6_dpg_smoke"] = (
    "Smoke of DPG (GT-prototype-supervised scene prototypes + hard cosine "
    "assignment): verifies prototype learning loss (Hungarian cosine pull "
    "to GT instance prototypes), prototype similarity, assignment accuracy, "
    "and the eval hard-assignment render path with K = GT count."
)
config_defaults["semantic_v6_dpg_smoke"] = Options(
    **{**_SEMANTIC_V6_DPG_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_dpg_smoke",
    experiment_name="semantic_v6_dpg_smoke",
)

config_doc["semantic_v6_dpg_train"] = (
    "DPG training (3000 steps): frozen DINO-enriched unit features, learned "
    "scene-specific prototypes supervised by GT instance prototypes "
    "(Hungarian cosine pull), hard cosine assignment.  K = GT instance "
    "count.  Checkpoints every 200; LSM evals at 500/1000/.../3000."
)
config_defaults["semantic_v6_dpg_train"] = Options(
    **_SEMANTIC_V6_DPG_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=6,
    max_iters_per_epoch=500,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_dpg_train_3000",
    experiment_name="semantic_v6_dpg_train_3000",
)

_SEMANTIC_V6_RENDER_SPACE_COMMON = {
    **_SEMANTIC_V6_UNIT_SHAPING_IMG_COMMON,
    # InstanceSplat-style rendered-space embedding supervision on top of
    # the DINO+InfoNCE unit embedding.  No new grouping mechanism; the
    # Agglomerative inference and all frozen components stay unchanged.
    "instance_branch_unit_dino": True,
    "instance_branch_unit_dino_dim": 64,
    "instance_branch_render_space": True,
    "instance_branch_render_space_pull": 1.0,
    "instance_branch_render_space_push": 0.5,
    "instance_branch_render_space_cross": 1.0,
    "instance_branch_render_space_margin_push": 0.5,
    "instance_branch_render_space_margin_cross": 0.2,
    "instance_branch_unit_resume": (
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_unit_shaping_img_dino_train_3000/"
        "model.safetensors"
    ),
}

config_doc["semantic_v6_render_space_smoke"] = (
    "Smoke of rendered-space embedding supervision: verifies the unit "
    "embedding renders through the frozen Gaussians, per-view 3D-consistent "
    "instance maps are built, prototype pull/push/cross-view losses run with "
    "gradients, and the Agglomerative eval path still renders masks."
)
config_defaults["semantic_v6_render_space_smoke"] = Options(
    **{**_SEMANTIC_V6_RENDER_SPACE_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_render_space_smoke",
    experiment_name="semantic_v6_render_space_smoke",
)

config_doc["semantic_v6_render_space_train"] = (
    "Rendered-space embedding supervision (3000 steps): DINO+InfoNCE unit "
    "embedding + prototype pull/push/cross-view rendered-space losses. "
    "Agglomerative inference unchanged. Checkpoints every 200; LSM evals at "
    "500/1000/.../3000."
)
config_defaults["semantic_v6_render_space_train"] = Options(
    **_SEMANTIC_V6_RENDER_SPACE_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_render_space_train_3000",
    experiment_name="semantic_v6_render_space_train_3000",
)

config_doc["semantic_v6_render_space_push_train"] = (
    "Rendered-space embedding supervision with a STRONGER prototype push: "
    "identical to semantic_v6_render_space_train except "
    "render_space_margin_push 0.5 -> 1.0 and render_space_push 0.5 -> 2.0 "
    "(the push hinge was nearly inactive at margin 0.5 vs prototype "
    "distance ~0.7).  Everything else unchanged.  Watch whether push "
    "activates, instance_proto_cos drops, and AP50 exceeds 0.325."
)
config_defaults["semantic_v6_render_space_push_train"] = Options(
    **{
        **_SEMANTIC_V6_RENDER_SPACE_COMMON,
        "instance_branch_render_space_margin_push": 1.0,
        "instance_branch_render_space_push": 2.0,
    },
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_render_space_push_train_3000",
    experiment_name="semantic_v6_render_space_push_train_3000",
)

config_doc["semantic_v6_render_space_info_train"] = (
    "Unit embedding + rendered-space pixel-level InfoNCE (3000 steps): "
    "DINO+InfoNCE unit embedding, rendered into target views and supervised "
    "by pixel-level InfoNCE (same-instance pixels pulled, different pushed) "
    "plus the prototype pull/push/cross terms.  Agglomerative inference "
    "unchanged.  Aim: break the 0.324 Agglomerative ceiling by making the "
    "rendered embedding itself instance-discriminative."
)
config_defaults["semantic_v6_render_space_info_train"] = Options(
    **{
        **_SEMANTIC_V6_RENDER_SPACE_COMMON,
        "instance_branch_render_space_margin_push": 1.0,
        "instance_branch_render_space_push": 2.0,
        "instance_branch_render_space_info_nce": 0.5,
        "instance_branch_render_space_info_temp": 0.2,
        "instance_branch_render_space_info_samples": 32,
        "lr": 1e-4,
        "lr_scheduler": "constant",
    },
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_render_space_info_train_3000",
    experiment_name="semantic_v6_render_space_info_train_3000",
)

config_doc["semantic_v6_rendered_grounding_train"] = (
    "InstanceSplat-style rendered-space instance grounding (minimal first "
    "version): a dedicated instance-feature head on the 8 local units "
    "(unit feature + DINO + 3D position), propagated to the frozen "
    "Gaussians through the GS->unit assignment, rendered into the target "
    "views, and supervised DIRECTLY by the per-view 2D GT instance masks "
    "via prototype pull, prototype push and cross-view consistency.  The "
    "identity InfoNCE / Agglomerative objective is disabled (embed_loss "
    "mode 'none'); unit entropy/compactness regularizers stay.  Eval "
    "clusters the grounding feature with the unchanged LSM protocol for a "
    "fair 0.324 comparison."
)
config_defaults["semantic_v6_rendered_grounding_train"] = Options(
    **{
        **_SEMANTIC_V6_RENDER_SPACE_COMMON,
        "instance_branch_grounding": True,
        "instance_branch_grounding_dim": 64,
        "instance_branch_grounding_3d_pull": 1.0,
        "instance_branch_grounding_3d_push": 0.5,
        "instance_branch_grounding_3d_margin": 0.2,
        "instance_branch_embed_loss_mode": "none",
        "instance_branch_render_space_info_nce": 0.0,
        "instance_branch_render_space_margin_push": 1.0,
        "instance_branch_render_space_push": 2.0,
    },
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_rendered_grounding_train_3000",
    experiment_name="semantic_v6_rendered_grounding_train_3000",
)

config_doc["semantic_v6_dynamic_query_train"] = (
    "Dynamic Scene Instance Query -> direct mask prediction (one-shot "
    "feed-forward).  TokenGS + DINO + 8 Local Units frozen.  Scene-specific "
    "instance queries are generated from the current scene's unit features: "
    "PointGroup-style per-unit 3D center offset prior -> FPS over adjusted "
    "centers -> Mask3D-style cross-attention refinement -> query-unit mask "
    "logits -> propagated to Gaussians through the frozen GS->unit "
    "assignment -> rendered instance masks.  Supervised by GT instance "
    "masks with the existing Hungarian + BCE/Dice + void/unmatched + usage "
    "entropy + 3D consistency.  Inference is a single forward pass that "
    "directly outputs masks (no Agglomerative / clustering / TTA)."
)
config_defaults["semantic_v6_dynamic_query_train"] = Options(
    **{
        **_SEMANTIC_V6_UNIT_SHAPING_IMG_COMMON,
        "instance_branch_unit_dino": True,
        "instance_branch_unit_dino_dim": 64,
        "instance_branch_dynamic_queries": True,
        "instance_branch_num_queries": 64,
        "instance_branch_query_dim": 128,
        "instance_branch_query_layers": 2,
        "instance_branch_query_diversity": 0.5,
        "instance_branch_query_diversity_margin": 0.3,
        "instance_group_num_groups": 64,
        "instance_group_usage_entropy": 0.2,
        "instance_group_min_instance_pixels": 8,
        "lambda_instance_group_void": 0.02,
        "lambda_instance_group_unmatched": 0.02,
    },
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_dynamic_query_train_3000",
    experiment_name="semantic_v6_dynamic_query_train_3000",
)

config_doc["semantic_v6_center_offset_train"] = (
    "PointGroup-style instance-center offset prediction on the frozen "
    "DINO + 8-local-unit representation.  Only a small per-unit 3D "
    "center-offset head trains (regression toward pseudo-GT instance "
    "centers from the 15-view projected per-GS GT labels).  Inference: "
    "offset-adjusted unit centers are tight per instance and grouped by a "
    "simple 3D center clustering (DBSCAN / agglomerative on centers), "
    "replacing the threshold-sensitive embedding clustering capped at "
    "0.369 oracle-eps.  Seeded-oracle evidence: best-center-unit prototype "
    "reaches AP50=0.44, GT-mean prototype 0.534."
)
config_defaults["semantic_v6_center_offset_train"] = Options(
    **{
        **_SEMANTIC_V6_UNIT_SHAPING_IMG_COMMON,
        "instance_branch_unit_dino": True,
        "instance_branch_unit_dino_dim": 64,
        "instance_branch_center_offset": True,
        "instance_branch_center_offset_hidden": 256,
        "instance_center_loss_weight": 1.0,
    },
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_center_offset_train_3000",
    experiment_name="semantic_v6_center_offset_train_3000",
)

_SEMANTIC_V6_DIRECT_GS_COMMON = {
    "model_type": "semantic_tokengs_v6",
    "prompt_training": True,
    "prompt_tokengs_checkpoint": "/space0/mawb/tokengs/workspace/tokengs_re10k/model.safetensors",
    "prompt_clip_model_path": "/space0/mawb/tokengs/checkpoints/clip-vit-large-patch14",
    "data_mode": (("scannet_prompt_small", 1),),
    "dataset_kwargs": {
        "small_manifest_path": (
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "scannet_prompt_full_wide_8x7.json"
        ),
        "wide_target_subsample": 0,
    },
    "prompt_mode": "manifest",
    "img_size": (256, 256),
    "num_input_views": 8,
    "num_views": 15,
    "batch_size": 1,
    "num_workers": 0,
    "use_instance_labels": True,
    "prompt_save_validation_checkpoints": True,
    "instance_branch_token_units": True,
    "instance_branch_direct_gs": True,
    "instance_branch_direct_gs_embed_dim": 8,
    "instance_branch_direct_gs_hidden": 128,
    "instance_branch_direct_gs_dino": True,
    "instance_branch_direct_gs_pull": 1.0,
    "instance_branch_direct_gs_push": 2.0,
    "instance_branch_direct_gs_cross": 1.0,
    "instance_branch_direct_gs_margin_push": 1.0,
    "instance_branch_direct_gs_margin_cross": 0.2,
    "instance_branch_direct_gs_info_nce": 1.0,
    "instance_branch_direct_gs_info_temp": 0.1,
    "instance_branch_direct_gs_info_samples": 32,
}
config_doc["semantic_v6_multidecoder_joint_train"] = (
    "Unified multi-decoder Gaussian representation on the shared TokenGS "
    "latent: GS Decoder (frozen activation head -> reconstruction), "
    "Instance Decoder (DirectGS per-GS compact instance embedding rendered "
    "to 2D, supervised by GT masks via prototype pull/push/cross-view) and "
    "Semantic Decoder (per-GS semantic feature rendered and distilled "
    "toward frozen LSeg dense features, cosine+L1, no GT semantic needed). "
    "The instance/semantic losses back-propagate into the shared token "
    "latent (decoder tail trainable) while L_recon anchors reconstruction "
    "stability.  One-shot feed-forward; no TTA; no Agglomerative."
)
config_defaults["semantic_v6_multidecoder_joint_train"] = Options(
    **{
        **_SEMANTIC_V6_DIRECT_GS_COMMON,
        "semantic_branch_version": "source_projected",
        "lambda_semantic_feature": 1.0,
        "lambda_semantic_cosine": 1.0,
        "lambda_semantic_l1": 0.05,
        "lambda_semantic_source": 0.5,
        "prompt_unfreeze_tokengs": True,
        "instance_branch_backprop_token": True,
    },
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    lr=1e-4,
    lr_scheduler="constant",
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_multidecoder_joint_train_3000",
    experiment_name="semantic_v6_multidecoder_joint_train_3000",
)


_SEMANTIC_V6_MULTIDECODER_GSDINO_COMMON = {
    **_SEMANTIC_V6_DIRECT_GS_COMMON,
    # Same unified multi-task setup as semantic_v6_multidecoder_joint_train
    # (shared gs_token_hidden -> semantic LSeg-lifting head + instance
    # branch + reconstruction decoder), but the instance branch is replaced
    # by the best-known 0.324 recipe (DINO + 8 local units + patch features
    # + soft InfoNCE, Agglomerative eps=0.5 at inference) instead of the
    # DirectGS per-GS embedding.  The semantic / instance branches attach to
    # the shared latent through GradScale gates (forward identity, backward
    # x alpha_sem / alpha_ins) so their gradients are muted relative to the
    # reconstruction path (alpha=1) -- this is the ONLY new mechanism.
    "semantic_branch_version": "source_projected",
    "lambda_semantic_feature": 1.0,
    "lambda_semantic_cosine": 1.0,
    "lambda_semantic_l1": 0.05,
    "lambda_semantic_source": 0.5,
    "prompt_unfreeze_tokengs": True,
    "instance_branch_backprop_token": True,
    "lambda_rgb": 1.0,
    # --- instance branch: 0.324 recipe ---
    "instance_branch_direct_gs": False,
    "instance_branch_unit_embedding": True,
    "instance_branch_unit_encoder": False,
    "instance_branch_unit_image": True,
    "instance_branch_unit_image_dim": 64,
    "instance_branch_unit_dino": True,
    "instance_branch_unit_dino_dim": 64,
    "instance_branch_cluster_eps": 0.5,
    "instance_group_dense_decoder": True,
    "instance_group_dense_multiscale": False,
    "instance_branch_unit_resume": (
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_token_units_train_12000/"
        "checkpoints/model_step_006000.safetensors"
    ),
    "instance_group_num_groups": 100,
    # --- GradScale gates on the shared latent ---
    "grad_scale_sem": 0.3,
    "grad_scale_ins": 0.1,
}

config_doc["semantic_v6_multidecoder_gsdino_joint_train"] = (
    "Unified multi-task shared-representation joint training: the shared "
    "gs_token_hidden latent drives (1) the frozen-activation GS decoder "
    "(reconstruction, gradient alpha=1), (2) the semantic LSeg-lifting head "
    "(gradient alpha_sem=0.3) and (3) the best-known instance branch "
    "(DINO + 8 local units + patch + InfoNCE, gradient alpha_ins=0.1) via "
    "forward-identity/backward-scaled GradScale gates.  Everything else "
    "(losses, data, inference = Agglomerative eps=0.5, eval protocol) is "
    "identical to the 0.324 recipe.  One-shot feed-forward; no TTA."
)
config_defaults["semantic_v6_multidecoder_gsdino_joint_train"] = Options(
    **_SEMANTIC_V6_MULTIDECODER_GSDINO_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    lr=1e-4,
    lr_scheduler="constant",
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_multidecoder_gsdino_joint_train_3000",
    experiment_name="semantic_v6_multidecoder_gsdino_joint_train_3000",
)


_SEMANTIC_V6_GENERATIVE_UNITS_COMMON = {
    # 0.324 instance recipe (DINO + 8 local units + patch + soft InfoNCE,
    # frozen backbone) UNCHANGED on the instance side, plus the generative
    # re-route: the reconstruction path now renders from unit-decoded GS
    # (zero-init residual on the frozen per-GS parameters).  The ONLY new
    # mechanism is that reconstruction gradients flow into the unit
    # formation, making the units the shared intermediate jointly optimized
    # by RGB and instance supervision.  Semantic branch stays off so the
    # experiment isolates reconstruction + instance.
    **_SEMANTIC_V6_UNIT_SHAPING_IMG_COMMON,
    "instance_branch_unit_dino": True,
    "instance_branch_unit_dino_dim": 64,
    "generative_units": True,
    "lambda_rgb": 1.0,
    "instance_branch_backprop_token": False,
    "prompt_unfreeze_tokengs": False,
}

config_doc["semantic_v6_generative_units_joint_train"] = (
    "Generative local units (v1): Token -> 8 local units -> unit-decoded GS. "
    "The unit Gaussian decoder is a zero-initialized residual on the frozen "
    "TokenGS per-GS parameters, so PSNR starts at the frozen level and the "
    "units become the shared intermediate that reconstruction (RGB) and "
    "instance supervision (DINO + 8 units + patch + InfoNCE, Agglomerative "
    "eps=0.5 at inference) jointly optimize.  Instance-side recipe is "
    "byte-identical to the 0.324 baseline; semantic branch off.  Verifies "
    "whether a generative unit representation resolves the joint-training "
    "conflict (success: PSNR stays >= ~19.5 and AP50 > 0.324)."
)
config_defaults["semantic_v6_generative_units_joint_train"] = Options(
    **_SEMANTIC_V6_GENERATIVE_UNITS_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_generative_units_joint_train_3000",
    experiment_name="semantic_v6_generative_units_joint_train_3000",
)


_SEMANTIC_V6_GENERATIVE_UNITS_MASK_COMMON = {
    **_SEMANTIC_V6_GENERATIVE_UNITS_COMMON,
    # v2: keep the generative units (Token -> 8 units -> unit-decoded GS)
    # but replace the InfoNCE embedding + Agglomerative instance side with
    # the rendered-mask assignment path: 100 group tokens cross-attend to
    # the units, unit->group(+void) assignment is propagated to GS, rendered
    # with the UNIT-DECODED Gaussians and supervised by per-scene Hungarian
    # BCE/Dice + 3D anchor consistency.  Because the mask render now
    # consumes the gradient-tracking unit-decoded GS, instance supervision
    # directly shapes the generative representation ("instance-pure units
    # reconstruct better" becomes the aligned goal).  Inference is
    # end-to-end rendered masks -- no Agglomerative, no embedding/clustering
    # mismatch.
    "instance_branch_unit_embedding": False,
    "instance_branch_unit_image": False,
    "instance_branch_unit_dino": False,
}

config_doc["semantic_v6_generative_units_mask_joint_train"] = (
    "Generative units + rendered-mask supervision (v2): Token -> 8 units -> "
    "unit-decoded GS (zero-init residual, PSNR starts at the frozen level); "
    "instance side = 100 group tokens -> unit assignment -> GS -> rendered "
    "instance masks supervised by Hungarian BCE/Dice + 3D consistency, with "
    "the mask loss flowing through the renderer into the unit-decoded GS "
    "(via opacity/geometry) and the unit formation.  Reconstruction (RGB "
    "through the same units) and instance supervision now jointly shape the "
    "generative unit representation.  Inference = rendered masks, no "
    "clustering."
)
config_defaults["semantic_v6_generative_units_mask_joint_train"] = Options(
    **_SEMANTIC_V6_GENERATIVE_UNITS_MASK_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_generative_units_mask_joint_train_3000",
    experiment_name="semantic_v6_generative_units_mask_joint_train_3000",
)

_SEMANTIC_V6_GENERATIVE_UNITS_TEACHER_COMMON = {
    **_SEMANTIC_V6_GENERATIVE_UNITS_MASK_COMMON,
    # v3 (token-aligned structure, teacher bootstrap):
    #  * weights: PURE tokengs_re10k reconstruction (encoder + token
    #    transformer + old GS head).  backbone_resume mirrors the same file
    #    so the generic trainer freezes exactly RE10K weights (no wide7l
    #    semantic fine-tune, no train/eval backbone mismatch).
    #  * units: fresh random 8-local-unit formation + unit Gaussian decoder
    #    (zero-init residual -> step-0 GS == frozen teacher GS); NO warm
    #    start from token-units@6000 because the units are now part of the
    #    generative geometry structure and must co-adapt with RE10K tokens.
    #  * old GS head = frozen teacher: GS-param + rendered-RGB distillation
    #    with a linear decay to zero over gen_teacher_decay_steps; the old
    #    activation-head path is only a bootstrap and is removed later.
    #  * instance side unchanged from v2: 100 unit-level instance queries
    #    (group tokens) cross-attend ALL units (cross-token), unit->group
    #    assignment propagates to GS and masks are rendered from the
    #    unit-decoded Gaussians (Hungarian BCE/Dice + void + 3D).
    "prompt_tokengs_checkpoint": (
        "/space0/mawb/tokengs/workspace/tokengs_re10k/model.safetensors"
    ),
    "instance_branch_unit_resume": "",
    "gen_teacher_distill": True,
    "gen_teacher_decay_steps": 1500,
    "gen_teacher_gs_weight": 1.0,
    "gen_teacher_rgb_weight": 1.0,
    "lambda_rgb": 200.0,
}

config_doc["semantic_v6_generative_units_teacher_train"] = (
    "Generative units + frozen-teacher bootstrap (v3, token-aligned): "
    "load PURE tokengs_re10k (encoder + token transformer + old GS head); "
    "Token->8 shared local units->8 GS/unit decode the FINAL Gaussians "
    "(zero-init residual so step 0 reproduces the teacher geometry).  The "
    "old GS head is a frozen teacher only for early render-space bootstrap: "
    "weighted Gaussian-parameter + rendered-RGB distillation decays linearly "
    "to zero over gen_teacher_decay_steps, after which the old path can be "
    "removed.  Instance supervision: unit-level instance queries (group "
    "tokens) assign units across tokens; masks render from the unit-decoded "
    "GS (Hungarian BCE/Dice).  No encoder adapter / semantic / MBM in this "
    "phase."
)
config_defaults["semantic_v6_generative_units_teacher_train"] = Options(
    **_SEMANTIC_V6_GENERATIVE_UNITS_TEACHER_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/tokengs_re10k/model.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_generative_units_teacher_train_3000",
    experiment_name="semantic_v6_generative_units_teacher_train_3000",
)

_SEMANTIC_V6_ABSOLUTE_UNITS_TEACHER_COMMON = {
    **_SEMANTIC_V6_GENERATIVE_UNITS_MASK_COMMON,
    # Absolute token-aligned student (v3-absolute):
    #  * student = Shared Local Units -> Unit GS Decoder -> COMPLETE student
    #    GS.  No residual on the old head; ``generative_units`` (the residual
    #    route) is OFF.
    #  * teacher = frozen old activation-head path, always no_grad, used only
    #    during the bootstrap/decay window (see abs_* schedule).
    #  * instance side reuses the unit-level group-token path; masks render
    #    from the student GS.
    #  * weights: PURE tokengs_re10k (encoder + token transformer + old head
    #    as teacher); unit formation + absolute decoder start fresh.
    #  (updated) main experiment loads the ScanNet pure-reconstruction
    #  base8k@8000 checkpoint (PSNR 20.12) for encoder + token transformer +
    #  frozen teacher.  The file is the TokenGS-only extraction of
    #  scannet_recon_finetune_base_8k@8000 (no semantic/instance weights;
    #  patch embeddings from the re10k init that the recon fine-tune kept).
    "generative_units": False,
    "instance_branch_abs_units": True,
    "instance_branch_unit_resume": "",
    "prompt_tokengs_checkpoint": (
        "/space0/mawb/tokengs/workspace/scannet_recon_finetune_base_8k/"
        "tokengs_backbone_step_008000.safetensors"
    ),
    "abs_bootstrap_steps": 600,
    "abs_teacher_decay_steps": 400,
    "abs_instance_warmup_steps": 800,
    "abs_teacher_gs_weight": 1.0,
    "abs_teacher_rgb_weight": 1.0,
    "lambda_rgb": 200.0,
}

config_doc["semantic_v6_absolute_units_teacher_train"] = (
    "Absolute token-aligned structure (independent config): load PURE "
    "tokengs_re10k; decoder tokens -> 8 shared local units -> 8 GS/unit -> "
    "COMPLETE student GS.  No additive/dependency on the old GS head: the "
    "frozen old head is only a no_grad teacher during the bootstrap, used "
    "for rendered-RGB + low-weight per-GS distillation (per-token Hungarian "
    "pairing).  Stage schedule: recon/teacher bootstrap "
    "(abs_bootstrap_steps), teacher decay to 0 with the teacher forward "
    "skipped (abs_teacher_decay_steps), then instance loss warm-up "
    "(abs_instance_warmup_steps).  After the teacher turns off the model "
    "must forward/backward/save/load purely through Token->Unit->GS.  "
    "Instance grouping reuses unit-level queries + masks rendered through "
    "the student GS.  The residual generative config above stays untouched "
    "as the对照."
)
config_defaults["semantic_v6_absolute_units_teacher_train"] = Options(
    **_SEMANTIC_V6_ABSOLUTE_UNITS_TEACHER_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/scannet_recon_finetune_base_8k/"
        "tokengs_backbone_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_absolute_units_teacher_base8k_train_3000",
    experiment_name="semantic_v6_absolute_units_teacher_base8k_train_3000",
)

# Ablation kept for comparison: identical absolute-student structure but the
# frozen teacher / encoder / token transformer load the PURE tokengs_re10k
# reconstruction weights (instead of base8k@8000).
config_doc["semantic_v6_absolute_units_teacher_re10k_train"] = (
    "Ablation of the absolute token-aligned student with PURE tokengs_re10k "
    "weights (same structure/stages as the base8k main experiment)."
)
config_defaults["semantic_v6_absolute_units_teacher_re10k_train"] = Options(
    **{
        **_SEMANTIC_V6_ABSOLUTE_UNITS_TEACHER_COMMON,
        "prompt_tokengs_checkpoint": (
            "/space0/mawb/tokengs/workspace/tokengs_re10k/model.safetensors"
        ),
    },
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/tokengs_re10k/model.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_absolute_units_teacher_re10k_train_3000",
    experiment_name="semantic_v6_absolute_units_teacher_re10k_train_3000",
)

# Reconstruction-only continuation (fork from model_step_001000 of the
# absolute base8k run): keep the frozen base8k teacher ON forever and keep
# instance/semantic completely OFF.  Teacher bootstrap steps are effectively
# infinite, so the old 600/1000 decay schedule is never reached; the run is
# a pure-reconstruction "continue bootstrap" to validate whether
# Token -> Unit -> GS can recover PSNR 12.5 -> ~19-20 (teacher level).
config_doc["semantic_v6_absolute_units_recon_continue"] = (
    "Absolute-unit pure-reconstruction continuation (fork): resume weights "
    "from semantic_v6_absolute_units_teacher_base8k_train_3000@1000, keep "
    "the frozen base8k teacher ON with constant weight (never decay, never "
    "enable instance).  Loss = RGB reconstruction + teacher RGB distill + "
    "low-weight GS distill only.  Validation/best = fixed mean PSNR of the "
    "student-only render.  Run 20 epochs x 200 = 4000 extra steps (saved "
    "step N corresponds to absolute step 1000+N, target ~5000 total)."
)
config_defaults["semantic_v6_absolute_units_recon_continue"] = Options(
    **{
        **_SEMANTIC_V6_ABSOLUTE_UNITS_TEACHER_COMMON,
        # Keep the teacher fully ON for this continuation run.
        "abs_bootstrap_steps": 1000000000,
        "abs_teacher_decay_steps": 1,
        "abs_instance_warmup_steps": 1000000000,
    },
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/scannet_recon_finetune_base_8k/"
        "tokengs_backbone_step_008000.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_absolute_units_recon_continue_4000",
    experiment_name="semantic_v6_absolute_units_recon_continue_4000",
)

# Full-data reconstruction-only run (fork from recon model_best): 3 complete
# epochs over the ~5680-sample ScanNet prompt set (no 200-iter cap), teacher
# ON at full weight forever, instance/semantic/MBM OFF, instance branch
# hard-frozen.  LR lowered to 3e-5 (overfit smoke showed large-gradient
# oscillation spikes at 1e-4).  Intra-epoch head checkpoints every 1000
# steps; model_best selected by mean validation PSNR each epoch.
config_doc["semantic_v6_absolute_units_recon_full3"] = (
    "Full-data reconstruction continuation of the absolute-unit student: "
    "resume model_best of semantic_v6_absolute_units_recon_continue_4000, "
    "freeze base8k backbone+teacher, train ONLY absolute_gs_head over 3 full "
    "epochs (~5680 samples/epoch, 17040 steps total), teacher on at full "
    "weight, instance/semantic/MBM off.  Loss = RGB + teacher RGB + low GS "
    "distill.  Best = mean validation PSNR."
)
config_defaults["semantic_v6_absolute_units_recon_full3"] = Options(
    **{
        **_SEMANTIC_V6_ABSOLUTE_UNITS_TEACHER_COMMON,
        "abs_bootstrap_steps": 1000000000,
        "abs_teacher_decay_steps": 1,
        "abs_instance_warmup_steps": 1000000000,
        "abs_freeze_instance": True,
        "abs_ckpt_every": 1000,
        "lr": 3.0e-5,
        "num_epochs": 3,
        "max_iters_per_epoch": 6000,
    },
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/scannet_recon_finetune_base_8k/"
        "tokengs_backbone_step_008000.safetensors"
    ),
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_absolute_units_recon_full3",
    experiment_name="semantic_v6_absolute_units_recon_full3",
)

config_doc["semantic_v6_absolute_units_joint_guarded"] = (
    "Guarded joint baseline (independent experiment): resume the full3 "
    "reconstruction student, keep the base8k encoder/token transformer/old "
    "GS head frozen with the teacher ON forever (RGB + teacher-RGB + "
    "low-weight per-token GS distill), and train ONLY the absolute Shared "
    "Local Unit decoder plus the unit-level instance query/assignment "
    "branch.  Stage schedule with full-data step semantics: "
    "[0, guarded_instance_warmup_steps) instance branch trains with the "
    "student GS detached (instance grads cannot reach absolute units); "
    "[warmup, guarded_instance_full_joint_steps) instance->unit grads and "
    "the effective instance weight ramp linearly; afterwards full joint.  "
    "semantic/prompt/CLIP/LSeg stay hard-frozen and all semantic losses are "
    "0.  model_best is still selected by validation PSNR, but the final "
    "joint model must be chosen with phase checkpoints (PSNR + AP50)."
)
config_defaults["semantic_v6_absolute_units_joint_guarded"] = Options(
    **{
        **_SEMANTIC_V6_ABSOLUTE_UNITS_TEACHER_COMMON,
        "abs_bootstrap_steps": 1000000000,
        "abs_teacher_decay_steps": 1,
        "abs_instance_warmup_steps": 1000000000,
        "abs_freeze_instance": False,
        "abs_ckpt_every": 1000,
        "abs_joint_guarded": True,
        "guarded_instance_head_reinit": True,
        "guarded_instance_warmup_steps": 1000,
        "guarded_instance_full_joint_steps": 5680,
        "guarded_lambda_instance_max": 0.05,
        "guarded_abs_lr": 1.0e-5,
        "guarded_instance_lr": 1.0e-4,
        # Two dedicated LR groups are used for the absolute head and the
        # instance branch; keep the scheduler constant so per-group LR is
        # not overwritten by OneCycleLR.
        "lr_scheduler": "constant",
        "lr": 3.0e-5,
        "num_epochs": 3,
        "max_iters_per_epoch": 6000,
        "prompt_tokengs_checkpoint": (
            "/space/mawb/tokengs/workspace/scannet_recon_finetune_base_8k/"
            "tokengs_backbone_step_008000.safetensors"
        ),
        "prompt_clip_model_path": (
            "/space/mawb/tokengs/checkpoints/clip-vit-large-patch14"
        ),
        "dataset_kwargs": {
            "small_manifest_path": (
                "/space/mawb/tokengs/data/scannet_prompt/"
                "scannet_prompt_full_wide_8x7.json"
            ),
            # ScanNetPromptTrain otherwise reads the protocol yaml whose
            # train-manifest path still points at the 108-era /space0 tree;
            # pass the 240-local provisional manifest explicitly instead.
            "train_manifest_path": (
                "/space/mawb/tokengs/data/scannet_prompt/"
                "scannet_c3g8_train_provisional.json"
            ),
            "query_bank_path": (
                "/space/mawb/tokengs/data/scannet_prompt/"
                "scannet_c3g8_query_bank.json"
            ),
        },
    },
    backbone_resume=(
        "/space/mawb/tokengs/workspace/scannet_recon_finetune_base_8k/"
        "tokengs_backbone_step_008000.safetensors"
    ),
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_absolute_units_joint_guarded",
    experiment_name="semantic_v6_absolute_units_joint_guarded",
)

config_doc["semantic_v6_absolute_units_true_shared_joint"] = (
    "True-Shared guarded joint baseline: one unit formation (q_abs from "
    "AbsoluteUnitDecoder) feeds BOTH the GS decoder (8 GS/unit) and the new "
    "SharedUnitInstanceHead (100 group queries over all 8192 units).  The "
    "legacy dual-unit instance branch is not instantiated.  Instance mask "
    "rendering detaches Student GS geometry, so instance gradients reach "
    "tok_norm/tok_proj/unit_queries/unit_readout but never "
    "center_mlp/gs_decoder/slot_emb.  Head loss weight and the "
    "instance->q_abs unit multiplier are independent schedules; RGB/teacher "
    "reconstruction guardrails stay ON.  Full3 absolute_gs_head is resumed "
    "strictly 24/24; SharedUnitInstanceHead starts fresh."
)
config_defaults["semantic_v6_absolute_units_true_shared_joint"] = Options(
    **{
        **_SEMANTIC_V6_ABSOLUTE_UNITS_TEACHER_COMMON,
        "abs_bootstrap_steps": 1000000000,
        "abs_teacher_decay_steps": 1,
        "abs_instance_warmup_steps": 1000000000,
        "abs_freeze_instance": False,
        "abs_ckpt_every": 1000,
        "abs_joint_guarded": False,
        "abs_true_shared_units": True,
        "instance_branch_token_units": False,
        "instance_branch_independent": False,
        "instance_branch_abs_units": True,
        "tsh_instance_warmup_steps": 1000,
        "tsh_instance_ramp_end_steps": 5680,
        "tsh_lambda_instance": 0.05,
        "tsh_unit_gradient_multiplier_max": 32.0,
        "tsh_abs_lr": 1.0e-5,
        "tsh_instance_lr": 1.0e-4,
        "lr_scheduler": "constant",
        "lr": 3.0e-5,
        "num_epochs": 3,
        "max_iters_per_epoch": 6000,
        "prompt_tokengs_checkpoint": (
            "/space/mawb/tokengs/workspace/scannet_recon_finetune_base_8k/"
            "tokengs_backbone_step_008000.safetensors"
        ),
        "prompt_clip_model_path": (
            "/space/mawb/tokengs/checkpoints/clip-vit-large-patch14"
        ),
        "dataset_kwargs": {
            "small_manifest_path": (
                "/space/mawb/tokengs/data/scannet_prompt/"
                "scannet_prompt_full_wide_8x7.json"
            ),
            "train_manifest_path": (
                "/space/mawb/tokengs/data/scannet_prompt/"
                "scannet_c3g8_train_provisional.json"
            ),
            "query_bank_path": (
                "/space/mawb/tokengs/data/scannet_prompt/"
                "scannet_c3g8_query_bank.json"
            ),
        },
    },
    backbone_resume=(
        "/space/mawb/tokengs/workspace/scannet_recon_finetune_base_8k/"
        "tokengs_backbone_step_008000.safetensors"
    ),
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_absolute_units_true_shared_joint",
    experiment_name="semantic_v6_absolute_units_true_shared_joint",
)

config_doc["semantic_v6_absolute_units_true_shared_joint_m4"] = (
    "Formal-ready True-Shared joint baseline with "
    "tsh_unit_gradient_multiplier_max=4.0.  Identical to the m0 head-only "
    "config except for the instance->q_abs unit multiplier."
)
config_defaults["semantic_v6_absolute_units_true_shared_joint_m4"] = Options(
    **{
        **_SEMANTIC_V6_ABSOLUTE_UNITS_TEACHER_COMMON,
        "abs_bootstrap_steps": 1000000000,
        "abs_teacher_decay_steps": 1,
        "abs_instance_warmup_steps": 1000000000,
        "abs_freeze_instance": False,
        "abs_ckpt_every": 1000,
        "abs_joint_guarded": False,
        "abs_true_shared_units": True,
        "instance_branch_token_units": False,
        "instance_branch_independent": False,
        "instance_branch_abs_units": True,
        "tsh_instance_warmup_steps": 1000,
        "tsh_instance_ramp_end_steps": 5680,
        "tsh_lambda_instance": 0.05,
        "tsh_unit_gradient_multiplier_max": 4.0,
        "tsh_abs_lr": 1.0e-5,
        "tsh_instance_lr": 1.0e-4,
        "lr_scheduler": "constant",
        "lr": 3.0e-5,
        "num_epochs": 3,
        "max_iters_per_epoch": 6000,
        "prompt_tokengs_checkpoint": (
            "/space/mawb/tokengs/workspace/scannet_recon_finetune_base_8k/"
            "tokengs_backbone_step_008000.safetensors"
        ),
        "prompt_clip_model_path": (
            "/space/mawb/tokengs/checkpoints/clip-vit-large-patch14"
        ),
        "dataset_kwargs": {
            "small_manifest_path": (
                "/space/mawb/tokengs/data/scannet_prompt/"
                "scannet_prompt_full_wide_8x7.json"
            ),
            "train_manifest_path": (
                "/space/mawb/tokengs/data/scannet_prompt/"
                "scannet_c3g8_train_provisional.json"
            ),
            "query_bank_path": (
                "/space/mawb/tokengs/data/scannet_prompt/"
                "scannet_c3g8_query_bank.json"
            ),
        },
    },
    backbone_resume=(
        "/space/mawb/tokengs/workspace/scannet_recon_finetune_base_8k/"
        "tokengs_backbone_step_008000.safetensors"
    ),
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_absolute_units_true_shared_joint_m4",
    experiment_name="semantic_v6_absolute_units_true_shared_joint_m4",
)

config_doc["semantic_v6_absolute_units_true_shared_head_only_m0"] = (
    "Head-only control: tsh_unit_gradient_multiplier_max=0.0, so the "
    "instance head trains normally but q_abs is permanently detached from "
    "the instance loss.  GT reconstruction still trains absolute unit "
    "formation and GS decoder."
)
config_defaults["semantic_v6_absolute_units_true_shared_head_only_m0"] = (
    Options(
        **{
            **_SEMANTIC_V6_ABSOLUTE_UNITS_TEACHER_COMMON,
            "abs_bootstrap_steps": 1000000000,
            "abs_teacher_decay_steps": 1,
            "abs_instance_warmup_steps": 1000000000,
            "abs_freeze_instance": False,
            "abs_ckpt_every": 1000,
            "abs_joint_guarded": False,
            "abs_true_shared_units": True,
            "instance_branch_token_units": False,
            "instance_branch_independent": False,
            "instance_branch_abs_units": True,
            "tsh_instance_warmup_steps": 1000,
            "tsh_instance_ramp_end_steps": 5680,
            "tsh_lambda_instance": 0.05,
            "tsh_unit_gradient_multiplier_max": 0.0,
            "tsh_abs_lr": 1.0e-5,
            "tsh_instance_lr": 1.0e-4,
            "lr_scheduler": "constant",
            "lr": 3.0e-5,
            "num_epochs": 3,
            "max_iters_per_epoch": 6000,
            "prompt_tokengs_checkpoint": (
                "/space/mawb/tokengs/workspace/"
                "scannet_recon_finetune_base_8k/"
                "tokengs_backbone_step_008000.safetensors"
            ),
            "prompt_clip_model_path": (
                "/space/mawb/tokengs/checkpoints/clip-vit-large-patch14"
            ),
            "dataset_kwargs": {
                "small_manifest_path": (
                    "/space/mawb/tokengs/data/scannet_prompt/"
                    "scannet_prompt_full_wide_8x7.json"
                ),
                "train_manifest_path": (
                    "/space/mawb/tokengs/data/scannet_prompt/"
                    "scannet_c3g8_train_provisional.json"
                ),
                "query_bank_path": (
                    "/space/mawb/tokengs/data/scannet_prompt/"
                    "scannet_c3g8_query_bank.json"
                ),
            },
        },
        backbone_resume=(
            "/space/mawb/tokengs/workspace/"
            "scannet_recon_finetune_base_8k/"
            "tokengs_backbone_step_008000.safetensors"
        ),
        print_freq=10,
        log_image_freq=100,
        mixed_precision="bf16",
        workspace=(
            "workspace/semantic_v6_absolute_units_true_shared_head_only_m0"
        ),
        experiment_name=(
            "semantic_v6_absolute_units_true_shared_head_only_m0"
        ),
    )
)

config_doc["semantic_v6_absolute_units_true_shared_joint_m4_ddp8"] = (
    "DDP8 variant of the m4 True-Shared baseline: stage thresholds are "
    "converted from single-GPU samples to 8-GPU optimizer steps "
    "(head warm-up 0-125, unit ramp 125-710, full joint after 710).  "
    "Same per-rank LR 1e-5 / 1e-4, multiplier 4.0, 3 epochs x ~710 "
    "optimizer steps = ~2130 steps, equivalent to 3 x 5680 global samples."
)
config_defaults[
    "semantic_v6_absolute_units_true_shared_joint_m4_ddp8"
] = config_defaults[
    "semantic_v6_absolute_units_true_shared_joint_m4"
].evolve(
    tsh_ddp8=True,
    tsh_instance_warmup_steps=125,
    tsh_instance_ramp_end_steps=710,
    abs_ckpt_every=125,
    workspace="workspace/semantic_v6_absolute_units_true_shared_joint_m4_ddp8",
    experiment_name=(
        "semantic_v6_absolute_units_true_shared_joint_m4_ddp8"
    ),
)

config_doc["semantic_v6_absolute_units_true_shared_head_only_m0_ddp8"] = (
    "DDP8 variant of the m0 head-only control: same head-loss schedule as "
    "m4_ddp8 (0-125 warm-up etc.), instance->q_abs multiplier stays 0."
)
config_defaults[
    "semantic_v6_absolute_units_true_shared_head_only_m0_ddp8"
] = config_defaults[
    "semantic_v6_absolute_units_true_shared_head_only_m0"
].evolve(
    tsh_ddp8=True,
    tsh_instance_warmup_steps=125,
    tsh_instance_ramp_end_steps=710,
    abs_ckpt_every=125,
    workspace="workspace/semantic_v6_absolute_units_true_shared_head_only_m0_ddp8",
    experiment_name=(
        "semantic_v6_absolute_units_true_shared_head_only_m0_ddp8"
    ),
)

config_doc["semantic_v6_absolute_units_true_shared_common_warmup_ddp8"] = (
    "Common warm-up experiment for the causal m0/m4 fork.  Train only to "
    "optimizer step 125 (use --num-epochs 1 --max-iters-per-epoch 125), "
    "which saves a full workspace state (model + optimizer + scheduler + "
    "metadata with optimizer_step=125).  The saved state is then copied to "
    "two fork workspaces and resumed with --tsh-fork-continue-step 125 so "
    "m0/m4 share an identical step-125 state before the unit multiplier "
    "diverges."
)
config_defaults[
    "semantic_v6_absolute_units_true_shared_common_warmup_ddp8"
] = config_defaults[
    "semantic_v6_absolute_units_true_shared_joint_m4_ddp8"
].evolve(
    workspace="workspace/semantic_v6_absolute_units_true_shared_common_warmup_ddp8",
    experiment_name=(
        "semantic_v6_absolute_units_true_shared_common_warmup_ddp8"
    ),
)

_SIU3R_MBM_DDP8_COMMON_OVERRIDES = {
    # Official SIU3R: MASt3R image encoder frozen, panoptic/decoder heads
    # trained; the Gaussian decoder / token transformer tail gets its own
    # low-LR group here (1e-6) while the old activation-head teacher stays
    # frozen and only runs during an early no_grad bootstrap window.
    "tsh_ddp8": True,
    "tsh_instance_warmup_steps": 125,
    "tsh_instance_ramp_end_steps": 710,
    "abs_ckpt_every": 710,
    "abs_bootstrap_steps": 125,
    "abs_teacher_decay_steps": 125,
    "abs_instance_warmup_steps": 1000000000,
    "num_epochs": 3,
    "max_iters_per_epoch": 710,
    "prompt_unfreeze_tokengs": True,
    "prompt_unfreeze_tokengs_mode": "decoder",
    "prompt_unfreeze_tokengs_lr": 1.0e-6,
    "tsh_mbm_decoder_tail_lr": 1.0e-6,
    # U->R schedule (optimizer steps, DDP8): no mask-guided geometry signal
    # before the instance head is reliable (step 710 = epoch-1 end), then a
    # linear 0 -> 1 ramp of the official 0.05 weight across epoch 2.
    "tsh_mbm_u2r_weight": 0.05,
    "tsh_mbm_u2r_warmup_start_step": 710,
    "tsh_mbm_u2r_warmup_steps": 710,
    "tsh_mbm_u2r_min_conf": 0.3,
    "tsh_mbm_u2r_min_alpha": 0.05,
}

config_doc["semantic_v6_absolute_units_true_shared_siu3r_mbm_r2u_ddp8"] = (
    "SIU3R-mapped R->U-only DDP8 variant.  Official Multi-View Mask "
    "Aggregation is an inference-time 2D-mask -> per-GS -> splat-render "
    "operation with no training module; TokenGS realizes it structurally "
    "with unit-level assignments rendered through the shared Student GS.  "
    "This variant therefore keeps the R->U evidence path (instance "
    "supervision, unit gradient multiplier 4) ON and the U->R depth "
    "smoothness OFF, replicating the m4 baseline while unlocking the "
    "decoder-tail LR group and the early-teacher schedule used by the "
    "MBM family.  Development init: m0_ddp8 model_step_001420."
)
config_defaults[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_r2u_ddp8"
] = config_defaults[
    "semantic_v6_absolute_units_true_shared_joint_m4_ddp8"
].evolve(
    **_SIU3R_MBM_DDP8_COMMON_OVERRIDES,
    tsh_mbm_mode="r2u",
    workspace=(
        "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_r2u_ddp8"
    ),
    experiment_name=(
        "semantic_v6_absolute_units_true_shared_siu3r_mbm_r2u_ddp8"
    ),
)

config_doc["semantic_v6_absolute_units_true_shared_siu3r_mbm_u2r_ddp8"] = (
    "SIU3R-mapped U->R-only DDP8 variant: official Mask-Guided Geometry "
    "Refinement (rendered-depth smoothness weighted by detached predicted "
    "instance interiors, pipeline.py:249-265) ON with the official 0.05 "
    "weight; instance->q_abs unit multiplier stays 0 so the ONLY new "
    "instance-driven geometry signal is the mask-guided depth smoothness.  "
    "This isolates whether U->R alone helps reconstruction/AP50."
)
config_defaults[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_u2r_ddp8"
] = config_defaults[
    "semantic_v6_absolute_units_true_shared_head_only_m0_ddp8"
].evolve(
    **_SIU3R_MBM_DDP8_COMMON_OVERRIDES,
    tsh_mbm_mode="u2r",
    tsh_unit_gradient_multiplier_max=0.0,
    workspace=(
        "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_u2r_ddp8"
    ),
    experiment_name=(
        "semantic_v6_absolute_units_true_shared_siu3r_mbm_u2r_ddp8"
    ),
)

config_doc["semantic_v6_absolute_units_true_shared_siu3r_mbm_both_ddp8"] = (
    "SIU3R-mapped MBM Both DDP8 variant (formal development target): the "
    "true-shared instance path with unit gradient multiplier 4 (R->U "
    "evidence at unit level) PLUS the official mask-guided rendered-depth "
    "smoothness (U->R), weight 0.05 ramping across epoch 2.  Teacher only "
    "stabilizes steps 0-250 then decays to 0 and its old-head forward is "
    "skipped; decoder-tail LR 1e-6; GT RGB reconstruction stays the "
    "primary supervision (lambda_rgb=200 TokenGS MSE regime)."
)
config_defaults[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_ddp8"
] = config_defaults[
    "semantic_v6_absolute_units_true_shared_joint_m4_ddp8"
].evolve(
    **_SIU3R_MBM_DDP8_COMMON_OVERRIDES,
    tsh_mbm_mode="both",
    workspace=(
        "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_ddp8"
    ),
    experiment_name=(
        "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_ddp8"
    ),
)

config_doc[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
] = (
    "Calibrated SIU3R-MBM Both variant (16-batch empirical calibration): "
    "U->R max weight 10.0 (median U->R/GT grad ratio ~1.5% on unit "
    "formation and ~0.5% on the GS decoder; P90 <= 3.4%; max <= 5.2%) and "
    "shared decoder-tail LR 3e-6.  Everything else (formula, mask policy, "
    "depth source, warm-up 710->1420, True Shared structure, instance "
    "lambda 0.05, unit multiplier 4) is identical to the W=0.05 baseline."
)
config_defaults[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
] = config_defaults[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_ddp8"
].evolve(
    tsh_mbm_u2r_weight=10.0,
    tsh_mbm_decoder_tail_lr=3.0e-6,
    prompt_unfreeze_tokengs_lr=3.0e-6,
    workspace=(
        "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_"
        "w10_t3e6_ddp8"
    ),
    experiment_name=(
        "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
    ),
)

config_doc[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_scene_hungarian_ddp8"
] = (
    "Scene-level Hungarian continuation from MBM Both@1420.  The True-Shared "
    "unit/GS pipeline, calibrated LR groups, and MBM weights are unchanged; "
    "only the target-view instance matching scope changes.  The continuation "
    "uses DDP8 optimizer-step schedules: head adaptation 0-125, unit and "
    "U->R ramps 125-355, then full joint through step 710.  No PGSR."
)
config_defaults[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_scene_hungarian_ddp8"
] = config_defaults[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
].evolve(
    instance_group_scene_level_matching=True,
    tsh_instance_warmup_steps=125,
    tsh_instance_ramp_end_steps=355,
    tsh_mbm_u2r_warmup_start_step=125,
    tsh_mbm_u2r_warmup_steps=230,
    tsh_per_gs_refine=False,
    abs_bootstrap_steps=0,
    abs_teacher_decay_steps=0,
    num_epochs=1,
    max_iters_per_epoch=710,
    abs_ckpt_every=710,
    abs_ckpt_steps_extra=(125, 355),
    workspace=(
        "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_"
        "scene_hungarian_ddp8"
    ),
    experiment_name=(
        "semantic_v6_absolute_units_true_shared_siu3r_mbm_scene_hungarian_ddp8"
    ),
)

config_doc[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_r2u_t3e6_ddp8"
] = (
    "Calibrated R->U control with the same decoder-tail LR 3e-6 as the "
    "W=10 Both variant and U->R disabled (MBM-off control for the short "
    "trainer comparison and later long ablation)."
)
config_defaults[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_r2u_t3e6_ddp8"
] = config_defaults[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_r2u_ddp8"
].evolve(
    tsh_mbm_u2r_weight=0.0,
    tsh_mbm_decoder_tail_lr=3.0e-6,
    prompt_unfreeze_tokengs_lr=3.0e-6,
    workspace=(
        "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_r2u_"
        "t3e6_ddp8"
    ),
    experiment_name=(
        "semantic_v6_absolute_units_true_shared_siu3r_mbm_r2u_t3e6_ddp8"
    ),
)

config_doc[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_w10_t3e6_pgsr_ddp8"
] = (
    "Per-GS slot refinement fork of the calibrated Both run (development "
    "checkpoint model_step_001420).  The only unit formation stays q_abs; "
    "SharedUnitInstanceHead outputs unit_logits, which are broadcast to the "
    "unit's 8 GS and refined by a zero-initialized per-GS residual branch "
    "(alpha zero-init, gate ramps 0 -> 1 over the first 125 optimizer "
    "steps).  Student-GS attributes enter the residual branch detached, so "
    "instance BCE/Dice never reach center_mlp/gs_decoder/reconstruction "
    "slot_emb.  One epoch = 710 DDP8 steps; stage checkpoints 125/355/710. "
    "Teacher permanently off; semantic/DINO/CLIP/LSeg off."
)
config_defaults[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_w10_t3e6_pgsr_ddp8"
] = config_defaults[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
].evolve(
    tsh_per_gs_refine=True,
    tsh_per_gs_ramp_steps=125,
    # Fork-continuation schedule inside the single 710-step epoch:
    # pretrained heads keep training immediately, unit gradients reach full
    # strength by step 50, U->R (W=10) ramps 50 -> 200, per-GS gate 0 -> 125.
    tsh_instance_warmup_steps=0,
    tsh_instance_ramp_end_steps=50,
    tsh_mbm_u2r_warmup_start_step=50,
    tsh_mbm_u2r_warmup_steps=150,
    abs_bootstrap_steps=0,
    abs_teacher_decay_steps=0,
    num_epochs=1,
    max_iters_per_epoch=710,
    abs_ckpt_every=710,
    abs_ckpt_steps_extra=(125, 355),
    workspace=(
        "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_w10_"
        "t3e6_pgsr_ddp8"
    ),
    experiment_name=(
        "semantic_v6_absolute_units_true_shared_siu3r_mbm_w10_t3e6_pgsr_ddp8"
    ),
)


config_doc["semantic_v6_direct_gs_smoke"] = (
    "Smoke of InstanceSplat-style direct per-GS instance embedding: verifies "
    "the compact embedding renders into target views, 2D GT-mask pull/push/"
    "cross losses run with gradients (backbone frozen), and the eval "
    "k-means clustering path produces a probability map."
)
config_defaults["semantic_v6_direct_gs_smoke"] = Options(
    **{**_SEMANTIC_V6_DIRECT_GS_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_direct_gs_smoke",
    experiment_name="semantic_v6_direct_gs_smoke",
)

config_doc["semantic_v6_direct_gs_train"] = (
    "InstanceSplat-style direct per-GS instance embedding (3000 steps): "
    "frozen TokenGS geometry, compact 8-D per-GS embedding rendered to "
    "target views and supervised by 2D GT masks (pull/push/cross). Eval "
    "clusters the rendered embedding per view (GT-count oracle). Checkpoints "
    "every 200; LSM evals at 1000/2000/3000."
)
config_defaults["semantic_v6_direct_gs_train"] = Options(
    **_SEMANTIC_V6_DIRECT_GS_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=15,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    lr=1e-4,
    lr_scheduler="constant",
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_direct_gs_train_3000",
    experiment_name="semantic_v6_direct_gs_train_3000",
)

_SEMANTIC_V6_UNIT_SHAPING_IMG_PROTO_COMMON = {
    **_SEMANTIC_V6_UNIT_SHAPING_IMG_CONF_COMMON,
    # Identity-embedding objective only: replace soft InfoNCE with a soft
    # prototype/center CE over per-scene soft instance centers (built from
    # the pseudo-label distributions, detached) plus a purity-weighted
    # margin separation. Unit formation + patch feature + agglomerative
    # clustering (eps=0.5) + rendering + all other supervision unchanged.
    "instance_branch_embed_loss_mode": "softproto_margin",
    "instance_branch_embed_proto_temp": 0.1,
    "instance_branch_embed_margin": 0.2,
    "instance_branch_embed_margin_weight": 1.0,
    "instance_branch_unit_resume": (
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_unit_shaping_img_train_3000/"
        "checkpoints/model_step_002000.safetensors"
    ),
}

config_doc["semantic_v6_unit_shaping_img_proto_smoke"] = (
    "Smoke of soft prototype + margin identity-embedding objective: verifies "
    "the per-scene soft centers, soft prototypical CE, purity-weighted margin "
    "and all monitors (same/diff similarity, margin, kNN agreement, collapse) "
    "train with the frozen-backbone recipe, and the eval agglomerative "
    "clustering path still renders."
)
config_defaults["semantic_v6_unit_shaping_img_proto_smoke"] = Options(
    **{**_SEMANTIC_V6_UNIT_SHAPING_IMG_PROTO_COMMON, "max_eval_iters": 1},
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_unit_shaping_img_proto_smoke",
    experiment_name="semantic_v6_unit_shaping_img_proto_smoke",
)

config_doc["semantic_v6_unit_shaping_img_proto_train"] = (
    "Soft prototype + margin identity-embedding (3000 steps, warm-started "
    "from unit-shaping+patch@2000): soft prototypical CE to per-scene "
    "instance centers + purity-weighted margin separation replace soft "
    "InfoNCE. Everything else identical to the 0.279 run (unit formation "
    "trainable, patch feature, agglomerative eps=0.5, pseudo conf 0.6/3/0.25). "
    "Checkpoints every 500; LSM evals at 1000/2000/3000."
)
config_defaults["semantic_v6_unit_shaping_img_proto_train"] = Options(
    **_SEMANTIC_V6_UNIT_SHAPING_IMG_PROTO_COMMON,
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=6,
    max_iters_per_epoch=500,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_unit_shaping_img_proto_train_3000",
    experiment_name="semantic_v6_unit_shaping_img_proto_train_3000",
)

_SEMANTIC_V6_LB_COMMON = {
    **_SEMANTIC_V6_DENSE_COMMON,
    # Official TokenGS latent-bottleneck backbone (DL3DV 6-view, SSIM
    # variant): 12-layer multiscale encoder -> 4096 scene latent tokens ->
    # 4096 Gaussian tokens -> 262,144 Gaussians. This is both a better
    # reconstruction backbone and 4x the Gaussian resolution of the old
    # 1024-token re10k backbone, attacking the two known ceilings (PSNR
    # 19.8 vs InstOk3D 22.4, and 64K vs 262K primitives for boundaries).
    # The backbone is frozen; the semantic + instance heads warm-start from
    # pgr3df2@12000 and retrain on the new token distribution.
    **_LATENT_D12_ARCH,
    "num_gs_tokens": 4096,
    "img_size": (256, 256),
    "camera_normalization_method": "mean_cam",
    "camera_scale_method": "constant",
    "prompt_tokengs_checkpoint": (
        "/space0/mawb/tokengs/checkpoints/dl3dv_latent_6v_ssim.safetensors"
    ),
    "instance_group_dense_decoder": False,
    "prompt_unfreeze_tokengs": False,
}

config_doc["semantic_v6_open_vocab_lb_smoke"] = (
    "Twenty-step smoke of the latent-bottleneck backbone (4096 GS tokens, "
    "262K Gaussians) with the pgr3df2 heads warm-started: checks strict "
    "backbone loading, render memory at 4x Gaussians, and that the head "
    "trains on the new token distribution."
)
config_defaults["semantic_v6_open_vocab_lb_smoke"] = Options(
    **{**_SEMANTIC_V6_LB_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_lb_smoke_20",
    experiment_name="semantic_v6_open_vocab_lb_smoke_20",
)

config_doc["semantic_v6_open_vocab_lb_train"] = (
    "Retrain the semantic + instance heads on the official latent-bottleneck "
    "backbone (4096 GS tokens / 262K Gaussians, DL3DV 6-view SSIM variant). "
    "Backbone frozen, heads warm-started from pgr3df2@12000 (AP50 0.231 on "
    "the old 64K-Gaussian backbone). 6000 steps: recent head variants peak "
    "at 2000~6000; checkpoints saved every 200 steps, LSM evals at "
    "2000/4000/6000. Watch render memory (4x Gaussians)."
)
config_defaults["semantic_v6_open_vocab_lb_train"] = Options(
    **_SEMANTIC_V6_LB_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_lb_train_6000",
    experiment_name="semantic_v6_open_vocab_lb_train_6000",
)

_SEMANTIC_V6_LB2_COMMON = {
    **_SEMANTIC_V6_LB_COMMON,
    # Instance-only training on the latent-bottleneck backbone. All semantic
    # supervision is off (open-vocab CE, CLIP feature distillation, prompt
    # BCE/Dice, RGB) so the ONLY training signal is the instance-group loss
    # (2D rendered Hungarian + 3D anchor-level) plus the cross-view instance
    # InfoNCE. The semantic adapters get no gradient and stay frozen in
    # practice; only instance_group_head updates. Warm start from the first
    # run's peak (lb@6000, AP50 0.164 / PSNR 26.27) and train 10000
    # uninterrupted steps (LSM evals at 6000/7000/8000/9000/10000).
    "lambda_ce_cosine": 0.0,
    "lambda_feat": 0.0,
    "prompt_lambda_bce": 0.0,
    "prompt_lambda_dice": 0.0,
    "lambda_rgb": 0.0,
    "lambda_instance_contrastive": 0.1,
}

config_doc["semantic_v6_open_vocab_lb2_smoke"] = (
    "Twenty-step smoke of instance-only latent-bottleneck training: all "
    "semantic losses off, cross-view instance InfoNCE on (0.1), warm "
    "started from the first run's peak lb@6000."
)
config_defaults["semantic_v6_open_vocab_lb2_smoke"] = Options(
    **{**_SEMANTIC_V6_LB2_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_lb_train_6000/"
        "checkpoints/model_step_006000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_lb2_smoke_20",
    experiment_name="semantic_v6_open_vocab_lb2_smoke_20",
)

config_doc["semantic_v6_open_vocab_lb2_train"] = (
    "Instance-only latent-bottleneck training: warm start from lb@6000 "
    "(first run peak, AP50 0.164 / PSNR 26.27), ALL semantic losses off "
    "(only instance-group 2D+3D loss and cross-view InfoNCE drive "
    "training), 10000 uninterrupted steps. Isolates whether instance-only "
    "supervision on the strong 262K-Gaussian backbone can break the old "
    "0.231 AP50 ceiling."
)
config_defaults["semantic_v6_open_vocab_lb2_train"] = Options(
    **_SEMANTIC_V6_LB2_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_lb_train_6000/"
        "checkpoints/model_step_006000.safetensors"
    ),
    num_epochs=50,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_lb2_train_10000",
    experiment_name="semantic_v6_open_vocab_lb2_train_10000",
)

_SEMANTIC_V6_LB3_COMMON = {
    **_SEMANTIC_V6_LB_COMMON,
    # Joint-loss latent-bottleneck training, fixed run: warm start from the
    # first run's peak lb@6000 (AP50 0.164) and train 10000 UNINTERRUPTED
    # steps (the earlier continuation reset the optimizer LR and overfit;
    # the instance-only lb2 run destroyed the warm-start because removing
    # the semantic losses changed the optimization landscape). Semantic CE
    # is down-weighted (1.0 -> 0.2) as a structured regularizer, CLIP feat
    # distillation and prompt losses stay on, cross-view instance InfoNCE
    # (0.1) is added. LR stays continuous at 1e-4 across the whole run.
    "lambda_ce_cosine": 0.2,
    "lambda_instance_contrastive": 0.1,
}

config_doc["semantic_v6_open_vocab_lb3_smoke"] = (
    "Twenty-step smoke of the joint-loss latent-bottleneck run: warm start "
    "lb@6000, semantic CE down-weighted to 0.2 (kept as regularizer), "
    "cross-view instance InfoNCE 0.1."
)
config_defaults["semantic_v6_open_vocab_lb3_smoke"] = Options(
    **{**_SEMANTIC_V6_LB3_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_lb_train_6000/"
        "checkpoints/model_step_006000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_lb3_smoke_20",
    experiment_name="semantic_v6_open_vocab_lb3_smoke_20",
)

config_doc["semantic_v6_open_vocab_lb3_train"] = (
    "Joint-loss latent-bottleneck training, 10000 uninterrupted steps from "
    "lb@6000: semantic CE 0.2 (regularizer), CLIP feat + prompt losses on, "
    "instance InfoNCE 0.1, LR continuous at 1e-4. Tests whether a single "
    "long run with the semantic regularizer kept can push the 262K-Gaussian "
    "backbone's instance AP50 past the first run's 0.164 toward the old "
    "0.231 ceiling. LSM evals at 6000/7000/8000/9000/10000."
)
config_defaults["semantic_v6_open_vocab_lb3_train"] = Options(
    **_SEMANTIC_V6_LB3_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_lb_train_6000/"
        "checkpoints/model_step_006000.safetensors"
    ),
    num_epochs=50,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_lb3_train_10000",
    experiment_name="semantic_v6_open_vocab_lb3_train_10000",
)

_SEMANTIC_V6_LB4_COMMON = {
    **_SEMANTIC_V6_LB3_COMMON,
    # First controlled fix for feed-forward grouping: ScanNet instance ids
    # are scene-global, so assign each instance to one group jointly across
    # all target views. Keep only the direct 2D group objective while this
    # behavior is validated; the old projected 3D labels and usage entropy
    # introduced conflicting supervision.
    "instance_group_scene_level_matching": True,
    "lambda_instance_group_3d": 0.0,
    "instance_group_usage_entropy": 0.0,
    "lambda_instance_contrastive": 0.0,
    # Instance-head-only ablation on the frozen latent TokenGS backbone.
    "lambda_ce_cosine": 0.0,
    "lambda_feat": 0.0,
    "prompt_lambda_bce": 0.0,
    "prompt_lambda_dice": 0.0,
    "lambda_rgb": 0.0,
    "lambda_instance_dense_aux": 0.0,
    "instance_group_lambda_warmup_steps": 0,
}

config_doc["semantic_v6_open_vocab_lb4_smoke"] = (
    "Twenty-step configuration smoke for scene-level multi-view Hungarian "
    "matching. Frozen latent backbone; only the direct 2D instance-group "
    "CE, BCE, Dice, void, and unmatched losses are active."
)
config_defaults["semantic_v6_open_vocab_lb4_smoke"] = Options(
    **{**_SEMANTIC_V6_LB4_COMMON, "max_eval_iters": 2},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_lb3_train_10000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_lb4_smoke_20",
    experiment_name="semantic_v6_open_vocab_lb4_smoke_20",
)

config_doc["semantic_v6_open_vocab_lb4_train"] = (
    "Scene-level multi-view Hungarian instance-head training from lb3@8000. "
    "The latent TokenGS backbone stays frozen; noisy projected 3D loss, "
    "usage entropy, semantic/prompt losses, and instance contrastive loss "
    "are disabled for this controlled 2D grouping ablation."
)
config_defaults["semantic_v6_open_vocab_lb4_train"] = Options(
    **_SEMANTIC_V6_LB4_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_lb3_train_10000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_lb4_train_6000",
    experiment_name="semantic_v6_open_vocab_lb4_train_6000",
)

_SEMANTIC_V6_LB5_COMMON = {
    **_SEMANTIC_V6_LB4_COMMON,
    # Render the same Gaussian group field into both the eight observed
    # context views and seven held-out target views. One scene-level matching
    # is shared across all 15 views, directly teaching the feed-forward head
    # to recover from the evidence it actually receives at inference time.
    "instance_group_supervise_input_views": True,
}

config_doc["semantic_v6_open_vocab_lb5_smoke"] = (
    "Twenty-step configuration smoke for 15-view scene-level instance "
    "supervision. The frozen latent backbone is warm-started from lb4@4000; "
    "input and target masks share one Hungarian assignment."
)
config_defaults["semantic_v6_open_vocab_lb5_smoke"] = Options(
    **{**_SEMANTIC_V6_LB5_COMMON, "max_eval_iters": 2},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_lb4_train_6000/"
        "checkpoints/model_step_004000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_lb5_smoke_20",
    experiment_name="semantic_v6_open_vocab_lb5_smoke_20",
)

config_doc["semantic_v6_open_vocab_lb5_train"] = (
    "All-view 2D instance-head training from lb4@4000: render group "
    "probabilities into all 8 input and 7 target views, perform one shared "
    "scene-level Hungarian assignment, and supervise all 15 views. The "
    "latent backbone stays frozen and projected 3D supervision stays off."
)
config_defaults["semantic_v6_open_vocab_lb5_train"] = Options(
    **_SEMANTIC_V6_LB5_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_lb4_train_6000/"
        "checkpoints/model_step_004000.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_lb5_train_4000",
    experiment_name="semantic_v6_open_vocab_lb5_train_4000",
)

_SEMANTIC_V6_GC2_COMMON = {
    **_SEMANTIC_V6_LB5_COMMON,
    # GC2: the same scene-adaptive object queries both condition the decoder
    # anchors before Gaussian activation and provide the instance assignment
    # consumed by the existing 15-view scene-level Hungarian objective.
    "instance_group_conditioned_gaussians": True,
    "instance_group_condition_dim": 256,
    "instance_group_condition_heads": 8,
    "instance_group_condition_layers": 2,
    "instance_group_condition_residual_scale": 0.01,
    "instance_group_condition_assignment_temperature": 10.0,
    "instance_group_condition_gaussian_blend": 0.1,
    # GC2 replaces the legacy post-hoc per-Gaussian residual head. Spatial
    # conditioning is built into the unified decoder from proposal centers.
    "instance_group_decoder": False,
    "instance_group_use_anchor_pos": False,
    "instance_group_per_gaussian": False,
    "instance_group_residual_head": False,
    "instance_group_dense_decoder": False,
    "instance_group_count_head": False,
    "instance_group_adaptive_count": False,
    "prompt_unfreeze_tokengs": False,
    # GC2 checkpoints contain only trainable heads. Record the actual frozen
    # latent TokenGS backbone so standalone evaluation restores it explicitly.
    "backbone_resume": (
        "/space0/mawb/tokengs/checkpoints/"
        "dl3dv_latent_6v_ssim.safetensors"
    ),
}

config_doc["semantic_v6_open_vocab_gc2_smoke"] = (
    "Two-step integration smoke for the shared-query group-conditioned "
    "anchor/Gaussian decoder. Warm starts semantic state from lb5@4000, "
    "keeps TokenGS frozen, and verifies 15-view scene-level supervision."
)
config_defaults["semantic_v6_open_vocab_gc2_smoke"] = Options(
    **{**_SEMANTIC_V6_GC2_COMMON, "max_eval_iters": 1},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_lb5_train_4000/"
        "checkpoints/model_step_004000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_gc2_smoke_2_blend",
    experiment_name="semantic_v6_open_vocab_gc2_smoke_2_blend",
)

config_doc["semantic_v6_open_vocab_gc2_train"] = (
    "Shared-query group-conditioned Gaussian training from lb5@4000. The "
    "pretrained latent TokenGS proposal path and activation weights remain "
    "frozen; only the unified GC2 instance decoder and existing semantic "
    "adapters are checkpointed. Uses lb5's 15-view scene-level supervision."
)
config_defaults["semantic_v6_open_vocab_gc2_train"] = Options(
    **_SEMANTIC_V6_GC2_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_lb5_train_4000/"
        "checkpoints/model_step_004000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_gc2_train_6000",
    experiment_name="semantic_v6_open_vocab_gc2_train_6000",
)

_SEMANTIC_V6_GC3_COMMON = {
    **_SEMANTIC_V6_GC2_COMMON,
    # GC3 keeps GC2's scene-level object queries, then computes an
    # independent assignment for every Gaussian using local proposal
    # geometry/appearance.  Only a bounded opacity residual is enabled;
    # xyz/scale/rotation stay on the frozen TokenGS proposal.
    "instance_group_condition_per_gaussian": True,
    "instance_group_condition_per_gaussian_opacity_scale": 0.05,
    "instance_group_condition_gaussian_blend": 0.05,
}

config_doc["semantic_v6_open_vocab_gc3_smoke"] = (
    "Two-step smoke for GC3 per-Gaussian group-conditioned assignments. "
    "Warm starts the shared-query GC2 decoder from gc2@6000 and verifies "
    "15-view scene-level supervision plus per-Gaussian tensor shapes."
)
config_defaults["semantic_v6_open_vocab_gc3_smoke"] = Options(
    **{**_SEMANTIC_V6_GC3_COMMON, "max_eval_iters": 1},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_gc2_train_6000/"
        "checkpoints/model_step_006000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_gc3_smoke_2",
    experiment_name="semantic_v6_open_vocab_gc3_smoke_2",
)

config_doc["semantic_v6_open_vocab_gc3_train"] = (
    "Per-Gaussian group-conditioned Gaussian training from gc2@6000. "
    "Shared group queries remain scene-level; each Gaussian receives local "
    "assignment evidence and a small opacity residual, while TokenGS geometry "
    "and RGB remain frozen. Uses lb5's 15-view scene-level supervision."
)
config_defaults["semantic_v6_open_vocab_gc3_train"] = Options(
    **_SEMANTIC_V6_GC3_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_gc2_train_6000/"
        "checkpoints/model_step_006000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_gc3_train_6000",
    experiment_name="semantic_v6_open_vocab_gc3_train_6000",
)

_SEMANTIC_V6_GC4_COMMON = {
    **_SEMANTIC_V6_GC3_COMMON,
    # GC4 trains the object-conditioned path together with reconstruction.
    # The encoder remains frozen; decoder/activation parameters use a small
    # geometry LR while the group/image head uses the main LR.
    "prompt_unfreeze_tokengs": True,
    "prompt_unfreeze_tokengs_lr": 1e-5,
    "prompt_unfreeze_tokengs_mode": "decoder",
    "instance_group_condition_decoder_checkpoint": True,
    "lambda_rgb": 200.0,
    "instance_group_lambda_warmup_steps": 500,
    "instance_group_condition_gaussian_blend": 0.1,
    "instance_group_condition_per_gaussian_opacity_scale": 0.02,
    # Pixel-aligned encoder evidence is projected to each 64-Gaussian
    # anchor's 3D center before the shared group queries are refined.
    "instance_group_dense_decoder": True,
    "instance_group_condition_image_anchors": True,
    "instance_group_condition_image_feature_dim": 64,
    "instance_group_condition_image_upsample": 2,
    # The latent-bottleneck encoder exposes one [B,V*P,1024] feature map;
    # use that actual shape here (the adapter still supports two-scale
    # backbones through the option).
    "instance_group_condition_image_multiscale": False,
    "instance_group_condition_image_scale": 1.0,
}

config_doc["semantic_v6_open_vocab_gc4_smoke"] = (
    "Two-step smoke for GC4: image-aligned anchor evidence plus joint RGB "
    "and instance training. The encoder is frozen, the TokenGS decoder is "
    "gradient-checkpointed, and the group/image head uses the main LR."
)
config_defaults["semantic_v6_open_vocab_gc4_smoke"] = Options(
    **{**_SEMANTIC_V6_GC4_COMMON, "max_eval_iters": 1},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_gc3_train_6000/"
        "checkpoints/model_step_006000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_gc4_smoke_2",
    experiment_name="semantic_v6_open_vocab_gc4_smoke_2",
)

config_doc["semantic_v6_open_vocab_gc4_train"] = (
    "GC4 joint image-aligned group-conditioned Gaussian training from "
    "gc3@6000. Frozen encoder patch features are projected to 4096 anchor "
    "centers, shared group queries consume token/position/image evidence, "
    "and decoder-only TokenGS fine-tuning is coupled to RGB reconstruction "
    "and the existing 15-view scene-level instance objective. Save every "
    "200 steps and evaluate at 1000/2000/3000/4000/5000/6000."
)
config_defaults["semantic_v6_open_vocab_gc4_train"] = Options(
    **_SEMANTIC_V6_GC4_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_gc3_train_6000/"
        "checkpoints/model_step_006000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_gc4_train_6000",
    experiment_name="semantic_v6_open_vocab_gc4_train_6000",
)

_SEMANTIC_V6_GC_COMMON = {
    **_SEMANTIC_V6_PGR3DF_COMMON,
    # First structural object-conditioned Gaussian experiment.  Group
    # queries attend to decoder anchors before activation, while the existing
    # residual instance head and 3D/2D losses remain available for a clean
    # comparison against pgr3df2.
    "instance_group_conditioned_gaussians": True,
    "instance_group_condition_dim": 256,
    "instance_group_condition_heads": 8,
    "instance_group_condition_layers": 2,
    "instance_group_condition_residual_scale": 1.0,
    "prompt_unfreeze_tokengs": False,
}

config_doc["semantic_v6_open_vocab_gc1_smoke"] = (
    "Twenty-step smoke of group-conditioned Gaussian generation. The new "
    "adapter is zero initialized, so the warm-started pgr3df2 geometry is "
    "unchanged before learning."
)
config_defaults["semantic_v6_open_vocab_gc1_smoke"] = Options(
    **{**_SEMANTIC_V6_GC_COMMON, "max_eval_iters": 2},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_gc1_smoke_20",
    experiment_name="semantic_v6_open_vocab_gc1_smoke_20",
)

config_doc["semantic_v6_open_vocab_gc1_train"] = (
    "Group-conditioned Gaussian generation from pgr3df2@12000. The new "
    "object-context adapter is trained with the existing scene-level 2D "
    "and 3D instance objectives while the TokenGS backbone remains frozen. "
    "Evaluate checkpoints early because the head-only recipes peak quickly."
)
config_defaults["semantic_v6_open_vocab_gc1_train"] = Options(
    **_SEMANTIC_V6_GC_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
        "checkpoints/model_step_008000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_gc1_train_6000",
    experiment_name="semantic_v6_open_vocab_gc1_train_6000",
)

_SEMANTIC_V6_FULL_CE2_WIDE7M_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_WIDE7L_COMMON,
    # Cross-view instance contrastive loss on top of the anchor+decoder
    # stack: ScanNet instance ids are consistent across the frames of a
    # scene, so same-instance pixels are aggregated across all 7 target
    # views into one prototype. Directly teaches cross-view consistent,
    # instance-discriminative token features -- the one remaining
    # training-side lever for feed-forward grouping (per-view contrastive
    # in wide7i did not help; this is the corrected cross-view version).
    "lambda_instance_contrastive": 0.1,
    # Head architecture is unchanged from wide7l (no reinit), so no warmup.
    "instance_group_lambda_warmup_steps": 0,
}

config_doc["semantic_v6_open_vocab_full_ce2_wide7m_smoke"] = (
    "Twenty-step smoke of the wide7m variant (anchor+decoder + cross-view "
    "instance contrastive) warm started from v6-wide7l@8000."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7m_smoke"] = Options(
    **{**_SEMANTIC_V6_FULL_CE2_WIDE7M_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7m_smoke_20",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7m_smoke_20",
)

config_doc["semantic_v6_open_vocab_full_ce2_wide7m_train"] = (
    "Wide7m: wide7l (anchor + decoder, 8000-step checkpoint) plus the "
    "cross-view instance-contrastive loss. Goal: make the token features "
    "cross-view consistent and instance-discriminative so the feed-forward "
    "grouping head generalizes better. Warm starts from v6-wide7l@8000 "
    "and runs 4000 steps."
)
config_defaults["semantic_v6_open_vocab_full_ce2_wide7m_train"] = Options(
    **_SEMANTIC_V6_FULL_CE2_WIDE7M_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/model.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_full_ce2_wide7m_train_4000",
    experiment_name="semantic_v6_open_vocab_full_ce2_wide7m_train_4000",
)

_SEMANTIC_V6_LIFTING_COMMON = {
    **_SEMANTIC_V6_FULL_CE2_WIDE7L_COMMON,
    # C3G-style LSeg lifting semantics (source_projected recipe from the
    # tokengs_c3g experiment): project context-view LSeg features onto the
    # Gaussians, refine with the learned fusion head, render to
    # target/source views, and fit to the LSeg features of those images.
    # Targets ~0.5 C3G8 mIoU while keeping the wide7l instance grouping.
    "lambda_semantic_feature": 1.0,
    "lambda_semantic_cosine": 1.0,
    "lambda_semantic_l1": 0.05,
    "semantic_use_depth_filter": False,
}

config_doc["semantic_v6_lifting_joint_smoke"] = (
    "Twenty-step smoke of the joint instance-grouping + LSeg-lifting "
    "semantics training, warm started from v6-wide7l@8000."
)
config_defaults["semantic_v6_lifting_joint_smoke"] = Options(
    **{**_SEMANTIC_V6_LIFTING_COMMON, "max_eval_iters": 8},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/model.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v6_lifting_joint_smoke_20",
    experiment_name="semantic_v6_lifting_joint_smoke_20",
)

config_doc["semantic_v6_lifting_joint_train"] = (
    "Joint training: wide7l instance grouping (anchor + decoder) plus "
    "C3G-style LSeg lifting semantics (projected features + learned fusion "
    "head, cosine+L1 vs target/source LSeg features). Goal: one model that "
    "carries both the instance structure (~0.20 AP50 feed-forward) and the "
    "lifted semantics (~0.5 C3G8 mIoU). Warm starts from v6-wide7l@8000 "
    "and runs 4000 steps."
)
config_defaults["semantic_v6_lifting_joint_train"] = Options(
    **_SEMANTIC_V6_LIFTING_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/model.safetensors"
    ),
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_lifting_joint_train_4000",
    experiment_name="semantic_v6_lifting_joint_train_4000",
)

_SEMANTIC_V3_FULL_COMMON = {
    **_SEMANTIC_V3_BIGDATA_COMMON,
    "dataset_kwargs": {"small_manifest_path": _PROMPT_FULL_MANIFEST},
}

config_doc["semantic_v3_open_vocab_full_smoke"] = (
    "Twenty-step open-vocabulary semantic_v3 smoke test on the full manifest."
)
config_defaults["semantic_v3_open_vocab_full_smoke"] = Options(
    **{**_SEMANTIC_V3_FULL_COMMON, "max_eval_iters": 8},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v3_open_vocab_full_ratio02_smoke_20",
    experiment_name="semantic_v3_open_vocab_full_ratio02_smoke_20",
)

config_doc["semantic_v3_open_vocab_full_train"] = (
    "Four-thousand-step open-vocabulary semantic_v3 training on the full "
    "1425-scene / 768-per-class manifest with prompt-conditioned prototypes "
    "and joint eight-class balanced-BCE supervision."
)
config_defaults["semantic_v3_open_vocab_full_train"] = Options(
    **_SEMANTIC_V3_FULL_COMMON,
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_full_ratio02_train_4000",
    experiment_name="semantic_v3_open_vocab_full_ratio02_train_4000",
)

config_doc["semantic_v3_open_vocab_full_eval"] = (
    "Evaluate the best full-data open-vocabulary semantic_v3 checkpoint on "
    "the fixed 24-sample proxy split."
)
config_defaults["semantic_v3_open_vocab_full_eval"] = Options(
    **_SEMANTIC_V3_FULL_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v3_open_vocab_full_ratio02_train_4000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v3_open_vocab_full_ratio02_validation_best",
    experiment_name="semantic_v3_open_vocab_full_ratio02_validation_best",
)

config_doc["prompt_scannet_query_diverse_smoke"] = (
    "Twenty-step smoke test with diverse cross-scene image queries."
)
config_defaults["prompt_scannet_query_diverse_smoke"] = Options(
    **_PROMPT_QUERY_DIVERSE_COMMON,
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/prompt_scannet_query_diverse_smoke_20",
    experiment_name="prompt_scannet_query_diverse_smoke_20",
)

config_doc["prompt_scannet_query_diverse_train"] = (
    "Two-thousand-step query-diversity ablation on the fixed 64/8-scene split."
)
config_defaults["prompt_scannet_query_diverse_train"] = Options(
    **_PROMPT_QUERY_DIVERSE_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_query_diverse_train_2000",
    experiment_name="prompt_scannet_query_diverse_train_2000",
)

config_doc["prompt_scannet_query_diverse_eval"] = (
    "Evaluate the query-diversity ablation on its fixed validation samples."
)
config_defaults["prompt_scannet_query_diverse_eval"] = Options(
    **{**_PROMPT_QUERY_DIVERSE_COMMON, "eval_n_media_dumps": 24},
    evaluating=True,
    resume="workspace/prompt_scannet_query_diverse_train_2000/model.safetensors",
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_query_diverse_validation",
    experiment_name="prompt_scannet_query_diverse_validation",
)


_PROMPT_MASKED_CLS_COMMON = {
    **_PROMPT_QUERY_DIVERSE_COMMON,
    "prompt_image_pooling": "masked_input_cls",
    "prompt_save_validation_checkpoints": True,
}

config_doc["prompt_scannet_masked_cls_smoke"] = (
    "Twenty-step mask-neutralized CLIP CLS image-query smoke test."
)
config_defaults["prompt_scannet_masked_cls_smoke"] = Options(
    **_PROMPT_MASKED_CLS_COMMON,
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/prompt_scannet_masked_cls_smoke_20",
    experiment_name="prompt_scannet_masked_cls_smoke_20",
)

config_doc["prompt_scannet_masked_cls_train"] = (
    "Two-thousand-step mask-neutralized CLIP CLS pooling ablation."
)
config_defaults["prompt_scannet_masked_cls_train"] = Options(
    **_PROMPT_MASKED_CLS_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_masked_cls_train_2000",
    experiment_name="prompt_scannet_masked_cls_train_2000",
)

config_doc["prompt_scannet_masked_cls_eval"] = (
    "Evaluate the best mask-neutralized CLIP CLS prompt checkpoint."
)
config_defaults["prompt_scannet_masked_cls_eval"] = Options(
    **{**_PROMPT_MASKED_CLS_COMMON, "eval_n_media_dumps": 24},
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "prompt_scannet_masked_cls_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/prompt_scannet_masked_cls_validation_best",
    experiment_name="prompt_scannet_masked_cls_validation_best",
)


_SEMANTIC_V2_COMMON = {
    **_PROMPT_SMALL_COMMON,
    "model_type": "semantic_tokengs_v2",
    "data_mode": (("scannet_semantic_small", 1),),
    "dataset_kwargs": {"small_manifest_path": _PROMPT_QUERY_DIVERSE_MANIFEST},
    "prompt_mode": "text_only",
    "prompt_image_probability": 0.0,
    "semantic_v2_dim": 256,
    "semantic_v2_temperature_init": 14.285714,
    "semantic_v2_balanced_bce": False,
    "semantic_v2_score_mode": "sigmoid",
    "prompt_save_validation_checkpoints": True,
    "eval_n_media_dumps": 1,
}

config_doc["semantic_v2_scannet_smoke"] = (
    "Twenty-step eight-class Semantic Token Adapter V2 smoke test."
)
config_defaults["semantic_v2_scannet_smoke"] = Options(
    **{**_SEMANTIC_V2_COMMON, "max_eval_iters": 2},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v2_scannet_smoke_20",
    experiment_name="semantic_v2_scannet_smoke_20",
)

config_doc["semantic_v2_scannet_train"] = (
    "Two-thousand-step Semantic Token Adapter V2 structural diagnostic."
)
config_defaults["semantic_v2_scannet_train"] = Options(
    **_SEMANTIC_V2_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v2_scannet_train_2000",
    experiment_name="semantic_v2_scannet_train_2000",
)

config_doc["semantic_v2_scannet_eval"] = (
    "Evaluate the best Semantic Token Adapter V2 checkpoint."
)
config_defaults["semantic_v2_scannet_eval"] = Options(
    **_SEMANTIC_V2_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v2_scannet_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v2_scannet_validation_best",
    experiment_name="semantic_v2_scannet_validation_best",
)


_SEMANTIC_V2_BALANCED_BCE_COMMON = {
    **_SEMANTIC_V2_COMMON,
    "semantic_v2_balanced_bce": True,
}

config_doc["semantic_v2_balanced_bce_scannet_smoke"] = (
    "Twenty-step V2 per-class positive/negative balanced BCE smoke test."
)
config_defaults["semantic_v2_balanced_bce_scannet_smoke"] = Options(
    **{**_SEMANTIC_V2_BALANCED_BCE_COMMON, "max_eval_iters": 2},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v2_balanced_bce_scannet_smoke_20",
    experiment_name="semantic_v2_balanced_bce_scannet_smoke_20",
)

config_doc["semantic_v2_balanced_bce_scannet_train"] = (
    "Two-thousand-step V2 per-class positive/negative balanced BCE experiment."
)
config_defaults["semantic_v2_balanced_bce_scannet_train"] = Options(
    **_SEMANTIC_V2_BALANCED_BCE_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v2_balanced_bce_scannet_train_2000",
    experiment_name="semantic_v2_balanced_bce_scannet_train_2000",
)

config_doc["semantic_v2_balanced_bce_scannet_eval"] = (
    "Evaluate the best V2 per-class positive/negative balanced BCE checkpoint."
)
config_defaults["semantic_v2_balanced_bce_scannet_eval"] = Options(
    **_SEMANTIC_V2_BALANCED_BCE_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v2_balanced_bce_scannet_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v2_balanced_bce_scannet_validation_best",
    experiment_name="semantic_v2_balanced_bce_scannet_validation_best",
)

_SEMANTIC_V2_BIGDATA_COMMON = {
    **_SEMANTIC_V2_BALANCED_BCE_COMMON,
    "dataset_kwargs": {"small_manifest_path": _PROMPT_TARGET_DIVERSE_BIGDATA_MANIFEST},
}

config_doc["semantic_v2_balanced_bce_bigdata_smoke"] = (
    "Twenty-step V2 balanced-BCE smoke test on the 128-scene manifest."
)
config_defaults["semantic_v2_balanced_bce_bigdata_smoke"] = Options(
    **{**_SEMANTIC_V2_BIGDATA_COMMON, "max_eval_iters": 2},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v2_balanced_bce_bigdata_smoke_20",
    experiment_name="semantic_v2_balanced_bce_bigdata_smoke_20",
)

config_doc["semantic_v2_balanced_bce_bigdata_train"] = (
    "Two-thousand-step V2 balanced-BCE training on the 128-scene manifest."
)
config_defaults["semantic_v2_balanced_bce_bigdata_train"] = Options(
    **_SEMANTIC_V2_BIGDATA_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v2_balanced_bce_bigdata_train_2000",
    experiment_name="semantic_v2_balanced_bce_bigdata_train_2000",
)

config_doc["semantic_v2_balanced_bce_bigdata_eval"] = (
    "Evaluate the best V2 big-data balanced-BCE checkpoint on the fixed "
    "24-sample proxy split."
)
config_defaults["semantic_v2_balanced_bce_bigdata_eval"] = Options(
    **_SEMANTIC_V2_BIGDATA_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v2_balanced_bce_bigdata_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v2_balanced_bce_bigdata_validation_best",
    experiment_name="semantic_v2_balanced_bce_bigdata_validation_best",
)


_SEMANTIC_V2_BALANCED_SOFTMAX_COMMON = {
    **_SEMANTIC_V2_BALANCED_BCE_COMMON,
    "semantic_v2_score_mode": "softmax",
}

config_doc["semantic_v2_balanced_softmax_scannet_smoke"] = (
    "Twenty-step V2 balanced-BCE token-class softmax smoke test."
)
config_defaults["semantic_v2_balanced_softmax_scannet_smoke"] = Options(
    **{**_SEMANTIC_V2_BALANCED_SOFTMAX_COMMON, "max_eval_iters": 2},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v2_balanced_softmax_scannet_smoke_20",
    experiment_name="semantic_v2_balanced_softmax_scannet_smoke_20",
)

config_doc["semantic_v2_balanced_softmax_scannet_train"] = (
    "Two-thousand-step V2 balanced-BCE token-class softmax experiment."
)
config_defaults["semantic_v2_balanced_softmax_scannet_train"] = Options(
    **_SEMANTIC_V2_BALANCED_SOFTMAX_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v2_balanced_softmax_scannet_train_2000",
    experiment_name="semantic_v2_balanced_softmax_scannet_train_2000",
)

config_doc["semantic_v2_balanced_softmax_scannet_eval"] = (
    "Evaluate the best V2 balanced-BCE token-class softmax checkpoint."
)
config_defaults["semantic_v2_balanced_softmax_scannet_eval"] = Options(
    **_SEMANTIC_V2_BALANCED_SOFTMAX_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v2_balanced_softmax_scannet_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v2_balanced_softmax_scannet_validation_best",
    experiment_name="semantic_v2_balanced_softmax_scannet_validation_best",
)


_SEMANTIC_V2_LAST_CROSS_ATTN_COMMON = {
    **_SEMANTIC_V2_BALANCED_SOFTMAX_COMMON,
    "semantic_v2_tune_last_cross_attention": True,
}

config_doc["semantic_v2_last_cross_attn_scannet_smoke"] = (
    "Twenty-step semantic-only tuning of the final TokenGS cross-attention."
)
config_defaults["semantic_v2_last_cross_attn_scannet_smoke"] = Options(
    **{**_SEMANTIC_V2_LAST_CROSS_ATTN_COMMON, "max_eval_iters": 2},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/semantic_v2_last_cross_attn_scannet_smoke_20",
    experiment_name="semantic_v2_last_cross_attn_scannet_smoke_20",
)

config_doc["semantic_v2_last_cross_attn_scannet_train"] = (
    "Two-thousand-step semantic-only final TokenGS cross-attention experiment."
)
config_defaults["semantic_v2_last_cross_attn_scannet_train"] = Options(
    **_SEMANTIC_V2_LAST_CROSS_ATTN_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v2_last_cross_attn_scannet_train_2000",
    experiment_name="semantic_v2_last_cross_attn_scannet_train_2000",
)

config_doc["semantic_v2_last_cross_attn_scannet_eval"] = (
    "Evaluate the best semantic-only final TokenGS cross-attention checkpoint."
)
config_defaults["semantic_v2_last_cross_attn_scannet_eval"] = Options(
    **_SEMANTIC_V2_LAST_CROSS_ATTN_COMMON,
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v2_last_cross_attn_scannet_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v2_last_cross_attn_scannet_validation_best",
    experiment_name="semantic_v2_last_cross_attn_scannet_validation_best",
)


_CONDITIONAL_V3_COMMON = {
    **_PROMPT_QUERY_DIVERSE_COMMON,
    "model_type": "conditional_prompt_tokengs",
    "prompt_mode": "manifest",
    "prompt_save_validation_checkpoints": True,
    "eval_n_media_dumps": 6,
}

_CONDITIONAL_V3_FROZEN_DECODER_COMMON = {
    **_CONDITIONAL_V3_COMMON,
    "conditional_v3_tune_last_cross_attention": False,
}

config_doc["conditional_v3_scannet_smoke"] = (
    "Twenty-step OV-DETR-style text/image conditional query smoke test."
)
config_defaults["conditional_v3_scannet_smoke"] = Options(
    **{**_CONDITIONAL_V3_COMMON, "max_eval_iters": 3},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/conditional_v3_scannet_smoke_20",
    experiment_name="conditional_v3_scannet_smoke_20",
)

config_doc["conditional_v3_scannet_train"] = (
    "Two-thousand-step shared text/image conditional query experiment."
)
config_defaults["conditional_v3_scannet_train"] = Options(
    **_CONDITIONAL_V3_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/conditional_v3_scannet_train_2000",
    experiment_name="conditional_v3_scannet_train_2000",
)

config_doc["conditional_v3_scannet_eval"] = (
    "Evaluate the best shared text/image conditional query checkpoint."
)
config_defaults["conditional_v3_scannet_eval"] = Options(
    **{**_CONDITIONAL_V3_COMMON, "eval_n_media_dumps": 24},
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "conditional_v3_scannet_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/conditional_v3_scannet_validation_best",
    experiment_name="conditional_v3_scannet_validation_best",
)

config_doc["conditional_v3_frozen_decoder_scannet_smoke"] = (
    "Twenty-step conditional-query smoke test with the copied decoder frozen."
)
config_defaults["conditional_v3_frozen_decoder_scannet_smoke"] = Options(
    **{**_CONDITIONAL_V3_FROZEN_DECODER_COMMON, "max_eval_iters": 3},
    num_epochs=1,
    max_iters_per_epoch=20,
    print_freq=1,
    log_image_freq=10,
    mixed_precision="no",
    workspace="workspace/conditional_v3_frozen_decoder_scannet_smoke_20",
    experiment_name="conditional_v3_frozen_decoder_scannet_smoke_20",
)

config_doc["conditional_v3_frozen_decoder_scannet_train"] = (
    "Two-thousand-step conditional-query experiment with decoder frozen."
)
config_defaults["conditional_v3_frozen_decoder_scannet_train"] = Options(
    **_CONDITIONAL_V3_FROZEN_DECODER_COMMON,
    num_epochs=10,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/conditional_v3_frozen_decoder_scannet_train_2000",
    experiment_name="conditional_v3_frozen_decoder_scannet_train_2000",
)

config_doc["conditional_v3_frozen_decoder_scannet_eval"] = (
    "Evaluate the best frozen-decoder conditional-query checkpoint."
)
config_defaults["conditional_v3_frozen_decoder_scannet_eval"] = Options(
    **{**_CONDITIONAL_V3_FROZEN_DECODER_COMMON, "eval_n_media_dumps": 24},
    evaluating=True,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "conditional_v3_frozen_decoder_scannet_train_2000/model_best.safetensors"
    ),
    num_epochs=0,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/conditional_v3_frozen_decoder_scannet_validation_best",
    experiment_name="conditional_v3_frozen_decoder_scannet_validation_best",
)

config_doc["finetune_scannet_latent_da_smoke"] = (
    "Two-step smoke of the ScanNet pure-reconstruction domain adaptation "
    "for the DL3DV latent-bottleneck backbone: native TokenGS training, "
    "LSM-style 8+7 ScanNet windows, L1+SSIM only, full finetune LR 4e-5."
)
config_defaults["finetune_scannet_latent_da_smoke"] = Options(
    data_mode=(("scannet_lsm_style_train", 1),),
    dataset_kwargs={"windows_per_scene": 4},
    num_input_views=8,
    num_views=15,
    img_size=(256, 256),
    num_gs_tokens=4096,
    lr=4e-5,
    num_epochs=1,
    max_iters_per_epoch=2,
    pct_start_steps=400,
    # Full-geometry finetune of 262K Gaussians x 15 views exceeds 47GB;
    # chunk the render/backward per view (official mean-of-gradients mode).
    mean_of_grads="per-view",
    mean_of_grads_view_chunk_size=1,
    resume=(
        "/space0/mawb/tokengs/checkpoints/"
        "dl3dv_latent_6v_ssim.safetensors"
    ),
    workspace="workspace/tokengs_scannet_latent_da_smoke_2",
    experiment_name="tokengs_scannet_latent_da_smoke_2",
    **_LATENT_D12_ARCH,
    **_SSIM_LOSS,
)

config_doc["finetune_scannet_latent_da"] = (
    "Pure-reconstruction domain adaptation of the DL3DV latent-bottleneck "
    "backbone (4096 GS tokens, 12-layer multiscale encoder, 262K Gaussians) "
    "on ScanNet. LSM-style 8+7 interleaved windows over the 1473-scene "
    "train split, only RGB reconstruction loss (L1+SSIM, the backbone's own "
    "loss family), full finetune at LR 4e-5 for 4000 steps. This pulls the "
    "token feature distribution from the DL3DV outdoor domain toward "
    "ScanNet before Step 2 instance training; the saved checkpoint becomes "
    "the domain-aligned backbone."
)
config_defaults["finetune_scannet_latent_da"] = Options(
    data_mode=(("scannet_lsm_style_train", 1),),
    dataset_kwargs={"windows_per_scene": 4},
    num_input_views=8,
    num_views=15,
    img_size=(256, 256),
    num_gs_tokens=4096,
    lr=4e-5,
    num_epochs=20,
    max_iters_per_epoch=200,
    pct_start_steps=400,
    # Chunked per-view render/backward keeps the full-geometry finetune
    # within a single 47GB GPU (official mean-of-gradients mode).
    mean_of_grads="per-view",
    mean_of_grads_view_chunk_size=1,
    resume=(
        "/space0/mawb/tokengs/checkpoints/"
        "dl3dv_latent_6v_ssim.safetensors"
    ),
    workspace="workspace/tokengs_scannet_latent_da_4000",
    experiment_name="tokengs_scannet_latent_da_4000",
    **_LATENT_D12_ARCH,
    **_SSIM_LOSS,
)

_SEMANTIC_V6_DA_COMMON = {
    **_SEMANTIC_V6_LB_COMMON,
    # Step-1 domain adaptation for the DL3DV latent backbone on ScanNet,
    # using the gc4-validated recipe (decoder-only unfreeze + RGB keeps
    # PSNR ~26 and fits one 47GB GPU): the frozen encoder/latent blocks run
    # without autograd, only decoder_blocks + activation_head + gs_tokens
    # are trainable at LR 1e-5, and ONLY the RGB loss is active (all
    # semantic/prompt/instance losses off, instance labels not loaded).
    "data_mode": (("scannet_lsm_style_train", 1),),
    "dataset_kwargs": {"windows_per_scene": 4},
    "prompt_unfreeze_tokengs": True,
    "prompt_unfreeze_tokengs_lr": 1e-5,
    "prompt_unfreeze_tokengs_mode": "decoder",
    "lambda_ce_cosine": 0.0,
    "lambda_feat": 0.0,
    "prompt_lambda_bce": 0.0,
    "prompt_lambda_dice": 0.0,
    "use_instance_labels": False,
}

config_doc["finetune_scannet_latent_da_v6_smoke"] = (
    "Two-step smoke of Step-1 domain adaptation via the semantic-v6 wrapper: "
    "latent backbone, decoder-only unfreeze LR 1e-5, RGB-only loss, "
    "ScanNet LSM-style 8+7 windows."
)
config_defaults["finetune_scannet_latent_da_v6_smoke"] = Options(
    **{**_SEMANTIC_V6_DA_COMMON, "max_eval_iters": 2},
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_scannet_latent_da_smoke_2",
    experiment_name="semantic_v6_scannet_latent_da_smoke_2",
)

config_doc["finetune_scannet_latent_da_v6"] = (
    "Step-1 domain adaptation (4000 steps): DL3DV latent-bottleneck "
    "backbone finetuned on ScanNet LSM-style 8+7 windows with only the RGB "
    "loss (lambda_rgb 200), decoder-only unfreeze at LR 1e-5, encoder and "
    "latent blocks frozen (no autograd through them). The saved checkpoint "
    "contains the adapted decoder/activation/gs_tokens and becomes the "
    "Step-2 backbone for instance training."
)
config_defaults["finetune_scannet_latent_da_v6"] = Options(
    **_SEMANTIC_V6_DA_COMMON,
    num_epochs=20,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_scannet_latent_da_4000",
    experiment_name="semantic_v6_scannet_latent_da_4000",
)

_SEMANTIC_V6_GC_DA_COMMON = {
    **_SEMANTIC_V6_GC3_COMMON,
    # Step-2 instance training on the Step-1 domain-adapted backbone.
    # GC3 structure (shared object queries conditioning the Gaussian
    # generation + per-Gaussian assignment + bounded opacity residual),
    # 15-view scene-level supervision, instance-only losses (the gc3 recipe
    # that reached 0.173 AP50 on the raw latent backbone). The head is
    # initialized from scratch (no resume) because the domain-adapted token
    # features have shifted; the backbone comes from the DA checkpoint.
    "data_mode": (("scannet_lsm_style_train", 1),),
    "dataset_kwargs": {"windows_per_scene": 4},
    "backbone_resume": (
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_scannet_latent_da_4000/"
        "checkpoints/model_step_004000.safetensors"
    ),
}

config_doc["semantic_v6_open_vocab_gc_da_smoke"] = (
    "Two-step smoke of Step-2: GC3 structure trained from scratch on the "
    "domain-adapted latent backbone (Step-1 output), ScanNet LSM-style "
    "8+7 windows, 15-view scene-level instance supervision."
)
config_defaults["semantic_v6_open_vocab_gc_da_smoke"] = Options(
    **{**_SEMANTIC_V6_GC_DA_COMMON, "max_eval_iters": 1},
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_gc_da_smoke_2",
    experiment_name="semantic_v6_open_vocab_gc_da_smoke_2",
)

config_doc["semantic_v6_open_vocab_gc_da_train"] = (
    "Step-2 instance training (6000 steps) on the Step-1 domain-adapted "
    "latent backbone: GC3 structure (group-conditioned Gaussians + "
    "per-Gaussian assignment + opacity residual), 15-view scene-level "
    "Hungarian supervision, instance-only losses, head initialized from "
    "scratch. train.py's load_fresh_backbone_resume loads the Step-1 "
    "decoder/GS-token subset at fresh start (no --resume), so training "
    "really runs on the ScanNet-domain-aligned backbone rather than the raw "
    "DL3DV one. This is the gc3 recipe (0.173 AP50 on the raw latent "
    "backbone) applied to the domain-aligned backbone."
)
config_defaults["semantic_v6_open_vocab_gc_da_train"] = Options(
    **_SEMANTIC_V6_GC_DA_COMMON,
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_gc_da_train_6000",
    experiment_name="semantic_v6_open_vocab_gc_da_train_6000",
)

config_doc["semantic_v6_open_vocab_gc_da_warm_smoke"] = (
    "Two-step smoke of gc_da with a warm-started head: resume the GC3 head "
    "from gc3@6000 (trained on the raw latent backbone) while switching the "
    "frozen backbone to the Step-1 domain-adapted checkpoint. Isolates the "
    "head-initialization confound of the gc_da comparison."
)
config_defaults["semantic_v6_open_vocab_gc_da_warm_smoke"] = Options(
    **{**_SEMANTIC_V6_GC_DA_COMMON, "max_eval_iters": 1},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_gc3_train_6000/"
        "checkpoints/model_step_006000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_gc_da_warm_smoke",
    experiment_name="semantic_v6_open_vocab_gc_da_warm_smoke",
)

config_doc["semantic_v6_open_vocab_gc_da_warm_train"] = (
    "Step-2 instance training (6000 steps) on the Step-1 domain-adapted "
    "latent backbone with the GC3 head warm-started from gc3@6000 (which "
    "was trained on the raw latent backbone). Same GC3 recipe and "
    "instance-only losses as gc_da, differing only in head initialization: "
    "this isolates whether the gc_da drop comes from the DA features or "
    "from the scratch head. The frozen decoder/GS tokens come from the DA "
    "checkpoint via the resume path's backbone_resume fallback."
)
config_defaults["semantic_v6_open_vocab_gc_da_warm_train"] = Options(
    **_SEMANTIC_V6_GC_DA_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_gc3_train_6000/"
        "checkpoints/model_step_006000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_gc_da_warm_train_6000",
    experiment_name="semantic_v6_open_vocab_gc_da_warm_train_6000",
)

config_doc["semantic_v6_open_vocab_da_pgr_smoke"] = (
    "Two-step smoke of Step-2 with the best head: pgr3df2 recipe "
    "(PerGaussianResidualHead + joint losses + warm-started head) on the "
    "Step-1 domain-adapted latent backbone, everything else unchanged."
)
config_defaults["semantic_v6_open_vocab_da_pgr_smoke"] = Options(
    **{**_SEMANTIC_V6_LB_COMMON, "max_eval_iters": 2},
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_scannet_latent_da_4000/"
        "checkpoints/model_step_004000.safetensors"
    ),
    num_epochs=1,
    max_iters_per_epoch=2,
    print_freq=1,
    log_image_freq=2,
    mixed_precision="no",
    workspace="workspace/semantic_v6_open_vocab_da_pgr_smoke_2",
    experiment_name="semantic_v6_open_vocab_da_pgr_smoke_2",
)

config_doc["semantic_v6_absolute_units_true_shared_query_memory_refine_probe_ddp8"] = (
    "Refiner-only probe from MBM Both@1420.  The existing q_abs, absolute "
    "GS decoder, decoder tail and TSH head are frozen; only the two-round "
    "assignment-conditioned GroupQueryMemoryRefiner is trainable."
)
config_defaults["semantic_v6_absolute_units_true_shared_query_memory_refine_probe_ddp8"] = config_defaults[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
].evolve(
    tsh_query_memory_refine=True,
    tsh_query_memory_refine_rounds=2,
    tsh_query_memory_refine_gate_steps=50,
    tsh_query_memory_refine_probe=True,
    prompt_unfreeze_tokengs=False,
    tsh_mbm_decoder_tail_lr=3.0e-6,
    tsh_mbm_mode="off",
    tsh_mbm_u2r_weight=0.0,
    tsh_instance_warmup_steps=0,
    tsh_instance_ramp_end_steps=0,
    tsh_unit_gradient_multiplier_max=0.0,
    tsh_per_gs_refine=False,
    instance_group_scene_level_matching=False,
    num_epochs=1,
    max_iters_per_epoch=125,
    abs_ckpt_every=125,
    abs_ckpt_steps_extra=(),
    workspace="workspace/semantic_v6_absolute_units_true_shared_query_memory_refine_probe_ddp8",
    experiment_name="semantic_v6_absolute_units_true_shared_query_memory_refine_probe_ddp8",
)

config_doc["semantic_v6_absolute_units_true_shared_query_memory_refine_head_joint_probe_ddp8"] = (
    "Joint TSH-head plus GroupQueryMemoryRefiner probe from MBM Both@1420; "
    "the reconstruction path and decoder remain frozen."
)
config_defaults["semantic_v6_absolute_units_true_shared_query_memory_refine_head_joint_probe_ddp8"] = config_defaults[
    "semantic_v6_absolute_units_true_shared_query_memory_refine_probe_ddp8"
].evolve(
    tsh_query_memory_refine_head_joint_probe=True,
    tsh_instance_lr=3.0e-5,
    tsh_query_memory_refine_lr=1.0e-4,
    workspace="workspace/semantic_v6_absolute_units_true_shared_query_memory_refine_head_joint_probe_ddp8",
    experiment_name="semantic_v6_absolute_units_true_shared_query_memory_refine_head_joint_probe_ddp8",
)

config_doc["semantic_v6_absolute_units_true_shared_ga_idu0_ddp8"] = (
    "GA-IDU-0 identity framework from Both@1420.  No GA-IDU parameters are "
    "trained; geometry, consistency, refiner and quality paths are absent."
)
config_defaults["semantic_v6_absolute_units_true_shared_ga_idu0_ddp8"] = config_defaults[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
].evolve(
    ga_idu_mode="0",
    ga_idu_input_dim=1024,
    prompt_unfreeze_tokengs=False,
    tsh_mbm_decoder_tail_lr=0.0,
    tsh_abs_lr=0.0,
    tsh_instance_lr=0.0,
    tsh_query_memory_refine=False,
    tsh_per_gs_refine=False,
    instance_group_scene_level_matching=False,
    tsh_mbm_mode="off",
    tsh_mbm_u2r_weight=0.0,
    tsh_unit_gradient_multiplier_max=0.0,
    num_epochs=1,
    max_iters_per_epoch=1,
    abs_ckpt_every=1,
    workspace="workspace/semantic_v6_absolute_units_true_shared_ga_idu0_ddp8",
    experiment_name="semantic_v6_absolute_units_true_shared_ga_idu0_ddp8",
)

config_doc["semantic_v6_absolute_units_true_shared_ga_idu1_ddp8"] = (
    "GA-IDU-1 probe from Both@1420: MemoryResampler, unit-aligned "
    "InstanceDecoder, BaseTSH-anchored group residual and channel-aligned "
    "assignment residual only.  Geometry and RGB remain frozen."
)
config_defaults["semantic_v6_absolute_units_true_shared_ga_idu1_ddp8"] = config_defaults[
    "semantic_v6_absolute_units_true_shared_ga_idu0_ddp8"
].evolve(
    ga_idu_mode="1",
    ga_idu_input_dim=1024,
    prompt_unfreeze_tokengs=False,
    tsh_mbm_decoder_tail_lr=0.0,
    tsh_abs_lr=0.0,
    tsh_instance_lr=0.0,
    ga_idu_instance_lr=1.0e-4,
    ga_idu_gate_steps=5,
    max_iters_per_epoch=3,
    workspace="workspace/semantic_v6_absolute_units_true_shared_ga_idu1_ddp8",
    experiment_name="semantic_v6_absolute_units_true_shared_ga_idu1_ddp8",
)

config_doc["semantic_v6_absolute_units_true_shared_ta_riu_v1_ddp8"] = (
    "TA-RIU v1 fixed-batch probe from Both@1420: a token-aligned shared "
    "memory mixer drives the complete TSH instance readout and bounded "
    "geometry/appearance reconstruction residuals.  The absolute student "
    "and all backbone/semantic modules remain frozen."
)
config_defaults["semantic_v6_absolute_units_true_shared_ta_riu_v1_ddp8"] = config_defaults[
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
].evolve(
    ta_riu_enabled=True,
    ta_riu_dim=256,
    ta_riu_memory_latents=256,
    ta_riu_heads=8,
    ta_riu_shared_lr=1.0e-5,
    ta_riu_instance_lr=3.0e-5,
    ta_riu_geometry_lr=1.0e-5,
    ta_riu_appearance_lr=1.0e-5,
    ta_riu_gate_steps=25,
    ta_riu_xyz_scale=0.02,
    ta_riu_log_scale_scale=0.05,
    ta_riu_rot_scale=0.05,
    ta_riu_opacity_scale=0.05,
    ta_riu_color_scale=0.05,
    ga_idu_mode="off",
    tsh_query_memory_refine=False,
    tsh_per_gs_refine=False,
    instance_group_scene_level_matching=False,
    tsh_mbm_mode="off",
    tsh_mbm_u2r_weight=0.0,
    tsh_mbm_decoder_tail_lr=0.0,
    tsh_abs_lr=0.0,
    tsh_instance_lr=3.0e-5,
    tsh_unit_gradient_multiplier_max=1.0,
    abs_bootstrap_steps=0,
    abs_teacher_decay_steps=0,
    abs_teacher_gs_weight=0.0,
    abs_teacher_rgb_weight=0.0,
    prompt_unfreeze_tokengs=False,
    num_epochs=1,
    max_iters_per_epoch=200,
    abs_ckpt_every=200,
    abs_ckpt_steps_extra=(1, 5, 25, 50, 100),
    abs_ckpt_full_state=False,
    workspace="workspace/semantic_v6_absolute_units_true_shared_ta_riu_v1_ddp8",
    experiment_name="semantic_v6_absolute_units_true_shared_ta_riu_v1_ddp8",
)

config_doc["semantic_v6_absolute_units_true_shared_ta_riu_v1_short250_ddp8"] = (
    "TA-RIU v1 short multi-scene generalization probe from Both@1420.  This "
    "keeps the fixed-batch-validated token-aligned coupling and frozen "
    "reconstruction boundary, but uses the formal 8-context/7-target "
    "multi-scene dataloader for exactly 250 DDP8 optimizer steps."
)
config_defaults[
    "semantic_v6_absolute_units_true_shared_ta_riu_v1_short250_ddp8"
] = config_defaults[
    "semantic_v6_absolute_units_true_shared_ta_riu_v1_ddp8"
].evolve(
    # The parent is the fixed-batch audit recipe; the short probe must use
    # the real formal multi-scene sampler instead of its one-sample override.
    prompt_overfit_single_batch=False,
    abs_instance_warmup_steps=0,
    num_epochs=1,
    max_iters_per_epoch=250,
    print_freq=10,
    log_image_freq=250,
    abs_ckpt_every=25,
    abs_ckpt_steps_extra=(25, 50, 100, 150, 200, 250),
    abs_ckpt_full_state=True,
    workspace=(
        "workspace/semantic_v6_absolute_units_true_shared_ta_riu_v1_"
        "short250_ddp8"
    ),
    experiment_name=(
        "semantic_v6_absolute_units_true_shared_ta_riu_v1_short250_ddp8"
    ),
)

config_doc[
    "semantic_v6_absolute_units_true_shared_ta_riu_v1_short250_retry_v2_ddp8"
] = (
    "Retry of the TA-RIU v1 short multi-scene probe after correcting the "
    "inherited absolute-student instance warm-up; this workspace is kept "
    "separate from the aborted preflight workspace."
)
config_defaults[
    "semantic_v6_absolute_units_true_shared_ta_riu_v1_short250_retry_v2_ddp8"
] = config_defaults[
    "semantic_v6_absolute_units_true_shared_ta_riu_v1_short250_ddp8"
].evolve(
    abs_instance_warmup_steps=0,
    workspace=(
        "workspace/semantic_v6_absolute_units_true_shared_ta_riu_v1_"
        "short250_retry_v2_ddp8"
    ),
    experiment_name=(
        "semantic_v6_absolute_units_true_shared_ta_riu_v1_short250_retry_v2_ddp8"
    ),
)

config_doc["semantic_v6_open_vocab_da_pgr_train"] = (
    "Step-2 instance training (6000 steps): the best head recipe "
    "(pgr3df2: PerGaussianResidualHead, joint losses, warm-started from "
    "pgr3df2@12000) with ONLY the backbone replaced by the Step-1 "
    "domain-adapted latent checkpoint. Everything else (data, supervision, "
    "loss balance, head structure) is identical to the 0.231-AP50 run, so "
    "any change is attributable to the backbone."
)
config_defaults["semantic_v6_open_vocab_da_pgr_train"] = Options(
    **_SEMANTIC_V6_LB_COMMON,
    resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_open_vocab_pgr3df2_train_12000/"
        "checkpoints/model_step_012000.safetensors"
    ),
    backbone_resume=(
        "/space0/mawb/tokengs/workspace/"
        "semantic_v6_scannet_latent_da_4000/"
        "checkpoints/model_step_004000.safetensors"
    ),
    num_epochs=30,
    max_iters_per_epoch=200,
    print_freq=10,
    log_image_freq=100,
    mixed_precision="bf16",
    workspace="workspace/semantic_v6_open_vocab_da_pgr_train_6000",
    experiment_name="semantic_v6_open_vocab_da_pgr_train_6000",
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
