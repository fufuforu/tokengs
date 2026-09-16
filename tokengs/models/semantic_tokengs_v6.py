"""v6: v4 open-vocabulary semantics plus native instance-group structure."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import torch

from tokengs.models.instance_group_head import (
    DenseInstanceResidualHead,
    GroupConditionedGaussianAdapter,
    GroupConditionedGaussianGenerator,
    GroupTokenPerGaussianHead,
    IndependentInstanceBranch,
    InstanceCountHead,
    InstanceGroupDecoder,
    InstanceGroupHead,
    TokenLocalUnitGrouping,
    PerGaussianInstanceGroupHead,
    PerGaussianResidualHead,
)
from tokengs.models.absolute_unit_decoder import AbsoluteUnitDecoder
from tokengs.models.semantic_tokengs_v4 import SemanticTokenGSv4
from tokengs.models.ta_riu import (
    AppearanceResidualHead,
    GeometryResidualHead,
    SharedUnitMixer,
)
from tokengs.models.ta_riu_v2 import (
    FrozenDINOv2Extractor,
    GeometryAlignedDINOUnitEncoder,
    build_unit_soft_instance_targets,
    soft_unit_info_nce,
)
from tokengs.models.ta_riu_v3 import TARIUV3DualStream
from tokengs.models.token_eru import TokenGSEarlyDualStreamDecoder
from tokengs.models.token_eru.dino_metric import (
    DINOUnitEvidence,
    DINOUnitFusion,
    HistoricalDINOUnitEncoder,
    MetricEmbeddingHead,
)
from tokengs.models.token_eru.historical_unit_infonce import (
    build_historical_soft_unit_targets,
    historical_soft_unit_infonce,
)
from tokengs.models.token_eru.metric_clustering import historical_metric_cluster
from tokengs.models.token_eru.unit_3d_anchor import Unit3DAnchor
from tokengs.models.token_eru.query_metric_coupling import QueryMetricCoupling
from tokengs.models.token_eru.early_query_codecoder import EarlyObjectQueryAdapter


class SemanticTokenGSv6(SemanticTokenGSv4):
    """Adds an anchor-to-instance group assignment head to v4.

    The 1024 decoder tokens are soft-assigned to ``num_groups`` instance
    groups (softmax competition), rendered as per-view instance probability
    maps, and supervised by ScanNet 2D instance masks through Hungarian
    matching (BCE + Dice + void + unmatched). The v4 per-Gaussian open
    vocabulary semantic field is kept unchanged, so the C3G8 semantic
    protocol, feature lifting, and prompt retrieval all continue to work;
    group-level semantic embeddings additionally support instance retrieval.
    """

    def __init__(self, opt):
        # v4's __init__ calls prompt_trainable_groups() before v6 builds the
        # count/conditioning heads, so the attributes must exist from the
        # start.
        self.instance_count_head = None
        self.instance_group_conditioner = None
        self.instance_branch = None
        self.tsh_instance_head = None
        self.ta_riu_shared_mixer = None
        self.ta_riu_geometry_head = None
        self.ta_riu_appearance_head = None
        self.ta_riu_v2_unit_encoder = None
        self.ta_riu_v3_dual_stream = None
        self.token_eru_decoder = None
        self.token_eru_unit_formation = None
        self.token_eru_dino_encoder = None
        self.token_eru_dino_fusion = None
        self.token_eru_metric_head = None
        self.token_eru_3d_anchor = None
        self.token_eru_query_metric_coupling = None
        self._token_eru_dino_metric_step = 500
        self._token_eru_query_metric_local_step = 0
        self._token_eru_early_query_local_step = 0
        self._token_eru_early_query_gate = 0.0
        self._token_eru_last_early_query_state = None
        super().__init__(opt)
        num_groups = int(getattr(self.opt, "instance_group_num_groups", 64))
        head_input_dim = int(self.opt.token_dim)
        if getattr(self.opt, "instance_group_use_anchor_pos", False):
            head_input_dim += 64
        if getattr(self.opt, "instance_branch_token_units", False):
            # Experiment D: Token -> local spatial units -> Group. Frozen
            # TokenGS geometry is only rendered through; unit formation +
            # group tokens + assignment train.
            self.instance_group_head = None
            self.instance_group_conditioner = None
            self.instance_count_head = None
            self.instance_branch = TokenLocalUnitGrouping(
                opt,
                token_dim=int(self.opt.token_dim),
                units_per_token=int(
                    getattr(self.opt, "instance_branch_units_per_token", 8)
                ),
                gs_feat_dim=int(
                    getattr(self.opt, "instance_branch_unit_feat_dim", 128)
                ),
                unit_layers=int(
                    getattr(self.opt, "instance_branch_unit_layers", 2)
                ),
                num_groups=int(
                    getattr(self.opt, "instance_branch_num_groups", 100)
                ),
                group_anchor_dim=int(
                    getattr(self.opt, "instance_branch_anchor_dim", 256)
                ),
                num_heads=int(
                    getattr(self.opt, "instance_branch_num_heads", 8)
                ),
                group_layers=int(
                    getattr(self.opt, "instance_branch_num_layers", 2)
                ),
                unit_temp=float(
                    getattr(self.opt, "instance_branch_unit_temp", 5.0)
                ),
                assignment_temperature=float(
                    getattr(
                        self.opt,
                        "instance_group_condition_assignment_temperature",
                        10.0,
                    )
                ),
                unit_entropy_weight=float(
                    getattr(self.opt, "instance_branch_unit_entropy", 0.05)
                ),
                unit_compactness_weight=float(
                    getattr(
                        self.opt,
                        "instance_branch_unit_compactness",
                        0.1,
                    )
                ),
                unit_purity=bool(
                    getattr(self.opt, "instance_branch_unit_purity", False)
                ),
                unit_purity_weight=float(
                    getattr(
                        self.opt,
                        "instance_branch_unit_purity_weight",
                        0.1,
                    )
                ),
                gs_refine=bool(
                    getattr(self.opt, "instance_branch_gs_refine", False)
                ),
                gs_refine_scale=float(
                    getattr(
                        self.opt, "instance_branch_gs_refine_scale", 0.1
                    )
                ),
                gs_refine_dim=int(
                    getattr(self.opt, "instance_branch_gs_refine_dim", 64)
                ),
                scene_prototypes=bool(
                    getattr(
                        self.opt, "instance_branch_scene_prototypes", False
                    )
                ),
                num_slots=int(
                    getattr(self.opt, "instance_branch_num_slots", 100)
                ),
                slot_dim=int(
                    getattr(self.opt, "instance_branch_slot_dim", 128)
                ),
                slot_iterations=int(
                    getattr(self.opt, "instance_branch_slot_iterations", 3)
                ),
                slot_temp=float(
                    getattr(self.opt, "instance_branch_slot_temp", 5.0)
                ),
                unit_embedding=bool(
                    getattr(self.opt, "instance_branch_unit_embedding", False)
                ),
                unit_encoder=bool(
                    getattr(self.opt, "instance_branch_unit_encoder", False)
                ),
                embed_dim=int(
                    getattr(self.opt, "instance_branch_embed_dim", 128)
                ),
                embed_temp=float(
                    getattr(self.opt, "instance_branch_embed_temp", 0.1)
                ),
                embed_loss=float(
                    getattr(self.opt, "instance_branch_embed_loss", 1.0)
                ),
                embed_loss_mode=str(
                    getattr(
                        self.opt,
                        "instance_branch_embed_loss_mode",
                        "info_nce",
                    )
                ),
                embed_sample=int(
                    getattr(
                        self.opt,
                        "instance_branch_embed_sample",
                        0,
                    )
                ),
                embed_proto_temp=float(
                    getattr(
                        self.opt,
                        "instance_branch_embed_proto_temp",
                        0.1,
                    )
                ),
                embed_margin=float(
                    getattr(self.opt, "instance_branch_embed_margin", 0.2)
                ),
                embed_margin_weight=float(
                    getattr(
                        self.opt,
                        "instance_branch_embed_margin_weight",
                        1.0,
                    )
                ),
                embed_center_push_margin=float(
                    getattr(
                        self.opt,
                        "instance_branch_embed_center_push_margin",
                        0.2,
                    )
                ),
                embed_center_push_weight=float(
                    getattr(
                        self.opt,
                        "instance_branch_embed_center_push_weight",
                        0.1,
                    )
                ),
                scene_assignment=bool(
                    getattr(
                        self.opt, "instance_branch_scene_assignment", False
                    )
                ),
                scene_slots=int(
                    getattr(self.opt, "instance_branch_scene_slots", 100)
                ),
                scene_slot_iters=int(
                    getattr(self.opt, "instance_branch_scene_slot_iters", 3)
                ),
                scene_slot_temp=float(
                    getattr(self.opt, "instance_branch_scene_slot_temp", 5.0)
                ),
                scene_slot_pos_weight=float(
                    getattr(
                        self.opt,
                        "instance_branch_scene_slot_pos_weight",
                        1.0,
                    )
                ),
                scene_unit_loss_weight=float(
                    getattr(self.opt, "lambda_scene_assignment_unit", 1.0)
                ),
                scene_slot_entropy=float(
                    getattr(
                        self.opt,
                        "instance_branch_scene_slot_entropy",
                        0.05,
                    )
                ),
                scene_slot_void=float(
                    getattr(self.opt, "instance_branch_scene_slot_void", 0.1)
                ),
                scene_slot_unmatched=float(
                    getattr(
                        self.opt,
                        "instance_branch_scene_slot_unmatched",
                        0.1,
                    )
                ),
                scene_slot_min_mass=float(
                    getattr(
                        self.opt,
                        "instance_branch_scene_slot_min_mass",
                        1.0,
                    )
                ),
                dpg=bool(getattr(self.opt, "instance_branch_dpg", False)),
                dpg_proto_dim=int(
                    getattr(self.opt, "instance_branch_dpg_proto_dim", 256)
                ),
                dpg_heads=int(
                    getattr(self.opt, "instance_branch_dpg_heads", 4)
                ),
                dpg_layers=int(
                    getattr(self.opt, "instance_branch_dpg_layers", 2)
                ),
                dpg_proto_weight=float(
                    getattr(self.opt, "instance_branch_dpg_proto_weight", 1.0)
                ),
                render_space=bool(
                    getattr(self.opt, "instance_branch_render_space", False)
                ),
                render_space_pull=float(
                    getattr(self.opt, "instance_branch_render_space_pull", 1.0)
                ),
                render_space_push=float(
                    getattr(self.opt, "instance_branch_render_space_push", 0.5)
                ),
                render_space_cross=float(
                    getattr(self.opt, "instance_branch_render_space_cross", 1.0)
                ),
                render_space_margin_push=float(
                    getattr(
                        self.opt,
                        "instance_branch_render_space_margin_push",
                        0.5,
                    )
                ),
                render_space_margin_cross=float(
                    getattr(
                        self.opt,
                        "instance_branch_render_space_margin_cross",
                        0.2,
                    )
                ),
                render_space_info_nce=float(
                    getattr(
                        self.opt,
                        "instance_branch_render_space_info_nce",
                        0.0,
                    )
                ),
                render_space_info_temp=float(
                    getattr(
                        self.opt,
                        "instance_branch_render_space_info_temp",
                        0.2,
                    )
                ),
                render_space_info_samples=int(
                    getattr(
                        self.opt,
                        "instance_branch_render_space_info_samples",
                        32,
                    )
                ),
                direct_gs=bool(
                    getattr(self.opt, "instance_branch_direct_gs", False)
                ),
                direct_gs_embed_dim=int(
                    getattr(self.opt, "instance_branch_direct_gs_embed_dim", 8)
                ),
                direct_gs_hidden=int(
                    getattr(self.opt, "instance_branch_direct_gs_hidden", 128)
                ),
                direct_gs_dino=bool(
                    getattr(self.opt, "instance_branch_direct_gs_dino", True)
                ),
                direct_gs_pull=float(
                    getattr(self.opt, "instance_branch_direct_gs_pull", 1.0)
                ),
                direct_gs_push=float(
                    getattr(self.opt, "instance_branch_direct_gs_push", 2.0)
                ),
                direct_gs_cross=float(
                    getattr(self.opt, "instance_branch_direct_gs_cross", 1.0)
                ),
                direct_gs_margin_push=float(
                    getattr(
                        self.opt,
                        "instance_branch_direct_gs_margin_push",
                        1.0,
                    )
                ),
                direct_gs_margin_cross=float(
                    getattr(
                        self.opt,
                        "instance_branch_direct_gs_margin_cross",
                        0.2,
                    )
                ),
                direct_gs_info_nce=float(
                    getattr(
                        self.opt, "instance_branch_direct_gs_info_nce", 1.0
                    )
                ),
                direct_gs_info_temp=float(
                    getattr(
                        self.opt, "instance_branch_direct_gs_info_temp", 0.1
                    )
                ),
                direct_gs_info_samples=int(
                    getattr(
                        self.opt,
                        "instance_branch_direct_gs_info_samples",
                        32,
                    )
                ),
                backprop_token=bool(
                    getattr(self.opt, "instance_branch_backprop_token", False)
                ),
                unit_image=bool(
                    getattr(self.opt, "instance_branch_unit_image", False)
                ),
                unit_image_dim=int(
                    getattr(self.opt, "instance_branch_unit_image_dim", 64)
                ),
                dino_unit=bool(
                    getattr(self.opt, "instance_branch_unit_dino", False)
                ),
                dino_unit_dim=int(
                    getattr(self.opt, "instance_branch_unit_dino_dim", 64)
                ),
                grounding=bool(
                    getattr(self.opt, "instance_branch_grounding", False)
                ),
                grounding_dim=int(
                    getattr(self.opt, "instance_branch_grounding_dim", 64)
                ),
                grounding_3d_pull=float(
                    getattr(
                        self.opt, "instance_branch_grounding_3d_pull", 0.0
                    )
                ),
                grounding_3d_push=float(
                    getattr(
                        self.opt, "instance_branch_grounding_3d_push", 0.0
                    )
                ),
                grounding_3d_margin=float(
                    getattr(
                        self.opt, "instance_branch_grounding_3d_margin", 0.2
                    )
                ),
                dynamic_queries=bool(
                    getattr(
                        self.opt, "instance_branch_dynamic_queries", False
                    )
                ),
                num_queries=int(
                    getattr(self.opt, "instance_branch_num_queries", 128)
                ),
                query_dim=int(
                    getattr(self.opt, "instance_branch_query_dim", 128)
                ),
                query_layers=int(
                    getattr(self.opt, "instance_branch_query_layers", 2)
                ),
                query_diversity=float(
                    getattr(
                        self.opt, "instance_branch_query_diversity", 0.0
                    )
                ),
                query_diversity_margin=float(
                    getattr(
                        self.opt,
                        "instance_branch_query_diversity_margin",
                        0.1,
                    )
                ),
                center_offset=bool(
                    getattr(
                        self.opt, "instance_branch_center_offset", False
                    )
                ),
                center_offset_hidden=int(
                    getattr(
                        self.opt,
                        "instance_branch_center_offset_hidden",
                        256,
                    )
                ),
                pseudo_conf=float(
                    getattr(self.opt, "instance_branch_pseudo_conf", 0.0)
                ),
                pseudo_min_views=int(
                    getattr(self.opt, "instance_branch_pseudo_min_views", 0)
                ),
                pseudo_unit_min_mass=float(
                    getattr(
                        self.opt,
                        "instance_branch_pseudo_unit_min_mass",
                        0.0,
                    )
                ),
                cluster_pos_weight=float(
                    getattr(
                        self.opt, "instance_branch_cluster_pos_weight", 1.0
                    )
                ),
                cluster_eps=float(
                    getattr(self.opt, "instance_branch_cluster_eps", 1.0)
                ),
                void_fg_share=float(
                    getattr(self.opt, "instance_branch_void_fg_share", 0.5)
                ),
                gaussians_per_token=self.num_gaussians_per_token,
                sic_units=bool(
                    getattr(self.opt, "instance_branch_sic_units", False)
                ),
                sic_queries=int(
                    getattr(
                        self.opt, "instance_branch_sic_queries", 128
                    )
                ),
                sic_dim=int(
                    getattr(self.opt, "instance_branch_sic_dim", 256)
                ),
                sic_heads=int(
                    getattr(self.opt, "instance_branch_sic_heads", 4)
                ),
                sic_layers=int(
                    getattr(self.opt, "instance_branch_sic_layers", 2)
                ),
                sic_usage=float(
                    getattr(self.opt, "instance_branch_sic_usage", 0.02)
                ),
                generative_units=bool(
                    getattr(self.opt, "generative_units", False)
                ),
            )
            self.instance_branch.requires_grad_(True)
            if bool(
                getattr(self.opt, "instance_branch_abs_units", False)
            ):
                # In absolute-student mode the instance mask render must
                # consume the gradient-tracking student GS (like the v2
                # generative path) so instance supervision can shape the
                # unit decoder geometry after the instance stage turns on.
                self.instance_branch.abs_render_grad = True
            if bool(
                getattr(self.opt, "instance_branch_scene_prototypes", False)
            ) or bool(
                getattr(self.opt, "instance_branch_unit_embedding", False)
            ) or bool(
                getattr(self.opt, "instance_branch_scene_assignment", False)
            ) or bool(
                getattr(self.opt, "generative_units", False)
            ):
                unit_resume = str(
                    getattr(self.opt, "instance_branch_unit_resume", "")
                )
                if unit_resume and Path(unit_resume).is_file():
                    from safetensors.torch import load_file

                    ck = load_file(unit_resume, device="cpu")
                    branch = self.instance_branch
                    native = dict(branch.named_parameters())
                    loaded = 0
                    for name, param in native.items():
                        key = f"instance_branch.{name}"
                        if key in ck and ck[key].shape == param.shape:
                            with torch.no_grad():
                                param.copy_(ck[key])
                            loaded += 1
                    freeze_formation = bool(
                        getattr(
                            self.opt,
                            "instance_branch_scene_prototypes",
                            False,
                        )
                    ) or bool(
                        getattr(
                            self.opt,
                            "instance_branch_scene_assignment",
                            False,
                        )
                    ) or bool(
                        getattr(self.opt, "instance_branch_dpg", False)
                    )
                    if freeze_formation:
                        for name, param in branch.named_parameters():
                            if name.startswith(
                                (
                                    "gs_feature_mlp.",
                                    "unit_queries",
                                    "unit_layers.",
                                    "log_unit_temp",
                                    "unit_image_net.",
                                    "dino_proj.",
                                )
                            ):
                                param.requires_grad_(False)
                        print(
                            f"[scene-proto] loaded {loaded} unit-formation "
                            f"keys from {unit_resume} and froze unit formation"
                        )
                    else:
                        print(
                            f"[feature-shaping] loaded {loaded} unit-formation "
                            f"keys from {unit_resume} (warm start, trainable)"
                        )
                else:
                    print(
                        "[scene-proto] WARNING: no unit_resume provided; "
                        "unit formation starts at init (eval must load the "
                        "checkpoint which contains the frozen unit weights)."
                    )
        # Absolute token-aligned student (independent config): decoder
        # tokens -> shared local units -> COMPLETE student GS (no residual /
        # no old-head dependence).  The old GS head is only a frozen no_grad
        # teacher for the bootstrap stage.  Instance grouping reuses the
        # token-units branch above (unit-level cross-token queries + masks
        # rendered through the student GS).
        self.absolute_gs_head = None
        if bool(
            getattr(self.opt, "instance_branch_abs_units", False)
        ):
            abs_units = int(
                getattr(self.opt, "instance_branch_units_per_token", 8)
            )
            self.absolute_gs_head = AbsoluteUnitDecoder(
                token_dim=int(self.opt.token_dim),
                units_per_token=abs_units,
                gaussians_per_unit=max(
                    1, self.num_gaussians_per_token // abs_units
                ),
            )
            self.absolute_gs_head.requires_grad_(True)
            # Absolute-student phase has semantic/prompt OFF: hard-freeze
            # those modules so their parameters cannot receive any updates
            # (including AdamW weight decay on zero grads).
            for module_name in (
                "prompt_matcher",
                "semantic_matcher",
                "semantic_lifting_head",
                "semantic_projector",
                "prompt_semantic_adapter",
                "semantic_head",
                "lseg_teacher",
            ):
                module = getattr(self, module_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad_(False)
            if getattr(self, "instance_branch", None) is not None:
                if bool(
                    getattr(self.opt, "abs_freeze_instance", False)
                ):
                    for param in self.instance_branch.parameters():
                        param.requires_grad_(False)
                else:
                    self.instance_branch.requires_grad_(True)
            self.absolute_gs_head.requires_grad_(True)
            if bool(
                getattr(self.opt, "abs_true_shared_units", False)
            ):
                from tokengs.models.shared_unit_instance_head import (
                    SharedUnitInstanceHead,
                )

                self.tsh_instance_head = SharedUnitInstanceHead(
                    unit_dim=int(self.absolute_gs_head.feat_dim),
                    units_per_token=int(
                        getattr(
                            self.opt,
                            "instance_branch_units_per_token",
                            8,
                        )
                    ),
                    gaussians_per_unit=int(
                        self.absolute_gs_head.gaussians_per_unit
                    ),
                    num_tokens=int(self.opt.num_gs_tokens),
                    num_groups=int(
                        getattr(self.opt, "tsh_num_groups", 100)
                    ),
                    num_heads=int(
                        getattr(self.opt, "tsh_num_heads", 8)
                    ),
                    num_layers=int(
                        getattr(self.opt, "tsh_num_layers", 2)
                    ),
                    query_memory_refine=bool(
                        getattr(self.opt, "tsh_query_memory_refine", False)
                    ),
                    query_memory_refine_rounds=int(
                        getattr(self.opt, "tsh_query_memory_refine_rounds", 2)
                    ),
                )
                self.tsh_instance_head.requires_grad_(True)
                self.ga_idu1_head = None
                if bool(getattr(self.opt, "ta_riu_v3_enabled", False)):
                    if bool(getattr(self.opt, "ta_riu_enabled", False)) or bool(
                        getattr(self.opt, "ta_riu_v2_enabled", False)
                    ):
                        raise ValueError("TA-RIU-v3 excludes TA-RIU-v1/v2")
                    if str(getattr(self.opt, "ga_idu_mode", "off")) != "off":
                        raise ValueError("TA-RIU-v3 requires ga_idu_mode=off")
                    if bool(getattr(self.opt, "tsh_query_memory_refine", False)):
                        raise ValueError("TA-RIU-v3 excludes query-memory refiner")
                    if bool(getattr(self.opt, "tsh_per_gs_refine", False)):
                        raise ValueError("TA-RIU-v3 excludes PGSR/per-GS refinement")
                    self.ta_riu_v3_dual_stream = TARIUV3DualStream(
                        dino_repo_path=str(
                            getattr(self.opt, "ta_riu_v3_dino_repo_path")
                        ),
                        dino_weight_path=str(
                            getattr(self.opt, "ta_riu_v3_dino_weight_path")
                        ),
                        dim=int(getattr(self.opt, "ta_riu_v3_dim", 256)),
                        context_views=int(
                            getattr(self.opt, "ta_riu_v3_context_views", 8)
                        ),
                        units_per_view=int(
                            getattr(self.opt, "ta_riu_v3_units_per_view", 1024)
                        ),
                        instance_depth=int(
                            getattr(self.opt, "ta_riu_v3_instance_depth", 2)
                        ),
                        num_heads=int(getattr(self.opt, "ta_riu_v3_num_heads", 8)),
                        mixer_hidden_dim=int(
                            getattr(self.opt, "ta_riu_v3_mixer_hidden_dim", 512)
                        ),
                    )
                    self.absolute_gs_head.requires_grad_(False)
                    for name, param in self.named_parameters():
                        param.requires_grad_(
                            name.startswith("tsh_instance_head.")
                            or name.startswith("ta_riu_v3_dual_stream.")
                        )
                    print(
                        "[ta-riu-v3] enabled: dual-stream aligned unit "
                        "path; TSH and v3 modules trainable, reconstruction "
                        "backbone/absolute head frozen"
                    )
                if bool(getattr(self.opt, "ta_riu_v2_enabled", False)):
                    if bool(getattr(self.opt, "ta_riu_enabled", False)):
                        raise ValueError("TA-RIU v1 and v2 are mutually exclusive")
                    if str(getattr(self.opt, "ga_idu_mode", "off")) != "off":
                        raise ValueError("TA-RIU v2 requires ga_idu_mode=off")
                    if bool(getattr(self.opt, "tsh_query_memory_refine", False)):
                        raise ValueError("TA-RIU v2 excludes query-memory refiner")
                    if bool(getattr(self.opt, "tsh_per_gs_refine", False)):
                        raise ValueError("TA-RIU v2 excludes PGSR/per-GS refinement")
                    dino = FrozenDINOv2Extractor(
                        str(getattr(self.opt, "ta_riu_v2_dino_repo_path")),
                        str(getattr(self.opt, "ta_riu_v2_dino_weight_path")),
                        model_name=str(getattr(self.opt, "ta_riu_v2_dino_model", "dinov2_vitb14")),
                    )
                    self.ta_riu_v2_unit_encoder = GeometryAlignedDINOUnitEncoder(
                        unit_dim=int(self.absolute_gs_head.feat_dim),
                        dino_dim=int(getattr(self.opt, "ta_riu_v2_dino_dim", 768)),
                        dino_proj_dim=int(getattr(self.opt, "ta_riu_v2_dino_proj_dim", 128)),
                        position_dim=int(getattr(self.opt, "ta_riu_v2_position_dim", 32)),
                        embedding_dim=int(getattr(self.opt, "ta_riu_v2_embedding_dim", 64)),
                        num_tokens=int(self.opt.num_gs_tokens),
                        units_per_token=int(self.absolute_gs_head.units_per_token),
                        gaussians_per_unit=int(self.absolute_gs_head.gaussians_per_unit),
                        dino_extractor=dino,
                    )
                    self.absolute_gs_head.requires_grad_(False)
                    for name, param in self.named_parameters():
                        param.requires_grad_(
                            name.startswith("tsh_instance_head.")
                            or name.startswith("ta_riu_v2_unit_encoder.")
                        )
                    print(
                        "[ta-riu-v2] enabled: TSH + geometry-aligned DINO "
                        "unit encoder trainable; reconstruction frozen"
                    )
                if bool(getattr(self.opt, "ta_riu_enabled", False)):
                    if str(getattr(self.opt, "ga_idu_mode", "off")) != "off":
                        raise ValueError("TA-RIU requires ga_idu_mode=off")
                    if bool(getattr(self.opt, "tsh_query_memory_refine", False)):
                        raise ValueError("TA-RIU excludes query-memory refiner")
                    if bool(getattr(self.opt, "tsh_per_gs_refine", False)):
                        raise ValueError("TA-RIU excludes PGSR/per-GS refinement")
                    self.ta_riu_shared_mixer = SharedUnitMixer(
                        input_dim=int(self.opt.token_dim),
                        dim=int(getattr(self.opt, "ta_riu_dim", 256)),
                        memories=int(getattr(self.opt, "ta_riu_memory_latents", 256)),
                        heads=int(getattr(self.opt, "ta_riu_heads", 8)),
                    )
                    self.ta_riu_geometry_head = GeometryResidualHead(
                        dim=int(getattr(self.opt, "ta_riu_dim", 256)),
                        hidden=int(getattr(self.opt, "ta_riu_dim", 256)),
                        gaussians_per_unit=int(self.absolute_gs_head.gaussians_per_unit),
                    )
                    self.ta_riu_appearance_head = AppearanceResidualHead(
                        dim=int(getattr(self.opt, "ta_riu_dim", 256)),
                        hidden=int(getattr(self.opt, "ta_riu_dim", 256)),
                        gaussians_per_unit=int(self.absolute_gs_head.gaussians_per_unit),
                    )
                    # This is a strict two-readout joint probe: q_abs and the
                    # base absolute GS remain frozen; only the complete TSH
                    # readout, shared mixer, and reconstruction residual heads
                    # are trainable.
                    self.absolute_gs_head.requires_grad_(False)
                    for name, param in self.named_parameters():
                        trainable = name.startswith(
                            (
                                "tsh_instance_head.",
                                "ta_riu_shared_mixer.",
                                "ta_riu_geometry_head.",
                                "ta_riu_appearance_head.",
                            )
                        )
                        param.requires_grad_(trainable)
                    print(
                        "[ta-riu] v1 enabled: full TSH + shared mixer + "
                        "geometry/appearance residuals trainable; all base "
                        "reconstruction modules frozen"
                    )
                if str(getattr(self.opt, "ga_idu_mode", "off")) in ("0", "1"):
                    from tokengs.models.ga_idu import GAIDU1

                    # GA-IDU-0 is the identity-capable public framework; the
                    # same module is constructed for GA-IDU-1.  It never
                    # creates geometry/appearance/quality or cross-view
                    # components, and is initialized fresh outside the base
                    # checkpoint namespace.
                    self.ga_idu1_head = GAIDU1(
                        input_dim=int(getattr(self.opt, "ga_idu_input_dim", 64)),
                        dim=int(getattr(self.opt, "ga_idu_dim", 256)),
                        groups=int(getattr(self.opt, "tsh_num_groups", 100)),
                        heads=int(getattr(self.opt, "ga_idu_heads", 8)),
                    )
                    if str(getattr(self.opt, "ga_idu_mode", "off")) == "0":
                        self.ga_idu1_head.requires_grad_(False)
                    else:
                        self.tsh_instance_head.requires_grad_(False)
                        self.absolute_gs_head.requires_grad_(False)
                if bool(
                    getattr(self.opt, "tsh_query_memory_refine_probe", False)
                ):
                    # Probe mode isolates the new readout: q_abs, the
                    # existing TSH head, absolute GS decoder and backbone
                    # remain frozen; only the new refiner is trainable.
                    self.absolute_gs_head.requires_grad_(False)
                    joint_probe = bool(getattr(
                        self.opt, "tsh_query_memory_refine_head_joint_probe", False
                    ))
                    for name, param in self.tsh_instance_head.named_parameters():
                        param.requires_grad_(
                            joint_probe or name.startswith("query_memory_refiner.")
                        )
                if bool(
                    getattr(self.opt, "tsh_per_gs_refine", False)
                ):
                    from tokengs.models.per_gs_slot_refine_head import (
                        PerGSSlotRefineHead,
                    )

                    self.tsh_slot_refine_head = PerGSSlotRefineHead(
                        unit_dim=int(self.absolute_gs_head.feat_dim),
                        num_groups=int(
                            getattr(self.opt, "tsh_num_groups", 100)
                        ),
                        units_per_token=int(
                            getattr(
                                self.opt,
                                "instance_branch_units_per_token",
                                8,
                            )
                        ),
                        gaussians_per_unit=int(
                            self.absolute_gs_head.gaussians_per_unit
                        ),
                        num_tokens=int(self.opt.num_gs_tokens),
                        hidden=int(
                            getattr(
                                self.opt, "tsh_per_gs_hidden", 128
                            )
                        ),
                        use_group_context=bool(
                            getattr(
                                self.opt,
                                "tsh_per_gs_use_group_context",
                                True,
                            )
                        ),
                    )
                    self.tsh_slot_refine_head.requires_grad_(True)
                    print(
                        "[tsh-pgsr] PerGSSlotRefineHead initialized "
                        f"(zero gate alpha, zero logit proj); "
                        f"params="
                        f"{sum(p.numel() for p in self.tsh_slot_refine_head.parameters())}"
                    )
                if (
                    float(getattr(self.opt, "tsh_mbm_decoder_tail_lr", 0.0)) > 0.0
                    and not bool(
                        getattr(self.opt, "tsh_query_memory_refine_probe", False)
                    )
                ):
                    # SIU3R-mapped joint training: the shared token
                    # transformer tail (decoder_blocks) is trainable at its
                    # own low LR.  The image encoder, the old activation-head
                    # teacher and the static gs_tokens stay hard-frozen.
                    unfrozen = 0
                    refrozen = 0
                    for name, param in self.named_parameters():
                        if name.startswith(
                            "enc_dec_backbone.decoder_blocks."
                        ):
                            param.requires_grad_(True)
                            unfrozen += 1
                        elif (
                            name.startswith(
                                (
                                    "enc_dec_backbone.encoder_blocks.",
                                    "enc_dec_backbone.encoder_norm.",
                                    "patch_embed.",
                                    "patch_plucker_embed.",
                                    "activation_head.",
                                    "anchor_pos_encoder.",
                                )
                            )
                            or name in ("gs_tokens", "gs_tokens_dynamic")
                        ):
                            param.requires_grad_(False)
                            refrozen += 1
                    print(
                        f"[tsh-mbm] decoder tail LR "
                        f"{getattr(self.opt, 'tsh_mbm_decoder_tail_lr', 0.0)}: "
                        f"unfrozen dec params={unfrozen}, "
                        f"re-frozen encoder/teacher/token params={refrozen}"
                    )
        elif getattr(self.opt, "instance_branch_independent", False):
            # Experiment C: fully independent instance-structured branch. The
            # frozen TokenGS reconstruction stays untouched; the branch builds
            # its own anchors + group tokens + Gaussian decoder and consumes
            # the frozen hidden/Gaussians only as detached inputs.
            self.instance_group_head = None
            self.instance_group_conditioner = None
            self.instance_count_head = None
            self.instance_branch = IndependentInstanceBranch(
                opt,
                token_dim=int(self.opt.token_dim),
                num_groups=int(
                    getattr(self.opt, "instance_branch_num_groups", 100)
                ),
                anchor_dim=int(
                    getattr(self.opt, "instance_branch_anchor_dim", 256)
                ),
                num_heads=int(
                    getattr(self.opt, "instance_branch_num_heads", 8)
                ),
                num_layers=int(
                    getattr(self.opt, "instance_branch_num_layers", 2)
                ),
                num_gaussians_per_anchor=int(
                    getattr(
                        self.opt,
                        "instance_branch_gaussians_per_anchor",
                        16,
                    )
                ),
                pos_offset_scale=float(
                    getattr(
                        self.opt, "instance_branch_pos_offset_scale", 0.2
                    )
                ),
                scale_delta_amp=float(
                    getattr(
                        self.opt, "instance_branch_scale_delta_amp", 0.5
                    )
                ),
                opacity_delta_amp=float(
                    getattr(
                        self.opt, "instance_branch_opacity_delta_amp", 0.3
                    )
                ),
                rgb_delta_amp=float(
                    getattr(self.opt, "instance_branch_rgb_delta_amp", 0.5)
                ),
                gs_z_offset=float(getattr(self.opt, "gaussian_z_offset", 1.0)),
            )
            self.instance_branch.requires_grad_(True)
        elif getattr(self.opt, "instance_group_conditioned_gaussians", False):
            if getattr(self.opt, "instance_group_condition_generator", False):
                # Experiment B: group tokens condition Gaussian *generation*
                # (token rewrite before the activation head + explicit bounded
                # geometry/opacity deltas), not just mask assignment.
                self.instance_group_head = GroupConditionedGaussianGenerator(
                    token_dim=int(self.opt.token_dim),
                    num_groups=num_groups,
                    condition_dim=int(
                        getattr(self.opt, "instance_group_condition_dim", 256)
                    ),
                    num_heads=int(
                        getattr(self.opt, "instance_group_condition_heads", 8)
                    ),
                    num_layers=int(
                        getattr(self.opt, "instance_group_condition_layers", 2)
                    ),
                    residual_scale=float(
                        getattr(
                            self.opt,
                            "instance_group_condition_residual_scale",
                            0.3,
                        )
                    ),
                    geometry_scale=float(
                        getattr(
                            self.opt,
                            "instance_group_condition_geometry_scale",
                            0.05,
                        )
                    ),
                    opacity_scale=float(
                        getattr(
                            self.opt,
                            "instance_group_condition_per_gaussian_opacity_scale",
                            0.05,
                        )
                    ),
                    num_gaussians_per_token=self.num_gaussians_per_token,
                    assignment_temperature=float(
                        getattr(
                            self.opt,
                            "instance_group_condition_assignment_temperature",
                            10.0,
                        )
                    ),
                )
            else:
                # GC2 uses one shared set of object queries for both Gaussian
                # conditioning and the supervised instance assignment.  Do not
                # construct a second post-hoc group head in this mode.
                self.instance_group_head = GroupConditionedGaussianAdapter(
                    token_dim=int(self.opt.token_dim),
                    num_groups=num_groups,
                    condition_dim=int(
                        getattr(self.opt, "instance_group_condition_dim", 256)
                    ),
                    num_heads=int(
                        getattr(self.opt, "instance_group_condition_heads", 8)
                    ),
                    num_layers=int(
                        getattr(self.opt, "instance_group_condition_layers", 2)
                    ),
                    residual_scale=float(
                        getattr(
                            self.opt,
                            "instance_group_condition_residual_scale",
                            1.0,
                        )
                    ),
                    num_gaussians_per_token=self.num_gaussians_per_token,
                    assignment_temperature=float(
                        getattr(
                            self.opt,
                            "instance_group_condition_assignment_temperature",
                            10.0,
                        )
                    ),
                    per_gaussian_assignment=bool(
                        getattr(
                            self.opt,
                            "instance_group_condition_per_gaussian",
                            False,
                        )
                    ),
                    per_gaussian_opacity_scale=float(
                        getattr(
                            self.opt,
                            "instance_group_condition_per_gaussian_opacity_scale",
                            0.05,
                        )
                    ),
                    image_aligned_anchors=bool(
                        getattr(
                            self.opt,
                            "instance_group_condition_image_anchors",
                            False,
                        )
                    ),
                    image_feature_dim=int(
                        getattr(
                            self.opt,
                            "instance_group_condition_image_feature_dim",
                            64,
                        )
                    ),
                    image_upsample=int(
                        getattr(
                            self.opt,
                            "instance_group_condition_image_upsample",
                            2,
                        )
                    ),
                    image_multiscale=bool(
                        getattr(
                            self.opt,
                            "instance_group_condition_image_multiscale",
                            True,
                        )
                    ),
                    image_scale=float(
                        getattr(
                            self.opt,
                            "instance_group_condition_image_scale",
                            1.0,
                        )
                    ),
                )
        elif getattr(self.opt, "instance_group_per_gaussian", False):
            # Per-Gaussian head consumes the raw token hidden states; the
            # anchor-position conditioning happens at Gaussian level inside
            # the head, so no token-level concat here.
            if getattr(self.opt, "instance_group_residual_head", False):
                head_kwargs = dict(
                    token_dim=int(self.opt.token_dim),
                    num_groups=num_groups,
                    feature_dim=int(
                        getattr(self.opt, "instance_group_feature_dim", 16)
                    ),
                    num_layers=int(
                        getattr(self.opt, "instance_group_decoder_layers", 2)
                    ),
                    use_anchor_pos=getattr(
                        self.opt, "instance_group_use_anchor_pos", True
                    ),
                    num_gaussians_per_token=self.num_gaussians_per_token,
                    residual_scale=float(
                        getattr(self.opt, "instance_group_residual_scale", 0.3)
                    ),
                    pos_attn_layers=int(
                        getattr(self.opt, "instance_group_pos_attn_layers", 0)
                    ),
                    pos_attn_scale=float(
                        getattr(
                            self.opt, "instance_group_pos_attn_scale", 1.0
                        )
                    ),
                )
                if getattr(self.opt, "instance_group_dense_decoder", False):
                    head_kwargs.update(
                        dict(
                            dense_feature_dim=int(
                                getattr(
                                    self.opt,
                                    "instance_group_dense_feature_dim",
                                    16,
                                )
                            ),
                            dense_upsample=int(
                                getattr(
                                    self.opt, "instance_group_dense_upsample", 2
                                )
                            ),
                            dense_multiscale=bool(
                                getattr(
                                    self.opt,
                                    "instance_group_dense_multiscale",
                                    True,
                                )
                            ),
                            dense_scale=float(
                                getattr(
                                    self.opt, "instance_group_dense_scale", 0.3
                                )
                            ),
                            dense_gate=bool(
                                getattr(
                                    self.opt, "instance_group_dense_gate", True
                                )
                            ),
                        )
                    )
                    self.instance_group_head = DenseInstanceResidualHead(
                        **head_kwargs
                    )
                else:
                    self.instance_group_head = PerGaussianResidualHead(
                        **head_kwargs
                    )
            else:
                if getattr(
                    self.opt, "instance_group_group_token_refine", False
                ):
                    # Group tokens (global instance proposal) + per-Gaussian
                    # refinement: the zero-init residual starts exactly at
                    # the token-level baseline and sharpens boundaries.
                    self.instance_group_head = GroupTokenPerGaussianHead(
                        token_dim=int(self.opt.token_dim),
                        num_groups=num_groups,
                        num_layers=int(
                            getattr(
                                self.opt, "instance_group_decoder_layers", 2
                            )
                        ),
                        num_gaussians_per_token=self.num_gaussians_per_token,
                        refine_scale_init=float(
                            getattr(
                                self.opt,
                                "instance_group_group_token_refine_scale_init",
                                0.05,
                            )
                        ),
                        refine_scale_max=float(
                            getattr(
                                self.opt,
                                "instance_group_group_token_refine_scale_max",
                                0.3,
                            )
                        ),
                    )
                else:
                    self.instance_group_head = PerGaussianInstanceGroupHead(
                        token_dim=int(self.opt.token_dim),
                        num_groups=num_groups,
                        feature_dim=int(
                            getattr(self.opt, "instance_group_feature_dim", 16)
                        ),
                        num_layers=int(
                            getattr(
                                self.opt, "instance_group_decoder_layers", 2
                            )
                        ),
                        use_anchor_pos=getattr(
                            self.opt, "instance_group_use_anchor_pos", True
                        ),
                        num_gaussians_per_token=self.num_gaussians_per_token,
                    )
        elif getattr(self.opt, "instance_group_decoder", False):
            self.instance_group_head = InstanceGroupDecoder(
                token_dim=head_input_dim,
                num_groups=num_groups,
                num_layers=int(
                    getattr(self.opt, "instance_group_decoder_layers", 2)
                ),
            )
        else:
            self.instance_group_head = InstanceGroupHead(
                token_dim=head_input_dim,
                num_groups=num_groups,
            )
        if self.instance_group_head is not None:
            self.instance_group_head.requires_grad_(True)
        # Scene-adaptive count head: predicts the per-scene instance count
        # (log-regression on pooled anchor features + positions), used to
        # prune inactive groups at eval and to supervise only the top-G
        # groups during training.
        if getattr(self.opt, "instance_group_count_head", False):
            count_input_dim = int(self.opt.token_dim)
            if getattr(self.opt, "instance_group_use_anchor_pos", False):
                count_input_dim += 64
            self.instance_count_head = InstanceCountHead(
                input_dim=count_input_dim,
                hidden_dim=int(
                    getattr(self.opt, "instance_group_count_hidden", 128)
                ),
            )
            self.instance_count_head.requires_grad_(True)
        # Dense image-evidence decoder: cache the frozen TokenGS encoder's
        # per-patch features (last + one mid block) so the instance head can
        # consume pixel-level evidence without re-running the encoder. The
        # cached tensors are detached (the backbone stays frozen).
        self._dense_feature_cache: dict[str, torch.Tensor] = {}
        self._dense_feature_hooks: list = []
        if getattr(self.opt, "instance_group_dense_decoder", False):

            def _make_dense_cache(name: str):
                def _cache(module, inputs, output):
                    self._dense_feature_cache[name] = output.detach()

                return _cache

            self._dense_feature_hooks.append(
                self.enc_dec_backbone.encoder.register_forward_hook(
                    _make_dense_cache("last")
                )
            )
            if getattr(self.opt, "instance_group_dense_multiscale", True):
                encoder = self.enc_dec_backbone.encoder
                mid_idx = max(0, len(encoder) // 2)
                self._dense_feature_hooks.append(
                    encoder[mid_idx].register_forward_hook(
                        _make_dense_cache("mid")
                    )
                )
        self._configure_token_eru()
        self.train(True)

    def _configure_token_eru(self) -> None:
        enabled = bool(getattr(self.opt, "token_eru_enabled", False))
        mode = str(getattr(self.opt, "token_eru_mode", "disabled"))
        if not enabled or mode in ("disabled", "identity"):
            if enabled and mode not in ("identity", "disabled"):
                raise ValueError(f"Unknown TokenGS-ERU mode: {mode}")
            return
        if mode != "early_dual_stream":
            raise ValueError(f"Unknown TokenGS-ERU mode: {mode}")
        if self.absolute_gs_head is None or self.tsh_instance_head is None:
            raise RuntimeError(
                "TokenGS-ERU requires the existing absolute unit and TSH heads"
            )
        self.token_eru_decoder = TokenGSEarlyDualStreamDecoder(
            self.enc_dec_backbone.decoder_blocks,
            decoder_dim=int(self.opt.enc_embed_dim),
            num_blocks=int(self.opt.dec_depth),
            adapter_bottleneck_dim=int(
                getattr(self.opt, "token_eru_adapter_bottleneck_dim", 128)
            ),
        )
        # This copy is deliberately created before loading the old checkpoint;
        # train.py calls initialize_token_eru_from_reconstruction() after the
        # old 24/50/324 namespaces have been restored.
        self.token_eru_unit_formation = copy.deepcopy(self.absolute_gs_head)
        self.requires_grad_(False)
        self.tsh_instance_head.requires_grad_(True)
        self.token_eru_decoder.understanding_decoder_blocks.requires_grad_(True)
        self.token_eru_decoder.reconstruction_to_understanding.requires_grad_(True)
        self.token_eru_decoder.understanding_to_reconstruction.requires_grad_(True)
        self.token_eru_unit_formation.requires_grad_(True)
        self.token_eru_decoder.set_gates(
            reconstruction_to_understanding=0.0,
            understanding_to_reconstruction=0.0,
        )
        if bool(getattr(self.opt, "token_eru_early_query_codecoder_enabled", False)):
            if int(self.opt.enc_embed_dim) != 1024:
                raise RuntimeError("EQC requires enc_embed_dim=1024")
            if int(getattr(self.opt, "tsh_num_groups", 100)) != 100:
                raise RuntimeError("EQC requires tsh_num_groups=100")
            if int(self.absolute_gs_head.feat_dim) != 256:
                raise RuntimeError("EQC requires native TSH query dim=256")
            cpu_rng = torch.get_rng_state()
            cuda_rng = None
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                cuda_rng = torch.cuda.get_rng_state_all()
            python_rng = random.getstate()
            try:
                self.token_eru_decoder.early_query_adapter = EarlyObjectQueryAdapter(
                    understanding_dim=1024,
                    query_dim=256,
                    attention_dim=256,
                    num_heads=8,
                    mlp_ratio=4.0,
                    dropout=0.0,
                    understanding_write_scale=0.25,
                )
            finally:
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                random.setstate(python_rng)
            self.token_eru_decoder.early_query_layers = tuple(
                int(value)
                for value in getattr(
                    self.opt, "token_eru_early_query_codecoder_layers", (2, 5, 8, 11)
                )
            )
            if self.token_eru_decoder.early_query_layers != (2, 5, 8, 11):
                raise RuntimeError("EQC interaction layers must be (2,5,8,11)")
            print("[token-eru-eqc] shared early query adapter constructed")
        print(
            "[token-eru] early dual stream constructed: "
            f"blocks={len(self.enc_dec_backbone.decoder_blocks)} "
            f"decoder_dim={self.opt.enc_embed_dim}"
        )
        self._configure_token_eru_dino_metric()
        self._configure_token_eru_3d_anchor()
        self._configure_token_eru_query_metric()

    def _configure_token_eru_3d_anchor(self) -> None:
        enabled = bool(getattr(self.opt, "token_eru_3d_anchor_enabled", False))
        if not enabled:
            return
        if self.token_eru_decoder is None or self.token_eru_unit_formation is None:
            raise RuntimeError("3D anchor requires the TokenGS-ERU dual stream")
        self.token_eru_3d_anchor = Unit3DAnchor(
            unit_dim=int(getattr(self.opt, "token_eru_3d_anchor_unit_dim", 256)),
            hidden_dim=int(getattr(self.opt, "token_eru_3d_anchor_hidden_dim", 256)),
            num_frequencies=int(getattr(self.opt, "token_eru_3d_anchor_num_frequencies", 6)),
            eps=float(getattr(self.opt, "token_eru_3d_anchor_eps", 1e-6)),
            min_scale=float(getattr(self.opt, "token_eru_3d_anchor_min_scale", 1e-3)),
            clamp_value=float(getattr(self.opt, "token_eru_3d_anchor_clamp_value", 10.0)),
            injection_scale=float(getattr(self.opt, "token_eru_3d_anchor_injection_scale", 1.0)),
            detach_statistics=bool(getattr(self.opt, "token_eru_3d_anchor_detach_statistics", True)),
        )
        self.token_eru_3d_anchor.requires_grad_(True)
        print("[token-eru-3d-anchor] reconstruction-derived unit anchor constructed")

    def _configure_token_eru_query_metric(self) -> None:
        enabled = bool(getattr(self.opt, "token_eru_query_metric_enabled", False))
        if not enabled:
            return
        required = {
            "token_eru_enabled": bool(getattr(self.opt, "token_eru_enabled", False)),
            "token_eru_dino_metric_enabled": bool(
                getattr(self.opt, "token_eru_dino_metric_enabled", False)
            ),
            "tsh_instance_head": self.tsh_instance_head is not None,
            "token_eru_metric_head": self.token_eru_metric_head is not None,
            "token_eru_3d_anchor_enabled": not bool(
                getattr(self.opt, "token_eru_3d_anchor_enabled", False)
            ),
            "token_eru_dino_eval_mode": str(
                getattr(self.opt, "token_eru_dino_eval_mode", "query")
            ) == "query",
            "tsh_num_groups": int(getattr(self.opt, "tsh_num_groups", 100)) == 100,
        }
        failed = [name for name, ok in required.items() if not ok]
        if failed:
            raise RuntimeError(
                "QMC requires the native ERU-DINO query path; failed="
                + ",".join(failed)
            )
        cpu_rng = torch.get_rng_state()
        cuda_rng = None
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            cuda_rng = torch.cuda.get_rng_state_all()
        python_rng = random.getstate()
        try:
            self.token_eru_query_metric_coupling = QueryMetricCoupling(
                unit_embedding_dim=int(
                    getattr(self.opt, "token_eru_query_metric_embedding_dim", 128)
                ),
                query_dim=int(
                    getattr(self.opt, "token_eru_query_metric_query_dim", 256)
                ),
                num_groups=int(getattr(self.opt, "token_eru_query_metric_num_groups", 100)),
                initial_temperature=float(
                    getattr(self.opt, "token_eru_query_metric_initial_temperature", 10.0)
                ),
                max_gate=float(
                    getattr(self.opt, "token_eru_query_metric_max_gate", 0.25)
                ),
            )
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            random.setstate(python_rng)
        self.token_eru_query_metric_coupling.requires_grad_(True)
        print("[token-eru-qmc] query-metric coupling constructed")

    def _configure_token_eru_dino_metric(self) -> None:
        enabled = bool(getattr(self.opt, "token_eru_dino_metric_enabled", False))
        if not enabled:
            return
        if self.token_eru_decoder is None or self.token_eru_unit_formation is None:
            raise RuntimeError(
                "TokenGS-ERU-DINO-Metric requires the early dual-stream ERU path"
            )
        cpu_rng = torch.get_rng_state()
        cuda_rng = None
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            cuda_rng = torch.cuda.get_rng_state_all()
        python_rng = random.getstate()
        try:
            self.token_eru_dino_encoder = HistoricalDINOUnitEncoder(
                getattr(
                    self.opt,
                    "token_eru_dino_repo_path",
                    "/space/mawb/.cache/torch/hub/facebookresearch_dinov2_main",
                ),
                getattr(
                    self.opt,
                    "token_eru_dino_weight_path",
                    "/space/mawb/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth",
                ),
                output_dim=256,
                num_context_views=8,
                num_tokens=1024,
                units_per_token=8,
            )
            self.token_eru_dino_fusion = DINOUnitFusion(256, 256)
            self.token_eru_metric_head = MetricEmbeddingHead(
                input_dim=256,
                hidden_dim=256,
                embedding_dim=int(getattr(self.opt, "token_eru_dino_embedding_dim", 128)),
            )
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            random.setstate(python_rng)
        self.token_eru_dino_encoder.requires_grad_(True)
        self.token_eru_dino_fusion.requires_grad_(True)
        self.token_eru_metric_head.requires_grad_(True)
        # The external DINO model is still frozen and is deliberately not a
        # registered child of FrozenDINOv2Extractor.
        self.token_eru_dino_encoder.dino_extractor.requires_grad_(False)
        print(
            "[token-eru-dino] metric branch constructed: "
            f"embedding_dim={getattr(self.opt, 'token_eru_dino_embedding_dim', 128)}"
        )

    def initialize_token_eru_from_reconstruction(self) -> dict[str, int]:
        """Copy loaded reconstruction modules into the fresh ERU branches."""
        if self.token_eru_decoder is None:
            return {"decoder_blocks": 0, "unit_keys": 0}
        self.token_eru_decoder.initialize_understanding_from_reconstruction()
        result = self.token_eru_unit_formation.load_state_dict(
            self.absolute_gs_head.state_dict(), strict=True
        )
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "TokenGS-ERU strict unit-formation copy failed: "
                f"missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
        return {
            "decoder_blocks": len(self.token_eru_decoder.understanding_decoder_blocks),
            "unit_keys": len(self.absolute_gs_head.state_dict()),
        }

    @staticmethod
    def token_eru_gates(step: int, opt) -> tuple[float, float]:
        if int(step) < 0:
            raise ValueError("TokenGS-ERU step must be non-negative")
        if bool(
            getattr(opt, "token_eru_dino_metric_joint_formation", False)
        ) or bool(
            getattr(opt, "token_eru_dino_metric_joint_formation_j2", False)
        ):
            # JointFormation starts from the already-native ERU@500 state;
            # its parent gates remain fixed rather than replaying the ERU
            # warm-up schedule from zero.
            return 1.0, 0.1
        r2u_steps = max(1, int(getattr(opt, "token_eru_r2u_ramp_steps", 25)))
        u2r_start = int(getattr(opt, "token_eru_u2r_start_step", 25))
        u2r_steps = max(1, int(getattr(opt, "token_eru_u2r_ramp_steps", 25)))
        u2r_max = float(getattr(opt, "token_eru_u2r_max_gate", 0.1))
        step = int(step)
        r2u = 0.0 if step == 0 else min(1.0, step / r2u_steps)
        if step <= u2r_start:
            u2r = 0.0
        else:
            u2r = u2r_max * min(1.0, (step - u2r_start) / u2r_steps)
        return float(r2u), float(u2r)

    def set_token_eru_step(self, completed_optimizer_step: int) -> dict[str, float]:
        if self.token_eru_decoder is None:
            return {
                "completed_optimizer_step": float(completed_optimizer_step),
                "reconstruction_to_understanding_gate": 0.0,
                "understanding_to_reconstruction_gate": 0.0,
            }
        r2u, u2r = self.token_eru_gates(completed_optimizer_step, self.opt)
        self.token_eru_decoder.set_gates(
            reconstruction_to_understanding=r2u,
            understanding_to_reconstruction=u2r,
        )
        if self.token_eru_dino_encoder is not None:
            self.set_token_eru_dino_metric_step(completed_optimizer_step)
        if (
            self.token_eru_query_metric_coupling is not None
            and bool(getattr(self.opt, "evaluating", False))
        ):
            self.set_token_eru_query_metric_step(completed_optimizer_step)
        return {
            "completed_optimizer_step": float(completed_optimizer_step),
            "reconstruction_to_understanding_gate": r2u,
            "understanding_to_reconstruction_gate": u2r,
        }

    @staticmethod
    def token_eru_query_metric_gate(local_stage_step: int, opt) -> float:
        if int(local_stage_step) < 0:
            raise ValueError("QMC local stage step must be non-negative")
        if int(local_stage_step) == 0:
            return 0.0
        ramp_steps = max(
            1,
            int(getattr(opt, "token_eru_query_metric_gate_ramp_steps", 25)),
        )
        return float(
            getattr(opt, "token_eru_query_metric_max_gate", 0.25)
        ) * min(1.0, int(local_stage_step) / float(ramp_steps))

    def set_token_eru_query_metric_step(self, local_stage_step: int) -> dict[str, float]:
        if int(local_stage_step) < 0:
            raise ValueError("QMC local stage step must be non-negative")
        self._token_eru_query_metric_local_step = int(local_stage_step)
        gate = self.token_eru_query_metric_gate(local_stage_step, self.opt)
        return {
            "local_stage_step": float(local_stage_step),
            "query_metric_gate": gate,
        }

    @staticmethod
    def token_eru_early_query_gate(local_stage_step: int, opt) -> float:
        if int(local_stage_step) < 0:
            raise ValueError("EQC local stage step must be non-negative")
        return min(1.0, int(local_stage_step) / 25.0)

    def set_token_eru_early_query_step(self, local_stage_step: int) -> dict[str, float]:
        if int(local_stage_step) < 0:
            raise ValueError("EQC local stage step must be non-negative")
        self._token_eru_early_query_local_step = int(local_stage_step)
        self._token_eru_early_query_gate = self.token_eru_early_query_gate(
            local_stage_step, self.opt
        )
        return {
            "local_stage_step": float(local_stage_step),
            "early_query_gate": float(self._token_eru_early_query_gate),
        }

    @staticmethod
    def token_eru_dino_schedule(step: int, opt) -> tuple[float, float]:
        step = int(step)
        joint_formation = bool(
            getattr(opt, "token_eru_dino_metric_joint_formation", False)
        ) or bool(
            getattr(opt, "token_eru_dino_metric_joint_formation_j2", False)
        )
        if not joint_formation and step < 500:
            return 0.0, 0.0
        start = int(
            getattr(
                opt,
                "token_eru_dino_gate_start_step",
                0 if joint_formation else 500,
            )
        )
        end = int(getattr(opt, "token_eru_dino_gate_end_step", 525))
        if end <= start:
            raise ValueError("DINO gate end step must be greater than start step")
        if step <= start:
            value = 0.0
        else:
            value = min(1.0, (step - start) / float(end - start))
        return float(value), float(value)

    def set_token_eru_dino_metric_step(self, completed_optimizer_step: int) -> dict[str, float]:
        if int(completed_optimizer_step) < 0:
            raise ValueError("DINO metric step must be non-negative")
        self._token_eru_dino_metric_step = int(completed_optimizer_step)
        gate, weight = self.token_eru_dino_schedule(completed_optimizer_step, self.opt)
        return {
            "completed_optimizer_step": float(completed_optimizer_step),
            "dino_gate": gate,
            "metric_loss_weight": float(getattr(self.opt, "token_eru_dino_metric_loss_weight", 0.0)) * weight,
        }

    def _forward_token_eru_dino_metric(
        self,
        data: dict,
        understanding_units: torch.Tensor,
        reconstruction_gaussians: torch.Tensor,
        model_input,
        training: bool,
    ) -> dict:
        if self.token_eru_dino_encoder is None:
            return {}
        if "images_input" not in data:
            raise RuntimeError("TokenGS-ERU-DINO-Metric requires context images_input")
        context = data["images_input"]
        if context.ndim != 5 or context.shape[1] != 8:
            raise RuntimeError(f"expected context RGB [B,8,3,H,W], got {tuple(context.shape)}")
        evidence: DINOUnitEvidence = self.token_eru_dino_encoder(
            context,
            {
                "base_gaussians": reconstruction_gaussians.detach(),
                "cam_to_world_input": data["cam_to_world_input"],
                "intrinsics_input": data["intrinsics_input"],
                "image_hw": tuple(int(x) for x in context.shape[-2:]),
            },
        )
        gate, ramp = self.token_eru_dino_schedule(
            self._token_eru_dino_metric_step, self.opt
        )
        fused = self.token_eru_dino_fusion(
            understanding_units, evidence.unit_features, gate
        )
        embeddings = self.token_eru_metric_head(fused)
        zero = embeddings.sum() * 0.0
        metric_stats = None
        if training and "instance_label_input" in data and "instance_label_output" in data:
            targets, valid = build_historical_soft_unit_targets(
                reconstruction_gaussians.detach(),
                data,
                tuple(int(x) for x in data["instance_label_output"].shape[-2:]),
            )
            metric_stats = historical_soft_unit_infonce(
                embeddings.reshape(embeddings.shape[0], 8192, -1),
                targets,
                valid,
                temperature=0.1,
            )
            metric_loss = metric_stats.loss
        else:
            metric_loss = zero
        loss_weight = float(getattr(self.opt, "token_eru_dino_metric_loss_weight", 0.0)) * ramp
        output = {
            "dino_unit_features": evidence.unit_features,
            "dino_patch_features": evidence.patch_features,
            "dino_alignment_weights": evidence.alignment_weights,
            "fused_understanding_units": fused,
            "unit_metric_embeddings": embeddings,
            "loss_instance_metric": metric_loss,
            "loss_instance_metric_weighted": metric_loss * loss_weight,
            "metric_loss_diagnostics": metric_stats,
            "dino_gate": torch.tensor(gate, device=fused.device),
            "metric_loss_weight_eff": torch.tensor(loss_weight, device=fused.device),
        }
        if metric_stats is not None:
            output.update(
                {
                    "metric_valid_unit_count": torch.tensor(
                        float(metric_stats.valid_unit_count), device=fused.device
                    ),
                    "metric_positive_pair_count": torch.tensor(
                        float(metric_stats.positive_pair_count), device=fused.device
                    ),
                    "metric_negative_pair_count": torch.tensor(
                        float(metric_stats.negative_pair_count), device=fused.device
                    ),
                    "metric_mean_positive_similarity": metric_stats.mean_positive_similarity,
                    "metric_mean_negative_similarity": metric_stats.mean_negative_similarity,
                    "metric_target_entropy": metric_stats.target_entropy,
                }
            )
        if not training:
            output["metric_cluster_output"] = None
        return output

    def build_token_eru_dino_metric_cluster(
        self,
        metric_embeddings: torch.Tensor,
        unit_logits: torch.Tensor,
        reconstruction_gaussians: torch.Tensor,
        model_input,
    ):
        """Build the formal GT-free metric-cluster output after TSH logits.

        Objectness is derived only from the predicted 101-way unit logits;
        no labels or pseudo-target cache enters this path.
        """
        if metric_embeddings.ndim != 4 or metric_embeddings.shape[1:3] != (1024, 8):
            raise ValueError("metric embeddings must be [B,1024,8,128]")
        if unit_logits.ndim == 3 and unit_logits.shape[1:] == (8192, 101):
            unit_logits = unit_logits.reshape(unit_logits.shape[0], 1024, 8, 101)
        elif unit_logits.ndim != 4 or unit_logits.shape[1:3] != (1024, 8):
            raise ValueError(
                "unit logits must be [B,1024,8,101] or canonical-flattened "
                f"[B,8192,101], got {tuple(unit_logits.shape)}"
            )
        probabilities = torch.softmax(unit_logits.float(), dim=-1)
        unit_objectness = (1.0 - probabilities[..., 100]).reshape(
            probabilities.shape[0], 8192
        )
        unit_positions = reconstruction_gaussians[..., :3].reshape(
            reconstruction_gaussians.shape[0], 1024, 8, 8, 3
        ).mean(dim=3).reshape(reconstruction_gaussians.shape[0], 8192, 3)
        return historical_metric_cluster(
            metric_embeddings.reshape(metric_embeddings.shape[0], 8192, -1),
            unit_positions,
            unit_objectness,
            {"values": reconstruction_gaussians, "renderer": self.gs},
            {
                "cam_view": model_input.decoder.cam_view,
                "intrinsics": model_input.decoder.intrinsics,
            },
            eps=float(getattr(self.opt, "token_eru_dino_cluster_eps", 0.5)),
            foreground_threshold=0.5,
            foreground_share=0.5,
        )

    def _forward_abs_hidden(self, model_input):
        if self.token_eru_decoder is None:
            return super()._forward_abs_hidden(model_input)
        self._token_eru_last_early_query_state = None
        self._token_eru_last_early_query_state_delta_norm = 0.0
        self._token_eru_last_early_query_u_residual_norm = 0.0
        encoder_latent = self.forward_encoder(model_input.encoder)
        self._last_encoder_values = encoder_latent.values
        self._last_encoder_memory_meta = {
            "shape": list(encoder_latent.values.shape),
            "layout": "[B, num_heads, sequence_length, head_dim]",
            "num_heads": int(encoder_latent.values.shape[1]),
            "sequence_length": int(encoder_latent.values.shape[2]),
            "head_dim": int(encoder_latent.values.shape[3]),
        }
        gs_tokens = self.get_gs_tokens(
            encoder_latent.keys.shape[0],
            encoder_latent=encoder_latent,
            decoder_input=model_input.decoder,
        )
        gs_tokens = self._apply_time_embedding_to_gs_tokens(
            gs_tokens, model_input.decoder
        )
        early_state = None
        early_enabled = bool(
            getattr(self.opt, "token_eru_early_query_codecoder_enabled", False)
        )
        if early_enabled:
            if self.tsh_instance_head is None:
                raise RuntimeError("EQC requires the native TSH instance head")
            early_state = self.tsh_instance_head.get_object_query_seed(
                int(gs_tokens.shape[0])
            )
            early_state_seed = early_state
            r_hidden, u_hidden, early_state = self.token_eru_decoder(
                gs_tokens,
                encoder_latent,
                early_query_state=early_state,
                early_query_gate=float(self._token_eru_early_query_gate),
            )
            self._token_eru_last_early_query_state = early_state
            self._token_eru_last_early_query_state_delta_norm = float(
                (early_state - early_state_seed).detach().float().norm().item()
            )
            self._token_eru_last_early_query_u_residual_norm = float(
                getattr(self.token_eru_decoder, "_last_early_query_u_residual_norm", 0.0)
            )
        else:
            r_hidden, u_hidden = self.token_eru_decoder(
                gs_tokens, encoder_latent
            )
        self._token_eru_last_reconstruction_tokens = r_hidden
        self._token_eru_last_understanding_tokens = u_hidden
        self._token_eru_understanding_units = self.token_eru_unit_formation.form_units(
            u_hidden
        )
        return r_hidden

    def forward(
        self,
        data: dict,
        skip_loss: bool = False,
        compute_quality_metrics: bool = False,
        **_kwargs,
    ) -> dict:
        if hasattr(self, "_dense_feature_cache"):
            self._dense_feature_cache.clear()
        return super().forward(
            data,
            skip_loss=skip_loss,
            compute_quality_metrics=compute_quality_metrics,
            **_kwargs,
        )

    def train(self, mode: bool = True) -> "SemanticTokenGSv6":
        # PromptTokenGS.train() forces every child (except prompt_matcher)
        # into eval mode; keep the independent instance branch in the model's
        # training mode so its supervision gating works.
        super().train(mode)
        if getattr(self, "instance_branch", None) is not None:
            self.instance_branch.train(mode)
        return self

    def prompt_trainable_groups(self) -> dict[str, list]:
        groups = super().prompt_trainable_groups()
        if self.instance_group_head is not None:
            groups["instance_group_head"] = list(
                self.instance_group_head.parameters()
            )
        elif getattr(self, "instance_branch", None) is not None:
            groups["instance_group_head"] = list(
                self.instance_branch.parameters()
            )
        if self.instance_count_head is not None:
            groups["instance_count_head"] = list(
                self.instance_count_head.parameters()
            )
        return groups

    semantic_trainable_groups = prompt_trainable_groups
