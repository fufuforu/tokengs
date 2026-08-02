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
    dataset_kwargs: dict[str, str] | None = None
    prompt_mode: Literal[
        "text_only", "image_only", "text_image_mixed", "manifest"
    ] = "text_image_mixed"
    query_image_size: tuple[int, int] = (224, 224)
    prompt_image_probability: float = 0.5
    prompt_min_target_pixels: int = 64

    # --- prompt-conditioned token matching
    prompt_training: bool = False
    prompt_tokengs_checkpoint: str = "/space0/mawb/tokengs/workspace/tokengs_re10k/model.safetensors"
    prompt_clip_model_path: str = "/space0/mawb/tokengs/checkpoints/clip-vit-base-patch32"
    prompt_hidden_dim: int = 128
    prompt_attention_heads: int = 4
    prompt_mixed_text_weight: float = 0.5
    prompt_image_pooling: Literal["masked_patch", "masked_input_cls"] = "masked_patch"
    prompt_lambda_bce: float = 1.0
    prompt_lambda_dice: float = 1.0
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
                "conditional_prompt_tokengs",
            ):
                raise ValueError(
                    "prompt_training=True requires a prompt or semantic model"
                )
            if self.num_gs_tokens != 1024:
                raise ValueError("Prompt training requires num_gs_tokens=1024")
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

config_doc["debug_scannet_prompt"] = (
    "Prompt-training ScanNet sample with a forced cross-scene image query."
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

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
