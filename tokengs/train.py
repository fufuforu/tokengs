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

import tyro
import time
import os
import json
import datetime
import glob
import shutil
import subprocess
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, DataLoaderConfiguration
from safetensors.torch import load_file, save_file

import imageio
import numpy as np

from tokengs.options import AllConfigs
from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.instance_group_loss import hungarian_instance_group_loss

import warnings

from tokengs.utils.gaussians import Gaussians
warnings.filterwarnings("ignore")

try:
    from _240_path_remap import remap_path as _240_remap_path
except Exception:  # pragma: no cover - non-240 checkout
    def _240_remap_path(value):
        return value


def setup_workspace_and_status(opt, accelerator):
    """Setup workspace directory and check for existing completion."""
    status_dir = os.path.join(opt.workspace, "status")
    complete_file = os.path.join(status_dir, "COMPLETE")

    if accelerator.is_main_process:
        os.makedirs(status_dir, exist_ok=True)
        if os.path.exists(complete_file):
            if os.path.exists(os.path.join(opt.workspace, "model.safetensors")):
                accelerator.print(
                    f"[workspace] COMPLETE marker found at {complete_file}; "
                    "continuing from the existing checkpoint "
                    "(pass --num_epochs larger than the previous run to "
                    "extend training, or use a fresh --workspace to restart)"
                )
                os.remove(complete_file)
            else:
                raise RuntimeError(
                    f"Found COMPLETE file at {complete_file} but no "
                    "model.safetensors to resume from; use a fresh workspace"
                )


def load_checkpoint_and_resume(opt, accelerator):
    """Load checkpoint and resume training state."""
    epoch_start = 0
    wandb_run_id = None
    
    if os.path.exists(f'{opt.workspace}/model.safetensors') and os.path.exists(f'{opt.workspace}/metadata.json'):
        if accelerator.is_main_process:
            print(f"Resuming from {opt.workspace}/model.safetensors")
        opt.resume = f'{opt.workspace}/model.safetensors'
        
        with open(f'{opt.workspace}/metadata.json', 'r') as f:
            dc = json.load(f)
            epoch_start = dc['epoch'] + 1
            if 'wandb_run_id' in dc:
                wandb_run_id = dc['wandb_run_id']
    
    return epoch_start, wandb_run_id


def setup_wandb(opt, accelerator, epoch_start, wandb_run_id):
    """Setup wandb logging."""
    if not accelerator.is_main_process or not opt.use_wandb:
        return None, None
    
    import wandb
    run_name = datetime.datetime.now().strftime("%b %d, %I:%M%p")
    
    # Initialize wandb - resume if we have a run_id, otherwise create new run
    if wandb_run_id and epoch_start > 0:
        wandb.init(config=asdict(opt), project=opt.project_name, group=opt.experiment_name, 
                  id=wandb_run_id, resume="must")
        print(f"Resuming wandb run {wandb_run_id}")
    else:
        wandb.init(config=asdict(opt), project=opt.project_name, group=opt.experiment_name, 
                  name=f"{opt.experiment_name} {run_name}")
        
    # Store wandb run id for potential future resuming
    wandb_run_id = wandb.run.id
    print(f"wandb run id: {wandb.run.id}")
    
    tensorboard_root_dir = f'{opt.out_dir}/{opt.experiment_name}' if opt.experiment_name else None
    wandb.tensorboard.patch(root_logdir=tensorboard_root_dir, save=False)
    writer = SummaryWriter(log_dir=tensorboard_root_dir)
    print(f"tensorboard root dir: {tensorboard_root_dir}")
    
    return wandb_run_id, writer


def apply_checkpoint_architecture(opt):
    """Patch architecture options from the resume checkpoint metadata.

    Eval and resumed training must build the same model as the checkpoint
    (CLIP variant, feature-field dimension, geometry conditioning, teacher
    projection), otherwise shape mismatches break strict loading.
    """
    resume = getattr(opt, "resume", None)
    if not resume or resume in (None, "None"):
        return
    checkpoint_path = os.path.abspath(resume)
    step_name = os.path.basename(checkpoint_path).replace(
        "model_step_", ""
    ).replace(".safetensors", "")
    candidates = [
        os.path.join(
            os.path.dirname(checkpoint_path),
            f"metadata_step_{step_name}.json",
        ),
        os.path.join(os.path.dirname(checkpoint_path), "metadata_best.json"),
        os.path.join(os.path.dirname(checkpoint_path), "metadata.json"),
    ]
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        with open(candidate, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if "prompt_clip_model_path" in meta:
            opt.prompt_clip_model_path = _240_remap_path(
                str(meta["prompt_clip_model_path"])
            )
        if "semantic_v4_feature_dim" in meta:
            opt.semantic_v4_feature_dim = int(meta["semantic_v4_feature_dim"])
            opt.semantic_v4_use_geometry = bool(
                meta.get("semantic_v4_use_geometry", True)
            )
            opt.semantic_v4_teacher_projection = str(
                meta.get("semantic_v4_teacher_projection", "frozen_random")
            )
        if "semantic_v2_class_weights" in meta:
            weights = tuple(float(value) for value in meta["semantic_v2_class_weights"])
            if len(weights) == 8:
                opt.semantic_v2_class_weights = weights
        print(
            f"[checkpoint-arch] loaded {os.path.basename(candidate)}: "
            f"clip={os.path.basename(str(opt.prompt_clip_model_path))} "
            f"feature_dim={getattr(opt, 'semantic_v4_feature_dim', 32)}"
        )
        return


def load_model_checkpoint(opt, model, accelerator, epoch_start):
    """Load model checkpoint with tolerance for shape mismatches."""
    if getattr(opt, "model_type", None) == "globalsplat_instance_v2":
        mode = str(getattr(opt, "gsi_v2_resume_mode", "official"))
        if mode == "official":
            report = model.load_initial_state()
            accelerator.print(f"[gsi-v2] official restore path={report['path']} sha256={report['sha256']} keys={report['keys']} numel={report['numel']}")
        elif mode == "phase_r_to_joint":
            if not opt.resume or not os.path.isfile(opt.resume):
                raise RuntimeError(f"[gsi-v2] Phase R checkpoint missing: {opt.resume}")
            state = load_file(opt.resume, device="cpu")
            report = model.load_phase_r_state_dict(state)
            accelerator.print(f"[gsi-v2] Phase R -> J restore path={os.path.abspath(opt.resume)} keys={report['loaded_reconstruction_keys']}/{report['reconstruction_keys']} missing={report['missing']}")
        elif mode == "strict":
            if not opt.resume or not os.path.isfile(opt.resume):
                raise RuntimeError(f"[gsi-v2] strict checkpoint missing: {opt.resume}")
            state = load_file(opt.resume, device="cpu")
            incompatible = model.load_state_dict(state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(f"[gsi-v2] strict restore failed: {incompatible}")
            accelerator.print(f"[gsi-v2] strict restore path={os.path.abspath(opt.resume)} keys={len(state)} missing=[] unexpected=[]")
        else:
            raise RuntimeError(f"[gsi-v2] unknown resume mode {mode}")
        return
    if opt.resume is None or opt.resume == 'None':
        return
    
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    source_checkpoint_keys = tuple(ckpt.keys())

    if getattr(opt, "prompt_training", False):
        checkpoint_label = (
            "SemanticTokenGSv2"
            if opt.model_type in (
                "semantic_tokengs_v2",
                "semantic_tokengs_v3",
                "semantic_tokengs_v4",
                "semantic_tokengs_v5",
            )
            else "PromptTokenGS"
        )
        checkpoint_path = os.path.abspath(opt.resume)
        metadata_name = (
            "metadata_best.json"
            if os.path.basename(checkpoint_path) == "model_best.safetensors"
            else "metadata.json"
        )
        metadata_path = os.path.join(os.path.dirname(checkpoint_path), metadata_name)
        checkpoint_step = "unknown"
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as handle:
                checkpoint_step = json.load(handle).get("step", "unknown")
        accelerator.print(
            f"[{checkpoint_label}] loading trainable checkpoint: {checkpoint_path}"
        )
        accelerator.print(f"[{checkpoint_label}] checkpoint step: {checkpoint_step}")
        state_dict = (
            nn.Module.state_dict(model)
            if bool(getattr(opt, "tsh_query_memory_refine_probe", False))
            or bool(getattr(opt, "ta_riu_v2_enabled", False))
            or bool(getattr(opt, "ta_riu_v3_enabled", False))
            else model.state_dict()
        )
        # Frozen-backbone recipes save prompt checkpoints without the
        # TokenGS backbone, but we resume from a full wide7l checkpoint
        # (which includes it). Load the frozen parts directly through the
        # native Module API (the PromptTokenGS.load_state_dict override only
        # knows about the trainable prompt subset), then drop them from the
        # prompt checkpoint so the strict validation below sees a clean set.
        backbone_resume_path = (
            getattr(opt, "backbone_resume", "") or ""
        )
        load_backbone = (
            not getattr(opt, "prompt_unfreeze_tokengs", False)
            or (backbone_resume_path and os.path.isfile(backbone_resume_path))
        )
        if load_backbone:
            frozen_prefixes = [
                "patch_embed.",
                "patch_plucker_embed.",
                "activation_head.",
                "anchor_pos_encoder.",
            ]
            # In the SIU3R-MBM recipes the decoder tail is trainable and is
            # restored by the guarded-joint branch below.  Do not consume it
            # here as part of the frozen-backbone load, otherwise the later
            # strict tail check sees an empty checkpoint and silently falls
            # back to the base8k constructor weights.
            if (
                float(getattr(opt, "tsh_mbm_decoder_tail_lr", 0.0)) <= 0.0
                and not bool(getattr(opt, "ta_riu_enabled", False))
                and not bool(getattr(opt, "ta_riu_v2_enabled", False))
            ):
                frozen_prefixes.append("enc_dec_backbone.")
            frozen_prefixes = tuple(frozen_prefixes)
            frozen_keys = [
                key
                for key in ckpt
                if key.startswith(frozen_prefixes) or key == "gs_tokens"
            ]
            backbone_ckpt = ckpt
            backbone_source = os.path.basename(checkpoint_path)
            if not frozen_keys:
                # Prompt-only checkpoint (frozen-backbone recipe) does not
                # carry the TokenGS backbone; fall back to the full
                # checkpoint that training actually used (backbone_resume).
                backbone_path = getattr(opt, "backbone_resume", "") or ""
                if backbone_path and os.path.isfile(backbone_path):
                    backbone_ckpt = load_file(backbone_path, device="cpu")
                    backbone_source = os.path.basename(backbone_path)
                    frozen_keys = [
                        key
                        for key in backbone_ckpt
                        if key.startswith(frozen_prefixes)
                        or key == "gs_tokens"
                    ]
            if frozen_keys:
                native_state = nn.Module.state_dict(model)
                loadable = {
                    key: backbone_ckpt[key]
                    for key in frozen_keys
                    if key in native_state
                    and native_state[key].shape == backbone_ckpt[key].shape
                }
                # TA-RIU's source is Both@1420: it contains the 324 decoder
                # tail keys but not the other frozen-backbone keys.  A
                # non-empty partial namespace must not suppress the fallback
                # to the exact backbone checkpoint used by Both.
                ta_missing_fallback = 0
                if bool(getattr(opt, "ta_riu_enabled", False)):
                    expected_frozen = [
                        key for key in native_state
                        if key.startswith(frozen_prefixes) or key == "gs_tokens"
                    ]
                    missing_frozen = [
                        key for key in expected_frozen if key not in loadable
                    ]
                    fallback_path = getattr(opt, "backbone_resume", "") or ""
                    if missing_frozen and fallback_path and os.path.isfile(fallback_path):
                        fallback_ckpt = load_file(fallback_path, device="cpu")
                        fallback_loadable = {
                            key: fallback_ckpt[key]
                            for key in missing_frozen
                            if key in fallback_ckpt
                            and key in native_state
                            and native_state[key].shape == fallback_ckpt[key].shape
                        }
                        loadable.update(fallback_loadable)
                        ta_missing_fallback = len(fallback_loadable)
                if loadable:
                    nn.Module.load_state_dict(model, loadable, strict=False)
                    for key in loadable:
                        if key in ckpt:
                            ckpt.pop(key)
                accelerator.print(
                    f"[{checkpoint_label}] frozen-backbone resume: loaded "
                    f"{len(loadable)}/"
                    f"{len(set(frozen_keys).union(loadable.keys()))} frozen keys "
                    f"(source={len(frozen_keys)}, fallback={ta_missing_fallback}) "
                    f"from {backbone_source}"
                )
        # Group-count changes resize only the final head projection; drop it
        # from the checkpoint so it is reinitialized from the fresh state.
        model_head_keys = {
            key
            for key in state_dict
            if key.startswith("instance_group_head.")
        }
        ckpt_head_keys = {
            key for key in ckpt if key.startswith("instance_group_head.")
        }
        head_shape_mismatch = any(
            key in state_dict and ckpt[key].shape != state_dict[key].shape
            for key in ckpt_head_keys
        )
        if (
            model_head_keys
            and ckpt_head_keys
            and (model_head_keys != ckpt_head_keys or head_shape_mismatch)
        ):
            # Key-level merge instead of a wholesale drop: keep every
            # instance_group_head weight that still exists with the same
            # shape (e.g. the warm-started base decoder of the per-Gaussian
            # residual head), drop keys whose architecture changed, and let
            # brand-new head parameters stay at their fresh initialization.
            dropped = 0
            kept = 0
            for key in list(ckpt):
                if not key.startswith("instance_group_head."):
                    continue
                if key in state_dict and ckpt[key].shape == state_dict[key].shape:
                    kept += 1
                else:
                    ckpt.pop(key)
                    dropped += 1
            accelerator.print(
                f"[{checkpoint_label}] instance_group_head architecture "
                f"changed: kept {kept} matching keys, dropped {dropped} "
                f"stale keys ({len(ckpt_head_keys)} ckpt -> "
                f"{len(model_head_keys)} model, "
                f"shape_mismatch={head_shape_mismatch}); new keys stay fresh"
            )
        for token_type in ("gs_tokens", "gs_tokens_dynamic"):
            if (
                token_type not in ckpt
                or token_type not in state_dict
                or ckpt[token_type].shape == state_dict[token_type].shape
            ):
                continue
            old_count = ckpt[token_type].shape[0]
            new_count = state_dict[token_type].shape[0]
            initialized = state_dict[token_type].clone()
            if new_count >= old_count:
                initialized[:old_count].copy_(ckpt[token_type])
                extra = new_count - old_count
                repeat_factor = (extra + old_count - 1) // old_count
                expanded = ckpt[token_type].repeat(repeat_factor, 1)[:extra]
                initialized[old_count:] = (
                    expanded + 0.01 * torch.randn_like(expanded)
                )
            else:
                indices = torch.linspace(0, old_count - 1, new_count).long()
                initialized.copy_(ckpt[token_type][indices])
            ckpt[token_type] = initialized
            accelerator.print(
                f"[{checkpoint_label}] resized {token_type}: "
                f"{old_count} -> {new_count}"
            )
        if bool(getattr(opt, "ta_riu_v3_enabled", False)):
            v3_prefix = "ta_riu_v3_dual_stream."
            forbidden_prefixes = (
                "ta_riu_",
                "ta_riu_v2_",
                "tsh_slot_refine_head.",
                "tsh_query_memory_refine",
                "instance_branch.",
            )
            if any(
                key.startswith(forbidden_prefixes)
                and not key.startswith(v3_prefix)
                for key in source_checkpoint_keys
            ):
                raise RuntimeError(
                    "TA-RIU-v3 source contains an incompatible instance/refine "
                    "module; use base8k, never Both or another instance fork"
                )
            ckpt_has_v3 = any(key.startswith(v3_prefix) for key in ckpt)
            native_state = nn.Module.state_dict(model)
            if ckpt_has_v3:
                required_prefixes = ("absolute_gs_head.", "tsh_instance_head.", v3_prefix)
                expected = [
                    key for key in native_state
                    if key.startswith(required_prefixes)
                ]
                loadable = {
                    key: ckpt[key]
                    for key in ckpt
                    if key.startswith(required_prefixes)
                    and key in native_state
                    and native_state[key].shape == ckpt[key].shape
                }
                missing = [key for key in expected if key not in loadable]
                unexpected = [
                    key for key in ckpt
                    if key.startswith(required_prefixes) and key not in loadable
                ]
                if missing or unexpected:
                    raise RuntimeError(
                        "TA-RIU-v3 strict continuation failed: "
                        f"missing={missing[:5]} unexpected={unexpected[:5]}"
                    )
                nn.Module.load_state_dict(model, loadable, strict=False)
                accelerator.print(
                    "[ta-riu-v3] strict continuation: "
                    f"absolute_gs_head loaded "
                    f"{sum(k.startswith('absolute_gs_head.') for k in loadable)}/24, "
                    f"tsh_instance_head loaded "
                    f"{sum(k.startswith('tsh_instance_head.') for k in loadable)}/50, "
                    f"v3 loaded "
                    f"{sum(k.startswith(v3_prefix) for k in loadable)} keys, "
                    "fresh reset=false"
                )
                return
            full3_path = str(
                getattr(opt, "ta_riu_v3_absolute_head_resume", "") or ""
            )
            if not full3_path:
                raise RuntimeError(
                    "TA-RIU-v3 requires ta_riu_v3_absolute_head_resume"
                )
            if not os.path.isfile(full3_path):
                raise RuntimeError(
                    f"TA-RIU-v3 absolute-head checkpoint missing: {full3_path}"
                )
            full3 = load_file(full3_path, device="cpu")
            abs_expected = [
                key for key in native_state
                if key.startswith("absolute_gs_head.")
            ]
            abs_loadable = {
                key: full3[key]
                for key in full3
                if key.startswith("absolute_gs_head.")
                and key in native_state
                and native_state[key].shape == full3[key].shape
            }
            abs_missing = [key for key in abs_expected if key not in abs_loadable]
            abs_unexpected = [
                key for key in full3
                if key.startswith("absolute_gs_head.") and key not in abs_loadable
            ]
            if len(abs_loadable) != 24 or abs_missing or abs_unexpected:
                raise RuntimeError(
                    "TA-RIU-v3 full3 absolute head must load strict 24/24: "
                    f"loaded={len(abs_loadable)} missing={abs_missing[:5]} "
                    f"unexpected_or_shape={abs_unexpected[:5]}"
                )
            nn.Module.load_state_dict(model, abs_loadable, strict=False)
            if any(
                key.startswith(
                    (
                        "tsh_instance_head.",
                        "ta_riu_v2_unit_encoder.",
                        "tsh_slot_refine_head.",
                    )
                )
                for key in source_checkpoint_keys
            ):
                raise RuntimeError(
                    "TA-RIU-v3 initial source must not contain TSH/v2/PGSR keys"
                )
            fresh_seed = (int(opt.seed) + 987654321) % (2**31)
            torch.manual_seed(fresh_seed)
            tsh_head = getattr(model, "tsh_instance_head", None)
            if tsh_head is None or not hasattr(tsh_head, "reset_parameters_fresh"):
                raise RuntimeError("TA-RIU-v3 requires a fresh TSH head")
            tsh_head.reset_parameters_fresh()
            accelerator.print(
                "[ta-riu-v3] initial lineage: base8k backbone loaded; "
                f"full3 absolute_gs_head loaded 24/24; TSH fresh seed={fresh_seed}; "
                "v3 fresh; fresh reset=false for imported modules"
            )
            return
        if bool(getattr(opt, "instance_branch_abs_units", False)):
            guarded = bool(getattr(opt, "abs_joint_guarded", False)) or bool(
                getattr(opt, "abs_true_shared_units", False)
            )
            if guarded:
                # Checkpoint init / continuation semantics:
                #  * A) first fork from full3 (no tsh_instance_head keys):
                #    strict abs 24/24 load, new head initialized from a
                #    FIXED fresh seed, no legacy dual-unit params imported.
                #  * B) resume of a true-shared training checkpoint (contains
                #    tsh_instance_head.*): abs + tsh head are both loaded and
                #    fresh reset is NEVER triggered.
                # Legacy dual-unit guarded runs keep their previous semantics.
                tsh_mode = bool(
                    getattr(opt, "abs_true_shared_units", False)
                )
                ckpt_has_tsh = any(
                    key.startswith("tsh_instance_head.")
                    for key in ckpt
                )
                ta_prefixes = (
                    "ta_riu_shared_mixer.",
                    "ta_riu_geometry_head.",
                    "ta_riu_appearance_head.",
                )
                ta_v2_prefix = "ta_riu_v2_unit_encoder."
                ckpt_has_ta_riu = any(
                    key.startswith(ta_prefixes) for key in ckpt
                )
                ckpt_has_ta_riu_v2 = any(
                    key.startswith(ta_v2_prefix) for key in ckpt
                )
                if tsh_mode:
                    reinit = not ckpt_has_tsh
                    prefixes = (
                        ("absolute_gs_head.", "tsh_instance_head.")
                        if ckpt_has_tsh
                        else ("absolute_gs_head.",)
                    )
                    # TA-RIU is a new fork: its modules are fresh when the
                    # source is Both@1420, but are strict on continuation.
                    if bool(getattr(opt, "ta_riu_enabled", False)) and ckpt_has_ta_riu:
                        prefixes = prefixes + ta_prefixes
                    if bool(getattr(opt, "ta_riu_v2_enabled", False)) and ckpt_has_ta_riu_v2:
                        prefixes = prefixes + (ta_v2_prefix,)
                else:
                    reinit = bool(
                        getattr(
                            opt, "guarded_instance_head_reinit", False
                        )
                    )
                    prefixes = (
                        ("absolute_gs_head.",)
                        if reinit
                        else (
                            "absolute_gs_head.",
                            "instance_branch.",
                        )
                    )
                native_state = (
                    nn.Module.state_dict(model)
                    if bool(getattr(opt, "ta_riu_v2_enabled", False))
                    else model.state_dict()
                )
                expected = [
                    key for key in native_state
                    if key.startswith(prefixes)
                ]
                loadable = {
                    key: ckpt[key]
                    for key in ckpt
                    if key.startswith(prefixes)
                    and key in native_state
                    and native_state[key].shape == ckpt[key].shape
                }
                missing = [
                    key for key in expected if key not in loadable
                ]
                if bool(getattr(opt, "tsh_query_memory_refine", False)):
                    # First fork from Both@1420: the original 50 TSH keys
                    # remain strict, while the new refiner is intentionally
                    # initialized fresh.  A checkpoint from this probe must
                    # contain all refiner keys and therefore remains strict
                    # on resume.
                    missing_refiner = [
                        key for key in missing
                        if key.startswith("tsh_instance_head.query_memory_refiner.")
                    ]
                    missing = [key for key in missing if key not in missing_refiner]
                if bool(getattr(opt, "ta_riu_v2_enabled", False)) and not ckpt_has_ta_riu_v2:
                    # First fork from Both@1420: v2 is intentionally fresh.
                    missing = [key for key in missing if not key.startswith(ta_v2_prefix)]
                if missing:
                    raise RuntimeError(
                        f"[{checkpoint_label}] guarded-joint resume failed: "
                        f"missing {len(missing)} head keys, e.g. "
                        f"{missing[:5]}"
                    )
                nn.Module.load_state_dict(
                    model, loadable, strict=False
                )
                # SIU3R-MBM family: the shared token-transformer tail has
                # its own low-LR group and must be restored together with
                # the two heads on any resume/eval load; it is saved in the
                # same checkpoint (trainable => kept by state_dict()).
                tail_lr = float(
                    getattr(opt, "tsh_mbm_decoder_tail_lr", 0.0)
                )
                if tail_lr > 0.0 or bool(getattr(opt, "ta_riu_enabled", False)) or bool(getattr(opt, "ta_riu_v2_enabled", False)):
                    tail_expected = [
                        key
                        for key in native_state
                        if key.startswith(
                            "enc_dec_backbone.decoder_blocks."
                        )
                    ]
                    tail_in_ckpt = {
                        key: ckpt[key]
                        for key in ckpt
                        if key.startswith(
                            "enc_dec_backbone.decoder_blocks."
                        )
                        and key in native_state
                        and native_state[key].shape == ckpt[key].shape
                    }
                    if not tail_in_ckpt:
                        accelerator.print(
                            f"[{checkpoint_label}] decoder-tail keys not "
                            f"present in checkpoint (dev fork from a "
                            f"frozen-tail run): keeping the constructor "
                            f"(base8k) decoder-tail initialization"
                        )
                    else:
                        tail_missing = [
                            key
                            for key in tail_expected
                            if key not in tail_in_ckpt
                        ]
                        if tail_missing:
                            raise RuntimeError(
                                f"[{checkpoint_label}] decoder-tail resume "
                                f"failed: missing {len(tail_missing)} keys, "
                                f"e.g. {tail_missing[:5]}"
                            )
                        nn.Module.load_state_dict(
                            model, tail_in_ckpt, strict=False
                        )
                        accelerator.print(
                            f"[{checkpoint_label}] decoder-tail restored "
                            f"{len(tail_in_ckpt)}/{len(tail_expected)} "
                            f"keys (tsh_mbm_decoder_tail_lr={tail_lr})"
                        )
                if bool(getattr(opt, "tsh_per_gs_refine", False)):
                    refine_expected = [
                        key
                        for key in native_state
                        if key.startswith("tsh_slot_refine_head.")
                    ]
                    refine_in_ckpt = {
                        key: ckpt[key]
                        for key in ckpt
                        if key.startswith("tsh_slot_refine_head.")
                        and key in native_state
                        and native_state[key].shape == ckpt[key].shape
                    }
                    if not refine_in_ckpt:
                        # First fork from the calibrated Both checkpoint:
                        # keep the freshly initialized (zero-gate) refine
                        # head; NEVER fresh-reset abs/tsh.
                        accelerator.print(
                            f"[{checkpoint_label}] per-GS refine keys not "
                            f"present (fork from Both checkpoint): keeping "
                            f"zero-initialized refine head, abs/tsh fully "
                            f"restored above"
                        )
                    else:
                        refine_missing = [
                            key
                            for key in refine_expected
                            if key not in refine_in_ckpt
                        ]
                        if refine_missing:
                            raise RuntimeError(
                                f"[{checkpoint_label}] per-GS refine resume "
                                f"failed: missing {len(refine_missing)} "
                                f"keys, e.g. {refine_missing[:5]}"
                            )
                        nn.Module.load_state_dict(
                            model, refine_in_ckpt, strict=False
                        )
                        accelerator.print(
                            f"[{checkpoint_label}] per-GS refine restored "
                            f"{len(refine_in_ckpt)}/"
                            f"{len(refine_expected)} keys"
                        )
                abs_keys = sum(
                    1 for key in loadable
                    if key.startswith("absolute_gs_head.")
                )
                ins_keys = len(loadable) - abs_keys
                tsh_keys = sum(
                    1
                    for key in loadable
                    if key.startswith("tsh_instance_head.")
                )
                ta_keys = sum(
                    1 for key in loadable if key.startswith(ta_prefixes)
                )
                ta_v2_keys = sum(
                    1 for key in loadable if key.startswith("ta_riu_v2_unit_encoder.")
                )
                accelerator.print(
                        f"[{checkpoint_label}] guarded-joint resume: "
                        f"true_shared={tsh_mode} "
                        f"absolute_gs_head loaded {abs_keys}/24 strict "
                    f"(expected {sum(1 for k in expected if k.startswith('absolute_gs_head.'))}), "
                    f"instance_branch keys loaded {ins_keys}, "
                    f"tsh_instance_head keys loaded {tsh_keys} "
                    f"ta_riu keys loaded {ta_keys} ta_riu_v2 keys loaded {ta_v2_keys}"
                    + (
                        (
                            "; first fork from full3: tsh head uses "
                            "FIXED fresh seed"
                            if tsh_mode
                            else "; instance branch left at FRESH init "
                            "(guarded_instance_head_reinit=True)"
                        )
                        if reinit
                        else (
                            "; checkpoint continuation: heads loaded, "
                            "fresh reset NOT applied"
                        )
                    )
                )
                if bool(getattr(opt, "ta_riu_enabled", False)) and not ckpt_has_ta_riu:
                    accelerator.print(
                        f"[{checkpoint_label}] TA-RIU modules absent in source "
                        "checkpoint: initialized fresh; abs/TSH continuation "
                        "was not reset"
                    )
                if bool(getattr(opt, "ta_riu_v2_enabled", False)) and not ckpt_has_ta_riu_v2:
                    accelerator.print(
                        f"[{checkpoint_label}] TA-RIU-v2 modules absent in source "
                        "checkpoint: initialized fresh; abs/TSH continuation "
                        "was not reset"
                    )
                if reinit:
                    # Fixed, reproducible fresh seed for the very first fork.
                    fresh_seed = (int(opt.seed) + 987654321) % (2**31)
                    torch.manual_seed(fresh_seed)
                    branch = getattr(model, "instance_branch", None)
                    if branch is not None and hasattr(
                        branch, "reset_parameters_fresh"
                    ):
                        branch.reset_parameters_fresh()
                    tsh_head = getattr(model, "tsh_instance_head", None)
                    if tsh_head is not None and hasattr(
                        tsh_head, "reset_parameters_fresh"
                    ):
                        tsh_head.reset_parameters_fresh()
                    accelerator.print(
                        f"[{checkpoint_label}] guarded-joint instance branch "
                        f"re-initialized with fresh seed {fresh_seed} "
                        f"(imported tsh/instance keys = 0; continuation never "
                        f"enters this branch)"
                    )
                return
            # Absolute-student head-only checkpoint: load every matching
            # weight tolerantly (the frozen TokenGS backbone comes from
            # backbone_resume).  The full-model strict validation below is
            # not applicable to a head-only fork checkpoint.
            matched = torch.nn.Module.load_state_dict(
                model, ckpt, strict=False
            )
            head_matched = len(ckpt) - len(matched.unexpected_keys)
            accelerator.print(
                f"[{checkpoint_label}] abs fork resume: tolerant load "
                f"(head keys matched {head_matched}/{len(ckpt)}; "
                f"full-model missing={len(matched.missing_keys)} expected "
                f"(backbone/CLIP loaded separately), "
                f"unexpected={len(matched.unexpected_keys)})"
            )
            return
        missing = sorted(set(state_dict) - set(ckpt))
        unexpected = sorted(set(ckpt) - set(state_dict))
        # DINOv2 is a frozen external extractor loaded lazily; if an older
        # checkpoint saved its weights (instance_branch._dino_model.*), they
        # are not part of the trainable model and must be ignored on resume.
        unexpected = [
            key
            for key in unexpected
            if not key.startswith("instance_branch._dino_model.")
        ]
        mismatched = sorted(
            (key, tuple(ckpt[key].shape), tuple(state_dict[key].shape))
            for key in set(state_dict) & set(ckpt)
            if ckpt[key].shape != state_dict[key].shape
        )
        accelerator.print(f"[{checkpoint_label}] resume missing keys: {missing}")
        accelerator.print(f"[{checkpoint_label}] resume unexpected keys: {unexpected}")
        accelerator.print(f"[{checkpoint_label}] resume shape-mismatched keys: {mismatched}")
        matcher_prefixes = (
            "semantic_token_adapter.",
            "prompt_semantic_adapter.",
            "prompt_adapter.",
            "log_temperature",
            "matching_decoder.",
        )
        matcher_missing = [
            key
            for key in missing
            if key.startswith(matcher_prefixes)
        ]
        if matcher_missing or unexpected or mismatched:
            raise RuntimeError("Prompt checkpoint failed strict validation")
        if missing:
            accelerator.print(
                f"[{checkpoint_label}] warm-start: initializing "
                f"{len(missing)} non-matcher keys (e.g. semantic last-decoder "
                "fork / unfrozen TokenGS) from the pretrained state"
            )
            fresh_state = model.state_dict()
            for key in missing:
                ckpt[key] = fresh_state[key]
        model.load_state_dict(ckpt, strict=True)
        return
    
    # tolerant load (only load matching shapes)
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict: 
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
        else:
            accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')


def load_fresh_backbone_resume(opt, model, accelerator):
    """Fresh-start (no resume) frozen-backbone recipe support.

    When a Step-2 run starts from scratch (``resume`` unset) on top of a
    domain-adapted backbone, the model constructor only restores the
    ``prompt_tokengs_checkpoint`` encoder/decoder.  If ``backbone_resume``
    points to a different checkpoint (e.g. Step-1 domain adaptation output),
    load its trainable backbone subset (decoder blocks / activation head /
    GS tokens) on top, leaving all heads at their fresh initialization.
    This lets instance heads train from scratch on domain-aligned token
    features instead of silently retraining on the raw pretrained backbone.
    """
    resume = getattr(opt, "resume", None)
    if resume not in (None, "", "None"):
        return
    if not getattr(opt, "prompt_training", False):
        return
    backbone_path = getattr(opt, "backbone_resume", "") or ""
    if not backbone_path or not os.path.isfile(backbone_path):
        return
    base_path = getattr(opt, "prompt_tokengs_checkpoint", "") or ""
    if os.path.abspath(backbone_path) == os.path.abspath(base_path):
        return
    try:
        ckpt = load_file(backbone_path, device="cpu")
    except Exception as exc:  # pragma: no cover - read failure only
        accelerator.print(
            f"[fresh-backbone] failed to read {backbone_path}: {exc}"
        )
        return
    prefixes = (
        "enc_dec_backbone.",
        "patch_embed.",
        "patch_plucker_embed.",
        "activation_head.",
        "anchor_pos_encoder.",
    )
    native_state = nn.Module.state_dict(model)
    loadable = {}
    for key, value in ckpt.items():
        if key == "gs_tokens" or key.startswith(prefixes):
            if key in native_state and native_state[key].shape == value.shape:
                loadable[key] = value
    if not loadable:
        accelerator.print(
            f"[fresh-backbone] no loadable backbone keys in {backbone_path}; "
            "keeping the constructor's pretrained backbone"
        )
        return
    nn.Module.load_state_dict(model, loadable, strict=False)
    accelerator.print(
        f"[fresh-backbone] loaded {len(loadable)} trainable backbone keys "
        f"from {backbone_path} (heads kept at fresh init)"
    )


def load_semantic_adapter_resume(opt, model, accelerator):
    """Warm-start the semantic adapters from a full checkpoint at fresh start.

    Used when semantic losses are turned on for a fresh-head run (e.g. the
    wide7l semantic-stabilizer test): the adapters (semantic_token_adapter,
    prompt_semantic_adapter, gaussian_feature_head, log_temperature) start
    from the trained checkpoint instead of random projections, so the
    semantic losses are meaningful from step 0. No-op when resume is set or
    the option is empty.
    """
    resume = getattr(opt, "resume", None)
    if resume not in (None, "", "None"):
        return
    path = getattr(opt, "prompt_semantic_adapter_resume", "") or ""
    if not path or not os.path.isfile(path):
        return
    try:
        ckpt = load_file(path, device="cpu")
    except Exception as exc:  # pragma: no cover - read failure only
        accelerator.print(
            f"[sem-adapter-resume] failed to read {path}: {exc}"
        )
        return
    prefixes = (
        "semantic_token_adapter.",
        "prompt_semantic_adapter.",
        "gaussian_feature_head.",
        "log_temperature",
    )
    native_state = nn.Module.state_dict(model)
    loadable = {}
    for key, value in ckpt.items():
        if not key.startswith(prefixes):
            continue
        target = key
        if key not in native_state:
            target = "prompt_matcher." + key
        if target in native_state and native_state[target].shape == value.shape:
            loadable[target] = value
    if not loadable:
        accelerator.print(
            "[sem-adapter-resume] no loadable semantic adapter keys in "
            f"{path}; keeping fresh init"
        )
        return
    nn.Module.load_state_dict(model, loadable, strict=False)
    accelerator.print(
        f"[sem-adapter-resume] loaded {len(loadable)} semantic adapter "
        f"keys from {path}"
    )

    if opt.init_tokens_from_existing and epoch_start == 0:
        _initialize_tokens_from_existing(ckpt, state_dict, accelerator)

    if opt.init_latents_from_existing and epoch_start == 0:
        _initialize_latents_from_existing(ckpt, state_dict, accelerator)
    
    if opt.init_dynamic_tokens_from_static and epoch_start == 0 and 'gs_tokens' in ckpt:
        _initialize_dynamic_tokens_from_static(ckpt, state_dict, accelerator)


def _initialize_tokens_from_existing(ckpt, state_dict, accelerator):
    """Initialize tokens from existing checkpoint."""
    with torch.no_grad():
        for token_type in ['gs_tokens', 'gs_tokens_dynamic']:
            if token_type not in ckpt or token_type not in state_dict:
                continue
            pretrained_tokens = ckpt[token_type]
            current_tokens = state_dict[token_type]    
            N_old = pretrained_tokens.shape[0]
            N_new = current_tokens.shape[0]
            
            if N_new != N_old:
                accelerator.print(f'[INFO] Initializing {token_type} from pretrained tokens, N_old: {N_old}, N_new: {N_new}')
            
            if N_new <= N_old:
                # Downsample / take subset if fewer tokens
                idx = torch.linspace(0, N_old - 1, N_new).long()
                current_tokens.copy_(pretrained_tokens[idx])
            else:
                # Copy existing tokens first
                current_tokens[:N_old].copy_(pretrained_tokens)

                # Initialize additional tokens by sampling from pretrained tokens + noise
                extra_tokens = current_tokens[N_old:]
                repeat_factor = (extra_tokens.shape[0] + N_old - 1) // N_old

                expanded = pretrained_tokens.repeat((repeat_factor, 1))[:extra_tokens.shape[0]]
                noise = 0.01 * torch.randn_like(expanded)   # small perturbation
                extra_tokens.copy_(expanded + noise)


def _initialize_latents_from_existing(ckpt, state_dict, accelerator):
    """Initialize latent bottleneck tokens from an existing latent checkpoint."""
    key = "enc_dec_backbone.latents"
    if key not in ckpt or key not in state_dict:
        return

    with torch.no_grad():
        pretrained_latents = ckpt[key]
        current_latents = state_dict[key]
        N_old = pretrained_latents.shape[0]
        N_new = current_latents.shape[0]

        if N_new != N_old:
            accelerator.print(
                f"[INFO] Initializing latent bottleneck from checkpoint, N_old: {N_old}, N_new: {N_new}"
            )

        if N_new <= N_old:
            idx = torch.linspace(0, N_old - 1, N_new).long()
            current_latents.copy_(pretrained_latents[idx])
        else:
            current_latents[:N_old].copy_(pretrained_latents)
            extra_latents = current_latents[N_old:]
            repeat_factor = (extra_latents.shape[0] + N_old - 1) // N_old
            expanded = pretrained_latents.repeat((repeat_factor, 1))[: extra_latents.shape[0]]
            noise = 0.01 * torch.randn_like(expanded)
            extra_latents.copy_(expanded + noise)


def _initialize_dynamic_tokens_from_static(ckpt, state_dict, accelerator):
    """Initialize dynamic tokens from static tokens."""
    accelerator.print(f'[INFO] Initializing dynamic tokens from static tokens')
    
    with torch.no_grad():
        static_tokens = ckpt["gs_tokens"]  # shape: [N_static, D]
        dynamic_tokens = state_dict['gs_tokens_dynamic']          # shape: [N_dynamic, D]
        N_dynamic = dynamic_tokens.shape[0]
        N_static = static_tokens.shape[0]

        if N_dynamic == N_static:
            # direct copy
            dynamic_tokens.copy_(static_tokens)
        elif N_dynamic > N_static:
            # replicate or pad
            repeat_factor = (N_dynamic + N_static - 1) // N_static
            dynamic_tokens.copy_(
                static_tokens.repeat((repeat_factor, 1))[:N_dynamic]
            )
        else:
            # random subset if fewer dynamic tokens
            idx = torch.randperm(N_static)[:N_dynamic]
            dynamic_tokens.copy_(static_tokens[idx])


def _setup_gsi_v2_optimizer(opt, model, accelerator, epoch_start):
    reconstruction = list(model.reconstruction_named_parameters())
    instance = list(model.instance_named_parameters())
    instance_ids = {id(parameter) for _, parameter in instance}
    reconstruction = [(name, parameter) for name, parameter in reconstruction if id(parameter) not in instance_ids]
    seen = [id(parameter) for _, parameter in reconstruction + instance]
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if len(seen) != len(set(seen)) or set(seen) != trainable:
        raise RuntimeError("[gsi-v2] optimizer parameter groups are not a disjoint complete partition")
    groups = []
    def add(entries, lr):
        decay = [p for _, p in entries if p.ndim != 1 and not getattr(p, "_no_weight_decay", False)]
        nodecay = [p for _, p in entries if p.ndim == 1 or getattr(p, "_no_weight_decay", False)]
        if decay:
            groups.append({"params": decay, "lr": float(lr), "weight_decay": float(opt.weight_decay)})
        if nodecay:
            groups.append({"params": nodecay, "lr": float(lr), "weight_decay": 0.0})
    if model.phase == "reconstruction":
        add(reconstruction, opt.gsi_v2_reconstruction_lr)
    else:
        add(reconstruction, opt.gsi_v2_joint_reconstruction_lr)
        add(instance, opt.gsi_v2_instance_lr)
    try:
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=True)
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    if epoch_start > 0 and os.path.isfile(os.path.join(opt.workspace, "optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.workspace, "optimizer.pth"), map_location="cpu"))
    accelerator.print("[gsi-v2] optimizer groups: " + ", ".join(f"{len(g['params'])} tensors lr={g['lr']} wd={g['weight_decay']}" for g in groups))
    return optimizer


def setup_optimizer(opt, model, accelerator, epoch_start):
    """Setup optimizer. Call before accelerator.prepare()."""
    if getattr(opt, "model_type", None) == "globalsplat_instance_v2":
        return _setup_gsi_v2_optimizer(opt, model, accelerator, epoch_start)
    decay_params, nodecay_params = [], []
    geometry_decay, geometry_nodecay = [], []
    guarded_abs_decay, guarded_abs_nodecay = [], []
    guarded_ins_decay, guarded_ins_nodecay = [], []
    refine_decay, refine_nodecay = [], []
    ta_shared_decay, ta_shared_nodecay = [], []
    ta_geo_decay, ta_geo_nodecay = [], []
    ta_app_decay, ta_app_nodecay = [], []
    ta_v2_decay, ta_v2_nodecay = [], []
    v3_dino_decay, v3_dino_nodecay = [], []
    v3_instance_decay, v3_instance_nodecay = [], []
    v3_mixer_decay, v3_mixer_nodecay = [], []
    v3_group_names = {
        "dino_projection": [],
        "instance_stream": [],
        "pair_mixer": [],
    }
    tsh = bool(getattr(opt, "abs_true_shared_units", False))
    guarded = bool(getattr(opt, "abs_joint_guarded", False)) or tsh
    if tsh:
        instance_prefixes = ("tsh_instance_head.",)
        if str(getattr(opt, "ga_idu_mode", "off")) == "1":
            instance_prefixes = instance_prefixes + ("ga_idu1_head.",)
        if bool(getattr(opt, "tsh_per_gs_refine", False)):
            instance_prefixes = instance_prefixes + (
                "tsh_slot_refine_head.",
            )
    else:
        instance_prefixes = ("instance_branch.",)
    abs_group_lr = (
        float(getattr(opt, "tsh_abs_lr", 1e-5))
        if tsh
        else float(getattr(opt, "guarded_abs_lr", 1e-5))
    )
    instance_group_lr = (
        float(getattr(opt, "ga_idu_instance_lr", 1e-4))
        if str(getattr(opt, "ga_idu_mode", "off")) == "1"
        else float(getattr(opt, "tsh_instance_lr", 1e-4))
        if tsh
        else float(getattr(opt, "guarded_instance_lr", 1e-4))
    )
    refine_group_lr = float(getattr(opt, "tsh_query_memory_refine_lr", 1e-4))
    joint_refine = bool(getattr(opt, "tsh_query_memory_refine_head_joint_probe", False))
    geometry_lr = float(getattr(opt, "prompt_unfreeze_tokengs_lr", 0.0))
    separate_geometry = (
        bool(getattr(opt, "prompt_unfreeze_tokengs", False))
        and geometry_lr > 0.0
    )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if bool(getattr(opt, "ta_riu_enabled", False)) and name.startswith(
            (
                "ta_riu_shared_mixer.",
                "ta_riu_geometry_head.",
                "ta_riu_appearance_head.",
            )
        ):
            if name.startswith("ta_riu_shared_mixer."):
                target_decay, target_nodecay = ta_shared_decay, ta_shared_nodecay
            elif name.startswith("ta_riu_geometry_head."):
                target_decay, target_nodecay = ta_geo_decay, ta_geo_nodecay
            else:
                target_decay, target_nodecay = ta_app_decay, ta_app_nodecay
            if param.dim() == 1 or getattr(param, "_no_weight_decay", False):
                target_nodecay.append(param)
            else:
                target_decay.append(param)
            continue
        if bool(getattr(opt, "ta_riu_v2_enabled", False)) and name.startswith(
            "ta_riu_v2_unit_encoder."
        ):
            target = ta_v2_nodecay if (param.dim() == 1 or getattr(param, "_no_weight_decay", False)) else ta_v2_decay
            target.append(param)
            continue
        if bool(getattr(opt, "ta_riu_v3_enabled", False)) and name.startswith(
            "ta_riu_v3_dual_stream."
        ):
            if name.startswith("ta_riu_v3_dual_stream.context_dino."):
                target_decay, target_nodecay = v3_dino_decay, v3_dino_nodecay
                group_name = "dino_projection"
            elif name.startswith(
                (
                    "ta_riu_v3_dual_stream.instance_query_embedding.",
                    "ta_riu_v3_dual_stream.instance_unit_former.",
                )
            ):
                target_decay, target_nodecay = v3_instance_decay, v3_instance_nodecay
                group_name = "instance_stream"
            elif name.startswith("ta_riu_v3_dual_stream.pair_mixer."):
                target_decay, target_nodecay = v3_mixer_decay, v3_mixer_nodecay
                group_name = "pair_mixer"
            else:
                raise RuntimeError(f"unclassified TA-RIU-v3 parameter: {name}")
            v3_group_names[group_name].append(name)
            target = target_nodecay if (
                param.dim() == 1 or getattr(param, "_no_weight_decay", False)
            ) else target_decay
            target.append(param)
            continue
        # Only the actual TokenGS reconstruction path belongs to the lower
        # geometry learning-rate group. Instance/group heads are semantic
        # predictors and must keep the main LR; the old broad predicate put
        # them in the geometry group whenever decoder fine-tuning was enabled.
        is_geometry = separate_geometry and (
            name.startswith("enc_dec_backbone.")
            or name.startswith("activation_head.")
            or name in ("gs_tokens", "gs_tokens_dynamic")
        )
        if param.dim() == 1 or getattr(param, '_no_weight_decay', False):
            if guarded and name.startswith("absolute_gs_head."):
                guarded_abs_nodecay.append(param)
            elif joint_refine and name.startswith("tsh_instance_head.query_memory_refiner."):
                refine_nodecay.append(param)
            elif guarded and any(
                name.startswith(p) for p in instance_prefixes
            ):
                guarded_ins_nodecay.append(param)
            elif is_geometry:
                geometry_nodecay.append(param)
            else:
                nodecay_params.append(param)
        else:
            if guarded and name.startswith("absolute_gs_head."):
                guarded_abs_decay.append(param)
            elif joint_refine and name.startswith("tsh_instance_head.query_memory_refiner."):
                refine_decay.append(param)
            elif guarded and any(
                name.startswith(p) for p in instance_prefixes
            ):
                guarded_ins_decay.append(param)
            elif is_geometry:
                geometry_decay.append(param)
            else:
                decay_params.append(param)

    optim_groups = []
    if len(decay_params) > 0:
        optim_groups.append({'params': decay_params, 'weight_decay': opt.weight_decay})
    if len(nodecay_params) > 0:
        optim_groups.append({'params': nodecay_params, 'weight_decay': 0.0})
    if guarded:
        guarded_lr_groups = (
            (
                guarded_abs_decay,
                guarded_abs_nodecay,
                abs_group_lr,
            ),
            (
                guarded_ins_decay,
                guarded_ins_nodecay,
                instance_group_lr,
            ),
            (refine_decay, refine_nodecay, refine_group_lr),
        )
        for decay_list, nodecay_list, group_lr in guarded_lr_groups:
            if len(decay_list) > 0:
                optim_groups.append(
                    {
                        'params': decay_list,
                        'weight_decay': opt.weight_decay,
                        'lr': group_lr,
                    }
                )
            if len(nodecay_list) > 0:
                optim_groups.append(
                    {
                        'params': nodecay_list,
                        'weight_decay': 0.0,
                        'lr': group_lr,
                    }
                )
        if bool(getattr(opt, "ta_riu_enabled", False)):
            ta_lr_groups = (
                (ta_shared_decay, ta_shared_nodecay, float(getattr(opt, "ta_riu_shared_lr", 1.0e-5))),
                (ta_geo_decay, ta_geo_nodecay, float(getattr(opt, "ta_riu_geometry_lr", 1.0e-5))),
                (ta_app_decay, ta_app_nodecay, float(getattr(opt, "ta_riu_appearance_lr", 1.0e-5))),
            )
            for decay_list, nodecay_list, group_lr in ta_lr_groups:
                if len(decay_list) > 0:
                    optim_groups.append(
                        {
                            'params': decay_list,
                            'weight_decay': opt.weight_decay,
                            'lr': group_lr,
                        }
                    )
                if len(nodecay_list) > 0:
                    optim_groups.append(
                        {
                            'params': nodecay_list,
                            'weight_decay': 0.0,
                            'lr': group_lr,
                        }
                    )
        if bool(getattr(opt, "ta_riu_v2_enabled", False)):
            v2_lr = float(getattr(opt, "ta_riu_v2_fusion_lr", 1.0e-4))
            if ta_v2_decay:
                optim_groups.append(
                    {"params": ta_v2_decay, "weight_decay": opt.weight_decay, "lr": v2_lr}
                )
            if ta_v2_nodecay:
                optim_groups.append(
                    {"params": ta_v2_nodecay, "weight_decay": 0.0, "lr": v2_lr}
                )
        if bool(getattr(opt, "ta_riu_v3_enabled", False)):
            v3_lr_groups = (
                (
                    "dino_projection",
                    v3_dino_decay,
                    v3_dino_nodecay,
                    float(getattr(opt, "ta_riu_v3_dino_projection_lr", 1.0e-4)),
                ),
                (
                    "instance_stream",
                    v3_instance_decay,
                    v3_instance_nodecay,
                    float(getattr(opt, "ta_riu_v3_instance_lr", 1.0e-4)),
                ),
                (
                    "pair_mixer",
                    v3_mixer_decay,
                    v3_mixer_nodecay,
                    float(getattr(opt, "ta_riu_v3_mixer_lr", 5.0e-5)),
                ),
            )
            for _, decay_list, nodecay_list, group_lr in v3_lr_groups:
                if decay_list:
                    optim_groups.append(
                        {
                            "params": decay_list,
                            "weight_decay": opt.weight_decay,
                            "lr": group_lr,
                        }
                    )
                if nodecay_list:
                    optim_groups.append(
                        {"params": nodecay_list, "weight_decay": 0.0, "lr": group_lr}
                    )
            for group_name, names in v3_group_names.items():
                accelerator.print(
                    f"[optimizer] ta_riu_v3 group={group_name} "
                    f"lr={dict((x[0], x[3]) for x in v3_lr_groups)[group_name]} "
                    f"keys={len(names)} params={sum(p.numel() for n, p in model.named_parameters() if n in names)}"
                )
        accelerator.print(
            f"[optimizer] guarded joint groups: "
            f"abs_lr={abs_group_lr} instance_lr={instance_group_lr} "
            f"refine_lr={refine_group_lr} "
            f"(true_shared={tsh}) "
            f"abs_params={len(guarded_abs_decay) + len(guarded_abs_nodecay)} "
            f"instance_params={len(guarded_ins_decay) + len(guarded_ins_nodecay)} "
            f"ta_shared_params={len(ta_shared_decay) + len(ta_shared_nodecay)} "
            f"ta_geo_params={len(ta_geo_decay) + len(ta_geo_nodecay)} "
            f"ta_app_params={len(ta_app_decay) + len(ta_app_nodecay)}"
            f" ta_v2_params={len(ta_v2_decay) + len(ta_v2_nodecay)}"
        )
    if separate_geometry:
        if len(geometry_decay) > 0:
            optim_groups.append(
                {
                    'params': geometry_decay,
                    'weight_decay': opt.weight_decay,
                    'lr': geometry_lr,
                }
            )
        if len(geometry_nodecay) > 0:
            optim_groups.append(
                {
                    'params': geometry_nodecay,
                    'weight_decay': 0.0,
                    'lr': geometry_lr,
                }
            )
        accelerator.print(
            f"[optimizer] separate geometry LR={geometry_lr} "
            f"({len(geometry_decay) + len(geometry_nodecay)} params)"
        )

    optimizer = torch.optim.AdamW(optim_groups, lr=opt.lr, betas=(0.9, 0.95), fused=True)

    fork_continue = bool(
        getattr(opt, "tsh_fork_continue_step", 0) > 0
    )
    if epoch_start > 0 or (
        fork_continue
        and os.path.isfile(os.path.join(opt.workspace, "optimizer.pth"))
    ):
        optimizer.load_state_dict(torch.load(os.path.join(opt.workspace, 'optimizer.pth'), map_location='cpu'))

    return optimizer


def setup_scheduler(opt, optimizer, iters_per_epoch, accelerator, epoch_start):
    """Setup scheduler. Call after accelerator.prepare() with per-GPU iters_per_epoch."""
    if getattr(opt, "model_type", None) == "globalsplat_instance_v2":
        warmup = int(opt.gsi_v2_lr_warmup_steps)
        total_steps = max(1, int(opt.num_epochs) * int(iters_per_epoch // max(1, opt.gradient_accumulation_steps)))
        minimum = float(opt.gsi_v2_lr_min_ratio)
        def multiplier(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
            return minimum + 0.5 * (1.0 - minimum) * (1.0 + float(np.cos(np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
        if epoch_start > 0 and os.path.isfile(os.path.join(opt.workspace, "scheduler.pth")):
            scheduler_state = torch.load(
                os.path.join(opt.workspace, "scheduler.pth"), map_location="cpu"
            )
            scheduler.load_state_dict(scheduler_state)
            # LambdaLR restores last_epoch/_last_lr but does not update the
            # optimizer param groups.  Explicitly restore the saved current
            # LR so the first resumed update is continuous with step1000.
            for group, lr in zip(optimizer.param_groups, scheduler_state["_last_lr"]):
                group["lr"] = float(lr)
        return scheduler
    if opt.lr_scheduler == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        fork_continue = bool(
            getattr(opt, "tsh_fork_continue_step", 0) > 0
        )
        if epoch_start > 0 or (
            fork_continue
            and os.path.isfile(os.path.join(opt.workspace, "scheduler.pth"))
        ):
            scheduler.load_state_dict(torch.load(os.path.join(opt.workspace, 'scheduler.pth')))
        return scheduler

    steps_per_epoch = iters_per_epoch // opt.gradient_accumulation_steps
    total_steps = opt.num_epochs * steps_per_epoch
    pct_start = min(opt.pct_start_steps / total_steps, 0.99)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=opt.lr, total_steps=total_steps,
        pct_start=pct_start, final_div_factor=opt.final_div_factor,
    )

    if epoch_start > 0:
        scheduler.load_state_dict(torch.load(os.path.join(opt.workspace, 'scheduler.pth')))

    return scheduler


def save_checkpoint(
    opt, accelerator, model, optimizer, scheduler, epoch, wandb_run_id, global_step
):
    """Save model checkpoint and metadata."""
    accelerator.wait_for_everyone()
    accelerator.save_model(model, opt.workspace)
    
    if accelerator.is_main_process:
        torch.save(optimizer.state_dict(), os.path.join(opt.workspace, 'optimizer.pth'))
        torch.save(scheduler.state_dict(), os.path.join(opt.workspace, 'scheduler.pth'))
        
        metadata = {'epoch': epoch, 'step': int(global_step)}
        if getattr(opt, "model_type", None) == "globalsplat_instance_v2":
            try:
                from tokengs.models.globalsplat_instance_v2.dependency import sha256_file
                vgg_sha = sha256_file(opt.gsi_v2_vgg_weight_path)
            except Exception:
                vgg_sha = None
            metadata.update(accelerator.unwrap_model(model).lineage_metadata())
            metadata.update({
                "optimizer_step": int(global_step),
                "equivalent_samples": int(global_step) * int(opt.batch_size) * int(getattr(accelerator, "num_processes", 1)),
                "world_size": int(getattr(accelerator, "num_processes", 1)),
                "global_batch": int(opt.batch_size) * int(getattr(accelerator, "num_processes", 1)),
                "gsi_v2_vgg_sha256": vgg_sha,
            })
        if bool(getattr(opt, "tsh_ddp8", False)):
            world = int(getattr(accelerator, "num_processes", 1))
            gbs = int(opt.batch_size) * world
            metadata.update(
                {
                    "optimizer_step": int(global_step),
                    "equivalent_global_samples": int(global_step) * gbs,
                    "epoch": int(epoch),
                    "world_size": world,
                    "per_gpu_batch_size": int(opt.batch_size),
                    "global_batch_size": gbs,
                }
            )
        if getattr(opt, "prompt_training", False):
            metadata["tokengs_checkpoint"] = opt.prompt_tokengs_checkpoint
            metadata["prompt_checkpoint"] = os.path.join(opt.workspace, "model.safetensors")
            metadata["model_type"] = opt.model_type
            if opt.model_type == "prompt_tokengs":
                metadata["prompt_tune_last_cross_attention"] = bool(
                    opt.prompt_tune_last_cross_attention
                )
            if opt.model_type == "conditional_prompt_tokengs":
                metadata["conditional_v3_tune_last_cross_attention"] = bool(
                    opt.conditional_v3_tune_last_cross_attention
                )
        if wandb_run_id:
            metadata['wandb_run_id'] = wandb_run_id
            
        with open(f'{opt.workspace}/metadata.json', 'w') as f:
            json.dump(metadata, f)


def save_gsi_v2_step0_snapshot(opt, accelerator, model, optimizer, scheduler):
    """Save the explicit R1 optimizer-step-0 sidecars on a fresh workspace."""
    if getattr(opt, "model_type", None) != "globalsplat_instance_v2":
        return
    ckpt_dir = os.path.join(opt.workspace, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in unwrapped.state_dict().items()
        }
        save_file(state, os.path.join(ckpt_dir, "model_step_000000.safetensors"))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer_step_000000.pth"))
        torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler_step_000000.pth"))
        try:
            from tokengs.models.globalsplat_instance_v2.dependency import sha256_file
            vgg_sha = sha256_file(opt.gsi_v2_vgg_weight_path)
        except Exception:
            vgg_sha = None
        metadata = {
            "optimizer_step": 0,
            "equivalent_global_samples": 0,
            "epoch": 0,
            "world_size": int(getattr(accelerator, "num_processes", 1)),
            "per_gpu_batch_size": int(opt.batch_size),
            "global_batch_size": int(opt.batch_size) * int(getattr(accelerator, "num_processes", 1)),
            "gsi_v2_vgg_sha256": vgg_sha,
            "checkpoint_keys": len(state),
            "checkpoint_numel": int(sum(value.numel() for value in state.values())),
            "lineage": unwrapped.lineage_metadata(),
        }
        with open(os.path.join(ckpt_dir, "metadata_step_000000.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    if bool(getattr(opt, "abs_ckpt_full_state", False)):
        import random
        torch.save(
            {
                "optimizer_step": 0, "epoch": 0,
                "rank": int(getattr(accelerator, "process_index", -1)),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "python_random": random.getstate(),
            },
            os.path.join(ckpt_dir, f"rng_step_000000_rank{int(getattr(accelerator, 'process_index', -1)):02d}.pth"),
        )
    accelerator.wait_for_everyone()


def save_fork_rng_state(
    opt, accelerator, epoch, global_step
):
    """Per-rank RNG / sampling state for causal-fork continuation."""
    if not bool(getattr(opt, "tsh_ddp8", False)):
        return
    state_dir = os.path.join(opt.workspace, ".fork_state")
    os.makedirs(state_dir, exist_ok=True)
    rank = int(getattr(accelerator, "process_index", -1))
    import random

    payload = {
        "epoch": int(epoch),
        "optimizer_step": int(global_step),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "python_random": random.getstate(),
    }
    torch.save(
        payload,
        os.path.join(state_dir, f"rank{rank}.pt"),
    )


def load_fork_rng_state(opt, accelerator):
    """Restore per-rank RNG state saved by the common warm-up run."""
    if int(getattr(opt, "tsh_fork_continue_step", 0)) <= 0:
        return
    rank = int(getattr(accelerator, "process_index", -1))
    path = os.path.join(
        opt.workspace, ".fork_state", f"rank{rank}.pt"
    )
    if not os.path.isfile(path):
        raise RuntimeError(
            f"missing per-rank fork state for rank {rank}: {path}"
        )
    payload = torch.load(path, map_location="cpu")
    import random

    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state(payload["cuda_rng"])
    random.setstate(payload["python_random"])
    if accelerator.is_main_process:
        print(
            f"[fork] restored rank {rank} RNG state at "
            f"optimizer_step={payload['optimizer_step']}"
        )


def _atomic_touch(path: str) -> None:
    """Create a shared-filesystem marker without exposing a partial file."""
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write("ok\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_text(path: str, text: str) -> None:
    """Write a small diagnostic marker atomically on the shared filesystem."""
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _wait_for_shared_checkpoint(
    ckpt_dir: str,
    step: int,
    required_names: tuple[str, ...],
    rank: int,
    timeout_seconds: int = 1800,
) -> None:
    """Wait for rank0's completed checkpoint using filesystem polling only."""
    complete = os.path.join(ckpt_dir, f"step_{step:06d}.complete")
    failed = os.path.join(ckpt_dir, f"step_{step:06d}.failed")
    rank_failed = os.path.join(ckpt_dir, f"step_{step:06d}.rank*.failed")
    deadline = time.monotonic() + timeout_seconds
    while True:
        if os.path.isfile(failed):
            raise RuntimeError(
                f"checkpoint step={step} failed; rank={rank}; see {failed}"
            )
        if glob.glob(rank_failed):
            raise RuntimeError(
                f"checkpoint step={step} has a failed rank; rank={rank}; "
                f"markers={rank_failed}"
            )
        if os.path.isfile(complete):
            missing = [
                name for name in required_names
                if not os.path.isfile(os.path.join(ckpt_dir, name))
            ]
            if missing:
                raise RuntimeError(
                    f"checkpoint sentinel appeared with missing files; "
                    f"rank={rank} step={step} missing={missing}"
                )
            return
        if time.monotonic() >= deadline:
            missing = [
                name for name in required_names
                if not os.path.isfile(os.path.join(ckpt_dir, name))
            ]
            raise TimeoutError(
                f"timed out waiting for checkpoint sentinel; rank={rank} "
                f"step={step} missing={missing} timeout={timeout_seconds}s"
            )
        time.sleep(1.0)


def _wait_for_rank_markers(
    ckpt_dir: str, step: int, marker_names: tuple[str, ...], rank: int,
    timeout_seconds: int = 1800,
) -> None:
    """Wait for all per-rank RNG markers before rank0 starts long I/O."""
    failed = os.path.join(ckpt_dir, f"step_{step:06d}.failed")
    deadline = time.monotonic() + timeout_seconds
    while True:
        if os.path.isfile(failed) or glob.glob(
            os.path.join(ckpt_dir, f"step_{step:06d}.rank*.failed")
        ):
            raise RuntimeError(
                f"checkpoint step={step} has a failed rank before rank0 save; "
                f"rank={rank}"
            )
        missing = [
            name for name in marker_names
            if not os.path.isfile(os.path.join(ckpt_dir, name))
        ]
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for rank RNG markers; rank={rank} "
                f"step={step} missing={missing} timeout={timeout_seconds}s"
            )
        time.sleep(1.0)


def save_gsi_v2_intra_epoch_checkpoint_synchronized(
    opt, accelerator, model, optimizer, scheduler, epoch, completed_step
):
    """Save a resumable GSI checkpoint without a barrier during rank0 I/O.

    Every rank enters this function and writes its own RNG sidecar.  Rank0
    waits for the per-rank sidecars using the shared filesystem, writes the
    model/optimizer/scheduler/config/metadata atomically, then publishes one
    completion sentinel.  Only after that sentinel exists do all ranks use a
    short matching Accelerator barrier.
    """
    if not bool(getattr(opt, "abs_ckpt_full_state", False)):
        raise RuntimeError(
            "synchronized GSI intra-epoch save requires abs_ckpt_full_state=true"
        )
    ckpt_dir = os.path.join(opt.workspace, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    step = int(completed_step)
    rank = int(getattr(accelerator, "process_index", -1))
    world = int(getattr(accelerator, "num_processes", 1))
    prefix = f"step_{step:06d}"
    model_name = f"model_{prefix}.safetensors"
    optimizer_name = f"optimizer_{prefix}.pth"
    scheduler_name = f"scheduler_{prefix}.pth"
    metadata_name = f"metadata_{prefix}.json"
    config_name = f"config_{prefix}.yaml"
    required = (model_name, optimizer_name, scheduler_name, metadata_name)
    config_path = os.path.join(opt.workspace, "config.yaml")
    if os.path.isfile(config_path):
        required += (config_name,)

    final_paths = [os.path.join(ckpt_dir, name) for name in required]
    for path in final_paths + [
        os.path.join(ckpt_dir, f"{prefix}.complete"),
        os.path.join(ckpt_dir, f"{prefix}.failed"),
    ]:
        if os.path.exists(path):
            raise FileExistsError(
                f"refusing to overwrite existing GSI checkpoint artifact: {path}"
            )

    import random
    rng_name = f"rng_{prefix}_rank{rank:02d}.pth"
    rng_done_name = f"rng_{prefix}_rank{rank:02d}.complete"
    rng_path = os.path.join(ckpt_dir, rng_name)
    rng_tmp = f"{rng_path}.tmp.{os.getpid()}"
    try:
        torch.save(
            {
                "optimizer_step": step,
                "epoch": int(epoch),
                "rank": rank,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state()
                if torch.cuda.is_available()
                else None,
                "python_random": random.getstate(),
            },
            rng_tmp,
        )
        os.replace(rng_tmp, rng_path)
        _atomic_touch(os.path.join(ckpt_dir, rng_done_name))
    except Exception as exc:
        try:
            _atomic_write_text(
                os.path.join(ckpt_dir, f"{prefix}.rank{rank:02d}.failed"),
                f"rank={rank} step={step} rng save failed: {exc!r}\n",
            )
        finally:
            if os.path.exists(rng_tmp):
                os.unlink(rng_tmp)
        raise

    rank_done_names = tuple(
        f"rng_{prefix}_rank{value:02d}.complete" for value in range(world)
    )
    try:
        if accelerator.is_main_process:
            _wait_for_rank_markers(
                ckpt_dir, step, rank_done_names, rank, timeout_seconds=1800
            )
            from safetensors.torch import save_file

            unwrapped = accelerator.unwrap_model(model)
            state = {
                key: value.detach().cpu().contiguous()
                for key, value in unwrapped.state_dict().items()
            }
            model_path = os.path.join(ckpt_dir, model_name)
            model_tmp = f"{model_path}.tmp.{os.getpid()}"
            save_file(state, model_tmp)
            os.replace(model_tmp, model_path)
            del state

            optimizer_path = os.path.join(ckpt_dir, optimizer_name)
            optimizer_tmp = f"{optimizer_path}.tmp.{os.getpid()}"
            optimizer_state = optimizer.state_dict()
            torch.save(optimizer_state, optimizer_tmp)
            del optimizer_state
            os.replace(optimizer_tmp, optimizer_path)

            scheduler_path = os.path.join(ckpt_dir, scheduler_name)
            scheduler_tmp = f"{scheduler_path}.tmp.{os.getpid()}"
            scheduler_state = scheduler.state_dict()
            torch.save(scheduler_state, scheduler_tmp)
            del scheduler_state
            os.replace(scheduler_tmp, scheduler_path)

            if os.path.isfile(config_path):
                config_dst = os.path.join(ckpt_dir, config_name)
                config_tmp = f"{config_dst}.tmp.{os.getpid()}"
                shutil.copy2(config_path, config_tmp)
                os.replace(config_tmp, config_dst)

            try:
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=os.getcwd(),
                    text=True,
                ).strip()
            except Exception:
                git_commit = "unknown"
            gbs = int(opt.batch_size) * world
            metadata = {
                "optimizer_step": step,
                "equivalent_global_samples": step * gbs,
                "epoch": int(epoch),
                "world_size": world,
                "per_gpu_batch_size": int(opt.batch_size),
                "global_batch_size": gbs,
                "git_commit": git_commit,
                "config_path": os.path.abspath(config_path),
                "optimizer_state": os.path.abspath(optimizer_path),
                "scheduler_state": os.path.abspath(scheduler_path),
                "rng_state_pattern": os.path.abspath(
                    os.path.join(ckpt_dir, f"rng_{prefix}_rank*.pth")
                ),
                "checkpoint_protocol": "shared_fs_sentinel_v1",
            }
            metadata_path = os.path.join(ckpt_dir, metadata_name)
            metadata_tmp = f"{metadata_path}.tmp.{os.getpid()}"
            with open(metadata_tmp, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(metadata_tmp, metadata_path)
            _atomic_touch(os.path.join(ckpt_dir, f"{prefix}.complete"))
        else:
            _wait_for_shared_checkpoint(ckpt_dir, step, required, rank)
    except Exception as exc:
        if accelerator.is_main_process:
            try:
                _atomic_write_text(
                    os.path.join(ckpt_dir, f"{prefix}.failed"),
                    f"rank={rank} step={step} checkpoint save failed: {exc!r}\n",
                )
            except Exception:
                pass
        raise
    finally:
        for temporary in glob.glob(os.path.join(ckpt_dir, f"{prefix}*.tmp.*")):
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    # This is intentionally the only NCCL synchronization after rank0's
    # long file I/O.  The published sentinel makes entry symmetric.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print(
            f"[abs-ckpt] saved synchronized intra-epoch full-state checkpoint "
            f"step={step} sentinel={os.path.join(ckpt_dir, f'{prefix}.complete')}"
        )
def save_prompt_validation_checkpoint(
    opt, accelerator, model, epoch, global_step, validation_metrics, is_best
):
    """Save prompt-only validation snapshots and the current best checkpoint."""
    if not getattr(opt, "prompt_save_validation_checkpoints", False):
        return
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_dir = os.path.join(opt.workspace, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, f"model_step_{int(global_step):06d}.safetensors"
        )
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in accelerator.unwrap_model(model).state_dict().items()
        }
        save_file(state, checkpoint_path)
        metadata = {
            "epoch": int(epoch),
            "step": int(global_step),
            "mask_iou": float(validation_metrics["mask_iou"]),
            "validation_loss": float(validation_metrics["loss"]),
            "tokengs_checkpoint": os.path.abspath(opt.prompt_tokengs_checkpoint),
            "prompt_checkpoint": os.path.abspath(checkpoint_path),
            "model_type": opt.model_type,
        }
        if bool(getattr(opt, "tsh_ddp8", False)):
            world = int(getattr(accelerator, "num_processes", 1))
            gbs = int(opt.batch_size) * world
            metadata.update(
                {
                    "optimizer_step": int(global_step),
                    "equivalent_global_samples": int(global_step) * gbs,
                    "world_size": world,
                    "per_gpu_batch_size": int(opt.batch_size),
                    "global_batch_size": gbs,
                }
            )
        if "macro_miou" in validation_metrics:
            metadata["macro_miou"] = float(validation_metrics["macro_miou"])
            metadata["macc"] = float(validation_metrics["macc"])
            metadata["mask_accuracy"] = float(
                validation_metrics.get("mask_accuracy", float("nan"))
            )
        for key in (
            "predicted_foreground_ratio",
            "gt_foreground_ratio",
            "foreground_probability",
            "background_probability",
        ):
            if key in validation_metrics:
                metadata[key] = float(validation_metrics[key])
        if opt.model_type in ("prompt_tokengs", "conditional_prompt_tokengs"):
            metadata["prompt_image_pooling"] = opt.prompt_image_pooling
            metadata["conditional_query_decoder"] = (
                opt.model_type == "conditional_prompt_tokengs"
            )
            metadata["prompt_balanced_bce"] = bool(opt.prompt_balanced_bce)
            if opt.model_type == "prompt_tokengs":
                metadata["prompt_tune_last_cross_attention"] = bool(
                    opt.prompt_tune_last_cross_attention
                )
            if opt.model_type == "conditional_prompt_tokengs":
                metadata["conditional_v3_tune_last_cross_attention"] = bool(
                    opt.conditional_v3_tune_last_cross_attention
                )
        elif opt.model_type in (
            "semantic_tokengs_v2",
            "semantic_tokengs_v3",
            "semantic_tokengs_v4",
            "semantic_tokengs_v5",
            "semantic_tokengs_v6",
        ):
            unwrapped = accelerator.unwrap_model(model)
            metadata["semantic_v2_dim"] = int(opt.semantic_v2_dim)
            metadata["prompt_clip_model_path"] = str(opt.prompt_clip_model_path)
            metadata["semantic_v2_class_weights"] = list(
                getattr(opt, "semantic_v2_class_weights", (1.0,) * 8)
            )
            if opt.model_type in (
                "semantic_tokengs_v4",
                "semantic_tokengs_v5",
                "semantic_tokengs_v6",
            ):
                metadata["semantic_v4_feature_dim"] = int(
                    getattr(opt, "semantic_v4_feature_dim", 32)
                )
                metadata["semantic_v4_use_geometry"] = bool(
                    getattr(opt, "semantic_v4_use_geometry", True)
                )
            metadata["semantic_v4_teacher_projection"] = str(
                getattr(
                    opt, "semantic_v4_teacher_projection", "frozen_random"
                )
            )
            metadata["instance_group_num_groups"] = int(
                getattr(opt, "instance_group_num_groups", 64)
            )
            metadata["instance_group_conditioned_gaussians"] = bool(
                getattr(opt, "instance_group_conditioned_gaussians", False)
            )
            metadata["instance_group_condition_dim"] = int(
                getattr(opt, "instance_group_condition_dim", 256)
            )
            metadata["instance_group_condition_heads"] = int(
                getattr(opt, "instance_group_condition_heads", 8)
            )
            metadata["instance_group_condition_layers"] = int(
                getattr(opt, "instance_group_condition_layers", 2)
            )
            metadata["instance_group_condition_residual_scale"] = float(
                getattr(opt, "instance_group_condition_residual_scale", 1.0)
            )
            metadata[
                "instance_group_condition_assignment_temperature"
            ] = float(
                getattr(
                    opt,
                    "instance_group_condition_assignment_temperature",
                    10.0,
                )
            )
            metadata["instance_group_condition_gaussian_blend"] = float(
                getattr(opt, "instance_group_condition_gaussian_blend", 0.1)
            )
            metadata["instance_group_condition_per_gaussian"] = bool(
                getattr(opt, "instance_group_condition_per_gaussian", False)
            )
            metadata[
                "instance_group_condition_per_gaussian_opacity_scale"
            ] = float(
                getattr(
                    opt,
                    "instance_group_condition_per_gaussian_opacity_scale",
                    0.05,
                )
            )
            metadata["instance_group_per_gaussian"] = bool(
                getattr(opt, "instance_group_per_gaussian", False)
            )
            metadata["instance_group_decoder"] = bool(
                getattr(opt, "instance_group_decoder", False)
            )
            metadata["instance_group_decoder_layers"] = int(
                getattr(opt, "instance_group_decoder_layers", 2)
            )
            metadata["instance_group_use_anchor_pos"] = bool(
                getattr(opt, "instance_group_use_anchor_pos", False)
            )
            metadata["instance_group_residual_head"] = bool(
                getattr(opt, "instance_group_residual_head", False)
            )
            metadata["instance_group_residual_scale"] = float(
                getattr(opt, "instance_group_residual_scale", 0.3)
            )
            metadata["instance_group_adaptive_count"] = bool(
                getattr(opt, "instance_group_adaptive_count", False)
            )
            metadata["instance_group_count_head"] = bool(
                getattr(opt, "instance_group_count_head", False)
            )
            metadata["instance_group_count_hidden"] = int(
                getattr(opt, "instance_group_count_hidden", 128)
            )
            metadata["lambda_instance_group_count"] = float(
                getattr(opt, "lambda_instance_group_count", 0.0)
            )
            metadata["instance_group_pos_attn_layers"] = int(
                getattr(opt, "instance_group_pos_attn_layers", 0)
            )
            metadata["instance_group_pos_attn_scale"] = float(
                getattr(opt, "instance_group_pos_attn_scale", 1.0)
            )
            # For frozen-backbone recipes the resume checkpoint is prompt
            # only, so point the eval's backbone loader at the full
            # checkpoint that training actually used.
            metadata["resume"] = str(
                getattr(opt, "backbone_resume", "")
                or getattr(opt, "resume", "")
                or ""
            )
            if getattr(opt, "backbone_resume", ""):
                metadata["backbone_resume"] = str(opt.backbone_resume)
            metadata["semantic_v2_balanced_bce"] = bool(opt.semantic_v2_balanced_bce)
            metadata["semantic_v2_score_mode"] = opt.semantic_v2_score_mode
            metadata["semantic_v2_tune_last_cross_attention"] = bool(
                opt.semantic_v2_tune_last_cross_attention
            )
            metadata["prompt_unfreeze_tokengs"] = bool(
                opt.prompt_unfreeze_tokengs
            )
            abs_best = bool(
                getattr(opt, "instance_branch_abs_units", False)
            )
            selection_metric = (
                "psnr"
                if abs_best
                else (
                    "argmax_macro_miou"
                    if opt.semantic_v2_score_mode == "softmax"
                    else "mask_iou"
                )
            )
            metadata["selection_metric"] = selection_metric
            if abs_best:
                metadata["selection_psnr"] = float(
                    validation_metrics[selection_metric]
                )
            else:
                metadata["selection_iou"] = float(
                    validation_metrics[selection_metric]
                )
            metadata["temperature"] = float(
                unwrapped.semantic_matcher.temperature.detach().cpu()
            )
        metadata_path = os.path.join(
            checkpoint_dir, f"metadata_step_{int(global_step):06d}.json"
        )
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        accelerator.print(
            f"[prompt-checkpoint] step={global_step} path={os.path.abspath(checkpoint_path)}"
        )
        if is_best:
            best_path = os.path.join(opt.workspace, "model_best.safetensors")
            save_file(state, best_path)
            best_metadata = dict(metadata)
            best_metadata["prompt_checkpoint"] = os.path.abspath(best_path)
            with open(
                os.path.join(opt.workspace, "metadata_best.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(best_metadata, handle, indent=2)
            if bool(getattr(opt, "instance_branch_abs_units", False)):
                accelerator.print(
                    f"[prompt-best] step={global_step} "
                    f"selection_metric=psnr "
                    f"selection_psnr={validation_metrics['psnr']:.4f} "
                    f"path={os.path.abspath(best_path)}"
                )
            else:
                accelerator.print(
                    f"[prompt-best] step={global_step} "
                    f"selection_metric={metadata.get('selection_metric', 'mask_iou')} "
                    f"selection_iou={metadata.get('selection_iou', validation_metrics['mask_iou']):.6f} "
                    f"mask_iou={validation_metrics['mask_iou']:.6f} "
                    f"path={os.path.abspath(best_path)}"
                )
    accelerator.wait_for_everyone()


def log_training_images(
    opt, accelerator, data, out, epoch, i, writer, is_train=True, global_step=None
):
    """Log training/evaluation images or videos."""
    if not accelerator.is_main_process:
        return
    if getattr(opt, "prompt_training", False):
        if opt.model_type in (
            "semantic_tokengs_v2",
            "semantic_tokengs_v3",
            "semantic_tokengs_v4",
            "semantic_tokengs_v5",
            "semantic_tokengs_v6",
        ):
            log_semantic_v2_images(
                opt, data, out, epoch, i, is_train=is_train, global_step=global_step
            )
        else:
            log_prompt_training_images(
                opt, data, out, epoch, i, is_train=is_train, global_step=global_step
            )
        return
        
    prefix = "train" if is_train else "eval"
    
    # Ensure images directory exists
    os.makedirs(f'{opt.workspace}/images', exist_ok=True)
    
    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
    pred_images = np.clip(out['images_pred'].detach().cpu().numpy(), 0, 1) # [B, V, 3, output_size, output_size]
    
    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[4], 3) # [B*output_size, V*output_size, 3]
    imageio.imwrite(f'{opt.workspace}/images/{prefix}_gt_images_{epoch}_{i}.jpg', (np.clip(gt_images, 0, 1) * 255).astype(np.uint8))

    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[4], 3)
    imageio.imwrite(f'{opt.workspace}/images/{prefix}_pred_images_{epoch}_{i}.jpg', (np.clip(pred_images, 0, 1) * 255).astype(np.uint8))

    if opt.use_wandb:
        writer.add_image(f'image/{prefix}_gt', gt_images.clip(0,1.0), epoch, dataformats='HWC')
        writer.add_image(f'image/{prefix}_pred', pred_images.clip(0,1.0), epoch, dataformats='HWC')


def _rgb_uint8(tensor):
    return tensor.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255


def _mask_uint8(tensor):
    return tensor.detach().float().clamp(0, 1).squeeze().cpu().numpy() * 255


def log_prompt_training_images(
    opt, data, out, epoch, iteration, is_train=True, global_step=None
):
    """Save the complete first-stage prompt segmentation debug panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = "train" if is_train else "eval"
    if global_step is None:
        tag = f"{prefix}_e{epoch:04d}_i{iteration:06d}"
    elif is_train:
        tag = f"{prefix}_step_{global_step:06d}"
    else:
        tag = f"{prefix}_step_{global_step:06d}_i{iteration:03d}"
    output_dir = os.path.join(opt.workspace, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    input_rgb = data["images_input"][0, 0]
    target_rgb = data["images_output"][0, 0]
    rendered_rgb = out["images_pred"][0, 0]
    target_mask = out["target_prompt_mask"][0, 0, 0, 0]
    probability = out["rendered_prompt_probability"][0, 0, 0, 0]
    threshold_mask = probability >= float(opt.prompt_threshold)
    query_rgb = data["query_image"][0]
    query_mask = data["query_mask"][0]

    imageio.imwrite(os.path.join(output_dir, f"{tag}_input_rgb.png"), _rgb_uint8(input_rgb).astype(np.uint8))
    imageio.imwrite(os.path.join(output_dir, f"{tag}_target_rgb.png"), _rgb_uint8(target_rgb).astype(np.uint8))
    imageio.imwrite(os.path.join(output_dir, f"{tag}_rendered_rgb.png"), _rgb_uint8(rendered_rgb).astype(np.uint8))
    imageio.imwrite(os.path.join(output_dir, f"{tag}_target_mask.png"), _mask_uint8(target_mask).astype(np.uint8))
    imageio.imwrite(os.path.join(output_dir, f"{tag}_predicted_probability.png"), _mask_uint8(probability).astype(np.uint8))
    imageio.imwrite(os.path.join(output_dir, f"{tag}_threshold_prediction.png"), _mask_uint8(threshold_mask).astype(np.uint8))
    imageio.imwrite(os.path.join(output_dir, f"{tag}_query_image.png"), _rgb_uint8(query_rgb).astype(np.uint8))
    imageio.imwrite(os.path.join(output_dir, f"{tag}_query_mask.png"), _mask_uint8(query_mask).astype(np.uint8))

    target_np = _rgb_uint8(target_rgb) / 255.0
    probability_np = probability.detach().float().clamp(0, 1).squeeze().cpu().numpy()
    overlay = target_np.copy()
    overlay[..., 0] = np.maximum(overlay[..., 0], probability_np)
    overlay[..., 1:] *= 1.0 - 0.45 * probability_np[..., None]
    imageio.imwrite(
        os.path.join(output_dir, f"{tag}_rgb_mask_overlay.png"),
        (overlay.clip(0, 1) * 255).astype(np.uint8),
    )
    with open(os.path.join(output_dir, f"{tag}_text_prompt.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"positive: {data['positive_text_prompt'][0]}\n")
        handle.write(f"negative: {data['negative_text_prompt'][0]}\n")

    token_scores = out["token_logits"][0, 0].detach().sigmoid().float().cpu().numpy()
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(token_scores, bins=40, range=(0, 1))
    axis.set(xlabel="token foreground probability", ylabel="count", xlim=(0, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{tag}_token_score_histogram.png"), dpi=120)
    plt.close(fig)


def log_semantic_v2_images(
    opt, data, out, epoch, iteration, is_train=True, global_step=None
):
    """Save fixed eight-class targets, probabilities, predictions, and diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = "train" if is_train else "eval"
    if global_step is None:
        tag = f"{prefix}_e{epoch:04d}_i{iteration:06d}"
    elif is_train:
        tag = f"{prefix}_step_{global_step:06d}"
    else:
        tag = f"{prefix}_step_{global_step:06d}_i{iteration:03d}"
    output_dir = os.path.join(opt.workspace, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    input_rgb = data["images_input"][0, 0]
    target_rgb = data["images_output"][0, 0]
    rendered_rgb = out["images_pred"][0, 0]
    imageio.imwrite(
        os.path.join(output_dir, f"{tag}_input_rgb.png"),
        _rgb_uint8(input_rgb).astype(np.uint8),
    )
    imageio.imwrite(
        os.path.join(output_dir, f"{tag}_target_rgb.png"),
        _rgb_uint8(target_rgb).astype(np.uint8),
    )
    imageio.imwrite(
        os.path.join(output_dir, f"{tag}_rendered_rgb.png"),
        _rgb_uint8(rendered_rgb).astype(np.uint8),
    )

    target_np = _rgb_uint8(target_rgb) / 255.0
    class_names = tuple(out["semantic_class_names"])
    token_scores = out["token_probabilities"][0].detach().float().cpu().numpy()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
    for class_index, class_name in enumerate(class_names):
        target_mask = out["target_prompt_mask"][0, class_index, 0, 0]
        probability = out["rendered_prompt_probability"][0, class_index, 0, 0]
        prediction = probability >= float(opt.prompt_threshold)
        stem = f"{tag}_{class_index + 1:02d}_{class_name}"
        imageio.imwrite(
            os.path.join(output_dir, f"{stem}_gt.png"),
            _mask_uint8(target_mask).astype(np.uint8),
        )
        imageio.imwrite(
            os.path.join(output_dir, f"{stem}_probability.png"),
            _mask_uint8(probability).astype(np.uint8),
        )
        imageio.imwrite(
            os.path.join(output_dir, f"{stem}_prediction.png"),
            _mask_uint8(prediction).astype(np.uint8),
        )
        probability_np = probability.detach().float().clamp(0, 1).cpu().numpy()
        overlay = target_np.copy()
        overlay[..., 0] = np.maximum(overlay[..., 0], probability_np)
        overlay[..., 1:] *= 1.0 - 0.45 * probability_np[..., None]
        imageio.imwrite(
            os.path.join(output_dir, f"{stem}_overlay.png"),
            (overlay.clip(0, 1) * 255).astype(np.uint8),
        )
        axis = axes.flat[class_index]
        axis.hist(token_scores[class_index], bins=40, range=(0, 1))
        axis.set_title(class_name)
    fig.supxlabel("token foreground probability")
    fig.supylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{tag}_token_score_histograms.png"), dpi=120)
    plt.close(fig)

    prototype_cosine = out["prototype_cosine_matrix"].detach().float().cpu().numpy()
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(prototype_cosine, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(8), class_names, rotation=45, ha="right")
    axis.set_yticks(range(8), class_names)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{tag}_prototype_cosine.png"), dpi=120)
    plt.close(fig)

    with open(
        os.path.join(output_dir, f"{tag}_semantic_diagnostics.txt"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("prompts: " + ", ".join(class_names) + "\n")
        handle.write(f"temperature: {float(out['temperature']):.6f}\n")
        handle.write(
            f"semantic_token_variance: {float(out['semantic_token_variance']):.8f}\n"
        )
        handle.write(
            "semantic_token_adjacent_cosine: "
            f"{float(out['semantic_token_adjacent_cosine']):.8f}\n"
        )
        handle.write("prototype_cosine_matrix:\n")
        np.savetxt(handle, prototype_cosine, fmt="%.6f")


def train_epoch(opt, accelerator, model, optimizer, scheduler, train_dataloader, 
                iters_per_epoch, epoch, writer, start_time, train_dataset,
                global_step_offset: int = 0,
                sample_skip: int = 0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_psnr = 0
    log_time = time.time()
    epoch_iters = min(iters_per_epoch, opt.max_iters_per_epoch)
    if sample_skip > 0:
        epoch_iters = max(0, epoch_iters - sample_skip)

    def grad_norm(parameters):
        squared = []
        for parameter in parameters:
            if parameter.grad is not None:
                squared.append(parameter.grad.detach().float().square().sum())
        if not squared:
            device = accelerator.device
            return torch.zeros((), device=device)
        return torch.stack(squared).sum().sqrt()

    def train_step(data, global_step, save_after_step):
        """Execute a single training step and return metrics."""
        optimizer.zero_grad()

        global_step_for_aux = global_step
        completed_step = global_step + 1
        unwrapped_model = accelerator.unwrap_model(model)
        if getattr(opt, "model_type", None) == "globalsplat_instance_v2":
            schedule_step = (
                completed_step
                if bool(getattr(opt, "gsi_v2_schedule_uses_completed_step", False))
                else global_step_for_aux
            )
            unwrapped_model.set_train_step(schedule_step)
        if hasattr(unwrapped_model, "compute_lambda_dyn_aux_eff"):
            unwrapped_model.lambda_dyn_aux_eff = unwrapped_model.compute_lambda_dyn_aux_eff(
                global_step_for_aux,
                opt,
            )
        if hasattr(unwrapped_model, "compute_instance_group_lambda_eff"):
            unwrapped_model.instance_group_lambda_eff = (
                unwrapped_model.compute_instance_group_lambda_eff(
                    global_step_for_aux, opt
                )
            )
        if hasattr(unwrapped_model, "compute_teacher_lambda_eff"):
            unwrapped_model.teacher_lambda_eff = (
                unwrapped_model.compute_teacher_lambda_eff(
                    global_step_for_aux, opt
                )
            )
        if hasattr(unwrapped_model, "compute_instance_stage_eff"):
            unwrapped_model.instance_stage_eff = (
                unwrapped_model.compute_instance_stage_eff(
                    global_step_for_aux, opt
                )
            )
        if bool(getattr(opt, "abs_joint_guarded", False)) and hasattr(
            unwrapped_model, "compute_guarded_instance_effs"
        ):
            (
                unwrapped_model.guarded_instance_loss_weight_eff,
                unwrapped_model.guarded_instance_unit_grad_eff,
            ) = unwrapped_model.compute_guarded_instance_effs(
                global_step_for_aux, opt
            )
        if bool(getattr(opt, "abs_true_shared_units", False)) and hasattr(
            unwrapped_model, "compute_tsh_effs"
        ):
            if bool(getattr(opt, "ta_riu_enabled", False)) or bool(
                getattr(opt, "ta_riu_v3_enabled", False)
            ):
                # TA-RIU's only warm-up is the explicit residual/mixer gate;
                # the complete TSH head is trainable from the first forward.
                unwrapped_model.tsh_instance_loss_weight_eff = 1.0
                unwrapped_model.tsh_unit_grad_eff = 1.0
            else:
                (
                    unwrapped_model.tsh_instance_loss_weight_eff,
                    unwrapped_model.tsh_unit_grad_eff,
                ) = unwrapped_model.compute_tsh_effs(
                    global_step_for_aux, opt
                )
        if bool(getattr(opt, "abs_true_shared_units", False)) and hasattr(
            unwrapped_model, "compute_tsh_mbm_u2r_eff"
        ):
            unwrapped_model.tsh_mbm_u2r_eff = (
                unwrapped_model.compute_tsh_mbm_u2r_eff(
                    global_step_for_aux, opt
                )
            )
        if bool(getattr(opt, "tsh_per_gs_refine", False)) and hasattr(
            unwrapped_model, "compute_tsh_per_gs_gate_eff"
        ):
            unwrapped_model.tsh_per_gs_gate_eff = (
                unwrapped_model.compute_tsh_per_gs_gate_eff(
                    global_step_for_aux, opt
                )
            )
        if bool(getattr(opt, "tsh_query_memory_refine", False)) and hasattr(
            unwrapped_model, "compute_tsh_query_memory_refine_eff"
        ):
            unwrapped_model.tsh_query_memory_refine_gate_eff = (
                unwrapped_model.compute_tsh_query_memory_refine_eff(
                    global_step_for_aux, opt
                )
            )
        if str(getattr(opt, "ga_idu_mode", "off")) == "1" and hasattr(
            unwrapped_model, "compute_ga_idu_gate_eff"
        ):
            unwrapped_model.ga_idu_gate_eff = unwrapped_model.compute_ga_idu_gate_eff(
                global_step_for_aux, opt
            )
        if bool(getattr(opt, "ta_riu_enabled", False)) and hasattr(
            unwrapped_model, "compute_ta_riu_gate_eff"
        ):
            ta_gate = unwrapped_model.compute_ta_riu_gate_eff(
                global_step_for_aux, opt
            )
            unwrapped_model.ta_riu_gate_eff = ta_gate
            unwrapped_model.ta_riu_geo_gate_eff = ta_gate
            unwrapped_model.ta_riu_app_gate_eff = ta_gate
        if bool(getattr(opt, "ta_riu_v2_enabled", False)) and hasattr(
            unwrapped_model, "compute_ta_riu_v2_gate_eff"
        ):
            unwrapped_model.ta_riu_v2_gate_eff = (
                unwrapped_model.compute_ta_riu_v2_gate_eff(
                    global_step_for_aux, opt
                )
            )
        if bool(getattr(opt, "ta_riu_v3_enabled", False)) and hasattr(
            unwrapped_model, "compute_ta_riu_v3_gate_eff"
        ):
            unwrapped_model.ta_riu_v3_gate_eff = (
                unwrapped_model.compute_ta_riu_v3_gate_eff(
                    global_step_for_aux, opt
                )
            )

        compute_quality_metrics = (
            save_after_step or completed_step % opt.print_freq == 0
        )
        if getattr(opt, "prompt_training", False):
            out = model(data, compute_quality_metrics=compute_quality_metrics)
        else:
            out = model(data)
        loss = out['loss']
        psnr = out['psnr']

        if (
            getattr(opt, "prompt_overfit_single_batch", False)
            and global_step == 0
            and accelerator.is_main_process
        ):
            log_training_images(
                opt, accelerator, data, out, epoch, 0, writer,
                is_train=True, global_step=0,
            )

        backward_loss = out.get('backward_loss', loss)
        accelerator.backward(backward_loss)

        if (
            bool(getattr(opt, "instance_branch_abs_units", False))
            and accelerator.is_main_process
            and completed_step % max(1, opt.print_freq) == 0
        ):
            base_model = accelerator.unwrap_model(model)

            def _grad_sum(prefixes):
                total = 0.0
                for name, param in base_model.named_parameters():
                    if param.grad is None:
                        continue
                    if any(name.startswith(prefix) for prefix in prefixes):
                        total += float(param.grad.abs().sum())
                return total

            abs_dec_grad = _grad_sum(("absolute_gs_head.",))
            instance_grad = _grad_sum(
                ("instance_branch.", "tsh_instance_head.")
            )
            backbone_grad = _grad_sum(
                (
                    "enc_dec_backbone.",
                    "patch_embed.",
                    "patch_plucker_embed.",
                    "activation_head.",
                    "anchor_pos_encoder.",
                )
            )
            teacher_grad = _grad_sum(("activation_head.",))
            semantic_prefixes = (
                "prompt_matcher.",
                "semantic_lifting_head.",
                "semantic_projector.",
                "prompt_semantic_adapter.",
                "gaussian_feature_head.",
            )
            if not bool(
                getattr(opt, "abs_joint_guarded", False)
            ):
                semantic_prefixes = semantic_prefixes + ("instance_branch.",)
            sem_grad = _grad_sum(semantic_prefixes)
            abs_param_norm = 0.0
            for _name, param in base_model.absolute_gs_head.named_parameters():
                abs_param_norm += float(param.detach().float().abs().sum())
            scene_name = (
                data["scene_name"][0] if "scene_name" in data else "?"
            )
            frame0 = (
                int(data["frame_ids"][0][0])
                if "frame_ids" in data
                else -1
            )
            accelerator.print(
                f"[abs-train] step={completed_step} "
                f"sample={scene_name}:{frame0} "
                f"loss_rgb={float(out['loss_rgb']):.4f} "
                f"teacher_gs={float(out['loss_teacher_gs']):.4f} "
                f"teacher_rgb={float(out['loss_teacher_rgb']):.4f} "
                f"teacher_eff={getattr(base_model, 'teacher_lambda_eff', 0.0):.4f} "
                f"inst={float(out.get('loss_instance_group', 0.0)):.4f} "
                f"inst_eff={getattr(base_model, 'instance_stage_eff', 1.0):.4f} "
                f"inst_w={float(out.get('tsh_instance_loss_weight', out.get('guarded_instance_loss_weight', 0.0))):.6f} "
                f"inst_unit_eff={float(out.get('tsh_unit_grad_eff', out.get('guarded_instance_unit_grad_eff', 0.0))):.4f} "
                f"teacher_called={int(bool(getattr(base_model, 'teacher_called', False)))} "
                f"abs_grad={abs_dec_grad:.2f} abs_norm={abs_param_norm:.2f} "
                f"inst_grad={instance_grad:.2f} "
                f"backbone_grad={backbone_grad:.2f} "
                f"teacher_grad={teacher_grad:.2f} "
                f"sem_grad={sem_grad:.2f} "
                f"gaussians_source=absolute_student old_gs_head_calls=0 "
                f"psnr={float(psnr):.2f}"
            )

        if getattr(opt, "prompt_training", False):
            base_model = accelerator.unwrap_model(model)
            if (
                opt.model_type
                in (
                "semantic_tokengs_v2",
                "semantic_tokengs_v3",
                "semantic_tokengs_v4",
                "semantic_tokengs_v5",
                "semantic_tokengs_v6",
                )
            ):
                groups = base_model.semantic_trainable_groups()
                out["token_adapter_grad_norm"] = grad_norm(groups["token_adapter"])
                out["prompt_adapter_grad_norm"] = grad_norm(groups["prompt_adapter"])
                out["temperature_grad_norm"] = grad_norm(groups["temperature"])
                if "gaussian_feature_head" in groups:
                    out["gaussian_feature_head_grad_norm"] = grad_norm(
                        groups["gaussian_feature_head"]
                    )
                if "instance_group_head" in groups:
                    out["instance_group_head_grad_norm"] = grad_norm(
                        groups["instance_group_head"]
                    )
                adapter_group_names = (
                    "token_adapter", "prompt_adapter", "temperature"
                )
                out["semantic_adapter_grad_norm"] = grad_norm(
                    parameter
                    for name in adapter_group_names
                    for parameter in groups[name]
                )
                if "last_cross_attention" in groups:
                    out["last_cross_attention_grad_norm"] = grad_norm(
                        groups["last_cross_attention"]
                    )
                if "tokengs" in groups:
                    out["tokengs_grad_norm"] = grad_norm(groups["tokengs"])
                    frozen_parameters = ()
                else:
                    frozen_parameters = (
                        parameter
                        for parameter in base_model.parameters()
                        if not parameter.requires_grad
                    )
            elif opt.model_type == "conditional_prompt_tokengs":
                groups = base_model.conditional_trainable_groups()
                out["condition_projection_grad_norm"] = grad_norm(
                    groups["condition_projection"]
                )
                if groups["last_cross_attention"]:
                    out["conditional_cross_attention_grad_norm"] = grad_norm(
                        groups["last_cross_attention"]
                    )
                else:
                    out["conditional_cross_attention_grad_norm"] = torch.zeros(
                        (), device=out["loss"].device
                    )
                out["prompt_decoder_grad_norm"] = grad_norm(
                    parameter
                    for parameters in groups.values()
                    for parameter in parameters
                )
                frozen_parameters = (
                    parameter
                    for parameter in base_model.parameters()
                    if not parameter.requires_grad
                )
            else:
                groups = base_model.prompt_trainable_groups()
                out["prompt_decoder_grad_norm"] = grad_norm(
                    groups["matching_decoder"]
                )
                if "last_cross_attention" in groups:
                    out["last_cross_attention_grad_norm"] = grad_norm(
                        groups["last_cross_attention"]
                    )
                frozen_parameters = (
                    parameter
                    for parameter in base_model.parameters()
                    if not parameter.requires_grad
                )
            if "tokengs_grad_norm" not in out:
                out["tokengs_grad_norm"] = grad_norm(frozen_parameters)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

        optimizer.step()
        if accelerator.sync_gradients:
            scheduler.step()

        if save_after_step and accelerator.is_main_process:
            with torch.inference_mode():
                if getattr(opt, "prompt_training", False):
                    visualization_out = model(data, compute_quality_metrics=False)
                else:
                    visualization_out = model(data)
            log_training_images(
                opt, accelerator, data, visualization_out, epoch, completed_step,
                writer, is_train=True, global_step=completed_step,
            )

        loss_value = loss.detach()
        psnr_value = psnr.detach()
        loss_value_detailed = {}
        for key in (
            "loss_rgb",
            "loss_feat",
            "loss_ce",
            "loss_ce_cosine",
            "loss_bce",
            "loss_dice",
            "loss_boundary_rgb",
            "loss_instance_group",
            "instance_group_gt_count",
            "instance_group_matched_count",
            "instance_group_active_count",
            "instance_group_supervised_view_count",
            "loss_instance_group_dice",
            "loss_instance_group_mask",
            "loss_instance_group_void",
            "loss_instance_group_unmatched",
            "loss_instance_group_ce",
            "loss_instance_group_entropy",
            "loss_mbm_u2r",
            "tsh_mbm_u2r_eff",
            "mbm_u2r_loss_x",
            "mbm_u2r_loss_y",
            "mbm_u2r_interior_share_x",
            "mbm_u2r_interior_share_y",
            "mbm_u2r_valid_share",
            "mbm_u2r_conf_mean",
            "mbm_u2r_depth_has_nan",
            "tsh_per_gs_alpha",
            "tsh_per_gs_unit_logit_diff_mean",
            "tsh_per_gs_unit_logit_diff_max",
            "tsh_per_gs_unit_prob_diff_mean",
            "loss_instance_branch_rgb",
            "loss_instance_branch_rgb_weighted",
            "loss_unit_entropy",
            "loss_unit_compactness",
            "loss_unit_purity",
            "unit_purity_monitor",
            "loss_unit_embedding",
            "unit_embedding_same_sim",
            "unit_embedding_diff_sim",
            "unit_embedding_collapse",
            "unit_embedding_margin",
            "unit_embedding_center_cos_mean",
            "unit_embedding_center_margin",
            "unit_knn_agreement",
            "pseudo_gs_kept_ratio",
            "pseudo_unit_kept_ratio",
            "loss_unit_assignment",
            "unit_assignment_match",
            "unit_assignment_void",
            "unit_assignment_agreement",
            "slot_usage_entropy",
            "loss_dpg_prototype",
            "dpg_proto_similarity",
            "dpg_assignment_accuracy",
            "render_space_pull",
            "render_space_push",
            "render_space_cross",
            "render_space_info_nce",
            "instance_proto_cos",
            "instance_proto_margin",
            "loss_direct_gs_pull",
            "loss_direct_gs_push",
            "loss_direct_gs_cross",
            "direct_gs_proto_cos",
            "direct_gs_proto_margin",
            "direct_gs_pixel_same",
            "direct_gs_pixel_diff",
            "direct_gs_pixel_gap",
            "num_clusters",
            "loss_instance_contrastive",
            "loss_instance_dense_aux",
            "loss_semantic_lifting",
            "mask_iou",
            "foreground_probability",
            "background_probability",
            "predicted_foreground_ratio",
            "gt_foreground_ratio",
            "prompt_decoder_grad_norm",
            "condition_projection_grad_norm",
            "conditional_cross_attention_grad_norm",
            "semantic_adapter_grad_norm",
            "token_adapter_grad_norm",
            "prompt_adapter_grad_norm",
            "gaussian_feature_head_grad_norm",
            "instance_group_head_grad_norm",
            "temperature_grad_norm",
            "last_cross_attention_grad_norm",
            "tokengs_grad_norm",
            "gc2_assignment_entropy",
            "gc2_assignment_max_probability",
            "gc2_assignment_void_share",
            "gc2_active_group_count",
            "gc2_conditioning_residual_ratio",
            "gc2_gaussian_abs_delta",
            "gc2_xyz_abs_delta",
            "gc3_gaussian_assignment_entropy",
            "gc3_gaussian_assignment_void_share",
            "gc4_image_anchor_valid_share",
            "gc4_image_anchor_gate",
            "macro_miou",
            "macc",
            "temperature",
            "semantic_token_variance",
            "semantic_token_adjacent_cosine",
            "prototype_offdiag_cosine_mean",
            "prototype_offdiag_cosine_max",
            "probability_sum",
            "top1_top2_margin",
            "simultaneous_high_ratio",
            "argmax_accuracy",
            "ssim",
            "lpips",
        ):
            if key in out:
                loss_value_detailed[key] = out[key].detach()
        if 'loss_ssim' in out:
            loss_value_detailed['loss_ssim'] = out['loss_ssim'].detach()
        if 'loss_visibility' in out:
            loss_value_detailed['loss_visibility'] = out['loss_visibility'].detach()
        if 'loss_opacity' in out:
            loss_value_detailed['loss_opacity'] = out['loss_opacity'].detach()
        for key in ('loss_dyn_aux', 'psnr_dyn_aux', 'lambda_dyn_aux_eff'):
            if key in out:
                loss_value_detailed[key] = out[key].detach()
        
        return loss_value, loss_value_detailed, psnr_value

    train_dataset.set_rng_epoch(0 if getattr(opt, "prompt_overfit_single_batch", False) else epoch)
    if accelerator.is_main_process:
        print(f"[INFO] Setting RNG epoch to {epoch}")

    iterator = iter(train_dataloader)
    if sample_skip > 0:
        if accelerator.is_main_process:
            print(
                f"[fork] skipping first {sample_skip} local samples "
                f"to continue from step {global_step_offset}"
            )
        for _ in range(sample_skip):
            next(iterator)
        if accelerator.is_main_process:
            print(f"[fork] skip done; {epoch_iters} steps remain in epoch")
    for i in range(epoch_iters):
        data = next(iterator)
        global_step = global_step_offset + i
        completed_step = global_step + 1
        if getattr(opt, "prompt_overfit_single_batch", False):
            save_after_step = completed_step in set(opt.prompt_visualization_steps)
            if global_step == 0 and accelerator.is_main_process:
                scene = data["scene_name"][0]
                frames = [int(value) for value in data["frame_ids"][0].tolist()]
                prompt_class = int(data["prompt_class_id"][0])
                foreground_ratio = float(data["binary_mask_output"][0].float().mean())
                query_scene = data["query_scene_name"][0]
                query_frame = int(data["query_frame_id"][0])
                query_class = int(data["query_class_id"][0])
                accelerator.print(
                    "[prompt-overfit] fixed batch: "
                    f"scene={scene} frames={frames} class={prompt_class} "
                    f"text={data['positive_text_prompt'][0]!r} "
                    f"gt_foreground_ratio={foreground_ratio:.6f} "
                    f"query_scene={query_scene} query_frame={query_frame} "
                    f"query_class={query_class}"
                )
        else:
            save_after_step = (
                opt.log_image_freq > 0
                and accelerator.is_main_process
                and completed_step % opt.log_image_freq == 0
            )
            
        with accelerator.accumulate(model):
            loss_value, loss_value_detailed, psnr_value = train_step(
                data, global_step, save_after_step
            )
            total_loss += loss_value
            total_psnr += psnr_value

            if opt.use_wandb and accelerator.is_main_process:
                writer.add_scalar(f"psnr/train_iteration", psnr_value.item(), completed_step)
                writer.add_scalar(f"loss/train_iteration", loss_value.item(), completed_step)
                writer.add_scalar(f"lr/train_iteration", scheduler.get_last_lr()[0], completed_step)
                try:
                    writer.add_scalar(f"time/train_iteration", time.time() - start_time, completed_step)
                    for key, value in loss_value_detailed.items():
                        writer.add_scalar(f"metrics/{key}/train_iteration", value.item(), completed_step)
                except: 
                    pass

        if bool(getattr(opt, "tsh_ddp8", False)) and os.environ.get(
            "TOKENG_DDP_TRACE", "0"
        ) == "1" and completed_step % max(1, opt.print_freq) == 0:
            trace_dir = os.path.join(opt.workspace, ".ddp_trace")
            os.makedirs(trace_dir, exist_ok=True)
            scene = (
                str(data["scene_name"][0])
                if "scene_name" in data
                else "?"
            )
            rank = int(getattr(accelerator, "process_index", -1))
            with open(
                os.path.join(trace_dir, "rank_samples.txt"),
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    f"rank={rank} step={completed_step} epoch={epoch} "
                    f"scene={scene}\n"
                )
            import hashlib

            unwrapped = accelerator.unwrap_model(model)
            digest = hashlib.sha256()
            hash_prefixes = (
                "absolute_gs_head.",
                "tsh_instance_head.",
            )
            if os.environ.get("TOKENG_DDP_TRACE_FULL_HASH", "0") == "1":
                # GSI-v2 has a reconstruction namespace rather than the
                # historical absolute/TSH head prefixes.  An empty prefix
                # tuple means the complete registered model is hashed.
                hash_prefixes = ()
            if bool(getattr(opt, "tsh_per_gs_refine", False)):
                hash_prefixes = hash_prefixes + (
                    "tsh_slot_refine_head.",
                )
            for name, param in unwrapped.named_parameters():
                if not hash_prefixes or name.startswith(hash_prefixes):
                    digest.update(
                        name.encode()
                        + param.detach().float().cpu().contiguous()
                        .numpy()
                        .tobytes()
                    )
            with open(
                os.path.join(trace_dir, "rank_hashes.txt"),
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    f"rank={rank} step={completed_step} "
                    f"sha={digest.hexdigest()}\n"
                )

        if accelerator.is_main_process:
            # logging
            if completed_step % opt.print_freq == 0:
                if torch.cuda.is_available():
                    mem_free, mem_total = torch.cuda.mem_get_info()
                    memory = f"{(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G"
                else:
                    memory = "cpu"
                elapsed = time.time() - log_time
                speed = opt.print_freq / elapsed if elapsed > 0 else 0
                details = " ".join(
                    f"{key}={value.item():.6f}"
                    for key, value in loss_value_detailed.items()
                )
                if bool(
                    getattr(opt, "abs_joint_guarded", False)
                ) or bool(getattr(opt, "abs_true_shared_units", False)):
                    lr_str = str(
                        [group["lr"] for group in optimizer.param_groups]
                    )
                else:
                    lr_str = f"{scheduler.get_last_lr()[0]:.10f}"
                print(f"[INFO] step={completed_step} epoch={epoch} {i}/{epoch_iters} mem: {memory} lr: {lr_str} loss: {loss_value.item():.6f} {details} psnr={psnr_value.item():.4f} speed: {speed:.2f} it/s")
                log_time = time.time()
        checkpoint_due = (
            (
                bool(getattr(opt, "instance_branch_abs_units", False))
                or getattr(opt, "model_type", None) == "globalsplat_instance_v2"
            )
            and int(getattr(opt, "abs_ckpt_every", 0)) > 0
            and (
                completed_step % int(opt.abs_ckpt_every) == 0
                or completed_step
                in set(int(value) for value in getattr(
                    opt, "abs_ckpt_steps_extra", ()
                ))
            )
        )
        if checkpoint_due and not (
            bool(getattr(opt, "tsh_ddp8", False))
            and bool(getattr(opt, "abs_ckpt_full_state", False))
            and getattr(opt, "model_type", None) == "globalsplat_instance_v2"
        ) and accelerator.is_main_process:
            # Intra-epoch periodic head checkpoints for recovery/staging
            # (optimizer state stays in the workspace-level saves).
            ckpt_dir = os.path.join(opt.workspace, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            from safetensors.torch import save_file

            state = {
                key: value.detach().cpu().contiguous()
                for key, value in accelerator.unwrap_model(model).state_dict().items()
            }
            save_file(
                state,
                os.path.join(
                    ckpt_dir, f"model_step_{completed_step:06d}.safetensors"
                ),
            )
            if bool(getattr(opt, "tsh_ddp8", False)):
                world = int(getattr(accelerator, "num_processes", 1))
                gbs = int(opt.batch_size) * world
                intra_meta = {
                    "optimizer_step": int(completed_step),
                    "equivalent_global_samples": int(completed_step) * gbs,
                    "epoch": int(epoch),
                    "world_size": world,
                    "per_gpu_batch_size": int(opt.batch_size),
                    "global_batch_size": gbs,
                }
                if bool(getattr(opt, "abs_ckpt_full_state", False)):
                    torch.save(
                        optimizer.state_dict(),
                        os.path.join(
                            ckpt_dir,
                            f"optimizer_step_{completed_step:06d}.pth",
                        ),
                    )
                    torch.save(
                        scheduler.state_dict(),
                        os.path.join(
                            ckpt_dir,
                            f"scheduler_step_{completed_step:06d}.pth",
                        ),
                    )
                    config_path = os.path.join(opt.workspace, "config.yaml")
                    if os.path.isfile(config_path):
                        shutil.copy2(
                            config_path,
                            os.path.join(
                                ckpt_dir,
                                f"config_step_{completed_step:06d}.yaml",
                            ),
                        )
                    try:
                        git_commit = subprocess.check_output(
                            ["git", "rev-parse", "HEAD"],
                            cwd=os.getcwd(),
                            text=True,
                        ).strip()
                    except Exception:
                        git_commit = "unknown"
                    intra_meta.update(
                        {
                            "git_commit": git_commit,
                            "config_path": os.path.abspath(config_path),
                            "optimizer_state": os.path.abspath(
                                os.path.join(
                                    ckpt_dir,
                                    f"optimizer_step_{completed_step:06d}.pth",
                                )
                            ),
                            "scheduler_state": os.path.abspath(
                                os.path.join(
                                    ckpt_dir,
                                    f"scheduler_step_{completed_step:06d}.pth",
                                )
                            ),
                            "rng_state_pattern": os.path.abspath(
                                os.path.join(
                                    ckpt_dir,
                                    f"rng_step_{completed_step:06d}_rank*.pth",
                                )
                            ),
                        }
                    )
                with open(
                    os.path.join(
                        ckpt_dir,
                        f"metadata_step_{completed_step:06d}.json",
                    ),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(intra_meta, handle, indent=2)
            print(
                f"[abs-ckpt] saved intra-epoch head checkpoint "
                f"step={completed_step}"
            )
        if checkpoint_due and bool(getattr(opt, "abs_ckpt_full_state", False)) and not (
            bool(getattr(opt, "tsh_ddp8", False))
            and getattr(opt, "model_type", None) == "globalsplat_instance_v2"
        ):
            # Every rank writes its own RNG state.  This is intentionally a
            # sidecar-only operation and does not alter model/loss semantics.
            import random

            ckpt_dir = os.path.join(opt.workspace, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                {
                    "optimizer_step": int(completed_step),
                    "epoch": int(epoch),
                    "rank": int(getattr(accelerator, "process_index", -1)),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state()
                    if torch.cuda.is_available()
                    else None,
                    "python_random": random.getstate(),
                },
                os.path.join(
                    ckpt_dir,
                    f"rng_step_{completed_step:06d}_rank"
                    f"{int(getattr(accelerator, 'process_index', -1)):02d}.pth",
                ),
            )
            accelerator.wait_for_everyone()

        if (
            checkpoint_due
            and bool(getattr(opt, "tsh_ddp8", False))
            and bool(getattr(opt, "abs_ckpt_full_state", False))
            and getattr(opt, "model_type", None) == "globalsplat_instance_v2"
        ):
            save_gsi_v2_intra_epoch_checkpoint_synchronized(
                opt,
                accelerator,
                model,
                optimizer,
                scheduler,
                epoch,
                completed_step,
            )

    total_loss = accelerator.gather_for_metrics(total_loss).mean()
    total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
    
    if accelerator.is_main_process:
        total_loss /= max(1, epoch_iters)
        total_psnr /= max(1, epoch_iters)
        accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")
        if bool(getattr(opt, "instance_branch_abs_units", False)):
            accelerator.print(
                f"[abs-recon-epoch] epoch={epoch} "
                f"samples_this_epoch={epoch_iters} "
                f"cumulative_step={global_step_offset + epoch_iters}"
            )

        if opt.use_wandb:
            writer.add_scalar(f"psnr/train", total_psnr.item(), epoch)
            writer.add_scalar(f"loss/train", total_loss.item(), epoch)


def log_gaussian_histograms(opt, all_gaussians, epoch, writer):
    """Log histograms of Gaussian properties to wandb."""
    # Concatenate all gaussians: [B, N, 14] -> [B*N, 14]
    gaussians = Gaussians.from_raw(torch.cat(all_gaussians, dim=0).reshape(-1, 14))
    
    # Log position histograms (x, y, z separately)
    # Clamp to bin range so out-of-bounds values appear in extreme bins
    pos_bins = torch.linspace(-25, 25, 100)
    for i, label in enumerate("xyz"):
        writer.add_histogram(f'gaussian/pos_{label}', gaussians.xyz[:, i].clamp(-25, 25), global_step=epoch, bins=pos_bins)
    
    # Log opacity histogram [0, 1]
    opacity_bins = torch.linspace(0, 1, 100)
    writer.add_histogram('gaussian/opacity', gaussians.opacity.flatten().clamp(0, 1), global_step=epoch, bins=opacity_bins)
    
    scale_bins = torch.linspace(0, opt.gaussian_scale_cap, 100)
    for i, label in enumerate("xyz"):
        writer.add_histogram(f'gaussian/scale_{label}', gaussians.scaling[:, i].clamp(0, opt.gaussian_scale_cap), global_step=epoch, bins=scale_bins)
    
    # Log rotation histograms [-1, 1] (quaternion components)
    rotation_bins = torch.linspace(-1, 1, 100)
    for i, label in enumerate("wxyz"):
        writer.add_histogram(f'gaussian/rotation_{label}', gaussians.rotation[:, i].clamp(-1, 1), global_step=epoch, bins=rotation_bins)
    
    # Log RGB histograms [0, 1]
    rgb_bins = torch.linspace(0, 1, 100)
    for i, label in enumerate("rgb"):
        writer.add_histogram(f'gaussian/rgb_{label}', gaussians.rgb[:, i].clamp(0, 1), global_step=epoch, bins=rgb_bins)


def _instance_eval_stats(opt, out, data, model, accelerator):
    """Lightweight instance proxy for the training-time eval.

    The instance-group loss is gated by ``self.training`` in the model
    forward, so the training eval never reported any instance signal. This
    recomputes the full-strength (lambda_eff=1) Hungarian-matched BCE+Dice
    loss and the matched/GT ratio on the target views from the rendered
    group probabilities, giving a real-time instance quality readout during
    training. Returns None when the head / labels are unavailable.
    """
    unwrapped = accelerator.unwrap_model(model)
    head = getattr(unwrapped, "instance_group_head", None)
    if head is None:
        head = getattr(unwrapped, "instance_branch", None)
    if head is None:
        return None
    rendered = out.get("rendered_instance_group_probability")
    labels = data.get("instance_label_output")
    if rendered is None or labels is None:
        return None
    if rendered.ndim != 6 or labels.ndim != 4:
        return None
    loss, stats = hungarian_instance_group_loss(
        rendered.float(),
        labels.long(),
        # Void is the last rendered channel; scene-adaptive cluster counts
        # (unit-embedding clustering) vary per scene.
        num_groups=int(rendered.shape[1] - 1),
        min_instance_pixels=int(
            getattr(opt, "instance_group_min_instance_pixels", 32)
        ),
        dice_weight=float(getattr(opt, "lambda_instance_group_dice", 1.0)),
        mask_weight=float(getattr(opt, "lambda_instance_group_mask", 1.0)),
        void_weight=float(getattr(opt, "lambda_instance_group_void", 0.1)),
        unmatched_weight=float(
            getattr(opt, "lambda_instance_group_unmatched", 0.1)
        ),
        lambda_eff=1.0,
        area_alpha=float(getattr(opt, "instance_group_area_alpha", 0.0)),
        match_area_norm=bool(
            getattr(opt, "instance_group_match_area_norm", False)
        ),
        ce_weight=float(getattr(opt, "lambda_instance_group_ce", 0.0)),
        match_topk=int(getattr(opt, "instance_group_match_topk", 1)),
        secondary_pair_weight=float(
            getattr(opt, "instance_group_secondary_pair_weight", 0.3)
        ),
        usage_entropy_weight=float(
            getattr(opt, "instance_group_usage_entropy", 0.0)
        ),
        scene_level_matching=bool(
            getattr(opt, "instance_group_scene_level_matching", False)
        ),
    )
    gt_count = float(stats["instance_group_gt_count"].detach())
    matched_count = float(stats["instance_group_matched_count"].detach())
    return {
        "eval_loss_instance_group": float(loss.detach()),
        "eval_instance_gt_count": gt_count,
        "eval_instance_matched_count": matched_count,
        "eval_instance_match_ratio": (
            matched_count / gt_count if gt_count > 0 else 0.0
        ),
    }


def evaluate_epoch(
    opt, accelerator, model, test_dataloader, epoch, writer, global_step=None
):
    """Evaluate for one epoch."""
    use_input_supervision = opt.use_input_supervision
    opt.use_input_supervision = False
    # The instance-group lambda is updated by train_step (linear warm-up),
    # which would wrongly scale the reported eval instance loss to ~0 during
    # the warm-up phase. Evaluation should always report full-strength
    # losses/metrics, so force lambda_eff = 1.0 here.
    if hasattr(accelerator.unwrap_model(model), "instance_group_lambda_eff"):
        accelerator.unwrap_model(model).instance_group_lambda_eff = 1.0
    if hasattr(accelerator.unwrap_model(model), "teacher_lambda_eff"):
        accelerator.unwrap_model(model).teacher_lambda_eff = 0.0
    if hasattr(accelerator.unwrap_model(model), "instance_stage_eff"):
        accelerator.unwrap_model(model).instance_stage_eff = 1.0
    with torch.inference_mode():
        model.eval()

        total_psnr = 0
        prompt_metric_totals = {}
        class_iou_totals = {}
        class_counts = {}
        class_acc_totals = {}
        mode_iou_totals = {}
        mode_counts = {}
        mode_acc_totals = {}
        semantic_v2_totals = {}
        semantic_v2_vectors = {}
        semantic_v2_prototype_cosine = None
        semantic_v2_region_embeddings = []
        semantic_v2_per_class_data = None
        instance_eval_totals = None
        all_gaussians = []
        for i, data in enumerate(iter(test_dataloader)):
            if opt.max_eval_iters > 0 and i >= opt.max_eval_iters:
                break
            if getattr(opt, "prompt_training", False):
                out = model(data, compute_quality_metrics=True)
            else:
                out = model(data)
            instance_stats = _instance_eval_stats(
                opt, out, data, model, accelerator
            )
            if instance_stats is not None:
                if instance_eval_totals is None:
                    instance_eval_totals = {
                        key: 0.0 for key in instance_stats
                    }
                for key, value in instance_stats.items():
                    instance_eval_totals[key] += value
            psnr = out['psnr']
            total_psnr += psnr.detach()
            if (
                not bool(getattr(opt, "instance_branch_abs_units", False))
                and opt.model_type in (
                "semantic_tokengs_v2",
                "semantic_tokengs_v3",
                "semantic_tokengs_v4",
                "semantic_tokengs_v5",
                "semantic_tokengs_v6",
                )
            ):
                for key in (
                    "loss",
                    "loss_bce",
                    "loss_dice",
                    "loss_instance_group",
                    "instance_group_gt_count",
                    "instance_group_matched_count",
                    "instance_group_active_count",
                    "instance_group_supervised_view_count",
                    "loss_instance_dense_aux",
                    "loss_boundary_rgb",
                    "foreground_probability",
                    "background_probability",
                    "predicted_foreground_ratio",
                    "gt_foreground_ratio",
                    "semantic_token_variance",
                    "semantic_token_adjacent_cosine",
                    "prototype_offdiag_cosine_mean",
                    "prototype_offdiag_cosine_max",
                    "temperature",
                    "probability_sum",
                    "top1_top2_margin",
                    "simultaneous_high_ratio",
                    "argmax_accuracy",
                    "ssim",
                    "lpips",
                ):
                    if key in out:
                        semantic_v2_totals[key] = (
                            semantic_v2_totals.get(key, 0) + out[key].detach()
                        )
                for key in (
                    "class_intersection",
                    "class_union",
                    "class_target_count",
                    "class_predicted_count",
                    "class_valid_count",
                    "class_foreground_probability_sum",
                    "class_background_rejection_sum",
                    "class_background_count",
                    "class_token_score_mean",
                    "class_token_score_std",
                    "class_token_score_min",
                    "class_token_score_max",
                    "argmax_confusion",
                ):
                    semantic_v2_vectors[key] = (
                        semantic_v2_vectors.get(key, 0) + out[key].detach()
                    )
                matrix = out["prototype_cosine_matrix"].detach()
                semantic_v2_prototype_cosine = (
                    matrix
                    if semantic_v2_prototype_cosine is None
                    else semantic_v2_prototype_cosine + matrix
                )
                region_embeddings = out["class_region_embeddings"].detach().cpu()
                target_present = (
                    out["target_prompt_mask"].detach().sum(dim=(2, 3, 4, 5)) > 0
                ).cpu()
                for batch_index, scene_name in enumerate(data["scene_name"]):
                    for class_index in range(8):
                        if bool(target_present[batch_index, class_index]):
                            semantic_v2_region_embeddings.append(
                                (
                                    str(scene_name),
                                    class_index,
                                    region_embeddings[batch_index, class_index],
                                )
                            )
            elif (
                not bool(getattr(opt, "instance_branch_abs_units", False))
                and getattr(opt, "prompt_training", False)
            ):
                for key in (
                    "loss",
                    "loss_bce",
                    "loss_dice",
                    "mask_iou",
                    "mask_accuracy",
                    "foreground_probability",
                    "background_probability",
                    "predicted_foreground_ratio",
                    "gt_foreground_ratio",
                    "ssim",
                    "lpips",
                ):
                    prompt_metric_totals[key] = prompt_metric_totals.get(key, 0) + out[key].detach()
                probabilities = out["rendered_prompt_probability"]
                targets = out["target_prompt_mask"]
                valid_masks = out["valid_mask"]
                for batch_index, class_id_tensor in enumerate(data["prompt_class_id"]):
                    class_id = int(class_id_tensor)
                    prompt_mode = data["prompt_mode"][batch_index]
                    probability = probabilities[batch_index]
                    target = targets[batch_index] >= 0.5
                    valid = valid_masks[batch_index]
                    predicted = probability >= float(opt.prompt_threshold)
                    intersection = (predicted & target & valid).sum().float()
                    union = ((predicted | target) & valid).sum().float()
                    sample_iou = intersection / union.clamp_min(1e-6)
                    target_count = (target & valid).sum().float()
                    sample_acc = intersection / target_count.clamp_min(1e-6)
                    class_iou_totals[class_id] = (
                        class_iou_totals.get(class_id, 0) + sample_iou
                    )
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    class_acc_totals[class_id] = (
                        class_acc_totals.get(class_id, 0) + sample_acc
                    )
                    mode_iou_totals[prompt_mode] = (
                        mode_iou_totals.get(prompt_mode, 0) + sample_iou
                    )
                    mode_counts[prompt_mode] = mode_counts.get(prompt_mode, 0) + 1
                    mode_acc_totals[prompt_mode] = (
                        mode_acc_totals.get(prompt_mode, 0) + sample_acc
                    )
            
            # Collect gaussians for histogram logging
            if accelerator.is_main_process:
                if (
                    not getattr(opt, "prompt_training", False)
                    and getattr(opt, "model_type", None) != "globalsplat_instance_v2"
                ):
                    all_gaussians.append(out['gaussians'].detach().cpu())
            
            # save some images
            if i < int(opt.eval_n_media_dumps):
                log_training_images(
                    opt,
                    accelerator,
                    data,
                    out,
                    epoch,
                    i,
                    writer,
                    is_train=False,
                    global_step=global_step,
                )

        torch.cuda.empty_cache()

        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        eval_count = min(len(test_dataloader), opt.max_eval_iters) if opt.max_eval_iters > 0 else len(test_dataloader)
        evaluation_summary = {}
        if bool(getattr(opt, "instance_branch_abs_units", False)):
            # Absolute student: validation only monitors reconstruction.
            eval_psnr = float(total_psnr) / max(1, eval_count)
            accelerator.print(
                f"[eval] epoch: {epoch} psnr: {eval_psnr:.4f} "
                f"(count={eval_count})"
            )
            return {
                "psnr": eval_psnr,
                "mask_iou": 0.0,
                "loss": 0.0,
            }
        if opt.model_type in (
            "semantic_tokengs_v2",
            "semantic_tokengs_v3",
            "semantic_tokengs_v4",
            "semantic_tokengs_v5",
            "semantic_tokengs_v6",
        ):
            count = max(1, eval_count)
            intersection = semantic_v2_vectors["class_intersection"]
            union = semantic_v2_vectors["class_union"]
            target_count = semantic_v2_vectors["class_target_count"]
            valid_count = semantic_v2_vectors["class_valid_count"]
            predicted_count = semantic_v2_vectors["class_predicted_count"]
            per_class_iou = intersection / union.clamp_min(1e-6)
            per_class_recall = intersection / target_count.clamp_min(1e-6)
            per_class_pred_ratio = predicted_count / valid_count.clamp_min(1e-6)
            per_class_gt_ratio = target_count / valid_count.clamp_min(1e-6)
            semantic_v2_per_class_data = {
                ("wall", "floor", "ceiling", "chair", "table", "sofa", "bed", "other")[class_index]: {
                    "mask_iou": float(per_class_iou[class_index].item()),
                    "mask_acc": float(per_class_recall[class_index].item()),
                }
                for class_index in range(8)
            }
            per_class_fg_probability = (
                semantic_v2_vectors["class_foreground_probability_sum"]
                / target_count.clamp_min(1e-6)
            )
            per_class_bg_probability = (
                semantic_v2_vectors["class_background_rejection_sum"]
                / semantic_v2_vectors["class_background_count"].clamp_min(1e-6)
            )
            evaluation_summary = {
                key: float((value / count).item())
                for key, value in semantic_v2_totals.items()
            }
            evaluation_summary.update(
                {
                    "mask_iou": float(per_class_iou.mean().item()),
                    "macro_miou": float(per_class_iou.mean().item()),
                    "macc": float(per_class_recall.mean().item()),
                    "predicted_foreground_ratio": float(
                        per_class_pred_ratio.mean().item()
                    ),
                    "gt_foreground_ratio": float(per_class_gt_ratio.mean().item()),
                    "foreground_probability": float(
                        per_class_fg_probability.mean().item()
                    ),
                    "background_probability": float(
                        per_class_bg_probability.mean().item()
                    ),
                }
            )
            confusion = semantic_v2_vectors["argmax_confusion"].float()
            argmax_tp = confusion.diagonal()
            argmax_union = (
                confusion.sum(dim=0) + confusion.sum(dim=1) - argmax_tp
            )
            argmax_class_iou = argmax_tp / argmax_union.clamp_min(1e-6)
            argmax_class_recall = argmax_tp / confusion.sum(dim=1).clamp_min(1e-6)
            evaluation_summary.update(
                {
                    "argmax_macro_miou": float(argmax_class_iou.mean().item()),
                    "argmax_macc": float(argmax_class_recall.mean().item()),
                    "argmax_pixel_accuracy": float(
                        argmax_tp.sum().div(confusion.sum().clamp_min(1e-6)).item()
                    ),
                }
            )
        if instance_eval_totals is not None:
            eval_count_instance = (
                min(len(test_dataloader), opt.max_eval_iters)
                if opt.max_eval_iters > 0
                else len(test_dataloader)
            )
            eval_count_instance = max(1, eval_count_instance)
            for key, value in instance_eval_totals.items():
                evaluation_summary[key] = value / eval_count_instance
        elif getattr(opt, "prompt_training", False):
            evaluation_summary = {
                key: float((value / max(1, eval_count)).item())
                for key, value in prompt_metric_totals.items()
            }
            if class_counts:
                class_ids = sorted(class_counts)
                macro_miou = sum(
                    class_iou_totals[class_id] / class_counts[class_id]
                    for class_id in class_ids
                ) / len(class_ids)
                macro_acc = sum(
                    class_acc_totals[class_id] / class_counts[class_id]
                    for class_id in class_ids
                ) / len(class_ids)
                evaluation_summary["macro_miou"] = float(macro_miou.item())
                evaluation_summary["macc"] = float(macro_acc.item())
        if accelerator.is_main_process:
            total_psnr /= max(1, eval_count)
            accelerator.print(f"[eval] epoch: {epoch} psnr: {total_psnr:.4f}")
            if instance_eval_totals is not None:
                step_text = "none" if global_step is None else str(global_step)
                instance_text = " ".join(
                    f"{key}={evaluation_summary[key]:.4f}"
                    for key in (
                        "eval_loss_instance_group",
                        "eval_instance_match_ratio",
                        "eval_instance_gt_count",
                        "eval_instance_matched_count",
                    )
                    if key in evaluation_summary
                )
                accelerator.print(
                    f"[eval-instance] step={step_text} epoch={epoch} "
                    f"{instance_text}"
                )
            if opt.model_type in (
                "semantic_tokengs_v2",
                "semantic_tokengs_v3",
                "semantic_tokengs_v4",
                "semantic_tokengs_v5",
                "semantic_tokengs_v6",
            ):
                step_text = "none" if global_step is None else str(global_step)
                scalar_text = " ".join(
                    f"{key}={value:.6f}"
                    for key, value in evaluation_summary.items()
                    if key not in ("mask_iou",)
                )
                accelerator.print(
                    f"[eval-semantic-v2] step={step_text} epoch={epoch} {scalar_text}"
                )
                class_names = (
                    "wall", "floor", "ceiling", "chair",
                    "table", "sofa", "bed", "other",
                )
                score_mean = semantic_v2_vectors["class_token_score_mean"] / max(
                    1, eval_count
                )
                score_std = semantic_v2_vectors["class_token_score_std"] / max(
                    1, eval_count
                )
                score_min = semantic_v2_vectors["class_token_score_min"] / max(
                    1, eval_count
                )
                score_max = semantic_v2_vectors["class_token_score_max"] / max(
                    1, eval_count
                )
                for class_index, class_name in enumerate(class_names):
                    accelerator.print(
                        f"[eval-v2-class] step={step_text} class_id={class_index + 1} "
                        f"class_name={class_name} iou={per_class_iou[class_index].item():.6f} "
                        f"recall={per_class_recall[class_index].item():.6f} "
                        f"predicted_ratio={per_class_pred_ratio[class_index].item():.6f} "
                        f"gt_ratio={per_class_gt_ratio[class_index].item():.6f} "
                        f"score_mean={score_mean[class_index].item():.6f} "
                        f"score_std={score_std[class_index].item():.6f} "
                        f"score_min={score_min[class_index].item():.6f} "
                        f"score_max={score_max[class_index].item():.6f}"
                    )
                prototype_matrix = (
                    semantic_v2_prototype_cosine / max(1, eval_count)
                ).float().cpu().tolist()
                accelerator.print(
                    f"[eval-v2-prototypes] step={step_text} cosine_matrix="
                    f"{json.dumps(prototype_matrix, separators=(',', ':'))}"
                )
                same_scene_class_cosines = []
                different_class_cosines = []
                for left_index, left in enumerate(semantic_v2_region_embeddings):
                    for right in semantic_v2_region_embeddings[left_index + 1:]:
                        if left[0] == right[0]:
                            continue
                        cosine = float(torch.dot(left[2], right[2]))
                        if left[1] == right[1]:
                            same_scene_class_cosines.append(cosine)
                        else:
                            different_class_cosines.append(cosine)
                same_class_cosine = (
                    float(np.mean(same_scene_class_cosines))
                    if same_scene_class_cosines
                    else float("nan")
                )
                different_class_cosine = (
                    float(np.mean(different_class_cosines))
                    if different_class_cosines
                    else float("nan")
                )
                accelerator.print(
                    f"[eval-v2-cross-scene] step={step_text} "
                    f"same_class_cosine={same_class_cosine:.6f} "
                    f"different_class_cosine={different_class_cosine:.6f} "
                    f"margin={same_class_cosine - different_class_cosine:.6f} "
                    f"same_pairs={len(same_scene_class_cosines)} "
                    f"different_pairs={len(different_class_cosines)}"
                )
                confusion = semantic_v2_vectors["argmax_confusion"].long().cpu()
                for class_index, class_name in enumerate(class_names):
                    accelerator.print(
                        f"[eval-v2-argmax-class] step={step_text} "
                        f"class_id={class_index + 1} class_name={class_name} "
                        f"iou={argmax_class_iou[class_index].item():.6f} "
                        f"recall={argmax_class_recall[class_index].item():.6f}"
                    )
                accelerator.print(
                    f"[eval-v2-argmax] step={step_text} "
                    f"macro_miou={evaluation_summary['argmax_macro_miou']:.6f} "
                    f"macc={evaluation_summary['argmax_macc']:.6f} "
                    f"pixel_accuracy={evaluation_summary['argmax_pixel_accuracy']:.6f}"
                )
                accelerator.print(
                    f"[eval-v2-confusion] step={step_text} matrix="
                    f"{json.dumps(confusion.tolist(), separators=(',', ':'))}"
                )
            elif getattr(opt, "prompt_training", False):
                prompt_summary = " ".join(
                    f"{key}={(value / max(1, eval_count)).item():.6f}"
                    for key, value in prompt_metric_totals.items()
                )
                step_text = "none" if global_step is None else str(global_step)
                accelerator.print(
                    f"[eval-prompt] step={step_text} epoch={epoch} {prompt_summary}"
                )
                class_names = (
                    "wall", "floor", "ceiling", "chair",
                    "table", "sofa", "bed", "other",
                )
                for class_id in sorted(class_counts):
                    class_iou = class_iou_totals[class_id] / class_counts[class_id]
                    class_acc = class_acc_totals[class_id] / class_counts[class_id]
                    accelerator.print(
                        f"[eval-class] step={step_text} class_id={class_id} "
                        f"class_name={class_names[class_id - 1]} "
                        f"count={class_counts[class_id]} mask_iou={class_iou.item():.6f} "
                        f"mask_acc={class_acc.item():.6f}"
                    )
                for prompt_mode in sorted(mode_counts):
                    mode_iou = mode_iou_totals[prompt_mode] / mode_counts[prompt_mode]
                    mode_acc = mode_acc_totals[prompt_mode] / mode_counts[prompt_mode]
                    accelerator.print(
                        f"[eval-mode] step={step_text} prompt_mode={prompt_mode} "
                        f"count={mode_counts[prompt_mode]} mask_iou={mode_iou.item():.6f} "
                        f"mask_acc={mode_acc.item():.6f}"
                    )
                if class_counts:
                    accelerator.print(
                        f"[eval-prompt-macro] step={step_text} epoch={epoch} "
                        f"macro_miou={evaluation_summary['macro_miou']:.6f} "
                        f"macc={evaluation_summary['macc']:.6f} "
                        f"mask_accuracy={evaluation_summary['mask_accuracy']:.6f}"
                    )

            if opt.use_wandb:
                writer.add_scalar(f"psnr/eval", total_psnr.item(), epoch)
                
                # Log Gaussian property histograms
                if len(all_gaussians) > 0:
                    log_gaussian_histograms(opt, all_gaussians, epoch, writer)

    if accelerator.is_main_process and getattr(opt, "evaluating", False):
        metrics_payload = {
            "evaluation_summary": {
                key: float(value)
                for key, value in evaluation_summary.items()
            },
            "psnr": float(total_psnr.item()),
            "num_samples": int(eval_count),
            "per_class": {},
            "per_mode": {},
        }
        if semantic_v2_per_class_data is not None:
            metrics_payload["per_class"] = semantic_v2_per_class_data
        if class_counts:
            metrics_payload["per_class"] = {
                class_names[class_id - 1]: {
                    "count": int(class_counts[class_id]),
                    "mask_iou": float(
                        (class_iou_totals[class_id] / class_counts[class_id]).item()
                    ),
                    "mask_acc": float(
                        (class_acc_totals[class_id] / class_counts[class_id]).item()
                    ),
                }
                for class_id in sorted(class_counts)
            }
        if mode_counts:
            metrics_payload["per_mode"] = {
                prompt_mode: {
                    "count": int(mode_counts[prompt_mode]),
                    "mask_iou": float(
                        (mode_iou_totals[prompt_mode] / mode_counts[prompt_mode]).item()
                    ),
                    "mask_acc": float(
                        (mode_acc_totals[prompt_mode] / mode_counts[prompt_mode]).item()
                    ),
                }
                for prompt_mode in sorted(mode_counts)
            }
        metrics_path = os.path.join(opt.workspace, "eval_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, indent=2)
        accelerator.print(
            f"[eval-metrics] saved to {os.path.abspath(metrics_path)}"
        )

    opt.use_input_supervision = use_input_supervision
    return evaluation_summary


def main():    
    start_time = time.time()
    opt = tyro.cli(AllConfigs)

    if getattr(opt, "model_type", None) == "globalsplat_instance_v2":
        if getattr(opt, "gsi_v2_recon_loss_mode", "mse_smoke") == "official_vgg":
            vgg_path = os.path.abspath(opt.gsi_v2_vgg_weight_path)
            if not os.path.isfile(vgg_path) or os.path.getsize(vgg_path) <= 0:
                raise RuntimeError(f"[gsi-v2] formal training is blocked: explicit VGG asset missing at {vgg_path}")

    torch.manual_seed(opt.seed)

    ddp_kwargs = None
    if (
        bool(getattr(opt, "abs_true_shared_units", False))
        and (
            str(getattr(opt, "tsh_mbm_mode", "off")) != "off"
            or float(getattr(opt, "tsh_mbm_decoder_tail_lr", 0.0)) > 0.0
        )
    ):
        # The SIU3R-MBM forward runs the frozen old-GS teacher inside a
        # no_grad block and the shared decoder tail a second time with
        # autograd inside the same DDP forward.  DDP's default reducer
        # mis-detects those decoder-tail parameters as unused; the
        # find_unused_parameters pass resolves it (trainable params are
        # still used every step; legacy m0/m4 runs are unaffected).
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=True
        )

    if getattr(opt, "model_type", None) == "globalsplat_instance_v2":
        # GSI-v2 has deliberately gated branches during Phase-J warm-up:
        # instance-to-reconstruction paths are unused while their gate is
        # zero.  Match the validated preflight reducer configuration so DDP
        # handles those intentional per-step unused parameters without
        # changing any forward, loss, or optimizer semantics.
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=True
        )

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[ddp_kwargs] if ddp_kwargs is not None else None,
    )

    # Setup workspace and status
    setup_workspace_and_status(opt, accelerator)

    # Load checkpoint and resume
    epoch_start, wandb_run_id = load_checkpoint_and_resume(opt, accelerator)

    # Setup wandb
    wandb_run_id, writer = setup_wandb(opt, accelerator, epoch_start, wandb_run_id)
    
    if accelerator.is_main_process:
        print(opt)
        if getattr(opt, "prompt_training", False):
            print("[prompt-loss] BCE function: torch.nn.functional.binary_cross_entropy")
            print(
                "[prompt-loss] BCE input: rendered sigmoid probability clamped to "
                "[eps, 1-eps]; target is a binary mask (BCEWithLogitsLoss is not used)"
            )
            if opt.model_type in (
                "semantic_tokengs_v2",
                "semantic_tokengs_v3",
                "semantic_tokengs_v4",
                "semantic_tokengs_v5",
                "semantic_tokengs_v6",
            ):
                bce_reduction = (
                    "per-class positive/negative balanced mean"
                    if opt.semantic_v2_balanced_bce
                    else "valid-pixel mean"
                )
                print(f"[prompt-loss] Semantic V2 BCE reduction: {bce_reduction}")
                print(
                    "[semantic-v2] token score normalization: "
                    f"{opt.semantic_v2_score_mode}"
                )
                print(
                    "[semantic-v2] semantic-only final cross-attention tuning: "
                    f"{opt.semantic_v2_tune_last_cross_attention}"
                )
            elif opt.model_type == "conditional_prompt_tokengs":
                print(
                    "[conditional-v3] copied final cross-attention tuning: "
                    f"{opt.conditional_v3_tune_last_cross_attention}"
                )

        config_save_path = os.path.join(opt.workspace, "config.yaml")
        if not os.path.exists(config_save_path):
            with open(config_save_path, "w") as f:
                f.write(tyro.extras.to_yaml(opt))
            print(f"[INFO] Config saved to {config_save_path=}")

    # model
    apply_checkpoint_architecture(opt)
    model = model_registry[opt.model_type](opt)

    # Load model checkpoint
    load_model_checkpoint(opt, model, accelerator, epoch_start)
    load_fresh_backbone_resume(opt, model, accelerator)
    load_semantic_adapter_resume(opt, model, accelerator)
    
    # Data
    train_dataloader, test_dataloader, train_dataset, test_dataset = get_multi_dataloader(opt, accelerator)

    # Optimizer (before prepare)
    optimizer = setup_optimizer(opt, model, accelerator, epoch_start)

    # accelerate (shards dataloader across GPUs)
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    # Compute per-GPU iterations from the prepared (sharded) dataloader
    iters_per_epoch = min(len(train_dataloader), opt.max_iters_per_epoch)
    if bool(getattr(opt, "tsh_ddp8", False)):
        accelerator.print(
            f"[ddp8] rank={getattr(accelerator, 'process_index', -1)} "
            f"world={int(getattr(accelerator, 'num_processes', 1))} "
            f"local_dataloader_len={len(train_dataloader)} "
            f"iters_per_epoch={iters_per_epoch} "
            f"train_dataset_len={len(train_dataset)}"
        )

    if opt.evaluating:
        os.makedirs(opt.workspace, exist_ok=True)
        evaluate_epoch(
            opt,
            accelerator,
            model,
            test_dataloader,
            epoch=-1,
            writer=writer,
            global_step=0,
        )
        accelerator.wait_for_everyone()
        return

    # Scheduler (after prepare, so iters_per_epoch is correct)
    # NOTE: do NOT call accelerator.prepare(scheduler) here -- total_steps is already
    # computed from the per-GPU iters_per_epoch, and prepare() would divide it again
    # by num_processes, causing the LR to decay to zero far too early.
    scheduler = setup_scheduler(opt, optimizer, iters_per_epoch, accelerator, epoch_start)

    # loop
    os.makedirs(opt.workspace, exist_ok=True)

    if (
        getattr(opt, "model_type", None) == "globalsplat_instance_v2"
        and epoch_start == 0
        and not os.path.exists(os.path.join(opt.workspace, "checkpoints", "model_step_000000.safetensors"))
    ):
        save_gsi_v2_step0_snapshot(opt, accelerator, model, optimizer, scheduler)
        if accelerator.is_main_process:
            print("[gsi-v2] saved explicit optimizer-step-0 snapshot")

    if (
        not getattr(opt, "prompt_overfit_single_batch", False)
        and opt.eval_before_training
        and not bool(getattr(opt, "gsi_v2_disable_training_eval", False))
    ):
        evaluate_epoch(
            opt,
            accelerator,
            model,
            test_dataloader,
            epoch_start - 1,
            writer,
            global_step=epoch_start * iters_per_epoch,
        )
    
    best_prompt_iou = float("-inf")
    best_metadata_path = os.path.join(opt.workspace, "metadata_best.json")
    if os.path.isfile(best_metadata_path):
        with open(best_metadata_path, "r", encoding="utf-8") as handle:
            best_metadata = json.load(handle)
            best_prompt_iou = float(
                best_metadata.get(
                    "selection_iou", best_metadata.get("mask_iou", float("-inf"))
                )
            )

    epoch = epoch_start
    fork_step = int(getattr(opt, "tsh_fork_continue_step", 0))
    fork_pending = fork_step > 0
    fork_epoch = fork_step // max(1, iters_per_epoch)
    fork_step_in_epoch = fork_step % max(1, iters_per_epoch)
    if fork_pending:
        # Resume from an arbitrary optimizer step.  The earlier fork logic
        # only handled step<iters_per_epoch; step1000 is epoch 1, batch 290
        # for the 710-step R1 schedule.
        epoch_start = fork_epoch
        epoch = fork_epoch
        load_fork_rng_state(opt, accelerator)
    while epoch < opt.num_epochs:
        # train
        offset = (
            fork_step
            if fork_pending
            else epoch * iters_per_epoch
        )
        sample_skip = fork_step_in_epoch if fork_pending else 0
        train_epoch(opt, accelerator, model, optimizer, scheduler, train_dataloader, 
                    iters_per_epoch, epoch, writer, start_time, train_dataset,
                    global_step_offset=offset,
                    sample_skip=sample_skip)
        fork_pending = False
        
        # checkpoint
        global_step = (epoch + 1) * iters_per_epoch
        save_checkpoint(
            opt,
            accelerator,
            model,
            optimizer,
            scheduler,
            epoch,
            wandb_run_id,
            global_step,
        )
        save_fork_rng_state(opt, accelerator, epoch, global_step)

        # eval
        if (
            not getattr(opt, "prompt_overfit_single_batch", False)
            and not bool(getattr(opt, "gsi_v2_disable_training_eval", False))
        ):
            validation_metrics = evaluate_epoch(
                opt,
                accelerator,
                model,
                test_dataloader,
                epoch,
                writer,
                global_step=global_step,
            )
            if getattr(opt, "prompt_training", False):
                if bool(
                    getattr(opt, "instance_branch_abs_units", False)
                ):
                    # Absolute student: select the best checkpoint by
                    # reconstruction PSNR, not semantic mask IoU.
                    selection_metric = "psnr"
                else:
                    selection_metric = (
                        "argmax_macro_miou"
                        if opt.model_type in (
                            "semantic_tokengs_v2",
                            "semantic_tokengs_v3",
                            "semantic_tokengs_v4",
                            "semantic_tokengs_v5",
                            "semantic_tokengs_v6",
                        )
                        and opt.semantic_v2_score_mode == "softmax"
                        else "mask_iou"
                    )
                current_iou = float(validation_metrics[selection_metric])
                is_best = current_iou > best_prompt_iou
                save_prompt_validation_checkpoint(
                    opt,
                    accelerator,
                    model,
                    epoch,
                    global_step,
                    validation_metrics,
                    is_best,
                )
                if is_best:
                    best_prompt_iou = current_iou

        epoch += 1
            
    # If we get here, we've completed all epochs
    # Signal successful completion
    if accelerator.is_main_process:
        status_dir = os.path.join(opt.workspace, "status")
        with open(os.path.join(status_dir, "COMPLETE"), "w") as f:
            f.write(f"Training completed successfully after {opt.num_epochs} epochs")
        print(f"[INFO] Training completed successfully after {opt.num_epochs} epochs")
    
    accelerator.wait_for_everyone()
    return


if __name__ == "__main__":
    main()
