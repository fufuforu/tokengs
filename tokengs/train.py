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
from dataclasses import asdict

import torch
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, DataLoaderConfiguration
from safetensors.torch import load_file, save_file

import imageio
import numpy as np

from tokengs.options import AllConfigs
from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry

import warnings

from tokengs.utils.gaussians import Gaussians
warnings.filterwarnings("ignore")


def setup_workspace_and_status(opt, accelerator):
    """Setup workspace directory and check for existing completion."""
    status_dir = os.path.join(opt.workspace, "status")
    complete_file = os.path.join(status_dir, "COMPLETE")
    
    if accelerator.is_main_process:
        os.makedirs(status_dir, exist_ok=True)
        if os.path.exists(complete_file):
            raise RuntimeError(f"Found existing COMPLETE file at {complete_file}, remove it if you want to run the job again")


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


def load_model_checkpoint(opt, model, accelerator, epoch_start):
    """Load model checkpoint with tolerance for shape mismatches."""
    if opt.resume is None or opt.resume == 'None':
        return
    
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')

    if getattr(opt, "prompt_training", False):
        checkpoint_label = (
            "SemanticTokenGSv2"
            if opt.model_type == "semantic_tokengs_v2"
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
        state_dict = model.state_dict()
        missing = sorted(set(state_dict) - set(ckpt))
        unexpected = sorted(set(ckpt) - set(state_dict))
        mismatched = sorted(
            (key, tuple(ckpt[key].shape), tuple(state_dict[key].shape))
            for key in set(state_dict) & set(ckpt)
            if ckpt[key].shape != state_dict[key].shape
        )
        accelerator.print(f"[{checkpoint_label}] resume missing keys: {missing}")
        accelerator.print(f"[{checkpoint_label}] resume unexpected keys: {unexpected}")
        accelerator.print(f"[{checkpoint_label}] resume shape-mismatched keys: {mismatched}")
        if missing or unexpected or mismatched:
            raise RuntimeError("Prompt checkpoint failed strict validation")
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


def setup_optimizer(opt, model, accelerator, epoch_start):
    """Setup optimizer. Call before accelerator.prepare()."""
    decay_params, nodecay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() == 1 or getattr(param, '_no_weight_decay', False):
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = []
    if len(decay_params) > 0:
        optim_groups.append({'params': decay_params, 'weight_decay': opt.weight_decay})
    if len(nodecay_params) > 0:
        optim_groups.append({'params': nodecay_params, 'weight_decay': 0.0})

    optimizer = torch.optim.AdamW(optim_groups, lr=opt.lr, betas=(0.9, 0.95), fused=True)

    if epoch_start > 0:
        optimizer.load_state_dict(torch.load(os.path.join(opt.workspace, 'optimizer.pth'), map_location='cpu'))

    return optimizer


def setup_scheduler(opt, optimizer, iters_per_epoch, accelerator, epoch_start):
    """Setup scheduler. Call after accelerator.prepare() with per-GPU iters_per_epoch."""
    if opt.lr_scheduler == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        if epoch_start > 0:
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
        if opt.model_type in ("prompt_tokengs", "conditional_prompt_tokengs"):
            metadata["prompt_image_pooling"] = opt.prompt_image_pooling
            metadata["conditional_query_decoder"] = (
                opt.model_type == "conditional_prompt_tokengs"
            )
            if opt.model_type == "prompt_tokengs":
                metadata["prompt_tune_last_cross_attention"] = bool(
                    opt.prompt_tune_last_cross_attention
                )
            if opt.model_type == "conditional_prompt_tokengs":
                metadata["conditional_v3_tune_last_cross_attention"] = bool(
                    opt.conditional_v3_tune_last_cross_attention
                )
        elif opt.model_type == "semantic_tokengs_v2":
            unwrapped = accelerator.unwrap_model(model)
            metadata["semantic_v2_dim"] = int(opt.semantic_v2_dim)
            metadata["semantic_v2_balanced_bce"] = bool(opt.semantic_v2_balanced_bce)
            metadata["semantic_v2_score_mode"] = opt.semantic_v2_score_mode
            metadata["semantic_v2_tune_last_cross_attention"] = bool(
                opt.semantic_v2_tune_last_cross_attention
            )
            selection_metric = (
                "argmax_macro_miou"
                if opt.semantic_v2_score_mode == "softmax"
                else "mask_iou"
            )
            metadata["selection_metric"] = selection_metric
            metadata["selection_iou"] = float(validation_metrics[selection_metric])
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
        if opt.model_type == "semantic_tokengs_v2":
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
                iters_per_epoch, epoch, writer, start_time, train_dataset):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_psnr = 0
    log_time = time.time()

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
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, "compute_lambda_dyn_aux_eff"):
            unwrapped_model.lambda_dyn_aux_eff = unwrapped_model.compute_lambda_dyn_aux_eff(
                global_step_for_aux,
                opt,
            )

        completed_step = global_step + 1
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

        if getattr(opt, "prompt_training", False):
            base_model = accelerator.unwrap_model(model)
            if opt.model_type == "semantic_tokengs_v2":
                groups = base_model.semantic_trainable_groups()
                out["token_adapter_grad_norm"] = grad_norm(groups["token_adapter"])
                out["prompt_adapter_grad_norm"] = grad_norm(groups["prompt_adapter"])
                out["temperature_grad_norm"] = grad_norm(groups["temperature"])
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
            "loss_bce",
            "loss_dice",
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
            "temperature_grad_norm",
            "last_cross_attention_grad_norm",
            "tokengs_grad_norm",
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

    for i, data in enumerate(iter(train_dataloader)):
        if i >= opt.max_iters_per_epoch:
            break

        global_step = epoch * iters_per_epoch + i
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
            save_after_step = accelerator.is_main_process and (
                completed_step % opt.log_image_freq == 0
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
                print(f"[INFO] step={completed_step} epoch={epoch} {i}/{iters_per_epoch} mem: {memory} lr: {scheduler.get_last_lr()[0]:.10f} loss: {loss_value.item():.6f} {details} psnr={psnr_value.item():.4f} speed: {speed:.2f} it/s")
                log_time = time.time()

    total_loss = accelerator.gather_for_metrics(total_loss).mean()
    total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
    
    if accelerator.is_main_process:
        total_loss /= iters_per_epoch
        total_psnr /= iters_per_epoch
        accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")

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


def evaluate_epoch(
    opt, accelerator, model, test_dataloader, epoch, writer, global_step=None
):
    """Evaluate for one epoch."""
    use_input_supervision = opt.use_input_supervision
    opt.use_input_supervision = False
    with torch.inference_mode():
        model.eval()

        total_psnr = 0
        prompt_metric_totals = {}
        class_iou_totals = {}
        class_counts = {}
        mode_iou_totals = {}
        mode_counts = {}
        semantic_v2_totals = {}
        semantic_v2_vectors = {}
        semantic_v2_prototype_cosine = None
        semantic_v2_region_embeddings = []
        all_gaussians = []
        for i, data in enumerate(iter(test_dataloader)):
            if opt.max_eval_iters > 0 and i >= opt.max_eval_iters:
                break
            if getattr(opt, "prompt_training", False):
                out = model(data, compute_quality_metrics=True)
            else:
                out = model(data)
            psnr = out['psnr']
            total_psnr += psnr.detach()
            if opt.model_type == "semantic_tokengs_v2":
                for key in (
                    "loss",
                    "loss_bce",
                    "loss_dice",
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
            elif getattr(opt, "prompt_training", False):
                for key in (
                    "loss",
                    "loss_bce",
                    "loss_dice",
                    "mask_iou",
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
                    class_iou_totals[class_id] = (
                        class_iou_totals.get(class_id, 0) + sample_iou
                    )
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    mode_iou_totals[prompt_mode] = (
                        mode_iou_totals.get(prompt_mode, 0) + sample_iou
                    )
                    mode_counts[prompt_mode] = mode_counts.get(prompt_mode, 0) + 1
            
            # Collect gaussians for histogram logging
            if accelerator.is_main_process:
                if not getattr(opt, "prompt_training", False):
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
        if opt.model_type == "semantic_tokengs_v2":
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
        elif getattr(opt, "prompt_training", False):
            evaluation_summary = {
                key: float((value / max(1, eval_count)).item())
                for key, value in prompt_metric_totals.items()
            }
        if accelerator.is_main_process:
            total_psnr /= max(1, eval_count)
            accelerator.print(f"[eval] epoch: {epoch} psnr: {total_psnr:.4f}")
            if opt.model_type == "semantic_tokengs_v2":
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
                    accelerator.print(
                        f"[eval-class] step={step_text} class_id={class_id} "
                        f"class_name={class_names[class_id - 1]} "
                        f"count={class_counts[class_id]} mask_iou={class_iou.item():.6f}"
                    )
                for prompt_mode in sorted(mode_counts):
                    mode_iou = mode_iou_totals[prompt_mode] / mode_counts[prompt_mode]
                    accelerator.print(
                        f"[eval-mode] step={step_text} prompt_mode={prompt_mode} "
                        f"count={mode_counts[prompt_mode]} mask_iou={mode_iou.item():.6f}"
                    )

            if opt.use_wandb:
                writer.add_scalar(f"psnr/eval", total_psnr.item(), epoch)
                
                # Log Gaussian property histograms
                if len(all_gaussians) > 0:
                    log_gaussian_histograms(opt, all_gaussians, epoch, writer)

    opt.use_input_supervision = use_input_supervision
    return evaluation_summary


def main():    
    start_time = time.time()
    opt = tyro.cli(AllConfigs)

    torch.manual_seed(opt.seed)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
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
            if opt.model_type == "semantic_tokengs_v2":
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
    model = model_registry[opt.model_type](opt)

    # Load model checkpoint
    load_model_checkpoint(opt, model, accelerator, epoch_start)
    
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
        not getattr(opt, "prompt_overfit_single_batch", False)
        and opt.eval_before_training
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
    while epoch < opt.num_epochs:
        # train
        train_epoch(opt, accelerator, model, optimizer, scheduler, train_dataloader, 
                    iters_per_epoch, epoch, writer, start_time, train_dataset)
        
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

        # eval
        if not getattr(opt, "prompt_overfit_single_batch", False):
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
                selection_metric = (
                    "argmax_macro_miou"
                    if opt.model_type == "semantic_tokengs_v2"
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
