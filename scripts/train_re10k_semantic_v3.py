"""Train the V3 token_decoder_lowrank semantic field on RE10K (2-view).

Reproduces the tokengs_c3g recipe that reached ~0.536 ScanNet C3G8 mIoU:
frozen TokenGS RE10K 2-view backbone + token-compressed semantic field,
supervised by cosine feature distillation against the LSeg teacher on the
target and source views.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import tyro
from accelerate import Accelerator
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.input_types import split_data
from tokengs.options import config_defaults


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--ckpt_freq", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--index_file",
        default="/space0/mawb/tokengs/workspace/re10k_index.json",
        help="Cache the RE10K scene index (first run scans ~14min).",
    )
    args = parser.parse_args()

    accelerator = Accelerator()
    opt = config_defaults["eval_scannet_lsm_instance"]
    opt.data_mode = (("re10k", 1),)
    opt.num_input_views = 2
    opt.num_views = 3
    opt.batch_size = 1
    opt.num_workers = args.num_workers
    opt.evaluating = False
    opt.dataset_kwargs = {"index_file": args.index_file}

    # V3 token_decoder_lowrank recipe (tokengs_c3g re10k_semantic_lseg_v3_token).
    opt.semantic_branch_version = "token_decoder_lowrank"
    opt.lambda_semantic_feature = 1.0
    opt.lambda_semantic_cosine = 1.0
    opt.lambda_semantic_l1 = 0.0
    opt.lambda_semantic_ce = 0.0
    opt.semantic_feature_use_alpha_mask = False
    opt.semantic_stream_compressed_features = True
    opt.semantic_render_scale = 0.5
    opt.semantic_render_chunk = 32
    opt.semantic_detach_tokens = True
    opt.semantic_detach_geometry = True
    opt.semantic_use_depth_filter = False
    opt.prompt_tokengs_checkpoint = (
        "/space0/mawb/tokengs/workspace/tokengs_re10k/model.safetensors"
    )
    opt.prompt_unfreeze_tokengs = False
    opt.use_instance_labels = False
    opt.lambda_instance_group = 0.0

    model = model_registry[opt.model_type](opt)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.semantic_head.parameters():
        parameter.requires_grad_(True)
    # Match the tokengs_c3g optimizer: weight decay only on 2-D parameters
    # (biases / LayerNorm weights are excluded), betas (0.9, 0.95).
    decay_params = []
    no_decay_params = []
    for name, parameter in model.semantic_head.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dim() <= 1:
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)
    trainable = decay_params + no_decay_params
    print(
        "[re10k-v3] trainable semantic params:",
        sum(parameter.numel() for parameter in trainable),
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.05},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr, args.lr],
        total_steps=args.num_steps,
        pct_start=args.warmup_steps / max(1, args.num_steps),
        final_div_factor=1000.0,
    )

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "config.yaml"
    if not config_path.exists():
        config_path.write_text(tyro.extras.to_yaml(opt), encoding="utf-8")
    checkpoint_dir = workspace / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    _, loader, _, _ = get_multi_dataloader(opt, accelerator)
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    model.train()

    iterator = iter(loader)
    total_loss = 0.0
    for step in range(1, args.num_steps + 1):
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            data = next(iterator)
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        model_input, _ = split_data(data, opt)
        optimizer.zero_grad()
        with torch.no_grad():
            reconstruction, gs_token_hidden, _ = (
                model._forward_prompt_reconstruction(model_input)
            )
        loss = model.compute_semantic_v3_loss(
            data,
            reconstruction.gaussians,
            gs_token_hidden,
            model_input,
            bg_color=reconstruction.background_color,
        )
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.detach())
        if step % args.log_freq == 0 and accelerator.is_main_process:
            print(
                f"[re10k-v3] step={step}/{args.num_steps} "
                f"loss={total_loss / args.log_freq:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
            total_loss = 0.0
        if step % args.ckpt_freq == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            save_file(
                unwrapped.state_dict(),
                str(checkpoint_dir / f"model_step_{step:06d}.safetensors"),
            )
            print(f"[re10k-v3] saved model_step_{step:06d}.safetensors")
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_file(unwrapped.state_dict(), str(workspace / "model.safetensors"))
        print("[re10k-v3] done")


if __name__ == "__main__":
    main()
