"""Standalone training for the InstanceSplat-style direct per-GS instance
embedding (bypasses train.py's accelerate/DDP integration, which was found
to save non-discriminative weights despite training logs showing separation).

The frozen TokenGS backbone + frozen DINO extractor stay untouched; only
``direct_gs_head`` (plus its output scale) trains with rendered-space
supervision (pull / push / cross-view / pixel InfoNCE).  Checkpoints are
saved directly with safetensors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.input_types import split_data
from tokengs.options import config_defaults

from eval_instance_lsm_protocol import _LocalAccelerator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--num_steps", type=int, default=3000)
    parser.add_argument("--ckpt_freq", type=int, default=200)
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iters_per_epoch", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--data-mode",
        default="scannet_prompt_small",
        choices=("scannet_prompt_small", "scannet_lsm_style_train"),
        help=(
            "scannet_prompt_small = close-frame 64-scene manifest; "
            "scannet_lsm_style_train = wide-gap 8+7 interleaved windows "
            "over the ScanNet train set (matches the LSM eval distribution)."
        ),
    )
    parser.add_argument("--info_nce_weight", type=float, default=0.3)
    parser.add_argument("--info_temp", type=float, default=0.3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = config_defaults["semantic_v6_direct_gs_train"]
    opt.workspace = args.workspace
    opt.experiment_name = out_dir.name
    opt.max_iters_per_epoch = args.max_iters_per_epoch
    opt.num_epochs = (args.num_steps + args.max_iters_per_epoch - 1) // (
        args.max_iters_per_epoch
    )
    opt.evaluating = False
    if args.data_mode == "scannet_lsm_style_train":
        opt.data_mode = (("scannet_lsm_style_train", 1),)
        opt.dataset_kwargs = {"windows_per_scene": 16}
    else:
        opt.data_mode = (("scannet_prompt_small", 1),)
        opt.dataset_kwargs = {
            "small_manifest_path": (
                "/space0/mawb/tokengs/data/scannet_prompt/"
                "scannet_prompt_full_wide_8x7.json"
            ),
            "wide_target_subsample": 0,
        }
    # Gentler pixel InfoNCE: smoother loss, less overfitting to the training
    # distribution (the strong 0.1/1.0 setting overfit: train gap 0.9, test
    # gap 0.001).
    opt.instance_branch_direct_gs_info_nce = args.info_nce_weight
    opt.instance_branch_direct_gs_info_temp = args.info_temp
    import tyro

    (out_dir / "config.yaml").write_text(
        tyro.extras.to_yaml(opt), encoding="utf-8"
    )

    model = model_registry[opt.model_type](opt)
    model.train()
    model = model.cuda()

    # Freeze everything except the direct-GS embedding head.
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.instance_branch.direct_gs_head.parameters():
        p.requires_grad_(True)
    trainable = [
        p for p in model.instance_branch.direct_gs_head.parameters()
        if p.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    print(f"[direct-gs-train] trainable params: "
          f"{sum(p.numel() for p in trainable)}")

    _, loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    iterator = iter(loader)

    t_start = time.time()
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
        optimizer.zero_grad()
        out = model(data)
        loss = out["loss_instance_group"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step % args.print_freq == 0 or step == args.num_steps:
            gap = out.get("direct_gs_pixel_gap")
            pc = out.get("direct_gs_proto_cos")
            pull = out.get("loss_direct_gs_pull")
            push = out.get("loss_direct_gs_push")
            elapsed = time.time() - t_start
            print(
                f"[direct-gs-train] step={step}/{args.num_steps} "
                f"loss={float(loss):.3f} pull={float(pull):.3f} "
                f"push={float(push):.3f} pixel_gap={float(gap):.3f} "
                f"proto_cos={float(pc):.3f} elapsed={elapsed:.0f}s",
                flush=True,
            )

        if step % args.ckpt_freq == 0 or step == args.num_steps:
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            path = ckpt_dir / f"model_step_{step:06d}.safetensors"
            state = {
                k: v.detach().cpu().contiguous()
                for k, v in model.state_dict().items()
            }
            from safetensors.torch import save_file

            save_file(state, str(path))
            metadata = {
                "epoch": step // args.max_iters_per_epoch,
                "step": step,
                "model_type": opt.model_type,
                "tokengs_checkpoint": opt.prompt_tokengs_checkpoint,
                "prompt_checkpoint": str(path),
            }
            (out_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            # Final/best model at the workspace root for convenience.
            save_file(state, str(out_dir / "model.safetensors"))
            print(f"[direct-gs-train] saved {path}", flush=True)

    print("[direct-gs-train] done")


if __name__ == "__main__":
    main()
