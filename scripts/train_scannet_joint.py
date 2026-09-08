"""Joint ScanNet training: instance grouping + V3 LSeg semantic field.

Warm-starts the instance branch from the best wide7l checkpoint and the
semantic head from the old 0.535 C3G8 checkpoint, then trains both on the
full ScanNet train set (``scannet_prompt_train``) with frozen TokenGS
geometry. Instance supervision: ScanNet 2D instance-filt masks (Hungarian
BCE+Dice). Semantic supervision: LSeg feature distillation (target+source).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import tyro
from accelerate import Accelerator
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.options import config_defaults


WIDE7L_CHECKPOINT = (
    "/space0/mawb/tokengs/workspace/"
    "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/model.safetensors"
)
V3_SEMANTIC_CHECKPOINT = (
    "/space0/mawb/tokengs_c3g/workspace/"
    "re10k_semantic_lseg_v3_token_r16/model.safetensors"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--num_steps", type=int, default=8000)
    parser.add_argument("--lr_instance", type=float, default=1e-4)
    parser.add_argument("--lr_semantic", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--ckpt_freq", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=15)
    parser.add_argument("--resume_instance", default=WIDE7L_CHECKPOINT)
    parser.add_argument("--resume_semantic", default=V3_SEMANTIC_CHECKPOINT)
    parser.add_argument("--num_groups", type=int, default=128)
    parser.add_argument("--windows_per_scene", type=int, default=16)
    parser.add_argument("--lr_geometry", type=float, default=3e-5)
    parser.add_argument("--lambda_rgb", type=float, default=1.0)
    parser.add_argument("--lambda_boundary_rgb", type=float, default=0.1)
    parser.add_argument("--lambda_instance_3d", type=float, default=1.0)
    parser.add_argument(
        "--unfreeze_mode",
        choices=("decoder", "all"),
        default="decoder",
        help=(
            "When geometry is unfrozen: 'decoder' updates decoder blocks, "
            "activation head, and GS tokens only; 'all' also updates the "
            "encoder and patch embeddings."
        ),
    )
    parser.add_argument(
        "--instance_per_gaussian",
        action="store_true",
        help=(
            "Use the per-Gaussian instance residual head. Off by default so "
            "the wide7l instance head warm-starts without stale-key mixing."
        ),
    )
    parser.add_argument(
        "--freeze_geometry",
        action="store_true",
        help="Keep the TokenGS geometry frozen (heads only).",
    )
    parser.add_argument(
        "--resume_train",
        default="",
        help="Resume from a joint checkpoint (optimizer/scheduler state is "
        "read from the workspace, step from train_state.json).",
    )
    parser.add_argument(
        "--data_mode",
        default="scannet_lsm_style_train",
        help="Training dataset mode (scannet_lsm_style_train for eval-like "
        "wide-gap windows, scannet_prompt_small for the close-frame 64-scene "
        "protocol that produced wide7l).",
    )
    args = parser.parse_args()

    accelerator = Accelerator()
    opt = config_defaults["eval_scannet_lsm_instance"]
    opt.data_mode = ((args.data_mode, 1),)
    opt.dataset_kwargs = {
        "windows_per_scene": args.windows_per_scene,
    }
    opt.num_input_views = args.num_input_views
    opt.num_views = args.num_views
    if args.data_mode == "scannet_prompt_small":
        # ScanNetPromptSmall is driven entirely by its manifest.
        opt.prompt_mode = "manifest"
        # The current small manifest contains 2-input + 1-target prompt
        # samples (the 8+7 manifest used by the original wide7l run no
        # longer exists), so train with the dataset's native view counts.
        if args.num_input_views >= 8:
            opt.num_input_views = 2
            opt.num_views = 3
            print(
                "[scannet-joint] scannet_prompt_small is 2+1; "
                "forcing num_input_views=2, num_views=3"
            )
    opt.batch_size = 1
    opt.num_workers = args.num_workers
    opt.evaluating = False

    # --- instance grouping (wide7l recipe) ---
    opt.use_instance_labels = True
    opt.instance_group_num_groups = args.num_groups
    opt.instance_group_decoder = True
    opt.instance_group_decoder_layers = 2
    opt.instance_group_use_anchor_pos = True
    opt.instance_group_render_scale = 1.0
    opt.instance_group_alpha_threshold = 0.05
    opt.instance_group_min_instance_pixels = 32
    opt.instance_group_match_area_norm = True
    opt.instance_group_match_topk = 3
    opt.instance_group_area_alpha = 0.5
    opt.instance_group_secondary_pair_weight = 0.3
    opt.instance_group_usage_entropy = 0.05
    opt.instance_group_lambda_warmup_steps = 500
    opt.instance_group_per_gaussian = args.instance_per_gaussian
    opt.instance_group_residual_head = args.instance_per_gaussian
    opt.lambda_instance_group_ce = 1.0
    opt.lambda_instance_group_dice = 1.0
    opt.lambda_instance_group_mask = 1.0
    opt.lambda_instance_group_unmatched = 0.1
    opt.lambda_instance_group_void = 0.1
    opt.lambda_instance_group_3d = args.lambda_instance_3d
    opt.lambda_instance_group_3d_ce = 1.0
    opt.instance_group_3d_min_gs = 16
    opt.instance_group_3d_match_topk = 1

    # --- V3 semantic field ---
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

    # --- disable the CLIP prompt path (only instance + LSeg semantic loss) ---
    opt.prompt_lambda_bce = 0.0
    opt.prompt_lambda_dice = 0.0
    opt.lambda_ce = 0.0
    opt.lambda_ce_cosine = 0.0
    opt.lambda_feat = 0.0
    opt.lambda_rgb = 0.0
    opt.lambda_instance_contrastive = 0.0

    opt.prompt_tokengs_checkpoint = (
        "/space0/mawb/tokengs/workspace/tokengs_re10k/model.safetensors"
    )
    # End-to-end mode: unfreeze the geometry so tokens can adapt to
    # instance boundaries (InstanceSplat-style), protected by the
    # reconstruction loss. ``prompt_unfreeze_tokengs`` must be True so
    # ``_forward_prompt_reconstruction`` keeps gradients flowing.
    opt.prompt_unfreeze_tokengs = not args.freeze_geometry
    opt.prompt_unfreeze_tokengs_mode = args.unfreeze_mode
    opt.gradient_clip = 1.0
    # A frozen geometry has no path from the RGB loss back to parameters.
    # Keeping it in the total loss would only dominate the reported value.
    opt.lambda_rgb = 0.0 if args.freeze_geometry else args.lambda_rgb
    opt.lambda_boundary_rgb = (
        0.0 if args.freeze_geometry else args.lambda_boundary_rgb
    )

    model = model_registry[opt.model_type](opt)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Warm start: instance branch + fine-tuned backbone from wide7l, and
    # ONLY the semantic head from the old 0.535 V3 checkpoint. The v3
    # checkpoint also contains the vanilla base backbone; loading it would
    # overwrite wide7l's geometry-adapted backbone and break the instance
    # head (its tokens are tuned to wide7l's geometry).
    torch.nn.Module.load_state_dict(
        model, load_file(args.resume_instance, device="cpu"), strict=False
    )
    semantic_state = {
        key: value
        for key, value in load_file(
            args.resume_semantic, device="cpu"
        ).items()
        if key.startswith("semantic_head.")
    }
    missing_semantic, unexpected_semantic = torch.nn.Module.load_state_dict(
        model, semantic_state, strict=False
    )
    print(
        "[scannet-joint] semantic warm-start: "
        f"loaded={len(semantic_state)} missing={len(missing_semantic)} "
        f"unexpected={len(unexpected_semantic)}"
    )
    resume_step = 1
    if args.resume_train:
        torch.nn.Module.load_state_dict(
            model, load_file(args.resume_train, device="cpu"), strict=False
        )
        print(f"[scannet-joint] resume model weights from {args.resume_train}")

    for module in (
        model.semantic_head,
        model.instance_group_head,
        model.anchor_pos_encoder,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    geometry_modules = []
    if not args.freeze_geometry:
        if args.unfreeze_mode == "decoder":
            geometry_modules = [
                model.enc_dec_backbone.decoder_blocks,
                model.activation_head,
            ]
        elif args.unfreeze_mode == "all":
            geometry_modules = [
                model.enc_dec_backbone,
                model.patch_embed,
                model.patch_plucker_embed,
                model.activation_head,
            ]
        else:  # pragma: no cover - argparse restricts this branch
            raise ValueError(f"Unsupported unfreeze_mode={args.unfreeze_mode}")
        for module in geometry_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        model.gs_tokens.requires_grad_(True)

    def split_decay(params):
        decay, no_decay = [], []
        for parameter in params:
            if not parameter.requires_grad:
                continue
            (no_decay if parameter.dim() <= 1 else decay).append(parameter)
        return decay, no_decay

    sem_decay, sem_no_decay = split_decay(model.semantic_head.parameters())
    inst_decay, inst_no_decay = split_decay(model.instance_group_head.parameters())
    inst_decay += [p for p in model.anchor_pos_encoder.parameters() if p.dim() > 1 and p.requires_grad]
    inst_no_decay += [p for p in model.anchor_pos_encoder.parameters() if p.dim() <= 1 and p.requires_grad]

    optim_groups = [
        {"params": sem_decay, "lr": args.lr_semantic, "weight_decay": 0.05},
        {"params": sem_no_decay, "lr": args.lr_semantic, "weight_decay": 0.0},
        {"params": inst_decay, "lr": args.lr_instance, "weight_decay": 0.05},
        {"params": inst_no_decay, "lr": args.lr_instance, "weight_decay": 0.0},
    ]
    max_lrs = [args.lr_semantic, args.lr_semantic, args.lr_instance, args.lr_instance]
    if not args.freeze_geometry:
        geometry_params = []
        for module in geometry_modules:
            geometry_params.extend(module.parameters())
        geometry_params.append(model.gs_tokens)
        geo_decay, geo_no_decay = split_decay(geometry_params)
        optim_groups = [
            {"params": geo_decay, "lr": args.lr_geometry, "weight_decay": 0.05},
            {"params": geo_no_decay, "lr": args.lr_geometry, "weight_decay": 0.0},
            *optim_groups,
        ]
        max_lrs = [args.lr_geometry, args.lr_geometry, *max_lrs]

    optimizer = torch.optim.AdamW(
        optim_groups,
        betas=(0.9, 0.95),
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=args.num_steps,
        pct_start=min(args.warmup_steps / max(1, args.num_steps), 0.99),
        final_div_factor=1000.0,
    )

    trainable = sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    print("[scannet-joint] trainable params:", trainable)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "config.yaml"
    if not config_path.exists():
        config_path.write_text(tyro.extras.to_yaml(opt), encoding="utf-8")
    checkpoint_dir = workspace / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    if args.resume_train:
        optimizer_path = workspace / "optimizer.pth"
        scheduler_path = workspace / "scheduler.pth"
        train_state_path = workspace / "train_state.json"
        if not (optimizer_path.exists() and scheduler_path.exists()):
            raise RuntimeError(
                f"Missing optimizer/scheduler state for resume: {workspace}"
            )
        optimizer.load_state_dict(
            torch.load(optimizer_path, map_location="cpu", weights_only=False)
        )
        scheduler.load_state_dict(
            torch.load(scheduler_path, map_location="cpu", weights_only=False)
        )
        if train_state_path.exists():
            resume_step = int(
                json.loads(train_state_path.read_text())["step"]
            )
        print(
            f"[scannet-joint] resuming from step {resume_step} "
            f"(last_lr={scheduler.get_last_lr()})"
        )

    _, loader, _, _ = get_multi_dataloader(opt, accelerator)
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    model.train()

    def _save_state(unwrapped) -> dict:
        """Heads + geometry so checkpoints are self-contained at eval time.

        The filtered ``state_dict`` only keeps trainable heads; include the
        (frozen but geometry-adapted) backbone and GS tokens as well, so the
        eval scripts do not fall back to the vanilla base geometry.
        """
        state = unwrapped.state_dict()
        for key, value in unwrapped.enc_dec_backbone.state_dict().items():
            state[f"enc_dec_backbone.{key}"] = value.detach().cpu()
        for name in ("patch_embed", "patch_plucker_embed", "activation_head"):
            module = getattr(unwrapped, name, None)
            if module is not None:
                for key, value in module.state_dict().items():
                    state[f"{name}.{key}"] = value.detach().cpu()
        state["gs_tokens"] = unwrapped.gs_tokens.detach().cpu()
        return state

    iterator = iter(loader)
    total_loss = 0.0
    for step in range(1, args.num_steps + 1):
        if step <= resume_step:
            continue
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            data = next(iterator)
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.instance_group_lambda_eff = (
            unwrapped.compute_instance_group_lambda_eff(step, opt)
        )
        optimizer.zero_grad()
        out = model(data)
        loss = out["loss"]
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.detach())
        if step % args.log_freq == 0 and accelerator.is_main_process:
            print(
                f"[scannet-joint] step={step}/{args.num_steps} "
                f"loss={total_loss / args.log_freq:.4f} "
                f"instance={float(out.get('loss_instance_group', torch.zeros(()))):.4f} "
                f"sem_v3={float(out.get('loss_semantic_v3', torch.zeros(()))):.4f} "
                f"psnr={float(out['psnr']):.2f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
            total_loss = 0.0
        if step % args.ckpt_freq == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            save_file(
                _save_state(unwrapped),
                str(checkpoint_dir / f"model_step_{step:06d}.safetensors"),
            )
            torch.save(
                optimizer.state_dict(),
                str(workspace / "optimizer.pth"),
            )
            torch.save(
                scheduler.state_dict(),
                str(workspace / "scheduler.pth"),
            )
            (workspace / "train_state.json").write_text(
                json.dumps({"step": step}), encoding="utf-8"
            )
            print(f"[scannet-joint] saved model_step_{step:06d}.safetensors")
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_file(_save_state(unwrapped), str(workspace / "model.safetensors"))
        print("[scannet-joint] done")


if __name__ == "__main__":
    main()
