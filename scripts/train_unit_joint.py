"""Standalone joint training: TokenGS reconstruction + per-Unit instance
embedding, with the instance loss allowed to backprop into the token hidden
/ encoder (``instance_branch_backprop_token``) and a reconstruction-dominant
loss with an instance-loss warm-up so RGB reconstruction is not broken.

The instance branch is the 8-Local-Unit embedding + rendered-space pixel
InfoNCE (unit shaping + patch + DINO).  TokenGS encoder/decoder are unfrozen
(``prompt_unfreeze_tokengs``) but the instance loss weight warms from 0 so
the reconstruction stays dominant.
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


def reconstruction_only_forward(
    model, model_input, return_hidden=False, return_gaussians=False
):
    """Reconstruction-only forward with gradients (no semantic / instance
    branch), used for re10k batches which have no semantic/instance labels.
    Mirrors TokenGS._forward_prompt_reconstruction without the no_grad."""
    encoder_latent = model.forward_encoder(model_input.encoder)
    hidden = model.get_gs_tokens(
        encoder_latent.keys.shape[0],
        encoder_latent=encoder_latent,
        decoder_input=model_input.decoder,
    )
    hidden = model._apply_time_embedding_to_gs_tokens(
        hidden, model_input.decoder
    )
    for layer in model.enc_dec_backbone.decoder_blocks[:-1]:
        hidden = layer(
            gs_tokens=hidden,
            keys=encoder_latent.keys,
            values=encoder_latent.values,
        )
    geometry_hidden = model.enc_dec_backbone.decoder_blocks[-1](
        gs_tokens=hidden,
        keys=encoder_latent.keys,
        values=encoder_latent.values,
    )
    gaussians = model.activation_head(geometry_hidden)
    gaussians[..., 2] = gaussians[..., 2] + model.opt.gaussian_z_offset
    reconstruction = model._reconstruction_from_gaussians(gaussians)
    rgb_results = model.render_reconstruction(
        reconstruction, model_input.decoder
    )
    if return_hidden:
        if return_gaussians:
            return rgb_results, geometry_hidden, gaussians
        return rgb_results, geometry_hidden
    if return_gaussians:
        return rgb_results, gaussians
    return rgb_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--config-name",
        default="semantic_v6_unit_shaping_img_dino_train",
        choices=(
            "semantic_v6_unit_shaping_img_dino_train",
            "semantic_v6_unit_shaping_img_dino_k64_train",
            "semantic_v6_unit_shaping_img_dino_sic_smoke",
            "semantic_v6_unit_shaping_img_dino_sic_train",
            "semantic_v6_unit_shaping_img_dino_base8k_train",
            "semantic_v6_render_space_info_train",
            "semantic_v6_rendered_grounding_train",
            "semantic_v6_dynamic_query_train",
            "semantic_v6_center_offset_train",
            "semantic_v6_multidecoder_joint_train",
            "semantic_v6_multidecoder_gsdino_joint_train",
            "semantic_v6_generative_units_joint_train",
            "semantic_v6_generative_units_mask_joint_train",
        ),
        help=(
            "Model config. The default is the best-known 0.324 recipe "
            "(DINO + 8 local units + patch + InfoNCE, frozen backbone). "
            "semantic_v6_render_space_info_train additionally applies the "
            "rendered-space prototype push/pull (failed to beat 0.324 on "
            "ScanNet)."
        ),
    )
    parser.add_argument("--num_steps", type=int, default=3000)
    parser.add_argument("--ckpt_freq", type=int, default=200)
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lambda_rgb", type=float, default=200.0)
    parser.add_argument("--lambda_instance_final", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iters_per_epoch", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--decoder_unfreeze_tail",
        type=int,
        default=2,
        help=(
            "Number of FINAL decoder blocks to unfreeze in the protected "
            "decoder joint recipe (encoder / activation head / gs_tokens / "
            "unit formation stay frozen; a frozen pretrained TokenGS "
            "teacher distills the decoder hidden + RGB so PSNR does not "
            "collapse)."
        ),
    )
    parser.add_argument(
        "--lambda_distill",
        type=float,
        default=1.0,
        help="Weight of teacher/student decoder-hidden feature distillation.",
    )
    parser.add_argument(
        "--lambda_distill_rgb",
        type=float,
        default=1.0,
        help="Weight of teacher/student rendered-RGB distillation.",
    )
    parser.add_argument(
        "--lambda_distill_gs",
        type=float,
        default=1000.0,
        help=(
            "Weight of teacher/student Gaussian-parameter distillation "
            "(position/scale/opacity).  This is the direct geometry "
            "constraint that stops the unfrozen decoder tail from drifting "
            "enough to collapse PSNR."
        ),
    )
    parser.add_argument(
        "--scan-ratio",
        type=float,
        default=0.3,
        help="Fraction of steps using ScanNet (reconstruction + instance).",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help=(
            "Mixed data: re10k (4867 scenes, reconstruction only) + "
            "ScanNet LSM-style (1473 scenes, reconstruction + instance).  "
            "TokenGS reconstruction generalizes on re10k; instance "
            "supervision comes from ScanNet."
        ),
    )
    parser.add_argument(
        "--scanpp",
        action="store_true",
        help=(
            "Use ScanNet++ instance masks (87 processed scenes so far) as "
            "the instance-supervision data instead of ScanNet.  More/denser "
            "per-frame instance masks; verify whether instance learning "
            "generalizes better than ScanNet."
        ),
    )
    parser.add_argument(
        "--scanpp_windows",
        type=int,
        default=32,
        help=(
            "LSM-style windows per ScanNet++ scene.  More windows = more "
            "sample diversity for the same scene (only changes indices, "
            "not data cost)."
        ),
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help=(
            "Keep TokenGS encoder/decoder/geometry frozen and only train "
            "the instance branch (8 local units + identity embedding).  "
            "This matches the best-known 0.324 recipe; joint unfreezing "
            "has repeatedly failed."
        ),
    )
    parser.add_argument(
        "--data-mode",
        default="scannet_prompt_small",
        choices=("scannet_prompt_small", "scannet_lsm_style_train"),
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help=(
            "Joint train from random initialization (no re10k / DINO@3000 "
            "weights): reconstruction + instance supervision from step 0, "
            "on the full ScanNet LSM-style windows.  Risk: ScanNet-only "
            "reconstruction may converge below re10k pretrained PSNR."
        ),
    )
    parser.add_argument(
        "--center_offset",
        action="store_true",
        help=(
            "PointGroup-style center-offset mode: freeze everything except "
            "the small per-unit instance-center offset head and train only "
            "the center regression (use --config-name "
            "semantic_v6_center_offset_train)."
        ),
    )
    parser.add_argument(
        "--multi_decoder",
        action="store_true",
        help=(
            "Unified multi-decoder joint: the instance branch (DirectGS or "
            "DINO+8-local-units per config, gated by GradScale) + semantic "
            "LSeg-lifting decoder share the TokenGS latent; the "
            "instance/semantic losses back-propagate into the decoder tail "
            "while L_recon anchors reconstruction (no teacher / detach)."
        ),
    )
    parser.add_argument(
        "--unfreeze_level",
        choices=("tail", "decoder", "all"),
        default="tail",
        help=(
            "Joint fine-tune scope for the shared TokenGS latent: 'tail' "
            "= last --decoder_unfreeze_tail blocks; 'decoder' = all decoder "
            "blocks (activation head stays frozen); 'all' = encoder + "
            "decoder + activation head + gs_tokens (full fine-tune, risky)."
        ),
    )
    parser.add_argument(
        "--lambda_semantic",
        type=float,
        default=1.0,
        help="Weight of the semantic LSeg-distillation loss (multi_decoder).",
    )
    parser.add_argument(
        "--recon-only",
        action="store_true",
        help=(
            "Pure reconstruction fine-tune on ScanNet: scan batches run "
            "reconstruction_only_forward (RGB MSE only, no instance / "
            "semantic losses, no teacher).  Use with --unfreeze_level all "
            "and --data-mode scannet_lsm_style_train to adapt the frozen "
            "TokenGS geometry to the ScanNet domain before re-running the "
            "0.324 instance recipe."
        ),
    )
    parser.add_argument(
        "--backbone-resume",
        default="",
        help=(
            "Override the config's backbone_resume (the checkpoint whose "
            "frozen TokenGS weights are loaded).  Point at tokengs_re10k "
            "to fine-tune the base pretrained model on ScanNet."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = config_defaults[args.config_name]
    opt.workspace = args.workspace
    opt.experiment_name = out_dir.name
    opt.max_iters_per_epoch = args.max_iters_per_epoch
    opt.num_epochs = (args.num_steps + args.max_iters_per_epoch - 1) // (
        args.max_iters_per_epoch
    )
    opt.evaluating = False
    opt.prompt_unfreeze_tokengs = not args.freeze_backbone
    opt.prompt_unfreeze_tokengs_mode = "all"
    opt.instance_branch_backprop_token = not args.freeze_backbone
    if args.backbone_resume:
        opt.backbone_resume = args.backbone_resume
    if args.decoder_unfreeze_tail > 0 and not args.multi_decoder:
        # Protected-decoder joint: the reconstruction path runs from the
        # detached hidden (monitor only), so the unfrozen decoder tail is
        # updated by instance supervision + teacher distillation only.
        opt.prompt_detach_reconstruction_tokens = True
    if args.center_offset:
        opt.prompt_unfreeze_tokengs = False
        opt.instance_branch_backprop_token = False
    if args.multi_decoder:
        opt.prompt_unfreeze_tokengs = True
        opt.instance_branch_backprop_token = True
    opt.batch_size = args.batch_size
    opt.num_workers = args.num_workers
    if args.from_scratch:
        opt.prompt_tokengs_checkpoint = ""
        opt.instance_branch_unit_resume = ""
        opt.backbone_resume = ""
        if not args.scanpp:
            args.data_mode = "scannet_lsm_style_train"
    if args.data_mode == "scannet_lsm_style_train":
        opt.data_mode = (("scannet_lsm_style_train", 1),)
        opt.dataset_kwargs = {"windows_per_scene": 16}
    elif args.scanpp:
        opt.data_mode = (("scannetpp_instance_train", 1),)
        opt.dataset_kwargs = {
            "windows_per_scene": args.scanpp_windows,
        }
    else:
        opt.data_mode = (("scannet_prompt_small", 1),)
        opt.dataset_kwargs = {
            "small_manifest_path": (
                "/space0/mawb/tokengs/data/scannet_prompt/"
                "scannet_prompt_full_wide_8x7.json"
            ),
            "wide_target_subsample": 0,
        }
    import tyro

    (out_dir / "config.yaml").write_text(
        tyro.extras.to_yaml(opt), encoding="utf-8"
    )

    _LocalAccelerator()  # noqa: F841 (required by get_multi_dataloader)
    train_loader_scan, _, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    iter_scan = iter(train_loader_scan)
    if args.mixed:
        opt_mix = config_defaults[args.config_name]
        opt_mix.data_mode = (("re10k", 1),)
        opt_mix.dataset_kwargs = {
            "index_file": (
                "/space0/mawb/tokengs/workspace/re10k_index.json"
            )
        }
        opt_mix.batch_size = args.batch_size
        opt_mix.num_workers = args.num_workers
        opt_mix.num_views = 15
        opt_mix.num_input_views = 8
        opt_mix.evaluating = False
        opt_mix.prompt_unfreeze_tokengs = not args.freeze_backbone
        opt_mix.prompt_unfreeze_tokengs_mode = "all"
        opt_mix.instance_branch_backprop_token = not args.freeze_backbone
        train_loader_re10k, _, _, _ = get_multi_dataloader(
            opt_mix, _LocalAccelerator()
        )
        iter_re10k = iter(train_loader_re10k)
        print(
            "[unit-joint] mixed data: re10k + "
            + ("ScanNet++" if args.scanpp else "ScanNet")
        )
    else:
        iter_re10k = None

    model = model_registry[opt.model_type](opt)
    model.train()
    model = model.cuda()

    # Protected-decoder recipe: freeze everything except (a) the last
    # ``--decoder_unfreeze_tail`` decoder blocks and (b) the instance
    # branch.  A frozen pretrained TokenGS teacher distills its decoder
    # hidden + rendered RGB so the unfrozen tail cannot collapse PSNR.
    tail_blocks = (
        list(model.enc_dec_backbone.decoder_blocks)[-args.decoder_unfreeze_tail:]
        if (
            args.decoder_unfreeze_tail > 0
            and args.unfreeze_level == "tail"
        )
        else []
    )
    if args.unfreeze_level in ("decoder", "all"):
        tail_blocks = list(model.enc_dec_backbone.decoder_blocks)
    tail_param_ids = set()
    for blk in tail_blocks:
        for p in blk.parameters():
            tail_param_ids.add(id(p))
    for name, param in model.named_parameters():
        param.requires_grad_(False)
    # trainable: DirectGS instance head + semantic lifting modules
    for module_name in ("semantic_lifting_head", "semantic_projector",
                        "prompt_semantic_adapter"):
        module = getattr(model, module_name, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad_(True)
    # Trainable instance-branch params for the DINO + 8-local-units recipe
    # (unit formation extractor + patch/DINO projections).  ``direct_gs_head``
    # is kept for backward compatibility with the old DirectGS config.
    instance_trainable_tokens = (
        "gs_feature_mlp.",
        "unit_queries",
        "unit_layers.",
        "log_unit_temp",
        "unit_image_net.",
        "dino_proj.",
        # Direction B0: scene-conditioned instance queries + zero-gated
        # readout into unit-query initialization (new params only).
        "sic_module.",
        "sic_gate",
        "sic_readout.",
        "sic_pos_emb.",
        "sic_h_proj.",
        "sic_dense_proj.",
        "sic_dino_proj.",
        "sic_desc_norm.",
        "sic_q_proj.",
        "identity_encoder.",
        "direct_gs_head",
        "unit_gaussian_decoder",
        # v2 rendered-mask assignment path (group tokens -> unit assignment)
        "unit_ctx_mlp",
        "unit_pos_mlp",
        "unit_norm",
        "group_tokens",
        "group_layers",
        "group_norm",
        "unit_assignment_proj",
        "group_assignment_proj",
        "log_assignment_temperature",
        "void_head",
    )
    for name, param in model.named_parameters():
        if name.startswith("instance_branch.") and any(
            tok in name for tok in instance_trainable_tokens
        ):
            param.requires_grad_(True)
    for name, param in model.named_parameters():
        if id(param) in tail_param_ids:
            param.requires_grad_(True)
    if args.unfreeze_level == "all":
        for name, param in model.named_parameters():
            if (
                name.startswith("enc_dec_backbone.")
                or name in ("gs_tokens", "gs_tokens_dynamic")
                or name.startswith("activation_head.")
            ):
                param.requires_grad_(True)
    if args.center_offset:
        # Center-offset mode: ONLY the per-unit center offset head trains.
        for name, param in model.named_parameters():
            param.requires_grad_(
                "center_offset_head" in name
            )

    teacher = None
    if (
        tail_blocks
        and not args.center_offset
        and not args.multi_decoder
        and not args.recon_only
    ):
        teacher = model_registry[opt.model_type](opt)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        teacher = teacher.cuda()

    decoder_tail_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in tail_param_ids:
            decoder_tail_params.append(param)
        else:
            head_params.append(param)
    optim_groups = [{"params": head_params, "lr": args.lr}]
    if decoder_tail_params:
        optim_groups.append(
            {"params": decoder_tail_params, "lr": args.lr_backbone}
        )
    optimizer = torch.optim.AdamW(optim_groups, weight_decay=0.05)
    print(
        f"[unit-joint] trainable: decoder_tail={sum(p.numel() for p in decoder_tail_params)} "
        f"({len(tail_blocks)} blocks) head={sum(p.numel() for p in head_params)}"
    )

    t_start = time.time()
    rng = np.random.default_rng(0)
    for step in range(1, args.num_steps + 1):
        use_scan = (
            iter_re10k is None
            or rng.random() < args.scan_ratio
        )
        iterator = iter_scan if use_scan else iter_re10k
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(train_loader_scan) if use_scan else iter(
                train_loader_re10k
            )
            data = next(iterator)
            if use_scan:
                iter_scan = iterator
            else:
                iter_re10k = iterator
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        warmup = min(1.0, step / max(1, args.warmup_steps))
        if args.center_offset:
            # Center-offset regression needs NO warmup: it is a stable
            # regression with a well-defined target (unlike the instance
            # grouping losses); the forward already applies
            # instance_center_loss_weight.
            lam_inst = 1.0
        else:
            lam_inst = args.lambda_instance_final * warmup
        optimizer.zero_grad()
        out = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if not use_scan or args.recon_only:
                model_input, supervision = split_data(data, opt)
                rgb_results = reconstruction_only_forward(model, model_input)
                pred = rgb_results["images_pred"]
                gt = supervision.images_output
                loss_rgb = (pred - gt.float()).square().mean()
                loss_inst = loss_rgb.detach() * 0.0
                sem_loss = None
                loss = args.lambda_rgb * loss_rgb
                src_label = "re10k" if not use_scan else "scan-recon"
            else:
                out = model(data)
                loss_rgb = out["loss_rgb"]
                loss_inst = out.get("loss_instance_group")
                if loss_inst is None:
                    loss_inst = loss_rgb.detach() * 0.0
                loss = args.lambda_rgb * loss_rgb + lam_inst * loss_inst
                src_label = "scan"
                sem_loss = out.get("loss_semantic_lifting")
                if args.multi_decoder and sem_loss is not None:
                    loss = loss + args.lambda_semantic * sem_loss
                else:
                    sem_loss = loss_rgb.detach() * 0.0
                distill_feat = torch.zeros((), device=loss.device)
                distill_rgb = torch.zeros((), device=loss.device)
                distill_gs = torch.zeros((), device=loss.device)
                feat_dev = torch.zeros((), device=loss.device)
                gs_dev = torch.zeros((), device=loss.device)
                if teacher is not None:
                    model_input, _ = split_data(data, opt)
                    with torch.no_grad():
                        teacher_rgb, teacher_hidden, teacher_gs = (
                            reconstruction_only_forward(
                                teacher,
                                model_input,
                                return_hidden=True,
                                return_gaussians=True,
                            )
                        )
                    student_hidden = out["gs_token_hidden"]
                    distill_feat = F.mse_loss(
                        F.normalize(student_hidden.float(), dim=-1),
                        F.normalize(teacher_hidden.float(), dim=-1),
                    )
                    distill_rgb = F.mse_loss(
                        out["images_pred"].clamp(0, 1).float(),
                        teacher_rgb["images_pred"].clamp(0, 1).float(),
                    )
                    # Gaussian-parameter distillation (position, scale,
                    # opacity) - the direct geometry constraint.
                    sg = out["gaussians"].float()
                    tg = teacher_gs.float()
                    w_pos = torch.ones(sg.shape[-1], device=sg.device)
                    # gaussian layout: pos(0-2) opacity(3) scale(4-6)
                    # rot(7-10) color(11-13)
                    w_pos[4:7] = 10.0  # scale drift is more sensitive
                    w_pos[11:] = 0.0  # exclude color (RGB distill covers it)
                    distill_gs = (
                        (sg - tg).square() * w_pos.view(1, 1, -1)
                    ).mean()
                    feat_dev = (
                        student_hidden.float() - teacher_hidden.float()
                    ).norm(dim=-1).mean()
                    gs_dev = (sg - tg).norm(dim=-1).mean()
                    loss = (
                        loss
                        + args.lambda_distill * distill_feat
                        + args.lambda_distill_rgb * distill_rgb
                        + args.lambda_distill_gs * distill_gs
                    )
        # Per-task gradient norms on the shared latent (effective
        # contribution after the GradScale gates): reconstruction vs
        # instance vs semantic.  Only at print steps to avoid 3 extra
        # backwards per step in the long run.
        hidden = (
            out.get("gs_token_hidden")
            if (use_scan and not args.recon_only)
            else None
        )

        def _hidden_grad(loss_term):
            if (
                loss_term is None
                or hidden is None
                or not loss_term.requires_grad
                or not hidden.requires_grad
            ):
                return None
            try:
                g = torch.autograd.grad(
                    loss_term,
                    hidden,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                return g
            except RuntimeError:
                return None

        want_grad_log = step % args.print_freq == 0 or step == args.num_steps
        if want_grad_log:
            g_recon = _hidden_grad(loss_rgb)
            g_inst = _hidden_grad(loss_inst)
            g_sem = _hidden_grad(sem_loss)
            grad_recon_norm = (
                float(g_recon.norm()) if g_recon is not None else float("nan")
            )
            grad_inst_norm = (
                float(g_inst.norm()) if g_inst is not None else float("nan")
            )
            grad_sem_norm = (
                float(g_sem.norm()) if g_sem is not None else float("nan")
            )
        else:
            grad_recon_norm = float("nan")
            grad_inst_norm = float("nan")
            grad_sem_norm = float("nan")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        if step % args.print_freq == 0 or step == args.num_steps:
            use_full_forward = use_scan and not args.recon_only
            psnr = out.get("psnr") if use_full_forward else (
                -10.0 * torch.log10(loss_rgb.detach().clamp_min(1e-10))
            )
            gap = out.get("direct_gs_pixel_gap") if use_full_forward else None
            diff = (
                out.get("unit_embedding_diff_sim")
                if use_full_forward
                else None
            )
            g3d_same = (
                out.get("grounding_3d_same_cos")
                if use_full_forward
                else None
            )
            g3d_diff = (
                out.get("grounding_3d_diff_cos")
                if use_full_forward
                else None
            )
            center_err = (
                out.get("center_err") if use_full_forward else None
            )
            sem_s = f"{float(sem_loss):.4f}" if use_full_forward else "N/A"
            distill_s = (
                f"feat={float(distill_feat):.4f}/rgb={float(distill_rgb):.4f}"
                f"/gs={float(distill_gs):.4f}"
                if teacher is not None and use_scan
                else "N/A"
            )
            fev_s = f"{float(feat_dev):.4f}/{float(gs_dev):.4f}" if (
                teacher is not None and use_scan
            ) else "N/A"
            elapsed = time.time() - t_start
            gap_s = f"{float(gap):.3f}" if gap is not None else "N/A"
            diff_s = f"{float(diff):.3f}" if diff is not None else "N/A"
            ct_same = (
                out.get("unit_same_tok_sim")
                if use_full_forward
                else None
            )
            ct_cross = (
                out.get("unit_cross_tok_sim")
                if use_full_forward
                else None
            )
            ct_diff = (
                out.get("unit_diff_sim") if use_full_forward else None
            )
            sic_act = (
                out.get("sic_active_ratio")
                if use_full_forward
                else None
            )
            sic_gate_v = (
                out.get("sic_gate_value")
                if use_full_forward
                else None
            )
            ct_s = (
                f"{float(ct_same):.3f}/{float(ct_cross):.3f}/"
                f"{float(ct_diff):.3f}"
                if ct_same is not None
                else "N/A"
            )
            sic_s = (
                f"act={float(sic_act):.3f}/gate={float(sic_gate_v):.4f}"
                if sic_act is not None
                else "N/A"
            )
            g3d_s = (
                f"{float(g3d_same):.3f}/{float(g3d_diff):.3f}"
                if g3d_same is not None
                else "N/A"
            )
            cerr_s = (
                f"{float(center_err):.4f}"
                if center_err is not None
                else "N/A"
            )
            grad_s = (
                f"recon={grad_recon_norm:.4f}/ins={grad_inst_norm:.4f}/"
                f"sem={grad_sem_norm:.4f}"
                if want_grad_log
                else "N/A"
            )
            grad_alpha_s = (
                f"alpha_sem={float(getattr(opt, 'grad_scale_sem', 0.3)):.2f}/"
                f"alpha_ins={float(getattr(opt, 'grad_scale_ins', 0.1)):.2f}"
            )
            print(
                f"[unit-joint] step={step}/{args.num_steps} "
                f"src={src_label} loss={float(loss):.3f} "
                f"rgb={float(loss_rgb):.3f} "
                f"inst={float(loss_inst):.3f} lam_inst={lam_inst:.3f} "
                f"psnr={float(psnr):.2f} pixel_gap={gap_s} "
                f"diff_sim={diff_s} g3d_same/diff={g3d_s} "
                f"ct_same/cross/diff={ct_s} sic={sic_s} "
                f"center_err={cerr_s} "
                f"sem={sem_s} "
                f"grad_hidden[{grad_s}] {grad_alpha_s} "
                f"distill={distill_s} feat_dev={fev_s} "
                f"elapsed={elapsed:.0f}s",
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
            save_file(state, str(out_dir / "model.safetensors"))
            print(f"[unit-joint] saved {path}", flush=True)

    print("[unit-joint] done")


if __name__ == "__main__":
    main()
