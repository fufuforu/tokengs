"""Offline 3D-consistency diagnostic for instance grouping (no training).

For each LSM scene, runs the frozen backbone + trained group head once and
measures how "3D" the per-Gaussian grouping is:

- rendered_vs_gs_label_agreement: fraction of visible Gaussians whose own
  argmax group matches the rendered argmax label at their projected pixel
  (mask sharpness / alpha-blending coherence).
- rendered_cross_view_agreement: for Gaussians visible in two views, whether
  the rendered labels at the two projected pixels agree (render-level 3D
  consistency).
- knn3d_same_group: mean fraction of 3D k-NN neighbours sharing the GS group
  (3D spatial coherence of the assignment).
- token_purity: per-token mode fraction of the 64 GS labels.
- gt_fragmentation: mean number of distinct groups covering one GT instance's
  Gaussians (over-segmentation at the GS level).
- gt_purity: mean best-group coverage per GT instance (max fraction of its GS
  in one group).
- gt_merging: mean number of distinct GT instances covered per active group
  (under-segmentation).
- void_share: fraction of Gaussians assigned to the void channel.

Optionally (--ttt-steps > 0) adapts only the instance head on the 8 context
views' GT masks before measuring, which quantifies how much per-scene readout
supervision restores 3D structure (the TTT=0.72 control).

Usage:
    python scripts/diagnose_3d_consistency.py \
        --resume workspace/semantic_v6_open_vocab_pgr3df2_train_12000/checkpoints/model_step_012000.safetensors \
        --workspace workspace/diag_3d_pgr3df2 \
        --label pgr3df2 \
        [--ttt-steps 30]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch._dynamo.config.disable = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.input_types import split_data
from tokengs.options import config_defaults


class _LocalAccelerator:
    is_main_process = True


def _load_arch_from_checkpoint(args, opt) -> None:
    """Mirror the LSM eval's architecture passthrough from checkpoint config."""
    checkpoint_path = Path(args.resume)
    meta = {}
    for name in (
        f"metadata_step_{checkpoint_path.stem.replace('model_step_', '')}.json",
        "metadata_best.json",
        "metadata.json",
    ):
        candidate = checkpoint_path.parent / name
        if candidate.is_file():
            meta = json.load(open(candidate, encoding="utf-8"))
            break
    config_yaml = checkpoint_path.parent / "config.yaml"
    if not config_yaml.is_file():
        config_yaml = checkpoint_path.parent.parent / "config.yaml"
    if config_yaml.is_file():
        try:
            import yaml

            def _options_ctor(loader, tag_suffix, node):
                return loader.construct_mapping(node, deep=True)

            yaml.add_multi_constructor("!dataclass:", _options_ctor, Loader=yaml.UnsafeLoader)
            yaml_cfg = yaml.load(
                config_yaml.read_text(encoding="utf-8"), Loader=yaml.UnsafeLoader
            )
            for key, value in yaml_cfg.items():
                meta.setdefault(key, value)
        except Exception as exc:  # pragma: no cover
            print(f"[diag-3d] config.yaml parse failed: {exc}")
    if not meta:
        return
    for field, cast in (
        ("instance_group_num_groups", int),
        ("instance_group_per_gaussian", bool),
        ("instance_group_decoder", bool),
        ("instance_group_residual_head", bool),
        ("instance_group_use_anchor_pos", bool),
        ("instance_group_render_scale", float),
        ("num_gs_tokens", int),
        ("num_dynamic_gs_tokens", int),
        ("gs_token_init", str),
        ("anchor_3d_query_source", str),
        ("anchor_3d_pos_scale", float),
        ("anchor_3d_pos_freqs", int),
        ("anchor_3d_pass_decoder", bool),
        ("instance_branch_independent", bool),
        ("instance_branch_num_groups", int),
        ("instance_branch_anchor_dim", int),
        ("instance_branch_num_heads", int),
        ("instance_branch_num_layers", int),
        ("instance_branch_gaussians_per_anchor", int),
        ("instance_branch_pos_offset_scale", float),
        ("instance_branch_scale_delta_amp", float),
        ("instance_branch_opacity_delta_amp", float),
        ("instance_branch_rgb_delta_amp", float),
        ("instance_branch_token_units", bool),
        ("instance_branch_units_per_token", int),
        ("instance_branch_unit_feat_dim", int),
        ("instance_branch_unit_layers", int),
        ("instance_branch_unit_temp", float),
        ("instance_branch_unit_entropy", float),
        ("instance_branch_unit_compactness", float),
    ):
        if meta.get(field) is not None:
            setattr(opt, field, cast(meta[field]))
    if meta.get("instance_group_num_groups") is not None:
        opt.instance_group_num_groups = int(meta["instance_group_num_groups"])
    backbone_resume = meta.get("backbone_resume") or meta.get("resume")
    if backbone_resume:
        opt.backbone_resume = str(backbone_resume)
    for field in (
        "enc_depth",
        "dec_depth",
        "dec_patch_size",
        "enc_embed_dim",
        "enc_num_heads",
        "mlp_ratio",
        "patch_size",
        "dec_init_values",
        "clip_head_readout_std",
        "clip_head_z_init",
        "gaussian_z_offset",
        "opacity_bias",
        "gs_token_std",
        "use_multiscale_encoder",
        "use_latent_bottleneck",
        "num_latents",
        "latent_cross_attn_depth",
        "camera_normalization_method",
        "camera_scale_method",
        "img_size",
        "num_views",
        "num_input_views",
    ):
        if meta.get(field) is not None:
            value = meta[field]
            if field == "img_size":
                value = tuple(int(v) for v in value)
            setattr(opt, field, type(getattr(opt, field))(value))
    if meta.get("prompt_unfreeze_tokengs") is not None:
        opt.prompt_unfreeze_tokengs = bool(meta["prompt_unfreeze_tokengs"])
    if meta.get("token_dim") is not None:
        opt.token_dim = int(meta["token_dim"])


def _project_gs_to_views(means, cam_views, intrinsics, image_size):
    """Project GS centers into views; returns (px, py, z, valid) [V,N]."""
    height, width = int(image_size[0]), int(image_size[1])
    view_count = cam_views.shape[0]
    world_to_cam = cam_views.transpose(-1, -2).float()  # [V,4,4]
    homo = torch.cat([means.float(), torch.ones_like(means[..., :1])], dim=-1)
    cam = torch.einsum("vij,nj->vni", world_to_cam, homo)
    z = cam[..., 2]
    fx = intrinsics[:, 0]
    fy = intrinsics[:, 1]
    cx = intrinsics[:, 2]
    cy = intrinsics[:, 3]
    px = fx[:, None] * cam[..., 0] / z.clamp_min(1e-6) + cx[:, None]
    py = fy[:, None] * cam[..., 1] / z.clamp_min(1e-6) + cy[:, None]
    valid = (z > 0.01) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    return px, py, valid


def _compute_metrics(
    model,
    model_input,
    data,
    opt,
    num_gaussians_per_token,
    knn_sample: int = 4096,
    knn_k: int = 16,
    min_gt_pixels: int = 32,
) -> dict:
    independent = bool(
        getattr(opt, "instance_branch_independent", False)
        or getattr(opt, "instance_branch_token_units", False)
    )
    with torch.no_grad():
        if independent:
            branch = model.instance_branch
            model(data, compute_quality_metrics=False)  # fills branch.last_*
            gaussians = branch.last_instance_gaussians
            gp = branch.last_group_probs
            per_anchor = getattr(branch, "num_gaussians_per_anchor", None)
            if (
                per_anchor is not None
                and gp.shape[1] == gaussians.shape[1] // per_anchor
            ):
                gaussian_group_probs = gp.repeat_interleave(per_anchor, dim=1)
            else:
                gaussian_group_probs = gp
        else:
            reconstruction, gs_token_hidden, _ = (
                model._forward_prompt_reconstruction(model_input)
            )
            gaussians = reconstruction.gaussians
            group_probs, _, _ = model._instance_group_head_forward(
                gs_token_hidden, gaussians, data=data
            )
            if group_probs.shape[1] == gaussians.shape[1]:
                gaussian_group_probs = group_probs
            else:
                gaussian_group_probs = group_probs.repeat_interleave(
                    num_gaussians_per_token, dim=1
                )

        # Per-GS argmax label (last channel = void).
        gs_labels = torch.argmax(gaussian_group_probs[0], dim=-1)  # [N]
        num_groups = gaussian_group_probs.shape[-1] - 1
        void_channel = num_groups
        void_share = float((gs_labels == void_channel).float().mean())
        gs_conf = gaussian_group_probs[0].max(dim=-1).values

        means = gaussians[0, :, :3].float()  # [N,3]
        n_gs = means.shape[0]

        # --- 3D k-NN same-group rate ---
        idx = torch.randperm(n_gs, device=means.device)[: min(knn_sample, n_gs)]
        sampled_means = means[idx]
        sampled_labels = gs_labels[idx]
        dist = torch.cdist(sampled_means, sampled_means)  # [S,S]
        nn = dist.topk(knn_k + 1, dim=-1, largest=False).indices[:, 1:]
        non_void = sampled_labels != void_channel
        same = (
            sampled_labels[non_void].unsqueeze(1)
            == sampled_labels[nn[non_void]].float()
        ).float()
        knn_same_group = float(same.mean()) if same.numel() else float("nan")

        # --- token purity ---
        token_labels = gs_labels.view(
            1, -1, num_gaussians_per_token
        )  # [1,T,P]
        counts = torch.zeros(
            token_labels.shape[1], void_channel + 1, device=gs_labels.device
        )
        counts.scatter_add_(
            1,
            token_labels[0],
            torch.ones_like(token_labels[0], dtype=torch.float32),
        )
        token_purity = float(
            (counts.max(dim=-1).values / num_gaussians_per_token).mean()
        )

        # --- context-view rendering + projections ---
        cam_view = data["cam_view_input"].float()  # [B,V,4,4]
        intrinsics = data["intrinsics_input"].float()  # [B,V,4]
        render = model.gs.render_feature_channels(
            gaussians,
            gaussian_group_probs,
            cam_view,
            intrinsics=intrinsics,
            opacity_scale=float(getattr(opt, "instance_group_render_scale", 1.0)),
        )
        rendered_groups = render["images_pred"][0]  # [V,G,H,W]
        rendered_alpha = render["alphas_pred"][0]  # [V,1,H,W]
        rendered_channels = rendered_groups / (rendered_alpha + 1e-5)
        rendered_probs = rendered_channels / rendered_channels.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        rendered_labels = torch.argmax(rendered_probs, dim=1)  # [V,H,W]
        view_count = cam_view.shape[1]
        px, py, valid = _project_gs_to_views(
            means,
            cam_view[0],
            intrinsics[0],
            tuple(opt.img_size),
        )  # [V,N]

        # --- rendered vs GS label agreement ---
        px_l = px.clamp(0, int(opt.img_size[1]) - 1).long()
        py_l = py.clamp(0, int(opt.img_size[0]) - 1).long()
        rendered_at_gs = torch.stack(
            [
                rendered_labels[v][py_l[v], px_l[v]]
                for v in range(view_count)
            ]
        )  # [V,N]
        vis = valid & (gs_labels.unsqueeze(0) != void_channel)
        agree = (rendered_at_gs == gs_labels.unsqueeze(0))
        rendered_gs_agreement = (
            float(agree[vis].float().mean()) if vis.any() else float("nan")
        )

        # --- cross-view rendered agreement ---
        cross_vals = []
        for v1 in range(view_count):
            for v2 in range(v1 + 1, view_count):
                both = valid[v1] & valid[v2] & (
                    gs_labels != void_channel
                )
                if not both.any():
                    continue
                same_label = (
                    rendered_at_gs[v1][both] == rendered_at_gs[v2][both]
                )
                cross_vals.append(float(same_label.float().mean()))
        cross_view_agreement = (
            float(np.mean(cross_vals)) if cross_vals else float("nan")
        )

        # --- GT-instance fragmentation / purity / merging (context views) ---
        labels = data["instance_label_input"][0].long()  # [V,H,W]
        fragmentation = []
        purity = []
        group_instances: dict[int, set] = {}
        for v in range(view_count):
            inst_ids = torch.unique(labels[v])
            for iid in inst_ids.tolist():
                if iid in (0, 255, -1):
                    continue
                mask = labels[v] == iid
                if int(mask.sum()) < min_gt_pixels:
                    continue
                in_mask = (
                    valid[v]
                    & mask[py_l[v].clamp(0, labels.shape[1] - 1), px_l[v].clamp(0, labels.shape[2] - 1)]
                )
                votes = gs_labels[in_mask]
                votes = votes[votes != void_channel]
                if votes.numel() == 0:
                    continue
                uniq, counts_ = torch.unique(votes, return_counts=True)
                fragmentation.append(float(uniq.numel()))
                purity.append(float(counts_.float().max() / votes.numel()))
                for g in uniq.tolist():
                    group_instances.setdefault(g, set()).add((v, iid))
        gt_fragmentation = (
            float(np.mean(fragmentation)) if fragmentation else float("nan")
        )
        gt_purity = float(np.mean(purity)) if purity else float("nan")
        per_group_counts = [
            len(v) for v in group_instances.values()
        ]
        gt_merging = (
            float(np.mean(per_group_counts)) if per_group_counts else float("nan")
        )

        stats = {
            "num_gaussians": int(n_gs),
            "void_share": void_share,
            "mean_gs_conf": float(gs_conf.mean()),
            "knn3d_same_group": knn_same_group,
            "token_purity": token_purity,
            "rendered_gs_agreement": rendered_gs_agreement,
            "cross_view_agreement": cross_view_agreement,
            "gt_fragmentation": gt_fragmentation,
            "gt_purity": gt_purity,
            "gt_merging": gt_merging,
            "n_gt_instances_covered": len(group_instances),
        }
        return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument(
        "--backbone-resume",
        default="",
        help="Override the frozen backbone path recovered from config.",
    )
    parser.add_argument(
        "--lsm_manifest",
        default=(
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "lsm_instance_eval_manifest.json"
        ),
    )
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=15)
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--knn_sample", type=int, default=4096)
    parser.add_argument("--knn_k", type=int, default=16)
    parser.add_argument("--ttt_steps", type=int, default=0)
    parser.add_argument("--ttt_lr", type=float, default=1e-3)
    args = parser.parse_args()

    opt = config_defaults["eval_scannet_lsm_instance"]
    opt.model_type = args.model_type
    opt.resume = args.resume
    opt.workspace = args.workspace
    opt.experiment_name = Path(args.workspace).name
    opt.num_input_views = int(args.num_input_views)
    opt.num_views = int(args.num_views)
    opt.dataset_kwargs = {
        **dict(opt.dataset_kwargs or {}),
        "lsm_manifest_path": args.lsm_manifest,
    }
    _load_arch_from_checkpoint(args, opt)
    if args.backbone_resume:
        opt.backbone_resume = args.backbone_resume

    Path(args.workspace).mkdir(parents=True, exist_ok=True)
    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())

    from safetensors.torch import load_file

    model = model_registry[opt.model_type](opt)
    resume_ckpt = load_file(args.resume, device="cpu")
    torch.nn.Module.load_state_dict(model, resume_ckpt, strict=False)
    if not any(key.startswith("enc_dec_backbone.") for key in resume_ckpt):
        backbone_path = str(getattr(opt, "backbone_resume", "") or "")
        if backbone_path and Path(backbone_path).is_file():
            backbone_ckpt = load_file(backbone_path, device="cpu")
            frozen_prefixes = (
                "enc_dec_backbone.",
                "patch_embed.",
                "patch_plucker_embed.",
                "activation_head.",
                "anchor_pos_encoder.",
            )
            native_state = torch.nn.Module.state_dict(model)
            loadable = {
                key: value
                for key, value in backbone_ckpt.items()
                if (key.startswith(frozen_prefixes) or key == "gs_tokens")
                and key in native_state
                and native_state[key].shape == value.shape
            }
            torch.nn.Module.load_state_dict(model, loadable, strict=False)
            print(
                f"[diag-3d] loaded {len(loadable)} backbone keys from "
                f"{backbone_path}"
            )
        else:
            print("[diag-3d] WARNING: no backbone loaded for frozen recipe")
    model.eval()
    model = model.cuda()
    num_gaussians_per_token = opt.dec_patch_size**2
    if (
        args.ttt_steps > 0
        and getattr(opt, "instance_branch_independent", False)
    ):
        print("[diag-3d] TTT variant not supported for the independent branch; disabling")
        args.ttt_steps = 0

    def _run_ttt(model, data, model_input) -> None:
        with torch.no_grad():
            reconstruction, gs_token_hidden, _ = (
                model._forward_prompt_reconstruction(model_input)
            )
            gaussians = reconstruction.gaussians
        saved = {}
        for name, parameter in model.named_parameters():
            saved[name] = parameter.requires_grad
            parameter.requires_grad_(False)
        for parameter in model.instance_group_head.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(
            [
                p
                for p in model.instance_group_head.parameters()
                if p.requires_grad
            ],
            lr=args.ttt_lr,
        )
        cam_view = data["cam_view_input"]
        intrinsics = data["intrinsics_input"]
        labels = data["instance_label_input"]
        for _ in range(args.ttt_steps):
            optimizer.zero_grad()
            loss, _ = model.instance_group_loss_on_views(
                gs_token_hidden, gaussians, cam_view, intrinsics, labels
            )
            loss.backward()
            optimizer.step()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(saved[name])

    per_scene = {}
    for i, data in enumerate(test_loader):
        if args.max_scenes > 0 and i >= args.max_scenes:
            break
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        scene_name = data["scene_name"][0]
        model_input, _ = split_data(data, opt)
        t0 = time.time()
        if args.ttt_steps > 0:
            _run_ttt(model, data, model_input)
        with torch.no_grad():
            stats = _compute_metrics(
                model,
                model_input,
                data,
                opt,
                num_gaussians_per_token,
                knn_sample=args.knn_sample,
                knn_k=args.knn_k,
            )
        per_scene[scene_name] = stats
        print(
            f"[diag-3d] {scene_name}: "
            + " ".join(f"{k}={v:.3f}" for k, v in stats.items())
            + f" ({time.time() - t0:.1f}s)"
        )

    agg = {}
    for key in (
        "void_share",
        "mean_gs_conf",
        "knn3d_same_group",
        "token_purity",
        "rendered_gs_agreement",
        "cross_view_agreement",
        "gt_fragmentation",
        "gt_purity",
        "gt_merging",
    ):
        values = [per_scene[s][key] for s in per_scene]
        values = [v for v in values if v == v]  # drop NaN
        agg[key] = (
            {"mean": float(np.mean(values)), "std": float(np.std(values))}
            if values
            else None
        )
    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "num_scenes": len(per_scene),
        "ttt_steps": args.ttt_steps,
        "aggregate": agg,
        "per_scene": per_scene,
    }
    out = Path(args.workspace) / "diag_3d_consistency.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n===== aggregate =====")
    for key, value in agg.items():
        print(f"  {key}: {value}")
    print(f"\n[diag-3d] wrote {out}")


if __name__ == "__main__":
    main()
