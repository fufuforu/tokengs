"""Visualize "one Token covers multiple GT instances".

Offline only (no model modification, no training). Uses the frozen
wide7l@8000 + pgr3df2 best checkpoint on selected LSM scenes:
  - scene-level token-purity histogram,
  - for representative tokens (pure / 2-major mixed / severely mixed):
      * the token's 64 GS projected into a reference view, colored by the
        majority-vote GT instance, overlaid on the GT instance map,
      * their 3D positions colored by GT instance (grey = scene context),
  - global pooled histograms over all 40 scenes.

Usage:
    python scripts/visualize_token_instance_mixing.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch._dynamo.config.disable = True
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_token_instance_mixing as dtm

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.input_types import split_data
from tokengs.models.instance_group_loss import (
    _gs_majority_target,
    _project_gs_to_views,
)
from tokengs.options import config_defaults


class _LocalAccelerator:
    is_main_process = True


_TAB20 = plt.get_cmap("tab20")


def _instance_color_map(instance_ids) -> dict[int, tuple]:
    ids = sorted(int(i) for i in instance_ids if i > 0)
    return {i: _TAB20(idx % 20) for idx, i in enumerate(ids)}


def _project_single_view(means, cam_view, intrinsics, image_size):
    """Project GS centers into one view; returns (px, py, valid)."""
    height, width = int(image_size[0]), int(image_size[1])
    world_to_cam = cam_view.transpose(-1, -2).float()  # [1,4,4]
    homo = torch.cat([means.float(), torch.ones_like(means[..., :1])], dim=-1)
    cam = torch.einsum("ij,nj->ni", world_to_cam[0], homo)
    z = cam[..., 2]
    fx, fy, cx, cy = intrinsics[0]
    px = fx * cam[..., 0] / z.clamp_min(1e-6) + cx
    py = fy * cam[..., 1] / z.clamp_min(1e-6) + cy
    valid = (z > 0.01) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    return px, py, valid


def _draw_gt_overlay(ax, gt_map, covered_ids, color_map):
    """Overlay the GT instance map on ax; highlight covered instances."""
    overlay = np.zeros((*gt_map.shape, 4), dtype=np.float32)
    overlay[..., 3] = 0.0
    for iid in covered_ids:
        mask = gt_map == iid
        rgba = to_rgba(color_map[int(iid)])
        overlay[mask] = (rgba[0], rgba[1], rgba[2], 0.45)
    ax.imshow(overlay)


def _select_tokens(tokens, n_voted_min=16):
    usable = [t for t in tokens if t.get("n_voted_gs", 0) >= n_voted_min]
    if not usable:
        usable = [t for t in tokens if t.get("n_voted_gs", 0) >= 8]
    if not usable:
        return None, None, None

    def pick(pred):
        cand = [t for t in usable if pred(t)]
        return max(cand, key=lambda t: t["n_voted_gs"]) if cand else None

    pure = pick(lambda t: t["purity"] >= 0.9)
    mixed2 = pick(lambda t: t["mixed_major"] and t["n_instances"] <= 4)
    severe = pick(lambda t: t["n_instances"] >= 10)
    return pure, mixed2, severe


def _plot_token_figure(
    token,
    gs_gt_token,
    px_t,
    py_t,
    valid_t,
    means_t,
    gt_map,
    color_map,
    scene_name,
    ref_view,
    rgb,
    out_path,
    context_means,
):
    covered = [int(i) for i in torch.unique(gs_gt_token[valid_t]) if i > 0]
    fig = plt.figure(figsize=(20, 6))
    ax_rgb = fig.add_subplot(1, 3, 1)
    ax_top = fig.add_subplot(1, 3, 2)
    ax_3d = fig.add_subplot(1, 3, 3, projection="3d")
    # Panel 1: RGB view with the token's GS colored by GT instance.
    ax_rgb.imshow(rgb.permute(1, 2, 0).cpu().numpy())
    _draw_gt_overlay(ax_rgb, gt_map, covered, color_map)
    vis = valid_t
    for iid in covered:
        sel = vis & (gs_gt_token == iid)
        if sel.sum() == 0:
            continue
        ax_rgb.scatter(
            px_t[sel].cpu().numpy(),
            py_t[sel].cpu().numpy(),
            s=42,
            color=color_map[iid],
            edgecolors="black",
            linewidths=0.5,
            label=f"inst {iid}",
        )
    ax_rgb.set_title(
        f"{scene_name} view{ref_view} | token {token['token']}\n"
        f"purity={token.get('purity', float('nan')):.2f} "
        f"n_inst={token.get('n_instances', 'na')} "
        f"entropy={token.get('entropy', float('nan')):.2f} "
        f"spread_rms={token['spread_rms']:.2f}",
        fontsize=10,
    )
    ax_rgb.legend(fontsize=7, loc="upper right")
    # Panel 2: top-down (x-z) view of the token's GS.
    sub = context_means[
        torch.randperm(context_means.shape[0])[: min(3000, context_means.shape[0])]
    ]
    ax_top.scatter(
        sub[:, 0].cpu().numpy(), sub[:, 2].cpu().numpy(), s=2, c="lightgrey"
    )
    for iid in covered:
        sel = gs_gt_token == iid
        if sel.sum() == 0:
            continue
        ax_top.scatter(
            means_t[sel, 0].cpu().numpy(),
            means_t[sel, 2].cpu().numpy(),
            s=48,
            color=color_map[iid],
            edgecolors="black",
            linewidths=0.5,
            label=f"inst {iid}",
        )
    ax_top.set_title("top-down (x,z)", fontsize=10)
    ax_top.legend(fontsize=7)
    # Panel 3: 3D view.
    ax_3d.scatter(
        sub[:, 0].cpu().numpy(),
        sub[:, 1].cpu().numpy(),
        sub[:, 2].cpu().numpy(),
        s=1,
        c="lightgrey",
        alpha=0.4,
    )
    for iid in covered:
        sel = gs_gt_token == iid
        if sel.sum() == 0:
            continue
        ax_3d.scatter(
            means_t[sel, 0].cpu().numpy(),
            means_t[sel, 1].cpu().numpy(),
            means_t[sel, 2].cpu().numpy(),
            s=40,
            color=color_map[iid],
            edgecolors="black",
            linewidths=0.5,
            label=f"inst {iid}",
        )
    ax_3d.set_title("3D (x,y,z)", fontsize=10)
    ax_3d.view_init(elev=18, azim=-60)
    ax_3d.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=(
            "/space0/mawb/tokengs/workspace/"
            "semantic_v6_open_vocab_pgr3df2_train_12000/"
            "checkpoints/model_step_012000.safetensors"
        ),
    )
    parser.add_argument(
        "--workspace", default="workspace/viz_token_mixing"
    )
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument(
        "--scenes",
        default="scene0692_00,scene0701_00,scene0686_01,scene0703_00",
    )
    parser.add_argument("--ref-view", type=int, default=3)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--min-votes", type=int, default=8)
    parser.add_argument("--major-share", type=float, default=0.2)
    parser.add_argument("--purity-threshold", type=float, default=0.9)
    parser.add_argument(
        "--lsm_manifest",
        default=(
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "lsm_instance_eval_manifest.json"
        ),
    )
    args = parser.parse_args()

    selected = {s.strip() for s in args.scenes.split(",") if s.strip()}
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = config_defaults["eval_scannet_lsm_instance"]
    opt.model_type = args.model_type
    opt.resume = args.resume
    opt.workspace = args.workspace
    opt.experiment_name = args.workspace.split("/")[-1]
    opt.num_input_views = 8
    opt.num_views = 15
    opt.dataset_kwargs = {
        **dict(opt.dataset_kwargs or {}),
        "lsm_manifest_path": args.lsm_manifest,
    }
    dtm._load_arch_from_checkpoint(args, opt)
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
    model.eval()
    model = model.cuda()
    gaussians_per_token = opt.dec_patch_size**2

    global_records = []
    selected_data = {}
    for i, data in enumerate(test_loader):
        if args.max_scenes > 0 and i >= args.max_scenes:
            break
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        scene_name = data["scene_name"][0]
        with torch.no_grad():
            model_input, _ = split_data(data, opt)
            reconstruction, gs_token_hidden, _ = (
                model._forward_prompt_reconstruction(model_input)
            )
            gaussians = reconstruction.gaussians
            means = gaussians[0, :, :3].float()
            group_probs, _, _ = model._instance_group_head_forward(
                gs_token_hidden, gaussians, data=data
            )
            if group_probs.shape[1] == gaussians.shape[1]:
                gs_group = torch.argmax(group_probs[0], dim=-1)
            else:
                gs_group = torch.argmax(
                    group_probs[0].repeat_interleave(
                        gaussians_per_token, dim=0
                    ),
                    dim=-1,
                )
            cam_views = torch.cat(
                [data["cam_view_input"], data["cam_view"]], dim=1
            )[0]
            intrinsics_all = torch.cat(
                [data["intrinsics_input"], data["intrinsics"]], dim=1
            )[0]
            labels_all = torch.cat(
                [
                    data["instance_label_input"],
                    data["instance_label_output"],
                ],
                dim=1,
            )[0]
            ids, valid = _project_gs_to_views(
                means, cam_views, intrinsics_all, labels_all, tuple(opt.img_size)
            )
            gs_gt = _gs_majority_target(ids, valid)
            scene_rms = float(
                (means - means.mean(dim=0)).square().sum(-1).mean().sqrt()
            )
        tokens = dtm._token_stats(
            gs_gt,
            gs_group,
            means,
            gaussians_per_token,
            scene_rms,
            args.major_share,
            args.min_votes,
        )
        for t in tokens:
            if "purity" in t:
                global_records.append(
                    {
                        "purity": t["purity"],
                        "n_instances": t["n_instances"],
                        "entropy": t["entropy"],
                        "spread_relative_rms": t["spread_relative_rms"],
                        "scene": scene_name,
                    }
                )
        if scene_name in selected:
            pure, mixed2, severe = _select_tokens(tokens)
            selected_data[scene_name] = {
                "tokens": tokens,
                "gs_gt": gs_gt,
                "means": means,
                "scene_rms": scene_rms,
                "rgb": data["images_input"][0, args.ref_view].float(),
                "gt_map": data["instance_label_input"][0, args.ref_view].long(),
                "cam_view": data["cam_view_input"][0, args.ref_view : args.ref_view + 1],
                "intrinsics": data["intrinsics_input"][0, args.ref_view : args.ref_view + 1],
                "selected": {"pure": pure, "mixed2": mixed2, "severe": severe},
            }
        print(f"[viz-token] {scene_name} ({len(tokens)} tokens)")

    # ---- global figures ----
    purities = np.array([r["purity"] for r in global_records])
    n_inst = np.array([r["n_instances"] for r in global_records])
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
    axes[0].hist(purities, bins=40, color="#4C72B0")
    axes[0].axvline(0.9, color="r", ls="--", label="pure threshold")
    axes[0].set_title(
        f"token purity (n={len(purities)}, pure={np.mean(purities>=0.9)*100:.1f}%)"
    )
    axes[0].legend()
    axes[1].hist(n_inst, bins=60, color="#55A868")
    axes[1].set_title(f"instances per token (mean={n_inst.mean():.1f})")
    axes[2].scatter(n_inst, purities, s=3, alpha=0.15)
    axes[2].set_xlabel("n GT instances"); axes[2].set_ylabel("purity")
    axes[2].set_title("purity vs n_instances")
    fig.tight_layout()
    fig.savefig(out_dir / "global_token_mixing.png", dpi=110)
    plt.close(fig)

    # ---- per-scene figures ----
    summary = {}
    for scene_name, sd in selected_data.items():
        tokens = sd["tokens"]
        usable = [t for t in tokens if "purity" in t]
        scene_purities = np.array([t["purity"] for t in usable])
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.hist(scene_purities, bins=25, color="#4C72B0")
        ax.axvline(0.9, color="r", ls="--")
        ax.set_title(
            f"{scene_name}: token purity  "
            f"(pure={np.mean(scene_purities>=0.9)*100:.0f}%, "
            f"mixed_major={np.mean([t['mixed_major'] for t in usable])*100:.0f}%)"
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{scene_name}_purity_hist.png", dpi=110)
        plt.close(fig)

        gs_gt = sd["gs_gt"]
        means = sd["means"]
        gt_map = sd["gt_map"].cpu().numpy()
        color_map = _instance_color_map(
            torch.unique(gs_gt[gs_gt > 0]).tolist()
        )
        px, py, valid = _project_single_view(
            means, sd["cam_view"], sd["intrinsics"], tuple(opt.img_size)
        )
        context_means = means
        summary[scene_name] = {}
        for kind, token in sd["selected"].items():
            if token is None:
                continue
            t_idx = token["token"]
            sel = torch.arange(64) + t_idx * 64
            means_t = means[sel]
            gs_gt_t = gs_gt[sel]
            px_t, py_t, valid_t = px[sel], py[sel], valid[sel]
            out_png = out_dir / f"{scene_name}_token{t_idx}_{kind}.png"
            _plot_token_figure(
                token,
                gs_gt_t,
                px_t,
                py_t,
                valid_t,
                means_t,
                gt_map,
                color_map,
                scene_name,
                args.ref_view,
                sd["rgb"],
                out_png,
                context_means,
            )
            summary[scene_name][kind] = {
                "token": t_idx,
                "purity": token.get("purity"),
                "n_instances": token.get("n_instances"),
                "entropy": token.get("entropy"),
                "spread_rms": token["spread_rms"],
                "spread_relative_rms": token["spread_relative_rms"],
                "png": str(out_png),
            }
            print(
                f"[viz-token] {scene_name} {kind} token={t_idx} "
                f"purity={token.get('purity')} n_inst={token.get('n_instances')}"
            )

    json_path = out_dir / "viz_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[viz-token] wrote figures + {json_path} -> {out_dir}")


if __name__ == "__main__":
    main()
