"""Offline diagnostic: how badly does Token granularity split GT instances?

No model modification, no training. Uses the frozen wide7l@8000 backbone +
pgr3df2 head (best checkpoint, AP50=0.232) on the LSM 40 held-out scenes.

For every Gaussian we know (a) its parent token (GS index // 64) and (b) its
ground-truth instance, obtained by projecting the GS center into all 15
labeled views and majority-voting the GT instance id (the exact convention of
the 3D anchor-level loss). For each token's 64 GS we then measure:
  - unique GT instance count,
  - purity (max-instance share of the voted GS),
  - instance entropy,
  - mixed flags (>=2 instances, >=2 major instances with share>=major_share),
  - cross-instance GS ratio (1 - purity),
  - 3D spread (RMS radius / max pairwise distance, plus scene-relative RMS),
  - predicted-group purity of the head's own assignment.

Outputs per-scene stats + a global summary, and answers the question:
"does token granularity seriously hinder instance grouping?"

Usage:
    python scripts/diagnose_token_instance_mixing.py \
        [--resume workspace/semantic_v6_open_vocab_pgr3df2_train_12000/checkpoints/model_step_012000.safetensors]
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
from tokengs.models.instance_group_loss import (
    _gs_majority_target,
    _project_gs_to_views,
)
from tokengs.options import config_defaults


class _LocalAccelerator:
    is_main_process = True


def _load_arch_from_checkpoint(args, opt) -> None:
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

            yaml.add_multi_constructor(
                "!dataclass:", _options_ctor, Loader=yaml.UnsafeLoader
            )
            yaml_cfg = yaml.load(
                config_yaml.read_text(encoding="utf-8"),
                Loader=yaml.UnsafeLoader,
            )
            for key, value in yaml_cfg.items():
                meta.setdefault(key, value)
        except Exception as exc:  # pragma: no cover
            print(f"[token-diag] config.yaml parse failed: {exc}")
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
        ("instance_branch_independent", bool),
        ("instance_branch_token_units", bool),
        ("instance_branch_num_groups", int),
        ("instance_branch_units_per_token", int),
        ("instance_branch_unit_feat_dim", int),
        ("instance_branch_unit_layers", int),
        ("instance_branch_unit_temp", float),
        ("instance_branch_unit_entropy", float),
        ("instance_branch_unit_compactness", float),
        ("instance_branch_unit_embedding", bool),
        ("instance_branch_unit_encoder", bool),
        ("instance_branch_embed_dim", int),
        ("instance_branch_embed_temp", float),
        ("instance_branch_embed_loss", float),
        ("instance_branch_unit_image", bool),
        ("instance_branch_unit_image_dim", int),
        ("instance_branch_pseudo_conf", float),
        ("instance_branch_pseudo_min_views", int),
        ("instance_branch_pseudo_unit_min_mass", float),
        ("instance_branch_cluster_eps", float),
        ("instance_branch_cluster_pos_weight", float),
        ("instance_branch_void_fg_share", float),
        ("instance_group_dense_decoder", bool),
        ("instance_group_dense_multiscale", bool),
        ("instance_branch_anchor_dim", int),
        ("instance_branch_num_heads", int),
        ("instance_branch_num_layers", int),
    ):
        if meta.get(field) is not None:
            setattr(opt, field, cast(meta[field]))
    if (
        getattr(opt, "instance_branch_independent", False)
        or getattr(opt, "instance_branch_token_units", False)
    ):
        opt.instance_group_num_groups = int(
            getattr(opt, "instance_branch_num_groups", 100)
        )
    if meta.get("prompt_unfreeze_tokengs") is not None:
        opt.prompt_unfreeze_tokengs = bool(meta["prompt_unfreeze_tokengs"])
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
    if meta.get("token_dim") is not None:
        opt.token_dim = int(meta["token_dim"])


def _token_stats(
    gs_gt: torch.Tensor,
    gs_group: torch.Tensor,
    means: torch.Tensor,
    gaussians_per_token: int,
    scene_rms: float,
    major_share: float,
    min_votes: int,
) -> list[dict]:
    """Per-token stats from per-GS GT instance ids and predicted groups."""
    n_tokens = gs_gt.numel() // gaussians_per_token
    gs_gt_t = gs_gt.view(n_tokens, gaussians_per_token)
    gs_group_t = gs_group.view(n_tokens, gaussians_per_token)
    means_t = means.view(n_tokens, gaussians_per_token, 3)
    tokens = []
    for t in range(n_tokens):
        voted = gs_gt_t[t]
        valid = voted > 0
        n_voted = int(valid.sum())
        row = {
            "token": t,
            "n_voted_gs": n_voted,
            "n_background_gs": int((voted == 0).sum()),
        }
        if n_voted >= min_votes:
            counts = torch.bincount(voted[valid])
            probs = counts.float() / n_voted
            n_inst = int(counts.numel())
            purity = float(probs.max())
            entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
            norm_entropy = entropy / np.log(n_inst) if n_inst > 1 else 0.0
            n_major = int((probs >= major_share).sum())
            row.update(
                {
                    "n_instances": n_inst,
                    "purity": purity,
                    "entropy": entropy,
                    "norm_entropy": norm_entropy,
                    "cross_gs_ratio": 1.0 - purity,
                    "mixed_any": n_inst >= 2,
                    "mixed_major": n_major >= 2,
                    "pred_group_purity": float(
                        torch.bincount(gs_group_t[t]).float().max() / 64
                    ),
                }
            )
        centers = means_t[t]
        centroid = centers.mean(dim=0)
        rms = float((centers - centroid).square().sum(-1).mean().sqrt())
        pairwise = torch.cdist(centers, centers)
        flat = pairwise[torch.triu(torch.ones(64, 64, dtype=torch.bool), 1)]
        row.update(
            {
                "spread_rms": rms,
                "spread_max_pair": float(flat.max()),
                "spread_p90_pair": float(torch.quantile(flat, 0.90)),
                "spread_relative_rms": rms / max(scene_rms, 1e-6),
            }
        )
        tokens.append(row)
    return tokens


def _scene_summary(tokens: list[dict], purity_threshold: float) -> dict:
    usable = [t for t in tokens if "purity" in t]
    if not usable:
        return {"n_tokens": len(tokens), "n_usable": 0}
    vals = {
        key: [t[key] for t in usable]
        for key in (
            "n_instances",
            "purity",
            "entropy",
            "norm_entropy",
            "cross_gs_ratio",
            "spread_rms",
            "spread_relative_rms",
            "spread_max_pair",
        )
    }

    def agg(values):
        values = sorted(values)
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
        }

    pure = sum(1 for t in usable if t["purity"] >= purity_threshold)
    mixed_any = sum(1 for t in usable if t["mixed_any"])
    mixed_major = sum(1 for t in usable if t["mixed_major"])
    pure_spread = [
        t["spread_rms"] for t in usable if t["purity"] >= purity_threshold
    ]
    mixed_spread = [t["spread_rms"] for t in usable if t["mixed_major"]]
    pred_pure = [t["pred_group_purity"] for t in usable]
    return {
        "n_tokens": len(tokens),
        "n_usable": len(usable),
        "n_background_only": len(tokens) - len(usable),
        "instance": {k: agg(v) for k, v in vals.items()},
        "pure_token_ratio": pure / len(usable),
        "mixed_any_ratio": mixed_any / len(usable),
        "mixed_major_ratio": mixed_major / len(usable),
        "pred_group_purity_mean": float(np.mean(pred_pure)),
        "pure_vs_mixed_spread_ratio": (
            float(np.mean(pure_spread))
            / max(float(np.mean(mixed_spread)), 1e-6)
            if pure_spread and mixed_spread
            else None
        ),
        "purity_hist": {
            f"{int(b*10)}0s": sum(
                1 for t in usable if int(t["purity"] * 10) == int(b * 10)
            )
            for b in range(10)
        },
    }


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
    parser.add_argument("--workspace", default="workspace/diag_token_mixing")
    parser.add_argument("--label", default="pgr3df2_token_mixing")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
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
    parser.add_argument("--min_votes", type=int, default=8)
    parser.add_argument("--major_share", type=float, default=0.2)
    parser.add_argument("--purity_threshold", type=float, default=0.9)
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
                f"[token-diag] loaded {len(loadable)} backbone keys from "
                f"{backbone_path}"
            )
    model.eval()
    model = model.cuda()
    gaussians_per_token = opt.dec_patch_size**2

    per_scene = {}
    for i, data in enumerate(test_loader):
        if args.max_scenes > 0 and i >= args.max_scenes:
            break
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        scene_name = data["scene_name"][0]
        t0 = time.time()
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

            # GT instance per GS: project into all 15 labeled views and
            # majority-vote (same convention as the 3D anchor-level loss).
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
        tokens = _token_stats(
            gs_gt,
            gs_group,
            means,
            gaussians_per_token,
            scene_rms,
            args.major_share,
            args.min_votes,
        )
        per_scene[scene_name] = _scene_summary(tokens, args.purity_threshold)
        s = per_scene[scene_name]
        print(
            f"[token-diag] {scene_name}: pure={s['pure_token_ratio']:.2f} "
            f"mixed_major={s['mixed_major_ratio']:.2f} "
            f"n_inst_mean={s['instance']['n_instances']['mean']:.2f} "
            f"purity_mean={s['instance']['purity']['mean']:.2f} "
            f"({time.time() - t0:.1f}s)"
        )

    # Global aggregation: mean over scenes (scene-macro) + token-pooled.
    keys = [
        "n_instances",
        "purity",
        "entropy",
        "norm_entropy",
        "cross_gs_ratio",
        "spread_rms",
        "spread_relative_rms",
    ]
    macro = {
        k: {
            stat: float(
                np.mean([per_scene[s]["instance"][k][stat] for s in per_scene])
            )
            for stat in ("mean", "median", "p90")
        }
        for k in keys
    }
    macro["pure_token_ratio"] = float(
        np.mean([per_scene[s]["pure_token_ratio"] for s in per_scene])
    )
    macro["mixed_any_ratio"] = float(
        np.mean([per_scene[s]["mixed_any_ratio"] for s in per_scene])
    )
    macro["mixed_major_ratio"] = float(
        np.mean([per_scene[s]["mixed_major_ratio"] for s in per_scene])
    )
    macro["pred_group_purity_mean"] = float(
        np.mean([per_scene[s]["pred_group_purity_mean"] for s in per_scene])
    )
    ratios = [
        per_scene[s]["pure_vs_mixed_spread_ratio"]
        for s in per_scene
        if per_scene[s]["pure_vs_mixed_spread_ratio"] is not None
    ]
    macro["pure_vs_mixed_spread_ratio"] = (
        float(np.mean(ratios)) if ratios else None
    )

    pure_tokens = sum(
        int(per_scene[s]["pure_token_ratio"] * per_scene[s]["n_usable"])
        for s in per_scene
    )
    mixed_tokens = sum(
        int(per_scene[s]["mixed_major_ratio"] * per_scene[s]["n_usable"])
        for s in per_scene
    )
    usable_tokens = sum(per_scene[s]["n_usable"] for s in per_scene)
    macro["pooled_pure_token_ratio"] = pure_tokens / max(usable_tokens, 1)
    macro["pooled_mixed_major_ratio"] = mixed_tokens / max(usable_tokens, 1)
    macro["n_usable_tokens_total"] = usable_tokens

    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "num_scenes": len(per_scene),
        "config": {
            "min_votes": args.min_votes,
            "major_share": args.major_share,
            "purity_threshold": args.purity_threshold,
        },
        "global": macro,
        "per_scene": per_scene,
    }
    out = Path(args.workspace) / "token_instance_mixing.json"
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n===== GLOBAL SUMMARY =====")
    for k in keys:
        print(f"  {k}: {macro[k]}")
    print(
        f"  pure_token_ratio(>=0.9): {macro['pure_token_ratio']:.3f} "
        f"(pooled {macro['pooled_pure_token_ratio']:.3f})"
    )
    print(
        f"  mixed_major_ratio(>=2 major): {macro['mixed_major_ratio']:.3f} "
        f"(pooled {macro['pooled_mixed_major_ratio']:.3f})"
    )
    print(
        f"  mixed_any_ratio(>=2 instances): {macro['mixed_any_ratio']:.3f}"
    )
    print(f"  pred_group_purity_mean: {macro['pred_group_purity_mean']:.3f}")
    print(
        f"  pure_vs_mixed_spread_ratio: "
        f"{macro['pure_vs_mixed_spread_ratio']:.3f}"
    )
    verdict_purity = macro["pure_token_ratio"]
    verdict_mixed = macro["mixed_major_ratio"]
    print("\n===== VERDICT =====")
    print(
        f"pure tokens (purity>=0.9): {verdict_purity*100:.1f}%  |  "
        f"mixed tokens (>=2 major GT instances): {verdict_mixed*100:.1f}%"
    )
    if verdict_purity < 0.5:
        print(
            "SEVERE: fewer than half the tokens are instance-pure; token "
            "granularity is a first-order obstacle to instance grouping."
        )
    elif verdict_mixed > 0.3:
        print(
            "MODERATE-SEVERE: a large share of tokens straddle >=2 instances."
        )
    elif verdict_purity < 0.75:
        print("MODERATE: many tokens are mixed, but the majority are pure.")
    else:
        print("MILD: token granularity is NOT the main obstacle.")
    print(f"\n[token-diag] wrote {out}")


if __name__ == "__main__":
    main()
