"""Inspect rendered instance-group probability maps on LSM eval scenes.

Debug-only (not the eval protocol): runs the model on a few eval scenes and
dumps per-view argmax statistics -- void share, number of active groups,
group pixel counts, and each active group's best IoU against the ScanNet
instance masks -- to understand why predicted instance counts are far below
GT counts (group collapse / void dominance).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch._dynamo.config.disable = True  # eager fallback; avoids inductor compile hangs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.options import config_defaults
from tokengs.utils.instance_ap import (
    gt_masks_from_instance_map,
    iou_matrix_vectorized,
)


class _LocalAccelerator:
    is_main_process = True


def _load_checkpoint_arch(args, opt) -> None:
    checkpoint_path = Path(args.resume)
    meta = {}
    metadata_path = checkpoint_path.parent / (
        "metadata_step_"
        + checkpoint_path.stem.replace("model_step_", "")
        + ".json"
    )
    if not metadata_path.is_file():
        metadata_path = checkpoint_path.parent / "metadata_best.json"
    if not metadata_path.is_file():
        metadata_path = checkpoint_path.parent / "metadata.json"
    if metadata_path.is_file():
        meta = json.load(open(metadata_path, encoding="utf-8"))
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
        except Exception as exc:  # pragma: no cover - fallback only
            print(f"[group-debug] config.yaml parse failed: {exc}")
    if meta.get("instance_group_num_groups") is not None:
        opt.instance_group_num_groups = int(meta["instance_group_num_groups"])
    if meta.get("instance_group_decoder") is not None:
        opt.instance_group_decoder = bool(meta["instance_group_decoder"])
    if meta.get("instance_group_decoder_layers") is not None:
        opt.instance_group_decoder_layers = int(
            meta["instance_group_decoder_layers"]
        )
    if meta.get("instance_group_use_anchor_pos") is not None:
        opt.instance_group_use_anchor_pos = bool(
            meta["instance_group_use_anchor_pos"]
        )
    if meta.get("num_gs_tokens") is not None:
        opt.num_gs_tokens = int(meta["num_gs_tokens"])
    if meta.get("num_dynamic_gs_tokens") is not None:
        opt.num_dynamic_gs_tokens = int(meta["num_dynamic_gs_tokens"])
    if meta.get("prompt_clip_model_path") is not None:
        opt.prompt_clip_model_path = str(meta["prompt_clip_model_path"])
    if meta.get("semantic_v4_feature_dim") is not None:
        opt.semantic_v4_feature_dim = int(meta["semantic_v4_feature_dim"])
    if meta.get("semantic_v4_use_geometry") is not None:
        opt.semantic_v4_use_geometry = bool(meta["semantic_v4_use_geometry"])
    if meta.get("semantic_v4_teacher_projection") is not None:
        opt.semantic_v4_teacher_projection = str(
            meta["semantic_v4_teacher_projection"]
        )
    if meta.get("prompt_unfreeze_tokengs") is not None:
        opt.prompt_unfreeze_tokengs = bool(meta["prompt_unfreeze_tokengs"])
    if meta.get("lseg_checkpoint_path") is not None:
        opt.lseg_checkpoint_path = str(meta["lseg_checkpoint_path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--lsm_manifest",
        default=(
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "lsm_instance_eval_manifest.json"
        ),
    )
    parser.add_argument("--max_scenes", type=int, default=2)
    parser.add_argument("--num_groups", type=int, default=128)
    parser.add_argument(
        "--render_scale",
        type=float,
        default=None,
        help="Override instance_group_render_scale (opacity sharpening).",
    )
    args = parser.parse_args()

    opt = config_defaults["eval_scannet_lsm_instance"]
    opt.resume = args.resume
    opt.instance_group_num_groups = args.num_groups
    opt.num_input_views = 8
    opt.num_views = 15
    opt.dataset_kwargs = {
        **dict(opt.dataset_kwargs or {}),
        "lsm_manifest_path": args.lsm_manifest,
    }
    _load_checkpoint_arch(args, opt)
    if args.render_scale is not None:
        opt.instance_group_render_scale = args.render_scale
    num_groups = int(opt.instance_group_num_groups)
    void_channel = num_groups

    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    from safetensors.torch import load_file

    model = model_registry[opt.model_type](opt)
    model.load_state_dict(load_file(args.resume, device="cpu"), strict=True)
    model.eval()
    model = model.cuda()

    summary = {}
    with torch.inference_mode():
        for i, data in enumerate(test_loader):
            if args.max_scenes > 0 and i >= args.max_scenes:
                break
            data = {
                k: (v.cuda() if torch.is_tensor(v) else v)
                for k, v in data.items()
            }
            out = model(data, compute_quality_metrics=False)
            probability = out["rendered_instance_group_probability"]
            token_probs = out["instance_group_probabilities"]  # [B,T,G+1]
            instance_labels = data["instance_label_output"].long()
            scene_name = data["scene_name"][0]
            _, _, view_count, _, height, width = probability.shape
            per_view = []
            distinct_groups: set[int] = set()
            token_argmax = torch.argmax(token_probs[0], dim=-1).cpu().numpy()
            token_active = [
                int(g)
                for g in range(num_groups + 1)
                if (token_argmax == g).sum() > 0 and g != void_channel
            ]
            token_void_share = float((token_argmax == void_channel).mean())
            token_max_prob = float(token_probs[0].max(dim=-1).values.mean())
            token_void_prob = float(token_probs[0][:, void_channel].mean())
            token_top2_margin = float(
                (
                    token_probs[0].topk(2, dim=-1).values[:, 0]
                    - token_probs[0].topk(2, dim=-1).values[:, 1]
                ).mean()
            )
            token_counts = {
                int(g): int((token_argmax == g).sum())
                for g in np.unique(token_argmax)
                if int(g) != void_channel
            }
            for v in range(view_count):
                probs = probability[0, :, v, 0].float().cpu().numpy()
                argmax = np.argmax(probs, axis=0)
                void_share = float((argmax == void_channel).mean())
                per_pixel_max = float(probs.max(axis=0).mean())
                logp = np.log(np.clip(probs, 1e-8, 1.0))
                per_pixel_entropy = float(
                    -(probs * logp).sum(axis=0).mean()
                )
                active = [
                    g
                    for g in range(num_groups)
                    if (argmax == g).sum() > 0
                ]
                distinct_groups.update(active)
                sizes = {
                    g: int((argmax == g).sum()) for g in active
                }
                top = sorted(sizes.items(), key=lambda kv: -kv[1])[:10]
                gt_map = instance_labels[0, v].cpu().numpy()
                gt_masks = gt_masks_from_instance_map(gt_map)
                gt_cover = float(
                    np.logical_or.reduce(gt_masks).mean()
                    if gt_masks
                    else 0.0
                )
                best_iou = {}
                if gt_masks:
                    candidates = [
                        (g, argmax == g)
                        for g in active
                        if sizes[g] >= 16
                    ]
                    if candidates:
                        matrix = iou_matrix_vectorized(
                            [mask for _, mask in candidates], gt_masks
                        )
                        for k, (g, _) in enumerate(candidates):
                            row = matrix[k]
                            best_iou[g] = float(
                                row.max() if row.size else 0.0
                            )
                per_view.append(
                    {
                        "view": int(v),
                        "void_share": void_share,
                        "void_mean_prob": float(probs[void_channel].mean()),
                        "per_pixel_max_prob": per_pixel_max,
                        "per_pixel_entropy": per_pixel_entropy,
                        "num_active_groups": len(active),
                        "group_pixels": sizes,
                        "top10_groups": top,
                        "gt_instances": len(gt_masks),
                        "gt_cover_frac": gt_cover,
                        "active_group_best_iou_vs_gt": best_iou,
                    }
                )
            summary[str(scene_name)] = {
                "per_view": per_view,
                "distinct_active_groups": sorted(distinct_groups),
                "token_level_active_groups": sorted(token_active),
                "token_level_void_share": token_void_share,
                "token_level_group_counts": token_counts,
                "token_max_prob": token_max_prob,
                "token_void_prob": token_void_prob,
                "token_top2_margin": token_top2_margin,
            }
            print(
                f"[group-debug] {scene_name}: distinct_groups="
                f"{len(distinct_groups)} "
                f"per-view active={[pv['num_active_groups'] for pv in per_view]} "
                f"void_share={[round(pv['void_share'], 3) for pv in per_view]} "
                f"px_max={[round(pv['per_pixel_max_prob'], 4) for pv in per_view]} "
                f"px_entropy={[round(pv['per_pixel_entropy'], 3) for pv in per_view]} "
                f"token_active={len(token_active)} "
                f"token_void={token_void_share:.3f} "
                f"token_max_prob={token_max_prob:.4f} "
                f"token_void_prob={token_void_prob:.4f} "
                f"token_top2_margin={token_top2_margin:.4f} "
                f"token_top={sorted(token_counts.items(), key=lambda kv: -kv[1])[:5]} "
                f"gt_per_view={[pv['gt_instances'] for pv in per_view]} "
                f"top_sizes={[pv['top10_groups'][:3] for pv in per_view]}"
            )

    if args.out:
        Path(args.out).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[group-debug] wrote {args.out}")


if __name__ == "__main__":
    main()
