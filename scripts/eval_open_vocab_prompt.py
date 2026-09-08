#!/usr/bin/env python3
"""Per-prompt open-vocabulary evaluation for SemanticTokenGSv3.

Loads a trained v3 checkpoint and evaluates the prompt-conditioned binary mask
for every sample in the fixed 24-sample prompt proxy (text_only / image_only /
text_image_mixed, including cross-scene image queries). Reports per-mode and
per-class mask IoU / accuracy and writes an eval_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.options import config_defaults


C3G8_CLASS_NAMES = ("wall", "floor", "ceiling", "chair", "table", "sofa", "bed", "other")


class _LocalAccelerator:
    is_main_process = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True, help="v3 checkpoint (safetensors)")
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--manifest",
        default=(
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "scannet_prompt_target_diverse_128_192.json"
        ),
    )
    parser.add_argument(
        "--model_type",
        default="semantic_tokengs_v3",
        help="Model registry key (semantic_tokengs_v3 / semantic_tokengs_v4).",
    )
    args = parser.parse_args()

    opt = config_defaults["semantic_v3_open_vocab_bigdata_smoke"]
    opt.model_type = args.model_type
    opt.dataset_kwargs = {"small_manifest_path": args.manifest}
    opt.evaluating = True
    opt.resume = args.resume
    opt.workspace = args.workspace
    opt.experiment_name = Path(args.workspace).name
    opt.max_eval_iters = 0
    opt.eval_n_media_dumps = 0
    opt.use_wandb = False

    # Match the checkpoint's architecture: if it was trained with the tunable
    # semantic last cross-attention fork, the eval model must use it too.
    checkpoint_path = Path(args.resume)
    metadata_path = checkpoint_path.parent / (
        "metadata_step_"
        + checkpoint_path.stem.replace("model_step_", "")
        + ".json"
    )
    if metadata_path.is_file():
        checkpoint_meta = json.load(open(metadata_path, encoding="utf-8"))
        if "prompt_clip_model_path" in checkpoint_meta:
            opt.prompt_clip_model_path = str(
                checkpoint_meta["prompt_clip_model_path"]
            )
            print(
                "[open-vocab] CLIP model from checkpoint metadata: "
                f"{opt.prompt_clip_model_path}"
            )
        if "semantic_v4_feature_dim" in checkpoint_meta:
            opt.semantic_v4_feature_dim = int(
                checkpoint_meta["semantic_v4_feature_dim"]
            )
            opt.semantic_v4_use_geometry = bool(
                checkpoint_meta.get("semantic_v4_use_geometry", True)
            )
            opt.semantic_v4_teacher_projection = str(
                checkpoint_meta.get(
                    "semantic_v4_teacher_projection", "frozen_random"
                )
            )
        if "semantic_v2_tune_last_cross_attention" in checkpoint_meta:
            opt.semantic_v2_tune_last_cross_attention = bool(
                checkpoint_meta["semantic_v2_tune_last_cross_attention"]
            )
            print(
                "[open-vocab] tune_last_cross_attention from checkpoint "
                f"metadata: {opt.semantic_v2_tune_last_cross_attention}"
            )
        if "prompt_unfreeze_tokengs" in checkpoint_meta:
            opt.prompt_unfreeze_tokengs = bool(
                checkpoint_meta["prompt_unfreeze_tokengs"]
            )
            print(
                "[open-vocab] unfreeze_tokengs from checkpoint metadata: "
                f"{opt.prompt_unfreeze_tokengs}"
            )

    Path(opt.workspace).mkdir(parents=True, exist_ok=True)
    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())

    model = model_registry[opt.model_type](opt)
    checkpoint = torch.load if args.resume.endswith(".pth") else None
    if checkpoint is not None:
        state = checkpoint
    else:
        from safetensors.torch import load_file

        state = load_file(args.resume, device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    model = model.cuda()

    class_totals: dict[int, dict[str, float]] = defaultdict(
        lambda: {"iou": 0.0, "acc": 0.0, "n": 0}
    )
    mode_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"iou": 0.0, "acc": 0.0, "n": 0}
    )
    all_rows = []

    with torch.inference_mode():
        for i, data in enumerate(test_loader):
            if opt.max_eval_iters > 0 and i >= opt.max_eval_iters:
                break
            data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in data.items()}
            out = model(data, compute_quality_metrics=False)
            probability = out["rendered_prompt_probability"]  # [B,8,V,1,H,W]
            valid = out["valid_mask"]  # [B,8,V,1,H,W]
            target = data["binary_mask_output"].float()  # [B,V,H,W]
            for b in range(probability.shape[0]):
                class_id = int(data["prompt_class_id"][b].item())
                prompt_mode = data["prompt_mode"][b]
                prob = probability[b, class_id - 1, 0, 0]  # [H,W]
                valid_mask = valid[b, class_id - 1, 0, 0]
                tgt = (target[b] >= 0.5) & valid_mask
                pred = (prob >= 0.5) & valid_mask
                intersection = (pred & tgt).sum().float()
                union = (pred | tgt).sum().float()
                iou = intersection / union.clamp_min(1e-6)
                acc = intersection / tgt.sum().clamp_min(1e-6)
                class_totals[class_id]["iou"] += float(iou)
                class_totals[class_id]["acc"] += float(acc)
                class_totals[class_id]["n"] += 1
                mode_totals[prompt_mode]["iou"] += float(iou)
                mode_totals[prompt_mode]["acc"] += float(acc)
                mode_totals[prompt_mode]["n"] += 1
                all_rows.append(
                    {
                        "index": i,
                        "scene": data["scene_name"][b],
                        "class_id": class_id,
                        "class_name": C3G8_CLASS_NAMES[class_id - 1],
                        "prompt_mode": prompt_mode,
                        "mask_iou": float(iou),
                        "mask_acc": float(acc),
                    }
                )

    per_class = {
        C3G8_CLASS_NAMES[class_id - 1]: {
            "count": int(stats["n"]),
            "mask_iou": stats["iou"] / max(1, stats["n"]),
            "mask_acc": stats["acc"] / max(1, stats["n"]),
        }
        for class_id, stats in sorted(class_totals.items())
    }
    per_mode = {
        mode: {
            "count": int(stats["n"]),
            "mask_iou": stats["iou"] / max(1, stats["n"]),
            "mask_acc": stats["acc"] / max(1, stats["n"]),
        }
        for mode, stats in sorted(mode_totals.items())
    }
    total_n = max(1, len(all_rows))
    payload = {
        "num_samples": len(all_rows),
        "mask_iou": sum(row["mask_iou"] for row in all_rows) / total_n,
        "mask_acc": sum(row["mask_acc"] for row in all_rows) / total_n,
        "per_class": per_class,
        "per_mode": per_mode,
        "samples": all_rows,
    }
    with open(os.path.join(args.workspace, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[open-vocab] overall iou={payload['mask_iou']:.4f} acc={payload['mask_acc']:.4f}")
    for mode, stats in per_mode.items():
        print(f"[open-vocab] mode={mode:17s} n={stats['count']} iou={stats['mask_iou']:.4f} acc={stats['mask_acc']:.4f}")
    for class_name, stats in per_class.items():
        print(f"[open-vocab] class={class_name:8s} n={stats['count']} iou={stats['mask_iou']:.4f} acc={stats['mask_acc']:.4f}")
    print(f"[open-vocab] saved to {os.path.abspath(os.path.join(args.workspace, 'eval_metrics.json'))}")


if __name__ == "__main__":
    main()
