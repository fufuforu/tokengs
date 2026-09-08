#!/usr/bin/env python3
"""Evaluate a trained SemanticTokenGSv3 under the C3G ScanNet protocol.

Replicates C3G's 3D scene understanding evaluation on the identical test
manifest (40 scenes x 30 frames, llff_hold=8, residues {1,4} -> 320 target
frames): per-frame multiclass argmax over the eight C3G8 classes, then
per-image mIoU (macro over classes, background ignored) and Acc (micro,
background ignored), averaged over frames -- exactly what model_wrapper.py
reports as test/mIoU and test/Acc.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.options import config_defaults


C3G8_CLASS_NAMES = ("wall", "floor", "ceiling", "chair", "table", "sofa", "bed", "other")


class _LocalAccelerator:
    is_main_process = True


def per_image_metrics(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int = 9
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multiclass mIoU (macro, ignore 0) and Acc (micro, ignore 0) per image."""
    valid = target != 0
    per_class_iou = []
    true_positive = torch.zeros((), dtype=torch.float32, device=pred.device)
    target_pixels = torch.zeros((), dtype=torch.float32, device=pred.device)
    for class_id in range(1, num_classes):
        predicted = (pred == class_id) & valid
        gt = (target == class_id) & valid
        intersection = (predicted & gt).sum().float()
        union = (predicted | gt).sum().float()
        per_class_iou.append(intersection / union.clamp_min(1e-6))
        true_positive = true_positive + intersection
        target_pixels = target_pixels + gt.sum().float()
    miou = torch.stack(per_class_iou).mean()
    acc = true_positive / target_pixels.clamp_min(1e-6)
    return miou, acc, torch.stack(per_class_iou)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--label",
        default="",
        help="Optional row label stored in the output JSON.",
    )
    parser.add_argument(
        "--model_type",
        default="semantic_tokengs_v3",
        help="Model registry key (semantic_tokengs_v3 / semantic_tokengs_v4).",
    )
    args = parser.parse_args()

    opt = config_defaults["eval_scannet_c3g8_semantic_v2"]
    opt.model_type = args.model_type
    opt.resume = args.resume
    opt.workspace = args.workspace
    opt.experiment_name = Path(args.workspace).name
    opt.max_eval_iters = 0
    opt.eval_n_media_dumps = 0
    opt.use_wandb = False

    checkpoint_path = Path(args.resume)
    metadata_path = checkpoint_path.parent / (
        "metadata_step_"
        + checkpoint_path.stem.replace("model_step_", "")
        + ".json"
    )
    meta = {}
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
                "!dataclass:",
                _options_ctor,
                Loader=yaml.UnsafeLoader,
            )
            yaml_cfg = yaml.load(
                config_yaml.read_text(encoding="utf-8"),
                Loader=yaml.UnsafeLoader,
            )
            for key, value in yaml_cfg.items():
                meta.setdefault(key, value)
        except Exception as exc:  # pragma: no cover - fallback only
            print(f"[c3g-eval] config.yaml parse failed: {exc}")
    if meta:
        if "prompt_clip_model_path" in meta:
            opt.prompt_clip_model_path = str(meta["prompt_clip_model_path"])
            print(
                "[c3g-eval] CLIP model from checkpoint metadata: "
                f"{opt.prompt_clip_model_path}"
            )
        if "semantic_v4_feature_dim" in meta:
            value = meta["semantic_v4_feature_dim"]
            if value is not None:
                opt.semantic_v4_feature_dim = int(value)
        if meta.get("semantic_v4_use_geometry") is not None:
            opt.semantic_v4_use_geometry = bool(meta["semantic_v4_use_geometry"])
        if meta.get("semantic_v4_teacher_projection") is not None:
            opt.semantic_v4_teacher_projection = str(
                meta["semantic_v4_teacher_projection"]
            )
        if meta.get("instance_group_num_groups") is not None:
            opt.instance_group_num_groups = int(meta["instance_group_num_groups"])
        if meta.get("prompt_unfreeze_tokengs") is not None:
            opt.prompt_unfreeze_tokengs = bool(meta["prompt_unfreeze_tokengs"])
        opt.semantic_v2_tune_last_cross_attention = bool(
            meta.get("semantic_v2_tune_last_cross_attention", False)
        )
        print(
            "[c3g-eval] unfreeze_tokengs:",
            opt.prompt_unfreeze_tokengs,
            "| tune_last_cross:",
            opt.semantic_v2_tune_last_cross_attention,
        )

    Path(args.workspace).mkdir(parents=True, exist_ok=True)
    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    model = model_registry[opt.model_type](opt)
    from safetensors.torch import load_file

    model.load_state_dict(load_file(args.resume, device="cpu"), strict=True)
    model.eval()
    model = model.cuda()

    per_image_iou = []
    per_image_acc = []
    per_class_iou_sum = None
    num_frames = 0
    with torch.inference_mode():
        for i, data in enumerate(test_loader):
            if opt.max_eval_iters > 0 and i >= opt.max_eval_iters:
                break
            data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in data.items()}
            out = model(data, compute_quality_metrics=False)
            target = data["semantic_label_output"].long()  # [B,V,H,W]
            pixel_logits = out.get("rendered_pixel_logits")  # [B,V,8,H,W] (v5)
            if pixel_logits is not None:
                probability = None
            else:
                probability = out["rendered_prompt_probability"]
            batch_size = (
                pixel_logits.shape[0]
                if pixel_logits is not None
                else probability.shape[0]
            )
            for b in range(batch_size):
                if pixel_logits is not None:
                    pred = pixel_logits[b, 0].argmax(dim=0) + 1
                else:
                    prob = probability[b, :, 0, 0]  # [8,H,W]
                    pred = prob.argmax(dim=0) + 1  # classes 1..8
                tgt = target[b, 0]
                miou, acc, per_class_iou = per_image_metrics(pred, tgt)
                per_image_iou.append(miou.item())
                per_image_acc.append(acc.item())
                if per_class_iou_sum is None:
                    per_class_iou_sum = per_class_iou
                else:
                    per_class_iou_sum = per_class_iou_sum + per_class_iou
                num_frames += 1

    mean_iou = sum(per_image_iou) / max(1, num_frames)
    mean_acc = sum(per_image_acc) / max(1, num_frames)
    per_class_iou_mean = (per_class_iou_sum / max(1, num_frames)).tolist()
    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "num_frames": num_frames,
        "mIoU": mean_iou,
        "Acc": mean_acc,
        "per_class_IoU": {
            name: value
            for name, value in zip(C3G8_CLASS_NAMES, per_class_iou_mean)
        },
    }
    with open(
        Path(args.workspace) / "c3g_protocol.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
    print(f"[c3g-eval] frames={num_frames} mIoU={mean_iou:.4f} Acc={mean_acc:.4f}")
    print(
        "[c3g-eval] per-class IoU: "
        + " ".join(f"{name}={value:.3f}" for name, value in payload["per_class_IoU"].items())
    )
    print(f"[c3g-eval] saved to {Path(args.workspace) / 'c3g_protocol.json'}")


if __name__ == "__main__":
    main()
