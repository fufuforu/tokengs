#!/usr/bin/env python3
"""Query a trained open-vocabulary SemanticTokenGSv3 with an arbitrary prompt.

Usage:
  python scripts/query_gs_prompt.py --resume CKPT --scene scene0000_00 --text "a chair"
  python scripts/query_gs_prompt.py --resume CKPT --scene scene0000_00 --image chair.png
  python scripts/query_gs_prompt.py --resume CKPT --scene scene0000_00 --text "a chair" --image chair.png

Options:
  --scene       target ScanNet scene (must not be one of the 40 held-out eval scenes)
  --frame-id    optional target frame id; default is the middle labeled frame
  --class       optional C3G8 class id (1-8) used as the GT mask for scoring/visualization
  --output-dir  where visualization.png / eval_metrics.json are written
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from tokengs.data import get_multi_dataloader
from tokengs.data.static.scannet import ScanNet, ScanNetSensReader
from tokengs.data.static.scannet_prompt import ScanNetPromptTrain
from tokengs.models import model_registry
from tokengs.options import config_defaults


C3G8_CLASS_NAMES = ("wall", "floor", "ceiling", "chair", "table", "sofa", "bed", "other")
_C3G8_EVAL_SCENES = None


def _load_c3g8_eval_scenes() -> list[str]:
    global _C3G8_EVAL_SCENES
    if _C3G8_EVAL_SCENES is None:
        manifest = json.load(
            open(
                "/space0/mawb/tokengs/data/scannet_prompt/"
                "scannet_c3g8_train_provisional.json",
                encoding="utf-8",
            )
        )
        _C3G8_EVAL_SCENES = sorted(manifest["excluded_eval_scenes"])
    return _C3G8_EVAL_SCENES


def _class_ratio(label_path: Path, class_id: int) -> float:
    raw = np.asarray(Image.open(label_path))
    if raw.ndim == 3:
        raw = raw[..., 0]
    lut, fallback, _ = ScanNet._load_c3g8_protocol(
        Path("/space0/mawb/tokengs/configs/semantic/scannet_c3g8.yaml")
    )
    mapped = np.full(raw.shape, fallback, dtype=np.int64)
    valid = (raw >= 0) & (raw < len(lut))
    mapped[valid] = lut[raw[valid]]
    height, width = mapped.shape
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    square = mapped[top : top + side, left : left + side]
    return float((square == class_id).mean())


def _pick_frames(
    scan_root: Path,
    scene: str,
    label_root: Path,
    frame_id: int | None,
    class_id: int | None = None,
) -> tuple[list[int], int]:
    reader = ScanNetSensReader(scan_root / scene / f"{scene}.sens", frame_stride=1)
    label_dir = label_root / scene / "label-filt"
    valid_ids = [
        frame
        for frame in reader.frame_ids
        if (label_dir / f"{frame}.png").is_file()
    ]
    if not valid_ids:
        raise RuntimeError(f"No labeled frames with valid poses in {scene}")
    if frame_id is not None:
        if frame_id not in reader.frame_ids:
            raise RuntimeError(
                f"frame {frame_id} is not a valid pose frame in {scene}"
            )
        target = frame_id
    elif class_id is not None:
        sample_count = min(60, len(valid_ids))
        indices = sorted(
            set(
                np.linspace(0, len(valid_ids) - 1, sample_count, dtype=int).tolist()
            )
        )
        best = None
        for index in indices:
            candidate = valid_ids[index]
            ratio = _class_ratio(label_dir / f"{candidate}.png", class_id)
            if ratio >= 0.01 and (best is None or ratio > best[1]):
                best = (candidate, ratio)
        if best is None:
            print(
                f"[query] no frame with visible class {class_id} in {scene}; "
                "falling back to the middle frame"
            )
            target = valid_ids[len(valid_ids) // 2]
        else:
            print(
                f"[query] picked frame {best[0]} with class {class_id} "
                f"foreground ratio {best[1]:.3f}"
            )
            target = best[0]
    else:
        target = valid_ids[len(valid_ids) // 2]
    before = [frame for frame in valid_ids if frame < target]
    after = [frame for frame in valid_ids if frame > target]
    if before and after:
        context = [before[-1], after[0]]
    elif before:
        context = before[-2:] if len(before) >= 2 else before
    elif after:
        context = after[:2] if len(after) >= 2 else after
    else:
        raise RuntimeError(f"Not enough context frames in {scene}")
    return sorted(context), target


def _build_temp_manifest(
    scene: str,
    context: list[int],
    target: int,
    class_id: int,
    dummy_scene: str,
    dummy_context: list[int],
    dummy_target: int,
) -> Path:
    excluded = _load_c3g8_eval_scenes()
    manifest = {
        "version": 1,
        "seed": 20260801,
        "provisional": True,
        "source_manifest": (
            "/space0/mawb/tokengs/data/scannet_prompt/scannet_c3g8_train_provisional.json"
        ),
        "query_bank": (
            "/space0/mawb/tokengs/data/scannet_prompt/scannet_c3g8_query_bank.json"
        ),
        "excluded_eval_scenes": excluded,
        "class_names": list(C3G8_CLASS_NAMES),
        "prompt_mode_ratio_for_image_classes": {
            "text_only": 0.4,
            "image_only": 0.3,
            "text_image_mixed": 0.3,
        },
        "train_scenes": [dummy_scene],
        "validation_scenes": [scene],
        "target_frames_per_scene": 0,
        "min_target_foreground_ratio": 0.0,
        "scene_split_from": None,
        "validation_samples_from": None,
        "train_samples": [
            {
                "sample_id": "dummy_0000",
                "scene": dummy_scene,
                "input_frame_ids": list(dummy_context),
                "target_frame_id": dummy_target,
                "class_id": 1,
                "foreground_pixels": 0,
                "image_pixels": 0,
                "foreground_ratio": 0.0,
                "prompt_mode": "text_only",
                "query": None,
            }
        ],
        "validation_samples": [
            {
                "sample_id": "query_0000",
                "scene": scene,
                "input_frame_ids": list(context),
                "target_frame_id": target,
                "class_id": class_id,
                "foreground_pixels": 0,
                "image_pixels": 0,
                "foreground_ratio": 0.0,
                "prompt_mode": "text_only",
                "query": None,
            }
        ],
        "distribution": {
            "train": {
                "classes": {"1": 1},
                "prompt_modes": {"text_only": 1},
                "scenes_represented": 1,
            },
            "validation": {
                "classes": {str(class_id): 1},
                "prompt_modes": {"text_only": 1},
                "scenes_represented": 1,
            },
        },
        "reader_reports": {},
    }
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix="query_manifest_", delete=False
    )
    json.dump(manifest, handle, indent=2)
    handle.close()
    return Path(handle.name)


def _load_query_image(image_path: str, mask_path: str | None) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8).copy()
    if mask_path is not None:
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32)
        if mask.shape != rgb.shape[:2]:
            raise ValueError("--image-mask must match --image spatial size")
        mask = (mask > 127).astype(np.float32)
    else:
        mask = np.ones(rgb.shape[:2], dtype=np.float32)
    if not mask.any():
        raise ValueError("query mask is empty")
    query_image, query_mask = ScanNetPromptTrain._letterbox_query(
        rgb, mask, (224, 224)
    )
    return query_image, query_mask


def _load_original_query(
    image_path: str, mask_path: str | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (original RGB in [0,1], masked RGB) for visualization panels."""
    rgb = (
        np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8).astype(
            np.float32
        )
        / 255.0
    )
    if mask_path is not None:
        mask = (
            np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 127
        ).astype(np.float32)
        if mask.shape != rgb.shape[:2]:
            raise ValueError("--image-mask must match --image spatial size")
    else:
        mask = np.ones(rgb.shape[:2], dtype=np.float32)
    return rgb, rgb * mask[..., None]


def _visualize(
    output_dir: Path,
    scene: str,
    target: int,
    class_id: int,
    input_rgb: torch.Tensor,
    rendered_rgb: torch.Tensor,
    gt_mask: torch.Tensor,
    threshold_pred: torch.Tensor,
    query_original: np.ndarray | None,
    query_masked: np.ndarray | None,
) -> None:
    panels = []
    labels = []
    panels.append(input_rgb.permute(1, 2, 0).clamp(0, 1).numpy())
    labels.append("new view gt rgb")
    panels.append(rendered_rgb.permute(1, 2, 0).clamp(0, 1).numpy())
    labels.append("new view rendered")
    panels.append(gt_mask.squeeze().numpy())
    labels.append(f"seg gt ({C3G8_CLASS_NAMES[class_id - 1]})")
    panels.append(threshold_pred.squeeze().numpy())
    labels.append("seg pred (th=0.5)")
    if query_original is not None:
        panels.append(query_original)
        labels.append("query original")
        panels.append(query_masked)
        labels.append("query masked")
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, panel, label in zip(axes, panels, labels):
        ax.imshow(
            panel,
            cmap="gray" if panel.ndim == 2 else None,
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{scene} frame {target} | prompt->{C3G8_CLASS_NAMES[class_id - 1]}")
    fig.tight_layout()
    fig.savefig(
        output_dir / "visualization.png",
        dpi=120,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True, help="v3 checkpoint (safetensors)")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--text", default=None, help="arbitrary text prompt")
    parser.add_argument("--image", default=None, help="image prompt path")
    parser.add_argument("--image-mask", default=None, help="optional query mask path")
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument(
        "--class", dest="class_id", type=int, default=1,
        help="C3G8 class id (1-8) used as GT mask for scoring",
    )
    parser.add_argument("--output-dir", default="workspace/query_gs")
    args = parser.parse_args()

    if args.text is None and args.image is None:
        parser.error("provide --text and/or --image")
    if not 1 <= args.class_id <= 8:
        parser.error("--class must be in 1..8")
    if args.scene in _load_c3g8_eval_scenes():
        parser.error(
            f"{args.scene} is a held-out C3G eval scene; "
            "query a training-domain scene instead"
        )

    scan_root = Path("/space0/mawb/tokengs/data/ScanNet/scans")
    label_root = Path("/space0/mawb/tokengs/data/scannet2d_labels")
    context, target = _pick_frames(
        scan_root, args.scene, label_root, args.frame_id, args.class_id
    )
    dummy_scene = "scene0001_00"
    dummy_context, dummy_target = _pick_frames(
        scan_root, dummy_scene, label_root, None, None
    )
    manifest_path = _build_temp_manifest(
        args.scene, context, target, args.class_id,
        dummy_scene, dummy_context, dummy_target,
    )

    opt = config_defaults["semantic_v3_open_vocab_full_smoke"]
    opt.dataset_kwargs = {"small_manifest_path": str(manifest_path)}
    opt.evaluating = True
    opt.max_eval_iters = 1
    opt.eval_n_media_dumps = 0

    class _LocalAccelerator:
        is_main_process = True

    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    model = model_registry[opt.model_type](opt)
    from safetensors.torch import load_file

    model.load_state_dict(load_file(args.resume, device="cpu"), strict=True)
    model.eval()
    model = model.cuda()

    data = next(iter(test_loader))
    data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in data.items()}
    if args.text is not None and args.image is not None:
        data["prompt_mode"] = ["text_image_mixed"]
        data["has_image_query"] = torch.tensor([True], device="cuda")
        data["positive_text_prompt"] = [args.text]
        query_image, query_mask = _load_query_image(args.image, args.image_mask)
        data["query_image"] = query_image.unsqueeze(0).cuda()
        data["query_mask"] = query_mask.unsqueeze(0).cuda()
        query_original, query_masked = _load_original_query(
            args.image, args.image_mask
        )
    elif args.image is not None:
        data["prompt_mode"] = ["image_only"]
        data["has_image_query"] = torch.tensor([True], device="cuda")
        data["positive_text_prompt"] = [C3G8_CLASS_NAMES[args.class_id - 1]]
        query_image, query_mask = _load_query_image(args.image, args.image_mask)
        data["query_image"] = query_image.unsqueeze(0).cuda()
        data["query_mask"] = query_mask.unsqueeze(0).cuda()
        query_original, query_masked = _load_original_query(
            args.image, args.image_mask
        )
    else:
        data["prompt_mode"] = ["text_only"]
        data["has_image_query"] = torch.tensor([False], device="cuda")
        data["positive_text_prompt"] = [args.text]
        query_original, query_masked = None, None

    with torch.inference_mode():
        out = model(data, compute_quality_metrics=False)

    probability = out["rendered_prompt_probability"][0, args.class_id - 1, 0, 0]
    valid = out["valid_mask"][0, args.class_id - 1, 0, 0]
    target_mask = data["binary_mask_output"][0]
    gt = (target_mask >= 0.5) & valid
    pred = (probability >= 0.5) & valid
    intersection = (pred & gt).sum().float()
    union = (pred | gt).sum().float()
    iou = intersection / union.clamp_min(1e-6)
    acc = intersection / gt.sum().clamp_min(1e-6)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _visualize(
        output_dir,
        args.scene,
        target,
        args.class_id,
        data["images_output"][0, 0].cpu(),
        out["images_pred"][0, 0].cpu(),
        gt.cpu(),
        pred.cpu(),
        query_original,
        query_masked,
    )
    with open(output_dir / "eval_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "scene": args.scene,
                "target_frame": target,
                "context_frames": context,
                "prompt_mode": data["prompt_mode"][0],
                "text_prompt": data["positive_text_prompt"][0],
                "image_prompt": args.image,
                "class_id": args.class_id,
                "mask_iou": float(iou),
                "mask_acc": float(acc),
            },
            handle,
            indent=2,
        )
    print(
        f"[query] scene={args.scene} frame={target} "
        f"mode={data['prompt_mode'][0]} class={C3G8_CLASS_NAMES[args.class_id - 1]} "
        f"mask_iou={float(iou):.4f} mask_acc={float(acc):.4f}"
    )
    print(f"[query] visualization saved to {output_dir / 'visualization.png'}")


if __name__ == "__main__":
    main()
