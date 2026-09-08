"""C3G-protocol LSeg lifting semantic evaluation (2 source + 1 target view).

Pure lifting (no learned semantic decoder): extract LSeg features from the
2 context views, project them onto the Gaussian centers, render to the
source/target views, and decode with LSeg's own output_conv + text decoder.
Works on any existing checkpoint (e.g. the wide7l instance-grouping model)
without retraining. Reports per-image mIoU / Acc on target and source views
plus target/source RGB PSNR, matching the numbers from the earlier
tokengs_c3g experiment (~0.536 target mIoU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_jaccard_index,
)

torch._dynamo.config.disable = True  # eager fallback; avoids inductor compile hangs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.input_types import split_data
from tokengs.models.lifting_semantic import SourceFeatureProjector
from tokengs.models.lseg_teacher import LSegTeacher
from tokengs.options import config_defaults


SCANNET_CLASSES = [
    "wall",
    "floor",
    "ceiling",
    "chair",
    "table",
    "sofa",
    "bed",
    "other",
]


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
            print(f"[lift-eval] config.yaml parse failed: {exc}")
    for key in (
        "prompt_clip_model_path",
        "semantic_v4_teacher_projection",
        "lseg_checkpoint_path",
    ):
        if meta.get(key) is not None:
            setattr(opt, key, str(meta[key]))
    for key, cast in (
        ("semantic_v4_feature_dim", int),
        ("semantic_v4_use_geometry", bool),
        ("prompt_unfreeze_tokengs", bool),
        ("instance_group_num_groups", int),
        ("instance_group_decoder", bool),
        ("instance_group_decoder_layers", int),
        ("instance_group_use_anchor_pos", bool),
        ("instance_group_render_scale", float),
        ("num_gs_tokens", int),
        ("num_dynamic_gs_tokens", int),
        ("lambda_semantic_feature", float),
        ("lambda_semantic_ce", float),
        ("lambda_semantic_source", float),
        ("semantic_pseudo_conf_threshold", float),
        ("semantic_classifier_hidden_dim", int),
        ("semantic_residual_scale", float),
    ):
        if meta.get(key) is not None:
            setattr(opt, key, cast(meta[key]))


def per_image_metrics(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int = 9
) -> tuple[torch.Tensor, torch.Tensor]:
    """C3G-compatible per-image metrics.

    Matches the old tokengs_c3g ``compute_c3g_semantic_metrics`` and C3G's
    torchmetrics stack: ``multiclass_jaccard_index(average='macro',
    ignore_index=0)`` and ``multiclass_accuracy(average='micro',
    ignore_index=0)`` per image. Note torchmetrics' macro averages only over
    the classes present in the target, which differs from averaging over all
    8 classes (the naive per-class mean systematically under-reports by
    roughly half on ScanNet C3G8).
    """
    pred_flat = pred.reshape(-1, *pred.shape[-2:])
    target_flat = target.reshape(-1, *target.shape[-2:])
    ious = []
    accuracies = []
    for pred_image, target_image in zip(pred_flat, target_flat):
        ious.append(
            multiclass_jaccard_index(
                pred_image,
                target_image,
                num_classes=num_classes,
                average="macro",
                ignore_index=0,
            )
        )
        accuracies.append(
            multiclass_accuracy(
                pred_image,
                target_image,
                num_classes=num_classes,
                average="micro",
                ignore_index=0,
            )
        )
    return torch.stack(ious), torch.stack(accuracies)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--num_groups", type=int, default=128)
    parser.add_argument(
        "--depth_filter",
        action="store_true",
        help="Use rendered source depth to filter the LSeg lifting.",
    )
    args = parser.parse_args()

    opt = config_defaults["eval_scannet_lsm_instance"]
    opt.data_mode = (("scannet_c3g8_eval", 1),)
    opt.num_input_views = 2
    opt.num_views = 3
    opt.resume = args.resume
    opt.workspace = args.workspace
    opt.experiment_name = Path(args.workspace).name
    opt.instance_group_num_groups = args.num_groups
    _load_checkpoint_arch(args, opt)
    Path(args.workspace).mkdir(parents=True, exist_ok=True)

    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    from safetensors.torch import load_file

    model = model_registry[opt.model_type](opt)
    # Load with the plain nn.Module path: ``PromptTokenGS.load_state_dict``
    # only knows about prompt-matcher keys, so it would silently ignore (or
    # reject) ``semantic_lifting_head`` weights. The lifting checkpoints
    # store exactly the prompt adapters + the semantic-lifting head; the
    # geometry backbone is already loaded by the model constructor.
    torch.nn.Module.load_state_dict(
        model, load_file(args.resume, device="cpu"), strict=False
    )
    model.eval()
    model = model.cuda()

    teacher = LSegTeacher(opt.lseg_checkpoint_path)
    projector = SourceFeatureProjector(image_hw=tuple(opt.img_size)).cuda()

    per_scene = {}
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if args.max_scenes > 0 and i >= args.max_scenes:
                break
            data = {
                k: (v.cuda() if torch.is_tensor(v) else v)
                for k, v in data.items()
            }
            model_input, _ = split_data(data, opt)
            reconstruction, gs_token_hidden, _ = (
                model._forward_prompt_reconstruction(model_input)
            )
            gaussians = reconstruction.gaussians
            means = gaussians[..., 0:3]

            input_rgb = data["images_input"]  # [B,V,3,H,W]
            batch_v, view_v, _, _, _ = input_rgb.shape
            source_lseg = teacher.extract(
                input_rgb.reshape(batch_v * view_v, 3, *tuple(opt.img_size))
            )  # [B*V,512,Hf,Wf]
            source_lseg = source_lseg.reshape(
                batch_v,
                view_v,
                source_lseg.shape[1],
                source_lseg.shape[2],
                source_lseg.shape[3],
            )  # [B,V,512,Hf,Wf]
            source_depth = None
            if args.depth_filter:
                source_depth = model.gs.render(
                    gaussians,
                    data["cam_view_input"],
                    bg_color=reconstruction.background_color,
                    intrinsics=data["intrinsics_input"],
                )["depths_pred"]
            fused, has_source, confidence = projector(
                xyz_world=means,
                source_features=source_lseg,
                source_c2w=data["cam_to_world_input"],
                source_intrinsics=data["intrinsics_input"],
                source_depth=source_depth,
            )
            if model.semantic_lifting_head is not None:
                fused = model.semantic_lifting_head(
                    decoded_tokens=gs_token_hidden,
                    gaussians=gaussians,
                    projected_features=fused,
                    has_source=has_source,
                    confidence=confidence,
                )

            target_render = model.gs.render_feature_channels(
                gaussians,
                fused,
                model_input.decoder.cam_view,
                intrinsics=model_input.decoder.intrinsics,
            )
            source_render = model.gs.render_feature_channels(
                gaussians,
                fused,
                data["cam_view_input"],
                intrinsics=data["intrinsics_input"],
            )

            def decode(rendered: torch.Tensor) -> torch.Tensor:
                flat = rendered.reshape(-1, rendered.shape[2], *rendered.shape[3:]).float()
                out = teacher.extractor.scratch.output_conv(flat)
                logits = teacher.extractor.decode_feature(
                    out, labelset=SCANNET_CLASSES
                )
                logits = F.interpolate(
                    logits,
                    size=tuple(opt.img_size),
                    mode="bilinear",
                    align_corners=False,
                )
                pred = logits.argmax(dim=1) + 1
                return pred.reshape(
                    rendered.shape[0], rendered.shape[1], *tuple(opt.img_size)
                )

            target_pred = decode(target_render["images_pred"])
            source_pred = decode(source_render["images_pred"])
            target_labels = data["semantic_label_output"].long()
            source_labels = data["semantic_label_input"].long()

            target_iou, target_acc = per_image_metrics(
                target_pred, target_labels
            )
            source_iou, source_acc = per_image_metrics(
                source_pred, source_labels
            )

            target_rgb = model.gs.render(
                gaussians,
                model_input.decoder.cam_view,
                bg_color=reconstruction.background_color,
                intrinsics=model_input.decoder.intrinsics,
            )["images_pred"]
            target_rgb_gt = data["images_output"]
            mse = (target_rgb.clamp(0, 1) - target_rgb_gt.clamp(0, 1)).square().mean()
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))

            scene_name = data["scene_name"][0]
            per_scene[str(scene_name)] = {
                "target_miou": float(target_iou.mean()),
                "target_acc": float(target_acc.mean()),
                "source_miou": float(source_iou.mean()),
                "source_acc": float(source_acc.mean()),
                "psnr": float(psnr),
            }
            print(
                f"[lift-eval] {scene_name}: target mIoU={float(target_iou.mean()):.4f} "
                f"Acc={float(target_acc.mean()):.4f} | source mIoU={float(source_iou.mean()):.4f} "
                f"Acc={float(source_acc.mean()):.4f} | PSNR={float(psnr):.2f}"
            )

    def mean(key: str) -> float:
        return float(
            sum(entry[key] for entry in per_scene.values())
            / max(1, len(per_scene))
        )

    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "num_scenes": len(per_scene),
        "target_miou": mean("target_miou"),
        "target_acc": mean("target_acc"),
        "source_miou": mean("source_miou"),
        "source_acc": mean("source_acc"),
        "psnr": mean("psnr"),
        "per_scene": per_scene,
    }
    out_path = Path(args.workspace) / "c3g_lifting.json"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[lift-eval] scenes={len(per_scene)} target mIoU={payload['target_miou']:.4f} "
        f"Acc={payload['target_acc']:.4f} source mIoU={payload['source_miou']:.4f} "
        f"Acc={payload['source_acc']:.4f} PSNR={payload['psnr']:.2f}"
    )
    print(f"[lift-eval] wrote {out_path}")


if __name__ == "__main__":
    main()
