"""C3G-protocol evaluation of the V3 token_decoder_lowrank semantic field.

Works with the old tokengs_c3g checkpoint (``re10k_semantic_lseg_v3_token_r16``,
~0.536 target mIoU) as well as freshly trained V3 checkpoints. The geometry
backbone comes from the checkpoint itself when present, otherwise from the
RE10K TokenGS pretrained weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.input_types import split_data
from tokengs.models.lseg_teacher import LSegTeacher
from tokengs.options import config_defaults
from scripts.eval_c3g_lifting_semantic import SCANNET_CLASSES, per_image_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument(
        "--lseg-checkpoint",
        default="/space0/mawb/tokengs/checkpoints/demo_e200.ckpt",
    )
    args = parser.parse_args()

    opt = config_defaults["eval_scannet_lsm_instance"]
    opt.data_mode = (("scannet_c3g_semantic_eval", 1),)
    opt.num_input_views = 2
    opt.num_views = 3
    opt.resume = args.resume
    opt.workspace = args.workspace
    opt.experiment_name = Path(args.workspace).name

    # V3 token_decoder_lowrank architecture.
    opt.semantic_branch_version = "token_decoder_lowrank"
    opt.lambda_semantic_feature = 1.0
    opt.lambda_semantic_cosine = 1.0
    opt.lambda_semantic_l1 = 0.0
    opt.semantic_feature_use_alpha_mask = False
    opt.semantic_stream_compressed_features = True
    opt.semantic_render_scale = 0.5
    opt.semantic_render_chunk = 32
    opt.semantic_detach_tokens = True
    opt.semantic_detach_geometry = True
    opt.lseg_checkpoint_path = args.lseg_checkpoint

    class _Acc:
        is_main_process = True

    _, loader, _, _ = get_multi_dataloader(opt, _Acc())
    model = model_registry[opt.model_type](opt)
    from safetensors.torch import load_file

    if args.resume.endswith(".safetensors"):
        torch.nn.Module.load_state_dict(
            model, load_file(args.resume, device="cpu"), strict=False
        )
    model.eval()
    model = model.cuda()
    print(
        "[v3-eval] semantic_head params:",
        sum(p.numel() for p in model.semantic_head.parameters()),
    )

    teacher = LSegTeacher(args.lseg_checkpoint)

    def decode(rendered: torch.Tensor) -> torch.Tensor:
        B, V, C, H, W = rendered.shape
        flat = rendered.reshape(-1, C, H, W).float()
        out = teacher.extractor.scratch.output_conv(flat)
        logits = teacher.extractor.decode_feature(out, labelset=SCANNET_CLASSES)
        logits = F.interpolate(
            logits, size=(256, 256), mode="bilinear", align_corners=False
        )
        return logits.argmax(dim=1).reshape(B, V, 256, 256) + 1

    per_scene = {}
    with torch.no_grad():
        for i, data in enumerate(loader):
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
            source_lseg = teacher.extract(
                data["images_input"].reshape(-1, 3, *tuple(opt.img_size))
            )
            source_lseg = source_lseg.reshape(
                data["images_input"].shape[0],
                data["images_input"].shape[1],
                *source_lseg.shape[1:],
            )
            semantic_3d = model.build_semantic_gaussian_features(
                gaussians=gaussians,
                decoded_tokens=gs_token_hidden,
                source_lseg_features=source_lseg,
                source_c2w=data["cam_to_world_input"],
                source_intrinsics=data["intrinsics_input"],
            )
            target_render = model.render_semantic_3d_features(
                gaussians,
                semantic_3d,
                model_input.decoder.cam_view,
                model_input.decoder.intrinsics,
            )
            source_render = model.render_semantic_3d_features(
                gaussians,
                semantic_3d,
                data["cam_view_input"],
                data["intrinsics_input"],
            )

            target_pred = decode(target_render["semantic_features_pred"])
            source_pred = decode(source_render["semantic_features_pred"])
            target_iou, target_acc = per_image_metrics(
                target_pred, data["semantic_label_output"].long()
            )
            source_iou, source_acc = per_image_metrics(
                source_pred, data["semantic_label_input"].long()
            )
            target_iou = target_iou.mean()
            target_acc = target_acc.mean()
            source_iou = source_iou.mean()
            source_acc = source_acc.mean()
            target_rgb = model.gs.render(
                gaussians,
                model_input.decoder.cam_view,
                bg_color=reconstruction.background_color,
                intrinsics=model_input.decoder.intrinsics,
            )["images_pred"]
            mse = (
                target_rgb.clamp(0, 1) - data["images_output"].clamp(0, 1)
            ).square().mean()
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))

            scene_name = data["scene_name"][0]
            per_scene[str(scene_name)] = {
                "target_miou": float(target_iou),
                "target_acc": float(target_acc),
                "source_miou": float(source_iou),
                "source_acc": float(source_acc),
                "psnr": float(psnr),
            }
            print(
                f"[v3-eval] {scene_name}: target mIoU={float(target_iou):.4f} "
                f"Acc={float(target_acc):.4f} | source mIoU={float(source_iou):.4f} "
                f"PSNR={float(psnr):.2f}"
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
    out_path = Path(args.workspace) / "c3g_v3.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[v3-eval] scenes={len(per_scene)} target mIoU={payload['target_miou']:.4f} "
        f"Acc={payload['target_acc']:.4f} source mIoU={payload['source_miou']:.4f} "
        f"PSNR={payload['psnr']:.2f}"
    )
    print(f"[v3-eval] wrote {out_path}")


if __name__ == "__main__":
    main()
