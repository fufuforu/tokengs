from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from safetensors.torch import load_file

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.options import config_defaults
from tokengs.utils.instance_ap import instance_ap, masks_from_group_probs, gt_masks_from_instance_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--resume_mode", required=True, choices=("official", "strict"))
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--min_pred_pixels", type=int, default=1)
    parser.add_argument("--min_gt_pixels", type=int, default=1)
    parser.add_argument("--max_predictions_per_image", type=int, default=100)
    return parser.parse_args()


def load_options(preset: str, resume: str, workspace: str):
    allowed = {"gsi_v2_recon_scannet_eval", "gsi_v2_joint_scannet_eval"}
    if preset not in allowed:
        raise ValueError(f"eval preset must be one of {sorted(allowed)}")
    opt = config_defaults[preset].evolve(resume=resume, workspace=workspace, gsi_v2_resume_mode="official" if resume.endswith(".ckpt") else "strict")
    return opt


def strict_load_model(opt, resume: str, device: torch.device):
    model = model_registry["globalsplat_instance_v2"](opt).to(device)
    if opt.gsi_v2_resume_mode == "strict":
        state = load_file(resume, device="cpu")
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"GSI-v2 strict eval restore failed: {incompatible}")
    model.set_eval_stage()
    model.eval()
    return model


def evaluate(model: torch.nn.Module, loader, opt, max_scenes: int = 0) -> dict:
    pred_masks, pred_scores, gt_masks = [], [], []
    pred_ids, gt_ids = [], []
    scene_rows = []
    seen = 0
    with torch.inference_mode():
        for data in loader:
            out = model(data)
            b = out["images_pred"].shape[0]
            for bi in range(b):
                scene = str(data.get("scene_name", [f"scene{seen}"])[bi])
                image_id = f"{scene}:b{bi}"
                probabilities = out.get("rendered_instance_group_probability")
                labels = data.get("instance_label_output")
                if probabilities is not None and labels is not None:
                    probs = probabilities[bi].detach().float().cpu().numpy()[:, :, 0]
                    label = labels[bi].detach().cpu().numpy()
                    for view in range(7):
                        masks, scores = masks_from_group_probs(probs[:, view], void_channel=100, min_mask_area=1)
                        pred_masks.extend(masks[:100]); pred_scores.extend(scores[:100]); pred_ids.extend([image_id] * min(100, len(masks)))
                        targets = gt_masks_from_instance_map(label[view], min_mask_area=1)
                        gt_masks.extend(targets); gt_ids.extend([image_id] * len(targets))
                mse = float(torch.nn.functional.mse_loss(out["images_pred"][bi].float(), data["images_output"][bi].float()).item())
                scene_rows.append({"scene_name": scene, "psnr": -10.0 * np.log10(max(mse, 1e-8))})
                seen += 1
                if max_scenes and seen >= max_scenes:
                    break
            if max_scenes and seen >= max_scenes:
                break
    result = instance_ap(pred_masks, pred_scores, gt_masks, pred_image_ids=pred_ids, gt_image_ids=gt_ids) if gt_masks else {"ap_25": 0.0, "ap_50": 0.0, "ap_75": 0.0, "ap_mean": 0.0}
    result.update({"scene_count": seen, "mean_psnr": float(np.mean([row["psnr"] for row in scene_rows])) if scene_rows else 0.0, "per_scene": scene_rows,
                   "resume": str(opt.resume), "resume_mode": opt.gsi_v2_resume_mode, "official_lineage": opt.lineage_metadata() if hasattr(opt, "lineage_metadata") else {}})
    return result


def main() -> None:
    args = parse_args()
    opt = load_options(args.preset, args.resume, args.workspace)
    accelerator = Accelerator(mixed_precision="no")
    _train, test, _train_ds, _test_ds = get_multi_dataloader(opt, accelerator)
    model = strict_load_model(opt, args.resume, accelerator.device)
    model, test = accelerator.prepare(model, test)
    result = evaluate(model, test, opt, args.max_scenes)
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        (workspace / "instance_ap.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
