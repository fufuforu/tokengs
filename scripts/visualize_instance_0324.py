"""Visualize the 0.324 instance pipeline on LSM target views.

Loads the frozen wide7l@8000 + DINO 8-local-unit checkpoint, runs the
one-shot forward on LSM scenes, and saves per-scene grids:

    [GT RGB] [pred RGB] [pred instance map] [GT instance map]

Predicted instance maps come from the rendered instance-group probability
(Agglomerative on the unit embedding, rendered through the frozen Gaussian
geometry).  GT instance maps are the ScanNet target-view labels.

Usage:
    python scripts/visualize_instance_0324.py \
        --workspace workspace/visualize_0324 --gpu 0 \
        --max_scenes 5 --views 1 3 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402

from eval_instance_lsm_protocol import (  # noqa: E402
    _LocalAccelerator,
    _audit_lsm_manifest,
)


def _tab20_colors(n: int = 20):
    from matplotlib import colormaps

    cmap = colormaps["tab20"]
    return np.asarray([cmap(i)[:3] for i in range(n)])


def _colorize(mask: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """[H,W] int labels -> [H,W,3] uint8 (0/negative = dark background)."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    valid = mask > 0
    ids = np.unique(mask[valid]) if valid.any() else []
    for iid in ids:
        sel = mask == iid
        rgb[sel] = colors[int(iid) % len(colors)]
    rgb[~valid] = (0.08, 0.08, 0.10)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _pred_instance_map(
    prob: np.ndarray, void_channel: int, conf_thresh: float = 0.25
) -> np.ndarray:
    """[G+1,H,W] -> [H,W] int map (0 = background/void/low-confidence)."""
    g_plus1, h, w = prob.shape
    amax = prob.argmax(axis=0)
    maxp = prob.max(axis=0)
    valid = (amax != void_channel) & (maxp >= conf_thresh)
    out = np.where(valid, amax + 1, 0).astype(np.int64)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=(
            "workspace/semantic_v6_unit_shaping_img_dino_train_3000/"
            "checkpoints/model_step_003000.safetensors"
        ),
    )
    parser.add_argument(
        "--workspace", default="workspace/visualize_0324"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_scenes", type=int, default=5)
    parser.add_argument(
        "--views", nargs="+", type=int, default=[1, 3, 5],
        help="Which of the 7 target views to show.",
    )
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = config_defaults["semantic_v6_unit_shaping_img_dino_train"]
    opt.data_mode = (("scannet_lsm_instance_eval", 1),)
    opt.dataset_kwargs = {
        "lsm_manifest_path": str(
            ROOT / "data/scannet_prompt/lsm_instance_eval_manifest.json"
        )
    }
    opt.workspace = str(out_dir)
    opt.experiment_name = out_dir.name
    _audit_lsm_manifest(
        str(ROOT / "data/scannet_prompt/lsm_instance_eval_manifest.json")
    )

    from safetensors.torch import load_file

    model = model_registry[opt.model_type](opt)
    ck = load_file(args.resume, device="cpu")
    torch.nn.Module.load_state_dict(model, ck, strict=False)
    if not any(key.startswith("enc_dec_backbone.") for key in ck):
        backbone_path = str(getattr(opt, "backbone_resume", "") or "")
        if backbone_path and Path(backbone_path).is_file():
            bck = load_file(backbone_path, device="cpu")
            prefixes = (
                "enc_dec_backbone.",
                "patch_embed.",
                "patch_plucker_embed.",
                "activation_head.",
                "anchor_pos_encoder.",
            )
            native = torch.nn.Module.state_dict(model)
            loadable = {
                key: value
                for key, value in bck.items()
                if (key.startswith(prefixes) or key == "gs_tokens")
                and key in native
                and native[key].shape == value.shape
            }
            torch.nn.Module.load_state_dict(model, loadable, strict=False)
    model.eval().cuda()
    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    colors = _tab20_colors()

    from PIL import Image

    for i, data in enumerate(test_loader):
        if args.max_scenes > 0 and i >= args.max_scenes:
            break
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        scene_name = str(data["scene_name"][0])
        with torch.inference_mode():
            out = model(data, compute_quality_metrics=False)
        prob_all = out["rendered_instance_group_probability"][0]  # [G+1,V,1,H,W]
        void_ch = prob_all.shape[0] - 1
        gt_rgb = data["images_output"][0].cpu().numpy()  # [V,3,H,W]
        pred_rgb = out["images_pred"][0].cpu().numpy()
        gt_inst = data["instance_label_output"][0].cpu().numpy()  # [V,H,W]
        view_count = gt_rgb.shape[0]
        views = [v for v in args.views if v < view_count] or [0]
        rows = []
        for v in views:
            prob_v = prob_all[:, v, 0].cpu().numpy()
            pred_map = _pred_instance_map(prob_v, void_ch, args.conf_thresh)
            gt_map = gt_inst[v].astype(np.int64)
            gt_map[gt_map < 0] = 0
            rows.append(
                [
                    _to_uint8(gt_rgb[v].transpose(1, 2, 0)),
                    _to_uint8(pred_rgb[v].transpose(1, 2, 0)),
                    _colorize(pred_map, colors),
                    _colorize(gt_map, colors),
                ]
            )
        h, w = rows[0][0].shape[:2]
        panel_h, panel_w = h + 24, w
        grid = np.full((len(rows) * panel_h, 4 * panel_w, 3), 255, dtype=np.uint8)
        from PIL import ImageFont

        for ri, row in enumerate(rows):
            for ci, panel in enumerate(row):
                y0 = ri * panel_h
                x0 = ci * panel_w
                grid[y0 : y0 + h, x0 : x0 + w] = panel
        img = Image.fromarray(grid)
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        labels = ["GT RGB", "Pred RGB", "Pred instance", "GT instance"]
        for ci, lab in enumerate(labels):
            draw.text((ci * panel_w + 8, len(rows) * panel_h - 20), lab,
                      fill=(0, 0, 0))
        path = out_dir / f"{scene_name}.png"
        img.save(path)
        print(
            f"[vis] {scene_name} views={views} -> {path} "
            f"({i + 1}/{min(args.max_scenes or 40, 40)})",
            flush=True,
        )

    print(f"[vis] done -> {out_dir}")


if __name__ == "__main__":
    main()
