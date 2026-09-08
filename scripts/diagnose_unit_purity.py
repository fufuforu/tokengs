"""Diagnose whether the learned 8 units/token formed instance-aligned
3D-local groups (Experiment D), using the trained token-units checkpoint.

For each scene: GT instance per GS (15-view majority vote, same convention),
the trained GS->unit assignment (soft), and the unit 3D centers. For every
unit we compute the (soft-weighted) GT instance distribution: purity,
number of instances, fraction of pure/mixed units, and spatial spread. This
is compared against token-level stats (64 GS sharing one label) to see how
much the learned units improved instance alignment.

No model modification / training. Offline only.

Usage:
    python scripts/diagnose_unit_purity.py
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=(
            "/space0/mawb/tokengs/workspace/"
            "semantic_v6_token_units_train_12000/"
            "checkpoints/model_step_006000.safetensors"
        ),
    )
    parser.add_argument("--workspace", default="workspace/diag_unit_purity")
    parser.add_argument("--label", default="token_units_6000")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument(
        "--lsm_manifest",
        default=(
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "lsm_instance_eval_manifest.json"
        ),
    )
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--min_gs", type=int, default=1)
    parser.add_argument("--major_share", type=float, default=0.2)
    parser.add_argument("--purity_threshold", type=float, default=0.9)
    args = parser.parse_args()

    opt = config_defaults["eval_scannet_lsm_instance"]
    opt.model_type = args.model_type
    opt.resume = args.resume
    opt.workspace = args.workspace
    opt.experiment_name = Path(args.workspace).name
    opt.num_input_views = 8
    opt.num_views = 15
    opt.dataset_kwargs = {
        **dict(opt.dataset_kwargs or {}),
        "lsm_manifest_path": args.lsm_manifest,
    }
    dtm._load_arch_from_checkpoint(args, opt)
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
    model.eval()
    model = model.cuda()
    br = model.instance_branch
    k = br.units_per_token

    per_scene = {}
    all_unit = []
    all_token = []
    for i, data in enumerate(test_loader):
        if args.max_scenes > 0 and i >= args.max_scenes:
            break
        data = {
            kk: (v.cuda() if torch.is_tensor(v) else v)
            for kk, v in data.items()
        }
        scene_name = data["scene_name"][0]
        t0 = time.time()
        with torch.no_grad():
            model(data, compute_quality_metrics=False)
            a = br.last_unit_assignment  # [B,T,P,K]
            centers = br.last_unit_centers  # [B,T,K,3]
            means = br.last_instance_gaussians[0, :, :3].float()
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
        a0 = a[0]  # [T,P,K]
        gs_gt_t = gs_gt.view(a0.shape[0], a0.shape[1])
        scene_units = []
        for t in range(a0.shape[0]):
            for u in range(k):
                w = a0[t, :, u]  # [P] soft weight of each GS for unit u
                if w.sum() < args.min_gs:
                    continue
                votes = gs_gt_t[t][w > 0.01]
                wv = w[w > 0.01]
                fg = votes > 0
                if not fg.any():
                    continue
                counts = {}
                for iid in torch.unique(votes[fg]).tolist():
                    counts[iid] = float(wv[fg][votes[fg] == iid].sum())
                total = sum(counts.values())
                purity = max(counts.values()) / total
                n_inst = len(counts)
                n_major = sum(1 for c in counts.values() if c / total >= args.major_share)
                # spatial spread of the unit's GS (weighted)
                pos = means.view(a0.shape[0], a0.shape[1], 3)[t]
                c = centers[0, t, u].float()
                dist2 = ((pos - c).square().sum(-1) * w).sum() / max(w.sum(), 1e-6)
                scene_units.append(
                    {
                        "purity": purity,
                        "n_instances": n_inst,
                        "mixed_major": n_major >= 2,
                        "n_gs": int((w > 0.01).sum()),
                        "spread_rms": float(dist2.sqrt()),
                    }
                )
        tokens = dtm._token_stats(
            gs_gt,
            torch.zeros_like(gs_gt),
            means,
            br.gaussians_per_token,
            1.0,
            args.major_share,
            args.min_gs,
        )
        token_usable = [t for t in tokens if "purity" in t]
        per_scene[scene_name] = {
            "n_units": len(scene_units),
            "unit_purity_mean": float(np.mean([u["purity"] for u in scene_units]))
            if scene_units else None,
            "unit_n_inst_mean": float(np.mean([u["n_instances"] for u in scene_units]))
            if scene_units else None,
            "unit_pure_ratio": float(
                np.mean([u["purity"] >= args.purity_threshold for u in scene_units])
            ) if scene_units else None,
            "unit_mixed_major_ratio": float(
                np.mean([u["mixed_major"] for u in scene_units])
            ) if scene_units else None,
            "unit_spread_rms_mean": float(
                np.mean([u["spread_rms"] for u in scene_units])
            ) if scene_units else None,
            "token_purity_mean": float(np.mean([t["purity"] for t in token_usable]))
            if token_usable else None,
            "token_n_inst_mean": float(
                np.mean([t["n_instances"] for t in token_usable])
            ) if token_usable else None,
        }
        all_unit.extend(scene_units)
        all_token.extend(token_usable)
        s = per_scene[scene_name]
        print(
            f"[unit-purity] {scene_name}: "
            f"unit_purity={s['unit_purity_mean']:.3f} "
            f"unit_n_inst={s['unit_n_inst_mean']:.1f} "
            f"unit_pure={s['unit_pure_ratio']:.2f} "
            f"unit_mixed={s['unit_mixed_major_ratio']:.2f} "
            f"(token_purity={s['token_purity_mean']:.3f} "
            f"token_n_inst={s['token_n_inst_mean']:.1f}) ({time.time()-t0:.1f}s)"
        )

    global_stats = {
        "unit_purity_mean": float(np.mean([u["purity"] for u in all_unit])),
        "unit_n_inst_mean": float(np.mean([u["n_instances"] for u in all_unit])),
        "unit_pure_ratio": float(
            np.mean([u["purity"] >= args.purity_threshold for u in all_unit])
        ),
        "unit_mixed_major_ratio": float(
            np.mean([u["mixed_major"] for u in all_unit])
        ),
        "unit_spread_rms_mean": float(np.mean([u["spread_rms"] for u in all_unit])),
        "token_purity_mean": float(np.mean([t["purity"] for t in all_token])),
        "token_n_inst_mean": float(np.mean([t["n_instances"] for t in all_token])),
        "n_units_total": len(all_unit),
    }
    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "num_scenes": len(per_scene),
        "global": global_stats,
        "per_scene": per_scene,
    }
    out = Path(args.workspace) / "unit_purity.json"
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n===== GLOBAL =====")
    for kk, vv in global_stats.items():
        print(f"  {kk}: {vv:.4f}" if isinstance(vv, float) else f"  {kk}: {vv}")
    print(f"\n[unit-purity] wrote {out}")


if __name__ == "__main__":
    main()
