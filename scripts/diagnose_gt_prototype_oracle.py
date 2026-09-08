"""GT-instance prototype matching oracle (offline, no training).

Uses the frozen wide7l@8000 backbone + LSM 40 held-out scenes. Per-GS GT
instance id comes from the 15-view projection + majority vote (same
convention as the 3D loss). For every GT instance a scene-specific prototype
is the mean of its GS features. Each foreground GS is assigned to the nearest
prototype (cosine), background stays void, and the assignment is rendered
through the real frozen Gaussian geometry into the 7 target views, evaluated
with the exact LSM AP protocol.

Variants (per-GS feature construction, no learned parameters):
  - hidden:      L2-normalized token hidden only (per-token feature; GS
                 inside a token are indistinguishable -> token-level ceiling)
  - feature:     token hidden + local geometry (relative 3D position within
                 the token + log-scale + opacity) -- no global position
  - feature3d:   feature + global scene-normalized 3D position

Answers: can the frozen TokenGS features separate instances if we give each
scene its GT prototypes? What is AP50 for feature-only vs feature+3D?

Usage:
    python scripts/diagnose_gt_prototype_oracle.py
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
from diagnose_hierarchical_oracle import _eval_scene, _render_group_labels

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


VARIANTS = {
    "hidden": {"h": 1.0},
    "feature": {"h": 1.0, "rel": 1.0, "so": 1.0},
    "feature3d": {"h": 1.0, "rel": 1.0, "so": 1.0, "gp": 1.0},
    "pos": {"gp": 1.0},
    "feature3d_w3": {"h": 1.0, "gp": 3.0},
}


def _build_features(
    token_hidden: torch.Tensor,
    means: torch.Tensor,
    gaussians: torch.Tensor,
    gaussians_per_token: int,
) -> dict[str, torch.Tensor]:
    """Per-GS feature blocks (L2-normalized), returned per block."""
    B, T, D = token_hidden.shape
    N = gaussians.shape[1]
    means = means.view(B, T, gaussians_per_token, 3)
    scene_center = means.mean(dim=(1, 2), keepdim=True)
    scene_scale = (
        (means - scene_center).square().mean(dim=(1, 2, 3), keepdim=True)
        .sqrt()
        .clamp_min(1e-3)
    )
    anchor = means.mean(dim=2, keepdim=True)
    rel_pos = (means - anchor) / scene_scale  # [B,T,P,3]
    global_pos = (means - scene_center) / scene_scale
    scale_op = torch.cat(
        [
            gaussians[..., 4:7].log().view(B, T, gaussians_per_token, 3),
            gaussians[..., 3:4].view(B, T, gaussians_per_token, 1),
        ],
        dim=-1,
    )  # [B,T,P,4]

    def norm(x):
        return _F_normalize(x, dim=-1)

    h_block = token_hidden.unsqueeze(2).expand(B, T, gaussians_per_token, D)
    h_norm = norm(h_block)
    rel_norm = norm(rel_pos)
    gp_norm = norm(global_pos)
    so_norm = norm(scale_op)
    flat = lambda x: x.reshape(B, N, -1)
    return {
        "h": flat(h_norm),
        "rel": flat(rel_norm),
        "so": flat(so_norm),
        "gp": flat(gp_norm),
    }


def _F_normalize(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-8)


def _prototype_assign(
    feats: dict[str, torch.Tensor], gs_gt: torch.Tensor, weights: dict[str, float]
):
    """Weighted nearest-prototype hard assignment (per scene)."""
    B, N, _ = feats["h"].shape
    feat = next(iter(feats.values()))
    device = feat.device
    labels = torch.zeros(B, N, dtype=torch.long, device=device)
    for b in range(B):
        ids = [int(i) for i in torch.unique(gs_gt[b][gs_gt[b] > 0])]
        if not ids:
            continue
        id_to_idx = {i: idx for idx, i in enumerate(ids)}
        fg = gs_gt[b] > 0
        sim = torch.zeros(int(fg.sum()), len(ids), device=device)
        for name, w in weights.items():
            block = feats[name]
            proto = torch.zeros(len(ids), block.shape[-1], device=device)
            for i, idx in id_to_idx.items():
                sel = gs_gt[b] == i
                proto[idx] = block[b][sel].mean(dim=0)
            proto = _F_normalize(proto)
            sim = sim + w * (block[b][fg] @ proto.t())
        assigned = torch.argmax(sim, dim=-1)
        fg_ids = torch.tensor([id_to_idx[i] for i in ids], device=device)
        labels[b][fg] = fg_ids[assigned]
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=(
            "/space0/mawb/tokengs/workspace/"
            "semantic_v6_open_vocab_full_ce2_wide7l_train_8000/"
            "checkpoints/model_step_008000.safetensors"
        ),
    )
    parser.add_argument(
        "--workspace", default="workspace/diag_gt_prototype_oracle"
    )
    parser.add_argument("--label", default="gt_prototype_oracle")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument(
        "--lsm_manifest",
        default=(
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "lsm_instance_eval_manifest.json"
        ),
    )
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--min_pred_pixels", type=int, default=1)
    parser.add_argument("--min_gt_pixels", type=int, default=1)
    parser.add_argument("--max_predictions_per_image", type=int, default=100)
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
    gaussians_per_token = opt.dec_patch_size**2

    per_config = {
        v: {
            "per_scene": {},
            "pred": [],
            "scores": [],
            "pids": [],
            "gt": [],
            "gids": [],
        }
        for v in VARIANTS
    }
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
            feats = _build_features(
                gs_token_hidden, means, gaussians, gaussians_per_token
            )
            instance_labels = data["instance_label_output"].long()
        for v in VARIANTS:
            labels = _prototype_assign(
                feats, gs_gt.unsqueeze(0), VARIANTS[v]
            )
            with torch.no_grad():
                rp, num_groups = _render_group_labels(
                    gaussians,
                    labels[0],
                    model_input.decoder.cam_view,
                    model_input.decoder.intrinsics,
                    opt,
                    model,
                )
            res, pm, psc, pid, gm, gid = _eval_scene(
                rp, instance_labels, scene_name, args
            )
            per_config[v]["per_scene"][scene_name] = res
            per_config[v]["pred"].extend(pm)
            per_config[v]["scores"].extend(psc)
            per_config[v]["pids"].extend(pid)
            per_config[v]["gt"].extend(gm)
            per_config[v]["gids"].extend(gid)
        print(
            f"[proto-oracle] {scene_name}: "
            + "  ".join(
                f"{v}={per_config[v]['per_scene'][scene_name]['ap50']:.3f}"
                for v in VARIANTS
            )
            + f" ({time.time()-t0:.1f}s)"
        )

    from tokengs.utils.instance_ap import instance_ap

    global_stats = {}
    for v in VARIANTS:
        per = per_config[v]["per_scene"]
        macro = {
            m: float(np.mean([e[m] for e in per.values()]))
            for m in ("ap", "ap25", "ap50", "ap75")
        }
        pooled = instance_ap(
            per_config[v]["pred"],
            per_config[v]["scores"],
            per_config[v]["gt"],
            thresholds=(0.25, 0.5, 0.75),
            vectorized=True,
            pred_image_ids=per_config[v]["pids"],
            gt_image_ids=per_config[v]["gids"],
        )
        macro.update(
            {
                "pooled_ap50": pooled["ap_50"],
                "num_pred_mean": float(
                    np.mean([e["num_pred"] for e in per.values()])
                ),
                "num_gt_mean": float(
                    np.mean([e["num_gt"] for e in per.values()])
                ),
            }
        )
        global_stats[v] = macro

    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "num_scenes": len(next(iter(per_config.values()))["per_scene"]),
        "note": (
            "GT prototypes = per-instance mean of frozen GS features; "
            "nearest-prototype cosine hard assignment; background -> void; "
            "exact LSM eval. feature = token hidden + rel pos + scale/opacity; "
            "feature3d adds global normalized 3D position."
        ),
        "global": global_stats,
        "per_scene": {
            v: per_config[v]["per_scene"] for v in VARIANTS
        },
    }
    out = Path(args.workspace) / "gt_prototype_oracle.json"
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n===== GLOBAL =====")
    for v in VARIANTS:
        g = global_stats[v]
        print(
            f"  {v:9s} AP25={g['ap25']:.3f} AP50={g['ap50']:.3f} "
            f"AP={g['ap']:.3f} pooledAP50={g['pooled_ap50']:.3f} "
            f"pred={g['num_pred_mean']:.1f} gt={g['num_gt_mean']:.1f}"
        )
    print(f"\n[proto-oracle] wrote {out}")


if __name__ == "__main__":
    main()
