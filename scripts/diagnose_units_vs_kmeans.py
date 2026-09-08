"""Compare learned 8 local units vs per-token 3D K-means K=8 oracle units.

Offline diagnostic, no training. Uses the learned 8-unit + GroupToken
checkpoint (v1 token-units) and LSM 40 scenes. For every original token's 64
GS:
  1. K-means (K=8, fixed seed) on GS 3D centers -> oracle spatial clusters;
  2. the model's learned soft GS->unit assignment (argmax for hard units);
  3. Hungarian-align learned units to k-means clusters (permutation allowed);
  4. stats: assignment agreement (hard + soft), ARI / NMI, matched cluster
     center distance (scene-normalized), each learned unit's k-means cluster
     coverage, and instance purity (15-view majority GT) of learned units vs
     oracle clusters.

Answer: are the learned units close to the ideal spatial K=8 units? If yes,
the bottleneck is unit->group binding; if no, unit formation needs work.

Usage:
    python scripts/diagnose_units_vs_kmeans.py
"""

from __future__ import annotations

import argparse
import json
import math
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


def _comb2(x):
    return x * (x - 1) / 2.0


def _ari_nmi(cont: np.ndarray) -> tuple[float, float]:
    """Adjusted Rand Index and normalized mutual information (8x8)."""
    n = int(cont.sum())
    if n == 0:
        return 0.0, 0.0
    a = cont.sum(axis=1)
    b = cont.sum(axis=0)
    sum_ij = float(_comb2(cont).sum())
    sum_a = float(_comb2(a).sum())
    sum_b = float(_comb2(b).sum())
    comb_n = _comb2(n)
    expected = sum_a * sum_b / comb_n if comb_n > 0 else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    ari = (sum_ij - expected) / (max_index - expected + 1e-12)

    p = cont / n
    p_i = a / n
    p_j = b / n
    eps = 1e-12
    mi = float(
        np.sum(p * np.log((p + eps) / (p_i[:, None] * p_j[None, :] + eps)))
    )
    h_u = -float(np.sum(p_i * np.log(p_i + eps)))
    h_v = -float(np.sum(p_j * np.log(p_j + eps)))
    nmi = mi / math.sqrt(h_u * h_v + eps)
    return ari, nmi


def _instance_purity_maps(gs_gt: np.ndarray, ids: np.ndarray, k: int):
    """Per-cluster instance purity (max GT-instance share among FG GS)."""
    purities = []
    for c in range(k):
        sel = ids == c
        votes = gs_gt[sel]
        fg = votes[votes > 0]
        if fg.size == 0:
            continue
        counts = np.bincount(fg)
        purities.append(float(counts.max() / fg.size))
    return purities


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
    parser.add_argument(
        "--workspace", default="workspace/diag_units_vs_kmeans"
    )
    parser.add_argument("--label", default="units_vs_kmeans")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument(
        "--lsm_manifest",
        default=(
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "lsm_instance_eval_manifest.json"
        ),
    )
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--kmeans_seed", type=int, default=0)
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
    from scipy.cluster.vq import kmeans2
    from scipy.optimize import linear_sum_assignment

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
    br = model.instance_branch
    k = int(args.k)
    gpt = br.gaussians_per_token

    per_scene = {}
    pooled = {
        "agreement": [],
        "soft_agreement": [],
        "ari": [],
        "nmi": [],
        "center_dist": [],
        "learned_cover": [],
        "learned_inst_purity": [],
        "oracle_inst_purity": [],
    }
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
            model_input, _ = split_data(data, opt)
            model(data, compute_quality_metrics=False)
            a = br.last_unit_assignment[0]  # [T,P,K] soft
            centers = br.last_unit_centers[0]  # [T,K,3]
            means = br.last_instance_gaussians[0, :, :3].float()  # [N,3]
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
            ids_p, valid = _project_gs_to_views(
                means, cam_views, intrinsics_all, labels_all, tuple(opt.img_size)
            )
            gs_gt = _gs_majority_target(ids_p, valid).cpu().numpy()
        scene_scale = float(
            (means - means.mean(0)).square().sum(-1).mean().sqrt()
        )
        T = a.shape[0]
        learned_hard = a.argmax(-1).cpu().numpy()  # [T,P]
        learned_soft = a.cpu().numpy()
        learned_centers = centers.cpu().numpy()
        means_np = means.cpu().numpy().reshape(T, gpt, 3)
        stats = {
            "agreement": [],
            "soft_agreement": [],
            "ari": [],
            "nmi": [],
            "center_dist": [],
            "learned_cover": [],
            "learned_inst_purity": [],
            "oracle_inst_purity": [],
        }
        for t in range(T):
            pts = means_np[t]
            _, kmeans_ids = kmeans2(
                pts, k, minit="points", seed=args.kmeans_seed
            )
            kmeans_ids = np.asarray(kmeans_ids, dtype=np.int64)
            kmeans_centers = np.stack(
                [
                    pts[kmeans_ids == c].mean(0) if (kmeans_ids == c).any()
                    else pts.mean(0)
                    for c in range(k)
                ]
            )
            l = learned_hard[t]
            c_ids = kmeans_ids
            cont = np.zeros((k, k), dtype=np.float64)
            for x, y in zip(l, c_ids):
                cont[x, y] += 1
            row, col = linear_sum_assignment(-cont)
            perm = np.zeros(k, dtype=np.int64)
            perm[row] = col
            aligned = perm[l]  # learned unit -> aligned kmeans cluster
            agreement = float((aligned == c_ids).mean())
            soft_agreement = float(
                learned_soft[t][np.arange(gpt), aligned].mean()
            )
            ari, nmi = _ari_nmi(cont)
            dists = [
                float(
                    np.linalg.norm(
                        learned_centers[t, u] - kmeans_centers[perm[u]]
                    )
                    / max(scene_scale, 1e-6)
                )
                for u in range(k)
            ]
            cover = []
            for u in range(k):
                sel = l == u
                if not sel.any():
                    continue
                vc = np.bincount(c_ids[sel], minlength=k)
                cover.append(float(vc.max() / vc.sum()))
            li = _instance_purity_maps(gs_gt.reshape(T, gpt)[t], l, k)
            oi = _instance_purity_maps(gs_gt.reshape(T, gpt)[t], c_ids, k)
            stats["agreement"].append(agreement)
            stats["soft_agreement"].append(soft_agreement)
            stats["ari"].append(ari)
            stats["nmi"].append(nmi)
            stats["center_dist"].append(float(np.mean(dists)))
            stats["learned_cover"].extend(cover)
            stats["learned_inst_purity"].extend(li)
            stats["oracle_inst_purity"].extend(oi)
        per_scene[scene_name] = {
            m: float(np.mean(v)) if v else None for m, v in stats.items()
        }
        for m in pooled:
            pooled[m].extend(stats[m])
        s = per_scene[scene_name]
        print(
            f"[units-vs-kmeans] {scene_name}: "
            f"agr={s['agreement']:.3f} soft={s['soft_agreement']:.3f} "
            f"ARI={s['ari']:.3f} NMI={s['nmi']:.3f} "
            f"center_d={s['center_dist']:.3f} "
            f"learned_inst_pur={s['learned_inst_purity']:.3f} "
            f"oracle_inst_pur={s['oracle_inst_purity']:.3f} "
            f"({time.time()-t0:.1f}s)"
        )

    global_stats = {
        m: float(np.mean(pooled[m])) if pooled[m] else None
        for m in pooled
    }
    global_stats["n_tokens"] = len(pooled["agreement"])
    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "num_scenes": len(per_scene),
        "k": k,
        "note": (
            "learned units (argmax of soft assignment) vs per-token 3D "
            "k-means K=8 (fixed seed); Hungarian-aligned; instance purity = "
            "max GT-instance share among foreground GS; center distance "
            "scene-normalized."
        ),
        "global": global_stats,
        "per_scene": per_scene,
    }
    out = Path(args.workspace) / "units_vs_kmeans.json"
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n===== GLOBAL (token-pooled) =====")
    for m, v in global_stats.items():
        if v is not None:
            print(f"  {m}: {v:.4f}" if isinstance(v, float) else f"  {m}: {v}")
    print(f"\n[units-vs-kmeans] wrote {out}")


if __name__ == "__main__":
    main()
