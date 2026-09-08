"""Diagnose the identity embedding's same/different-instance separability.

Offline, no training. Uses the best checkpoint (unit-shaping + patch @2000).
For every LSM scene it runs the frozen model forward, reads the trained unit
embeddings and the per-unit soft instance distributions p_u (from 15-view
majority-voted GS pseudo labels), then:
  - same-instance unit pairs (dominant instance equal, share >= 0.5, not
    background) vs different-instance pairs: cosine similarity distributions
    with mean / median / P10 / P90 / overlap / separation;
  - unit-level kNN instance agreement (from the model output);
  - pseudo-label confidence stats and small-instance retention under the
    high-confidence filter (conf>=0.6, >=3 views, unit mass >=0.25), so we
    can judge whether denoising would discard small instances.

Usage:
    python scripts/diagnose_embedding_distribution.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch._dynamo.config.disable = True
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_token_instance_mixing as dtm

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.options import config_defaults


class _LocalAccelerator:
    is_main_process = True


def _stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=(
            "/space0/mawb/tokengs/workspace/"
            "semantic_v6_unit_shaping_img_train_3000/"
            "checkpoints/model_step_002000.safetensors"
        ),
    )
    parser.add_argument(
        "--workspace", default="workspace/diag_embedding_distribution"
    )
    parser.add_argument("--label", default="embedding_distribution")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument(
        "--lsm_manifest",
        default=(
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "lsm_instance_eval_manifest.json"
        ),
    )
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--max_pairs", type=int, default=200000)
    parser.add_argument("--share_threshold", type=float, default=0.5)
    parser.add_argument("--conf_threshold", type=float, default=0.6)
    parser.add_argument("--min_views", type=int, default=3)
    parser.add_argument("--unit_min_mass", type=float, default=0.25)
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
    br = model.instance_branch

    all_same, all_diff = [], []
    knn_list = []
    conf_gs_keep, conf_unit_keep, inst_covered = [], [], []
    per_scene = {}
    for data in test_loader:
        if args.max_scenes > 0 and len(per_scene) >= args.max_scenes:
            break
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        scene_name = data["scene_name"][0]
        with torch.no_grad():
            out = model(data, compute_quality_metrics=False)
            e = br.last_unit_embeddings[0]  # [U,D]
            p_u = br.last_unit_pu[0]  # [U,m]
            knn_list.append(float(out["unit_knn_agreement"]))
            gs_gt, gs_conf, gs_nviews = br._pseudo_gs_labels(
                br.last_instance_gaussians[0, :, :3].float().unsqueeze(0),
                data,
                opt,
            )
        gs_gt0 = gs_gt[0].cpu().numpy()
        gs_conf0 = gs_conf[0].cpu().numpy()
        gs_nv0 = gs_nviews[0].cpu().numpy()
        kept = (gs_conf0 >= args.conf_threshold) & (
            gs_nv0 >= args.min_views
        )
        conf_gs_keep.append(float(kept.mean()))
        # per-instance retention (only foreground instances with >=4 GS)
        n_inst = 0
        n_inst_covered = 0
        for iid in np.unique(gs_gt0[gs_gt0 > 0]):
            sel = gs_gt0 == iid
            if sel.sum() < 4:
                continue
            n_inst += 1
            if kept[sel].any():
                n_inst_covered += 1
        inst_covered.append(n_inst_covered / max(n_inst, 1) if n_inst else 1.0)

        p_np = p_u.cpu().numpy()
        dom = p_np.argmax(-1)
        share = p_np.max(-1)
        bg_idx = int(br.last_unit_bg_idx)
        fg = (share >= args.share_threshold) & (dom != bg_idx)
        e_np = e.cpu().numpy()
        U = e_np.shape[0]
        rng = np.random.default_rng(0)
        # sample pairs among foreground-dominant units
        fg_idx = np.where(fg)[0]
        same_sims, diff_sims = [], []
        if fg_idx.size >= 2:
            n_pairs = min(args.max_pairs, fg_idx.size * 50)
            a_ = rng.choice(fg_idx, size=n_pairs)
            b_ = rng.choice(fg_idx, size=n_pairs)
            ok = a_ != b_
            a_, b_ = a_[ok], b_[ok]
            same_mask = dom[a_] == dom[b_]
            sims = (e_np[a_] * e_np[b_]).sum(-1)
            same_sims = sims[same_mask]
            diff_sims = sims[~same_mask]
        all_same.extend(same_sims.tolist())
        all_diff.extend(diff_sims.tolist())
        ss, ds = _stats(same_sims), _stats(diff_sims)
        per_scene[scene_name] = {
            "same": ss,
            "diff": ds,
            "knn": knn_list[-1],
            "gs_kept": conf_gs_keep[-1],
            "unit_kept": float(out["pseudo_unit_kept_ratio"]),
            "inst_covered": inst_covered[-1],
        }
        print(
            f"[embed-dist] {scene_name}: same={ss.get('mean', -1):.3f} "
            f"diff={ds.get('mean', -1):.3f} knn={knn_list[-1]:.3f} "
            f"gs_kept={conf_gs_keep[-1]:.2f} unit_kept="
            f"{float(out['pseudo_unit_kept_ratio']):.2f} "
            f"inst_covered={inst_covered[-1]:.2f}"
        )

    same_arr = np.asarray(all_same)
    diff_arr = np.asarray(all_diff)
    same_s, diff_s = _stats(same_arr), _stats(diff_arr)
    overlap = (
        float((diff_arr > np.median(same_arr)).mean()) if same_arr.size else None
    )
    pooled_std = float(
        np.sqrt(
            (same_arr.std() ** 2 + diff_arr.std() ** 2) / 2
        )
    )
    separation = (
        float((same_arr.mean() - diff_arr.mean()) / pooled_std)
        if same_arr.size and diff_arr.size and pooled_std > 0
        else None
    )
    global_stats = {
        "same": same_s,
        "diff": diff_s,
        "overlap_frac_above_same_median": overlap,
        "separation_in_pooled_std": separation,
        "knn_agreement_mean": float(np.mean(knn_list)),
        "gs_kept_mean": float(np.mean(conf_gs_keep)),
        "inst_covered_mean": float(np.mean(inst_covered)),
    }
    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "num_scenes": len(per_scene),
        "config": {
            "share_threshold": args.share_threshold,
            "conf_threshold": args.conf_threshold,
            "min_views": args.min_views,
            "unit_min_mass": args.unit_min_mass,
        },
        "global": global_stats,
        "per_scene": per_scene,
    }
    out = Path(args.workspace) / "embedding_distribution.json"
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1, 1, 81)
    ax.hist(same_arr, bins=bins, alpha=0.6, density=True, label="same-instance pairs")
    ax.hist(diff_arr, bins=bins, alpha=0.6, density=True, label="different-instance pairs")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.legend()
    ax.set_title(
        f"unit embedding similarity: same vs diff (kNN={global_stats['knn_agreement_mean']:.2f})"
    )
    fig.tight_layout()
    fig.savefig(Path(args.workspace) / "embedding_distribution.png", dpi=110)
    plt.close(fig)

    print("\n===== GLOBAL =====")
    for k, v in global_stats.items():
        if isinstance(v, dict):
            print(f"  {k}: {v}")
        elif v is not None:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\n[embed-dist] wrote {out}")


if __name__ == "__main__":
    main()
