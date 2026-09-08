"""Offline instance-level embedding dispersion diagnostic.

Uses the frozen unit-shaping + patch @2000 identity embedding and the soft
unit instance distributions (p_u, from the 15-view majority-voted pseudo
labels with the same confidence filtering the model uses).  Units are never
hard-assigned to a single instance: intra-instance statistics are weighted by
the soft masses, inter-instance statistics use per-instance mass-weighted
prototype centers, and background is separated via the background column.

Reports, per scene and globally:
  - intra-instance pairwise cosine / Euclidean distance (mean/median/P10/P90)
  - unit-to-own-center distance
  - inter-instance center cosine / distance
  - intra vs inter ratio and margin (own-center cosine minus best other)
  - same/different similarity distributions
  - failure cases: scenes with the lowest baseline LSM AP50 and their
    dispersion numbers.

No training, no checkpoint modification.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.instance_group_head import TokenLocalUnitGrouping
from tokengs.options import config_defaults

from eval_instance_lsm_protocol import (  # noqa: E402
    _LocalAccelerator,
    _load_checkpoint_arch,
)


def _stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p10": float("nan"), "p90": float("nan")}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=(
            "workspace/semantic_v6_unit_shaping_img_train_3000/"
            "checkpoints/model_step_002000.safetensors"
        ),
    )
    parser.add_argument(
        "--workspace", default="workspace/diag_embedding_dispersion"
    )
    parser.add_argument("--label", default="unit_shaping_img_2000")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument("--lsm_manifest", default=(
        "data/scannet_prompt/lsm_instance_eval_manifest.json"
    ))
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument(
        "--baseline_per_scene",
        default=(
            "workspace/clustering_ablation_unit_shaping_img_2000/"
            "clustering_ablation_per_scene.json"
        ),
        help="Per-scene baseline AP50 JSON (variant agg_avg_eps0.5_pw1).",
    )
    parser.add_argument("--max_pairs", type=int, default=200000)
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
        "lsm_manifest_path": str(ROOT / args.lsm_manifest),
    }
    _load_checkpoint_arch(args, opt)
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_ap50 = {}
    if Path(args.baseline_per_scene).is_file():
        bps = json.loads(
            Path(args.baseline_per_scene).read_text(encoding="utf-8")
        )
        for scene, entry in bps.items():
            if "agg_avg_eps0.5_pw1" in entry:
                baseline_ap50[scene] = entry["agg_avg_eps0.5_pw1"]["ap50"]
    print(f"[dispersion] baseline AP50 available for {len(baseline_ap50)} scenes")

    _orig = TokenLocalUnitGrouping._forward_unit_embedding

    def _patched(
        self,
        a,
        unit_feat,
        unit_center,
        means,
        gaussians,
        data,
        opt_inner,
        training,
        model_input,
        batch_size,
        token_count,
        n_gs,
        p,
        dense_features=None,
    ):
        return _orig(
            self,
            a,
            unit_feat,
            unit_center,
            means,
            gaussians,
            data,
            opt_inner,
            training,
            model_input,
            batch_size,
            token_count,
            n_gs,
            p,
            dense_features,
        )

    TokenLocalUnitGrouping._forward_unit_embedding = _patched

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
            print(
                f"[dispersion] frozen-backbone eval: loaded "
                f"{len(loadable)} backbone keys from {backbone_path}"
            )
    model.eval()
    model = model.cuda()
    br = model.instance_branch
    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())

    per_scene: dict[str, dict] = {}
    global_same: list[float] = []
    global_diff: list[float] = []
    global_intra_dist: list[float] = []
    global_inter_center_cos: list[float] = []
    t_start = time.time()
    for i, data in enumerate(test_loader):
        if args.max_scenes > 0 and i >= args.max_scenes:
            break
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        scene_name = str(data["scene_name"][0])
        with torch.inference_mode():
            model(data, compute_quality_metrics=False)
        e = br.last_unit_embeddings[0].cpu().numpy().astype(np.float32)
        pu = br.last_unit_pu[0].cpu().numpy().astype(np.float32)
        bg = int(br.last_unit_bg_idx)
        u = e.shape[0]
        if bg >= 0:
            fg_idx = [j for j in range(pu.shape[1]) if j != bg]
            bg_share = pu[:, bg].mean()
        else:
            fg_idx = list(range(pu.shape[1]))
            bg_share = 0.0
        w = pu[:, fg_idx]  # [U,F]
        total_fg = w.sum(axis=1)
        fg_units = total_fg > 0.05
        valid_w = w[fg_units]
        if valid_w.size == 0 or valid_w.shape[1] == 0:
            print(f"[dispersion] {scene_name}: no fg units")
            continue
        sim = e @ e.T  # [U,U]
        dist = np.sqrt(np.maximum(2.0 - 2.0 * sim, 0.0))
        # --- intra-instance weighted pairwise stats ---
        intra_cos_vals: list[float] = []
        intra_dist_vals: list[float] = []
        own_center_cos_vals: list[float] = []
        own_center_dist_vals: list[float] = []
        centers = np.zeros((valid_w.shape[1], e.shape[1]))
        for fi in range(valid_w.shape[1]):
            wi = valid_w[:, fi]
            if wi.sum() < 1e-6:
                continue
            ci = (wi[:, None] * e[fg_units]).sum(axis=0) / wi.sum()
            ci = ci / max(np.linalg.norm(ci), 1e-8)
            centers[fi] = ci
            sw = float(wi.sum()) ** 2
            if sw < 1e-6:
                continue
            intra_cos_vals.append(
                float(wi @ (sim[fg_units][:, fg_units] @ wi) / sw)
            )
            intra_dist_vals.append(
                float(wi @ (dist[fg_units][:, fg_units] @ wi) / sw)
            )
            c2u = sim[fg_units][:, fg_units] @ (wi / wi.sum())  # [U_fg]
            own_center_cos_vals.append(
                float((wi * c2u).sum() / wi.sum())
            )
            own_center_dist_vals.append(
                float((wi * np.sqrt(np.maximum(2 - 2 * c2u, 0))).sum() / wi.sum())
            )
        centers_n = centers / np.maximum(
            np.linalg.norm(centers, axis=1, keepdims=True), 1e-8
        )
        center_cos = centers_n @ centers_n.T
        off = ~np.eye(center_cos.shape[0], dtype=bool)
        inter_center_cos_vals = center_cos[off]
        inter_center_dist_vals = np.sqrt(
            np.maximum(2 - 2 * inter_center_cos_vals, 0)
        )
        # --- per-unit margin (own soft center vs best other center) ---
        p_norm = valid_w / valid_w.sum(axis=1, keepdims=True).clip(min=1e-8)
        cos_centers = e[fg_units] @ centers_n.T  # [U_fg, F]
        own = (p_norm * cos_centers).sum(axis=1)
        other_mask = (p_norm < 0.5).astype(np.float64)
        best_other = (cos_centers * other_mask).max(axis=1).clip(min=-1.0)
        margin = own - best_other
        mw = total_fg[fg_units]
        margin_mean = float((margin * mw).sum() / mw.sum())
        # --- sampled pairs for the global distributions ---
        rng = np.random.default_rng(0)
        fg_idx_u = np.where(fg_units)[0]
        n_pairs = min(args.max_pairs, len(fg_idx_u) * 50)
        if n_pairs >= 2:
            a_ = rng.choice(fg_idx_u, size=n_pairs)
            b_ = rng.choice(fg_idx_u, size=n_pairs)
            ok = a_ != b_
            a_, b_ = a_[ok], b_[ok]
            wgt = (pu[a_][:, fg_idx] * pu[b_][:, fg_idx]).sum(axis=1)
            same = wgt > 0.5
            global_same.extend(sim[a_[same], b_[same]].tolist())
            global_diff.extend(sim[a_[~same], b_[~same]].tolist())
            global_intra_dist.extend(dist[a_[same], b_[same]].tolist())
        global_inter_center_cos.extend(inter_center_cos_vals.tolist())
        intra_cos = np.asarray(intra_cos_vals)
        intra_dist = np.asarray(intra_dist_vals)
        inter_cos = np.asarray(inter_center_cos_vals)
        inter_dist = np.asarray(inter_center_dist_vals)
        per_scene[scene_name] = {
            "baseline_ap50": baseline_ap50.get(scene_name),
            "n_instances": int(centers_n.shape[0]),
            "bg_share": float(bg_share),
            "fg_unit_frac": float(fg_units.mean()),
            "intra_cos": _stats(intra_cos),
            "intra_dist": _stats(intra_dist),
            "own_center_cos": _stats(np.asarray(own_center_cos_vals)),
            "own_center_dist": _stats(np.asarray(own_center_dist_vals)),
            "inter_center_cos": _stats(inter_cos),
            "inter_center_dist": _stats(inter_dist),
            "ratio_intra_inter_dist": (
                float(intra_dist.mean() / inter_dist.mean())
                if inter_dist.size else float("nan")
            ),
            "margin": float(margin_mean),
        }
        elapsed = time.time() - t_start
        print(
            f"[dispersion] {scene_name} done "
            f"({i + 1}/{min(args.max_scenes or 40, 40)}) elapsed={elapsed:.0f}s"
        )

    def _agg(key: str) -> dict:
        vals = []
        for s in per_scene:
            v = per_scene[s][key]
            vals.append(v["mean"] if isinstance(v, dict) else v)
        return _stats(np.asarray(vals, dtype=np.float64))

    summary = {
        "num_scenes": len(per_scene),
        "intra_cos": _agg("intra_cos"),
        "intra_dist": _agg("intra_dist"),
        "own_center_cos": _agg("own_center_cos"),
        "own_center_dist": _agg("own_center_dist"),
        "inter_center_cos": _agg("inter_center_cos"),
        "inter_center_dist": _agg("inter_center_dist"),
        "ratio_intra_inter_dist": _agg("ratio_intra_inter_dist"),
        "margin": _agg("margin"),
        "global_same_cos": _stats(np.asarray(global_same)),
        "global_diff_cos": _stats(np.asarray(global_diff)),
        "global_intra_dist": _stats(np.asarray(global_intra_dist)),
        "global_inter_center_cos": _stats(np.asarray(global_inter_center_cos)),
    }
    ap_vals = [per_scene[s]["baseline_ap50"] for s in per_scene]
    if all(v is not None for v in ap_vals):
        ap = np.asarray(ap_vals)
        margin = np.asarray([per_scene[s]["margin"] for s in per_scene])
        intra = np.asarray([per_scene[s]["intra_cos"]["mean"] for s in per_scene])
        inter = np.asarray(
            [per_scene[s]["inter_center_cos"]["mean"] for s in per_scene]
        )
        summary["corr_ap_margin"] = float(np.corrcoef(ap, margin)[0, 1])
        summary["corr_ap_intra_cos"] = float(np.corrcoef(ap, intra)[0, 1])
        summary["corr_ap_inter_center_cos"] = float(np.corrcoef(ap, inter)[0, 1])
        order = np.argsort(ap)
        failure = []
        for idx in order[:8]:
            s = list(per_scene)[idx]
            failure.append(
                {
                    "scene": s,
                    "ap50": round(float(ap[idx]), 3),
                    "margin": round(float(margin[idx]), 3),
                    "intra_cos": round(float(intra[idx]), 3),
                    "inter_center_cos": round(float(inter[idx]), 3),
                }
            )
        summary["failure_cases"] = failure

    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "summary": summary,
        "per_scene": per_scene,
    }
    out_json = out_dir / "embedding_dispersion.json"
    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[dispersion] global summary:")
    for key in (
        "intra_cos",
        "intra_dist",
        "own_center_cos",
        "own_center_dist",
        "inter_center_cos",
        "inter_center_dist",
        "ratio_intra_inter_dist",
        "margin",
        "global_same_cos",
        "global_diff_cos",
    ):
        print(f"  {key:22s} {summary[key]}")
    if "failure_cases" in summary:
        print("  failure cases (lowest baseline AP50):")
        for f in summary["failure_cases"]:
            print(f"    {f}")
    print(f"[dispersion] wrote {out_json}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].hist(
            global_same, bins=80, alpha=0.6, label="same-instance"
        )
        axes[0].hist(
            global_diff, bins=80, alpha=0.6, label="different-instance"
        )
        axes[0].set_xlabel("cosine similarity")
        axes[0].legend()
        axes[0].set_title("Pairwise similarity distributions")
        axes[1].hist(
            global_intra_dist, bins=80, alpha=0.7, label="intra-instance"
        )
        axes[1].set_xlabel("Euclidean distance")
        axes[1].legend()
        axes[1].set_title("Intra-instance unit pair distance")
        if all(v is not None for v in ap_vals):
            axes[2].scatter(margin, ap, s=12)
            axes[2].set_xlabel("per-scene margin (own - best other center)")
            axes[2].set_ylabel("baseline AP50")
            axes[2].set_title("AP50 vs embedding margin")
        fig.tight_layout()
        fig.savefig(out_dir / "embedding_dispersion.png", dpi=150)
        print(f"[dispersion] wrote {out_dir / 'embedding_dispersion.png'}")
    except Exception as exc:  # pragma: no cover
        print(f"[dispersion] plot failed: {exc}")


if __name__ == "__main__":
    main()
