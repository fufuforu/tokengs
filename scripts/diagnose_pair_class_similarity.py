"""Offline diagnostic: are hard different-instance pairs mostly same-class?

Frozen checkpoint (unit-shaping + patch @2000), LSM 40 scenes.  For every
pair of different-GT-instance local units we split into:

  - same-class DI:  GT instance differs, dominant semantic class matches;
  - diff-class DI:  GT instance and semantic class both differ.

Semantic classes come from the existing 15-view ScanNet ``label-filt``
labels (c3g8 mapping, projected onto GS centers with the same machinery as
the instance pseudo-labels) -- no new LSeg pipeline is added.  LSeg features
are NOT computed at eval time in this pipeline (they only exist in the
training loss), so LSeg-similarity statistics are skipped and recorded as
unavailable.

Outputs per-scene + global JSON:
  - pair counts / shares for both categories;
  - identity-embedding cosine distribution (mean/median/P10/P90);
  - merge analysis: among units that end up in the same predicted cluster,
    what fraction of different-instance pairs are same-class, split by
    low/high AP50 scenes;
  - correlated-scene scatter (per-scene AP50 vs same-class-DI share).

No training, no model modification.
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
    _audit_lsm_manifest,
    _load_checkpoint_arch,
)


def _project_gs_labels(
    means: torch.Tensor,
    data: dict,
    opt,
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one 2D label field onto GS centers (15 views, majority vote).

    ``field`` is the data key holding per-view integer labels, e.g.
    "instance_label_input" or "semantic_label_input".  Returns
    (gs_label [N], conf [N]) as numpy arrays.
    """
    from tokengs.models.instance_group_head import (
        _gs_majority_target,
        _project_gs_to_views,
    )

    batch_size, n_gs, _ = means.shape
    cam_views_all = torch.cat(
        [data["cam_view_input"], data["cam_view"]], dim=1
    )
    intrinsics_all = torch.cat(
        [data["intrinsics_input"], data["intrinsics"]], dim=1
    )
    labels_all = torch.cat(
        [data[f"{field}_input"], data[f"{field}_output"]], dim=1
    )
    gs_label = torch.zeros(
        (batch_size, n_gs), dtype=torch.long, device=means.device
    )
    gs_conf = torch.zeros(
        (batch_size, n_gs), dtype=torch.float32, device=means.device
    )
    with torch.no_grad():
        for b in range(batch_size):
            ids, valid = _project_gs_to_views(
                means[b],
                cam_views_all[b],
                intrinsics_all[b],
                labels_all[b],
                tuple(opt.img_size),
            )
            gs_label[b] = _gs_majority_target(ids, valid)
            votes = torch.where(valid, ids, torch.full_like(ids, -1))
            n_valid = valid.sum(dim=0).clamp_min(1)
            best = torch.zeros(n_gs, device=means.device)
            for iid in torch.unique(ids[valid]).tolist():
                count = (votes == iid).sum(dim=0).float()
                best = torch.where(count > best, count, best)
            gs_conf[b] = best / n_valid
    return gs_label.cpu().numpy(), gs_conf.cpu().numpy()


def _quantiles(vals: np.ndarray) -> dict:
    if vals.size == 0:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
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
        "--workspace",
        default="workspace/diag_pair_class_similarity",
    )
    parser.add_argument("--label", default="pair_class_2000")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument("--lsm_manifest", default=(
        "data/scannet_prompt/lsm_instance_eval_manifest.json"
    ))
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=15)
    parser.add_argument("--min_pair_mass", type=float, default=0.1,
                        help="Min unit foreground mass to enter pair stats.")
    args = parser.parse_args()

    if args.num_input_views != 8 or args.num_views != 15:
        raise ValueError("LSM protocol requires 8 context + 7 target views")
    manifest_audit = _audit_lsm_manifest(str(ROOT / args.lsm_manifest))

    opt = config_defaults["eval_scannet_lsm_instance"]
    opt.model_type = args.model_type
    opt.resume = args.resume
    opt.workspace = args.workspace
    opt.experiment_name = Path(args.workspace).name
    opt.instance_group_num_groups = int(args.num_groups)
    opt.num_input_views = int(args.num_input_views)
    opt.num_views = int(args.num_views)
    opt.dataset_kwargs = {
        **dict(opt.dataset_kwargs or {}),
        "lsm_manifest_path": str(ROOT / args.lsm_manifest),
    }
    _load_checkpoint_arch(args, opt)
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    _orig_forward = TokenLocalUnitGrouping._forward_unit_embedding

    def _patched_forward(
        self, a, unit_feat, unit_center, means, gaussians, data, opt_inner,
        training, model_input, batch_size, token_count, n_gs, p,
        dense_features=None,
    ):
        out = _orig_forward(
            self, a, unit_feat, unit_center, means, gaussians, data,
            opt_inner, training, model_input, batch_size, token_count, n_gs,
            p, dense_features,
        )
        if not training:
            u_count = token_count * self.units_per_token
            center = unit_center.reshape(batch_size, u_count, 3).float()
            scene_center = center.mean(dim=1, keepdim=True)
            scene_scale = (
                (center - scene_center).square().mean(dim=(1, 2), keepdim=True)
                .sqrt().clamp_min(1e-3)
            )
            self._diag_buffers = {
                "a": a.detach().float(),
                "pos_norm": ((center - scene_center) / scene_scale)
                .detach().float(),
                "means": means.detach().float(),
            }
        return out

    TokenLocalUnitGrouping._forward_unit_embedding = _patched_forward

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
                f"[pair-class] frozen-backbone eval: loaded "
                f"{len(loadable)} backbone keys from {backbone_path}"
            )
    model.eval()
    model = model.cuda()
    br = model.instance_branch

    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())

    per_scene: dict[str, dict] = {}
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
        buf = br._diag_buffers
        batch_size = 1
        token_count = buf["a"].shape[1]
        k = br.units_per_token
        p = buf["a"].shape[2]
        u_count = token_count * k
        n_gs = token_count * p
        a3 = buf["a"][0].reshape(n_gs, k).cpu().numpy()  # [N,K]
        e = br.last_unit_embeddings[0].cpu().numpy()  # [U,D]
        means_np = buf["means"].reshape(batch_size, n_gs, 3)

        # GT instance + semantic class per GS (15-view majority vote).
        gs_inst, _, _ = br._pseudo_gs_labels(means_np, data, opt)
        gs_inst = gs_inst[0].cpu().numpy()
        gs_cls, _ = _project_gs_labels(means_np, data, opt, "semantic_label")
        gs_cls = gs_cls[0]

        # Unit-level soft distributions.
        unit_mass = a3.reshape(token_count, p, k).sum(axis=1)  # [T,K]
        unit_mass = unit_mass.reshape(-1)
        a4 = a3.reshape(token_count, p, k)  # [T,P,K]
        inst_onehot = np.zeros((n_gs, int(gs_inst.max()) + 1), dtype=np.float64)
        inst_onehot[np.arange(n_gs), gs_inst] = 1.0
        cls_onehot = np.zeros(
            (n_gs, int(gs_cls.max()) + 1), dtype=np.float64
        )
        cls_onehot[np.arange(n_gs), gs_cls] = 1.0
        u_inst_mass = np.einsum(
            "tpk,tpm->tkm", a4,
            inst_onehot.reshape(token_count, p, -1),
        ).reshape(u_count, -1)  # [U,M]
        u_cls_mass = np.einsum(
            "tpk,tpc->tkc", a4,
            cls_onehot.reshape(token_count, p, -1),
        ).reshape(u_count, -1)  # [U,C]
        total_mass = unit_mass.clip(min=1e-6)
        # Foreground units: dominant GT instance is not background.
        dom_inst = u_inst_mass.argmax(axis=1)
        fg = (u_inst_mass.max(axis=1) / total_mass) > 0.5
        fg = fg & (dom_inst != 0)
        # Dominant semantic class (0 = unknown/void background).
        dom_cls = u_cls_mass.argmax(axis=1)
        cls_valid = u_cls_mass.max(axis=1) / total_mass > 0.3

        fg_idx = np.where(fg)[0]
        if fg_idx.size < 2:
            per_scene[scene_name] = {
                "fg_units": int(fg_idx.size), "note": "too few fg units",
            }
            continue

        # Different-instance unit pairs (all foreground pairs).
        pairs_same_cls: list[float] = []
        pairs_diff_cls: list[float] = []
        n_same = 0
        n_diff = 0
        n_invalid_cls = 0
        rng = np.random.default_rng(0)
        # Sample different-instance pairs directly (memory-safe): draw
        # candidate index pairs until we collect enough differing-instance
        # pairs (bounded by the total number of fg pairs).
        n_fg = fg_idx.size
        max_pairs = min(n_fg * (n_fg - 1) // 2, 400000)
        aa = np.empty(0, dtype=np.int64)
        bb = np.empty(0, dtype=np.int64)
        n_cand = int(min(max_pairs * 3 + 5000, n_fg * n_fg))
        cand_a = rng.integers(0, n_fg, size=n_cand)
        cand_b = rng.integers(0, n_fg, size=n_cand)
        keep = cand_a != cand_b
        cand_a = fg_idx[cand_a[keep]]
        cand_b = fg_idx[cand_b[keep]]
        if cand_a.size:
            diff_inst = dom_inst[cand_a] != dom_inst[cand_b]
            aa = cand_a[diff_inst]
            bb = cand_b[diff_inst]
        if aa.size > max_pairs:
            sel = rng.choice(aa.size, max_pairs, replace=False)
            aa = aa[sel]
            bb = bb[sel]
        diff_inst = dom_inst[aa] != dom_inst[bb]
        aa = aa[diff_inst]
        bb = bb[diff_inst]
        if aa.size:
            cls_a = dom_cls[aa]
            cls_b = dom_cls[bb]
            valid_cls = cls_valid[aa] & cls_valid[bb] & (cls_a != 0) & (
                cls_b != 0
            )
            same_cls = valid_cls & (cls_a == cls_b)
            cos = (e[aa] * e[bb]).sum(axis=1)
            pairs_same_cls = cos[same_cls]
            pairs_diff_cls = cos[valid_cls & ~same_cls]
            n_same = int(same_cls.sum())
            n_diff = int((valid_cls & ~same_cls).sum())
            n_invalid_cls = int((~valid_cls).sum())

        # Merge analysis: units in the same predicted cluster (Agglomerative
        # eps=0.5, [embedding, pos]) whose GT instances differ.
        from scipy.cluster.hierarchy import fcluster, linkage

        feat = np.concatenate([e, buf["pos_norm"][0].cpu().numpy()], axis=-1)
        z = linkage(feat.astype(np.float32), method="average")
        labels = fcluster(z, t=0.5, criterion="distance") - 1
        same_cls_in_merge = 0
        diff_cls_in_merge = 0
        invalid_cls_in_merge = 0
        merge_pairs = 0
        for c in np.unique(labels):
            mem = np.where((labels == c) & fg)[0]
            if mem.size < 2:
                continue
            # pair units in the cluster
            if mem.size * mem.size <= 4_000_000:
                aa2, bb2 = np.meshgrid(mem, mem, indexing="ij")
                aa2 = aa2.ravel()
                bb2 = bb2.ravel()
                mask2 = aa2 < bb2
                aa2 = aa2[mask2]
                bb2 = bb2[mask2]
            else:
                n_pairs = mem.size * (mem.size - 1) // 2
                sample = min(n_pairs, 200000)
                ia = rng.integers(0, mem.size, size=sample)
                ib = rng.integers(0, mem.size, size=sample)
                ok = ia != ib
                aa2 = mem[ia[ok]]
                bb2 = mem[ib[ok]]
            if aa2.size == 0:
                continue
            di = dom_inst[aa2] != dom_inst[bb2]
            aa2 = aa2[di]
            bb2 = bb2[di]
            if aa2.size == 0:
                continue
            merge_pairs += int(aa2.size)
            valid_cls2 = (
                cls_valid[aa2]
                & cls_valid[bb2]
                & (dom_cls[aa2] != 0)
                & (dom_cls[bb2] != 0)
            )
            sc = valid_cls2 & (dom_cls[aa2] == dom_cls[bb2])
            same_cls_in_merge += int(sc.sum())
            diff_cls_in_merge += int((valid_cls2 & ~sc).sum())
            invalid_cls_in_merge += int((~valid_cls2).sum())

        same_cos_vals = np.asarray(pairs_same_cls, dtype=np.float64)
        diff_cos_vals = np.asarray(pairs_diff_cls, dtype=np.float64)
        # Keep raw values (capped) so the global distribution is exact.
        same_cos_raw = same_cos_vals[:200000].tolist()
        diff_cos_raw = diff_cos_vals[:200000].tolist()
        per_scene[scene_name] = {
            "fg_units": int(fg_idx.size),
            "di_pairs_total": int(n_same + n_diff),
            "di_pairs_same_class": int(n_same),
            "di_pairs_diff_class": int(n_diff),
            "di_pairs_invalid_class": int(n_invalid_cls),
            "same_class_share": float(
                n_same / max(1, n_same + n_diff)
            ),
            "valid_class_share": float(
                (n_same + n_diff) / max(1, n_same + n_diff + n_invalid_cls)
            ),
            "same_class_cos": _quantiles(same_cos_vals),
            "diff_class_cos": _quantiles(diff_cos_vals),
            "same_class_cos_raw": same_cos_raw,
            "diff_class_cos_raw": diff_cos_raw,
            "merge_pairs": int(merge_pairs),
            "merge_same_class_share": float(
                same_cls_in_merge / max(1, same_cls_in_merge + diff_cls_in_merge)
            ),
            "merge_valid_class_share": float(
                (same_cls_in_merge + diff_cls_in_merge)
                / max(1, merge_pairs)
            ),
            "ap50_ref": None,
        }
        elapsed = time.time() - t_start
        print(
            f"[pair-class] {scene_name} done "
            f"({i + 1}/{min(args.max_scenes or 40, 40)}) "
            f"same_class_share={per_scene[scene_name]['same_class_share']:.3f} "
            f"elapsed={elapsed:.0f}s"
        )

    # Merge reference AP50 from the existing baseline eval (best-known).
    ref_ap = {}
    ref_path = (
        ROOT / "workspace" / "lsm_unit_shaping_img_002000_eps0.5" /
        "instance_ap.json"
    )
    if ref_path.is_file():
        ref = json.load(open(ref_path, encoding="utf-8"))
        for sc, entry in ref["per_scene"].items():
            ref_ap[sc] = entry["ap_50"]

    global_pairs_same: list[float] = []
    global_pairs_diff: list[float] = []
    global_same_share = 0.0
    total_pairs = 0
    same_cos_all: list[float] = []
    diff_cos_all: list[float] = []
    for sc, entry in per_scene.items():
        entry["ap50_ref"] = ref_ap.get(sc)
        same_cos_all.extend(entry.get("same_class_cos_raw", []))
        diff_cos_all.extend(entry.get("diff_class_cos_raw", []))
        entry.pop("same_class_cos_raw", None)
        entry.pop("diff_class_cos_raw", None)
        n_s = int(entry.get("di_pairs_same_class", 0))
        n_d = int(entry.get("di_pairs_diff_class", 0))
        n_iv = int(entry.get("di_pairs_invalid_class", 0))
        global_same_share += n_s
        total_pairs += n_s + n_d + n_iv

    # Low-AP scene merge analysis.
    ap_vals = [entry["ap50_ref"] for entry in per_scene.values()
               if entry.get("ap50_ref") is not None]
    low_thr = float(np.median(ap_vals)) if ap_vals else 0.2
    low_scenes = [s for s, e in per_scene.items()
                  if e.get("ap50_ref") is not None and e["ap50_ref"] < low_thr]
    high_scenes = [s for s, e in per_scene.items()
                   if e.get("ap50_ref") is not None and e["ap50_ref"] >= low_thr]

    def _mean_key(scenes: list[str], key: str) -> float:
        vals = [per_scene[s][key] for s in scenes if key in per_scene[s]]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "num_scenes": len(per_scene),
        "total_di_pairs": int(total_pairs),
        "same_class_share_global": float(
            global_same_share / max(1, total_pairs)
        ),
        "same_class_cos": _quantiles(np.asarray(same_cos_all)),
        "diff_class_cos": _quantiles(np.asarray(diff_cos_all)),
        "low_ap_scenes": len(low_scenes),
        "low_ap_median_ap50": low_thr,
        "low_ap_same_class_share": _mean_key(low_scenes, "same_class_share"),
        "high_ap_same_class_share": _mean_key(high_scenes, "same_class_share"),
        "low_ap_merge_same_class_share": _mean_key(
            low_scenes, "merge_same_class_share"
        ),
        "high_ap_merge_same_class_share": _mean_key(
            high_scenes, "merge_same_class_share"
        ),
        "lseg_note": (
            "LSeg features are not computed at eval time in this pipeline "
            "(training-loss-only); skipped by design."
        ),
    }
    payload = {
        "label": args.label,
        "checkpoint": args.resume,
        "protocol": manifest_audit,
        "summary": summary,
        "per_scene": per_scene,
    }
    (out_dir / "pair_class_similarity.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[pair-class] summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
