"""GT-oracle empirical ceiling audit (read-only, no model training).

Optimizes two temporary assignment parameterizations on top of the FROZEN
Student GS of Both@1420 and compares with the learned prediction:

  * Unit oracle  : free logits per shared unit [T*U, K+1], softmax broadcast
                   to the unit's 8 GS;
  * Per-GS oracle: free logits per GS [N, K+1];

K = number of scene-global GT instance ids in the sample (no Hungarian, no
query permutation).  Rendering uses the exact model renderer / visibility /
void normalization at the official resolution.  Temporary parameters are
never written back.  Also reports Student-GS foreground/alpha coverage,
boundary F-scores, interior/boundary IoU and per-GS mixed-unit statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safetensors.torch import load_file  # noqa: E402
from scipy.ndimage import binary_dilation, binary_erosion  # noqa: E402

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


class _LocalAccelerator:
    is_main_process = True

    @staticmethod
    def print(*args, **kwargs):
        print(*args, **kwargs)


_CFG = (
    "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
)
_RESUME = (
    "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_"
    "t3e6_ddp8/checkpoints/model_step_001420.safetensors"
)


def _hash_model(model):
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        digest.update(name.encode())
        digest.update(
            param.detach().float().cpu().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()[:24]


def _load_heads(model, ckpt):
    abs_state = {
        key.split(".", 1)[1]: value
        for key, value in ckpt.items()
        if key.startswith("absolute_gs_head.")
    }
    res = model.absolute_gs_head.load_state_dict(abs_state, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    tsh_state = {
        key.split(".", 1)[1]: value
        for key, value in ckpt.items()
        if key.startswith("tsh_instance_head.")
    }
    res = model.tsh_instance_head.load_state_dict(tsh_state, strict=True)
    assert not res.missing_keys and not res.unexpected_keys


def _normalize_labels(arr):
    while arr.ndim > 3:
        arr = arr[0]
    return arr


def _render_probs(renderer, gs, feats, cam_view, intrinsics):
    render = renderer.render_feature_channels(
        gs, feats, cam_view, intrinsics=intrinsics
    )
    channels = render["images_pred"]  # [B,V,C,H,W]
    alpha = render["alphas_pred"]
    normed = channels / (alpha + 1e-5)
    probs = normed / normed.sum(dim=2, keepdim=True).clamp_min(1e-6)
    return probs


def _per_view_ce_dice(prob, target, void_channel, void_weight=0.1):
    """prob [B,C,H,W], target [B,H,W]."""
    p = prob.clamp(1e-6, 1.0 - 1e-6)
    losses = []
    for c in range(p.shape[1]):
        mask = target == c
        if not bool(mask.any()):
            continue
        w = void_weight if c == void_channel else 1.0
        losses.append(w * (-(p[:, c][mask]).log().mean()))
    ce = sum(losses) / max(1, len(losses))
    dice = []
    for c in range(p.shape[1]):
        if c == void_channel:
            continue
        gt_bin = (target == c).float()
        if bool(gt_bin.sum() < 25):
            continue
        dice.append(
            1.0
            - (2.0 * (p[:, c] * gt_bin).sum() + 1.0)
            / (p[:, c].sum() + gt_bin.sum() + 1.0)
        )
    return ce + (0.5 * sum(dice) / max(1, len(dice)) if dice else 0.0)


def _oracle_loss(probs, targets, void_channel):
    loss = None
    for v in range(probs.shape[1]):
        lv = _per_view_ce_dice(
            probs[:, v], targets[v : v + 1], void_channel
        )
        loss = lv if loss is None else loss + lv
    return loss / probs.shape[1]


def _optimize(logits, targets, void_channel, renderer, gs, cam_view,
              intrinsics, steps, unit_mode=False):
    param = torch.nn.Parameter(logits)
    optimizer = torch.optim.Adam([param], lr=1.5)
    history = []
    last = None
    t0 = time.time()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        soft = F.softmax(param.float(), dim=-1)
        if unit_mode:
            soft = soft.repeat_interleave(8, dim=0)
        feats = soft.unsqueeze(0).float()
        probs = _render_probs(renderer, gs, feats, cam_view, intrinsics)
        loss = _oracle_loss(probs, targets, void_channel)
        if not torch.isfinite(loss):
            history.append({"step": step, "loss": None})
            break
        loss.backward()
        optimizer.step()
        last = float(loss.detach().item())
        if step in (0, 25, 50, 100, 150, 200, 250, 300) or step == steps - 1:
            history.append({"step": step, "loss": last})
    return param.detach(), history, last, time.time() - t0


def _iou(a, b):
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _boundary_f(pred, gt, tol):
    ep = pred ^ binary_dilation(pred, iterations=1)
    eg = gt ^ binary_dilation(gt, iterations=1)
    tp = float(
        np.logical_or(
            np.logical_and(ep, binary_dilation(eg, iterations=tol)),
            np.logical_and(eg, binary_dilation(ep, iterations=tol)),
        ).sum()
    )
    np_ = float(ep.sum())
    ng = float(eg.sum())
    if np_ + ng == 0:
        return 1.0
    return 2 * tp / max(1e-6, 2 * tp + np_ + ng)


def _pair_metrics(pred, gt):
    iou = _iou(pred, gt)
    if not bool(pred.any()):
        b = {"1": 0.0, "2": 0.0, "4": 0.0, "8": 0.0}
    else:
        b = {str(t): _boundary_f(pred, gt, t) for t in (1, 2, 4, 8)}
    band = binary_dilation(gt, iterations=2) ^ binary_erosion(
        gt, iterations=2
    )
    interior = ~band
    interior_iou = _iou(
        np.logical_and(pred, interior),
        np.logical_and(gt, interior),
    ) if bool(gt[interior].any()) else 0.0
    band_iou = _iou(
        np.logical_and(pred, band),
        np.logical_and(gt, band),
    ) if bool(gt[band].any()) else 0.0
    return {
        "iou": iou,
        "boundary_f": b,
        "interior_iou": interior_iou,
        "band_iou": band_iou,
    }


def _summ(values):
    if not values:
        return None
    o = sorted(values)
    return {
        "min": o[0],
        "p10": o[max(0, int(0.1 * (len(o) - 1)))],
        "median": statistics.median(o),
        "p90": o[min(len(o) - 1, int(0.9 * (len(o) - 1)))],
        "max": o[-1],
        "mean": statistics.fmean(o),
        "count": len(o),
    }


def _metric_accumulate(records, name, pred_fn, gt_by_id, view_count):
    """pred_fn(view, gid) -> mask or None."""
    ious = []
    bfs = {t: [] for t in (1, 2, 4, 8)}
    interior = []
    band = []
    per_gt_best = {}
    for gid, masks in gt_by_id.items():
        best = 0.0
        for v in range(view_count):
            gt = masks[v]
            if gt is None or not bool(gt.any()):
                continue
            pred = pred_fn(v, gid)
            if pred is None:
                pm = {
                    "iou": 0.0,
                    "boundary_f": {str(t): 0.0 for t in (1, 2, 4, 8)},
                    "interior_iou": 0.0,
                    "band_iou": 0.0,
                }
            else:
                pm = _pair_metrics(pred, gt)
                best = max(best, pm["iou"])
            ious.append(pm["iou"])
            for t in bfs:
                bfs[t].append(pm["boundary_f"][str(t)])
            interior.append(pm["interior_iou"])
            band.append(pm["band_iou"])
        per_gt_best[gid] = best
    return {
        name + "_iou": _summ(ious),
        name + "_mean_iou": float(np.mean(ious)) if ious else None,
        name + "_best_gt_iou": float(np.mean(list(per_gt_best.values())))
        if per_gt_best
        else None,
        name + "_recall25": (
            sum(1 for x in ious if x >= 0.25) / max(1, len(ious))
        ),
        name + "_recall50": (
            sum(1 for x in ious if x >= 0.50) / max(1, len(ious))
        ),
        name + "_recall75": (
            sum(1 for x in ious if x >= 0.75) / max(1, len(ious))
        ),
        name + "_boundary_f": {t: _summ(bfs[t]) for t in bfs},
        name + "_interior_iou": _summ(interior),
        name + "_band_iou": _summ(band),
        name + "_per_gt_best": per_gt_best,
    }


def _foreground(pred_fn_per_view, gt_unions, void_label=1000000):
    pred_fg = [pred_fn_per_view(v) for v in range(len(gt_unions))]
    gt_all = np.concatenate(gt_unions)
    pred_all = np.concatenate(pred_fg)
    return {
        "iou": _iou(gt_all, pred_all),
        "precision": (
            float(np.logical_and(pred_all, gt_all).sum())
            / max(1, float(pred_all.sum()))
        ),
        "recall": (
            float(np.logical_and(pred_all, gt_all).sum())
            / max(1, float(gt_all.sum()))
        ),
        "gt_uncovered": (
            float(np.logical_and(gt_all, ~pred_all).sum())
            / max(1, float(gt_all.sum()))
        ),
        "bg_leak": (
            float(np.logical_and(pred_all, ~gt_all).sum())
            / max(1, float(pred_all.sum()))
        ),
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace/tsh_gs_ceiling")
    parser.add_argument("--resume", default=_RESUME)
    parser.add_argument("--config-name", default=_CFG)
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--n-scenes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = config_defaults[args.config_name]
    opt.workspace = str(out_dir)
    opt.num_workers = 0
    torch.manual_seed(42)
    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    iterator = iter(test_loader)
    for _ in range(args.scene_start):
        next(iterator)

    model = model_registry[opt.model_type](opt).cuda().eval()
    _load_heads(model, load_file(args.resume, device="cpu"))
    hash_before = _hash_model(model)
    renderer = model.gs
    reports = []
    seen_scenes = set()
    while len(reports) < args.n_scenes:
        try:
            data = next(iterator)
        except StopIteration:
            break
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        scene = str(data["scene_name"][0])
        if scene in seen_scenes:
            continue
        seen_scenes.add(scene)
        with torch.inference_mode():
            out = model(data, compute_quality_metrics=True)
        gs = out["gaussians"].detach().clone().float().contiguous()
        cam_view = data["cam_view"].float()
        intrinsics = data["intrinsics"].float()
        labels = _normalize_labels(
            data["instance_label_output"].long().cpu().numpy()
        )
        rendered_views = int(
            out["rendered_instance_group_probability"].shape[2]
        )
        labels = labels[:rendered_views]
        view_count = labels.shape[0]
        id_list = sorted(
            {
                int(x)
                for v in range(view_count)
                for x in np.unique(labels[v])
                if int(x) not in (0, 255)
            }
        )
        id2ch = {gid: i for i, gid in enumerate(id_list)}
        void_channel = len(id_list)
        targets = np.full(labels.shape, void_channel, dtype=np.int64)
        for gid, ch in id2ch.items():
            targets[labels == gid] = ch
        target_t = torch.from_numpy(targets).cuda().long()
        gt_by_id = {}
        gt_unions = []
        for v in range(view_count):
            union = np.zeros_like(labels[v], dtype=bool)
            for gid in id_list:
                m = labels[v] == gid
                if bool(m.any()):
                    gt_by_id.setdefault(gid, [None] * view_count)[v] = m
                    union |= m
            gt_unions.append(union)

        # alpha foreground coverage
        with torch.inference_mode():
            alpha = renderer.render_feature_channels(
                gs,
                torch.ones(
                    gs.shape[0], gs.shape[1], 1, device=gs.device
                ),
                cam_view,
                intrinsics=intrinsics,
            )["alphas_pred"][0, :, 0].cpu().numpy()
        fg_alpha = _foreground(
            lambda v: alpha[v] >= 0.5, gt_unions
        )

        # unit & per-GS oracle optimization
        n_unit = int(opt.num_gs_tokens) * 8
        n_gs = gs.shape[1]
        u_logits, u_hist, u_loss, u_t = _optimize(
            torch.zeros(n_unit, void_channel + 1, device=gs.device),
            target_t, void_channel, renderer, gs, cam_view, intrinsics,
            args.steps, unit_mode=True,
        )
        g_logits, g_hist, g_loss, g_t = _optimize(
            torch.zeros(n_gs, void_channel + 1, device=gs.device),
            target_t, void_channel, renderer, gs, cam_view, intrinsics,
            args.steps,
        )

        # render final class maps
        def _final_cmap(logits, per_gs):
            if per_gs:
                assign = F.softmax(logits, dim=-1)
            else:
                assign = F.softmax(logits, dim=-1).repeat_interleave(8, dim=0)
            feats = assign.unsqueeze(0).float()
            with torch.inference_mode():
                probs = _render_probs(
                    renderer, gs, feats, cam_view, intrinsics
                )
            cmap = probs[0].argmax(dim=1).cpu().numpy()  # [V,H,W]
            return cmap, assign.detach().cpu().numpy()

        unit_cmap, unit_assign = _final_cmap(u_logits, False)
        gs_cmap, gs_assign = _final_cmap(g_logits, True)

        # learned prediction (model output, 100 learned queries)
        lp = out["rendered_instance_group_probability"][0].float().cpu()
        learned_cmap = lp.argmax(dim=0)[:, 0].numpy()
        learned_void = lp.shape[0] - 1

        def _cmap_fn(cmap, void_ch):
            def pred_fn(v, gid):
                return cmap[v] == id2ch[gid]
            return pred_fn, (
                lambda v: cmap[v] != void_ch
            )

        unit_fn, unit_fg_fn = _cmap_fn(unit_cmap, void_channel)
        gs_fn, gs_fg_fn = _cmap_fn(gs_cmap, void_channel)

        def _learned_fn(v, gid):
            gt = gt_by_id[gid][v]
            if gt is None or not bool(gt.any()):
                return None
            best = None
            best_iou = 0.0
            for g in range(learned_void):
                pm = learned_cmap[v] == g
                if not bool(pm.any()):
                    continue
                i = _iou(pm, gt)
                if i > best_iou:
                    best_iou = i
                    best = pm
            return best

        scene_rec = {
            "scene": scene,
            "gt_ids": id_list,
            "learned": _metric_accumulate(
                {}, "learned", _learned_fn, gt_by_id, view_count
            ),
            "unit_oracle": _metric_accumulate(
                {}, "unit", unit_fn, gt_by_id, view_count
            ),
            "per_gs_oracle": _metric_accumulate(
                {}, "per_gs", gs_fn, gt_by_id, view_count
            ),
            "foreground_alpha": fg_alpha,
            "foreground_unit": _foreground(
                unit_fg_fn, gt_unions
            ),
            "foreground_per_gs": _foreground(
                gs_fg_fn, gt_unions
            ),
            "unit_history": u_hist,
            "per_gs_history": g_hist,
            "unit_final_loss": u_loss,
            "per_gs_final_loss": g_loss,
            "unit_seconds": u_t,
            "per_gs_seconds": g_t,
        }
        # per-GS mixed-unit statistics
        gs_labels = np.argmax(gs_assign, axis=-1)
        per_unit_probs = gs_assign.reshape(
            gs_assign.shape[0] // 8, 8, gs_assign.shape[1]
        )
        mean_unit_prob = per_unit_probs.mean(axis=1)
        unit_label = mean_unit_prob.argmax(axis=-1)
        unit_label_gs = gs_labels.reshape(-1, 8)
        distinct_nv = [
            len(
                {
                    int(x)
                    for x in row
                    if x != void_channel
                }
            )
            for row in unit_label_gs
        ]
        distinct_all = [
            len({int(x) for x in row}) for row in unit_label_gs
        ]
        mixed = [d >= 2 for d in distinct_nv]
        purity = mean_unit_prob.max(axis=-1)
        ent = -(mean_unit_prob * np.log(
            mean_unit_prob.clip(1e-8)
        )).sum(-1)
        gs_of_gt = {}
        unit_of_gt = {}
        for g in range(len(id_list)):
            sel = gs_labels == g
            if not bool(sel.any()):
                continue
            unit_ids = set((np.where(sel)[0] // 8).tolist())
            gs_of_gt[id_list[g]] = int(sel.sum())
            unit_of_gt[id_list[g]] = len(unit_ids)
        gs_np = gs.detach().float().cpu().numpy()
        if gs_np.ndim == 3:
            gs_np = gs_np[0]
        scale_per_gs = np.linalg.norm(gs_np[:, 4:7], axis=1)
        unit_scale = scale_per_gs.reshape(-1, 8).mean(axis=1)
        scene_rec["mixed_unit_stats"] = {
            "mixed_nv_ratio": float(np.mean(mixed)),
            "mixed_any_ratio": float(
                np.mean([d >= 2 for d in distinct_all])
            ),
            "distinct_nv": _summ(distinct_nv),
            "distinct_all": _summ(distinct_all),
            "unit_purity": _summ(purity.tolist()),
            "unit_entropy": _summ(ent.tolist()),
            "gs_per_gt": _summ(list(gs_of_gt.values())),
            "units_per_gt": _summ(list(unit_of_gt.values())),
            "mixed_rate_by_scale_quartile": {
                str(q + 1): float(
                    np.mean(
                        [
                            m
                            for m, s in zip(mixed, unit_scale.tolist())
                            if np.quantile(unit_scale, q / 4)
                            <= s
                            <= np.quantile(unit_scale, (q + 1) / 4)
                        ]
                    )
                )
                for q in range(4)
            },
        }
        reports.append(scene_rec)
        print(
            f"[ceiling] scene={scene} gt={len(id_list)} "
            f"unit={u_loss:.3f} gs={g_loss:.3f} "
            f"alpha_fg_iou={fg_alpha['iou']:.3f}",
            flush=True,
        )

    hash_after = _hash_model(model)
    report = {
        "resume": str(args.resume),
        "n_scenes": len(reports),
        "model_hash_before": hash_before,
        "model_hash_after": hash_after,
        "hash_unchanged": hash_before == hash_after,
        "scenes": reports,
    }
    out_path = out_dir / "audit_unit_gs_representation_ceiling.json"
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"[ceiling] wrote {out_path}")


if __name__ == "__main__":
    _main()
