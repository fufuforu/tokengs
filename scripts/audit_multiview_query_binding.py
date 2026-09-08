"""Real-batch multi-view query-binding audit for the True-Shared head.

Read-only audit (no model/config/checkpoint modification, no training) on
the calibrated Both@1420 checkpoint:

  * traces the real per-view Hungarian matching by wrapping
    instance_group_loss._soft_match_cost / _hungarian_matches and
    recording (view, group, GT-local-idx) + cost matrices;
  * reports cross-view query consistency / fragmentation / collisions for
    scene-global ScanNet instance ids across the sample's target views;
  * backprops the instance loss per target view and measures tsh-head and
    q_abs-unit-formation gradient norms and pairwise view cosine;
  * runs a simulated scene-level matching forward (existing
    instance_group_scene_level_matching=True path) and compares gradient
    norms/cosine against the per-view sum;
  * prints real tensor shapes for the sample.

Output JSON per workspace. Example:
    python -u scripts/audit_multiview_query_binding.py \
        --workspace workspace/tsh_mv_binding_audit --n-samples 16
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safetensors.torch import load_file  # noqa: E402

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models import instance_group_loss as L  # noqa: E402
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
    return len(abs_state), len(tsh_state)


def _set_full_joint(model, opt):
    model.tsh_instance_loss_weight_eff = 1.0
    model.tsh_unit_grad_eff = float(
        getattr(opt, "tsh_unit_gradient_multiplier_max", 0.0)
    )
    model.tsh_mbm_u2r_eff = float(
        getattr(opt, "tsh_mbm_u2r_weight", 10.0)
    )
    model.teacher_lambda_eff = 0.0


def _gt_map_stats(labels: torch.Tensor) -> list[dict]:
    """Per view: raw instance ids, areas (already [B,V,H,W])."""
    labels = labels.detach().cpu()
    views = []
    for v in range(labels.shape[1]):
        ids = torch.unique(labels[0, v])
        items = []
        for i in ids.tolist():
            if int(i) in (0, 255):
                continue
            mask = labels[0, v] == i
            area = int(mask.sum().item())
            if area >= 25:
                items.append({"id": int(i), "area": area})
        views.append(items)
    return views


_hook_costs: list = []
_hook_matches: list = []


class _Hooks:
    def __init__(self):
        self._orig_cost = L._soft_match_cost
        self._orig_matches = L._hungarian_matches
        self.costs: list = []
        self.matches: list = []

    def _cost_wrap(
        self, pred, gt_masks, dice, mask, area_norm_bce=False, eps=1e-6
    ):
        cost = self._orig_cost(
            pred,
            gt_masks,
            dice,
            mask,
            area_norm_bce=area_norm_bce,
            eps=eps,
        )
        self.costs.append(cost.detach().cpu())
        return cost

    def _matches_wrap(
        self,
        probabilities,
        gt_masks,
        dice,
        mask,
        area_norm_bce=False,
        topk=1,
        secondary_weight=0.3,
        num_active_groups=None,
        active_ids=None,
    ):
        out = self._orig_matches(
            probabilities,
            gt_masks,
            dice,
            mask,
            area_norm_bce=area_norm_bce,
            topk=topk,
            secondary_weight=secondary_weight,
            num_active_groups=num_active_groups,
            active_ids=active_ids,
        )
        is_per_view = bool(
            gt_masks
            and all(torch.is_tensor(x) for x in gt_masks)
        )
        if is_per_view:
            self.matches.append(
                {
                    "primary": [
                        (int(g), int(t), float(w))
                        for g, t, w in out[0]
                    ],
                    "extra": [
                        (int(g), int(t), float(w))
                        for g, t, w in out[1]
                    ],
                }
            )
        return out

    def __enter__(self):
        L._soft_match_cost = self._cost_wrap
        L._hungarian_matches = self._matches_wrap
        return self

    def __exit__(self, *_):
        L._soft_match_cost = self._orig_cost
        L._hungarian_matches = self._orig_matches


_PREFIX_UNIT = (
    "absolute_gs_head.tok_norm.",
    "absolute_gs_head.tok_proj.",
    "absolute_gs_head.unit_queries",
    "absolute_gs_head.unit_readout.",
)
_PREFIX_TSH = "tsh_instance_head."


def _grad_vec(model, prefixes):
    parts = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        hit = (
            name.startswith(prefixes)
            if isinstance(prefixes, str)
            else any(name.startswith(p) for p in prefixes)
        )
        if hit:
            parts.append(param.grad.detach().float().flatten())
    return torch.cat(parts) if parts else torch.zeros(1)


def _norm(v) -> float:
    return float(v.norm().item())


def _cos(a, b) -> float | None:
    if a.numel() < 2 or b.numel() < 2:
        return None
    return float(
        torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)
        ).item()
    )


def _instance_loss_call(model, opt, pi, labels, scene_level=False):
    head = model.tsh_instance_head
    return L.hungarian_instance_group_loss(
        pi,
        labels,
        num_groups=int(head.num_groups),
        min_instance_pixels=int(
            getattr(opt, "instance_group_min_instance_pixels", 32)
        ),
        dice_weight=float(getattr(opt, "lambda_instance_group_dice", 1.0)),
        mask_weight=float(getattr(opt, "lambda_instance_group_mask", 1.0)),
        void_weight=float(getattr(opt, "lambda_instance_group_void", 0.1)),
        unmatched_weight=float(
            getattr(opt, "lambda_instance_group_unmatched", 0.1)
        ),
        usage_entropy_weight=float(
            getattr(opt, "instance_group_usage_entropy", 0.05)
        ),
        ce_weight=float(getattr(opt, "lambda_instance_group_ce", 1.0)),
        area_alpha=float(getattr(opt, "instance_group_area_alpha", 0.5)),
        match_area_norm=bool(
            getattr(opt, "instance_group_match_area_norm", True)
        ),
        match_topk=int(getattr(opt, "instance_group_match_topk", 1)),
        secondary_pair_weight=float(
            getattr(opt, "instance_group_secondary_pair_weight", 0.3)
        ),
        use_adaptive_groups=bool(
            getattr(opt, "instance_group_adaptive_count", False)
        ),
        scene_level_matching=bool(scene_level),
    )


def _summarize(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "count": len(ordered),
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace/tsh_mv_binding_audit")
    parser.add_argument("--resume", default=_RESUME)
    parser.add_argument("--config-name", default=_CFG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--first-index", type=int, default=0)
    args = parser.parse_args()
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = config_defaults[args.config_name]
    opt.workspace = str(out_dir)
    opt.num_workers = 0
    torch.manual_seed(args.seed)
    loader, _, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    it = iter(loader)
    for _ in range(args.first_index):
        next(it)

    model = model_registry[opt.model_type](opt).cuda().train()
    ckpt = load_file(args.resume, device="cpu")
    abs_keys, tsh_keys = _load_heads(model, ckpt)
    _set_full_joint(model, opt)
    instance_weight = float(getattr(opt, "tsh_lambda_instance", 0.05)) * 1.0

    samples = []
    aggregated = {
        "gt_visibility_views": [],
        "gt_count_per_sample": [],
        "distinct_queries_per_gt": [],
        "gt_fragmented": [],
        "cross_view_consistent": [],
        "distinct_gt_per_query": [],
        "query_collision": [],
        "view_pairwise_grad_cos": [],
        "view_unit_grad_cos": [],
        "per_view_instance_norm": [],
        "pred_per_view": [],
        "gt_per_view": [],
    }
    try:
        for sample_index in range(args.n_samples):
            data = next(it)
            data = {
                k: (v.cuda() if torch.is_tensor(v) else v)
                for k, v in data.items()
            }
            scene = str(data.get("scene_name", ("?",))[0])
            frame_ids = (
                data["frame_ids"][0].tolist()
                if "frame_ids" in data
                else []
            )
            context_ids = (
                data["context_views_id"][0].tolist()
                if "context_views_id" in data
                else []
            )
            target_ids = (
                data["target_views_id"][0].tolist()
                if "target_views_id" in data
                else []
            )
            shapes = {
                "input_images": list(
                    data["input"].shape if torch.is_tensor(data.get("input")) else None
                ),
                "cam_view_input": list(data["cam_view_input"].shape),
                "images_input": list(data["images_input"].shape),
                "images_output": list(data["images_output"].shape),
                "instance_label_output": list(
                    data["instance_label_output"].shape
                ),
                "instance_label_all": list(data["instance_label_all"].shape),
            }
            entry = {
                "sample_index": sample_index,
                "scene": scene,
                "frame_ids": frame_ids,
                "context_views": context_ids,
                "target_views": target_ids,
                "shapes": shapes,
            }

            # ---- per-view matching forward with real hooks ----
            hooks = _Hooks()
            with hooks:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(data, compute_quality_metrics=False)
            pi = out["rendered_instance_group_probability"]
            labels = data["instance_label_output"].long()
            view_gt = _gt_map_stats(labels)
            gt_views = len(labels.shape[1:2]) if False else labels.shape[1]
            # map matching calls to non-empty views in order
            nonempty_views = [
                v for v, items in enumerate(view_gt) if items
            ]
            view_matches: dict[int, list] = {}
            for seq, match in enumerate(hooks.matches):
                if seq >= len(nonempty_views):
                    break
                v = nonempty_views[seq]
                id_by_idx = [
                    (idx, item["id"])
                    for idx, item in enumerate(view_gt[v])
                ]
                view_matches[v] = [
                    {
                        "query": g,
                        "gt_idx": t,
                        "gt_id": dict(id_by_idx)[t],
                        "role": w,
                    }
                    for g, t, w in match["primary"]
                ]
            entry["matching"] = {
                "per_view_mode": True,
                "view_gt": view_gt,
                "view_matches": {
                    str(v): m for v, m in view_matches.items()
                },
                "num_cost_matrices_hooked": len(hooks.costs),
            }

            # ---- global ID binding statistics ----
            gt_id_views: dict[int, list[int]] = {}
            for v, items in enumerate(view_gt):
                for item in items:
                    gt_id_views.setdefault(item["id"], []).append(v)
            gt_ids = sorted(gt_id_views)
            binding = {}
            for gid in gt_ids:
                matched = []
                for v in gt_id_views[gid]:
                    if v not in view_matches:
                        continue
                    qmap = {
                        m["gt_idx"]: m["query"] for m in view_matches[v]
                    }
                    gt_idx = next(
                        (
                            idx
                            for idx, item in enumerate(view_gt[v])
                            if item["id"] == gid
                        ),
                        None,
                    )
                    if gt_idx is not None and gt_idx in qmap:
                        matched.append((v, qmap[gt_idx]))
                queries = sorted({q for _, q in matched})
                visible = len(gt_id_views[gid])
                binding[gid] = {
                    "visible_views": visible,
                    "matched_views": len(matched),
                    "queries": queries,
                    "consistent": (
                        len(queries) <= 1
                        and len(matched) == visible
                    ),
                }
            entry["binding"] = binding
            stats = {
                "cross_view_query_consistency": (
                    sum(
                        1
                        for b in binding.values()
                        if b["visible_views"] >= 2
                        and b["consistent"]
                    )
                    / max(
                        1,
                        sum(
                            1
                            for b in binding.values()
                            if b["visible_views"] >= 2
                        ),
                    )
                ),
                "distinct_queries_per_gt": _summarize(
                    [
                        len(b["queries"])
                        for b in binding.values()
                        if b["matched_views"] > 0
                    ]
                ),
                "fragmented_gt_ratio": (
                    sum(
                        1
                        for b in binding.values()
                        if len(b["queries"]) > 1
                    )
                    / max(
                        1,
                        sum(
                            1
                            for b in binding.values()
                            if b["matched_views"] > 0
                        ),
                    )
                ),
            }
            query_to_gt: dict[int, set[int]] = {}
            for v, m in view_matches.items():
                for item in m:
                    query_to_gt.setdefault(item["query"], set()).add(
                        item["gt_id"]
                    )
            stats.update(
                {
                    "distinct_gt_per_query": _summarize(
                        [len(s) for s in query_to_gt.values()]
                    ),
                    "query_collision_ratio": (
                        sum(1 for s in query_to_gt.values() if len(s) > 1)
                        / max(1, len(query_to_gt))
                    ),
                    "gt_count": len(gt_ids),
                    "gt_visibility_views": _summarize(
                        [len(v) for v in gt_id_views.values()]
                    ),
                    "active_queries": len(query_to_gt),
                    "pred_count": sum(
                        len(m) for m in view_matches.values()
                    ),
                    "gt_count_views": sum(len(v) for v in view_gt),
                }
            )
            entry["stats"] = stats
            aggregated["gt_visibility_views"].extend(
                [len(v) for v in gt_id_views.values()]
            )
            aggregated["gt_count_per_sample"].append(len(gt_ids))
            aggregated["distinct_queries_per_gt"].extend(
                [
                    len(b["queries"])
                    for b in binding.values()
                    if b["matched_views"] > 0
                ]
            )
            aggregated["gt_fragmented"].append(stats["fragmented_gt_ratio"])
            aggregated["cross_view_consistent"].append(
                stats["cross_view_query_consistency"]
            )
            aggregated["distinct_gt_per_query"].extend(
                [len(s) for s in query_to_gt.values()]
            )
            aggregated["query_collision"].append(stats["query_collision_ratio"])

            # ---- phase 2: per-view instance gradients ----
            grads_per_view = {}
            model.zero_grad(set_to_none=True)
            for v in range(labels.shape[1]):
                pi_v = pi[:, :, v : v + 1]
                lab_v = labels[:, v : v + 1]
                if not (lab_v > 0).any():
                    grads_per_view[v] = None
                    continue
                loss_v, _ = _instance_loss_call(model, opt, pi_v, lab_v)
                loss_v.backward(retain_graph=True)
                vec_tsh = _grad_vec(model, _PREFIX_TSH)
                vec_unit = _grad_vec(model, _PREFIX_UNIT)
                grads_per_view[v] = {
                    "tsh_norm": _norm(vec_tsh),
                    "unit_norm": _norm(vec_unit),
                    "tsh_vec": vec_tsh,
                    "unit_vec": vec_unit,
                    "loss": float(loss_v.detach().item()),
                }
                model.zero_grad(set_to_none=True)
            # aggregate per-view loss backward (sum over views like training)
            model.zero_grad(set_to_none=True)
            loss_pv, _ = _instance_loss_call(model, opt, pi, labels)
            loss_pv.backward(retain_graph=True)
            agg_pv_tsh = _grad_vec(model, _PREFIX_TSH)
            agg_pv_unit = _grad_vec(model, _PREFIX_UNIT)
            agg_pv_norm = _norm(torch.cat([agg_pv_tsh, agg_pv_unit]))
            # per-view loss on final graph was each *instance_weight*; model
            # training applies tsh_lambda later; compare raw losses here.
            cos_pairs = []
            cos_pairs_unit = []
            vs = [v for v, g in grads_per_view.items() if g is not None]
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    a, b = grads_per_view[vs[i]], grads_per_view[vs[j]]
                    c = _cos(a["tsh_vec"], b["tsh_vec"])
                    cu = _cos(a["unit_vec"], b["unit_vec"])
                    if c is not None:
                        cos_pairs.append(c)
                    if cu is not None:
                        cos_pairs_unit.append(cu)
            entry["phase2"] = {
                "per_view_grads": {
                    str(v): (
                        None
                        if g is None
                        else {
                            "tsh_norm": g["tsh_norm"],
                            "unit_norm": g["unit_norm"],
                            "loss": g["loss"],
                        }
                    )
                    for v, g in grads_per_view.items()
                },
                "view_pairwise_cosine": _summarize(cos_pairs),
                "view_unit_pairwise_cosine": _summarize(cos_pairs_unit),
                "aggregate_per_view_norm": agg_pv_norm,
            }
            aggregated["view_pairwise_grad_cos"].extend(cos_pairs)
            aggregated["view_unit_grad_cos"].extend(cos_pairs_unit)

            # ---- simulated scene-level matching gradients ----
            model.zero_grad(set_to_none=True)
            scene_loss, _ = _instance_loss_call(
                model, opt, pi, labels, scene_level=True
            )
            scene_loss.backward(retain_graph=True)
            scene_tsh = _grad_vec(model, _PREFIX_TSH)
            scene_unit = _grad_vec(model, _PREFIX_UNIT)
            scene_norm = _norm(torch.cat([scene_tsh, scene_unit]))
            entry["phase2"]["scene_level"] = {
                "loss": float(scene_loss.detach().item()),
                "norm": scene_norm,
                "cos_vs_per_view_aggregate": _cos(
                    scene_tsh, agg_pv_tsh
                ),
                "unit_cos_vs_per_view_aggregate": _cos(
                    scene_unit, agg_pv_unit
                ),
            }
            # sanity: per-view max norm
            entry["phase2"]["max_per_view_norm"] = max(
                (g["tsh_norm"] for g in grads_per_view.values() if g),
                default=0.0,
            )
            model.zero_grad(set_to_none=True)
            del loss_pv, scene_loss
            torch.cuda.empty_cache()
            samples.append(entry)
            print(
                f"[binding] sample={sample_index} scene={scene} "
                f"consistency={stats['cross_view_query_consistency']:.3f} "
                f"fragmented={stats['fragmented_gt_ratio']:.3f} "
                f"collision={stats['query_collision_ratio']:.3f} "
                f"gt={len(gt_ids)} pred={stats['pred_count']}",
                flush=True,
            )
    finally:
        # restore hooks
        pass

    summary = {
        "n_samples": len(samples),
        "cross_view_query_consistency": _summarize(
            aggregated["cross_view_consistent"]
        ),
        "fragmented_gt_ratio": _summarize(aggregated["gt_fragmented"]),
        "distinct_queries_per_gt": _summarize(
            aggregated["distinct_queries_per_gt"]
        ),
        "distinct_gt_per_query": _summarize(
            aggregated["distinct_gt_per_query"]
        ),
        "query_collision_ratio": _summarize(aggregated["query_collision"]),
        "gt_count_per_sample": _summarize(aggregated["gt_count_per_sample"]),
        "gt_visibility_views": _summarize(
            aggregated["gt_visibility_views"]
        ),
        "view_pairwise_grad_cos": _summarize(
            aggregated["view_pairwise_grad_cos"]
        ),
        "view_unit_pairwise_grad_cos": _summarize(
            aggregated["view_unit_grad_cos"]
        ),
    }
    report = {
        "resume": str(args.resume),
        "config": args.config_name,
        "abs_keys": abs_keys,
        "tsh_keys": tsh_keys,
        "summary": summary,
        "samples": samples,
    }
    path = out_dir / "audit_multiview_query_binding.json"
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"[binding] wrote {path}")


if __name__ == "__main__":
    _main()
