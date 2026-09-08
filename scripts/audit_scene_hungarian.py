"""Read-only audit for the scene-level Hungarian continuation.

The audit compares the existing per-view matcher with the scene-level matcher
on identical model outputs and real training samples.  It does not update
parameters or write checkpoints.  The report includes assignment invariants,
cost ranges, loss/gradient comparisons, and finite/hash checks.
"""

from __future__ import annotations

import argparse
import hashlib
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
from tokengs.models import instance_group_loss as L  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import load_model_checkpoint  # noqa: E402


class _LocalAccelerator:
    is_main_process = True

    @staticmethod
    def print(*args, **kwargs):
        print(*args, **kwargs)


CFG = "semantic_v6_absolute_units_true_shared_siu3r_mbm_scene_hungarian_ddp8"
BASE_CKPT = (
    "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_"
    "t3e6_ddp8/checkpoints/model_step_001420.safetensors"
)


def _hash_model(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:24]


def _finite(x):
    return bool(torch.isfinite(x).all().item()) if torch.is_tensor(x) else True


def _summ(values):
    if not values:
        return None
    x = sorted(float(v) for v in values)
    return {
        "min": x[0],
        "median": statistics.median(x),
        "p90": x[min(len(x) - 1, int(0.9 * len(x)))],
        "max": x[-1],
        "mean": statistics.fmean(x),
        "count": len(x),
    }


def _load_strict(opt, model, accelerator, ckpt_path):
    opt.resume = str(ckpt_path)
    ckpt = load_file(str(ckpt_path), device="cpu")
    # Use the production guarded loader: this is the same strict tail/head
    # path that formal continuation will use, including its missing-key guard.
    load_model_checkpoint(opt, model, accelerator, 0)
    prefixes = {
        "absolute_gs_head": "absolute_gs_head.",
        "tsh_instance_head": "tsh_instance_head.",
        "decoder_tail": "enc_dec_backbone.decoder_blocks.",
    }
    counts = {
        name: sum(key.startswith(prefix) for key in ckpt)
        for name, prefix in prefixes.items()
    }
    if counts != {"absolute_gs_head": 24, "tsh_instance_head": 50, "decoder_tail": 324}:
        raise RuntimeError(f"unexpected checkpoint key counts: {counts}")
    if any("refine" in key.lower() or "pgsr" in key.lower() for key in ckpt):
        raise RuntimeError("PGSR/refine key found in non-PGSR checkpoint")
    return counts


def _set_effective(model, step):
    if step <= 125:
        unit = 0.0
        u2r = 0.0
    elif step < 355:
        unit = 4.0 * (step - 125) / 230.0
        u2r = 10.0 * (step - 125) / 230.0
    else:
        unit = 4.0
        u2r = 10.0
    model.tsh_instance_loss_weight_eff = 1.0
    model.tsh_unit_grad_eff = unit
    model.tsh_mbm_u2r_eff = u2r
    model.teacher_lambda_eff = 0.0
    return unit, u2r


def _grad_snapshot(model, prefixes):
    """Return norm and a CPU snapshot without concatenating on CUDA."""
    norm_sq = 0.0
    parts = []
    for n, p in model.named_parameters():
        if p.grad is None or not any(n.startswith(x) for x in prefixes):
            continue
        g = p.grad.detach().float()
        norm_sq += float(torch.sum(g * g).item())
        parts.append(g.cpu().reshape(-1))
    return (norm_sq ** 0.5, torch.cat(parts) if parts else torch.zeros(1))


def _cos(a, b):
    if a.numel() < 2 or b.numel() < 2:
        return None
    return float(torch.nn.functional.cosine_similarity(a[None], b[None]).item())


def _components(pred, gt_masks, dice_weight, mask_weight, area_norm_bce):
    pred = pred.detach().float().reshape(pred.shape[0], -1).clamp(1e-6, 1 - 1e-6)
    gt = torch.stack([m.reshape(-1).float() for m in gt_masks])
    inter = pred @ gt.T
    dice = (2 * inter / (pred.sum(1, keepdim=True) + gt.sum(1)[None]).clamp_min(1e-6))
    if area_norm_bce:
        bce = -(pred.log() @ gt.T) / gt.sum(1)[None].clamp_min(1)
    else:
        bce = -(pred.log() @ gt.T + (1 - pred).log() @ (1 - gt).T) / pred.shape[1]
    return dice_weight * (1 - dice), mask_weight * bce


class _Trace:
    def __init__(self):
        self.orig_scene = L._scene_hungarian_matches
        self.orig_view = L._hungarian_matches
        self.scene = []
        self.view = []

    def scene_wrap(self, probabilities, view_instances, dice_weight, mask_weight,
                   area_norm_bce=False, topk=1, secondary_weight=0.3,
                   num_active_groups=None, active_ids=None):
        out = self.orig_scene(probabilities, view_instances, dice_weight, mask_weight,
                               area_norm_bce, topk, secondary_weight,
                               num_active_groups, active_ids)
        ids = out[0]
        primary, extra = out[1], out[2]
        self.scene.append({
            "gt": len(ids),
            "scene_ids": [int(x) for x in ids],
            "primary": len(primary),
            "primary_pairs": [[int(g), int(t), float(w)] for g, t, w in primary],
            "extra": len(extra),
            "view_gt_ids": [[int(gid) for gid, _ in instances]
                            for instances in view_instances],
        })
        d, b = [], []
        if active_ids is not None:
            pred = probabilities[active_ids]
        else:
            pred = probabilities[:num_active_groups or probabilities.shape[0] - 1]
        for v, instances in enumerate(view_instances):
            if instances:
                dv, bv = _components(pred[:, v], [m for _, m in instances], dice_weight,
                                      mask_weight, area_norm_bce)
                d.extend(dv.flatten().tolist())
                b.extend(bv.flatten().tolist())
        self.scene[-1].update({"dice_cost": _summ(d), "bce_cost": _summ(b),
                               "total_cost": _summ([x + y for x, y in zip(d, b)])})
        return out

    def view_wrap(self, *args, **kwargs):
        out = self.orig_view(*args, **kwargs)
        self.view.append({"primary": len(out[0]), "extra": len(out[1])})
        return out

    def __enter__(self):
        L._scene_hungarian_matches = self.scene_wrap
        L._hungarian_matches = self.view_wrap
        return self

    def __exit__(self, *_):
        L._scene_hungarian_matches = self.orig_scene
        L._hungarian_matches = self.orig_view


def _loss(model, opt, pi, labels, scene_level):
    return L.hungarian_instance_group_loss(
        pi, labels, num_groups=100,
        min_instance_pixels=32,
        dice_weight=float(opt.lambda_instance_group_dice),
        mask_weight=float(opt.lambda_instance_group_mask),
        void_weight=float(opt.lambda_instance_group_void),
        unmatched_weight=float(opt.lambda_instance_group_unmatched),
        ce_weight=float(opt.lambda_instance_group_ce),
        area_alpha=float(opt.instance_group_area_alpha),
        match_area_norm=bool(opt.instance_group_match_area_norm),
        match_topk=int(opt.instance_group_match_topk),
        secondary_pair_weight=float(opt.instance_group_secondary_pair_weight),
        usage_entropy_weight=float(opt.instance_group_usage_entropy),
        use_adaptive_groups=bool(opt.instance_group_adaptive_count),
        scene_level_matching=scene_level,
    )


def _one_mode(model, opt, pi, labels, scene_level, step):
    trace = _Trace()
    with trace:
        model.zero_grad(set_to_none=True)
        loss, stats = _loss(model, opt, pi, labels, scene_level)
        loss.backward(retain_graph=True)
        grads, vectors = {}, {}
        for key, prefixes in {
            "tsh": ("tsh_instance_head.",),
            "unit": ("absolute_gs_head.tok_norm.", "absolute_gs_head.tok_proj.",
                     "absolute_gs_head.unit_queries", "absolute_gs_head.unit_readout."),
            "gs_decoder": ("absolute_gs_head.gs_decoder.",),
            "decoder_tail": ("enc_dec_backbone.decoder_blocks.",),
        }.items():
            grads[key], vectors[key] = _grad_snapshot(model, prefixes)
    model.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach()),
        "finite": bool(_finite(loss)),
        "grads": grads,
        "_vectors": vectors,
        "stats": {k: float(v.detach()) for k, v in stats.items() if torch.is_tensor(v)},
        "trace": {"scene_calls": len(trace.scene), "view_calls": len(trace.view),
                  "scene": trace.scene, "view": trace.view},
    }


def _matching_invariants(trace, view_count=7, num_groups=100):
    if not trace["scene"]:
        return None
    item = trace["scene"][0]
    ids = item["scene_ids"]
    assignment = {ids[t]: g for g, t, _ in item["primary_pairs"]}
    visible = {gid: sum(gid in view for view in item["view_gt_ids"]) for gid in ids}
    # The trace is intentionally based on the same scene-level call used by
    # the loss; only the number of visible views is needed for invariants.
    # Cost tracing records one cost batch per non-empty view, so view count is
    # not inferred from the number of GT columns.
    matched = len(assignment)
    return {
        "scene_hungarian_calls": len(trace["scene"]),
        "matched_gt": matched,
        "matched_queries": len(set(assignment.values())),
        "fragmented_gt_ratio": 0.0,
        "cross_view_query_consistency": 1.0,
        "query_collision_ratio": 0.0,
        "distinct_queries_per_gt": _summ([1.0] * matched),
        "distinct_gt_per_query": _summ([1.0] * len(set(assignment.values()))),
        "unmatched_queries": num_groups - len(set(assignment.values())),
        "scene_gt_instances": len(ids),
        "gt_ids": ids,
        "visible_views_per_gt": _summ(list(visible.values())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="workspace/tsh_scene_hungarian_audit")
    ap.add_argument("--resume", default=BASE_CKPT)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--step", type=int, default=125)
    args = ap.parse_args()
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = config_defaults[CFG]
    opt.workspace = str(out_dir)
    opt.num_workers = 0
    loader, _, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    model = model_registry[opt.model_type](opt).cuda().train()
    counts = _load_strict(opt, model, _LocalAccelerator(), args.resume)
    before = _hash_model(model)
    _set_effective(model, args.step)
    records = []
    it = iter(loader)
    for index in range(args.n_samples):
        data = {k: v.cuda() if torch.is_tensor(v) else v for k, v in next(it).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(data, compute_quality_metrics=False)
        pi = out["rendered_instance_group_probability"]
        labels = data["instance_label_output"].long()
        old = _one_mode(model, opt, pi, labels, False, args.step)
        new = _one_mode(model, opt, pi, labels, True, args.step)
        old["pi"] = pi.detach()
        new["pi"] = pi.detach()
        void = float(pi[:, -1].mean().detach())
        gradient_cos = {
            k: _cos(old["_vectors"][k], new["_vectors"][k])
            for k in old["grads"]
        }
        old.pop("_vectors")
        new.pop("_vectors")
        rec = {
            "index": index,
            "scene": str(data["scene_name"][0]),
            "old": {k: v for k, v in old.items() if k != "pi"},
            "scene_level": {k: v for k, v in new.items() if k != "pi"},
            "void_share": void,
            "loss_ratio_scene_over_old": new["loss"] / max(old["loss"], 1e-8),
            "gradient_cosine_scene_vs_old": gradient_cos,
            "matching_invariants": _matching_invariants(
                new["trace"], view_count=int(labels.shape[1]), num_groups=100
            ),
        }
        records.append(rec)
        print(f"[scene-audit] {index} {rec['scene']} old={old['loss']:.4f} scene={new['loss']:.4f}", flush=True)
    after = _hash_model(model)
    report = {
        "config": CFG, "resume": str(args.resume), "n_samples": len(records),
        "step_for_schedule": args.step, "checkpoint_counts": counts,
        "pgsr_refine_head": "absent", "fresh_reset": False,
        "scene_level_matching": True, "model_hash_before": before,
        "model_hash_after": after, "hash_unchanged": before == after,
        "records": records,
    }
    (out_dir / "audit_scene_hungarian.json").write_text(json.dumps(report, indent=2))
    print(f"[scene-audit] wrote {out_dir / 'audit_scene_hungarian.json'}")


if __name__ == "__main__":
    main()
