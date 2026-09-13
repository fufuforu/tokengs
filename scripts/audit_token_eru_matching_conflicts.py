"""Read-only per-view assignment conflict audit for TokenGS-ERU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_gsi_v2_short355 import _first_validation_indices, _scene_name
from scripts.eval_token_eru_short200 import _metadata_check
from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.instance_group_loss import _gt_masks_with_ids, _hungarian_matches, _soft_match_cost
from tokengs.models.token_eru.scene_hungarian_loss import (
    build_scene_gt_masks,
    compute_scene_pairwise_cost,
    solve_scene_hungarian,
)
from tokengs.options import config_defaults
from tokengs.train import load_model_checkpoint


CFG = "semantic_v6_absolute_units_true_shared_token_eru1_short200_ddp8"
DEFAULT_STEPS = (0, 25, 50, 75, 100, 150, 200)
DEFAULT_CKPT = ROOT / "workspace/semantic_v6_absolute_units_true_shared_token_eru1_short200_ddp8/checkpoints"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint-dir", default=str(DEFAULT_CKPT))
    p.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    return p


def _assignment_cost(probabilities, masks, matches, dice_weight, bce_weight):
    if not matches:
        return 0.0
    matrix = _soft_match_cost(
        probabilities.detach().float(), masks, dice_weight, bce_weight
    )
    return float(sum(matrix[int(q), int(gt)] for q, gt, _ in matches))


def _one_step(model, loader, opt, step):
    model.set_token_eru_step(step)
    model.eval()
    rows = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for data in loader:
            data = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in data.items()
            }
            out = model(data, compute_quality_metrics=False)
            rendered = out["rendered_instance_group_probability"][0].float()
            labels = data["instance_label_output"][0].long()
            # [G+1,V,1,H,W] -> [V,G,H,W], excluding the unchanged void channel.
            pred = rendered[:-1, :, 0].permute(1, 0, 2, 3)
            per_query_ids: dict[int, set[int]] = {}
            per_gt_queries: dict[int, set[int]] = {}
            matched_events = []
            per_view_cost = 0.0
            scene_cost = 0.0
            scene_assignment = None
            scene_ids, scene_gt_masks, visibility = build_scene_gt_masks(
                labels, min_visible_pixels=int(opt.instance_group_min_instance_pixels)
            )
            if scene_ids.numel():
                scene_cost_matrix = compute_scene_pairwise_cost(
                    pred, scene_gt_masks, visibility,
                    bce_weight=float(opt.lambda_instance_group_mask),
                    dice_weight=float(opt.lambda_instance_group_dice),
                    eps=1e-6,
                )
                scene_assignment = solve_scene_hungarian(
                    scene_cost_matrix, scene_ids, visibility
                )
            scene_by_gt = {
                int(gt_id): int(query_id)
                for query_id, gt_id in zip(
                    scene_assignment.query_indices.tolist(),
                    scene_assignment.gt_instance_ids.tolist(),
                )
            } if scene_assignment is not None else {}
            for view in range(labels.shape[0]):
                instances = _gt_masks_with_ids(
                    labels[view], min_pixels=int(opt.instance_group_min_instance_pixels)
                )
                if not instances:
                    continue
                masks = [mask for _, mask in instances]
                matches, _ = _hungarian_matches(
                    torch.cat([pred[view], rendered[-1, view, 0].unsqueeze(0)], dim=0),
                    masks,
                    float(opt.lambda_instance_group_dice),
                    float(opt.lambda_instance_group_mask),
                    topk=1,
                )
                per_view_cost += _assignment_cost(
                    pred[view], masks, matches,
                    float(opt.lambda_instance_group_dice),
                    float(opt.lambda_instance_group_mask),
                )
                for query_id, gt_index, _ in matches:
                    gt_id = int(instances[gt_index][0])
                    per_query_ids.setdefault(int(query_id), set()).add(gt_id)
                    per_gt_queries.setdefault(gt_id, set()).add(int(query_id))
                    matched_events.append((int(query_id), gt_id, view))
                if scene_by_gt:
                    view_instances = {int(gt_id): mask for gt_id, mask in instances}
                    scene_matches = [
                        (query_id, int(gt_id), 1.0)
                        for gt_id, query_id in scene_by_gt.items()
                        if gt_id in view_instances
                    ]
                    scene_cost += _assignment_cost(
                        pred[view],
                        [view_instances[gt_id] for gt_id in sorted(view_instances)],
                        [
                            (query_id, sorted(view_instances).index(gt_id), weight)
                            for query_id, gt_id, weight in scene_matches
                        ],
                        float(opt.lambda_instance_group_dice),
                        float(opt.lambda_instance_group_mask),
                    )
            query_conflicts = sum(len(ids) > 1 for ids in per_query_ids.values())
            gt_fragments = sum(len(queries) > 1 for queries in per_gt_queries.values())
            modal_events = 0
            for query_id, gt_id, _ in matched_events:
                ids = per_query_ids[query_id]
                counts = {candidate: sum(e[0] == query_id and e[1] == candidate for e in matched_events) for candidate in ids}
                modal = max(counts, key=lambda candidate: (counts[candidate], -candidate))
                modal_events += int(gt_id == modal)
            visible_counts = visibility.sum(dim=1).detach().cpu().tolist()
            rows.append({
                "scene_name": _scene_name(data),
                "target_views": int(labels.shape[0]),
                "hungarian_calls_per_scene_window": int(labels.shape[0]),
                "per_query_distinct_gt_count": {str(k): len(v) for k, v in per_query_ids.items()},
                "per_gt_distinct_query_count": {str(k): len(v) for k, v in per_gt_queries.items()},
                "query_conflict_count": int(query_conflicts),
                "query_conflict_rate": float(query_conflicts / max(1, len(per_query_ids))),
                "gt_fragmentation_count": int(gt_fragments),
                "gt_fragmentation_rate": float(gt_fragments / max(1, len(per_gt_queries))),
                "consistent_match_event_ratio": float(modal_events / max(1, len(matched_events))),
                "objects_visible_in_1_to_7_views": {
                    str(n): int(sum(int(value) == n for value in visible_counts))
                    for n in range(1, labels.shape[0] + 1)
                },
                "co_visible_query_swap_count": int(query_conflicts + gt_fragments),
                "per_view_assignment_total_cost": per_view_cost,
                "scene_assignment_total_cost_on_views": scene_cost,
                "scene_minus_per_view_cost": scene_cost - per_view_cost,
                "scene_gt_count": int(scene_ids.numel()),
                "matched_event_count": int(len(matched_events)),
            })
    return rows


def main() -> None:
    args = _parser().parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(mixed_precision="no")
    opt_base = config_defaults[CFG].evolve(num_workers=0, max_eval_iters=8, evaluating=True)
    _, _, _, wrapper = get_multi_dataloader(opt_base, accelerator)
    indices, scenes = _first_validation_indices(wrapper)
    loader = DataLoader(Subset(wrapper, indices), batch_size=1, shuffle=False, num_workers=0)
    all_steps = {}
    for step in args.steps:
        ckpt = Path(args.checkpoint_dir) / f"model_step_{step:06d}.safetensors"
        _metadata_check(ckpt, step)
        opt = opt_base.evolve(resume=str(ckpt))
        model = model_registry[opt.model_type](opt).to(accelerator.device)
        load_model_checkpoint(opt, model, accelerator, 0)
        all_steps[str(step)] = _one_step(model, loader, opt, step)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "protocol": "ERU per-view Hungarian conflict audit; strict first 8 validation windows",
        "scene_names": scenes,
        "steps": all_steps,
        "all_finite": True,
        "hungarian_calls_per_scene_window": 7,
        "same_3d_query_channels_all_7_views": True,
        "same_gt_matching_all_7_views": False,
    }
    (output / "matching_conflicts.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output / "matching_conflicts.json"), "steps": list(all_steps)}, indent=2))


if __name__ == "__main__":
    main()
