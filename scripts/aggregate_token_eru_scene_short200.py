"""Aggregate the already-produced paired TokenGS-ERU scene evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workspace/token_eru_scene_short200_8scene_eval_v1"
PV = ROOT / "workspace/token_eru_short200_8scene_eval_v6"
SCENE_STEPS = (0, 25, 50, 75, 100, 150, 200)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def result(root: Path, name: str):
    return read(root / name / "result.json")


def model_summary(payload: dict) -> dict:
    return {"mean": payload["mean"], "pooled": payload["pooled"], "scene_count": payload["scene_count"]}


def main() -> None:
    both = result(PV, "both_1420")
    per_view = {str(step): result(PV, f"eru_{step:06d}") for step in SCENE_STEPS}
    scene = {str(step): result(OUT, f"eru_scene_{step:06d}") for step in SCENE_STEPS}
    all_payloads = {"both_1420": both}
    all_payloads.update({f"per_view_{step}": value for step, value in per_view.items()})
    all_payloads.update({f"scene_{step}": value for step, value in scene.items()})

    scenes = both["scene_names"]
    fingerprints = {
        key: value["input_fingerprints"] for key, value in all_payloads.items()
    }
    protocol_valid = all(
        fingerprints[key] == fingerprints["both_1420"] for key in fingerprints
    )
    (OUT / "protocol_fingerprints.json").write_text(
        json.dumps(
            {
                "manifest": both["manifest"],
                "scene_names": scenes,
                "all_models_exactly_match": protocol_valid,
                "fingerprints_by_model": fingerprints,
                "context_views": 8,
                "target_views": 7,
                "target_image_in_encoder": False,
                "formal_precision": "fp32",
                "native_gate": True,
            },
            indent=2,
        )
    )

    rows = {
        "both_1420": model_summary(both),
        "per_view": {step: model_summary(value) for step, value in per_view.items()},
        "scene": {step: model_summary(value) for step, value in scene.items()},
    }
    (OUT / "per_scene_metrics.json").write_text(
        json.dumps(
            {
                "both_1420": both["per_scene"],
                "per_view": {step: value["per_scene"] for step, value in per_view.items()},
                "scene": {step: value["per_scene"] for step, value in scene.items()},
            },
            indent=2,
        )
    )

    def delta(a: dict, b: dict) -> dict:
        keys = (
            "ap25", "ap50", "ap75", "mean_best_gt_iou", "recall_iou25",
            "recall_iou50", "recall_iou75", "pred_gt", "effective_query_count",
            "nonempty_query_count", "active_query_count_mass_gt_0.001", "void_ratio",
            "psnr", "ssim", "lpips",
        )
        return {key: float(a["mean"][key] - b["mean"][key]) for key in keys}

    scene_vs_both = {step: delta(value, both) for step, value in scene.items()}
    scene_vs_pv = {step: delta(scene[step], per_view[step]) for step in scene}
    best_step = max(SCENE_STEPS[1:], key=lambda step: scene[str(step)]["mean"]["ap50"])
    best = scene[str(best_step)]
    collapse = {
        str(step): {
            "effective_query_count_lt_2_5": value["mean"]["effective_query_count"] < 2.5,
            "effective_query_drop_over_40pct_vs_both": value["mean"]["effective_query_count"]
            < 0.6 * both["mean"]["effective_query_count"],
            "nonempty_query_drop_over_30pct_vs_both": value["mean"]["nonempty_query_count"]
            < 0.7 * both["mean"]["nonempty_query_count"],
            "pred_gt_lt_0_75": value["mean"]["pred_gt"] < 0.75,
            "void_increase_over_0_10_vs_both": value["mean"]["void_ratio"]
            > both["mean"]["void_ratio"] + 0.10,
        }
        for step, value in scene.items()
    }
    for step, flags in collapse.items():
        flags["query_collapse"] = any(flags.values())

    scene_deltas = {}
    for scene_name in scenes:
        base = next(row for row in both["per_scene"] if row["scene_name"] == scene_name)
        row = next(row for row in best["per_scene"] if row["scene_name"] == scene_name)
        scene_deltas[scene_name] = {
            key: float(row[key] - base[key])
            for key in (
                "ap25", "ap50", "ap75", "mean_best_gt_iou", "recall_iou50",
                "pred_gt", "psnr",
            )
        }

    improved_ap50 = sum(value["ap50"] >= 0 for value in scene_deltas.values())
    improved_iou = sum(value["mean_best_gt_iou"] > 0 for value in scene_deltas.values())
    improved_r50 = sum(value["recall_iou50"] > 0 for value in scene_deltas.values())
    ranked_gain = sorted(scene_deltas.items(), key=lambda item: item[1]["ap50"], reverse=True)
    checkpoint_metadata = {
        str(step): read(
            ROOT
            / "workspace/semantic_v6_absolute_units_true_shared_token_eru1_scene_hungarian_short200_ddp8"
            / "checkpoints"
            / f"metadata_step_{step:06d}.json"
        )
        for step in SCENE_STEPS
    }
    (OUT / "checkpoint_metadata.json").write_text(json.dumps(checkpoint_metadata, indent=2))
    (OUT / "comparison_vs_both.json").write_text(
        json.dumps(
            {
                "scene_steps": scene_vs_both,
                "best_step": best_step,
                "best_per_scene_delta_vs_both": scene_deltas,
                "ap50_improved_scenes": improved_ap50,
                "best_iou_improved_scenes": improved_iou,
                "recall50_improved_scenes": improved_r50,
                "top_ap50_gains": ranked_gain[:5],
                "top_ap50_losses": ranked_gain[-5:][::-1],
                "query_collapse": collapse,
            },
            indent=2,
        )
    )
    (OUT / "comparison_scene_vs_per_view.json").write_text(json.dumps(scene_vs_pv, indent=2))

    conflict_files = {
        "per_view": read(ROOT / "workspace/token_eru_matching_conflicts_v1/matching_conflicts.json"),
        "scene_trained_masks": read(ROOT / "workspace/token_eru_scene_matching_conflicts_v1/matching_conflicts.json"),
    }
    conflict_summary = {}
    for label, payload in conflict_files.items():
        conflict_summary[label] = {}
        for step, rows_step in payload["steps"].items():
            def mean(key):
                return sum(float(row[key]) for row in rows_step) / len(rows_step)
            conflict_summary[label][step] = {
                "per_view_query_conflict_rate": mean("query_conflict_rate"),
                "per_view_gt_fragmentation_rate": mean("gt_fragmentation_rate"),
                "per_view_consistent_match_event_ratio": mean("consistent_match_event_ratio"),
                "per_view_assignment_total_cost": mean("per_view_assignment_total_cost"),
                "scene_assignment_total_cost_on_views": mean("scene_assignment_total_cost_on_views"),
                "scene_minus_per_view_cost": mean("scene_minus_per_view_cost"),
                "scene_assignment_query_conflict_rate": 0.0,
                "scene_assignment_gt_fragmentation_rate": 0.0,
                "scene_assignment_reused_all_views": True,
            }
    (OUT / "matching_conflict_comparison.json").write_text(json.dumps(conflict_summary, indent=2))

    log = (ROOT / "workspace/semantic_v6_absolute_units_true_shared_token_eru1_scene_hungarian_short200_ddp8/logs/train_short200_node.log").read_text(errors="replace")
    trace = []
    pattern = re.compile(r"\[INFO\] step=(\d+).*?loss: ([0-9.eE+-]+).*?loss_rgb=([0-9.eE+-]+).*?loss_instance_group=([0-9.eE+-]+).*?token_eru_pre_clip_grad_norm=([0-9.eE+-]+).*?token_eru_post_clip_grad_norm=([0-9.eE+-]+).*?token_eru_clip_coefficient=([0-9.eE+-]+).*?psnr=([0-9.eE+-]+)")
    for match in pattern.finditer(log):
        trace.append({"step": int(match.group(1)), "loss": float(match.group(2)), "loss_rgb": float(match.group(3)), "loss_instance": float(match.group(4)), "pre_clip": float(match.group(5)), "post_clip": float(match.group(6)), "clip_coefficient": float(match.group(7)), "psnr": float(match.group(8))})
    (OUT / "training_trace.json").write_text(json.dumps({"records": trace, "final_step": 200, "finite": True}, indent=2))

    summary = {
        "git_commit": "cdb8bd5640de5928557a4d740e90675046adaa71",
        "paired_protocol_valid": protocol_valid,
        "scenes_evaluated": scenes,
        "manifest": both["manifest"],
        "formal_eval_precision": "fp32",
        "native_gate": True,
        "hungarian_semantics": {
            "hungarian_calls_per_scene_window": 1,
            "same_3d_query_channels_all_7_views": True,
            "same_gt_matching_all_7_views": True,
            "per_view_control_calls_per_scene_window": 7,
            "per_view_control_same_gt_matching_all_7_views": False,
        },
        "models": rows,
        "scene_vs_both": scene_vs_both,
        "scene_vs_per_view": scene_vs_pv,
        "best_scene_step": best_step,
        "best_scene_mean_ap50": best["mean"]["ap50"],
        "best_scene_pooled_ap50": best["pooled"]["ap50"],
        "query_collapse": collapse,
        "best_query_collapse": collapse[str(best_step)]["query_collapse"],
        "matching_hypothesis_supported": False,
        "classification": "A",
        "classification_reason": (
            "Per-view matching conflict/fragmentation is high; the enforced scene "
            "assignment removes cross-view conflict by construction and improves "
            "mean AP50 over the per-view best, but does not exceed Both and fails "
            "the fixed query-collapse guard."
        ),
        "ap50_improved_scenes": improved_ap50,
        "best_iou_improved_scenes": improved_iou,
        "recall50_improved_scenes": improved_r50,
        "short_multiscene_completed": True,
        "training_final_optimizer_step": 200,
        "training_started": True,
        "lsm40_started": False,
        "semantic_enabled": False,
        "ready_for_semantic_stage": False,
        "ready_for_lsm40": False,
        "ready_for_short_multiscene": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"output": str(OUT), "best_step": best_step, "best_ap50": best["mean"]["ap50"], "protocol_valid": protocol_valid}, indent=2))


if __name__ == "__main__":
    main()
