"""Prepare-only evaluator entry point for query/metric-cluster comparisons.

The script intentionally requires an explicit ``--execute`` flag.  The
delivery scripts do not pass it, so preparing commands cannot accidentally
start formal evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("query", "metric_cluster"), default="query")
    parser.add_argument("--model", choices=("control", "treatment"), default="treatment")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.mode == "metric_cluster" and args.model != "treatment":
        raise ValueError("metric_cluster mode is only defined for the treatment")
    if not args.execute:
        commands = []
        for checkpoint in args.checkpoint:
            commands.append(
                {
                    "checkpoint": str(Path(checkpoint).resolve()),
                    "manifest": str(Path(args.manifest).resolve()),
                    "mode": args.mode,
                    "model": args.model,
                    "execute": True,
                }
            )
        out = Path(args.output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "prepared_commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
        print(json.dumps(commands, indent=2))
        return
    # Reuse the current paired ERU evaluator for query metrics.  Its AP,
    # mask conversion, and fingerprint protocol remain the single source of
    # truth.  This entry point only supplies the treatment config and manifest.
    import scripts.eval_token_eru_short200 as paired

    def metric_scene_instance(out, data):
        """Use the model's historical metric-cluster render output.

        The GT split, AP implementation, and mask-area rule remain those of
        the paired evaluator.  The formal GT-free cluster output includes an
        explicit final void channel.  ``masks_from_group_probs`` supplies the
        same argmax-mask and mean-confidence conversion used by the query
        evaluator.
        """
        cluster_output = out.get("metric_cluster_output")
        if cluster_output is None:
            raise RuntimeError(
                "metric_cluster mode requested but model returned no "
                "metric_cluster_output"
            )
        rendered = cluster_output.rendered_masks
        if rendered.ndim != 6 or rendered.shape[3] != 1:
            raise RuntimeError(
                "metric cluster masks must be [B,C,V,1,H,W], got "
                f"{tuple(rendered.shape)}"
            )
        probability = rendered[0].detach().float().cpu().numpy()[:, :, 0]
        labels = data["instance_label_output"][0].detach().long().cpu().numpy()
        predictions, scores, pred_ids = [], [], []
        ground_truth, gt_ids = [], []
        for view in range(probability.shape[1]):
            image_id = f"{paired._scene_name(data)}:target:{view}"
            masks, view_scores = paired.masks_from_group_probs(
                probability[:, view],
                void_channel=probability.shape[0] - 1,
                min_mask_area=1,
            )
            predictions.extend(masks)
            scores.extend(view_scores)
            pred_ids.extend([image_id] * len(masks))
            gt = paired.gt_masks_from_instance_map(labels[view], min_mask_area=1)
            ground_truth.extend(gt)
            gt_ids.extend([image_id] * len(gt))
        ap = paired.instance_ap(
            predictions,
            scores,
            ground_truth,
            thresholds=(0.25, 0.5, 0.75),
            vectorized=True,
            pred_image_ids=pred_ids,
            gt_image_ids=gt_ids,
        )
        diag = paired._mask_diagnostics(
            predictions, ground_truth, pred_ids, gt_ids
        )
        cluster_count = int(cluster_output.cluster_count[0])
        finite = all(
            np.isfinite(float(value))
            for value in (
                ap["ap_mean"],
                ap["ap_50"],
                diag["mean_best_gt_iou"],
            )
        )
        if not finite:
            raise FloatingPointError("non-finite metric-cluster evaluation output")
        return {
            "ap": float(ap["ap_mean"]),
            "ap25": float(ap["ap_25"]),
            "ap50": float(ap["ap_50"]),
            "ap75": float(ap["ap_75"]),
            "mean_best_gt_iou": float(diag["mean_best_gt_iou"]),
            "recall_iou25": float(diag["recall_iou25"]),
            "recall_iou50": float(diag["recall_iou50"]),
            "recall_iou75": float(diag["recall_iou75"]),
            "pred_gt": float(len(predictions) / max(1, len(ground_truth))),
            "prediction_count_nonempty": int(len(predictions)),
            "nonempty_query_count": cluster_count,
            "gt_count": int(len(ground_truth)),
            "active_query_count_mass_gt_0.001": cluster_count,
            "effective_query_count": float(cluster_count),
            "assignment_entropy": 0.0,
            "void_ratio": 0.0,
            "cluster_count": cluster_count,
            "predictions": predictions,
            "scores": scores,
            "prediction_ids": pred_ids,
            "ground_truth": ground_truth,
            "ground_truth_ids": gt_ids,
        }

    if args.mode == "metric_cluster":
        # Keep the established paired evaluator as the execution engine while
        # selecting the metric branch only through the resolved config.  No
        # second AP implementation or alternate dataloader is introduced.
        paired._scene_instance = metric_scene_instance

    paired.ERU_CFG = (
        "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
        "treatment_resume500_to700_ddp8"
        if args.model == "treatment"
        else "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
        "control_resume500_to700_ddp8"
    )
    original_cfg = paired.config_defaults[paired.ERU_CFG]
    if args.mode == "metric_cluster":
        paired.config_defaults[paired.ERU_CFG] = original_cfg.evolve(
            token_eru_dino_eval_mode="metric_cluster"
        )
    paired.MANIFEST = Path(args.manifest).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        for checkpoint in args.checkpoint:
            checkpoint = Path(checkpoint).resolve()
            step = int(checkpoint.stem.split("model_step_")[-1])
            output = output_root / f"{args.model}_{args.mode}_step_{step:06d}.json"
            if output.exists():
                raise RuntimeError(f"refusing to overwrite evaluation output: {output}")
            sys.argv = [
                "eval_token_eru_short200.py",
                "--checkpoint", str(checkpoint),
                "--output", str(output),
                "--optimizer-step", str(step),
                "--model", "eru",
                "--max-scenes", "8",
            ]
            paired.main()
    finally:
        paired.config_defaults[paired.ERU_CFG] = original_cfg


if __name__ == "__main__":
    main()
