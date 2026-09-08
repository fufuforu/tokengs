"""Offline ranking/mask decomposition for eval_instance_lsm_protocol caches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokengs.utils.instance_ap import instance_ap


def _load_scene(path: Path) -> dict:
    records = json.loads((path / "records.json").read_text())
    masks = np.load(path / "masks.npz")
    return {"records": records, "pred_masks": masks["pred_masks"].astype(bool),
            "gt_masks": masks["gt_masks"].astype(bool)}


def _ious(preds: list[np.ndarray], gts: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(preds), len(gts)), dtype=np.float32)
    for i, pred in enumerate(preds):
        pa = float(pred.sum())
        for j, gt in enumerate(gts):
            inter = float(np.logical_and(pred, gt).sum())
            union = pa + float(gt.sum()) - inter
            out[i, j] = inter / union if union > 0 else 0.0
    return out


def _match(order: list[int], matrix: np.ndarray, threshold: float) -> dict:
    unmatched = set(range(matrix.shape[1]))
    tp = []
    fp = []
    matched = {}
    for pred_idx in order:
        if not unmatched:
            fp.append(int(pred_idx))
            continue
        candidates = list(unmatched)
        gt_idx = max(candidates, key=lambda j: float(matrix[pred_idx, j]))
        if float(matrix[pred_idx, gt_idx]) >= threshold:
            tp.append(int(pred_idx))
            matched[int(pred_idx)] = int(gt_idx)
            unmatched.remove(gt_idx)
        else:
            fp.append(int(pred_idx))
    return {"tp": tp, "fp": fp, "matched": matched,
            "unmatched_gt": [int(i) for i in sorted(unmatched)]}


def _rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(_rank_values(x), _rank_values(y))[0, 1])


def _kendall(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt(
        (concordant + discordant + ties_x)
        * (concordant + discordant + ties_y)
    )
    return float((concordant - discordant) / denom) if denom else None


def _ap(pred_masks, scores, gt_masks, pred_ids, gt_ids) -> dict:
    return instance_ap(
        pred_masks, scores, gt_masks, thresholds=(0.25, 0.5, 0.75),
        vectorized=False, pred_image_ids=pred_ids, gt_image_ids=gt_ids,
    )


def _hist(values: np.ndarray) -> dict[str, int]:
    edges = [0.0, 0.25, 0.40, 0.45, 0.50, 0.55, 0.75, float("inf")]
    names = ["<0.25", "0.25-0.40", "0.40-0.45", "0.45-0.50",
             "0.50-0.55", "0.55-0.75", ">=0.75"]
    counts = {}
    for i, name in enumerate(names):
        if i == 0:
            mask = values < edges[1]
        elif i == len(names) - 1:
            mask = values >= edges[-2]
        else:
            mask = (values >= edges[i]) & (values < edges[i + 1])
        counts[name] = int(mask.sum())
    return counts


def _scene_audit(scene_path: Path) -> tuple[dict, list[dict], list[dict]]:
    data = _load_scene(scene_path)
    rec = data["records"]
    pred_records = [p for p in rec["predictions"] if p["kept_by_evaluator"]]
    all_records = rec["predictions"]
    gt_records = rec["ground_truth"]
    pred_masks = [data["pred_masks"][p["mask_index"]] for p in pred_records]
    gt_masks = [data["gt_masks"][g["mask_index"]] for g in gt_records]
    pred_ids = [p["image_id"] for p in pred_records]
    gt_ids = [g["image_id"] for g in gt_records]
    matrix = _ious(pred_masks, gt_masks)
    max_iou = matrix.max(axis=1) if len(pred_records) and len(gt_records) else np.zeros(len(pred_records))
    best_gt = matrix.max(axis=0) if len(gt_records) and len(pred_records) else np.zeros(len(gt_records))
    native_scores = np.asarray([p["raw_confidence"] for p in pred_records], dtype=np.float32)
    native_order = list(np.argsort(-native_scores, kind="stable"))
    oracle_order = list(np.argsort(-max_iou, kind="stable"))
    tp50_oracle_key = np.asarray(
        [float(max_iou[i] >= 0.5) * 2.0 + float(max_iou[i]) for i in range(len(max_iou))]
    )
    tp50_order = list(np.argsort(-tp50_oracle_key, kind="stable"))
    native = _ap(pred_masks, native_scores, gt_masks, pred_ids, gt_ids)
    oracle = _ap(pred_masks, max_iou, gt_masks, pred_ids, gt_ids)
    tp50_ap = _ap(pred_masks, tp50_oracle_key, gt_masks, pred_ids, gt_ids)
    native_match = _match(native_order, matrix, 0.5)
    oracle_match = _match(oracle_order, matrix, 0.5)
    tp50_match = _match(tp50_order, matrix, 0.5)
    tp_scores = native_scores[native_match["tp"]] if native_match["tp"] else np.array([])
    fp_scores = native_scores[native_match["fp"]] if native_match["fp"] else np.array([])
    duplicate = [i for i in native_match["fp"] if max_iou[i] >= 0.5]
    for p, iou in zip(pred_records, max_iou):
        p["max_iou"] = float(iou)
    for g, iou in zip(gt_records, best_gt):
        g["best_iou"] = float(iou)
    native_tp = set(native_match["tp"])
    native_dup = set(duplicate)
    native_gt = native_match["matched"]
    oracle_tp = set(oracle_match["tp"])
    tp50_oracle_tp = set(tp50_match["tp"])
    for i, p in enumerate(pred_records):
        p["native_tp50"] = bool(i in native_tp)
        p["native_fp50"] = bool(i not in native_tp)
        p["native_duplicate50"] = bool(i in native_dup)
        p["native_matched_gt_index"] = (
            int(native_gt[i]) if i in native_gt else None
        )
        p["oracle_max_iou_tp50"] = bool(i in oracle_tp)
        p["oracle_tp50_priority_tp50"] = bool(i in tp50_oracle_tp)
    summary = {
        "scene": scene_path.name,
        "gt_count": len(gt_records), "kept_prediction_count": len(pred_records),
        "all_candidate_count": len(all_records),
        "filtered_by_max_predictions": sum(p["filtered_by"] is not None for p in all_records),
        "score_threshold_filtered_count": 0,
        "native": native, "oracle_max_iou": oracle,
        "oracle_tp50_priority": tp50_ap,
        "ranking_headroom_ap50": float(oracle["ap_50"] - native["ap_50"]),
        "tp50_priority_headroom_ap50": float(tp50_ap["ap_50"] - native["ap_50"]),
        "spearman_score_max_iou": _spearman(native_scores, max_iou),
        "kendall_score_max_iou": _kendall(native_scores, max_iou),
        "native_tp50_count": len(native_match["tp"]),
        "native_fp_count": len(native_match["fp"]),
        "native_duplicate_prediction_count": len(duplicate),
        "native_tp_score_mean": float(tp_scores.mean()) if len(tp_scores) else None,
        "native_fp_score_mean": float(fp_scores.mean()) if len(fp_scores) else None,
        "native_tp_score_p50": float(np.percentile(tp_scores, 50)) if len(tp_scores) else None,
        "native_fp_score_p50": float(np.percentile(fp_scores, 50)) if len(fp_scores) else None,
        "best_iou_histogram": _hist(best_gt),
        "native_match50": native_match,
        "oracle_match50": oracle_match,
        "oracle_tp50_match50": tp50_match,
    }
    for p in pred_records:
        p["native_max_iou"] = p.pop("max_iou")
    return summary, pred_records, gt_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--joint", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    models = {"both_1420": Path(args.baseline), "head_joint_125": Path(args.joint)}
    summaries = {}
    best_by_model = {}
    detailed = {}
    pred_rows = []
    gt_rows = []
    for model, path in models.items():
        model_summaries = []
        model_best = {}
        for scene_path in sorted(p for p in path.iterdir() if p.is_dir()):
            summary, preds, gts = _scene_audit(scene_path)
            summary["model"] = model
            model_summaries.append(summary)
            detailed[f"{model}/{scene_path.name}"] = {
                "summary": summary, "predictions": preds, "ground_truth": gts
            }
            pred_rows.extend({"model": model, "scene": scene_path.name, **p} for p in preds)
            gt_rows.extend({"model": model, "scene": scene_path.name, **g} for g in gts)
            model_best.update({(scene_path.name, g["gt_id"]): g["best_iou"] for g in gts})
        summaries[model] = model_summaries
        best_by_model[model] = model_best

    (root / "summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (root / "per_scene_diagnostics.json").write_text(
        json.dumps(detailed, indent=2), encoding="utf-8"
    )
    for name, rows in [("per_prediction.csv", pred_rows), ("per_gt.csv", gt_rows)]:
        keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in keys} for row in rows)

    # Cross-model GT migration; GT ids are fixed by the manifest and view order.
    by_model_gt = {}
    for model, rows in summaries.items():
        by_model_gt[model] = best_by_model[model]
    migrations = []
    for key in sorted(set(by_model_gt["both_1420"]) & set(by_model_gt["head_joint_125"])):
        b = by_model_gt["both_1420"][key]; j = by_model_gt["head_joint_125"][key]
        migrations.append({"scene": key[0], "gt_id": key[1], "baseline_best_iou": b,
                           "joint_best_iou": j, "delta": j - b,
                           "baseline_bin": _hist(np.asarray([b])), "joint_bin": _hist(np.asarray([j]))})
    (root / "gt_migrations.json").write_text(json.dumps(migrations, indent=2), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps({"models": summaries, "gt_migrations": migrations}, indent=2), encoding="utf-8")
    print(json.dumps({model: [{k: s[k] for k in ["scene", "native", "oracle_max_iou", "ranking_headroom_ap50", "spearman_score_max_iou", "kendall_score_max_iou", "native_duplicate_prediction_count"]} for s in rows] for model, rows in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
