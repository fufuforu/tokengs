"""Audit the completed official SIU3R 1860-pair output tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_json(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    records = json.loads(args.pairs.read_text(encoding="utf-8"))
    val_root = args.output_root / "val" / "0"
    expected = []
    for index, record in enumerate(records):
        name = f"{record['scan']}_context{'_'.join(map(str, record['context_ids']))}"
        expected.append((index, record, val_root / name))
    required_dirs = ("rgb", "rgb_gt", "depth", "depth_gt", "context_seg_pred", "context_seg_gt", "target_seg_pred", "target_seg_gt")
    rows = []
    failures = []
    for index, record, directory in expected:
        row: dict[str, Any] = {"pair_index": index, "scan": record["scan"], "context_ids": record["context_ids"], "target_ids": record["target_ids"], "path": str(directory), "exists": directory.is_dir()}
        row["required_directories"] = {name: (directory / name).is_dir() for name in required_dirs}
        row["file_counts"] = {name: len(list((directory / name).glob("*.png"))) if (directory / name).is_dir() else 0 for name in required_dirs}
        row["pred_json_nonempty"] = all((directory / name / "pred.json").is_file() and bool(json.loads((directory / name / "pred.json").read_text())) for name in ("context_seg_pred", "target_seg_pred")) if directory.is_dir() else False
        row["render_scores_finite"] = False
        row["depth_scores_finite"] = False
        for name, key in (("render_scores.json", "render_scores_finite"), ("depth_scores.json", "depth_scores_finite")):
            path = directory / name
            if path.is_file():
                value = json.loads(path.read_text())
                row[key] = finite_json(value)
        row["all_required_counts"] = row["file_counts"] == {"rgb": 6, "rgb_gt": 6, "depth": 6, "depth_gt": 6, "context_seg_pred": 2, "context_seg_gt": 2, "target_seg_pred": 6, "target_seg_gt": 6}
        row["all_finite"] = bool(row["render_scores_finite"] and row["depth_scores_finite"])
        row["integrity"] = bool(row["exists"] and all(row["required_directories"].values()) and row["all_required_counts"] and row["pred_json_nonempty"] and row["all_finite"])
        rows.append(row)
        if not row["integrity"]:
            failures.append(index)
    actual = sorted(path.name for path in val_root.iterdir() if path.is_dir()) if val_root.is_dir() else []
    expected_names = sorted(row["path"].split("/")[-1] for row in rows)
    metrics_path = args.output_root / "official_metrics_raw.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    scenes = sorted({record["scan"] for record in records})
    payload = {
        "protocol": "siu3r_global_multiview_v1",
        "records_expected": len(records),
        "records_found": len(rows),
        "unique_scenes_expected": len(scenes),
        "unique_scenes_found": len({row["scan"] for row in rows}),
        "unique_pair_names": len(set(expected_names)),
        "duplicate_pair_names": len(expected_names) - len(set(expected_names)),
        "missing_pair_names": sorted(set(expected_names) - set(actual)),
        "unexpected_pair_names": sorted(set(actual) - set(expected_names)),
        "pairs_with_integrity_failure": failures,
        "all_referenced_pairs_found": not (set(expected_names) - set(actual)),
        "all_outputs_finite": all(row["all_finite"] for row in rows),
        "data_integrity_valid": not failures and expected_names == actual,
        "official_results_present": bool(metrics),
        "checkpoint_sha256": sha256(args.checkpoint),
        "pair_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(args.output)
    per_pair = []
    for row in rows:
        directory = Path(row["path"])
        render = json.loads((directory / "render_scores.json").read_text())
        depth = json.loads((directory / "depth_scores.json").read_text())
        per_pair.append({"pair_index": row["pair_index"], "scan": row["scan"], "context_ids": row["context_ids"], "target_ids": row["target_ids"], "render_mean": {key: float(np.mean([item[key] for item in render])) for key in ("psnr", "ssim", "lpips")}, "depth_mean": {key: float(np.mean([item[key] for item in depth])) for key in ("absrel", "rmse")}})
    per_pair_path = args.output.parent / "per_pair_metrics.json"
    per_pair_path.write_text(json.dumps(per_pair, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: payload[key] for key in ("records_expected", "unique_scenes_found", "unique_pair_names", "data_integrity_valid", "all_outputs_finite")}, indent=2))
    return 0 if payload["data_integrity_valid"] and payload["official_results_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
