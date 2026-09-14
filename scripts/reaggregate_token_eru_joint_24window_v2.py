"""Offline repair of the JointFormation 24-record audit status.

This script never loads a model and never recomputes predictions.  It only
recomputes numeric-finiteness and the AP101 protocol invariant from the
existing per-window JSON, and writes a new v2 audit directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workspace/token_eru_dino_joint_formation_v1_24window_paired_eval"
OUTPUT = ROOT / "workspace/token_eru_dino_joint_formation_v1_24window_paired_eval_v2"


def ap101_upper_bound(max_recall: float, eps: float = 1e-9) -> float:
    if not math.isfinite(max_recall):
        raise ValueError("max_recall must be finite")
    if max_recall < -eps or max_recall > 1.0 + eps:
        raise ValueError("max_recall must be in [0, 1]")
    recall = min(1.0, max(0.0, max_recall))
    return (math.floor(100.0 * recall + eps) + 1) / 101.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_numbers(value: Any, *, key: str = "") -> bool:
    if key in {"invariant_valid", "ap101_invariant_valid"}:
        return True
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_numbers(item, key=str(name)) for name, item in value.items())
    return True


def _fingerprint_key(fp: dict[str, Any]) -> tuple[Any, ...]:
    # sample_id/order are record identifiers, not scene-window identity.  The
    # content/frame/camera/GT fingerprint defines the unique scene-window.
    return (
        fp["scene_id"], tuple(fp["context_frame_ids"]), tuple(fp["target_frame_ids"]),
        fp["context_rgb_sha256"], fp["target_rgb_sha256"],
        fp["target_gt_instance_sha256"], fp["target_camera_sha256"],
    )


def _ap101_valid(native: dict[str, Any]) -> bool:
    valid = True
    for row in native["per_view"]:
        ap50 = float(row["ap50"])
        recall = float(row["max_achievable_recall50"])
        valid = valid and (
            math.isfinite(ap50)
            and -1e-9 <= ap50 <= 1.0 + 1e-9
            and math.isfinite(recall)
            and -1e-9 <= recall <= 1.0 + 1e-9
            and int(row["tp_iou50_count"]) <= int(row["gt_count"])
            and ap50 <= ap101_upper_bound(recall) + 1e-9
        )
    return bool(valid)


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUTPUT}")
    source = json.loads((SOURCE / "per_window_metrics.json").read_text(encoding="utf-8"))
    protocol = json.loads((SOURCE / "protocol_fingerprints.json").read_text(encoding="utf-8"))
    all_records = []
    repaired = {}
    model_status = {}
    for model_name, payload in source.items():
        records = []
        for original in payload["records"]:
            record = copy.deepcopy(original)
            numeric_finite = bool(
                _finite_numbers(record["native"])
                and _finite_numbers(record["reconstruction"])
            )
            invariant_valid = _ap101_valid(record["native"])
            record["numeric_finite"] = numeric_finite
            record["ap101_invariant_valid"] = invariant_valid
            record["finite"] = numeric_finite  # legacy display compatibility
            records.append(record)
            all_records.append(record)
        repaired[model_name] = {"records": records, "aggregate": payload["aggregate"]}
        model_status[model_name] = {
            "numeric_finite": all(r["numeric_finite"] for r in records),
            "ap101_invariant_valid": all(r["ap101_invariant_valid"] for r in records),
            "record_count": len(records),
        }

    groups = defaultdict(list)
    for record in repaired[next(iter(repaired))]["records"]:
        groups[_fingerprint_key(record["fingerprint"])].append(record["fingerprint"]["sample_id"])
    distribution = Counter(record["fingerprint"]["scene_id"] for record in all_records[:24])
    unique_rows = []
    for key, sample_ids in groups.items():
        unique_rows.append({
            "scene_id": key[0],
            "context_frame_ids": list(key[1]),
            "target_frame_ids": list(key[2]),
            "context_rgb_sha256": key[3],
            "target_rgb_sha256": key[4],
            "target_gt_instance_sha256": key[5],
            "target_camera_sha256": key[6],
            "record_count": len(sample_ids),
            "record_sample_ids": sample_ids,
        })
    protocol_v2 = copy.deepcopy(protocol)
    protocol_v2.update({
        "records_total": 24,
        "unique_scene_window_count": len(unique_rows),
        "held_out_scene_count": len(distribution),
        "scene_distribution": dict(sorted(distribution.items())),
        "scene_window_fingerprint_key": [
            "scene_id", "context_frame_ids", "target_frame_ids",
            "context_rgb_sha256", "target_rgb_sha256",
            "target_gt_instance_sha256", "target_camera_sha256",
        ],
        "description": "24 records / 8 unique scene-windows / 8 held-out scenes",
        "ap101_tolerance": 1e-9,
        "numeric_finite_separated_from_invariant": True,
        "unique_scene_windows": unique_rows,
    })
    summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    summary["protocol_fingerprints"] = protocol_v2
    summary["results"] = {
        **summary.get("results", {}),
        **{
            name: {
                **summary.get("results", {}).get(name, {}),
                **status,
                "all_finite": status["numeric_finite"],
            }
            for name, status in model_status.items()
        },
    }
    summary["validation_records"] = 24
    summary["unique_scene_window_count"] = len(unique_rows)
    summary["held_out_scene_count"] = len(distribution)
    summary["scene_distribution"] = dict(sorted(distribution.items()))
    summary["description"] = "24 records / 8 unique scene-windows / 8 held-out scenes"
    summary["cached_reaggregation"] = True
    summary["model_rerun"] = False

    OUTPUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads((SOURCE / "checkpoint_selection.json").read_text(encoding="utf-8"))
    for row in selection.get("candidates", []):
        status = model_status.get(f"joint_{int(row['step']):06d}", {})
        row["numeric_finite"] = bool(status.get("numeric_finite", False))
        row["ap101_invariant_valid"] = bool(status.get("ap101_invariant_valid", False))
        row["eligible"] = bool(
            row.get("eligible", False)
            and row["numeric_finite"]
            and row["ap101_invariant_valid"]
        )
    eligible_rows = [row for row in selection.get("candidates", []) if row.get("eligible")]
    eligible_rows.sort(key=lambda row: (
        -float(row["scene_macro_ap50"]), -float(row["pooled_ap50"]),
        -float(row["best_gt_iou"]), -float(row["recall50"]), int(row["step"]),
    ))
    selection["best_checkpoint"] = eligible_rows[0] if eligible_rows else None
    selection["ready_for_lsm40"] = bool(eligible_rows)

    for name, payload in {
        "per_window_metrics.json": repaired,
        "protocol_fingerprints.json": protocol_v2,
        "summary.json": summary,
        "per_scene_metrics.json": json.loads((SOURCE / "per_scene_metrics.json").read_text(encoding="utf-8")),
        "comparison_vs_eru500.json": json.loads((SOURCE / "comparison_vs_eru500.json").read_text(encoding="utf-8")),
        "checkpoint_selection.json": selection,
        "checkpoint_metadata.json": json.loads((SOURCE / "checkpoint_metadata.json").read_text(encoding="utf-8")),
        "audit_report.json": {
            "source": str(SOURCE.resolve()),
            "source_per_window_sha256": _sha256(SOURCE / "per_window_metrics.json"),
            "cached_reaggregation": True,
            "model_rerun": False,
            "model_status": model_status,
            "records_total": 24,
            "unique_scene_window_count": len(unique_rows),
            "held_out_scene_count": len(distribution),
            "scene_distribution": dict(sorted(distribution.items())),
            "ap101_tolerance": 1e-9,
        },
    }.items():
        (OUTPUT / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "reaggregation.log").write_text(
        "offline cached reaggregation; no model forward, training, or evaluation rerun\n",
        encoding="utf-8",
    )
    (OUTPUT / "status").mkdir()
    (OUTPUT / "status" / "COMPLETE").write_text("ok\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "models": model_status, "unique_scene_windows": len(unique_rows)}, indent=2))


if __name__ == "__main__":
    main()
