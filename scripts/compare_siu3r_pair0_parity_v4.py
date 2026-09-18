"""Compare official SIU3R JSON with the adapter on one identical bundle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _flatten(value: Any) -> list[float]:
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [float(value)]


def _mean(value: Any) -> float:
    values = _flatten(value)
    return sum(values) / len(values)


def _compare(name: str, official: Any, adapter: Any, tolerance: float) -> dict[str, Any]:
    left, right = _flatten(official), _flatten(adapter)
    if len(left) != len(right):
        return {"metric": name, "official": official, "adapter": adapter, "status": "FAIL", "reason": "shape/length mismatch"}
    diffs = [abs(a - b) for a, b in zip(left, right)]
    max_diff = max(diffs, default=0.0)
    denom = max(max(abs(a) for a in left), 1e-12)
    return {
        "metric": name,
        "official": official,
        "adapter": adapter,
        "absolute_difference": max_diff,
        "relative_difference": max_diff / denom,
        "tolerance": tolerance,
        "status": "PASS" if max_diff <= tolerance else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    official = json.loads(args.official.read_text(encoding="utf-8"))
    adapter = json.loads(args.adapter.read_text(encoding="utf-8"))["results"][0]
    metrics: list[dict[str, Any]] = []
    for name, left, right, tolerance in [
        ("psnr", official["psnr"], adapter["reconstruction"]["psnr"], 1e-6),
        ("ssim", official["ssim"], adapter["reconstruction"]["ssim"], 1e-6),
        ("lpips", official["lpips"], adapter["reconstruction"]["lpips"], 1e-5),
        ("absrel", official["absrel"], adapter["depth"]["absrel"], 1e-6),
        ("rmse", official["rmse"], adapter["depth"]["rmse"], 1e-6),
        ("context_miou", official["context_miou"], adapter["context_miou"], 1e-6),
        ("target_miou", official["target_miou"], adapter["target_miou"], 1e-6),
        ("context_pq", official["context_pq"], _mean(adapter["context_pq"]), 1e-6),
        ("target_pq", official["target_pq"], _mean(adapter["target_pq"]), 1e-6),
        ("context_pqs_per_class", official["context_pqs_per_class"], adapter["context_pq"], 1e-6),
        ("target_pqs_per_class", official["target_pqs_per_class"], adapter["target_pq"], 1e-6),
    ]:
        metrics.append(_compare(name, left, right, tolerance))
    for side in ("context_map", "target_map"):
        for key, tolerance in (("map", 1e-6), ("map_50", 1e-6), ("map_75", 1e-6), ("map_per_class", 1e-6), ("mar_100_per_class", 1e-6), ("classes", 0.0)):
            metrics.append(_compare(f"{side}.{key}", official[side][key], adapter[side][key], tolerance))
    for index, (left, right) in enumerate(zip(official.get("render_scores", []), adapter.get("reconstruction_per_target", []))):
        for key, tolerance in (("psnr", 1e-6), ("ssim", 1e-6), ("lpips", 1e-5)):
            metrics.append(_compare(f"target[{index}].{key}", left[key], right[key], tolerance))
    for index, (left, right) in enumerate(zip(official.get("depth_scores", []), adapter.get("target_depth", []))):
        for key in ("absrel", "rmse"):
            metrics.append(_compare(f"depth[{index}].{key}", left[key], right[key], 1e-6))
    failed = [item for item in metrics if item["status"] != "PASS"]
    payload = {
        "status": "NUMERICAL_PARITY_PASS" if not failed else "NUMERICAL_PARITY_FAIL",
        "same_prediction_bundle": True,
        "official_json": str(args.official.resolve()),
        "adapter_json": str(args.adapter.resolve()),
        "tolerances": {"fp32": 1e-6, "lpips_or_gpu_reduction": 1e-5},
        "metrics": metrics,
        "failed_metrics": [item["metric"] for item in failed],
        "max_metric_abs_diff": max((item.get("absolute_difference", math.inf) for item in metrics), default=0.0),
        "first_intermediate_divergence": failed[0]["metric"] if failed else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
