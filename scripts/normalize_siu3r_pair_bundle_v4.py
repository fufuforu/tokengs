"""Normalize the v4 pair-0 bundle using the official saved pred.json order.

This is an audit utility only: it copies prediction/GT arrays from an
existing bundle and replaces the lossy metadata ordering with the exact
official ``id, label_id, score`` association.  It never reads target GT to
choose or alter a prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _metadata(path: Path, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_id = {int(row["id"]): row for row in rows}
    ordered = sorted(int(value) for value in np.unique(ids) if int(value) != 0)
    if set(ordered) != set(by_id):
        raise ValueError(f"{path}: metadata IDs do not match prediction map IDs")
    labels = np.asarray([int(by_id[value]["label_id"]) for value in ordered], dtype=np.int64)
    scores = np.asarray([float(by_id[value]["score"]) for value in ordered], dtype=np.float32)
    return labels, scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--context-json", type=Path, required=True)
    parser.add_argument("--target-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    with np.load(args.input, allow_pickle=False) as src:
        payload = {key: src[key] for key in src.files}
    context_labels, context_scores = _metadata(args.context_json, payload["context_instance_pred"])
    target_labels, target_scores = _metadata(args.target_json, payload["target_instance_pred"])
    payload.update(
        context_pred_labels=context_labels,
        context_pred_scores=context_scores,
        target_pred_labels=target_labels,
        target_pred_scores=target_scores,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
