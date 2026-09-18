"""Strict TokenGS SIU3R pair-0 entrypoint.

The current J2 checkpoint is intentionally rejected before importing TokenGS
model code because its native contract is eight context views.  This guard is
kept in a separate entrypoint so a future variable-view checkpoint can add a
real forward implementation without changing the evaluator or legacy LSM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", default="3090")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    payload = {
        "protocol": "siu3r_global_multiview_v1",
        "pair_index": 0,
        "partition": args.partition,
        "strict_input": "NO",
        "messages": [
            "STRICT_SIU3R_INPUT_VIEW_PARITY: NO",
            "RETRAINING_OR_VARIABLE_VIEW_SUPPORT_REQUIRED: YES",
        ],
        "model_forward_started": False,
        "optimizer_step_executed": False,
        "ttt_started": False,
        "oracle_used": False,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "val_pair_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        "note": "Current J2 is configured for 8 context views; no 2-view TokenGS forward was attempted.",
    }
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(args.output)
    print(json.dumps(payload, indent=2))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
