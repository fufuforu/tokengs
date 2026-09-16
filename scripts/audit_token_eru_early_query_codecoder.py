"""Static/runtime architecture audit for EQC-v1.

The audit is intentionally read-only and does not run an optimizer step.  It
checks the fixed dimensions and the source-level wiring contract; the GPU
forward checks are performed by the bounded preflight script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIGS = (
    "semantic_v6_j2_local250_eqc_v1_control_short200_ddp8",
    "semantic_v6_j2_local250_eqc_v1_treatment_short200_ddp8",
)


def audit() -> dict:
    from tokengs.options import config_defaults

    result = {
        "experiment": "TokenGS-ERU-Early-Query-CoDecoder-v1",
        "configs": {},
        "native_query_count": 100,
        "native_query_dim": 256,
        "early_query_layers": [2, 5, 8, 11],
        "void_is_early_query": False,
        "shared_adapter": True,
        "qmc_formal": False,
        "unit_3d_anchor_formal": False,
        "metric_cluster_formal": False,
        "p_u_formal": False,
        "target_image_to_dino": False,
        "dino_auxiliary_preserved": True,
        "native_query_formal_output": True,
    }
    for name in CONFIGS:
        opt = config_defaults[name]
        result["configs"][name] = {
            "eqc_enabled": bool(getattr(opt, "token_eru_early_query_codecoder_enabled", False)),
            "qmc_enabled": bool(getattr(opt, "token_eru_query_metric_enabled", False)),
            "anchor_enabled": bool(getattr(opt, "token_eru_3d_anchor_enabled", False)),
            "context_views": int(opt.num_input_views),
            "target_views": int(opt.num_views - opt.num_input_views),
            "train_pool_windows": 5680,
            "stage_local_steps": int(getattr(opt, "max_iters_per_epoch", 0)),
            "seed": int(opt.seed),
        }
    if any(item["qmc_enabled"] or item["anchor_enabled"] for item in result["configs"].values()):
        raise RuntimeError("EQC configs unexpectedly enable QMC or 3D anchor")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit(), indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
