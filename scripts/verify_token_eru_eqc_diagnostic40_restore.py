"""Verify the model-only step-40 artifact from diagnostic40, without training."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import load_model_checkpoint  # noqa: E402


def sha(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ws = args.workspace.resolve()
    ckpt = ws / "checkpoints" / "model_step_000040.safetensors"
    meta = ws / "checkpoints" / "metadata_step_000040.json"
    marker = ws / "checkpoints" / "step_000040.complete"
    for path in (ckpt, meta, marker):
        if not path.is_file():
            raise RuntimeError(f"missing diagnostic checkpoint artifact: {path}")
    state = load_file(str(ckpt), device="cpu")
    if not state or not all(bool(torch.isfinite(v).all()) for v in state.values()):
        raise FloatingPointError("diagnostic checkpoint has non-finite tensor")
    opt = dataclasses.replace(config_defaults["semantic_v6_j2_local250_eqc_v1_treatment_short200_ddp8"])
    opt.workspace = str(ws)
    opt.resume = str(ckpt)
    model = model_registry[opt.model_type](opt)
    # Match the production ERU parent/model-only loader exactly.  The
    # ``gs_tokens`` parameter is intentionally included through the base
    # Module state dict even though SemanticTokenGSv4's filtered state_dict
    # view does not expose it in every construction mode.
    expected_state = nn.Module.state_dict(model)
    active_prefixes = (
        "enc_dec_backbone.decoder_blocks.",
        "absolute_gs_head.",
        "tsh_instance_head.",
        "token_eru_decoder.",
        "token_eru_unit_formation.",
        "token_eru_dino_encoder.unit_projector.",
        "token_eru_dino_fusion.",
        "token_eru_metric_head.",
        "token_eru_3d_anchor.",
        "token_eru_query_metric_coupling.",
        "token_eru_decoder.early_query_adapter.",
    )
    active_expected = {
        key for key in expected_state
        if key in ("gs_tokens", "gs_tokens_dynamic")
        or key.startswith(active_prefixes)
    }
    active_actual = {
        key for key in state
        if key in ("gs_tokens", "gs_tokens_dynamic")
        or key.startswith(active_prefixes)
    }
    missing = sorted(active_expected - active_actual)
    unexpected = sorted(active_actual - active_expected)
    mismatched = sorted(
        (key, tuple(state[key].shape), tuple(expected_state[key].shape))
        for key in active_expected & active_actual
        if state[key].shape != expected_state[key].shape
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"strict active-key mismatch missing={missing} unexpected={unexpected} "
            f"shape_mismatch={mismatched}"
        )

    class _LocalAccelerator:
        @staticmethod
        def print(*args, **kwargs):
            print(*args, **kwargs)

    # This is the same strict active-prefix restore used by the trainer and
    # does not construct an optimizer or execute a training step.
    load_model_checkpoint(opt, model, _LocalAccelerator(), 0)
    after = nn.Module.state_dict(model)
    max_diff = max(float((after[key].detach().cpu() - state[key]).abs().max()) for key in active_actual)
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    report = {
        "checkpoint": str(ckpt),
        "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "metadata": metadata,
        "strict_key_restore": True,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch": mismatched,
        "all_tensors_finite": True,
        "strict_restore_max_diff": max_diff,
        "model_state_key_count": len(state),
        "dino_backbone_checkpoint_keys": [
            key for key in state
            if key.startswith("token_eru_dino_backbone.")
            or key.startswith("dino_backbone.")
        ],
        "dino_auxiliary_projection_keys": [
            key for key in state
            if "token_eru_dino_encoder.unit_projector." in key
            or "token_eru_dino_fusion." in key
        ],
        "parameter_values_finite_after_restore": all(bool(torch.isfinite(v).all()) for v in after.values()),
        "formal_training_started": False,
        "optimizer_step_executed_by_verifier": False,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
