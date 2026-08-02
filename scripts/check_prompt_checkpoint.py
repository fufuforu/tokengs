#!/usr/bin/env python3
"""Validate a lightweight prompt-training checkpoint and its base-model provenance."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--checkpoint",
        default="model.safetensors",
        help="Checkpoint filename or path relative to --workspace.",
    )
    args = parser.parse_args()
    workspace = Path(args.workspace)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = workspace / checkpoint
    metadata_name = (
        "metadata_best.json"
        if checkpoint.name == "model_best.safetensors"
        else "metadata.json"
    )
    metadata_path = checkpoint.parent / metadata_name
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Expected {checkpoint} and {metadata_path}"
        )
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in keys}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    forbidden = [key for key in keys if "clip" in key or "gs_tokens" in key]
    if forbidden:
        raise RuntimeError(f"Frozen weights leaked into prompt checkpoint: {forbidden}")
    model_type = metadata.get("model_type", "prompt_tokengs")
    if model_type == "semantic_tokengs_v2":
        allowed = (
            "semantic_token_adapter.",
            "prompt_semantic_adapter.",
            "semantic_last_decoder.gs_cross_attn.",
            "semantic_last_decoder.gs_cross_attn_scale.",
        )
        invalid = [
            key
            for key in keys
            if key != "log_temperature" and not key.startswith(allowed)
        ]
        if invalid or "log_temperature" not in keys:
            raise RuntimeError(f"Invalid Semantic Adapter V2 keys: {invalid}")
    elif not keys or not all(key.startswith("matching_decoder.") for key in keys):
        raise RuntimeError("Checkpoint contains non-matching parameters")
    base_path = metadata.get("tokengs_checkpoint")
    if not base_path or not Path(base_path).is_file():
        raise RuntimeError(f"Recorded TokenGS checkpoint is invalid: {base_path}")
    print(
        json.dumps(
            {
                "prompt_checkpoint": str(checkpoint.resolve()),
                "tensor_count": len(keys),
                "keys": shapes,
                "forbidden_keys": forbidden,
                "tokengs_checkpoint": base_path,
                "model_type": model_type,
                "epoch": metadata.get("epoch"),
                "step": metadata.get("step"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
