#!/usr/bin/env python3
"""Run local-CLIP prompt matching forwards and report module footprint."""

import argparse
import json

import torch

from tokengs.models.prompt_matching import (
    DEFAULT_CLIP_MODEL_PATH,
    PromptConditionedTokenMatcher,
)


def _parameter_stats(module: torch.nn.Module) -> dict[str, int]:
    parameters = list(module.parameters())
    return {
        "total": sum(parameter.numel() for parameter in parameters),
        "trainable": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "bytes": sum(parameter.numel() * parameter.element_size() for parameter in parameters),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-model-path", default=str(DEFAULT_CLIP_MODEL_PATH))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = torch.cuda.memory_allocated(device)
    else:
        memory_before = None

    matcher = PromptConditionedTokenMatcher(args.clip_model_path).to(device).eval()
    image = torch.rand(1, 3, 224, 224, device=device)
    mask = torch.zeros(1, 1, 224, 224, device=device)
    mask[:, :, 24:200, 40:184] = 1
    shapes = {}
    with torch.no_grad():
        for mode in ("text_only", "image_only", "text_image_mixed"):
            output = matcher(
                torch.randn(1, 1024, 1024, device=device),
                mode=mode,
                text_query=["chair"],
                query_image=image,
                query_mask=mask,
            )
            shapes[mode] = {
                name: list(value.shape) for name, value in output.items()
            }
        output_4096 = matcher(
            torch.randn(1, 4096, 1024, device=device),
            mode="text_only",
            text_query=["wall"],
        )
        shapes["4096_tokens"] = list(output_4096["token_logits"].shape)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        cuda_memory = {
            "allocated_delta_bytes": torch.cuda.memory_allocated(device) - memory_before,
            "peak_delta_bytes": torch.cuda.max_memory_allocated(device) - memory_before,
        }
    else:
        cuda_memory = {"available": False}

    checkpoint = matcher.trainable_state_dict()
    report = {
        "device": str(device),
        "clip_model_path": str(matcher.prompt_encoder.model_path),
        "prompt_encoder": _parameter_stats(matcher.prompt_encoder),
        "matching_decoder": _parameter_stats(matcher.matching_decoder),
        "trainable_checkpoint_tensors": len(checkpoint),
        "trainable_checkpoint_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in checkpoint.values()
        ),
        "clip_keys_in_trainable_checkpoint": sum(
            "clip" in key for key in checkpoint
        ),
        "shapes": shapes,
        "cuda_memory": cuda_memory,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
