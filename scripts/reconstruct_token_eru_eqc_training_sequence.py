"""Reconstruct the matched first-200 global-batch sequence without training.

The production loader uses Accelerate's non-split BatchSamplerShard.  With a
batch size of one, rank ``r`` receives the raw shuffled sample at positions
``r, r+8, ...``.  This script materializes only stable sample identifiers and
never constructs a model or optimizer.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokengs.options import config_defaults  # noqa: E402

CONTROL = "semantic_v6_j2_local250_eqc_v1_control_short200_ddp8"
TREATMENT = "semantic_v6_j2_local250_eqc_v1_treatment_short200_ddp8"
WORLD_SIZE = 8
STEPS = 200


def _one(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.reshape(-1)[0].item()
        return value[0].tolist()
    if isinstance(value, (list, tuple)):
        return _one(value[0]) if len(value) == 1 else [_one(item) for item in value]
    return value


def stable_sample(batch: dict) -> dict:
    scene = str(_one(batch.get("scene_name", "?")))
    frame_ids = _one(batch.get("frame_ids", []))
    if isinstance(frame_ids, list) and frame_ids and isinstance(frame_ids[0], list):
        frame_ids = frame_ids[0]
    frame_ids = [int(value) for value in (frame_ids or [])]
    context = frame_ids[:8]
    target = frame_ids[8:]
    payload = {
        "scene_id": scene,
        "context_frame_ids": context,
        "target_frame_ids": target,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["window_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return payload


def collect(config_name: str) -> list[dict]:
    torch.manual_seed(42)
    random.seed(42)
    opt = dataclasses.replace(config_defaults[config_name])
    opt.num_workers = 0
    opt.batch_size = 1
    opt.evaluating = False
    manifest_path = Path(opt.dataset_kwargs["small_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = list(manifest["train_samples"])
    total = len(samples)
    indices = torch.randperm(
        total, generator=torch.Generator(device="cpu").manual_seed(opt.seed)
    ).tolist()

    selected_indices = indices[: STEPS * WORLD_SIZE]
    raw = []
    for global_index in selected_indices:
        sample = samples[global_index]
        scene_id = str(sample["scene"])
        context = list(sample["input_frame_ids"])
        target = list(sample["target_frame_ids"])
        payload = {
            "scene_id": scene_id,
            "context_frame_ids": [int(value) for value in context],
            "target_frame_ids": [int(value) for value in target],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["window_fingerprint"] = hashlib.sha256(encoded).hexdigest()
        raw.append(payload)
    records = []
    for step in range(1, STEPS + 1):
        ranks = []
        for rank in range(WORLD_SIZE):
            sample = raw[(step - 1) * WORLD_SIZE + rank]
            ranks.append({"rank": rank, **sample})
        records.append({"stage_local_step": step, "ranks": ranks})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    control = collect(CONTROL)
    treatment = collect(TREATMENT)
    if len(control) != STEPS or len(treatment) != STEPS:
        raise RuntimeError("sequence length is not 200")
    if control != treatment:
        for index, (left, right) in enumerate(zip(control, treatment), start=1):
            if left != right:
                raise RuntimeError(f"control/treatment sequence mismatch at step {index}")
        raise RuntimeError("control/treatment sequence mismatch")
    (output / "control_training_sequence.json").write_text(
        json.dumps({"config": CONTROL, "world_size": WORLD_SIZE, "steps": control}, indent=2),
        encoding="utf-8",
    )
    (output / "treatment_expected_sequence.json").write_text(
        json.dumps({"config": TREATMENT, "world_size": WORLD_SIZE, "steps": treatment}, indent=2),
        encoding="utf-8",
    )
    report = {
        "control_config": CONTROL,
        "treatment_config": TREATMENT,
        "steps": STEPS,
        "world_size": WORLD_SIZE,
        "batch_size_per_rank": 1,
        "batches_skipped": 0,
        "sequence_match": True,
        "accelerate_batch_sampler_mapping": "raw position modulo world_size",
        "control_first_step": control[0],
        "treatment_first_step": treatment[0],
    }
    (output / "sequence_match_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
