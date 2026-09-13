"""Fixed-batch smoke for TokenGS-ERU-DINO-Metric-v1.

This is intentionally a bounded diagnostic.  It never writes a formal
training workspace or starts a multi-scene evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import load_model_checkpoint, setup_optimizer  # noqa: E402


SOURCE = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_"
    "scene_hungarian_resume200_to500_persist_ddp8/checkpoints/"
    "model_step_000500.safetensors"
)


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    return value


def tensor_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            tensor = parameter.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--config", default="semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_treatment_resume500_to700_ddp8")
    parser.add_argument("--output-dir", default="workspace/token_eru_dino_metric_fixed_batch_audit_v1")
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        raise ValueError("fixed-batch smoke is limited to 1..100 optimizer steps")
    output_dir = ROOT / args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty smoke directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    opt = dataclasses.replace(config_defaults[args.config])
    opt.resume = str(SOURCE)
    opt.workspace = str(output_dir)
    opt.num_workers = 0
    opt.batch_size = 1
    opt.tsh_ddp8 = False
    opt.eval_before_training = False
    opt.use_wandb = False
    # Match tokengs.train initialization so fresh metric modules are identical
    # across ranks while the model itself preserves the source RNG trajectory.
    torch.manual_seed(opt.seed)
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    loader, _, _, _ = get_multi_dataloader(opt, accelerator)
    model = model_registry[opt.model_type](opt)
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        model.initialize_token_eru_from_reconstruction()
    optimizer = setup_optimizer(opt, model, accelerator, 0)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    unwrapped = accelerator.unwrap_model(model)
    batch = move(next(iter(loader)), accelerator.device)
    if batch["input"].shape[1] != 15 or batch["instance_label_output"].shape[1] != 7:
        raise RuntimeError("fixed batch is not the formal 8+7 protocol")
    fixed = {key: value for key, value in batch.items() if torch.is_tensor(value)}
    fixed_hash = hashlib.sha256(
        b"".join(key.encode() + value.detach().cpu().contiguous().numpy().tobytes() for key, value in sorted(fixed.items()))
    ).hexdigest()
    records = []
    for step in range(int(args.steps)):
        completed = 501 + step
        unwrapped.set_token_eru_step(completed)
        unwrapped.set_token_eru_dino_metric_step(completed)
        unwrapped.tsh_instance_loss_weight_eff = 1.0
        unwrapped.tsh_unit_grad_eff = 1.0
        # Keep the source ERU gate schedule unchanged; only the new DINO
        # gate/metric weight are ramped by this audit.
        unwrapped.teacher_lambda_eff = 0.0
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            output = model(batch, compute_quality_metrics=False)
        loss = output["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {completed}")
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        records.append(
            {
                "step": completed,
                "loss": float(loss.detach()),
                "loss_rgb": float(output["loss_rgb"].detach()),
                "loss_instance_group": float(output["loss_instance_group"].detach()),
                "loss_instance_metric": float(output["loss_instance_metric"].detach()),
                "dino_gate": float(output["dino_gate"].detach()),
                "metric_loss_weight_eff": float(output["metric_loss_weight_eff"].detach()),
                "parameter_hash": tensor_hash(unwrapped),
                "finite": True,
            }
        )
    report = {
        "source_checkpoint": str(SOURCE),
        "steps": records,
        "protocol": {"context_views": 8, "target_views": 7, "fixed_batch_hash": fixed_hash},
        "dino_context_only": True,
        "formal_training_started": False,
        "formal_evaluation_started": False,
    }
    (output_dir / "fixed_batch_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
