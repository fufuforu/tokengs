"""Read-only numerical step-0 identity check on the production DataLoader."""
from __future__ import annotations

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
from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.options import config_defaults
from tokengs.train import load_model_checkpoint

BASE_CFG = "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
TA_CFG = "semantic_v6_absolute_units_true_shared_ta_riu_v1_short250_retry_v2_ddp8"
CKPT = ROOT / "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors"


def move(value, device):
    if torch.is_tensor(value): return value.to(device)
    if isinstance(value, dict): return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, list): return [move(v, device) for v in value]
    if isinstance(value, tuple): return tuple(move(v, device) for v in value)
    return value


def set_zero_gate(model):
    for name, value in (
        ("ta_riu_eval_gate_override", 0.0),
        ("ta_riu_eval_geo_gate_override", 0.0),
        ("ta_riu_eval_app_gate_override", 0.0),
        ("ta_riu_gate_eff", 0.0),
        ("ta_riu_geo_gate_eff", 0.0),
        ("ta_riu_app_gate_eff", 0.0),
    ):
        setattr(model, name, value)


def maxdiff(a, b):
    return float((a.detach().float() - b.detach().float()).abs().max().cpu())


def main():
    out = ROOT / "workspace/tsh_ta_riu_v1_short250_identity_audit_v1"
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    opt = dataclasses.replace(config_defaults[TA_CFG])
    opt.resume = str(CKPT); opt.num_workers = 0; opt.batch_size = 1
    opt.evaluating = False; opt.eval_before_training = False; opt.use_wandb = False
    acc = Accelerator(mixed_precision=opt.mixed_precision, dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    loader, _, _, _ = get_multi_dataloader(opt, acc)
    batch = move(next(iter(loader)), acc.device)

    base_opt = dataclasses.replace(config_defaults[BASE_CFG]); base_opt.resume = str(CKPT); base_opt.num_workers = 0; base_opt.evaluating = False; base_opt.use_wandb = False
    torch.manual_seed(base_opt.seed)
    base = model_registry[base_opt.model_type](base_opt)
    load_model_checkpoint(base_opt, base, acc, 0)
    base = base.to(acc.device)
    base.eval()
    with torch.no_grad(), acc.autocast():
        base_out = base(batch, compute_quality_metrics=False)
    base_q = base._tsh_last_q_abs.detach()
    base_gs = base._tsh_last_student_gaussians.detach()

    torch.manual_seed(opt.seed)
    ta = model_registry[opt.model_type](opt)
    load_model_checkpoint(opt, ta, acc, 0)
    ta = ta.to(acc.device)
    set_zero_gate(ta); ta.eval()
    with torch.no_grad(), acc.autocast():
        ta_out = ta(batch, compute_quality_metrics=False)

    diffs = {
        "q_abs_max_diff": maxdiff(ta._tsh_last_q_abs, base_q),
        "student_gs_max_diff": maxdiff(ta._tsh_last_student_gaussians, base_gs),
        "rgb_max_diff": maxdiff(ta_out["images_pred"], base_out["images_pred"]),
        "unit_logits_max_diff": maxdiff(ta_out["unit_logits"], base_out["unit_logits"]),
        "rendered_masks_max_diff": maxdiff(ta_out["rendered_instance_group_probability"], base_out["rendered_instance_group_probability"]),
        "instance_loss_max_diff": maxdiff(ta_out["loss_instance_group"], base_out["loss_instance_group"]),
        "psnr_max_diff": maxdiff(ta_out["psnr"], base_out["psnr"]),
        "joint_base_max_diff": maxdiff(ta_out["ta_riu_joint_gaussians"], ta_out["ta_riu_base_gaussians"]),
    }
    result = {"batch": {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)}, "diffs": diffs, "pass": max(diffs.values()) <= 5e-5}
    (out / "identity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["pass"]: raise SystemExit(1)


if __name__ == "__main__": main()
