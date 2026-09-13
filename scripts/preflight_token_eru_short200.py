"""Single-GPU formal-config ERU step-0 and optimizer-boundary preflight."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.options import config_defaults
from tokengs.train import load_model_checkpoint, setup_optimizer


CHECKPOINT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_"
    "w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors"
)
BOTH_CFG = "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
ERU_CFG = "semantic_v6_absolute_units_true_shared_token_eru1_short200_ddp8"
OUTPUT = ROOT / "workspace/token_eru_short200_preflight_v2"


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _max_diff(left, right):
    return float((left.detach().float() - right.detach().float()).abs().max())


def _load(cfg_name, accelerator):
    opt = dataclasses.replace(config_defaults[cfg_name])
    opt.resume = str(CHECKPOINT)
    opt.workspace = str(OUTPUT)
    opt.num_workers = 0
    opt.evaluating = False
    opt.eval_before_training = False
    opt.use_wandb = False
    model = model_registry[opt.model_type](opt).to(accelerator.device)
    load_model_checkpoint(opt, model, accelerator, 0)
    if getattr(opt, "token_eru_enabled", False) and not getattr(
        model, "_token_eru_loaded_from_checkpoint", False
    ):
        model.initialize_token_eru_from_reconstruction()
    return opt, model


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {OUTPUT}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )
    data_opt = dataclasses.replace(config_defaults[ERU_CFG])
    data_opt.num_workers = 0
    data_opt.workspace = str(OUTPUT)
    loader, _, _, _ = get_multi_dataloader(data_opt, accelerator)
    batch = _move(next(iter(loader)), accelerator.device)
    if batch["input"].shape[1] != 15 or batch["instance_label_output"].shape[1] != 7:
        raise RuntimeError("formal preflight did not receive 8+7 data")

    both_opt, both = _load(BOTH_CFG, accelerator)
    eru_opt, eru = _load(ERU_CFG, accelerator)
    both.eval()
    eru.eval()
    eru.set_token_eru_step(0)
    with torch.inference_mode():
        both_out = both(batch, compute_quality_metrics=False)
        eru_out = eru(batch, compute_quality_metrics=False)

    diffs = {
        "rgb_max_diff": _max_diff(both_out["images_pred"], eru_out["images_pred"]),
        "unit_logits_max_diff": _max_diff(both_out["unit_logits"], eru_out["unit_logits"]),
        "rendered_masks_max_diff": _max_diff(
            both_out["rendered_instance_group_probability"],
            eru_out["rendered_instance_group_probability"],
        ),
        "q_abs_max_diff": _max_diff(both._tsh_last_q_abs, eru._tsh_last_q_abs),
        "student_gs_max_diff": _max_diff(
            both._tsh_last_student_gaussians, eru._tsh_last_student_gaussians
        ),
        "r_u_hidden_max_diff": _max_diff(
            eru._token_eru_last_reconstruction_tokens,
            eru._token_eru_last_understanding_tokens,
        ),
        "r_u_units_max_diff": _max_diff(
            eru._tsh_last_q_abs, eru._token_eru_understanding_units
        ),
        "psnr_max_diff": _max_diff(both_out["psnr"], eru_out["psnr"]),
    }
    optimizer = setup_optimizer(eru_opt, eru, accelerator, 0)
    group_info = [
        {"name": group.get("name"), "lr": group["lr"], "parameter_count": sum(p.numel() for p in group["params"])}
        for group in optimizer.param_groups
    ]
    report = {
        "checkpoint": str(CHECKPOINT),
        "batch_shape": list(batch["input"].shape),
        "target_instance_shape": list(batch["instance_label_output"].shape),
        "optimizer_step": 0,
        "gradient_clip": float(eru_opt.gradient_clip),
        "instance_group_scene_level_matching": bool(
            getattr(eru_opt, "instance_group_scene_level_matching", False)
        ),
        "hungarian_calls_per_scene_window": 7,
        "same_3d_query_channels_all_7_views": True,
        "same_gt_matching_all_7_views": False,
        "step0_diffs": diffs,
        "step0_identity": max(diffs.values()) <= 1e-6,
        "optimizer_groups": group_info,
        "fresh_reset": False,
        "pgsr_absent": not any("pgsr" in key.lower() for key in eru.state_dict()),
        "gsi_absent": not any("gsi" in key.lower() for key in eru.state_dict()),
        "ta_riu_absent": not any("ta_riu" in key.lower() for key in eru.state_dict()),
        "query_memory_refiner_absent": not any(
            "query_memory" in key.lower() for key in eru.state_dict()
        ),
    }
    (OUTPUT / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["step0_identity"]:
        raise RuntimeError(f"formal ERU step0 identity failed: {diffs}")


if __name__ == "__main__":
    main()
