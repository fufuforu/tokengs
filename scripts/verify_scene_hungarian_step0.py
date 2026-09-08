"""Compare Both@1420 and scene-Hungarian forward outputs on one real batch."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

torch._dynamo.config.disable = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import load_model_checkpoint  # noqa: E402


class _A:
    is_main_process = True

    @staticmethod
    def print(*args, **kwargs):
        print(*args, **kwargs)


CKPT = (
    "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_"
    "t3e6_ddp8/checkpoints/model_step_001420.safetensors"
)
BASE = "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
SCENE = "semantic_v6_absolute_units_true_shared_siu3r_mbm_scene_hungarian_ddp8"


def _run(opt_name, data, ckpt):
    opt = config_defaults[opt_name]
    opt.workspace = "workspace/tsh_scene_hungarian_step0_verify"
    opt.num_workers = 0
    model = model_registry[opt.model_type](opt).cuda().eval()
    opt.resume = ckpt
    load_model_checkpoint(opt, model, _A(), 0)
    captured = {}

    def hook(_module, _inputs, output):
        captured["q_abs"] = output[1].detach().float().cpu()

    handle = model.absolute_gs_head.register_forward_hook(hook)
    gpu_data = {k: v.cuda() if torch.is_tensor(v) else v for k, v in data.items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(gpu_data, compute_quality_metrics=False)
    handle.remove()
    result = {
        **captured,
        "gaussians": out["gaussians"].detach().float().cpu(),
        "images_pred": out["images_pred"].detach().float().cpu(),
        "unit_logits": out["unit_logits"].detach().float().cpu(),
        "rendered_instance_group_probability": out[
            "rendered_instance_group_probability"
        ].detach().float().cpu(),
        "psnr": float(out["psnr"]),
    }
    del out, model, gpu_data
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="workspace/tsh_scene_hungarian_step0_verify")
    ap.add_argument("--resume", default=CKPT)
    args = ap.parse_args()
    opt = config_defaults[BASE]
    opt.workspace = args.workspace
    opt.num_workers = 0
    loader, _, _, _ = get_multi_dataloader(opt, _A())
    data = next(iter(loader))
    a = _run(BASE, data, args.resume)
    b = _run(SCENE, data, args.resume)
    diffs = {}
    for key in ("q_abs", "gaussians", "images_pred", "unit_logits",
                "rendered_instance_group_probability"):
        diffs[key] = float((a[key] - b[key]).abs().max())
    report = {
        "resume": args.resume,
        "baseline_config": BASE,
        "scene_config": SCENE,
        "same_forward_batch": True,
        "max_abs_diffs": diffs,
        "allclose_1e-5": all(v <= 1e-5 for v in diffs.values()),
        "psnr_baseline": a["psnr"],
        "psnr_scene": b["psnr"],
    }
    Path(args.workspace).mkdir(parents=True, exist_ok=True)
    path = Path(args.workspace) / "verify_scene_hungarian_step0.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
