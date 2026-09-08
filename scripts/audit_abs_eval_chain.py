"""Audit the absolute-student LSM evaluation chain (no training).

For one checkpoint (or two, to compare), this loads the model exactly like
``eval_instance_lsm_protocol.py`` does and prints:
  * absolute_gs_head load status (loaded/missing), parameter norm/hash;
  * gaussians_source (must be absolute_student at eval);
  * old GS activation-head call count during the eval forward (must be 0);
  * per-scene RGB PSNR from the STUDENT Gaussians;
  * student-vs-old-base8k GS parameter difference on the same scene;
  * a causal perturbation check: zeroing the unit decoder changes both the
    rendered RGB and the rendered instance mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402

from eval_instance_lsm_protocol import (  # noqa: E402
    _LocalAccelerator,
    _load_checkpoint_arch,
)


def _hash_state_dict(state, prefix: str = "") -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        if key.startswith(prefix):
            digest.update(key.encode())
            digest.update(state[key].cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def _param_norm(model) -> float:
    total = 0.0
    for _name, param in model.absolute_gs_head.named_parameters():
        total += float(param.detach().float().abs().sum())
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument(
        "--lsm_manifest",
        default=(
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "lsm_instance_eval_manifest.json"
        ),
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    Path(args.workspace).mkdir(parents=True, exist_ok=True)

    from safetensors.torch import load_file

    resume_state = load_file(args.resume, device="cpu")
    abs_keys = [
        key for key in resume_state if key.startswith("absolute_gs_head.")
    ]
    report = {
        "resume": args.resume,
        "absolute_gs_head_keys_in_ckpt": len(abs_keys),
        "ckpt_abs_hash": _hash_state_dict(resume_state, "absolute_gs_head."),
    }

    a = SimpleNamespace(
        resume=args.resume,
        workspace=args.workspace,
        model_type="semantic_tokengs_v6",
        num_groups=100,
        max_scenes=1,
        lsm_manifest=args.lsm_manifest,
        num_input_views=8,
        num_views=15,
        ttt_steps=0,
        min_pred_pixels=1,
        min_gt_pixels=1,
        max_predictions_per_image=100,
        backbone_resume="",
        instance_branch_cluster_eps=None,
        instance_branch_cluster_pos_weight=None,
        instance_branch_void_fg_share=None,
        prune_top_k=0,
        ttt_lr=1e-3,
    )
    opt = config_defaults["eval_scannet_lsm_instance"]
    opt.model_type = a.model_type
    opt.resume = a.resume
    opt.workspace = args.workspace
    opt.experiment_name = Path(args.workspace).name
    opt.instance_group_num_groups = int(a.num_groups)
    opt.num_input_views = int(a.num_input_views)
    opt.num_views = int(a.num_views)
    opt.dataset_kwargs = {
        **dict(opt.dataset_kwargs or {}),
        "lsm_manifest_path": args.lsm_manifest,
    }
    _load_checkpoint_arch(a, opt)

    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    model = model_registry[opt.model_type](opt)
    torch.nn.Module.load_state_dict(model, resume_state, strict=False)
    if not any(key.startswith("enc_dec_backbone.") for key in resume_state):
        backbone_path = str(getattr(opt, "backbone_resume", "") or "")
        if backbone_path and Path(backbone_path).is_file():
            backbone_ckpt = load_file(backbone_path, device="cpu")
            frozen_prefixes = (
                "enc_dec_backbone.",
                "patch_embed.",
                "patch_plucker_embed.",
                "activation_head.",
                "anchor_pos_encoder.",
            )
            native_state = torch.nn.Module.state_dict(model)
            loadable = {
                key: value
                for key, value in backbone_ckpt.items()
                if (key.startswith(frozen_prefixes) or key == "gs_tokens")
                and key in native_state
                and native_state[key].shape == value.shape
            }
            torch.nn.Module.load_state_dict(model, loadable, strict=False)
            report["backbone_loaded_keys"] = len(loadable)
            report["backbone_path"] = backbone_path
    model.eval()
    model = model.cuda()

    abs_mode = (
        bool(getattr(opt, "instance_branch_abs_units", False))
        and hasattr(model, "absolute_gs_head")
    )
    report["gaussians_source"] = (
        "absolute_student" if abs_mode else "old_gs_head"
    )
    if abs_mode:
        native = {
            name: param
            for name, param in model.absolute_gs_head.named_parameters()
        }
        loaded = 0
        missing = []
        for name in native:
            key = f"absolute_gs_head.{name}"
            if key in resume_state and resume_state[key].shape == native[name].shape:
                loaded += 1
            else:
                missing.append(name)
        report["abs_loaded"] = f"{loaded}/{len(native)}"
        report["abs_missing"] = missing[:5]
        report["abs_param_norm"] = _param_norm(model)
        report["model_abs_hash"] = _hash_state_dict(
            model.absolute_gs_head.state_dict(), ""
        )

    data = next(iter(test_loader))
    data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in data.items()}

    # Teacher (old head) GS for the same scene, computed deliberately BEFORE
    # the call counter is armed.
    model_input = None
    from tokengs.models.input_types import split_data

    model_input, _ = split_data(data, opt)
    with torch.no_grad():
        teacher_recon, _, _ = model._forward_prompt_reconstruction(model_input)
    teacher_gs = teacher_recon.gaussians.detach().float()

    old_head_calls = 0
    if abs_mode:
        _orig = model.activation_head.forward

        def _counting(*args, **kwargs):
            nonlocal old_head_calls
            old_head_calls += 1
            return _orig(*args, **kwargs)

        model.activation_head.forward = _counting

    def _forward():
        with torch.inference_mode():
            return model(data, compute_quality_metrics=True)

    out = _forward()
    student_gs = out["gaussians"].detach().float()
    report["old_gs_head_calls_eval"] = old_head_calls
    report["psnr_student"] = float(out["psnr"].detach())
    pos_diff = (student_gs[..., :3] - teacher_gs[..., :3]).abs()
    report["student_vs_teacher_pos_maxdiff"] = float(pos_diff.max())
    report["student_vs_teacher_gs_meandiff"] = float(
        (student_gs - teacher_gs).abs().mean()
    )
    mask_ref = out["rendered_instance_group_probability"].detach()

    if args.perturb and abs_mode:
        saved = {
            name: param.detach().clone()
            for name, param in model.absolute_gs_head.named_parameters()
        }
        with torch.no_grad():
            for name, param in model.absolute_gs_head.named_parameters():
                if "gs_decoder" in name:
                    param.zero_()
        out2 = _forward()
        report["psnr_student_zeroed"] = float(out2["psnr"].detach())
        report["psnr_changed_after_zero"] = (
            report["psnr_student_zeroed"] != report["psnr_student"]
        )
        mask_new = out2["rendered_instance_group_probability"].detach()
        report["mask_changed_after_zero"] = float(
            (mask_ref - mask_new).abs().max()
        )
        with torch.no_grad():
            for name, param in model.absolute_gs_head.named_parameters():
                param.copy_(saved[name])

    (Path(args.workspace) / "audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
