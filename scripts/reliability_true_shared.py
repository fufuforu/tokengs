"""Checkpoint init/resume + fresh-seed + 16-batch stability audit."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safetensors.torch import load_file  # noqa: E402

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


class _LocalAccelerator:
    is_main_process = True


def _sha(t: torch.Tensor) -> str:
    b = t.detach().float().cpu().contiguous()
    import hashlib

    return hashlib.sha256(b.numpy().tobytes()).hexdigest()[:16]


def _head_hashes(model) -> dict:
    out = {}
    for name, param in model.named_parameters():
        if name.startswith(
            ("absolute_gs_head.", "tsh_instance_head.")
        ):
            out[name] = _sha(param)
    return out


def _load_heads_strict(model, ckpt, opt):
    abs_state = {
        key.split(".", 1)[1]: value
        for key, value in ckpt.items()
        if key.startswith("absolute_gs_head.")
    }
    res = model.absolute_gs_head.load_state_dict(abs_state, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    if any(key.startswith("tsh_instance_head.") for key in ckpt):
        tsh_state = {
            key.split(".", 1)[1]: value
            for key, value in ckpt.items()
            if key.startswith("tsh_instance_head.")
        }
        res = model.tsh_instance_head.load_state_dict(
            tsh_state, strict=True
        )
        assert not res.missing_keys and not res.unexpected_keys
    else:
        torch.manual_seed((int(opt.seed) + 987654321) % (2**31))
        model.tsh_instance_head.reset_parameters_fresh()
    return ckpt


def _grad_norm_params(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.double().norm().item() ** 2)
    return float(total ** 0.5)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        default="workspace/true_shared_reliability",
    )
    parser.add_argument("--resume-ckpt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-batches", type=int, default=16)
    parser.add_argument(
        "--config-name",
        default="semantic_v6_absolute_units_true_shared_joint_m4",
    )
    args = parser.parse_args()
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    opt = config_defaults[args.config_name]
    opt.workspace = str(out_dir)
    opt.num_workers = 0
    torch.manual_seed(args.seed)
    loader, _, _, _ = get_multi_dataloader(opt, _LocalAccelerator())

    def _fresh_model():
        m = model_registry[opt.model_type](opt).cuda().train()
        return m

    # ---- checkpoint load equivalence --------------------------------
    ckpt = load_file(args.resume_ckpt, device="cpu")
    m1 = _fresh_model()
    _load_heads_strict(m1, ckpt, opt)
    h1 = _head_hashes(m1)
    report["checkpoint"] = {
        "has_tsh_keys": any(
            k.startswith("tsh_instance_head.") for k in ckpt
        ),
        "abs_keys": sum(
            1 for k in ckpt if k.startswith("absolute_gs_head.")
        ),
        "tsh_keys": sum(
            1 for k in ckpt if k.startswith("tsh_instance_head.")
        ),
    }
    # second instance + same-batch output equality
    data = next(iter(loader))
    data = {k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()}
    m2 = _fresh_model()
    _load_heads_strict(m2, ckpt, opt)
    h2 = _head_hashes(m2)
    hash_equal = h1 == h2
    m1.teacher_lambda_eff = 1.0
    m1.tsh_instance_loss_weight_eff = 1.0
    m1.tsh_unit_grad_eff = float(opt.tsh_unit_gradient_multiplier_max)
    m2.teacher_lambda_eff = m1.teacher_lambda_eff
    m2.tsh_instance_loss_weight_eff = m1.tsh_instance_loss_weight_eff
    m2.tsh_unit_grad_eff = m1.tsh_unit_grad_eff
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o1 = m1(data)
        o2 = m2(data)
    output_equal = bool(
        torch.allclose(
            o1["loss"].float(), o2["loss"].float(), atol=1e-3
        )
        and torch.allclose(
            o1["gaussian_group_probabilities"].float(),
            o2["gaussian_group_probabilities"].float(),
            atol=1e-4,
        )
    )
    report["checkpoint"]["two_loads_hash_equal"] = hash_equal
    report["checkpoint"]["two_loads_output_equal"] = output_equal
    assert hash_equal and output_equal

    # ---- fresh seed determinism / completeness ----------------------
    seed_report = {}
    base = _head_hashes(m1)
    m1.tsh_instance_head.reset_parameters_fresh()
    after1 = _head_hashes(m1)
    abs_unchanged_1 = all(
        base[name] == after1[name]
        for name in base
        if name.startswith("absolute_gs_head.")
    )
    m3 = _fresh_model()
    _load_heads_strict(m3, ckpt, opt)
    torch.manual_seed(123456)
    m3.tsh_instance_head.reset_parameters_fresh()
    torch.manual_seed(123456)
    m4 = _fresh_model()
    _load_heads_strict(m4, ckpt, opt)
    torch.manual_seed(123456)
    m4.tsh_instance_head.reset_parameters_fresh()
    same_seed_hashes = _head_hashes(m3) == _head_hashes(m4)
    torch.manual_seed(999)
    m3.tsh_instance_head.reset_parameters_fresh()
    diff_names = [
        name
        for name in _head_hashes(m3)
        if name.startswith("tsh_instance_head.")
        and _head_hashes(m3)[name] != _head_hashes(m4)[name]
    ]
    inproj_changed = any("in_proj" in name for name in diff_names)
    seed_report = {
        "same_seed_same_hash": same_seed_hashes,
        "abs_unchanged_by_reset": abs_unchanged_1,
        "diff_seed_different_params": len(diff_names),
        "in_proj_weight_in_diff": inproj_changed,
        "sample_diff": sorted(diff_names)[:8],
    }
    report["fresh_seed"] = seed_report
    print("[reliability] fresh_seed", json.dumps(seed_report))
    assert same_seed_hashes and abs_unchanged_1
    assert inproj_changed

    # ---- 16-batch stability (full joint, model fixed from ckpt) ----
    m = _fresh_model()
    _load_heads_strict(m, ckpt, opt)
    m.train()
    m.teacher_lambda_eff = 1.0
    m.tsh_instance_loss_weight_eff = 1.0
    m.tsh_unit_grad_eff = float(opt.tsh_unit_gradient_multiplier_max)
    unit_params = [
        p for n, p in m.named_parameters()
        if n.startswith(
            (
                "absolute_gs_head.tok_norm",
                "absolute_gs_head.tok_proj",
                "absolute_gs_head.unit_queries",
                "absolute_gs_head.unit_readout",
            )
        )
    ]
    head_params = [
        p for n, p in m.named_parameters()
        if n.startswith("tsh_instance_head.")
    ]
    all_train = [
        p for p in m.parameters() if p.requires_grad
    ]
    rows = []
    for idx, batch in enumerate(loader):
        if idx >= args.n_batches:
            break
        if "instance_label_output" not in batch:
            continue
        batch = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(batch)
        scene = str(batch.get("scene_name", ["?"])[0])
        loss_inst_w = (
            o["loss_instance_group"]
            * float(opt.tsh_lambda_instance)
            * 1.0
        )
        unit_grads_inst = torch.autograd.grad(
            loss_inst_w, unit_params, retain_graph=True, allow_unused=True
        )
        inst_unit_norm = float(
            sum((g.double().norm().item() ** 2)
                for g in unit_grads_inst if g is not None) ** 0.5
        )
        unit_grads_gt = torch.autograd.grad(
            o["loss_rgb"], unit_params, retain_graph=True,
            allow_unused=True,
        )
        gt_unit_norm = float(
            sum((g.double().norm().item() ** 2)
                for g in unit_grads_gt if g is not None) ** 0.5
        )
        cos = -2.0
        if inst_unit_norm > 0 and gt_unit_norm > 0:
            a = torch.cat(
                [g.flatten().double() for g in unit_grads_inst
                 if g is not None]
            )
            b = torch.cat(
                [g.flatten().double() for g in unit_grads_gt
                 if g is not None]
            )
            cos = float((a @ b) / (a.norm() * b.norm()))
        head_grad = torch.autograd.grad(
            loss_inst_w, head_params, retain_graph=True,
            allow_unused=True,
        )
        head_norm = float(
            sum((g.double().norm().item() ** 2)
                for g in head_grad if g is not None) ** 0.5
        )
        total_grads = torch.autograd.grad(
            loss_inst_w,
            all_train,
            retain_graph=False,
            allow_unused=True,
        )
        total_norm = float(
            sum((g.double().norm().item() ** 2)
                for g in total_grads if g is not None) ** 0.5
        )
        m.zero_grad(set_to_none=True)
        rows.append(
            {
                "scene": scene,
                "gt_unit_norm": gt_unit_norm,
                "inst_unit_norm": inst_unit_norm,
                "r_unit": (
                    inst_unit_norm / gt_unit_norm
                    if gt_unit_norm > 0
                    else None
                ),
                "cosine": cos,
                "tsh_head_norm": head_norm,
                "instance_loss": float(o["loss_instance_group"].detach()),
                "void_share": float(
                    o["pi_unit"][..., -1].mean().detach()
                ),
                "max_group_prob": float(
                    o["pi_unit"][..., :-1].max(-1).values.mean().detach()
                ),
                "total_grad_norm": total_norm,
            }
        )
    r_vals = [r["r_unit"] for r in rows if r["r_unit"] is not None]
    cos_vals = [r["cosine"] for r in rows if r["cosine"] > -2.0]
    pct = lambda x: sorted(x)[int(len(x) * 0.90)] if x else None
    report["stability"] = {
        "rows": rows,
        "r_unit": {
            "min": min(r_vals) if r_vals else None,
            "median": statistics.median(r_vals) if r_vals else None,
            "p90": pct(r_vals),
            "max": max(r_vals) if r_vals else None,
        },
        "cosine": {
            "min": min(cos_vals) if cos_vals else None,
            "median": statistics.median(cos_vals) if cos_vals else None,
            "p90": pct(cos_vals),
            "max": max(cos_vals) if cos_vals else None,
        },
    }
    (out_dir / "reliability_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
