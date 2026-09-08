"""Real AdamW-step update measurement for the calibrated MBM joint run.

Because the trainer applies a global gradient clip (norm 1.0) before every
AdamW step, pre-clip norms alone do not describe what actually changes.  For
each real batch this script performs four real, isolated optimizer steps from
the same initialization (params restored between scenarios, fresh optimizer
state per scenario):

    joint   : full joint loss (RGB + teacher-off + instance + U->R)
    rgb     : GT-RGB loss only
    instance: weighted instance loss only
    u2r     : weighted U->R loss only

After clip+step it reports, per LR group, the post-clip grad norm, the true
parameter update norm (||theta_after - theta_before||), and the
update/parameter-norm ratio.  The default tail-LR is 1e-6; since the AdamW
first-step update is proportional to LR (update ~ lr * unit grad), the 3e-6 /
1e-5 numbers are also reported as exact linear scalings of the measured 1e-6
step (and validated in the short trainer comparison).

The U->R formula, mask policy, depth source and True Shared structure are
untouched.  No checkpoint/config is modified.
"""

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
from tokengs.train import setup_optimizer  # noqa: E402


class _LocalAccelerator:
    is_main_process = True

    @staticmethod
    def print(*args, **kwargs):
        print(*args, **kwargs)


def _load_heads_strict(model, ckpt, opt) -> int:
    abs_state = {
        key.split(".", 1)[1]: value
        for key, value in ckpt.items()
        if key.startswith("absolute_gs_head.")
    }
    res = model.absolute_gs_head.load_state_dict(abs_state, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    tsh_keys = 0
    if any(key.startswith("tsh_instance_head.") for key in ckpt):
        tsh_state = {
            key.split(".", 1)[1]: value
            for key, value in ckpt.items()
            if key.startswith("tsh_instance_head.")
        }
        res = model.tsh_instance_head.load_state_dict(tsh_state, strict=True)
        assert not res.missing_keys and not res.unexpected_keys
        tsh_keys = len(tsh_state)
    return tsh_keys


def _param_groups(model):
    return {
        "abs_unit": (
            "absolute_gs_head.tok_norm.",
            "absolute_gs_head.tok_proj.",
            "absolute_gs_head.unit_queries",
            "absolute_gs_head.unit_readout.",
        ),
        "abs_gs_decoder": (
            "absolute_gs_head.center_mlp.",
            "absolute_gs_head.gs_decoder.",
            "absolute_gs_head.slot_emb",
        ),
        "tail": "enc_dec_backbone.decoder_blocks.",
        "tsh": "tsh_instance_head.",
    }


def _grouped_params(model):
    groups = _param_groups(model)
    out = {}
    for label, prefixes in groups.items():
        plist = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if isinstance(prefixes, str):
                hit = name.startswith(prefixes)
            else:
                hit = any(name.startswith(p) for p in prefixes)
            if hit:
                plist.append(param)
        out[label] = plist
    return out


def _group_l2_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.double().norm().item() ** 2)
    return float(total ** 0.5)


def _param_norm(params) -> float:
    total = 0.0
    for p in params:
        total += float(p.detach().double().norm().item() ** 2)
    return float(total ** 0.5)


def _snapshot(params):
    return [p.detach().clone() for p in params]


def _restore(params, snap):
    with torch.no_grad():
        for p, s in zip(params, snap):
            p.copy_(s)


def _update_stats(params, snap):
    total = 0.0
    for p, s in zip(params, snap):
        total += float(((p.detach().double() - s.double()) ** 2).sum().item())
    return float(total ** 0.5)


def _run_scenario(model, result, scenario, opt, grouped, inst_weight, W):
    model.zero_grad(set_to_none=True)
    if scenario == "rgb":
        loss = result["loss_rgb"]
    elif scenario == "instance":
        loss = result["loss_instance_group"] * inst_weight
    elif scenario == "u2r":
        loss = getattr(model, "_tsh_last_mbm_u2r_loss", None)
        if loss is None:
            return None
    else:
        loss = result["loss"]
    loss.backward(retain_graph=(scenario != "joint"))
    total_norm = _group_l2_norm(list(model.parameters()))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer = setup_optimizer(opt, model, _LocalAccelerator(), 0)
    snap = {
        label: _snapshot(params)
        for label, params in grouped.items()
    }
    optimizer.step()
    out = {
        "scenario": scenario,
        "pre_clip_total_norm": total_norm,
    }
    for label, params in grouped.items():
        out[f"post_clip_norm_{label}"] = _group_l2_norm(params)
        out[f"update_norm_{label}"] = _update_stats(params, snap[label])
        out[f"param_norm_{label}"] = _param_norm(params)
        out[f"update_param_ratio_{label}"] = (
            out[f"update_norm_{label}"]
            / max(out[f"param_norm_{label}"], 1e-12)
        )
    return out


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--u2r-weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-batches", type=int, default=16)
    parser.add_argument(
        "--config-name",
        default=(
            "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_ddp8"
        ),
    )
    args = parser.parse_args()
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    W = float(args.u2r_weight)

    opt = config_defaults[args.config_name]
    opt.workspace = str(out_dir)
    opt.num_workers = 0
    opt.tsh_mbm_u2r_weight = W
    opt.tsh_mbm_u2r_warmup_start_step = 0
    opt.tsh_mbm_u2r_warmup_steps = 1
    torch.manual_seed(args.seed)
    loader, _, _, _ = get_multi_dataloader(opt, _LocalAccelerator())

    model = model_registry[opt.model_type](opt).cuda().train()
    ckpt = load_file(args.resume, device="cpu")
    tsh_keys = _load_heads_strict(model, ckpt, opt)
    model.tsh_instance_loss_weight_eff = 1.0
    model.tsh_unit_grad_eff = float(
        getattr(opt, "tsh_unit_gradient_multiplier_max", 0.0)
    )
    model.tsh_mbm_u2r_eff = W
    model.teacher_lambda_eff = 0.0
    grouped = _grouped_params(model)
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    all_snap = _snapshot(all_trainable)
    inst_weight = float(getattr(opt, "tsh_lambda_instance", 0.05)) * 1.0

    rows = []
    iterator = iter(loader)
    for batch_id in range(args.n_batches):
        try:
            data = next(iterator)
        except StopIteration:
            break
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        scene = str(data.get("scene_name", ("?",))[0])
        row = {"batch_id": batch_id, "scene": scene}
        for scenario in ("joint", "rgb", "instance", "u2r"):
            _restore(all_trainable, all_snap)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(data, compute_quality_metrics=False)
            stats = _run_scenario(
                model, result, scenario, opt, grouped, inst_weight, W
            )
            if stats is None:
                continue
            row[scenario] = stats
            print(
                f"[upd] batch={batch_id} scene={scene} "
                f"scenario={scenario} "
                f"unit_ratio={stats['update_param_ratio_abs_unit']:.3e} "
                f"gsdec_ratio={stats['update_param_ratio_abs_gs_decoder']:.3e} "
                f"tail_ratio={stats['update_param_ratio_tail']:.3e} "
                f"tsh_ratio={stats['update_param_ratio_tsh']:.3e}",
                flush=True,
            )
        rows.append(row)
        _restore(all_trainable, all_snap)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    def _summ(field, scenario):
        vals = []
        for r in rows:
            if scenario in r and field in r[scenario]:
                vals.append(r[scenario][field])
        if not vals:
            return None
        ordered = sorted(vals)
        return {
            "min": ordered[0],
            "median": statistics.median(ordered),
            "p90": ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))],
            "max": ordered[-1],
            "mean": statistics.fmean(ordered),
        }

    report = {
        "u2r_weight": W,
        "tail_lr_measured": opt.tsh_mbm_decoder_tail_lr,
        "scaled_tail_lrs": [1e-6, 3e-6, 1e-5],
        "tsh_keys_loaded": tsh_keys,
        "n_batches": len(rows),
        "summary": {},
    }
    for scenario in ("joint", "rgb", "instance", "u2r"):
        if scenario not in rows[0]:
            continue
        for label in ("abs_unit", "abs_gs_decoder", "tail", "tsh"):
            report["summary"][f"{scenario}_update_norm_{label}"] = _summ(
                f"update_norm_{label}", scenario
            )
            report["summary"][f"{scenario}_update_param_ratio_{label}"] = (
                _summ(f"update_param_ratio_{label}", scenario)
            )
        report["summary"][f"{scenario}_preclip_norm"] = _summ(
            "pre_clip_total_norm", scenario
        )
    report["per_batch"] = rows
    out_path = out_dir / f"optimizer_updates_w{W:g}.json"
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"[upd] wrote {out_path} ({len(rows)} batches)")


if __name__ == "__main__":
    _main()
