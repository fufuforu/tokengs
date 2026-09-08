"""Smoke + structural audit for the True-Shared guarded joint baseline.

Verifies:
  1. only one unit formation exists (absolute_gs_head);
  2. SharedUnitInstanceHead consumes the same q_abs tensor;
  3. pi_gs is block-constant over each unit's 8 GS;
  4. mask rendering detaches GS geometry (instance->geometry grads == 0);
  5. warm-up / ramp / full phases gate instance->q_abs gradients;
  6. gradient calibration table and r_unit;
  7. no NaN, old head not used for student outputs.

No formal training is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


_FULL3 = (
    "workspace/semantic_v6_absolute_units_recon_full3/model_best.safetensors"
)


def _tensor_hash(t: torch.Tensor) -> str:
    data = t.detach().float().cpu().contiguous()
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()[:16]


def _grad_norm_sum(prefixes) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if any(name.startswith(prefix) for prefix in prefixes):
            total += float(param.grad.double().norm().item() ** 2)
    return float(total ** 0.5)


def _group_grad_norms(model) -> dict:
    groups = {
        "abs_unit_formation": [
            "absolute_gs_head.tok_norm",
            "absolute_gs_head.tok_proj",
            "absolute_gs_head.unit_queries",
            "absolute_gs_head.unit_readout",
        ],
        "abs_gs_decoder": [
            "absolute_gs_head.center_mlp",
            "absolute_gs_head.slot_emb",
            "absolute_gs_head.gs_decoder",
        ],
        "tsh_instance_head": ["tsh_instance_head."],
        "backbone_tokens": [
            "enc_dec_backbone.",
            "patch_embed.",
            "patch_plucker_embed.",
            "anchor_pos_encoder.",
        ],
        "old_gs_teacher": ["activation_head."],
        "semantic_prompt": [
            "prompt_matcher.",
            "semantic_lifting_head.",
            "semantic_projector.",
            "prompt_semantic_adapter.",
            "gaussian_feature_head.",
        ],
    }
    out = {}
    for key, prefixes in groups.items():
        out[key] = _grad_norm_sum(prefixes)
    return out


model = None  # module-level helper uses this


def _main() -> None:
    global model
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        default="workspace/true_shared_unit_smoke",
    )
    parser.add_argument("--full3-resume", default=_FULL3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--full-joint", type=int, default=240)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    opt = config_defaults["semantic_v6_absolute_units_true_shared_joint"]
    opt.workspace = str(out_dir)
    opt.num_workers = 0
    opt.evaluating = False
    opt.tsh_instance_warmup_steps = args.warmup
    opt.tsh_instance_ramp_end_steps = args.full_joint

    train_loader, _, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    data = None
    for candidate in train_loader:
        if "instance_label_output" in candidate:
            data = candidate
            break
    if data is None:
        raise RuntimeError("no batch with instance labels found")
    data = {
        k: (v.cuda() if torch.is_tensor(v) else v)
        for k, v in data.items()
    }

    model = model_registry[opt.model_type](opt).cuda()
    model.train()
    abs_head = model.absolute_gs_head
    tsh_head = model.tsh_instance_head

    # ---- static: single unit formation ------------------------------------
    unit_param_names = [
        name for name, _ in model.named_parameters()
        if "unit_queries" in name or "unit_readout" in name
    ]
    report["static"] = {
        "unit_query_or_readout_params": unit_param_names,
        "legacy_instance_branch_instantiated": (
            getattr(model, "instance_branch", None) is not None
        ),
        "legacy_instance_branch_forbidden_params": [
            name
            for name, _ in model.named_parameters()
            if name.startswith(
                (
                    "instance_branch.gs_feature_mlp",
                    "instance_branch.unit_queries",
                    "instance_branch.unit_layers",
                    "instance_branch.log_unit_temp",
                )
            )
        ],
        "tsh_head_params": sum(
            1 for _ in tsh_head.parameters()
        ),
    }
    assert len(unit_param_names) == 5  # abs unit_queries + readout 2x(w+b)
    assert getattr(model, "instance_branch", None) is None

    # ---- load full3 abs strictly; head fresh ------------------------------
    ckpt = load_file(args.full3_resume, device="cpu")
    abs_state = {
        key.split(".", 1)[1]: value
        for key, value in ckpt.items()
        if key.startswith("absolute_gs_head.")
    }
    loaded = abs_head.load_state_dict(abs_state, strict=True)
    assert len(abs_state) == 24 and not loaded.missing_keys
    pre_hash = {
        name: _tensor_hash(param)
        for name, param in abs_head.named_parameters()
    }
    tsh_pre_hash = {
        name: _tensor_hash(param)
        for name, param in tsh_head.named_parameters()
    }
    torch.manual_seed(args.seed + 777)
    tsh_head.reset_parameters_fresh()
    post_hash = {
        name: _tensor_hash(param)
        for name, param in abs_head.named_parameters()
    }
    tsh_post_hash = {
        name: _tensor_hash(param)
        for name, param in tsh_head.named_parameters()
    }
    report["fresh_reset"] = {
        "abs_head_unchanged": all(
            pre_hash[name] == post_hash[name]
            for name in pre_hash
        ),
        "tsh_head_changed": [
            name for name in tsh_pre_hash
            if tsh_pre_hash[name] != tsh_post_hash[name]
        ],
        "abs_loaded_from_full3": len(abs_state),
    }
    assert report["fresh_reset"]["abs_head_unchanged"]

    # ---- forward capture --------------------------------------------------
    q_records = {}
    abs_gs_records = {}
    render_calls = []
    old_head_calls = {"n": 0}
    orig_abs = abs_head.forward

    def _abs_wrap(self, hidden):
        gs, q, center = orig_abs(hidden)
        q_records["q_abs"] = q
        abs_gs_records["gs"] = gs
        return gs, q, center

    abs_head.forward = _abs_wrap.__get__(abs_head)
    orig_tsh = tsh_head.forward

    def _tsh_wrap(self, q_abs):
        q_records["tsh_input"] = q_abs
        return orig_tsh(q_abs)

    tsh_head.forward = _tsh_wrap.__get__(tsh_head)
    orig_render = model.gs.render_feature_channels

    def _render_wrap(gaussians, features, *args, **kwargs):
        render_calls.append(
            {
                "gaussians": gaussians,
                "features": features,
                "gaussians_requires_grad": gaussians.requires_grad,
                "features_requires_grad": features.requires_grad,
            }
        )
        return orig_render(gaussians, features, *args, **kwargs)

    model.gs.render_feature_channels = _render_wrap
    orig_act = model.activation_head.forward

    def _act_wrap(*a, **k):
        old_head_calls["n"] += 1
        return orig_act(*a, **k)

    model.activation_head.forward = _act_wrap

    model.teacher_lambda_eff = 1.0
    model.tsh_instance_loss_weight_eff = 1.0
    model.tsh_unit_grad_eff = float(
        opt.tsh_unit_gradient_multiplier_max
    )
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(data)
    finally:
        abs_head.forward = orig_abs
        tsh_head.forward = orig_tsh
        model.gs.render_feature_channels = orig_render
        model.activation_head.forward = orig_act

    q_abs = q_records["q_abs"]
    tsh_in = q_records["tsh_input"]
    report["flow"] = {
        "abs_q_shape": list(q_abs.shape),
        "abs_q_data_ptr": q_abs.data_ptr(),
        "tsh_input_is_abs_q": q_abs is tsh_in,
        "tsh_input_data_ptr": tsh_in.data_ptr(),
        "tsh_input_values_equal_abs_q": bool(
            torch.allclose(
                tsh_in.float(), q_abs.float(), atol=1e-6
            )
        ),
        "tsh_input_edge": (
            "grad_scale(q_abs) identity edge (unit_gradient_multiplier)"
            if float(opt.tsh_unit_gradient_multiplier_max) != 1.0
            else "direct q_abs"
        ),
        "student_gs_from_abs": list(abs_gs_records["gs"].shape),
        "render_calls": [
            {
                "idx": i,
                "gaussians_requires_grad": call["gaussians_requires_grad"],
                "features_requires_grad": call["features_requires_grad"],
                "gaussians_ptr": call["gaussians"].data_ptr(),
                "features_shape": list(call["features"].shape),
                "gaussians_close_to_abs_gs": bool(
                    torch.allclose(
                        call["gaussians"].detach().float(),
                        abs_gs_records["gs"].detach().float(),
                        atol=1e-3,
                    )
                ),
            }
            for i, call in enumerate(render_calls)
        ],
        "old_head_total_calls": old_head_calls["n"],
        "teacher_called": bool(model.teacher_called),
        "non_teacher_old_head_calls": max(
            0, old_head_calls["n"] - int(model.teacher_called)
        ),
    }
    assert report["flow"]["tsh_input_values_equal_abs_q"]
    assert report["flow"]["non_teacher_old_head_calls"] == 0

    pi_gs = out["gaussian_group_probabilities"].detach().float()
    b, nt, k, g = 1, 1024, 8, 8
    block = pi_gs.reshape(b, nt, k, g, -1)
    report["flow"]["pi_gs_block_const"] = {
        "max_block_diff": float(
            (block - block[..., :1, :]).abs().max()
        ),
        "shape": list(pi_gs.shape),
    }
    assert float((block - block[..., :1, :]).abs().max()) == 0.0

    # ---- phase gradient audit -------------------------------------------
    def run_phase(label: str, step: int):
        head_eff, unit_eff = model.compute_tsh_effs(step, opt)
        model.tsh_instance_loss_weight_eff = head_eff
        model.tsh_unit_grad_eff = unit_eff
        model.teacher_lambda_eff = 1.0
        snap = {}

        def _make_scalar(kind: str):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(data)
            if kind == "instance_weighted":
                return (
                    o["loss_instance_group"]
                    * float(opt.tsh_lambda_instance)
                    * head_eff
                )
            if kind == "gt_rgb":
                return o["loss_rgb"]
            return o["loss_instance_group"]

        for term_name in ("instance_weighted", "gt_rgb", "instance_raw"):
            model.zero_grad(set_to_none=True)
            term = _make_scalar(term_name)
            term.backward(retain_graph=False)
            snap[term_name] = _group_grad_norms(model)
        return {
            "step": int(step),
            "head_eff": float(head_eff),
            "unit_eff": float(unit_eff),
            "grads": snap,
        }

    phases = {}
    for label, step in (
        ("warmup", args.warmup // 2),
        ("ramp", args.warmup + (args.full_joint - args.warmup) // 2),
        ("full_joint", args.full_joint + 1),
    ):
        phases[label] = run_phase(label, step)
    report["phases"] = phases

    w = phases["warmup"]["grads"]["instance_weighted"]
    f = phases["full_joint"]["grads"]["instance_weighted"]
    f_full = phases["full_joint"]["grads"]
    assert w["abs_unit_formation"] == 0.0
    assert w["abs_gs_decoder"] == 0.0
    assert f["abs_unit_formation"] > 0.0
    assert f["abs_gs_decoder"] == 0.0
    assert f["tsh_instance_head"] > 0.0
    assert f_full["gt_rgb"]["abs_unit_formation"] > 0
    assert f_full["gt_rgb"]["abs_gs_decoder"] > 0
    for key in ("backbone_tokens", "old_gs_teacher", "semantic_prompt"):
        assert f[key] == 0.0 and f_full["gt_rgb"][key] == 0.0

    # calibration at full joint
    unit_recon = f_full["gt_rgb"]["abs_unit_formation"]
    unit_inst = f["abs_unit_formation"]
    dec_recon = f_full["gt_rgb"]["abs_gs_decoder"]
    dec_inst = f["abs_gs_decoder"]
    report["calibration"] = {
        "tsh_lambda_instance": float(opt.tsh_lambda_instance),
        "unit_gradient_multiplier_max": float(
            opt.tsh_unit_gradient_multiplier_max
        ),
        "r_unit": (
            unit_inst / unit_recon if unit_recon > 0 else None
        ),
        "r_gs_decoder": (
            dec_inst / dec_recon if dec_recon > 0 else None
        ),
        "unit_grad_norms": {
            "instance_weighted": unit_inst,
            "gt_rgb": unit_recon,
        },
    }
    report["finite"] = {
        "loss_total": bool(torch.isfinite(out["loss"]).all()),
        "student_gs": bool(
            torch.isfinite(abs_gs_records["gs"]).all()
        ),
        "pi_gs": bool(torch.isfinite(pi_gs).all()),
    }
    assert report["finite"]["loss_total"]
    assert report["finite"]["student_gs"]
    assert report["finite"]["pi_gs"]

    (out_dir / "smoke_and_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
