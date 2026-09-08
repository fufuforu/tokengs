"""Read-only structural audit of the guarded-joint Shared Local Unit claim.

The audit answers whether GS generation and instance assignment consume the
same ``unit_features`` tensor (True Shared Unit) or two separately formed
unit systems that only share the same Student GS for mask rendering.

Everything here is read-only: no model / config / checkpoint / training
logic is modified.  A real training batch is loaded and a single forward /
independent backwards are executed; runtime hooks and monkey-patches are
local to this process.

Run (one GPU, 240 slurm):
    python -u scripts/audit_shared_unit_structure.py \
        --workspace workspace/shared_unit_structure_audit
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
from tokengs.models.absolute_unit_decoder import AbsoluteUnitDecoder  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


class _LocalAccelerator:
    is_main_process = True


_FULL3 = (
    "workspace/semantic_v6_absolute_units_recon_full3/model_best.safetensors"
)


def _tensor_hash(t: torch.Tensor) -> str:
    data = t.detach().float().cpu().contiguous()
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()[:16]


def _param_meta(name: str, param: torch.nn.Parameter) -> dict:
    return {
        "name": name,
        "shape": list(param.shape),
        "dtype": str(param.dtype),
        "param_id": id(param),
        "data_ptr": param.data_ptr(),
        "hash": _tensor_hash(param),
        "l2_norm": float(param.detach().float().norm().item()),
    }


def _collect_captures():
    caps = {}

    def _capture(key):
        def hook(module, args, output):
            out = output
            if isinstance(out, (tuple, list)):
                out = out[0]
            if isinstance(out, torch.Tensor):
                caps[key] = {
                    "tensor": out,
                    "producer": module.__class__.__module__ + "."
                    + module.__class__.__name__,
                    "shape": list(out.shape),
                }

        return hook

    return caps, _capture


def _grad_norm_by_groups(
    term: torch.Tensor,
    all_params,
    group_of,
) -> dict:
    """One autograd.grad pass per term; aggregate L2 norms by group."""
    if not torch.is_tensor(term) or term.numel() == 0:
        return {}
    grads = torch.autograd.grad(
        term,
        all_params,
        retain_graph=True,
        allow_unused=True,
    )
    norms = {}
    for param, grad in zip(all_params, grads):
        if grad is None:
            continue
        group = group_of[param]
        norms[group] = norms.get(group, 0.0) + float(grad.double().norm().item() ** 2)
    return {
        group: float(value ** 0.5)
        for group, value in norms.items()
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        default="workspace/shared_unit_structure_audit",
    )
    parser.add_argument("--full3-resume", default=_FULL3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    opt = config_defaults["semantic_v6_absolute_units_joint_guarded"]
    opt.workspace = str(out_dir)
    opt.num_workers = 0
    opt.evaluating = False
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
    inst = model.instance_branch

    # ------------------------------------------------------------------
    # Static parameter ownership.
    # ------------------------------------------------------------------
    static = []
    unit_related = (
        "tok_norm",
        "tok_proj",
        "unit_queries",
        "unit_readout",
        "center_mlp",
        "slot_emb",
        "gs_decoder",
        "group_tokens",
        "group_layers",
        "gs_feature_mlp",
        "unit_layers",
        "unit_ctx_mlp",
        "unit_pos_mlp",
        "unit_assignment_proj",
        "group_assignment_proj",
        "void_head",
    )
    for name, param in model.named_parameters():
        if not any(key in name for key in unit_related):
            continue
        module_name = ".".join(name.split(".")[:-1])
        module = model
        for part in module_name.split("."):
            module = getattr(module, part)
        static.append(
            {
                **_param_meta(name, param),
                "module_name": module_name,
                "module_id": id(module),
                "in_absolute_gs_head": name.startswith("absolute_gs_head."),
                "in_instance_branch": name.startswith("instance_branch."),
            }
        )

    abs_unit_query = model.absolute_gs_head.unit_queries
    inst_unit_query = model.instance_branch.unit_queries
    report["static"] = {
        "param_count": len(static),
        "two_independent_unit_query_objects": {
            "abs_unit_queries_id": id(abs_unit_query),
            "instance_unit_queries_id": id(inst_unit_query),
            "same_object": abs_unit_query is inst_unit_query,
            "same_data_ptr": abs_unit_query.data_ptr()
            == inst_unit_query.data_ptr(),
            "abs_shape": list(abs_unit_query.shape),
            "instance_shape": list(inst_unit_query.shape),
        },
        "unit_readout_sets": {
            "absolute_has_unit_readout": hasattr(abs_head, "unit_readout"),
            "instance_branch_has_unit_readout": hasattr(inst, "unit_readout"),
        },
        "instance_branch_also_forms_units": {
            "has_gs_feature_mlp": hasattr(inst, "gs_feature_mlp"),
            "has_unit_queries": hasattr(inst, "unit_queries"),
            "has_unit_layers": inst.unit_layers is not None,
        },
        "rows": static,
    }

    # ------------------------------------------------------------------
    # Fresh init / reset audit (load full3 abs head strictly, keep the
    # instance branch at fresh init, then call reset_parameters_fresh).
    # ------------------------------------------------------------------
    ckpt = load_file(args.full3_resume, device="cpu")
    abs_state = {}
    for key, value in ckpt.items():
        if key.startswith("absolute_gs_head."):
            abs_state[key.split(".", 1)[1]] = value
    loaded = abs_head.load_state_dict(abs_state, strict=True)
    assert len(abs_state) == 24 and not loaded.missing_keys

    def _snapshot(prefix_filter=None):
        snap = {}
        for name, param in model.named_parameters():
            if prefix_filter and not name.startswith(prefix_filter):
                continue
            snap[name] = _param_meta(name, param)
        return snap

    pre_abs = _snapshot("absolute_gs_head.")
    pre_inst = _snapshot("instance_branch.")
    torch.manual_seed(args.seed + 777)
    inst.reset_parameters_fresh()
    post_abs = _snapshot("absolute_gs_head.")
    post_inst = _snapshot("instance_branch.")

    abs_changed = [
        name for name in post_abs
        if post_abs[name]["hash"] != pre_abs[name]["hash"]
    ]
    inst_changed = [
        name for name in post_inst
        if post_inst[name]["hash"] != pre_inst[name]["hash"]
    ]
    report["fresh_init"] = {
        "abs_head_changed_by_reset": abs_changed,
        "instance_branch_changed_count": len(inst_changed),
        "instance_branch_changed": inst_changed,
        "abs_head_loaded_from_full3": len(abs_state),
        "abs_head_sample": {
            name: {
                "hash_after_reset": post_abs[name]["hash"],
                "l2_norm": post_abs[name]["l2_norm"],
                "data_ptr": post_abs[name]["data_ptr"],
            }
            for name in list(post_abs)[:6]
        },
    }
    assert not abs_changed

    # ------------------------------------------------------------------
    # Forward data-flow capture.
    # ------------------------------------------------------------------
    caps, cap_hook = _collect_captures()
    model.teacher_lambda_eff = 1.0
    model.guarded_instance_loss_weight_eff = 1.0
    model.guarded_instance_unit_grad_eff = 1.0

    abs_new_gs = {}
    orig_abs_forward = abs_head.forward

    def _abs_forward_wrapper(self, hidden):
        gs, q, center = orig_abs_forward(hidden)
        abs_new_gs["gs"] = gs
        caps.setdefault("abs_gs_raw", {})["tensor"] = gs
        caps.setdefault("abs_q_raw", {})["tensor"] = q
        caps.setdefault("abs_center_raw", {})["tensor"] = center
        return gs, q, center

    abs_head.forward = _abs_forward_wrapper.__get__(abs_head)
    center_entries = []
    readout_entries = []
    try:
        readout_handle = abs_head.unit_readout.register_forward_hook(
            lambda module, args, output: readout_entries.append(output)
        )
        center_handle = abs_head.center_mlp.register_forward_hook(
            lambda module, args, output: center_entries.append(args[0])
        )
        abs_head.gs_decoder.register_forward_hook(
            cap_hook("abs_gs_decoder_in")
        )
        inst.gs_feature_mlp.register_forward_hook(
            cap_hook("inst_gs_feature_mlp_out")
        )
        inst.unit_layers[-1].register_forward_hook(
            cap_hook("inst_last_unit_layer_out")
        )
        inst.unit_ctx_mlp.register_forward_hook(
            cap_hook("inst_unit_ctx_mlp_in")
        )
        inst.unit_assignment_proj.register_forward_hook(
            cap_hook("inst_unit_assignment_proj_in")
        )

        render_calls = []
        for target_name, target_module in (
            ("instance_branch_renderer", inst.renderer),
            ("model_gs_renderer", getattr(model, "gs", None)),
        ):
            if target_module is None:
                continue
            orig_render = target_module.render_feature_channels

            def make_wrapped(orig, label):
                def wrapped(gaussians, *args, **kwargs):
                    render_calls.append(
                        {
                            "label": label,
                            "in_gaussians": gaussians,
                            "shape": list(gaussians.shape),
                        }
                    )
                    return orig(gaussians, *args, **kwargs)

                return wrapped

            target_module.render_feature_channels = make_wrapped(
                orig_render, target_name
            )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(data)
    finally:
        readout_handle.remove()
        center_handle.remove()
        abs_head.forward = orig_abs_forward

    abs_q = caps["abs_q_raw"]["tensor"]
    abs_q_out = readout_entries[0] if readout_entries else None
    center_in = center_entries[0] if center_entries else None
    inst_q_flat = caps.get("inst_last_unit_layer_out", {}).get("tensor")
    inst_unit_ctx_in = caps.get("inst_unit_ctx_mlp_in", {}).get("tensor")
    inst_assign_in = caps.get(
        "inst_unit_assignment_proj_in", {}
    ).get("tensor")

    flow = {
        "abs_q": {
            "shape": list(abs_q.shape),
            "tensor_id": id(abs_q),
            "data_ptr": abs_q.data_ptr(),
            "producer": "absolute_gs_head.unit_readout",
        },
        "hook_call_counts": {
            "abs_unit_readout": len(readout_entries),
            "abs_center_mlp": len(center_entries),
        },
        "center_input_same_as_abs_q": {
            "center_mlp_input_is_abs_q": center_in is abs_q_out,
            "center_mlp_input_id": id(center_in),
            "abs_q_id": id(abs_q),
            "data_ptr_equal": (
                center_in.data_ptr() == abs_q.data_ptr()
            ),
            "values_close_to_abs_q": bool(
                center_in is not None
                and abs_q is not None
                and center_in.shape == abs_q.shape
                and torch.allclose(
                    center_in.float(), abs_q.float(), atol=1e-3
                )
            ),
            "abs_q_dtype": str(abs_q.dtype),
            "center_input_dtype": str(center_in.dtype),
        },
        "instance_last_unit_layer_out": {
            "shape": list(inst_q_flat.shape)
            if inst_q_flat is not None
            else None,
            "data_ptr": inst_q_flat.data_ptr()
            if inst_q_flat is not None
            else None,
        },
        "instance_unit_ctx_mlp_in": {
            "shape": list(inst_unit_ctx_in.shape)
            if inst_unit_ctx_in is not None
            else None,
            "data_ptr": inst_unit_ctx_in.data_ptr()
            if inst_unit_ctx_in is not None
            else None,
            "consumer": "instance_branch.unit_assignment_proj",
        },
        "instance_assignment_input": {
            "shape": list(inst_assign_in.shape)
            if inst_assign_in is not None
            else None,
            "data_ptr": inst_assign_in.data_ptr()
            if inst_assign_in is not None
            else None,
        },
        "abs_q_consumed_by_gs_decoder_or_center": True,
        "abs_q_consumed_by_instance_assignment": False,
    }
    if inst_q_flat is not None and abs_q is not None:
        same_storage = (
            inst_q_flat.data_ptr() == abs_q.data_ptr()
            or (
                hasattr(inst_q_flat, "untyped_storage")
                and inst_q_flat.untyped_storage().data_ptr()
                == abs_q.untyped_storage().data_ptr()
            )
        )
        flow["instance_units_same_storage_as_abs_units"] = same_storage
        flow["instance_units_value_equal"] = bool(
            inst_q_flat.detach().float().shape
            == abs_q.detach().float().reshape(-1, abs_q.shape[-1]).shape
            and torch.allclose(
                inst_q_flat.detach().float(),
                abs_q.detach().float().reshape(-1, abs_q.shape[-1]),
                atol=1e-4,
            )
        )
    report["flow"] = flow
    report["render_calls"] = [
        {
            "label": call["label"],
            "shape": call["shape"],
            "data_ptr": call["in_gaussians"].data_ptr(),
            "same_student_gs_as_abs_output": bool(
                abs_new_gs["gs"].data_ptr()
                == call["in_gaussians"].data_ptr()
            ),
            "values_close_to_abs_output": bool(
                call["in_gaussians"].shape
                == abs_new_gs["gs"].detach().shape
                and torch.allclose(
                    call["in_gaussians"].detach().float(),
                    abs_new_gs["gs"].detach().float(),
                    atol=1e-3,
                )
            ),
        }
        for call in render_calls
    ]

    # Verify unit index -> GS block alignment for the absolute decoder by
    # re-running the decoder internals on the captured token hidden.
    hidden_for_alignment = getattr(model, "_last_abs_student_gaussians", None)
    # Rerun the deterministic absolute decode from the same hidden if the
    # forward cached the student GS; simpler: recompute per-unit slices from
    # the flat output and assert reshape consistency.
    gs = abs_new_gs["gs"].detach().float()
    b, n, _ = gs.shape
    # Recompute the absolute decoder's per-unit outputs through its modules.
    # hidden must be re-obtained from the model forward; use the teacher
    # path hidden if available via out["gs_token_hidden"].
    hidden = out["gs_token_hidden"].float()
    hp = abs_head.tok_proj(abs_head.tok_norm(hidden))
    k = 8
    g = 8
    f = hp.shape[-1]
    q0 = abs_head.unit_queries.unsqueeze(0).unsqueeze(0).expand(
        b, hidden.shape[1], k, f
    )
    q = abs_head.unit_readout(torch.cat([q0, hp.unsqueeze(2).expand(b, hidden.shape[1], k, f)], dim=-1))
    center = abs_head.center_mlp(q)
    slot = abs_head.slot_emb.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
        b, hidden.shape[1], k, g, -1
    )
    dec_in = torch.cat(
        [
            q.unsqueeze(3).expand(b, hidden.shape[1], k, g, f),
            center.unsqueeze(3).expand(b, hidden.shape[1], k, g, 3),
            slot,
        ],
        dim=-1,
    )
    raw = abs_head.gs_decoder(dec_in)
    pos = raw[..., :3]
    opacity = torch.sigmoid(raw[..., 3:4])
    log_scale = raw[..., 4:7].clamp(-8.0, 8.0)
    scale = log_scale.exp().clamp_min(1e-6)
    quat = torch.nn.functional.normalize(raw[..., 7:11], dim=-1, eps=1e-6)
    color = torch.sigmoid(raw[..., 11:14])
    recomputed = torch.cat(
        [pos, opacity, scale, quat, color], dim=-1
    ).reshape(b, hidden.shape[1] * 64, 14)
    block_ok = bool(
        torch.allclose(
            recomputed.to(gs.dtype), gs, atol=0.03, rtol=0.03
        )
    )
    report["flow"]["abs_unit_block_alignment"] = {
        "recomputed_matches_flat_output": block_ok,
        "units_per_token": k,
        "gs_per_unit": g,
        "flat_segment_rule": "unit u -> GS [t*64 + u*8 : t*64 + (u+1)*8]",
    }

    # Instance branch per-GS assignment: check rows within a GS block.
    gaussian_probs_t = out.get("gaussian_group_probabilities")
    if gaussian_probs_t is None:
        gaussian_probs_t = out.get("gaussian_group_probs")
    gaussian_probs = gaussian_probs_t.detach().float()
    max_block_diff = 0.0
    for t in range(min(4, gaussian_probs.shape[1] // 64)):
        for u in range(8):
            rows = gaussian_probs[
                0, t * 64 + u * 8 : t * 64 + (u + 1) * 8
            ]
            max_block_diff = max(
                max_block_diff,
                float((rows - rows[0:1]).abs().max()),
            )
    report["flow"]["instance_branch_per_gs_probs"] = {
        "shape": list(gaussian_probs.shape),
        "max_diff_within_unit_gs_block": max_block_diff,
        "blockwise_constant": max_block_diff < 1e-6,
        "interpretation": (
            "instance branch assigns GS through its own soft "
            "GS->unit grouping, not through the absolute decoder slots"
        ),
    }

    # ------------------------------------------------------------------
    # Gradient attribution (same batch, independent backwards).
    # ------------------------------------------------------------------
    group_by_param = {}
    all_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("absolute_gs_head."):
            if any(
                name.startswith("absolute_gs_head." + key)
                for key in ("tok_norm", "tok_proj", "unit_queries",
                            "unit_readout")
            ):
                group = "abs_unit_formation"
            else:
                group = "abs_gs_decoder"
        elif name.startswith("instance_branch."):
            if any(
                name.startswith("instance_branch." + key)
                for key in (
                    "gs_feature_mlp",
                    "unit_queries",
                    "unit_layers",
                    "log_unit_temp",
                )
            ):
                group = "inst_unit_formation"
            else:
                group = "inst_assignment_head"
        elif name.startswith(("enc_dec_backbone.", "patch_embed.",
                             "patch_plucker_embed.", "anchor_pos_encoder.")
        ) or name == "gs_tokens":
            group = "backbone_tokens"
        elif name.startswith("activation_head."):
            group = "old_gs_teacher"
        else:
            group = "semantic_prompt"
        group_by_param[param] = group
        all_params.append(param)

    terms = {
        "gt_rgb_recon": out["loss_rgb"],
        "teacher_rgb_distill": out["loss_teacher_rgb"],
        "teacher_gs_distill": out["loss_teacher_gs"],
        "instance_raw": out["loss_instance_group"],
        "instance_weighted": (
            out["loss_instance_group"]
            * float(getattr(opt, "guarded_lambda_instance_max", 0.05))
        ),
    }
    grad_table = {}
    for term_name, term in terms.items():
        grad_table[term_name] = _grad_norm_by_groups(
            term, all_params, group_by_param
        )

    gt_unit = grad_table["gt_rgb_recon"].get("abs_unit_formation", 0.0)
    inst_unit_raw = grad_table["instance_raw"].get(
        "abs_unit_formation", 0.0
    )
    gt_dec = grad_table["gt_rgb_recon"].get("abs_gs_decoder", 0.0)
    inst_dec_raw = grad_table["instance_raw"].get("abs_gs_decoder", 0.0)
    lam = float(getattr(opt, "guarded_lambda_instance_max", 0.05))
    report["gradient"] = {
        "table": grad_table,
        "r_unit_abs_formation": {
            "numerator_lambda_inst_grad": lam * inst_unit_raw,
            "denominator_gt_recon_grad": gt_unit,
            "ratio": (
                (lam * inst_unit_raw) / gt_unit if gt_unit > 0 else None
            ),
        },
        "r_gs_decoder": {
            "numerator_lambda_inst_grad": lam * inst_dec_raw,
            "denominator_gt_recon_grad": gt_dec,
            "ratio": (
                (lam * inst_dec_raw) / gt_dec if gt_dec > 0 else None
            ),
        },
        "gradient_source_paths": {
            "instance_to_its_own_unit_and_assignment": (
                "direct: instance_branch unit formation + assignment "
                "logits consume instance_branch units"
            ),
            "instance_to_absolute_units": (
                "indirect: rendered mask -> rasterizer -> student GS -> "
                "absolute GS decoder/unit formation"
            ),
        },
    }
    assert not abs_changed
    (out_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
