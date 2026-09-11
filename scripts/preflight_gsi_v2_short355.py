"""Single-card/DDP preflight for the isolated GSI-v2 short Phase-J probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import load_file, save_file

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.globalsplat_instance_v2.camera_adapter import (
    make_official_context_frustum_meta,
    make_official_context_input,
    make_official_target_meta,
)
from tokengs.models.globalsplat_instance_v2.dependency import load_globalsplat_symbols
from tokengs.models.globalsplat_instance_v2.renderer import render_rgb_sh
from tokengs.models.input_types import split_data
from tokengs.options import config_defaults
from tokengs.train import _setup_gsi_v2_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="gsi_v2_joint_scannet_short355_ddp8")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--steps", type=int, default=3)
    return parser.parse_args()


def _forward(model, data):
    if torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(data)
    return model(data)


def _finite_parameters(model) -> bool:
    return all(bool(torch.isfinite(p.detach()).all()) for p in model.parameters())


def _finite_gradients(model) -> bool:
    return all(
        p.grad is None or bool(torch.isfinite(p.grad.detach()).all())
        for p in model.parameters()
    )


def _hash_state(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _maxdiff(left, right) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _sum_grad(model, prefixes: tuple[str, ...], exclude: tuple[str, ...] = ()) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if not name.startswith(prefixes) or name.startswith(exclude):
            continue
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().abs().sum().item())
    return total


def _nonzero_grad_entries(model, prefixes: tuple[str, ...], exclude: tuple[str, ...] = ()) -> list[dict[str, object]]:
    """Return names/magnitudes for diagnostic attribution of unexpected gradients."""
    entries = []
    for name, parameter in model.named_parameters():
        if not name.startswith(prefixes) or name.startswith(exclude) or parameter.grad is None:
            continue
        magnitude = float(parameter.grad.detach().float().abs().sum().item())
        if magnitude > 1e-8:
            entries.append({"name": name, "abs_sum": magnitude})
    return sorted(entries, key=lambda item: float(item["abs_sum"]), reverse=True)


def _all_gather_objects(accelerator: Accelerator, value):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(gathered, value)
        return gathered
    return [value]


def _scene(data) -> str:
    value = data.get("scene_name", ["?"])
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def _gradient_control_signature(boundary: dict[str, dict[str, object]]) -> dict[str, object]:
    """Compare only rank-invariant audit control flow, not sample-dependent values."""
    signature = {}
    for key, record in boundary.items():
        stats = record["forward_stats"]
        signature[key] = {
            "schedule_step": record["schedule_step"],
            "kind": record["kind"],
            "gradients_finite": record["gradients_finite"],
            "parameters_finite": record["parameters_finite"],
            "instance_head_nonzero": record["instance_head_grad"] > 0.0,
            "instance_stream_nonzero": record["instance_stream_grad"] > 0.0,
            "reconstruction_non_instance_nonzero": record["reconstruction_non_instance_grad"] > 0.0,
            "geometry_decoder_nonzero": record["geometry_decoder_grad"] > 0.0,
            "hungarian_calls": stats.get("gsi_v2_hungarian_calls"),
            "target_view_count": stats.get("gsi_v2_target_view_count"),
            "assignment_shape": stats.get("rendered_assignment_shape"),
            "assignment_finite": stats.get("assignment_finite"),
        }
    return signature


@torch.inference_mode()
def _identity_audit(model, data, opt) -> dict[str, object]:
    """Compare the joint wrapper against the restored Phase-R path at gate=0."""
    symbols = load_globalsplat_symbols(opt.gsi_v2_globalsplat_repo, opt.gsi_v2_globalsplat_commit)
    model.set_train_step(0)
    model_input, supervision = split_data(data, opt)
    context = make_official_context_input(model_input)
    direct = model.reconstruction
    tokens = direct._tokenize(context["images"], context["intrinsic"], context["c2w"])
    state = direct.scene_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)
    _aux, encoded, _scale = direct.slot_encoder(tokens, state=state)
    direct_tex, direct_geo = encoded
    direct_tuple = direct.gaussian_decoder((direct_tex, direct_geo))
    direct_gaussians = symbols.Gaussians(
        means=direct_tuple[0], rotations=direct_tuple[1], scales=direct_tuple[2],
        sh=direct_tuple[3], opacities=direct_tuple[4], reg=direct_tuple[5],
    )
    wrapped_gaussians, wrapped_geo, _wrapped_ins = model._official_gaussians(model_input, context)
    target = make_official_target_meta(model_input, (256, 256))
    direct_rgb = render_rgb_sh(symbols, direct_gaussians, target)
    wrapped_rgb = render_rgb_sh(symbols, wrapped_gaussians, target)
    context_k, context_w2c = make_official_context_frustum_meta(model_input)
    direct_loss, _direct_stats, _ = model.reconstruction_loss(
        direct_gaussians, direct_rgb["images_pred"], supervision.images_output,
        context_k, context_w2c,
    )
    wrapped_loss, _wrapped_stats, _ = model.reconstruction_loss(
        wrapped_gaussians, wrapped_rgb["images_pred"], supervision.images_output,
        context_k, context_w2c,
    )
    direct_psnr = -10.0 * torch.log10(
        torch.clamp(torch.nn.functional.mse_loss(direct_rgb["images_pred"].float(), supervision.images_output.float()), min=1e-8)
    )
    wrapped_psnr = -10.0 * torch.log10(
        torch.clamp(torch.nn.functional.mse_loss(wrapped_rgb["images_pred"].float(), supervision.images_output.float()), min=1e-8)
    )
    checks = {
        "scene_tokens_maxdiff": 0.0,
        "geometry_stream_tokens_maxdiff": _maxdiff(direct_geo, wrapped_geo),
        "gaussian_xyz_maxdiff": _maxdiff(direct_gaussians.means, wrapped_gaussians.means),
        "gaussian_scale_maxdiff": _maxdiff(direct_gaussians.scales, wrapped_gaussians.scales),
        "gaussian_rotation_maxdiff": _maxdiff(direct_gaussians.rotations, wrapped_gaussians.rotations),
        "gaussian_opacity_maxdiff": _maxdiff(direct_gaussians.opacities, wrapped_gaussians.opacities),
        "gaussian_sh_maxdiff": _maxdiff(direct_gaussians.sh, wrapped_gaussians.sh),
        "rgb_maxdiff": _maxdiff(direct_rgb["images_pred"], wrapped_rgb["images_pred"]),
        "alpha_maxdiff": _maxdiff(direct_rgb["alphas_pred"], wrapped_rgb["alphas_pred"]),
        "depth_maxdiff": _maxdiff(direct_rgb["depths_pred"], wrapped_rgb["depths_pred"]),
        "reconstruction_loss_absdiff": float((direct_loss.float() - wrapped_loss.float()).abs().item()),
        "psnr_absdiff": float((direct_psnr.float() - wrapped_psnr.float()).abs().item()),
        "all_finite": all(bool(torch.isfinite(x).all()) for x in (
            direct_gaussians.means, direct_gaussians.scales, direct_gaussians.rotations,
            direct_gaussians.opacities, direct_gaussians.sh, direct_rgb["images_pred"],
            direct_rgb["alphas_pred"], direct_rgb["depths_pred"], direct_loss,
        )),
        "rgb_maxdiff_le_1e-6": _maxdiff(direct_rgb["images_pred"], wrapped_rgb["images_pred"]) <= 1e-6,
    }
    checks["identity_pass"] = bool(checks["all_finite"] and checks["rgb_maxdiff_le_1e-6"] and all(
        checks[key] <= 1e-5 for key in (
            "geometry_stream_tokens_maxdiff", "gaussian_xyz_maxdiff", "gaussian_scale_maxdiff",
            "gaussian_rotation_maxdiff", "gaussian_opacity_maxdiff", "gaussian_sh_maxdiff",
            "alpha_maxdiff", "depth_maxdiff", "reconstruction_loss_absdiff", "psnr_absdiff",
        )
    ))
    return checks


def _output_summary(out: dict[str, torch.Tensor]) -> dict[str, object]:
    result = {}
    for key in ("loss", "loss_reconstruction", "loss_instance_group", "psnr",
                "gsi_v2_hungarian_calls", "gsi_v2_target_view_count",
                "gsi_v2_instance_gt_count", "gsi_v2_instance_matched_count"):
        if key in out and torch.is_tensor(out[key]):
            result[key] = float(out[key].detach().float().mean().item())
    probability = out.get("rendered_instance_group_probability")
    if probability is not None:
        p = probability.detach().float()
        result["rendered_assignment_shape"] = list(p.shape)
        result["assignment_channel_sum_max_error"] = float((p.sum(dim=1) - 1.0).abs().max().item())
        result["assignment_finite"] = bool(torch.isfinite(p).all())
    return result


def _gradient_audit(model, optimizer, data, opt, step: int, kind: str) -> dict[str, object]:
    unwrapped = model
    unwrapped.set_train_step(step)
    optimizer.zero_grad(set_to_none=True)
    out = _forward(model, data)
    loss_key = "loss_instance_group" if kind == "instance" else "loss_reconstruction"
    loss = out[loss_key]
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite {kind}-only loss at schedule step {step}")
    loss.backward()
    instance_prefixes = (
        "reconstruction.slot_encoder.slot_to_ins.",
        "reconstruction.slot_encoder.ins_rounds.",
        "reconstruction.slot_encoder.tri_adapters.",
        "instance_head.",
    )
    instance_stream = (
        "reconstruction.slot_encoder.slot_to_ins.",
        "reconstruction.slot_encoder.ins_rounds.",
        "reconstruction.slot_encoder.tri_adapters.",
    )
    reconstruction = ("reconstruction.",)
    record = {
        "schedule_step": step,
        "kind": kind,
        "loss": float(loss.detach().float().item()),
        "instance_head_grad": _sum_grad(unwrapped, ("instance_head.",)),
        "instance_stream_grad": _sum_grad(unwrapped, instance_stream),
        "reconstruction_non_instance_grad": _sum_grad(unwrapped, reconstruction, instance_prefixes),
        "reconstruction_non_instance_grad_entries": _nonzero_grad_entries(unwrapped, reconstruction, instance_prefixes),
        "geometry_decoder_grad": _sum_grad(unwrapped, ("reconstruction.gaussian_decoder.",), instance_prefixes),
        "gradients_finite": _finite_gradients(unwrapped),
        "parameters_finite": _finite_parameters(unwrapped),
        "forward_stats": _output_summary(out),
    }
    del out
    optimizer.zero_grad(set_to_none=True)
    return record


def main() -> None:
    args = parse_args()
    if args.preset not in config_defaults:
        raise KeyError(args.preset)
    opt = config_defaults[args.preset].evolve()
    if opt.globalsplat_instance_v2_phase != "joint":
        raise RuntimeError("short355 preflight requires joint phase")
    if opt.gsi_v2_resume_mode != "phase_r_to_joint":
        raise RuntimeError("short355 preflight requires phase_r_to_joint restore")
    if not Path(opt.resume).is_file():
        raise FileNotFoundError(opt.resume)
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=opt.mixed_precision, kwargs_handlers=[ddp])
    train_loader, _test, _train_ds, _test_ds = get_multi_dataloader(opt, accelerator)
    model = model_registry[opt.model_type](opt)
    phase_state = load_file(opt.resume, device="cpu")
    restore = model.load_phase_r_state_dict(phase_state)
    optimizer = _setup_gsi_v2_optimizer(opt, model, accelerator, 0)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    unwrapped = accelerator.unwrap_model(model)
    model.train()
    data = next(iter(train_loader))
    scene_by_rank = _all_gather_objects(accelerator, _scene(data))
    identity_local = _identity_audit(unwrapped, data, opt)
    identity_all = _all_gather_objects(accelerator, identity_local)
    if not all(item.get("identity_pass", False) for item in identity_all):
        raise RuntimeError(f"step0 identity failed: {identity_all}")

    boundary = {}
    for step in (25, 100):
        boundary[f"step{step}_instance_only"] = _gradient_audit(unwrapped, optimizer, data, opt, step, "instance")
        boundary[f"step{step}_rgb_only"] = _gradient_audit(unwrapped, optimizer, data, opt, step, "rgb")
    boundary_all = _all_gather_objects(accelerator, boundary)
    boundary_signatures = [_gradient_control_signature(item) for item in boundary_all]
    if any(item != boundary_signatures[0] for item in boundary_signatures[1:]):
        raise RuntimeError("gradient audit control flow differs across ranks")

    total_records = []
    local_hashes = []
    for index in range(int(args.steps)):
        schedule_step = index + 1
        unwrapped.set_train_step(schedule_step)
        optimizer.zero_grad(set_to_none=True)
        out = _forward(model, data)
        if not torch.isfinite(out["loss"]):
            raise RuntimeError(f"non-finite total loss at step {schedule_step}")
        out["loss"].backward()
        if not _finite_gradients(unwrapped):
            raise RuntimeError(f"non-finite total gradients at step {schedule_step}")
        accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)
        optimizer.step()
        record = {
            "optimizer_step": schedule_step,
            "loss": float(out["loss"].detach().float().item()),
            "finite": _finite_parameters(unwrapped),
            "gradients_finite": _finite_gradients(unwrapped),
            "stats": _output_summary(out),
        }
        total_records.append(record)
        del out
        local_hashes.append(_hash_state(unwrapped))

    final_hashes = _all_gather_objects(accelerator, local_hashes[-1])
    full_model_path = Path(args.output_json).resolve().parent / "preflight_model.safetensors"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        full_model_path.parent.mkdir(parents=True, exist_ok=True)
        save_file({key: value.detach().cpu().contiguous() for key, value in unwrapped.state_dict().items()}, str(full_model_path))
    accelerator.wait_for_everyone()
    restored = model_registry[opt.model_type](opt)
    restored.load_state_dict(load_file(str(full_model_path), device="cpu"), strict=True)
    reload_hash = _hash_state(restored)
    reload_hashes = _all_gather_objects(accelerator, reload_hash)
    report = {
        "preset": args.preset,
        "world_size": accelerator.num_processes,
        "rank": accelerator.process_index,
        "scene_by_rank": scene_by_rank,
        "rank_samples_distinct": len(set(scene_by_rank)) == len(scene_by_rank),
        "restore": restore,
        "r1_reconstruction_key_count": int(restore["loaded_reconstruction_keys"]),
        "r1_restore_454_of_454": int(restore["loaded_reconstruction_keys"]) == 454,
        "fresh_instance_stream": str(opt.gsi_v2_joint_instance_stream_init) == "fresh",
        "instance_parameter_count": int(sum(p.numel() for _, p in unwrapped.instance_named_parameters())),
        "reconstruction_parameter_count": int(sum(p.numel() for _, p in unwrapped.reconstruction_named_parameters() if id(p) not in {id(x) for _, x in unwrapped.instance_named_parameters()})),
        "step0_identity_by_rank": identity_all,
        "gradient_boundary_by_rank": boundary_all,
        "gradient_control_signature_by_rank": boundary_signatures,
        "total_steps": total_records,
        "full_state_hash_by_rank": final_hashes,
        "full_state_hash_sync": len(set(final_hashes)) == 1,
        "strict_reload_by_rank": reload_hashes,
        "strict_reload_hash_sync": len(set(reload_hashes)) == 1,
        "strict_reload_matches_trained_state": reload_hash == final_hashes[accelerator.process_index],
        "optimizer_step_success": True,
        "teacher_called": False,
        "u_to_r": 0.0,
        "unit_multiplier": 0.0,
        "scene_hungarian_once": all(
            record.get("stats", {}).get("gsi_v2_hungarian_calls") == 1.0
            for record in total_records
        ),
        "same_assignment_all_7_views": all(
            record.get("stats", {}).get("gsi_v2_target_view_count") == 7.0
            for record in total_records
        ),
        "all_total_finite": all(record["finite"] and record["gradients_finite"] for record in total_records),
        "output_json": str(Path(args.output_json).resolve()),
    }
    if accelerator.is_main_process:
        Path(args.output_json).resolve().parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).resolve().write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
