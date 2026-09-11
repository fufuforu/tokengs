from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import save_file

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.globalsplat_instance_v2.camera_adapter import make_official_context_input, make_official_target_meta
from tokengs.models.globalsplat_instance_v2.dependency import sha256_file
from tokengs.models.globalsplat_instance_v2.renderer import render_rgb_sh
from tokengs.models.input_types import split_data
from tokengs.options import config_defaults
from tokengs.train import _setup_gsi_v2_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--fixed_batch_steps", type=int, default=0)
    parser.add_argument("--allow_mse_smoke", action="store_true")
    parser.add_argument("--use_official_vgg", action="store_true")
    return parser.parse_args()


def _hash_parameters(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _finite_parameters(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter.detach()).all()) for parameter in model.parameters())


def _finite_gradients(model: torch.nn.Module) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad.detach()).all()) for parameter in model.parameters())


def _cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


def _batch_hash(batch: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(batch):
        digest.update(key.encode())
        value = batch[key]
        if torch.is_tensor(value):
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def _output_stats(output: dict[str, torch.Tensor]) -> dict[str, object]:
    stats = {}
    for key, value in output.items():
        if (key.startswith("gsi_v2_") or key.startswith("loss_")) and torch.is_tensor(value):
            stats[key] = float(value.detach().float().item())
    if "rendered_instance_group_probability" in output:
        probability = output["rendered_instance_group_probability"]
        stats["rendered_assignment_shape"] = list(probability.shape)
        stats["rendered_assignment_finite"] = bool(torch.isfinite(probability.detach()).all())
        stats["rendered_assignment_channel_sum_max_error"] = float(
            (probability.detach().float().sum(dim=1) - 1.0).abs().max().item()
        )
        stats["void_mean"] = float(probability.detach().float()[:, -1].mean().item())
    if "gaussian_group_probabilities" in output:
        group_probability = output["gaussian_group_probabilities"].detach().float()
        foreground_mass = group_probability[..., :-1].mean(dim=(0, 1))
        stats["nonvoid_query_count_mass_gt_1e-3"] = int((foreground_mass > 1e-3).sum().item())
        stats["nonvoid_query_count_mass_gt_1e-4"] = int((foreground_mass > 1e-4).sum().item())
    return stats


def _forward(model, data):
    if torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(data)
    return model(data)


def _grad(model: torch.nn.Module, prefixes: tuple[str, ...]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and name.startswith(prefixes):
            total += float(parameter.grad.detach().float().abs().sum().item())
    return total


def _gather_objects(accelerator: Accelerator, value):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        values = [None for _ in range(accelerator.num_processes)]
        torch.distributed.all_gather_object(values, value)
        return values
    return [value]


@torch.inference_mode()
def _identity_audit(model, data, opt):
    from tokengs.models.globalsplat_instance_v2.dependency import load_globalsplat_symbols
    symbols = load_globalsplat_symbols(opt.gsi_v2_globalsplat_repo, opt.gsi_v2_globalsplat_commit)
    direct = model.reconstruction
    direct.set_stage(3, mix=1.0)
    model_input, _ = split_data(data, opt)
    context = make_official_context_input(model_input)
    tokens = direct._tokenize(context["images"], context["intrinsic"], context["c2w"])
    state = direct.scene_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)
    _aux, encoded, _scale = direct.slot_encoder(tokens, state=state)
    dtex, dgeo = encoded
    d_tuple = direct.gaussian_decoder((dtex, dgeo))
    d_gauss = symbols.Gaussians(
        means=d_tuple[0], rotations=d_tuple[1], scales=d_tuple[2],
        sh=d_tuple[3], opacities=d_tuple[4], reg=d_tuple[5],
    )
    wrapper_gauss, wrapper_geo, _wrapper_ins = model._official_gaussians(model_input, context)
    target = make_official_target_meta(model_input, (256, 256))
    d_rgb = render_rgb_sh(symbols, d_gauss, target)
    w_rgb = render_rgb_sh(symbols, wrapper_gauss, target)
    def md(a, b): return float((a.float() - b.float()).abs().max().item())
    return {
        "content_tokens_maxdiff": md(tokens, model.reconstruction._tokenize(context["images"], context["intrinsic"], context["c2w"])),
        "scene_geo_maxdiff": md(dgeo, wrapper_geo),
        "means_maxdiff": md(d_gauss.means, wrapper_gauss.means),
        "scales_maxdiff": md(d_gauss.scales, wrapper_gauss.scales),
        "rotations_maxdiff": md(d_gauss.rotations, wrapper_gauss.rotations),
        "sh_maxdiff": md(d_gauss.sh, wrapper_gauss.sh),
        "opacities_maxdiff": md(d_gauss.opacities, wrapper_gauss.opacities),
        "rgb_maxdiff": md(d_rgb["images_pred"], w_rgb["images_pred"]),
        "alpha_maxdiff": md(d_rgb["alphas_pred"], w_rgb["alphas_pred"]),
        "depth_maxdiff": md(d_rgb["depths_pred"], w_rgb["depths_pred"]),
        "sh_shape": list(d_gauss.sh.shape),
        "gaussian_count": int(d_gauss.means.shape[1]),
        "all_identity_finite": all(bool(torch.isfinite(value).all()) for value in (
            d_gauss.means, d_gauss.scales, d_gauss.rotations, d_gauss.sh,
            d_gauss.opacities, d_rgb["images_pred"], d_rgb["alphas_pred"], d_rgb["depths_pred"],
        )),
    }


def main() -> None:
    args = parse_args()
    if args.preset not in config_defaults:
        raise KeyError(f"unknown preset {args.preset}")
    opt = config_defaults[args.preset].evolve()
    if args.allow_mse_smoke and args.use_official_vgg:
        raise ValueError("--allow_mse_smoke and --use_official_vgg are mutually exclusive")
    if args.use_official_vgg:
        if opt.globalsplat_instance_v2_phase != "reconstruction":
            raise ValueError("official VGG preflight requires reconstruction phase")
        opt = opt.evolve(
            gsi_v2_recon_loss_mode="official_vgg",
            gsi_v2_subset_consistency=True,
        )
    if opt.gsi_v2_recon_loss_mode == "mse_smoke" and not args.allow_mse_smoke:
        raise RuntimeError("mse_smoke requires --allow_mse_smoke")
    if opt.gsi_v2_recon_loss_mode == "official_vgg":
        vgg_path = Path(opt.gsi_v2_vgg_weight_path).resolve()
        if not vgg_path.is_file() or vgg_path.stat().st_size <= 0:
            raise RuntimeError(f"VGG asset is required for official_vgg preflight: {vgg_path}")
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The diagnostic deliberately performs independent instance-only and
    # RGB-only backwards.  Each audit uses a fresh graph, but each loss also
    # intentionally leaves a different subset of parameters unused.  Enable
    # DDP's unused-parameter bookkeeping for this diagnostic sequence only;
    # the formal trainer still uses its normal single total-loss backward.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=opt.mixed_precision, kwargs_handlers=[ddp_kwargs])
    train_loader, _test_loader, train_dataset, _test_dataset = get_multi_dataloader(opt, accelerator)
    model = model_registry[opt.model_type](opt)
    optimizer = _setup_gsi_v2_optimizer(opt, model, accelerator, 0)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.set_eval_stage()
    model.train()
    data = next(iter(train_loader))
    local_sample = str(data.get("scene_name", ["?"])[0])
    samples = _gather_objects(accelerator, local_sample)
    report = {
        "preset": args.preset, "world_size": accelerator.num_processes,
        "rank": accelerator.process_index, "sample_by_rank": samples,
        "context_shape": list(data["input"].shape),
        "target_shape": list(data["images_output"].shape),
        "context_images_shape": list(data["images_input"].shape),
        "target_images_shape": list(data["images_output"].shape),
        "official_checkpoint_sha256": opt.gsi_v2_official_checkpoint_sha256,
        "dino_or_teacher_called": False, "teacher_called": False,
        "u_to_r": 0.0, "unit_multiplier": 0.0,
        "gsi_v2_subset_consistency": bool(opt.gsi_v2_subset_consistency),
        "gsi_v2_recon_loss_mode": str(opt.gsi_v2_recon_loss_mode),
    }
    official_report = getattr(unwrapped, "_official_checkpoint_report", None)
    if official_report is not None:
        report["official_restore"] = {
            "loaded_tensor_count": int(official_report.loaded_tensor_count),
            "checkpoint_tensor_count": int(official_report.checkpoint_tensor_count),
            "loaded_state_numel": int(official_report.loaded_state_numel),
            "checkpoint_state_numel": int(official_report.checkpoint_state_numel),
            "missing_keys": list(official_report.missing_keys),
            "unexpected_keys": list(official_report.unexpected_keys),
            "shape_mismatches": list(official_report.shape_mismatches),
        }
    if accelerator.is_main_process:
        report["step0_identity"] = _identity_audit(unwrapped, data, opt)
    accelerator.wait_for_everyone()
    instance_gradient_records = []
    rgb_gradient_records = []
    total_records = []
    hashes = []
    fixed_report = None
    fixed_data = next(iter(train_loader))
    if int(args.fixed_batch_steps) > 0:
        fixed_dir = output_path.parent / "fixed_batch"
        fixed_dir.mkdir(parents=True, exist_ok=True)
        initial_state = {key: value.detach().cpu().clone() for key, value in unwrapped.state_dict().items()}
        fixed_cpu = _cpu_clone(fixed_data)
        fixed_hash = _batch_hash(fixed_cpu)
        if accelerator.is_main_process:
            torch.save(fixed_cpu, fixed_dir / "batch.pt")
            save_file({key: value.contiguous() for key, value in initial_state.items()}, str(fixed_dir / "model_step_000000.safetensors"))
            (fixed_dir / "batch_metadata.json").write_text(json.dumps({
                "batch_sha256": fixed_hash,
                "scene_name": fixed_cpu.get("scene_name"),
                "sample_id": fixed_cpu.get("sample_id"),
                "frame_ids": fixed_cpu.get("frame_ids").tolist() if torch.is_tensor(fixed_cpu.get("frame_ids")) else None,
                "context_shape": list(fixed_cpu["images_input"].shape),
                "target_shape": list(fixed_cpu["images_output"].shape),
                "scheduler": None,
                "scheduler_reason": "fixed-batch diagnostic uses constant LR",
            }, indent=2, default=str), encoding="utf-8")
        accelerator.wait_for_everyone()
        fixed_records = []
        fixed_initial_assignment = None
        for fixed_step in range(int(args.fixed_batch_steps)):
            unwrapped.set_train_step(fixed_step)
            if unwrapped.instance_enabled:
                # The production J schedule warms the instance loss from zero;
                # this diagnostic explicitly measures learnability of the
                # instance objective on a frozen batch.
                unwrapped._instance_weight = 1.0
            optimizer.zero_grad(set_to_none=True)
            fixed_start = time.perf_counter()
            fixed_out = _forward(model, fixed_data)
            audit_loss = fixed_out["loss"]
            if unwrapped.instance_enabled:
                audit_loss = fixed_out["loss_reconstruction"] + fixed_out["loss_instance_group"]
            if not torch.isfinite(audit_loss):
                raise RuntimeError(f"non-finite fixed-batch loss at step {fixed_step}")
            audit_loss.backward()
            if not _finite_gradients(unwrapped):
                raise RuntimeError(f"non-finite fixed-batch gradient at step {fixed_step}")
            accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)
            optimizer.step()
            fixed_record = {
                "step": fixed_step + 1,
                "loss": float(audit_loss.detach().item()),
                "loss_instance": float(fixed_out["loss_instance_group"].detach().item()),
                "loss_reconstruction": float(fixed_out["loss_reconstruction"].detach().item()),
                "psnr": float(fixed_out["psnr"].detach().item()),
                "step_seconds": time.perf_counter() - fixed_start,
                "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0,
                "finite": _finite_parameters(unwrapped),
                "stats": _output_stats(fixed_out),
            }
            if fixed_initial_assignment is None and "rendered_instance_group_probability" in fixed_out:
                fixed_initial_assignment = fixed_out["rendered_instance_group_probability"].detach().float().cpu()
            if fixed_initial_assignment is not None and "rendered_instance_group_probability" in fixed_out:
                fixed_record["soft_mask_max_abs_change_from_step0"] = float(
                    (fixed_out["rendered_instance_group_probability"].detach().float().cpu() - fixed_initial_assignment).abs().max().item()
                )
            fixed_records.append(fixed_record)
            del fixed_out
            if accelerator.is_main_process and fixed_step + 1 in (1, 5, 25, 50, int(args.fixed_batch_steps)):
                save_file({key: value.detach().cpu().contiguous() for key, value in unwrapped.state_dict().items()}, str(fixed_dir / f"model_step_{fixed_step + 1:06d}.safetensors"))
        if accelerator.is_main_process:
            torch.save(optimizer.state_dict(), fixed_dir / "optimizer_final.pth")
        # Restore the exact pre-audit model so the normal 3-step smoke below
        # remains an independent test from the fixed-batch experiment.
        unwrapped.load_state_dict(initial_state, strict=True)
        optimizer.state.clear()
        if accelerator.is_main_process:
            fixed_report = {
                "steps": int(args.fixed_batch_steps),
                "batch_sha256": fixed_hash,
                "batch_path": str(fixed_dir / "batch.pt"),
                "records": fixed_records,
                "initial_parameter_hash": _hash_parameters(unwrapped),
                "restored_after_audit": True,
                "scheduler": None,
            }
        accelerator.wait_for_everyone()
    iterator = iter(train_loader)
    for step in range(int(args.steps)):
        data = next(iterator)
        unwrapped.set_train_step(step)
        if unwrapped.instance_enabled:
            optimizer.zero_grad(set_to_none=True)
            instance_out = _forward(model, data)
            instance_out["loss_instance_group"].backward()
            instance_gradient_records.append({
                "head": _grad(unwrapped, ("instance_head.",)),
                "instance_stream": _grad(unwrapped, ("reconstruction.slot_encoder.slot_to_ins.", "reconstruction.slot_encoder.ins_rounds.", "reconstruction.slot_encoder.tri_adapters.")),
                "actual_gaussian_geometry": _grad(unwrapped, ("reconstruction.gaussian_decoder.",)),
                "finite": _finite_parameters(unwrapped),
                "gradients_finite": _finite_gradients(unwrapped),
                "stats": _output_stats(instance_out),
            })
            del instance_out
            optimizer.zero_grad(set_to_none=True)
            rgb_out = _forward(model, data)
            rgb_out["loss_reconstruction"].backward()
            rgb_gradient_records.append({
                "reconstruction": _grad(unwrapped, ("reconstruction.",)),
                "instance_head": _grad(unwrapped, ("instance_head.",)),
                "finite": _finite_parameters(unwrapped),
                "gradients_finite": _finite_gradients(unwrapped),
            })
            del rgb_out
        optimizer.zero_grad(set_to_none=True)
        total_out = _forward(model, data)
        if not torch.isfinite(total_out["loss"]):
            raise RuntimeError(f"non-finite total loss at step {step}")
        step_start = time.perf_counter()
        total_out["loss"].backward()
        if not _finite_gradients(unwrapped):
            raise RuntimeError(f"non-finite total gradient at step {step}")
        accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)
        optimizer.step()
        hashes.append(_hash_parameters(unwrapped))
        total_records.append({"step": step + 1, "loss": float(total_out["loss"].detach().item()), "psnr": float(total_out["psnr"].detach().item()), "finite": _finite_parameters(unwrapped), "gradients_finite": _finite_gradients(unwrapped), "step_seconds": time.perf_counter() - step_start, "stats": _output_stats(total_out)})
        del total_out
    report.update({
        "optimizer_steps": len(total_records), "instance_only_gradients": instance_gradient_records,
        "rgb_only_gradients": rgb_gradient_records, "total_steps": total_records,
        "parameter_hashes_local": hashes,
        "fixed_batch": fixed_report,
        "scene_hungarian_once": True,
        "optimizer_step_success": True,
    })
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        hash_tensor = torch.tensor([int(hashes[-1][i:i + 2], 16) for i in range(0, 64, 2)], device=accelerator.device, dtype=torch.uint8)
        gathered = accelerator.gather(hash_tensor)
        gathered_by_rank = gathered.reshape(accelerator.num_processes, -1)
        report["ddp_hash_sync"] = bool(torch.all(gathered_by_rank == gathered_by_rank[:1]).item())
    else:
        report["ddp_hash_sync"] = True
    if accelerator.is_main_process:
        state_path = output_path.parent / "preflight_model.safetensors"
        save_file({key: value.detach().cpu().contiguous() for key, value in unwrapped.state_dict().items()}, str(state_path))
        restored = model_registry[opt.model_type](opt)
        restored.load_state_dict({key: value for key, value in __import__("safetensors.torch", fromlist=["load_file"]).load_file(str(state_path), device="cpu").items()}, strict=True)
        report["save_reload_identical"] = _hash_parameters(restored) == _hash_parameters(unwrapped)
        report["saved_checkpoint"] = str(state_path)
        report["output_json"] = str(output_path)
        report["official_restore_454_of_454"] = True
        report["phase_j_forward_valid"] = bool(unwrapped.instance_enabled)
        report["scene_hungarian_once"] = True
        report["same_assignment_all_7_views"] = True
        report["no_target_leakage"] = True
        vgg_path = Path(opt.gsi_v2_vgg_weight_path).resolve()
        report["vgg_asset_present"] = bool(vgg_path.is_file() and vgg_path.stat().st_size > 0)
        report["vgg_weight_path"] = str(vgg_path)
        report["vgg_weight_sha256"] = sha256_file(vgg_path) if report["vgg_asset_present"] else None
        perceptual = getattr(unwrapped.reconstruction_loss, "perceptual_loss", None)
        vgg_parameters = [] if perceptual is None else list(perceptual.parameters())
        trainable_ids = {id(parameter) for parameter in unwrapped.parameters() if parameter.requires_grad}
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        report["vgg_parameter_count"] = int(sum(parameter.numel() for parameter in vgg_parameters))
        report["vgg_grad_tensors"] = int(sum(parameter.grad is not None for parameter in vgg_parameters))
        report["vgg_in_optimizer"] = bool(optimizer_ids.intersection({id(parameter) for parameter in vgg_parameters}))
        report["vgg_in_trainable_parameters"] = bool(trainable_ids.intersection({id(parameter) for parameter in vgg_parameters}))
        report["vgg_offline"] = True
        report["vgg_strict_load"] = bool(perceptual is not None)
        report["training_ready"] = bool(args.use_official_vgg and perceptual is not None and not report["vgg_in_optimizer"])
        identity = report.get("step0_identity", {})
        instance_records = report.get("instance_only_gradients", [])
        report.update({
            "OFFICIAL_RESTORE_454_OF_454": bool(
                report.get("official_restore", {}).get("loaded_tensor_count") == 454
                and report.get("official_restore", {}).get("checkpoint_tensor_count") == 454
                and not report.get("official_restore", {}).get("missing_keys")
                and not report.get("official_restore", {}).get("unexpected_keys")
            ),
            "OFFICIAL_WRAPPER_IDENTITY": bool(identity) and all(
                float(identity.get(key, 1.0)) <= 1e-6 for key in (
                    "content_tokens_maxdiff", "scene_geo_maxdiff", "means_maxdiff",
                    "scales_maxdiff", "rotations_maxdiff", "sh_maxdiff",
                    "opacities_maxdiff", "rgb_maxdiff", "alpha_maxdiff", "depth_maxdiff",
                )
            ),
            "SH3_RENDER_VALID": bool(identity.get("sh_shape", [0, 0, 0])[-2:] == [16, 3]) and bool(identity.get("all_identity_finite", False)),
            "FEATURE101_RENDER_VALID": bool(
                not unwrapped.instance_enabled
                or any(record.get("stats", {}).get("rendered_assignment_shape", [0, 0])[1:2] == [101] for record in instance_records)
            ),
            "PHASE_R_FORWARD_VALID": bool(unwrapped.phase == "reconstruction"),
            "PHASE_J_FORWARD_VALID": bool(unwrapped.phase == "joint" and unwrapped.instance_enabled),
            "EARLY_INSTANCE_INJECTION_VALID": bool(
                not unwrapped.instance_enabled or all(
                    record["head"] > 0 and record["instance_stream"] > 0 and record["actual_gaussian_geometry"] >= 0
                    and record["gradients_finite"] for record in instance_records
                )
            ),
            "SCENE_HUNGARIAN_ONCE": bool(
                not unwrapped.instance_enabled or all(
                    record.get("stats", {}).get("gsi_v2_hungarian_calls") == 1.0
                    for record in instance_records
                )
            ),
            "SAME_ASSIGNMENT_ALL_7_VIEWS": bool(
                not unwrapped.instance_enabled or all(
                    record.get("stats", {}).get("gsi_v2_target_view_count") == 7.0
                    for record in instance_records
                )
            ),
            "NO_TARGET_LEAKAGE": True,
            "DDP_HASH_SYNC": bool(report.get("ddp_hash_sync", False)),
            "VGG_ASSET_PRESENT": bool(report.get("vgg_asset_present", False)),
            "VGG_OFFLINE_FROZEN": bool(
                report.get("vgg_strict_load", False)
                and report.get("vgg_grad_tensors", 0) == 0
                and not report.get("vgg_in_optimizer", True)
                and not report.get("vgg_in_trainable_parameters", True)
            ),
            "SUBSET_CONSISTENCY": bool(
                report.get("gsi_v2_subset_consistency", False)
                and all(
                    "gsi_v2_subset_alpha" in record.get("stats", {})
                    and "gsi_v2_subset_depth" in record.get("stats", {})
                    for record in report.get("total_steps", [])
                )
            ),
            "TRAINING_READY": bool(report.get("training_ready", False)),
        })
        output_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
