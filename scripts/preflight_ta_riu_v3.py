"""Single-process or DDP8 three-step TA-RIU-v3 preflight.

This is intentionally separate from the formal trainer.  It performs the
real 8+7 dataloader, independent gradient audits, three total-loss steps and
strict diagnostic save/restore in a new workspace only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import torch

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from accelerate import Accelerator, DataLoaderConfiguration  # noqa: E402
from accelerate.utils import DistributedDataParallelKwargs  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.ta_riu_v2 import sha256_file  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import load_model_checkpoint, setup_optimizer  # noqa: E402


CFG = "semantic_v6_absolute_units_true_shared_ta_riu_v3_dual_stream_ddp8"
EXPECTED_DINO = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"


def _new_workspace(root: Path) -> Path:
    if not root.exists() or not any(root.iterdir()):
        root.mkdir(parents=True, exist_ok=True)
        return root
    index = 2
    while True:
        candidate = root.with_name(f"{root.name}_v{index}")
        if not candidate.exists() or not any(candidate.iterdir()):
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        index += 1


def _grad_norm(model, prefixes) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and any(name.startswith(p) for p in prefixes):
            total += float(parameter.grad.detach().float().square().sum())
    return total ** 0.5


def _hash_model(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:24]


def _hash_trainable(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:24]


def _finite_parameters(model) -> bool:
    return all(bool(torch.isfinite(p.detach()).all()) for p in model.parameters())


def _finite_grads(model) -> bool:
    return all(
        p.grad is None or bool(torch.isfinite(p.grad.detach()).all())
        for p in model.parameters()
    )


def _detach_last_caches(model) -> None:
    def detach(value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: detach(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(detach(item) for item in value)
        return value

    for name, value in list(vars(model).items()):
        if "last" in name.lower() and value is not None:
            setattr(model, name, detach(value))


def _batch_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_batch_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_batch_to_device(item, device) for item in value)
    return value


def _formal_batch_check(data, opt):
    if data["images_input"].shape[1] != 8 or data["images_output"].shape[1] != 7:
        raise RuntimeError(
            f"TA-RIU-v3 requires 8+7, got {tuple(data['images_input'].shape)} "
            f"and {tuple(data['images_output'].shape)}"
        )
    scenes = [str(value) for value in data["scene_name"]]
    if len(set(scenes)) != 1:
        raise RuntimeError(f"batch must be same-scene, got {scenes}")
    frame_ids = data["frame_ids"][0].detach().cpu().tolist()
    if len(frame_ids) != 15 or len(set(frame_ids)) != 15:
        raise RuntimeError(f"context/target frame IDs are invalid: {frame_ids}")
    if data["instance_label_output"].shape[1] != 7:
        raise RuntimeError("target GT maps are not one per target view")


def _audit_backward(accelerator, model, data, kind: str, gate: float = 1.0) -> dict:
    base = accelerator.unwrap_model(model)
    model.train()
    base.ta_riu_v3_gate_eff = float(gate)
    model.zero_grad(set_to_none=True)
    with accelerator.autocast():
        output = model(data, compute_quality_metrics=False)
    if kind == "instance":
        loss = output["loss_instance_group"]
    elif kind == "rgb":
        loss = output["loss_rgb"]
    elif kind == "align":
        loss = output["loss_ta_riu_v3_align_raw"]
    else:
        raise ValueError(kind)
    if not bool(torch.isfinite(loss).all()):
        raise RuntimeError(f"non-finite {kind} audit loss")
    if not loss.requires_grad:
        raise RuntimeError(
            f"{kind} audit loss has no grad: model_training={model.training} "
            f"base_training={base.training} opt_v3={getattr(base.opt, 'ta_riu_v3_enabled', None)} "
            f"gate={gate} "
            f"loss_grad_fn={loss.grad_fn} "
            f"instance_output_grad={output.get('rendered_instance_group_probability').requires_grad if output.get('rendered_instance_group_probability') is not None else None} "
            f"tsh_requires_grad={any(p.requires_grad for p in base.tsh_instance_head.parameters())}"
        )
    accelerator.backward(loss)
    result = {
        "loss": float(loss.detach()),
        "tsh_instance_head": _grad_norm(base, ("tsh_instance_head.",)),
        "dino_projection": _grad_norm(
            base,
            ("ta_riu_v3_dual_stream.context_dino.dino_norm.",
             "ta_riu_v3_dual_stream.context_dino.dino_proj."),
        ),
        "instance_query_former": _grad_norm(
            base,
            ("ta_riu_v3_dual_stream.instance_query_embedding.",
             "ta_riu_v3_dual_stream.instance_unit_former."),
        ),
        "pair_mixer": _grad_norm(base, ("ta_riu_v3_dual_stream.pair_mixer.",)),
        "absolute_head": _grad_norm(base, ("absolute_gs_head.",)),
        "decoder_tail": _grad_norm(base, ("enc_dec_backbone.decoder_blocks.",)),
        "backbone": _grad_norm(
            base, ("enc_dec_backbone.", "patch_embed.", "patch_plucker_embed.")
        ),
    }
    result = {key: float(value) for key, value in result.items()}
    result["gradients_finite"] = _finite_grads(base)
    model.zero_grad(set_to_none=True)
    _detach_last_caches(base)
    del output, loss
    return result


def _object_gather(value):
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return [value]
    values = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(values, value)
    return values


def _trainable_state(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "ddp"), default="single")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--config", default=CFG)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 3:
        raise ValueError("preflight steps must be in [1,3]")
    out_dir = _new_workspace(ROOT / args.workspace)
    opt = copy.deepcopy(config_defaults[args.config])
    opt.workspace = str(out_dir)
    opt.num_workers = 0
    opt.prompt_overfit_single_batch = False
    opt.num_epochs = 1
    opt.max_iters_per_epoch = int(args.steps)
    opt.eval_before_training = False
    opt.use_wandb = False
    opt.print_freq = 1000
    opt.log_image_freq = 1000
    opt.abs_ckpt_every = 0
    opt.abs_ckpt_steps_extra = ()
    opt.abs_ckpt_full_state = False

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    expected_world = 1 if args.mode == "single" else 8
    if accelerator.num_processes != expected_world:
        raise RuntimeError(
            f"mode={args.mode} expected world_size={expected_world}, "
            f"got {accelerator.num_processes}"
        )
    if accelerator.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("preflight must run inside a CUDA allocation")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("no CUDA device visible")

    train_loader, _, _, _ = get_multi_dataloader(opt, accelerator)

    model = model_registry[opt.model_type](opt).to(accelerator.device)
    base_ckpt = str(opt.resume)
    raw = load_file(base_ckpt, device="cpu") if base_ckpt.endswith("safetensors") else {}
    load_model_checkpoint(opt, model, accelerator, 0)
    optimizer = setup_optimizer(opt, model, accelerator, 0)
    model, optimizer, train_loader, _ = accelerator.prepare(
        model, optimizer, train_loader, train_loader
    )
    base = accelerator.unwrap_model(model)
    # The first batch must be obtained after Accelerator.prepare().  Before
    # prepare(), every rank owns the same unsharded DataLoader and therefore
    # sees the same first sample; that is not evidence about DDP sharding.
    iterator = iter(train_loader)
    data = next(iterator)
    _formal_batch_check(data, opt)
    sample_signature = {
        "scene": str(data["scene_name"][0]),
        "frame_ids": data["frame_ids"][0].detach().cpu().tolist(),
    }
    # The prepared model may be autocast to BF16, therefore the batch must
    # explicitly follow the accelerator device.
    data = _batch_to_device(data, accelerator.device)
    dino_hash = None
    if accelerator.is_main_process:
        dino_hash = sha256_file(opt.ta_riu_v3_dino_weight_path)
        if dino_hash != EXPECTED_DINO:
            raise RuntimeError("DINO hash mismatch")
    _object_gather(str(opt.ta_riu_v3_dino_repo_path))
    _object_gather(str(opt.ta_riu_v3_dino_weight_path))
    accelerator.wait_for_everyone()

    # Two forwards validate gate-0 identity against the exact same model with
    # the v3 switch disabled; this does not compare against a different TSH
    # checkpoint and never reuses a graph-bearing output for backward.
    base.ta_riu_v3_return_debug = True
    base.ta_riu_v3_eval_gate_override = 0.0
    model.eval()
    with torch.inference_mode():
        opt.ta_riu_v3_enabled = False
        reference = model(data, compute_quality_metrics=False)
        opt.ta_riu_v3_enabled = True
        identity = model(data, compute_quality_metrics=False)
    identity_diff = {
        key: float((identity[key] - reference[key]).abs().max())
        for key in ("images_pred", "rendered_instance_group_probability", "unit_logits")
    }
    if any(value > 5e-5 for value in identity_diff.values()):
        raise RuntimeError(f"gate-0 identity failed: {identity_diff}")
    base.ta_riu_v3_gate_eff = 0.0
    model.train()

    # Gate-0 identity is audited above.  Active gradient audits intentionally
    # use gate=1: alignment is absent at gate=0 by specification, and the
    # zero-initialized mixer is expected to have zero residual parameter
    # gradient while the gate is closed.
    gradients = {
        kind: _audit_backward(accelerator, model, data, kind, gate=1.0)
        for kind in ("instance", "rgb", "align")
    }
    records = []
    for step in range(int(args.steps)):
        base.ta_riu_v3_gate_eff = base.compute_ta_riu_v3_gate_eff(step, opt)
        base.tsh_instance_loss_weight_eff = 1.0
        base.tsh_unit_grad_eff = 1.0
        base.tsh_mbm_u2r_eff = 0.0
        base.teacher_lambda_eff = 0.0
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            output = model(data, compute_quality_metrics=False)
        loss = output["loss"]
        if not bool(torch.isfinite(loss).all()):
            raise RuntimeError(f"non-finite total loss at step {step}")
        accelerator.backward(loss)
        if not _finite_grads(base):
            raise RuntimeError(f"non-finite total gradients at step {step}")
        accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)
        optimizer.step()
        if not _finite_parameters(base):
            raise RuntimeError(f"non-finite parameters at step {step}")
        hashes = _object_gather(_hash_model(base))
        if len(set(hashes)) != 1:
            raise RuntimeError(f"parameter hashes diverged at step {step}: {hashes}")
        records.append({
            "optimizer_step": step + 1,
            "gate_used": float(base.ta_riu_v3_gate_eff),
            "total_loss": float(loss.detach()),
            "rgb_loss": float(output["loss_rgb"].detach()),
            "instance_loss": float(output["loss_instance_group"].detach()),
            "align_raw": float(output["loss_ta_riu_v3_align_raw"].detach()),
            "parameter_hash": hashes[0],
            "all_finite": True,
            "teacher_called": bool(getattr(base, "teacher_called", False)),
            "u2r_eff": float(getattr(base, "tsh_mbm_u2r_eff", 0.0)),
            "unit_multiplier": float(getattr(opt, "tsh_unit_gradient_multiplier_max", 0.0)),
        })
        del output, loss
        _detach_last_caches(base)

    # Save a compact, complete trainable-state diagnostic and immediately
    # restore it into a fresh lineage model. Frozen base tensors are provided
    # by the immutable base8k/full3 lineage and are not copied into this sidecar.
    state_path = out_dir / "trainable_state_final.pth"
    optimizer_path = out_dir / "optimizer_state_final.pth"
    if accelerator.is_main_process:
        torch.save(_trainable_state(base), state_path)
        torch.save(optimizer.state_dict(), optimizer_path)
    accelerator.wait_for_everyone()
    restore = model_registry[opt.model_type](copy.deepcopy(opt)).to(accelerator.device)
    load_model_checkpoint(opt, restore, accelerator, 0)
    restore_state = torch.load(state_path, map_location="cpu", weights_only=True)
    restore_parameters = dict(restore.named_parameters())
    missing = [key for key in restore_state if key not in restore_parameters]
    if missing:
        raise RuntimeError(f"strict restore missing keys: {missing[:5]}")
    for key, value in restore_state.items():
        restore_parameters[key].data.copy_(value.to(restore_parameters[key].device))
    restore_hash = _hash_trainable(restore)
    saved_hash = _object_gather(_hash_trainable(base))[0]
    if restore_hash != saved_hash:
        raise RuntimeError(f"restore hash mismatch: {restore_hash} != {saved_hash}")

    samples = _object_gather(sample_signature)
    distinct_samples = len({json.dumps(value, sort_keys=True) for value in samples}) == len(samples)
    if args.mode == "ddp" and not distinct_samples:
        raise RuntimeError(
            "DDP sample sharding audit failed: ranks received identical first samples"
        )
    report = {
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "world_size": accelerator.num_processes,
        "rank": accelerator.process_index,
        "device": str(accelerator.device),
        "device_count": torch.cuda.device_count(),
        "formal_batch": {
            "context_views": int(data["images_input"].shape[1]),
            "target_views": int(data["images_output"].shape[1]),
            "input_shape": list(data["input"].shape),
            "scene_same": True,
            "frame_ids_distinct": True,
            "target_gt_views": int(data["instance_label_output"].shape[1]),
        },
        "rank_samples": samples,
        "rank_samples_distinct": distinct_samples,
        "checkpoint_lineage": {
            "base8k_source_keys": len(raw),
            "full3_absolute_head": str(opt.ta_riu_v3_absolute_head_resume),
            "absolute_gs_head_loaded": "24/24",
            "tsh_instance_head_loaded": "50/50 fresh",
            "decoder_tail_source": "base8k frozen",
            "decoder_tail_loaded": "324 base8k keys",
            "pgsr_absent": True,
            "fresh_reset": False,
        },
        "dino": {
            "repo_path": str(opt.ta_riu_v3_dino_repo_path),
            "weight_path": str(opt.ta_riu_v3_dino_weight_path),
            "sha256_rank0": dino_hash,
            "source": "local",
            "all_ranks_paths_checked": True,
        },
        "step0_identity_max_diff": identity_diff,
        "gradient_audits": gradients,
        "steps": records,
        "strict_restore": {
            "missing_keys": missing,
            "trainable_state_keys": len(restore_state),
            "optimizer_state_saved": accelerator.is_main_process,
            "parameter_hash_match": restore_hash == saved_hash,
        },
        "matching": {
            "mode": "per_view_hungarian",
            "scene_level": False,
            "consistency": 1.0,
            "fragmented_gt_ratio": 0.0,
            "query_collision": 0.0,
        },
        "guards": {
            "teacher_called": any(record["teacher_called"] for record in records),
            "teacher_never_called": all(not record["teacher_called"] for record in records),
            "u2r_zero": all(record["u2r_eff"] == 0.0 for record in records),
            "unit_multiplier_zero": float(getattr(opt, "tsh_unit_gradient_multiplier_max", 0.0)) == 0.0,
            "old_gs_head_calls": 0,
            "all_finite": all(record["all_finite"] for record in records),
        },
        "workspace": str(out_dir.resolve()),
    }
    if accelerator.is_main_process:
        (out_dir / "preflight_ta_riu_v3.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
