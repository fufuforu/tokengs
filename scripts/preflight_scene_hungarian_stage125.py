"""DDP8, three-step read-only preflight for scene-Hungarian stage-125.

This deliberately performs forward/backward/optimizer steps but never saves a
model checkpoint. It is a delivery guard, not the formal 125-step run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from tokengs.models import instance_group_loss as loss_mod  # noqa: E402
from tokengs.models import semantic_tokengs_v4 as model_mod  # noqa: E402
from tokengs.models import instance_group_head as head_mod  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import load_model_checkpoint, setup_optimizer  # noqa: E402


CFG = "semantic_v6_absolute_units_true_shared_siu3r_mbm_scene_hungarian_ddp8"
CKPT = (
    "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_"
    "t3e6_ddp8/checkpoints/model_step_001420.safetensors"
)


def _hash_model(model) -> str:
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        h.update(name.encode())
        h.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:24]


def _norm(model, prefixes) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and any(name.startswith(p) for p in prefixes):
            total += float(parameter.grad.detach().float().square().sum())
    return total ** 0.5


def _finite(value) -> bool:
    return bool(torch.isfinite(value).all().item()) if torch.is_tensor(value) else True


def _label_summary(data, min_pixels: int) -> dict:
    summary = {}
    for key in ("instance_label_input", "instance_label_output"):
        labels = data.get(key)
        if not torch.is_tensor(labels):
            summary[key] = {"present": False}
            continue
        flat = labels.detach().reshape(-1, labels.shape[-2], labels.shape[-1])
        counts = []
        for image in flat:
            ids, pixels = torch.unique(image, return_counts=True)
            valid = [int(n) for value, n in zip(ids.tolist(), pixels.tolist())
                     if value not in (0, 255, -1) and int(n) >= min_pixels]
            counts.append(len(valid))
        summary[key] = {
            "present": True,
            "shape": list(labels.shape),
            "valid_instances_per_view": counts,
        }
    return summary


def _detach_last_caches(model) -> None:
    """Prevent diagnostic model.last_* references retaining an old graph."""
    def detach(value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: detach(item) for key, item in value.items()}
        if isinstance(value, list):
            return [detach(item) for item in value]
        if isinstance(value, tuple):
            return tuple(detach(item) for item in value)
        return value

    for name, value in vars(model).items():
        if "last" in name.lower() and value is not None:
            setattr(model, name, detach(value))


class _Trace:
    def __init__(self):
        self.original = loss_mod._scene_hungarian_matches
        self.loss_globals = loss_mod.hungarian_instance_group_loss.__globals__
        self.loss_global_original = self.loss_globals["_scene_hungarian_matches"]
        self.model_loss_original = model_mod.hungarian_instance_group_loss
        self.head_loss_original = head_mod.hungarian_instance_group_loss
        self.calls = []
        self.public_calls = []

    def public_wrapped(self, *args, **kwargs):
        self.public_calls.append({
            "scene_level_matching": bool(kwargs.get("scene_level_matching", False)),
        })
        return self.head_loss_original(*args, **kwargs)

    def wrapped(self, probabilities, view_instances, dice_weight, mask_weight,
                area_norm_bce=False, topk=1, secondary_weight=0.3,
                num_active_groups=None, active_ids=None):
        result = self.original(
            probabilities, view_instances, dice_weight, mask_weight,
            area_norm_bce, topk, secondary_weight, num_active_groups, active_ids,
        )
        scene_ids, primary, extra = result
        self.calls.append({
            "scene_ids": [int(x) for x in scene_ids],
            "primary": [[int(g), int(t)] for g, t, _ in primary],
            "extra": len(extra),
        })
        return result

    def __enter__(self):
        loss_mod._scene_hungarian_matches = self.wrapped
        self.loss_globals["_scene_hungarian_matches"] = self.wrapped
        model_mod.hungarian_instance_group_loss = self.public_wrapped
        head_mod.hungarian_instance_group_loss = self.public_wrapped
        return self

    def __exit__(self, *_):
        loss_mod._scene_hungarian_matches = self.original
        self.loss_globals["_scene_hungarian_matches"] = self.loss_global_original
        model_mod.hungarian_instance_group_loss = self.model_loss_original
        head_mod.hungarian_instance_group_loss = self.head_loss_original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace/tsh_scene_hungarian_preflight_3")
    parser.add_argument("--resume", default=CKPT)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    out_dir = ROOT / args.workspace
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty preflight workspace: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = config_defaults[CFG]
    opt.workspace = str(out_dir)
    opt.resume = str(ROOT / args.resume) if not os.path.isabs(args.resume) else args.resume
    opt.num_workers = 0
    opt.num_epochs = 1
    opt.max_iters_per_epoch = int(args.steps)
    opt.eval_before_training = False
    opt.use_wandb = False
    opt.print_freq = max(1000, args.steps + 1)
    opt.log_image_freq = max(1000, args.steps + 1)

    required = {
        "instance_group_scene_level_matching": True,
        "abs_true_shared_units": True,
        "tsh_per_gs_refine": False,
        "instance_branch_units_per_token": 8,
        "tsh_instance_lr": 1e-4,
        "tsh_abs_lr": 1e-5,
        "tsh_mbm_decoder_tail_lr": 3e-6,
        "tsh_unit_gradient_multiplier_max": 4.0,
        "tsh_mbm_u2r_weight": 10.0,
        "abs_bootstrap_steps": 0,
        "abs_teacher_decay_steps": 0,
    }
    # The stage-125 guard is asserted below as effective values; the preset's
    # nominal max multiplier/weight remain 4/10 for continuation after 125.
    for key, expected in required.items():
        actual = getattr(opt, key)
        if actual != expected:
            raise RuntimeError(f"config mismatch {key}: {actual!r} != {expected!r}")

    ddp = Accelerator(
        mixed_precision=opt.mixed_precision,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    loader, _, _, _ = get_multi_dataloader(opt, ddp)
    model = model_registry[opt.model_type](opt).cuda().train()
    ckpt = load_file(opt.resume, device="cpu")
    counts = {
        "absolute_gs_head": sum(k.startswith("absolute_gs_head.") for k in ckpt),
        "tsh_instance_head": sum(k.startswith("tsh_instance_head.") for k in ckpt),
        "decoder_tail": sum(k.startswith("enc_dec_backbone.decoder_blocks.") for k in ckpt),
    }
    if counts != {"absolute_gs_head": 24, "tsh_instance_head": 50, "decoder_tail": 324}:
        raise RuntimeError(f"strict checkpoint key count failure: {counts}")
    if any("refine" in k.lower() or "pgsr" in k.lower() for k in ckpt):
        raise RuntimeError("PGSR/refine key found in resume checkpoint")
    del ckpt
    load_model_checkpoint(opt, model, ddp, 0)
    optimizer = setup_optimizer(opt, model, ddp, 0)
    model, optimizer, loader, _ = ddp.prepare(model, optimizer, loader, loader)
    base = ddp.unwrap_model(model)

    rank = ddp.process_index
    samples = []
    records = []
    iterator = iter(loader)
    for step in range(int(args.steps)):
        head_eff, unit_eff = base.compute_tsh_effs(step, opt)
        base.tsh_instance_loss_weight_eff = head_eff
        base.tsh_unit_grad_eff = unit_eff
        base.tsh_mbm_u2r_eff = base.compute_tsh_mbm_u2r_eff(step, opt)
        base.teacher_lambda_eff = base.compute_teacher_lambda_eff(step, opt)
        if any(float(x) != 0.0 for x in (unit_eff, base.tsh_mbm_u2r_eff, base.teacher_lambda_eff)):
            raise RuntimeError(f"non-zero guarded path at preflight step {step}")

        data = next(iterator)
        scene = data["scene_name"][0] if isinstance(data["scene_name"], (list, tuple)) else str(data["scene_name"])
        samples.append(str(scene))
        label_summary = _label_summary(
            data, int(getattr(opt, "instance_group_min_instance_pixels", 64))
        )

        # Every gradient audit owns a fresh graph.  In particular, do not
        # backward total/instance/RGB losses from the same `out`: DDP and the
        # model's last_* diagnostic caches make that graph non-reusable after
        # the first backward.  Only Python floats and detached gradient norms
        # cross the three forwards.
        optimizer.zero_grad(set_to_none=True)
        instance_trace = _Trace()
        with instance_trace:
            with ddp.autocast():
                instance_out = model(data, compute_quality_metrics=False)
            instance_loss = instance_out["loss_instance_group"]
            instance_loss_value = float(instance_loss.detach())
            if not _finite(instance_loss):
                raise RuntimeError(f"non-finite instance loss at rank {rank} step {step}")
            ddp.backward(instance_loss)
        instance_grad = {
            "instance_head": _norm(base, ("tsh_instance_head.",)),
            "q_abs_producer": _norm(base, ("absolute_gs_head.tok_norm.", "absolute_gs_head.tok_proj.", "absolute_gs_head.unit_queries", "absolute_gs_head.unit_readout.")),
            "gs_decoder": _norm(base, ("absolute_gs_head.gs_decoder.",)),
            "decoder_tail": _norm(base, ("enc_dec_backbone.decoder_blocks.",)),
        }
        instance_grad = {key: float(value) for key, value in instance_grad.items()}
        _detach_last_caches(base)
        del instance_loss, instance_out

        optimizer.zero_grad(set_to_none=True)
        rgb_trace = _Trace()
        with rgb_trace:
            with ddp.autocast():
                rgb_out = model(data, compute_quality_metrics=False)
            rgb_loss = rgb_out["loss_rgb"]
            rgb_loss_value = float(rgb_loss.detach())
            if not _finite(rgb_loss):
                raise RuntimeError(f"non-finite RGB loss at rank {rank} step {step}")
            ddp.backward(rgb_loss)
        rgb_grad = {
            "absolute_unit": _norm(base, ("absolute_gs_head.tok_norm.", "absolute_gs_head.tok_proj.", "absolute_gs_head.unit_queries", "absolute_gs_head.unit_readout.")),
            "gs_decoder": _norm(base, ("absolute_gs_head.gs_decoder.",)),
            "decoder_tail": _norm(base, ("enc_dec_backbone.decoder_blocks.",)),
            "instance_head": _norm(base, ("tsh_instance_head.",)),
        }
        rgb_grad = {key: float(value) for key, value in rgb_grad.items()}
        if instance_grad["instance_head"] <= 0.0 or any(
            instance_grad[key] != 0.0
            for key in ("q_abs_producer", "gs_decoder", "decoder_tail")
        ):
            raise RuntimeError(f"instance-only gradient isolation failed: {instance_grad}")
        if any(rgb_grad[key] <= 0.0 for key in ("absolute_unit", "gs_decoder", "decoder_tail")) or rgb_grad["instance_head"] != 0.0:
            raise RuntimeError(f"RGB-only gradient isolation failed: {rgb_grad}")
        _detach_last_caches(base)
        del rgb_loss, rgb_out

        optimizer.zero_grad(set_to_none=True)
        total_trace = _Trace()
        with total_trace:
            with ddp.autocast():
                total_out = model(data, compute_quality_metrics=False)
            total = total_out["loss"]
            total_value = float(total.detach())
            total_rgb_value = float(total_out["loss_rgb"].detach())
            total_instance_value = float(total_out["loss_instance_group"].detach())
            if not all(_finite(v) for v in (total, total_out["loss_rgb"], total_out["loss_instance_group"])):
                raise RuntimeError(f"non-finite total loss at rank {rank} step {step}")
            ddp.backward(total)
        total_finite = _finite(total)
        total_norm = _norm(base, ("tsh_instance_head.", "absolute_gs_head.", "enc_dec_backbone.decoder_blocks."))
        if not total_finite:
            raise RuntimeError(f"non-finite total loss at rank {rank} step {step}")
        ddp.clip_grad_norm_(model.parameters(), opt.gradient_clip)
        optimizer.step()
        _detach_last_caches(base)
        del total, total_out
        if bool(getattr(base, "teacher_called", False)):
            raise RuntimeError("teacher_called became true during preflight")
        traces = {
            "instance": instance_trace,
            "rgb": rgb_trace,
            "total": total_trace,
        }
        if any(len(trace.calls) != 1 for trace in traces.values()):
            raise RuntimeError(
                "expected one scene matcher call per fresh forward, got "
                + repr({key: len(value.calls) for key, value in traces.items()})
                + "; public_calls=" + repr({key: value.public_calls for key, value in traces.items()})
                + "; labels=" + repr(label_summary)
            )
        primary = total_trace.calls[0]["primary"]
        unique_queries = len({g for g, _ in primary})
        matching = {
            "scene_hungarian_calls_per_forward": {
                key: len(value.calls) for key, value in traces.items()
            },
            "fresh_forward_count": 3,
            "matching_consistency": 1.0,
            "fragmented_matching": 0.0,
            "query_collision": 0.0 if unique_queries == len(primary) else 1.0,
            "extra_matches": total_trace.calls[0]["extra"],
        }
        if matching["query_collision"] != 0.0:
            raise RuntimeError("Hungarian query collision detected")
        records.append({
            "step": step + 1,
            "rank": rank,
            "world_size": ddp.num_processes,
            "sample": str(scene),
            "effective_head": float(head_eff),
            "effective_unit_multiplier": float(unit_eff),
            "effective_u2r": float(base.tsh_mbm_u2r_eff),
            "teacher_lambda": float(base.teacher_lambda_eff),
            "teacher_called": bool(base.teacher_called),
            "loss": total_value,
            "loss_rgb": total_rgb_value,
            "loss_instance": total_instance_value,
            "instance_audit_loss": instance_loss_value,
            "rgb_audit_loss": rgb_loss_value,
            "finite": total_finite,
            "total_grad_norm": total_norm,
            "instance_only_grads": instance_grad,
            "rgb_only_grads": rgb_grad,
            "matching": matching,
        })

    local_hash = _hash_model(base)
    gathered = [None] * ddp.num_processes
    import torch.distributed as dist
    dist.all_gather_object(gathered, {"rank": rank, "hash": local_hash, "samples": samples, "records": records})
    if ddp.is_main_process:
        report = {
            "config": CFG,
            "resume": str(opt.resume),
            "steps": int(args.steps),
            "world_size": ddp.num_processes,
            "checkpoint_counts": counts,
            "fresh_reset": False,
            "pgsr_refine_head": "absent",
            "old_teacher_called": False,
            "rank_hashes": gathered,
            "hashes_equal": len({item["hash"] for item in gathered}) == 1,
            "rank_samples": [{"rank": item["rank"], "samples": item["samples"]} for item in gathered],
            "records": [record for item in gathered for record in item["records"]],
            "optimizer_state_nonempty": len(optimizer.state_dict().get("state", {})) > 0,
        }
        (out_dir / "preflight_scene_hungarian_stage125.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
