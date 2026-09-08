"""Small DDP preflight for the True-Shared query-memory refiner probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.options import config_defaults
from tokengs.train import load_model_checkpoint, setup_optimizer


CFG = "semantic_v6_absolute_units_true_shared_query_memory_refine_probe_ddp8"
CKPT = "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors"


def model_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:24]


def grad_norm(model, prefixes):
    value = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None and any(name.startswith(x) for x in prefixes):
            value += float(p.grad.detach().float().square().sum())
    return value ** 0.5


def grad_norm_excluding(model, prefix, excluded):
    value = 0.0
    for name, p in model.named_parameters():
        if name.startswith(prefix) and not name.startswith(excluded) and p.grad is not None:
            value += float(p.grad.detach().float().square().sum())
    return value ** 0.5


def max_diff(a, b):
    return float((a.detach().float() - b.detach().float()).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--overfit-steps", type=int, default=0)
    ap.add_argument("--config", default=CFG)
    ap.add_argument("--resume", default=CKPT)
    args = ap.parse_args()
    root = ROOT
    out = root / args.workspace
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing non-empty workspace: {out}")
    out.mkdir(parents=True, exist_ok=True)

    cfg_name = args.config
    opt = config_defaults[cfg_name]
    opt.workspace = str(out)
    opt.resume = str(root / args.resume) if not os.path.isabs(args.resume) else args.resume
    opt.num_workers = 0
    opt.num_epochs = 1
    opt.max_iters_per_epoch = int(args.steps)
    opt.eval_before_training = False
    opt.use_wandb = False
    opt.print_freq = 1000
    opt.log_image_freq = 1000

    ddp = Accelerator(
        mixed_precision=opt.mixed_precision,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    loader, _, _, _ = get_multi_dataloader(opt, ddp)
    model = model_registry[opt.model_type](opt).cuda().train()
    raw = load_file(opt.resume, device="cpu")
    counts = {
        "absolute_gs_head": sum(k.startswith("absolute_gs_head.") for k in raw),
        "tsh_instance_head": sum(k.startswith("tsh_instance_head.") for k in raw),
        "decoder_tail": sum(k.startswith("enc_dec_backbone.decoder_blocks.") for k in raw),
    }
    assert counts == {"absolute_gs_head": 24, "tsh_instance_head": 50, "decoder_tail": 324}, counts
    assert not any("refine" in k.lower() or "pgsr" in k.lower() for k in raw)
    load_model_checkpoint(opt, model, ddp, 0)
    optimizer = setup_optimizer(opt, model, ddp, 0)
    model, optimizer, loader, _ = ddp.prepare(model, optimizer, loader, loader)
    base = ddp.unwrap_model(model)
    trainable = [n for n, p in base.named_parameters() if p.requires_grad]
    joint_probe = bool(getattr(opt, "tsh_query_memory_refine_head_joint_probe", False))
    if joint_probe:
        assert trainable and all(n.startswith("tsh_instance_head.") for n in trainable), trainable[:10]
        assert any(n.startswith("tsh_instance_head.query_memory_refiner.") for n in trainable)
    else:
        assert trainable and all(n.startswith("tsh_instance_head.query_memory_refiner.") for n in trainable), trainable[:10]
    assert not any(n.startswith("absolute_gs_head.") for n in trainable)
    if not joint_probe:
        assert not any(
            n.startswith("tsh_instance_head.")
            and not n.startswith("tsh_instance_head.query_memory_refiner.")
            for n in trainable
        )

    iterator = iter(loader)
    samples, records = [], []
    if int(args.overfit_steps) > 0:
        # Short learnability probe on one fixed batch.  This is deliberately
        # separate from the DDP audit below and never writes a formal model.
        data = next(iterator)
        losses = []
        residual_max = []
        for step in range(int(args.overfit_steps)):
            base.tsh_query_memory_refine_gate_eff = max(
                base.compute_tsh_query_memory_refine_eff(step, opt), 1.0 / 50.0
            )
            base.tsh_instance_loss_weight_eff = 1.0
            base.tsh_unit_grad_eff = 0.0
            base.tsh_mbm_u2r_eff = 0.0
            base.teacher_lambda_eff = 0.0
            optimizer.zero_grad(set_to_none=True)
            with ddp.autocast():
                out_step = model(data, compute_quality_metrics=False)
            loss_step = out_step["loss_instance_group"]
            assert torch.isfinite(loss_step).all()
            ddp.backward(loss_step)
            ddp.clip_grad_norm_(model.parameters(), opt.gradient_clip)
            optimizer.step()
            losses.append(float(loss_step.detach()))
            residual_max.append(float(out_step["residual_logits"].detach().abs().max()))
        if ddp.is_main_process:
            report = {
                "mode": "fixed_batch_overfit",
                "steps": int(args.overfit_steps),
                "initial_instance_loss": losses[0],
                "final_instance_loss": losses[-1],
                "losses": losses,
                "max_residual_logits": max(residual_max),
                "all_finite": all(torch.isfinite(torch.tensor(losses))),
            }
            (out / "overfit_query_memory_refine_probe.json").write_text(
                json.dumps(report, indent=2)
            )
            print(json.dumps(report, indent=2))
        return
    for step in range(int(args.steps)):
        gate = base.compute_tsh_query_memory_refine_eff(step, opt)
        base.tsh_query_memory_refine_gate_eff = gate
        base.tsh_instance_loss_weight_eff = 1.0
        base.tsh_unit_grad_eff = 0.0
        base.tsh_mbm_u2r_eff = 0.0
        base.teacher_lambda_eff = 0.0
        data = next(iterator)
        scene = data["scene_name"][0] if isinstance(data["scene_name"], (list, tuple)) else str(data["scene_name"])
        samples.append(str(scene))

        # Step-0 identity and representation checks use two fresh forwards.
        optimizer.zero_grad(set_to_none=True)
        base.tsh_query_memory_refine_gate_eff = 0.0
        # The existing TSH blocks are stochastic in train mode.  Identity is
        # a value-preservation check, so make only this paired comparison
        # deterministic; all gradient and optimizer checks below run in the
        # normal train mode.
        model.eval()
        base.tsh_query_memory_refine_eval_gate_override = 0.0
        with ddp.autocast():
            zero = model(data, compute_quality_metrics=False)
        zero_q = base._tsh_last_q_abs.detach().clone()
        zero_gs = base._tsh_last_student_gaussians.detach().clone()
        zero_rgb = base._tsh_last_rgb.detach().clone()
        zero_mask = zero["rendered_instance_group_probability"].detach().clone()
        assert max_diff(zero["unit_logits"], zero["base_unit_logits"]) == 0.0

        base.tsh_query_memory_refine_gate_eff = gate
        with ddp.autocast():
            probe = model(data, compute_quality_metrics=False)
        assert max_diff(zero_q, base._tsh_last_q_abs) == 0.0
        assert max_diff(zero_gs, base._tsh_last_student_gaussians) == 0.0
        assert max_diff(zero_rgb, base._tsh_last_rgb) == 0.0
        assert max_diff(zero_mask, probe["rendered_instance_group_probability"]) == 0.0
        assert torch.isfinite(probe["loss_instance_group"]).all()
        delattr(base, "tsh_query_memory_refine_eval_gate_override")
        model.train()

        # Independent instance-only backward: only the refiner may receive grad.
        optimizer.zero_grad(set_to_none=True)
        base.tsh_query_memory_refine_gate_eff = max(gate, 1.0 / 50.0)
        with ddp.autocast():
            instance_out = model(data, compute_quality_metrics=False)
        instance_loss = instance_out["loss_instance_group"]
        instance_loss_value = float(instance_loss.detach())
        ddp.backward(instance_loss)
        refiner_grad = grad_norm(base, ("tsh_instance_head.query_memory_refiner.",))
        old_head_grad = grad_norm_excluding(
            base,
            "tsh_instance_head.",
            "tsh_instance_head.query_memory_refiner.",
        )
        non_refiner = grad_norm(base, ("absolute_gs_head.", "enc_dec_backbone.", "instance_branch."))
        assert refiner_grad > 0.0, refiner_grad
        if joint_probe:
            assert old_head_grad > 0.0, old_head_grad
        assert non_refiner == 0.0, non_refiner
        del instance_out, instance_loss

        # RGB-only fresh forward: all probe parameters are frozen off the RGB path.
        optimizer.zero_grad(set_to_none=True)
        with ddp.autocast():
            rgb_out = model(data, compute_quality_metrics=False)
        assert torch.isfinite(rgb_out["loss_rgb"]).all()
        assert not rgb_out["loss_rgb"].requires_grad
        del rgb_out

        # Third fresh forward is the only optimizer backward/step.
        optimizer.zero_grad(set_to_none=True)
        with ddp.autocast():
            total_out = model(data, compute_quality_metrics=False)
        total = total_out["loss"]
        assert torch.isfinite(total).all()
        ddp.backward(total)
        assert torch.isfinite(torch.as_tensor(grad_norm(base, ("tsh_instance_head.query_memory_refiner.",))))
        ddp.clip_grad_norm_(model.parameters(), opt.gradient_clip)
        optimizer.step()
        records.append({
            "step": step + 1,
            "rank": ddp.process_index,
            "sample": str(scene),
            "gate": float(gate),
            "fresh_forwards": 5,
            "instance_refiner_grad": float(refiner_grad),
            "instance_non_refiner_grad": float(non_refiner),
            "old_tsh_grad": float(old_head_grad),
            "loss": float(total.detach()),
            "loss_instance": instance_loss_value,
            "loss_rgb": float(total_out["loss_rgb"].detach()),
        })

    # Validate save/restore without touching the formal checkpoint.
    # The model's public state_dict intentionally contains only checkpointable
    # trainable subsets.  This diagnostic round-trip is a strict full-module
    # restore, so use the native Module implementation rather than that
    # reduced training-checkpoint view.
    local_state = {
        k: v.detach().cpu().contiguous()
        for k, v in torch.nn.Module.state_dict(base).items()
    }
    save_path = out / "probe_checkpoint.safetensors"
    if ddp.is_main_process:
        save_file(local_state, str(save_path))
    if ddp.num_processes > 1:
        dist.barrier()
    restored = load_file(str(save_path), device="cpu")
    missing, unexpected = base.load_state_dict(restored, strict=True)
    assert not missing and not unexpected
    gathered = [{"rank": ddp.process_index, "hash": model_hash(base), "samples": samples, "records": records}]
    if ddp.num_processes > 1:
        gathered = [None] * ddp.num_processes
        dist.all_gather_object(gathered, {"rank": ddp.process_index, "hash": model_hash(base), "samples": samples, "records": records})
    if ddp.is_main_process:
        report = {
        "config": cfg_name,
            "world_size": ddp.num_processes,
            "checkpoint_counts": counts,
            "fresh_reset": False,
            "pgsr_refine_head": "absent",
            "trainable_only": (
                "tsh_instance_head.*"
                if joint_probe
                else "tsh_instance_head.query_memory_refiner.*"
            ),
            "rank_hashes": gathered,
            "hashes_equal": len({x["hash"] for x in gathered}) == 1,
            "rank_samples": [{"rank": x["rank"], "samples": x["samples"]} for x in gathered],
            "checkpoint_restore_strict": True,
            "records": [r for x in gathered for r in x["records"]],
        }
        (out / "preflight_query_memory_refine_probe.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
