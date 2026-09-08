"""GA-IDU-0/1 single-process or DDP preflight; never a formal trainer."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
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

CKPT = "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors"

def norm(model, prefixes):
    s = 0.0
    for n, p in model.named_parameters():
        if p.grad is not None and any(n.startswith(x) for x in prefixes):
            s += float(p.grad.detach().float().square().sum())
    return s ** 0.5

def module_norm(model, module_name):
    return norm(model, (f"ga_idu1_head.{module_name}.",))

def module_stats(model, module_name, before):
    prefix = f"ga_idu1_head.{module_name}."
    grad_sq = update_sq = param_sq = count = 0.0
    for name, p in model.named_parameters():
        if not name.startswith(prefix):
            continue
        count += p.numel(); param_sq += float(p.detach().float().square().sum())
        if p.grad is not None:
            grad_sq += float(p.grad.detach().float().square().sum())
        if name in before:
            update_sq += float((p.detach().float() - before[name]).square().sum())
    raw = grad_sq ** 0.5
    return {"raw_grad_norm": raw, "grad_norm_sqrt_param": raw / max(count, 1.0) ** 0.5, "parameter_update_norm": update_sq ** 0.5, "update_norm_parameter_norm": (update_sq / max(param_sq, 1e-30)) ** 0.5, "parameter_count": int(count)}

def phash(model):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        h.update(n.encode()); h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:24]

def finite(x):
    return bool(torch.isfinite(torch.as_tensor(x)).all())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="semantic_v6_absolute_units_true_shared_ga_idu1_ddp8")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--resume", default=CKPT)
    a = ap.parse_args()
    out = ROOT / a.workspace
    opt = config_defaults[a.config]
    opt.workspace = str(out); opt.resume = str(ROOT / a.resume) if not os.path.isabs(a.resume) else a.resume
    opt.num_workers = 0; opt.num_epochs = 1; opt.max_iters_per_epoch = a.steps
    opt.eval_before_training = False; opt.use_wandb = False; opt.print_freq = 999999; opt.log_image_freq = 999999
    acc = Accelerator(mixed_precision=opt.mixed_precision, dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    if acc.is_main_process and out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing non-empty workspace: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if acc.num_processes > 1:
        dist.barrier()
    loader, _, _, _ = get_multi_dataloader(opt, acc)
    model = model_registry[opt.model_type](opt).cuda().train()
    raw = load_file(opt.resume, device="cpu")
    counts = {
        "absolute_gs_head": sum(k.startswith("absolute_gs_head.") for k in raw),
        "tsh_instance_head": sum(k.startswith("tsh_instance_head.") for k in raw),
        "decoder_tail": sum(k.startswith("enc_dec_backbone.decoder_blocks.") for k in raw),
    }
    assert counts == {"absolute_gs_head": 24, "tsh_instance_head": 50, "decoder_tail": 324}, counts
    assert not any("pgsr" in k.lower() or "query_memory_refiner" in k.lower() for k in raw)
    load_model_checkpoint(opt, model, acc, 0)
    optimizer = setup_optimizer(opt, model, acc, 0)
    model, optimizer, loader, _ = acc.prepare(model, optimizer, loader, loader)
    base = acc.unwrap_model(model)
    trainable = [n for n,p in base.named_parameters() if p.requires_grad]
    if a.config.endswith("ga_idu1_ddp8"):
        assert trainable and all(n.startswith("ga_idu1_head.") for n in trainable), trainable[:10]
    else:
        # GA-IDU-0 is an identity audit, not a training configuration.  The
        # base TSH parameters may still have requires_grad=True because the
        # shared model constructor owns that flag; its configured optimizer
        # LR is zero.  Do not mistake that bookkeeping state for GA-IDU-0
        # trainability, and do not backward/step in the identity audit.
        assert all(
            n.startswith(("absolute_gs_head.", "tsh_instance_head."))
            for n in trainable
        ), trainable[:10]
    iterator = iter(loader); records=[]; samples=[]
    for step in range(a.steps):
        data = next(iterator)
        scene = data["scene_name"][0] if isinstance(data["scene_name"], (list,tuple)) else str(data["scene_name"])
        samples.append(scene)
        base.ga_idu_gate_eff = 0.0
        base.tsh_instance_loss_weight_eff = 1.0
        base.tsh_unit_grad_eff = 0.0; base.tsh_mbm_u2r_eff = 0.0; base.teacher_lambda_eff = 0.0
        model.eval()
        base.ga_idu_eval_gate_override = 0.0
        with acc.autocast(): z = model(data, compute_quality_metrics=False)
        identity = float((z["unit_logits"] - z["base_unit_logits"]).abs().max())
        rgb0 = z["images_pred"].detach().clone(); gs0 = z["gaussians"].detach().clone()
        assert identity <= 1e-6 and finite(z["loss_instance_group"])
        if a.config.endswith("ga_idu0_ddp8"):
            records.append({
                "step": step + 1,
                "scene": scene,
                "gate": 0.0,
                "identity_max_diff": identity,
                "memory_meta": getattr(base, "_last_encoder_memory_meta", {}),
                "loss": float(z["loss"].detach()),
                "loss_instance": float(z["loss_instance_group"].detach()),
                "loss_rgb": float(z["loss_rgb"].detach()),
                "rgb_max_diff": 0.0,
                "gs_max_diff": 0.0,
                "parameter_hash": phash(base),
                "identity_only": True,
            })
            continue
        delattr(base, "ga_idu_eval_gate_override")
        base.ga_idu_gate_eff = min(1.0, (step + 1) / float(max(1, opt.ga_idu_gate_steps)))
        model.train(); optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        step_start = time.perf_counter()
        with acc.autocast(): ins = model(data, compute_quality_metrics=False)
        li = ins["loss_instance_group"]; assert li.requires_grad and finite(li)
        acc.backward(li)
        g_instance = norm(base, ("ga_idu1_head.",)); g_old = norm(base, ("tsh_instance_head.", "absolute_gs_head.", "enc_dec_backbone."))
        assert g_instance > 0.0, g_instance
        assert g_old == 0.0, g_old
        module_grads = {
            "memory_resampler": module_norm(base, "memory_resampler"),
            "instance_decoder": module_norm(base, "instance_decoder"),
            "group_residual_decoder": module_norm(base, "group_residual_decoder"),
            "assignment_residual": module_norm(base, "assignment_residual"),
        }
        if step >= 1:
            assert all(value > 0.0 for value in module_grads.values()), module_grads
        before = {n: p.detach().float().clone() for n,p in base.named_parameters() if n.startswith("ga_idu1_head.")}
        optimizer.zero_grad(set_to_none=True)
        with acc.autocast(): rgb = model(data, compute_quality_metrics=False)
        assert finite(rgb["loss_rgb"])
        rgb_value = float(rgb["loss_rgb"].detach()); rgb_requires_grad = bool(rgb["loss_rgb"].requires_grad)
        optimizer.zero_grad(set_to_none=True)
        with acc.autocast(): total = model(data, compute_quality_metrics=False)
        assert finite(total["loss"]) and finite(total["loss_rgb"])
        acc.backward(total["loss"]); acc.clip_grad_norm_(model.parameters(), opt.gradient_clip); optimizer.step()
        records.append({"step": step+1, "scene":scene, "gate":base.ga_idu_gate_eff, "identity_max_diff":identity, "memory_meta":getattr(base, "_last_encoder_memory_meta", {}), "module_grads":module_grads, "module_stats":{m:module_stats(base,m,before) for m in ("memory_resampler","instance_decoder","group_residual_decoder","assignment_residual")}, "frozen_grad":g_old, "rgb_requires_grad":rgb_requires_grad, "loss":float(total["loss"].detach()), "loss_instance":float(li.detach()), "loss_rgb":rgb_value, "rgb_max_diff":float((rgb["images_pred"].detach()-rgb0).abs().max()), "gs_max_diff":float((rgb["gaussians"].detach()-gs0).abs().max()), "step_seconds":time.perf_counter()-step_start, "peak_cuda_memory_mb":torch.cuda.max_memory_allocated()/1024**2, "parameter_hash":phash(base)})
    local = {"rank":acc.process_index,"samples":samples,"records":records,"hash":phash(base)}
    gathered=[local]
    if acc.num_processes > 1:
        gathered=[None]*acc.num_processes; dist.all_gather_object(gathered, local)
    save_path=out/"ga_idu_preflight_checkpoint.safetensors"
    if acc.is_main_process:
        save_file({k:v.detach().cpu().contiguous() for k,v in torch.nn.Module.state_dict(base).items()}, str(save_path))
    if acc.num_processes > 1: dist.barrier()
    restored=load_file(str(save_path), device="cpu"); missing,unexpected=torch.nn.Module.load_state_dict(base, restored, strict=True); assert not missing and not unexpected
    report={"config":a.config,"world_size":acc.num_processes,"checkpoint_counts":counts,"fresh_reset":False,"pgsr_absent":True,"trainable":trainable,"hashes_equal":len({x["hash"] for x in gathered})==1,"rank_samples_distinct":len({x["samples"][0] for x in gathered})==acc.num_processes,"step_hashes_equal":all(len({x["records"][i]["parameter_hash"] for x in gathered})==1 for i in range(a.steps)),"ranks":gathered,"strict_restore":True,"ga_idu0_identity":a.config.endswith("ga_idu0_ddp8")}
    if acc.is_main_process:
        (out/"ga_idu_preflight.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
