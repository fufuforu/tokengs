"""One fixed-batch learnability audit with mask metrics at 0/25/50/100."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.options import config_defaults
from tokengs.train import load_model_checkpoint, setup_optimizer
from scripts.eval_instance_lsm_protocol import (
    masks_from_group_probs, gt_masks_from_instance_map, instance_ap,
    _mask_diagnostics,
)

CFG = "semantic_v6_absolute_units_true_shared_query_memory_refine_probe_ddp8"
CKPT = ROOT / "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors"

def norms(model):
    out = {"tsh_total": 0.0}
    for name, p in model.named_parameters():
        if name.startswith("tsh_instance_head.query_memory_refiner."):
            key = name.split(".")[2]
            out[key] = out.get(key, 0.0) + float((p.grad.detach().float().square().sum().sqrt()) if p.grad is not None else 0.0)
        if name.startswith("tsh_instance_head."):
            out["tsh_total"] += float((p.grad.detach().float().square().sum().sqrt()) if p.grad is not None else 0.0)
    return out

def evaluate(out, data, base_logits):
    prob = out["rendered_instance_group_probability"][0].float().detach().cpu().numpy()
    labels = data["instance_label_output"][0].long().cpu().numpy()
    scene = str(data["scene_name"][0])
    preds, scores, pids, gts, gids = [], [], [], [], []
    ent = []
    for v in range(prob.shape[1]):
        p = prob[:, v, 0]
        nonvoid = p[:-1]
        q = nonvoid / np.maximum(nonvoid.sum(0, keepdims=True), 1e-8)
        ent.append(float(-(q * np.log(np.maximum(q, 1e-8))).sum(0).mean()))
        masks, sc = masks_from_group_probs(p, void_channel=prob.shape[0]-1, min_mask_area=1)
        image = f"{scene}:b0"
        preds += masks; scores += sc; pids += [image] * len(masks)
        gm = gt_masks_from_instance_map(labels[v], min_mask_area=1)
        gts += gm; gids += [image] * len(gm)
    ap = instance_ap(preds, scores, gts, thresholds=(.25,.5,.75), vectorized=True, pred_image_ids=pids, gt_image_ids=gids)
    diag = _mask_diagnostics(preds, gts, pids, gids)
    pi = out["instance_group_probabilities"][0].float().detach()
    usage = pi[..., :-1].mean(dim=(0,1,2))
    logits = out["unit_logits"][0].float().detach()
    base = base_logits[0].float().detach()
    margin = torch.topk(base[..., :-1], 2, dim=-1).values
    margin = margin[..., 0] - margin[..., 1]
    changed = (logits[..., :-1].argmax(-1) != base[..., :-1].argmax(-1)).float().mean()
    res = (out["residual_logits"][0].float().detach() if "residual_logits" in out else torch.zeros_like(logits))
    flat = res.abs().flatten().cpu().numpy()
    nonempty_queries = []
    for v in range(prob.shape[1]):
        masks, _ = masks_from_group_probs(prob[:, v, 0], void_channel=prob.shape[0]-1, min_mask_area=1)
        nonempty_queries.append(len(masks))
    return {
        "instance_loss": float(out["loss_instance_group"].detach()),
        "psnr": float(out["psnr"].detach()) if "psnr" in out else None,
        "ssim": float(out["ssim"].detach()) if "ssim" in out else None,
        "lpips": float(out["lpips"].detach()) if "lpips" in out else None,
        "ap25": float(ap["ap_25"]), "ap50": float(ap["ap_50"]), "ap75": float(ap["ap_75"]),
        **diag, "pred_gt": len(preds) / max(1, len(gts)),
        "active_queries_legacy_gt_0.01": int((usage > .01).sum()),
        "mass_query_count_gt_0.001": int((usage > .001).sum()),
        "nonempty_prediction_queries_mean": float(np.mean(nonempty_queries)),
        "effective_query_count": float(np.exp(np.mean(ent))),
        "void_ratio": float(prob[-1].mean()), "assignment_entropy": float(np.mean(ent)),
        "unit_argmax_changed_vs_base": float(changed),
        "residual": {k: float(v) for k, v in zip(("mean","std","p50","p90","p99","max"), (flat.mean(), flat.std(), *np.percentile(flat,[50,90,99]), flat.max()))},
        "residual_gt_0.9": float((flat > .9).mean()), "residual_gt_0.95": float((flat > .95).mean()),
        "base_top1_top2_margin_mean": float(margin.mean()),
        "base_top1_top2_margin_p50": float(torch.quantile(margin.flatten(), .5)),
        "base_top1_top2_margin_p90": float(torch.quantile(margin.flatten(), .9)),
        "gate": float(out["query_memory_refine_gate"].detach()),
    }

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workspace", required=True); ap.add_argument("--steps", type=int, default=100); ap.add_argument("--config", default="semantic_v6_absolute_units_true_shared_query_memory_refine_probe_ddp8")
    a = ap.parse_args(); outdir = ROOT / a.workspace
    if outdir.exists() and any(outdir.iterdir()): raise RuntimeError(f"refusing non-empty workspace: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    opt = config_defaults[a.config]; opt.workspace=str(outdir); opt.resume=str(CKPT); opt.num_workers=0; opt.num_epochs=1; opt.max_iters_per_epoch=a.steps; opt.eval_before_training=False; opt.use_wandb=False
    ddp=Accelerator(mixed_precision=opt.mixed_precision, dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    loader,_,_,_=get_multi_dataloader(opt,ddp); model=model_registry[opt.model_type](opt).cuda().train(); raw=load_file(str(CKPT),device="cpu"); load_model_checkpoint(opt,model,ddp,0); optimizer=setup_optimizer(opt,model,ddp,0); model,optimizer,loader=ddp.prepare(model,optimizer,loader); base=ddp.unwrap_model(model); data=next(iter(loader))
    trainable=[n for n,p in base.named_parameters() if p.requires_grad]
    if bool(getattr(opt,"tsh_query_memory_refine_head_joint_probe",False)):
        assert trainable and all(n.startswith("tsh_instance_head.") for n in trainable), trainable[:10]
    else:
        assert trainable and all(n.startswith("tsh_instance_head.query_memory_refiner.") for n in trainable), trainable[:10]
    base.tsh_query_memory_refine_eval_gate_override=0.0; model.eval()
    with torch.no_grad():
        initial=model(data,compute_quality_metrics=False)
    base_logits=initial["base_unit_logits"].detach().clone()
    model.train(); base.tsh_query_memory_refine_gate_eff=0.0
    with torch.no_grad(): initial_loss_out=model(data,compute_quality_metrics=False)
    metrics={"step0":evaluate(initial,data,base_logits),"gradient_norms_step1":None,"gradient_norms_step100":None}
    metrics["step0"]["instance_loss"]=float(initial_loss_out["loss_instance_group"])
    delattr(base,"tsh_query_memory_refine_eval_gate_override"); model.train()
    for step in range(a.steps):
        base.tsh_query_memory_refine_gate_eff=max(base.compute_tsh_query_memory_refine_eff(step,opt),1/50); base.tsh_instance_loss_weight_eff=1.; base.tsh_unit_grad_eff=0.; base.tsh_mbm_u2r_eff=0.; base.teacher_lambda_eff=0.; optimizer.zero_grad(set_to_none=True)
        with ddp.autocast(): cur=model(data,compute_quality_metrics=False)
        loss=cur["loss_instance_group"]; assert torch.isfinite(loss).all(); ddp.backward(loss); gn=norms(base); ddp.clip_grad_norm_(model.parameters(),opt.gradient_clip); optimizer.step()
        if step+1 in (1,25,50,100):
            model.eval()
            with torch.no_grad(): snap=model(data,compute_quality_metrics=False)
            metrics[f"step{step+1}"]=evaluate(snap,data,base_logits); model.train()
            base.tsh_query_memory_refine_gate_eff=max(base.compute_tsh_query_memory_refine_eff(step,opt),1/50)
            with torch.no_grad(): loss_snap=model(data,compute_quality_metrics=False)
            metrics[f"step{step+1}"]["instance_loss"]=float(loss_snap["loss_instance_group"])
            metrics[f"gradient_norms_step{step+1}"]=gn
    (outdir/"fixed_batch_audit.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
if __name__ == "__main__": main()
