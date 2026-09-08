"""Reproducible GA-IDU-1 fixed-batch audit (diagnostic only).

The batch is selected once from the canonical manifest, serialized after
collation, and reused for every milestone. No formal checkpoint/workspace is
modified.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
torch._dynamo.config.disable = True
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.instance_group_loss import hungarian_instance_group_loss
from tokengs.options import config_defaults
from tokengs.train import load_model_checkpoint, setup_optimizer
from scripts.eval_instance_lsm_protocol import _mask_diagnostics, gt_masks_from_instance_map, instance_ap, masks_from_group_probs

CFG = "semantic_v6_absolute_units_true_shared_ga_idu1_ddp8"
CKPT = ROOT / "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors"
MANIFEST = ROOT / "data/scannet_prompt/scannet_prompt_full_wide_8x7.json"
MILESTONES = (0, 1, 5, 25, 50, 100)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value):
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def file_info(path):
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path), "exists": True, "size": st.st_size,
        "mtime_ns": st.st_mtime_ns, "sha256": sha256_file(path),
    }


def code_provenance():
    script = Path(__file__).resolve()
    diff = subprocess.check_output(["git", "diff", "--binary", "--", str(script)], cwd=ROOT)
    return {
        "script_path": str(script),
        "script_sha256": sha256_file(script),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def tensor_hash(x):
    h = hashlib.sha256(); h.update(x.detach().cpu().contiguous().numpy().tobytes()); return h.hexdigest()


def batch_hashes(value, prefix=""):
    out = {}
    if torch.is_tensor(value):
        out[prefix] = {"sha256": tensor_hash(value), "shape": list(value.shape), "dtype": str(value.dtype)}
    elif isinstance(value, dict):
        for k, v in value.items(): out.update(batch_hashes(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value): out.update(batch_hashes(v, f"{prefix}[{i}]"))
    return out


def model_hash(model):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        h.update(n.encode()); h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def ga_hash(model):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        if n.startswith("ga_idu1_head."):
            h.update(n.encode()); h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def move_cpu(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value, dict): return {k: move_cpu(v) for k, v in value.items()}
    if isinstance(value, list): return [move_cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(move_cpu(v) for v in value)
    return value


def cache_output(output):
    keys = ("base_unit_logits", "unit_logits", "pi_unit", "pi_gs_refined", "rendered_instance_group_probability", "gaussians", "images_pred")
    return {k: output[k].detach().cpu() for k in keys if k in output}


def reset_audit_eval_rng():
    """Make diagnostic forwards reproducible across independent processes."""
    torch.manual_seed(1729)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1729)


def loss_parts(output, data, opt):
    rendered = output["rendered_instance_group_probability"]
    labels = data["instance_label_output"].long()
    loss, stats = hungarian_instance_group_loss(
        rendered, labels, num_groups=int(opt.tsh_num_groups),
        min_instance_pixels=int(opt.instance_group_min_instance_pixels),
        dice_weight=float(opt.lambda_instance_group_dice),
        mask_weight=float(opt.lambda_instance_group_mask),
        void_weight=float(opt.lambda_instance_group_void),
        unmatched_weight=float(opt.lambda_instance_group_unmatched),
        lambda_eff=1.0, area_alpha=float(opt.instance_group_area_alpha),
        match_area_norm=bool(opt.instance_group_match_area_norm),
        ce_weight=float(opt.lambda_instance_group_ce),
        match_topk=int(opt.instance_group_match_topk),
        secondary_pair_weight=float(opt.instance_group_secondary_pair_weight),
        usage_entropy_weight=float(opt.instance_group_usage_entropy),
    )
    parts = {"total": float(loss.detach())}
    for k, v in stats.items():
        if torch.is_tensor(v): parts[k] = float(v.detach())
    return parts


def module_stats(model, before):
    result = {}
    for module_name in ("memory_resampler", "instance_decoder", "group_residual_decoder", "assignment_residual"):
        prefix = f"ga_idu1_head.{module_name}."
        params = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]
        grad_sq = sum(float(p.grad.detach().float().square().sum()) for _, p in params if p.grad is not None)
        update_sq = sum(float((p.detach().float() - before[n]).square().sum()) for n, p in params)
        param_sq = sum(float(p.detach().float().square().sum()) for _, p in params)
        count = sum(p.numel() for _, p in params)
        result[module_name] = {
            "parameter_count": count,
            "grad_norm": grad_sq ** 0.5,
            "grad_norm_over_sqrt_param_count": (grad_sq ** 0.5) / max(count ** 0.5, 1.0),
            "update_norm_from_step0": update_sq ** 0.5,
            "update_over_parameter_norm": (update_sq / max(param_sq, 1e-30)) ** 0.5,
        }
    return result


def evaluate(output, data, base_logits, opt, reference_output=None, model=None, before=None):
    prob = output["rendered_instance_group_probability"][0].float().detach().cpu().numpy()
    labels = data["instance_label_output"][0].long().cpu().numpy()
    preds, scores, pids, gts, gids = [], [], [], [], []
    ent = []
    for v in range(prob.shape[1]):
        p = prob[:, v, 0]; nonvoid = p[:-1]
        q = nonvoid / np.maximum(nonvoid.sum(0, keepdims=True), 1e-8)
        ent.append(float(-(q * np.log(np.maximum(q, 1e-8))).sum(0).mean()))
        masks, sc = masks_from_group_probs(p, void_channel=prob.shape[0]-1, min_mask_area=1)
        image = f"{data['scene_name'][0]}:b0"; preds += masks; scores += sc; pids += [image] * len(masks)
        gm = gt_masks_from_instance_map(labels[v], min_mask_area=1); gts += gm; gids += [image] * len(gm)
    ap = instance_ap(preds, scores, gts, thresholds=(.25, .5, .75), vectorized=True, pred_image_ids=pids, gt_image_ids=gids)
    diag = _mask_diagnostics(preds, gts, pids, gids)
    pi = output["instance_group_probabilities"][0].float().detach(); usage = pi[..., :-1].mean(dim=(0, 1, 2))
    logits = output["unit_logits"].float().detach(); base = base_logits.float().detach()
    delta = logits[..., :-1] - base[..., :-1]
    margin = torch.topk(base[..., :-1], 2, dim=-1).values; margin = margin[..., 0] - margin[..., 1]
    final_pi = torch.softmax(logits, -1); base_pi = torch.softmax(base, -1)
    flat = delta.abs().flatten().cpu().numpy()
    visible = base[..., :-1].amax(-1) > 0
    visible_change = ((logits[..., :-1].argmax(-1) != base[..., :-1].argmax(-1)) & visible).float()
    result = {
        "audit_eval_loss": loss_parts(output, data, opt),
        "ap25": float(ap["ap_25"]), "ap50": float(ap["ap_50"]), "ap75": float(ap["ap_75"]), **diag,
        "pred_gt": len(preds) / max(1, len(gts)), "nonempty_prediction_count": len(preds),
        "mass_query_count_gt_0.001": int((usage > .001).sum()), "effective_query_count": float(np.exp(np.mean(ent))),
        "void_ratio": float(prob[-1].mean()), "assignment_entropy": float(np.mean(ent)),
        "unit_argmax_changed_vs_base": float((logits[..., :-1].argmax(-1) != base[..., :-1].argmax(-1)).float().mean()),
        "visible_unit_argmax_changed_vs_base": float(visible_change.mean()),
        "delta_logits": {"mean": float(flat.mean()), "std": float(flat.std()), "p50": float(np.percentile(flat,50)), "p90": float(np.percentile(flat,90)), "p95": float(np.percentile(flat,95)), "p99": float(np.percentile(flat,99)), "max": float(flat.max()), "gt_0.1": float((flat>.1).mean()), "gt_0.25": float((flat>.25).mean()), "gt_0.5": float((flat>.5).mean()), "gt_1": float((flat>1).mean())},
        "base_margin": {"mean": float(margin.mean()), "p50": float(torch.quantile(margin.flatten(),.5)), "p75": float(torch.quantile(margin.flatten(),.75)), "p90": float(torch.quantile(margin.flatten(),.9)), "p95": float(torch.quantile(margin.flatten(),.95)), "p99": float(torch.quantile(margin.flatten(),.99))},
        "delta_crosses_base_margin_fraction": float((delta.abs().amax(-1)>margin).float().mean()),
        "soft_probability_mean_abs_diff": float((final_pi-base_pi).abs().mean()), "soft_probability_max_abs_diff": float((final_pi-base_pi).abs().max()),
        "parameter_hash": model_hash(output["_model_for_audit"]) if "_model_for_audit" in output else None,
        "logit_hashes": {"base": tensor_hash(output["base_unit_logits"]), "delta": tensor_hash(output["unit_logits"]-output["base_unit_logits"]), "final": tensor_hash(output["unit_logits"])},
    }
    if reference_output is not None:
        ref_prob = reference_output["rendered_instance_group_probability"][0].float().detach()
        cur_prob = output["rendered_instance_group_probability"][0].float().detach()
        ref_binary = ref_prob[:-1].amax(0) > ref_prob[-1]
        cur_binary = cur_prob[:-1].amax(0) > cur_prob[-1]
        result["binary_mask_pixel_change_vs_step0"] = float((ref_binary != cur_binary).float().mean())
        result["rendered_soft_mask_max_diff_vs_step0"] = float((cur_prob - ref_prob).abs().max())
        for key in ("gaussians", "images_pred"):
            if key in output and key in reference_output:
                result[f"{key}_max_diff_vs_step0"] = float((output[key].detach().float() - reference_output[key].detach().float()).abs().max())
    if model is not None and before is not None:
        result["module_grad_update_stats"] = module_stats(model, before)
    return result


def make_opt():
    opt = config_defaults[CFG]
    # Preserve the resolved formal GA-IDU data path: full_wide_8x7 manifest,
    # 8 context views and 7 target views. No manifest or sampler override.
    opt.num_workers = 0; opt.batch_size = 1; opt.num_epochs = 1; opt.max_iters_per_epoch = 100
    opt.eval_before_training = False; opt.use_wandb = False; opt.print_freq = 100000; opt.log_image_freq = 100000
    return opt


def main():
    p = argparse.ArgumentParser(); p.add_argument("--workspace", required=True); p.add_argument("--mode", choices=("run","verify"), default="run"); p.add_argument("--steps", type=int, default=100); p.add_argument("--fixed-batch", default=None)
    a = p.parse_args(); out = ROOT / a.workspace; out.mkdir(parents=True, exist_ok=True)
    if a.mode == "verify":
        batch = torch.load(out / "fixed_batch_cpu.pt", map_location="cpu", weights_only=False)
        saved = json.loads((out / "fixed_batch_hashes.json").read_text()); assert batch_hashes(batch) == saved["tensor_hashes"]
        opt = make_opt(); opt.workspace = str(out); opt.resume = str(CKPT)
        opt.prompt_overfit_single_batch = True; opt.prompt_overfit_sample_index = 0
        acc = Accelerator(mixed_precision=opt.mixed_precision, dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        model = model_registry[opt.model_type](opt).cuda().eval()
        load_model_checkpoint(opt, model, acc, 0)
        optimizer = setup_optimizer(opt, model, acc, 0)
        model, optimizer = acc.prepare(model, optimizer)
        base = acc.unwrap_model(model)
        state = torch.load(out / "step_100.pt", map_location="cpu", weights_only=False)
        base.ga_idu1_head.load_state_dict(state["ga_idu1_head"], strict=True)
        gpu_batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
        base.ga_idu_eval_gate_override = 1.0
        reset_audit_eval_rng()
        with torch.no_grad(): current = model(gpu_batch, compute_quality_metrics=False)
        cache = torch.load(out / "cache_step_100.pt", map_location="cpu", weights_only=False)
        checks = {}
        for key in ("base_unit_logits", "unit_logits", "rendered_instance_group_probability"):
            checks[key] = float((current[key].detach().cpu().float() - cache[key].float()).abs().max())
        report = {"batch_hash_stable": True, "parameter_hash_match": ga_hash(base) == state["parameter_hash"], "cache_max_diffs": checks, "restore_match": all(v <= 1e-5 for v in checks.values())}
        (out / "independent_restore_check.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2)); return
    if any(out.iterdir()): raise RuntimeError(f"refusing non-empty workspace: {out}")
    opt = make_opt(); opt.workspace = str(out); opt.resume = str(CKPT); opt.prompt_overfit_single_batch = True
    manifest = json.loads(MANIFEST.read_text()); sample_index = 0; opt.prompt_overfit_sample_index = sample_index
    acc = Accelerator(mixed_precision=opt.mixed_precision, dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    loader, _, train_dataset, _ = get_multi_dataloader(opt, acc); data = next(iter(loader)); cpu_batch = move_cpu(data)
    provider = train_dataset.datasets[0]; dataset = provider.dataset; sample = dataset.sample_list[sample_index]
    assert cpu_batch["images_input"].shape[1] == 8 and cpu_batch["images_output"].shape[1] == 7
    assert cpu_batch["instance_label_output"].shape[1] == 7
    assert len(set(str(v) for v in data["scene_name"])) == 1
    assert int((cpu_batch["instance_label_output"] > 0).sum()) > 0
    tensor_hashes = batch_hashes(cpu_batch); torch.save(cpu_batch, out / "fixed_batch_cpu.pt"); (out / "fixed_batch_hashes.json").write_text(json.dumps({"tensor_hashes": tensor_hashes}, indent=2))
    (out / "manifest_snapshot.json").write_text(json.dumps(manifest, indent=2));
    scene_id = str(data["scene_name"][0]); frame_ids = [int(v) for v in data["frame_ids"][0].tolist()]
    target_gt_instance_ids = [
        sorted(int(v) for v in torch.unique(cpu_batch["instance_label_output"][0, view]).tolist() if int(v) > 0)
        for view in range(int(cpu_batch["instance_label_output"].shape[1]))
    ]
    root_path = Path(provider.root_path) if hasattr(provider, "root_path") else Path(dataset.root_path)
    label_root = Path(provider.label_root) if hasattr(provider, "label_root") else Path(dataset.label_root)
    scene_dir = root_path / scene_id; label_dir = label_root / scene_id
    source_files = [scene_dir / f"{scene_id}.sens"]
    for raw_id in frame_ids[8:]:
        source_files.extend([label_dir / "label-filt" / f"{raw_id}.png", label_dir / "instance-filt" / f"{raw_id}.png"])
    fixed_info = {"manifest_path": str(MANIFEST.resolve()), "manifest_sha256": sha256_file(MANIFEST), "sample_index": sample_index, "sample": jsonable(sample), "scene_id": scene_id, "context_frame_ids": frame_ids[:8], "target_frame_ids": frame_ids[8:], "context_views": 8, "target_views": 7, "same_scene": True, "target_gt_view_count": int(cpu_batch["instance_label_output"].shape[1]), "target_gt_instance_ids": target_gt_instance_ids, "target_gt_instance_count_per_view": [len(v) for v in target_gt_instance_ids], "target_gt_nonzero_pixels": int((cpu_batch["instance_label_output"] > 0).sum()), "source_files": [file_info(path) for path in source_files], "data_root": str(root_path), "label_root": str(label_root), "fixed_batch_shapes": {k: list(v.shape) for k,v in cpu_batch.items() if torch.is_tensor(v)}, "fixed_batch_dtypes": {k: str(v.dtype) for k,v in cpu_batch.items() if torch.is_tensor(v)}, "code_provenance": code_provenance(), "fixed_batch_hashes": tensor_hashes}
    (out / "fixed_batch_manifest.json").write_text(json.dumps(fixed_info, indent=2))
    model = model_registry[opt.model_type](opt).cuda().train(); raw = load_file(str(CKPT), device="cpu")
    counts = {"absolute_gs_head": sum(k.startswith("absolute_gs_head.") for k in raw), "tsh_instance_head": sum(k.startswith("tsh_instance_head.") for k in raw), "decoder_tail": sum(k.startswith("enc_dec_backbone.decoder_blocks.") for k in raw)}
    assert counts == {"absolute_gs_head":24,"tsh_instance_head":50,"decoder_tail":324}; assert not any("pgsr" in k.lower() or "query_memory_refiner" in k.lower() for k in raw)
    load_model_checkpoint(opt, model, acc, 0); optimizer = setup_optimizer(opt, model, acc, 0); model, optimizer, loader = acc.prepare(model, optimizer, loader)
    base = acc.unwrap_model(model); trainable = [n for n,p0 in base.named_parameters() if p0.requires_grad]; assert trainable and all(n.startswith("ga_idu1_head.") for n in trainable)
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k,v in cpu_batch.items()}
    base.ga_idu_eval_gate_override = 0.; model.eval()
    reset_audit_eval_rng()
    with torch.no_grad(): zero = model(batch, compute_quality_metrics=False)
    base_logits = zero["base_unit_logits"].detach().clone(); assert float((zero["unit_logits"]-zero["base_unit_logits"]).abs().max()) <= 1e-6
    metrics = {"environment": {"python": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}, "checkpoint": {"path": str(CKPT), "sha256": sha256_file(CKPT), "counts": counts, "fresh_reset": False, "pgsr_absent": True}, "code_provenance": fixed_info["code_provenance"], "scheduler": {"state": None, "reason": "fixed-batch smoke uses constant LR"}, "config": {k: str(getattr(opt,k)) for k in vars(opt)}, "fixed_batch": fixed_info, "trainable": trainable, "step0": {**evaluate(zero, batch, base_logits, opt), "gate": 0., "identity_max_diff": float((zero["unit_logits"]-zero["base_unit_logits"]).abs().max()), "model_hash": model_hash(base), "ga_hash": ga_hash(base), "train_loss": None}}
    torch.save({"ga_idu1_head": {k[len("ga_idu1_head."):]:v.detach().cpu() for k,v in base.state_dict().items() if k.startswith("ga_idu1_head.")}, "optimizer": optimizer.state_dict(), "scheduler": None, "scheduler_reason": "fixed-batch smoke uses constant LR", "step": 0, "gate": 0., "parameter_hash": ga_hash(base), "rng": torch.get_rng_state()}, out / "step_000.pt")
    torch.save(cache_output(zero), out / "cache_step_000.pt")
    delattr(base, "ga_idu_eval_gate_override"); model.train(); initial = {n:p0.detach().float().clone() for n,p0 in base.named_parameters() if n.startswith("ga_idu1_head.")}
    records = []
    for step in range(a.steps):
        base.ga_idu_gate_eff = min(1., (step+1)/float(max(1,opt.ga_idu_gate_steps))); base.tsh_instance_loss_weight_eff=1.; base.tsh_unit_grad_eff=0.; base.tsh_mbm_u2r_eff=0.; base.teacher_lambda_eff=0.
        optimizer.zero_grad(set_to_none=True)
        with acc.autocast(): cur = model(batch, compute_quality_metrics=False)
        train_parts = loss_parts(cur, batch, opt); loss = cur["loss_instance_group"]; assert torch.isfinite(loss).all(); acc.backward(loss); pre=float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))); post=float(torch.nn.utils.clip_grad_norm_(model.parameters(), opt.gradient_clip)); optimizer.step()
        if step+1 in MILESTONES[1:]:
            model.eval();
            reset_audit_eval_rng()
            with torch.no_grad(): snap=model(batch,compute_quality_metrics=False)
            rec = evaluate(snap,batch,base_logits,opt,reference_output=zero,model=base,before=initial); rec.update({"gate":float(base.ga_idu_gate_eff),"model_hash":model_hash(base),"ga_hash":ga_hash(base),"train_loss":train_parts,"grad_norm_before_clip":pre,"grad_norm_after_clip":post,"optimizer_step":step+1})
            metrics[f"step{step+1}"]=rec; model.train()
            torch.save(cache_output(snap), out / f"cache_step_{step+1:03d}.pt")
        if step+1 in MILESTONES[1:]:
            torch.save({"ga_idu1_head": {k[len("ga_idu1_head."):]:v.detach().cpu() for k,v in base.state_dict().items() if k.startswith("ga_idu1_head.")}, "optimizer": optimizer.state_dict(), "scheduler": None, "scheduler_reason": "fixed-batch smoke uses constant LR", "step":step+1, "gate":float(base.ga_idu_gate_eff), "parameter_hash":ga_hash(base), "rng":torch.get_rng_state()}, out/f"step_{step+1:03d}.pt")
    torch.save({"ga_idu1_head": {k[len("ga_idu1_head."):]:v.detach().cpu() for k,v in base.state_dict().items() if k.startswith("ga_idu1_head.")}, "optimizer": optimizer.state_dict(), "scheduler": None, "scheduler_reason": "fixed-batch smoke uses constant LR", "step":a.steps, "parameter_hash":ga_hash(base)}, out/"step_100.pt")
    metrics["steps"]=a.steps; metrics["all_finite"]=True; (out/"per_step_metrics.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))

if __name__ == "__main__": main()
