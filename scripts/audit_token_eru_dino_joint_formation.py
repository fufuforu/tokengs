"""100-step matched fixed-batch audit for TokenGS-ERU JointFormation.

This diagnostic starts from the persisted ERU@500 model-only parent, creates a
fresh optimizer, and never starts the formal one-epoch run.  It deliberately
uses the native query branch as the formal output; metric clustering is only a
diagnostic side channel.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.token_eru.historical_unit_infonce import (  # noqa: E402
    build_historical_soft_unit_targets,
    historical_soft_unit_infonce,
)
from tokengs.models.token_eru.metric_clustering import (  # noqa: E402
    historical_metric_cluster_oracle_audit,
)
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import (  # noqa: E402
    configure_joint_formation_trainability,
    load_model_checkpoint,
    setup_optimizer,
)
from tokengs.utils.instance_ap import (  # noqa: E402
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)

CONFIG_NAME = (
    "semantic_v6_absolute_units_true_shared_token_eru1_"
    "dino_metric_joint_formation_v1_ddp8"
)
SOURCE = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_"
    "scene_hungarian_resume200_to500_persist_ddp8/checkpoints/"
    "model_step_000500.safetensors"
)
SOURCE_SHA256 = "cca23e2d24af7a374be6a62f2cf4026645afdf0df49686722cb5277c9e479209"
EXPECTED_BATCH_HASH = "ce2227280c967941a53e39b819d5370d47f3cfab09325a285c1767634dff32ec"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _batch_hash(batch) -> str:
    return hashlib.sha256(
        b"".join(
            key.encode() + value.detach().cpu().contiguous().numpy().tobytes()
            for key, value in sorted(batch.items())
            if torch.is_tensor(value)
        )
    ).hexdigest()


def _parameter_hash(model, trainable_only=False) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if trainable_only and not parameter.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        count += 1
    if count == 0:
        raise RuntimeError("parameter hash covers an empty set")
    return digest.hexdigest()


def _finite_model(model) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def _prediction_metrics(probability, labels):
    predictions, scores, ids = [], [], []
    ground_truth, gt_ids = [], []
    for view in range(probability.shape[1]):
        image_id = f"fixed:target:{view}"
        masks, view_scores = masks_from_group_probs(
            probability[:, view], void_channel=probability.shape[0] - 1, min_mask_area=1
        )
        predictions.extend(masks)
        scores.extend(view_scores)
        ids.extend([image_id] * len(masks))
        gt = gt_masks_from_instance_map(labels[view], min_mask_area=1)
        ground_truth.extend(gt)
        gt_ids.extend([image_id] * len(gt))
    ap = instance_ap(
        predictions, scores, ground_truth, thresholds=(0.25, 0.5, 0.75),
        vectorized=True, pred_image_ids=ids, gt_image_ids=gt_ids,
    )
    best, r25, r50, r75 = [], [], [], []
    for gt, gt_id in zip(ground_truth, gt_ids):
        best_iou = 0.0
        for pred, pred_id in zip(predictions, ids):
            if pred_id != gt_id:
                continue
            inter = float((pred * gt).sum())
            union = float(pred.sum() + gt.sum() - inter)
            best_iou = max(best_iou, inter / union if union else 0.0)
        best.append(best_iou)
    for threshold, output in ((0.25, r25), (0.5, r50), (0.75, r75)):
        output.extend(value >= threshold for value in best)
    return {
        "ap25": float(ap["ap_25"]), "ap50": float(ap["ap_50"]), "ap75": float(ap["ap_75"]),
        "best_iou": float(sum(best) / max(1, len(best))),
        "recall25": float(sum(r25) / max(1, len(r25))),
        "recall50": float(sum(r50) / max(1, len(r50))),
        "recall75": float(sum(r75) / max(1, len(r75))),
        "pred_gt": float(len(predictions) / max(1, len(ground_truth))),
        "prediction_count": len(predictions), "gt_count": len(ground_truth),
    }


def _snapshot_tensors(result, model, path: Path) -> None:
    cluster = result.get("metric_cluster_output")
    if cluster is None:
        raise RuntimeError("GT-free metric cluster output missing")
    tensors = {
        "unit_metric_embeddings": result["unit_metric_embeddings"].detach().cpu(),
        "unit_logits": result["unit_logits"].detach().cpu(),
        "unit_objectness": (
            1.0 - torch.softmax(result["unit_logits"].float(), dim=-1)[..., 100]
        ).detach().cpu(),
        "native_rendered_masks": result["rendered_instance_group_probability"].detach().cpu(),
        "gtfree_cluster_ids": cluster.gaussian_cluster_ids.detach().cpu(),
        "gtfree_rendered_masks": cluster.rendered_masks.detach().cpu(),
    }
    torch.save(tensors, path)


def _build_runtime(output: Path):
    base = config_defaults[CONFIG_NAME]
    opt = dataclasses.replace(
        base, resume=str(SOURCE), workspace=str(output), num_workers=0,
        batch_size=1, tsh_ddp8=False, eval_before_training=False,
        use_wandb=False, token_eru_dino_eval_mode="metric_cluster",
    )
    torch.manual_seed(int(opt.seed))
    random.seed(int(opt.seed))
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    loader, _, _, _ = get_multi_dataloader(opt, accelerator)
    model = model_registry[opt.model_type](opt)
    load_model_checkpoint(opt, model, accelerator, 0)
    if not getattr(model, "_token_eru_loaded_from_checkpoint", False):
        model.initialize_token_eru_from_reconstruction()
    configure_joint_formation_trainability(model, opt)
    optimizer = setup_optimizer(opt, model, accelerator, 0)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    return opt, accelerator, model, optimizer, loader


def _evaluate(model, unwrapped, batch, local_step: int, tensor_path: Path | None = None):
    unwrapped.eval()
    unwrapped.set_token_eru_step(local_step)
    unwrapped.set_token_eru_dino_metric_step(local_step)
    with torch.inference_mode():
        result = model(batch, compute_quality_metrics=False)
    cluster = result.get("metric_cluster_output")
    if cluster is None:
        raise RuntimeError("JointFormation eval did not produce GT-free cluster diagnostic")
    labels = batch["instance_label_output"][0].detach().long().cpu().numpy()
    query_prob = result["rendered_instance_group_probability"][0].detach().float().cpu().numpy()[:, :, 0]
    cluster_prob = cluster.rendered_masks[0].detach().float().cpu().numpy()[:, :, 0]
    query_metrics = _prediction_metrics(query_prob, labels)
    cluster_metrics = _prediction_metrics(cluster_prob, labels)
    targets, valid = build_historical_soft_unit_targets(
        result["gaussians"].detach(), batch, tuple(labels.shape[-2:])
    )
    embeddings = result["unit_metric_embeddings"].reshape(1, 8192, -1)
    logits = result["unit_logits"].reshape(1, 8192, -1)
    objectness = 1.0 - torch.softmax(logits.float(), dim=-1)[..., 100]
    positions = result["gaussians"][..., :3].reshape(1, 1024, 8, 8, 3).mean(dim=3).reshape(1, 8192, 3)
    oracle = historical_metric_cluster_oracle_audit(
        embeddings, positions, targets,
        {"values": result["gaussians"], "renderer": unwrapped.gs},
        {"cam_view": batch["cam_view"], "intrinsics": batch["intrinsics"]},
        background_index=0, eps=0.5, foreground_threshold=0.5, foreground_share=0.5,
    )
    oracle_prob = oracle.rendered_masks[0].detach().float().cpu().numpy()[:, :, 0]
    oracle_metrics = _prediction_metrics(oracle_prob, labels)
    metric_stats = historical_soft_unit_infonce(embeddings, targets, valid, temperature=0.1)
    if tensor_path is not None:
        _snapshot_tensors(result, unwrapped, tensor_path)
    return {
        "local_step": int(local_step),
        "native_query": query_metrics,
        "metric_cluster_gtfree": {
            **cluster_metrics,
            "foreground_cluster_count": int(cluster.cluster_count[0]),
            "void_ratio": float(cluster_prob[-1].mean()),
        },
        "metric_cluster_oracle_audit": {"oracle": True, **oracle_metrics},
        "oracle_no_gt_ap50_gap": float(cluster_metrics["ap50"] - oracle_metrics["ap50"]),
        "metric_loss": {
            "loss": float(metric_stats.loss),
            "valid_unit_count": int(metric_stats.valid_unit_count),
            "positive_pair_count": int(metric_stats.positive_pair_count),
            "negative_pair_count": int(metric_stats.negative_pair_count),
            "mean_positive_similarity": float(metric_stats.mean_positive_similarity),
            "mean_negative_similarity": float(metric_stats.mean_negative_similarity),
            "target_entropy": float(metric_stats.target_entropy),
        },
        "dino_gate": float(result["dino_gate"]),
        "finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="workspace/token_eru_dino_joint_formation_v1_fixed_batch")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty fixed-batch workspace: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if _sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("parent ERU@500 SHA256 mismatch")
    opt, accelerator, model, optimizer, loader = _build_runtime(output)
    unwrapped = accelerator.unwrap_model(model)
    batch = _move(next(iter(loader)), accelerator.device)
    batch_hash = _batch_hash(batch)
    if batch_hash != EXPECTED_BATCH_HASH:
        raise RuntimeError(f"fixed-batch mismatch: expected {EXPECTED_BATCH_HASH}, got {batch_hash}")
    if batch["input"].shape[1] != 15 or batch["instance_label_output"].shape[1] != 7:
        raise RuntimeError("fixed batch is not the formal 8+7 protocol")
    report = {
        "config": CONFIG_NAME, "source_checkpoint": str(SOURCE),
        "source_sha256": SOURCE_SHA256, "batch_hash": batch_hash,
        "trainable_parameter_hash_step0": _parameter_hash(unwrapped, True),
        "trainability": configure_joint_formation_trainability(unwrapped, opt),
        "milestones": [], "training_records": [],
        "formal_training_started": False, "formal_evaluation_started": False,
    }
    milestone_set = {0, 1, 5, 25, 50, 100}
    for local_step in range(101):
        if local_step > 0:
            unwrapped.train()
            unwrapped.set_token_eru_step(local_step)
            unwrapped.set_token_eru_dino_metric_step(local_step)
            optimizer.zero_grad(set_to_none=True)
            train_output = model(batch, compute_quality_metrics=False)
            loss = train_output["loss"]
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"non-finite loss at step {local_step}")
            accelerator.backward(loss)
            pre_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            if not _finite_model(unwrapped):
                raise FloatingPointError(f"non-finite parameter at step {local_step}")
            report["training_records"].append({
                "local_step": local_step, "loss": float(loss.detach()),
                "loss_rgb": float(train_output.get("loss_rgb", 0.0)),
                "loss_instance_group": float(train_output.get("loss_instance_group", 0.0)),
                "loss_instance_metric": float(train_output.get("loss_instance_metric", 0.0)),
                "pre_clip_grad_norm": pre_clip,
                "parameter_hash": _parameter_hash(unwrapped, True), "finite": True,
            })
        if local_step in milestone_set:
            record = _evaluate(
                model, unwrapped, batch, local_step,
                tensor_path=output / "step_100_eval_tensors.pt" if local_step == 100 else None,
            )
            record["parameter_hash"] = _parameter_hash(unwrapped, True)
            report["milestones"].append(record)
            if local_step in (25, 50, 100):
                state = {
                    name: parameter.detach().cpu().contiguous()
                    for name, parameter in unwrapped.named_parameters()
                    if parameter.requires_grad
                }
                path = output / f"step_{local_step:03d}_trainable.safetensors"
                save_file(state, str(path))
                report.setdefault("snapshots", []).append({
                    "local_step": local_step, "path": str(path),
                    "sha256": _sha256(path), "keys": sorted(state),
                })
    report["step100_independent_restore"] = False
    restore_dir = output / "restore_runtime"
    restore_dir.mkdir()
    opt2, acc2, model2, _, loader2 = _build_runtime(restore_dir)
    base2 = acc2.unwrap_model(model2)
    state = load_file(str(output / "step_100_trainable.safetensors"), device="cpu")
    result = torch.nn.Module.load_state_dict(
        base2, {key: value.to(acc2.device) for key, value in state.items()}, strict=False
    )
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected trainable-only keys: {result.unexpected_keys}")
    batch2 = _move(next(iter(loader2)), acc2.device)
    tensor_expected = torch.load(output / "step_100_eval_tensors.pt", map_location="cpu", weights_only=False)
    base2.eval(); base2.set_token_eru_step(100); base2.set_token_eru_dino_metric_step(100)
    with torch.inference_mode():
        got_result = model2(batch2, compute_quality_metrics=False)
    got_cluster = got_result["metric_cluster_output"]
    got = {
        "unit_metric_embeddings": got_result["unit_metric_embeddings"].detach().cpu(),
        "unit_logits": got_result["unit_logits"].detach().cpu(),
        "unit_objectness": (1.0 - torch.softmax(got_result["unit_logits"].float(), dim=-1)[..., 100]).detach().cpu(),
        "native_rendered_masks": got_result["rendered_instance_group_probability"].detach().cpu(),
        "gtfree_cluster_ids": got_cluster.gaussian_cluster_ids.detach().cpu(),
        "gtfree_rendered_masks": got_cluster.rendered_masks.detach().cpu(),
    }
    diffs = {key: float((got[key].float() - value.float()).abs().max()) for key, value in tensor_expected.items()}
    if any(value > 1e-5 for value in diffs.values()):
        raise RuntimeError(f"step100 independent restore mismatch: {diffs}")
    report["step100_independent_restore"] = {"valid": True, "max_abs_diff": diffs}
    (output / "fixed_batch_learning_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
