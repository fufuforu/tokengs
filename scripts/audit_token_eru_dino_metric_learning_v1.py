"""Recoverable 100-step fixed-batch audit for ERU-DINO-Metric-v1.

This is deliberately an isolated audit workspace.  It resumes the persisted
ERU step-500 model, uses one deterministic formal 8+7 batch, and never writes
to an ERU training workspace or starts a multi-scene evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_cached_gsplat_extension() -> None:
    so_path = os.environ.get("GSPLAT_PRECOMPILED_SO")
    if not so_path or not os.path.isfile(so_path) or "gsplat_cuda" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location("gsplat_cuda", so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load cached gsplat extension: {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["gsplat_cuda"] = module
    import gsplat

    sys.modules.setdefault("gsplat.csrc", module)
    setattr(gsplat, "csrc", module)


_load_cached_gsplat_extension()

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
from tokengs.train import load_model_checkpoint, setup_optimizer  # noqa: E402
from tokengs.utils.instance_ap import (  # noqa: E402
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)
from scripts.eval_token_eru_short200 import _mask_diagnostics  # noqa: E402


SOURCE = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_"
    "scene_hungarian_resume200_to500_persist_ddp8/checkpoints/"
    "model_step_000500.safetensors"
)
EXPECTED_SOURCE_SHA256 = (
    "cca23e2d24af7a374be6a62f2cf4026645afdf0df49686722cb5277c9e479209"
)
EXPECTED_V12_BATCH_HASH = (
    "ce2227280c967941a53e39b819d5370d47f3cfab09325a285c1767634dff32ec"
)
CONFIG_NAME = (
    "semantic_v6_absolute_units_true_shared_token_eru1_"
    "dino_metric_treatment_resume500_to700_ddp8"
)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _trainable_parameter_hash(model) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        count += 1
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    if count == 0:
        raise RuntimeError("trainable parameter hash would cover an empty set")
    return digest.hexdigest()


def _batch_hash(batch) -> str:
    # This is intentionally byte-compatible with the previous v12 audit.
    tensors = {
        key: value
        for key, value in batch.items()
        if torch.is_tensor(value)
    }
    return hashlib.sha256(
        b"".join(
            key.encode() + value.detach().cpu().contiguous().numpy().tobytes()
            for key, value in sorted(tensors.items())
        )
    ).hexdigest()


def _field_hash(value) -> str | None:
    if not torch.is_tensor(value):
        return None
    return _tensor_hash(value)


def _json_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _input_fingerprint(batch) -> dict:
    frame_ids = batch.get("frame_ids")
    if torch.is_tensor(frame_ids):
        frame_ids = frame_ids.detach().cpu().tolist()
    scene = batch.get("scene_name", ["unknown"])
    if isinstance(scene, (list, tuple)):
        scene = [str(value) for value in scene]
    else:
        scene = [str(scene)]
    result = {
        "scene": scene,
        "frame_ids": _json_value(frame_ids),
        "context_views": 8,
        "target_views": 7,
        "batch_hash_v12_compatible": _batch_hash(batch),
        "tensor_fields": {},
    }
    for key in sorted(batch):
        value = batch[key]
        if torch.is_tensor(value):
            result["tensor_fields"][key] = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": _field_hash(value),
            }
    return result


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {name: 0.0 for name in ("mean", "p10", "p50", "p90")}
    return {
        "mean": float(values.mean()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def _rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _prediction_bundle(probability: np.ndarray, labels: np.ndarray) -> tuple[dict, dict]:
    predictions, scores, pred_ids = [], [], []
    ground_truth, gt_ids = [], []
    duplicate_count = 0
    for view in range(probability.shape[1]):
        image_id = f"fixed:target:{view}"
        masks, view_scores = masks_from_group_probs(
            probability[:, view],
            void_channel=probability.shape[0] - 1,
            min_mask_area=1,
        )
        # The existing evaluator's duplicate diagnostic is IoU >= .5 among
        # masks in one image; it is reported here only as a diagnostic.
        for left in range(len(masks)):
            for right in range(left):
                a = np.asarray(masks[left], dtype=np.float32)
                b = np.asarray(masks[right], dtype=np.float32)
                inter = float((a * b).sum())
                union = float(a.sum() + b.sum() - inter)
                if union > 0 and inter / union >= 0.5:
                    duplicate_count += 1
        predictions.extend(masks)
        scores.extend(view_scores)
        pred_ids.extend([image_id] * len(masks))
        gt = gt_masks_from_instance_map(labels[view], min_mask_area=1)
        ground_truth.extend(gt)
        gt_ids.extend([image_id] * len(gt))
    ap = instance_ap(
        predictions,
        scores,
        ground_truth,
        thresholds=(0.25, 0.5, 0.75),
        vectorized=True,
        pred_image_ids=pred_ids,
        gt_image_ids=gt_ids,
    )
    diag = _mask_diagnostics(predictions, ground_truth, pred_ids, gt_ids)
    return (
        {
            "ap25": float(ap["ap_25"]),
            "ap50": float(ap["ap_50"]),
            "ap75": float(ap["ap_75"]),
            "best_iou": float(diag["mean_best_gt_iou"]),
            "recall25": float(diag["recall_iou25"]),
            "recall50": float(diag["recall_iou50"]),
            "recall75": float(diag["recall_iou75"]),
            "pred_gt": float(len(predictions) / max(1, len(ground_truth))),
            "prediction_count": int(len(predictions)),
            "gt_count": int(len(ground_truth)),
            "duplicate_prediction_count": int(duplicate_count),
        },
        {
            "predictions": predictions,
            "scores": scores,
            "prediction_ids": pred_ids,
            "ground_truth": ground_truth,
            "ground_truth_ids": gt_ids,
        },
    )


def _mask_pair_iou(left, right) -> float:
    inter = float((left * right).sum())
    union = float(left.sum() + right.sum() - inter)
    return inter / union if union > 0 else 0.0


def _query_cluster_iou(query_probability, cluster_probability) -> float:
    _, query = _prediction_bundle(query_probability, np.zeros((7, query_probability.shape[-2], query_probability.shape[-1]), dtype=np.int64))
    _, cluster = _prediction_bundle(cluster_probability, np.zeros((7, cluster_probability.shape[-2], cluster_probability.shape[-1]), dtype=np.int64))
    values = []
    for image_id in sorted(set(query["prediction_ids"]) | set(cluster["prediction_ids"])):
        q = [mask for mask, ident in zip(query["predictions"], query["prediction_ids"]) if ident == image_id]
        c = [mask for mask, ident in zip(cluster["predictions"], cluster["prediction_ids"]) if ident == image_id]
        for mask in c:
            values.append(max((_mask_pair_iou(mask, other) for other in q), default=0.0))
    return float(np.mean(values)) if values else 0.0


def _embedding_diagnostics(embeddings, targets, valid, cluster_ids, cluster_counts, objectness):
    emb = torch.nn.functional.normalize(embeddings.float(), dim=-1).detach().cpu().numpy()
    target = targets.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy().astype(bool)
    objectness_np = objectness.detach().cpu().numpy()
    same_values, different_values = [], []
    fragmentation, collision_labels = [], []
    rng = np.random.default_rng(3407)
    nn_purities = []
    aucs = []
    for b in range(emb.shape[0]):
        labels = target[b].argmax(axis=-1)
        fg = valid_np[b] & (labels != 0)
        indices = np.flatnonzero(fg)
        if indices.size > 2048:
            indices = rng.choice(indices, size=2048, replace=False)
        if indices.size:
            cosine = emb[b, indices] @ emb[b, indices].T
            same = labels[indices][:, None] == labels[indices][None, :]
            upper = np.triu(np.ones_like(same, dtype=bool), 1)
            same_values.extend(cosine[same & upper].tolist())
            different_values.extend(cosine[(~same) & upper].tolist())
            nearest = cosine.copy()
            np.fill_diagonal(nearest, -np.inf)
            nn = nearest.argmax(axis=1)
            nn_purity = float((labels[indices] == labels[indices][nn]).mean())
        else:
            nn_purity = 0.0
        nn_purities.append(nn_purity)
        cluster_units = cluster_ids[b].detach().cpu().numpy().reshape(-1, 8)[:, 0]
        for gt in np.unique(labels[fg]):
            members = fg & (labels == gt)
            fragmentation.append(int(np.unique(cluster_units[members]).size))
        for cluster in range(int(cluster_counts[b])):
            members = fg & (cluster_units == cluster)
            collision_labels.append(int(np.unique(labels[members]).size > 1))
        fg_scores = objectness_np[b, fg]
        bg_scores = objectness_np[b, valid_np[b] & (labels == 0)]
        auc = _rank_auc(
            np.concatenate([fg_scores, bg_scores]),
            np.concatenate([np.ones(fg_scores.size), np.zeros(bg_scores.size)]),
        )
        aucs.append(auc)
    same = np.asarray(same_values, dtype=np.float32)
    different = np.asarray(different_values, dtype=np.float32)
    return {
        "same_instance_cosine": _percentiles(same),
        "different_instance_cosine": _percentiles(different),
        "positive_negative_cosine_gap": float((same.mean() if same.size else 0.0) - (different.mean() if different.size else 0.0)),
        "one_nn_instance_purity": float(np.mean(nn_purities)) if nn_purities else 0.0,
        "gt_clusters_per_instance": _percentiles(np.asarray(fragmentation, dtype=np.float32)),
        "gt_fragmentation_rate": float(np.mean(np.asarray(fragmentation) > 1)) if fragmentation else 0.0,
        "cluster_collision_rate": float(np.mean(collision_labels)) if collision_labels else 0.0,
        "foreground_objectness": _percentiles(np.concatenate([objectness_np[b, valid_np[b] & (target[b].argmax(-1) != 0)] for b in range(emb.shape[0])])),
        "background_objectness": _percentiles(np.concatenate([objectness_np[b, valid_np[b] & (target[b].argmax(-1) == 0)] for b in range(emb.shape[0])])),
        "foreground_auroc": float(np.mean(aucs)) if aucs else 0.0,
    }


def _snapshot(model, output: Path, local_step: int, completed_step: int, batch_fp: dict, rng: dict) -> dict:
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    state = {}
    for name, tensor in model.state_dict().items():
        if name in trainable:
            state[name] = tensor.detach().cpu().contiguous()
    if not state:
        raise RuntimeError("diagnostic snapshot contains no trainable keys")
    path = output / f"step_{local_step:03d}_trainable.safetensors"
    from safetensors.torch import save_file

    save_file(state, str(path))
    metadata = {
        "local_step": int(local_step),
        "optimizer_step": int(completed_step),
        "source_step": 500,
        "checkpoint_path": str(path),
        "checkpoint_sha256": _sha256_file(path),
        "state_key_count": len(state),
        "state_keys": sorted(state),
        "batch_fingerprint": batch_fp,
        "trainable_parameter_names": sorted(trainable),
        "trainable_parameter_hash": _trainable_parameter_hash(model),
        "rng": rng,
    }
    (output / f"metadata_step_{local_step:06d}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _rng_state() -> dict:
    result = {"cpu": torch.get_rng_state().tolist(), "python": repr(random.getstate())}
    if torch.cuda.is_available():
        result["cuda"] = [item.tolist() for item in torch.cuda.get_rng_state_all()]
    return result


def _build_runtime(output: Path):
    base = config_defaults[CONFIG_NAME]
    opt = dataclasses.replace(
        base,
        resume=str(SOURCE),
        workspace=str(output),
        num_workers=0,
        batch_size=1,
        tsh_ddp8=False,
        eval_before_training=False,
        use_wandb=False,
        token_eru_dino_eval_mode="metric_cluster",
    )
    torch.manual_seed(int(opt.seed))
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
    optimizer = setup_optimizer(opt, model, accelerator, 0)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    return opt, accelerator, model, optimizer, loader


def _eval_once(model, unwrapped, batch, completed_step: int, output: Path, save_tensors: bool = False) -> dict:
    unwrapped.eval()
    unwrapped.opt.token_eru_dino_eval_mode = "metric_cluster"
    unwrapped.set_token_eru_step(completed_step)
    unwrapped.set_token_eru_dino_metric_step(completed_step)
    with torch.inference_mode():
        with torch.autocast(device_type=batch["input"].device.type, enabled=False):
            result = model(batch, compute_quality_metrics=False)
    if result.get("metric_cluster_output") is None:
        raise RuntimeError("GT-free metric cluster output missing at eval")
    query_prob = result["rendered_instance_group_probability"][0].detach().float().cpu().numpy()[:, :, 0]
    cluster_prob = result["metric_cluster_output"].rendered_masks[0].detach().float().cpu().numpy()[:, :, 0]
    labels = batch["instance_label_output"][0].detach().long().cpu().numpy()
    query_metrics, query_bundle = _prediction_bundle(query_prob, labels)
    cluster_metrics, cluster_bundle = _prediction_bundle(cluster_prob, labels)
    oracle_targets, valid = build_historical_soft_unit_targets(
        result["gaussians"].detach(), batch, tuple(int(x) for x in labels.shape[-2:])
    )
    embeddings = result["unit_metric_embeddings"].reshape(1, 8192, -1)
    unit_logits = result["unit_logits"].reshape(1, 8192, -1)
    objectness = (1.0 - torch.softmax(unit_logits.float(), dim=-1)[..., 100])
    positions = result["gaussians"][..., :3].reshape(1, 1024, 8, 8, 3).mean(dim=3).reshape(1, 8192, 3)
    oracle = historical_metric_cluster_oracle_audit(
        embeddings, positions, oracle_targets, {"values": result["gaussians"], "renderer": unwrapped.gs},
        {"cam_view": batch["cam_view"], "intrinsics": batch["intrinsics"]},
        background_index=0, eps=0.5, foreground_threshold=0.5, foreground_share=0.5,
    )
    oracle_prob = oracle.rendered_masks[0].detach().float().cpu().numpy()[:, :, 0]
    oracle_metrics, _ = _prediction_bundle(oracle_prob, labels)
    metric_stats = historical_soft_unit_infonce(embeddings, oracle_targets, valid, temperature=0.1)
    cluster = result["metric_cluster_output"]
    emb_diag = _embedding_diagnostics(
        embeddings, oracle_targets, valid, cluster.gaussian_cluster_ids, cluster.cluster_count, objectness
    )
    gtfree_oracle_gap = float(cluster_metrics["ap50"] - oracle_metrics["ap50"])
    snapshot_outputs = {
        "unit_metric_embeddings": embeddings[0].detach().cpu(),
        "unit_logits": unit_logits[0].detach().cpu(),
        "unit_objectness": objectness[0].detach().cpu(),
        "native_rendered_masks": torch.from_numpy(query_prob),
        "gtfree_cluster_ids": cluster.gaussian_cluster_ids[0].detach().cpu(),
        "gtfree_rendered_masks": torch.from_numpy(cluster_prob),
    }
    if save_tensors:
        torch.save(snapshot_outputs, output / f"step_{completed_step - 500:03d}_eval_tensors.pt")
    return {
        "local_step": int(completed_step - 500),
        "completed_optimizer_step": int(completed_step),
        "gates": {"dino": float(result["dino_gate"]), "r2u_u2r": list(unwrapped.token_eru_gates(completed_step, unwrapped.opt))},
        "native_query": query_metrics,
        "metric_cluster_gtfree": {
            **cluster_metrics,
            "foreground_cluster_count": int(cluster.cluster_count[0]),
            "nonempty_cluster_count": int(np.unique(cluster.gaussian_cluster_ids[0].detach().cpu().numpy()).size - 1),
            "void_ratio": float(cluster_prob[-1].mean()),
        },
        "metric_cluster_oracle_audit": {
            "oracle": True,
            **oracle_metrics,
            "foreground_cluster_count": int(oracle.cluster_count[0]),
        },
        "oracle_no_gt_ap50_gap": gtfree_oracle_gap,
        "query_cluster_matched_iou": _query_cluster_iou(query_prob, cluster_prob),
        "metric_loss": {
            "loss": float(metric_stats.loss),
            "valid_unit_count": int(metric_stats.valid_unit_count),
            "positive_pair_count": int(metric_stats.positive_pair_count),
            "negative_pair_count": int(metric_stats.negative_pair_count),
            "mean_positive_similarity": float(metric_stats.mean_positive_similarity),
            "mean_negative_similarity": float(metric_stats.mean_negative_similarity),
            "target_entropy": float(metric_stats.target_entropy),
        },
        "embedding_diagnostics": emb_diag,
        "finite": True,
        "gt_free_cluster_formal": True,
        "oracle_formal": False,
    }


def _restore_check(snapshot_dir: Path, output: Path) -> dict:
    snapshot = snapshot_dir / "step_100_trainable.safetensors"
    expected = torch.load(snapshot_dir / "step_100_eval_tensors.pt", map_location="cpu", weights_only=False)
    opt, accelerator, model, _, loader = _build_runtime(output / "restore_runtime")
    unwrapped = accelerator.unwrap_model(model)
    from safetensors.torch import load_file

    state = load_file(str(snapshot), device="cpu")
    native_current = torch.nn.Module.state_dict(unwrapped)
    missing = sorted(set(state) - set(native_current))
    shape_bad = sorted(
        key
        for key in state
        if key in native_current and tuple(state[key].shape) != tuple(native_current[key].shape)
    )
    if missing or shape_bad:
        raise RuntimeError(f"snapshot restore mismatch missing={missing} shape_bad={shape_bad}")
    # PromptTokenGS has a compatibility load_state_dict override for legacy
    # checkpoints.  A trainable-only audit snapshot is already namespaced in
    # the live module, so bypass that legacy filter and use PyTorch's native
    # loader with strict=False for the intentionally absent frozen keys.
    native_result = torch.nn.Module.load_state_dict(
        unwrapped,
        {key: value.to(next(unwrapped.parameters()).device) for key, value in state.items()},
        strict=False,
    )
    if native_result.unexpected_keys:
        raise RuntimeError(f"unexpected trainable-only snapshot keys: {native_result.unexpected_keys}")
    batch = _move(next(iter(loader)), accelerator.device)
    actual = _eval_once(model, unwrapped, batch, 600, output / "restore_runtime", save_tensors=False)
    actual_tensors = {
        "unit_metric_embeddings": actual,
    }
    # Re-run the compact tensor capture through the same path without writing
    # a second snapshot; the eval record is independently checked below.
    unwrapped.eval()
    with torch.inference_mode():
        result = model(batch, compute_quality_metrics=False)
    cluster = result["metric_cluster_output"]
    got = {
        "unit_metric_embeddings": result["unit_metric_embeddings"].reshape(1, 8192, -1)[0].detach().cpu(),
        "unit_logits": result["unit_logits"].reshape(1, 8192, -1)[0].detach().cpu(),
        "unit_objectness": (1.0 - torch.softmax(result["unit_logits"].reshape(1,8192,-1).float(), dim=-1)[...,100])[0].detach().cpu(),
        "native_rendered_masks": result["rendered_instance_group_probability"][0].detach().float().cpu()[:, :, 0],
        "gtfree_cluster_ids": cluster.gaussian_cluster_ids[0].detach().cpu(),
        "gtfree_rendered_masks": cluster.rendered_masks[0].detach().float().cpu()[:, :, 0],
    }
    diffs = {key: float((got[key].float() - expected[key].float()).abs().max()) for key in expected}
    if any(value > 1e-5 for value in diffs.values()):
        raise RuntimeError(f"step100 independent restore mismatch: {diffs}")
    return {"strict": True, "max_abs_diff": diffs, "all_match": True, "optimizer_step_executed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="workspace/token_eru_dino_metric_learning_audit_v1")
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty audit directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source_sha = _sha256_file(SOURCE)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"source SHA256 mismatch: {source_sha}")
    opt, accelerator, model, optimizer, loader = _build_runtime(output)
    unwrapped = accelerator.unwrap_model(model)
    batch = _move(next(iter(loader)), accelerator.device)
    if batch["input"].shape[1] != 15 or batch["instance_label_output"].shape[1] != 7:
        raise RuntimeError("fixed batch is not formal 8+7")
    batch_fp = _input_fingerprint(batch)
    if batch_fp["batch_hash_v12_compatible"] != EXPECTED_V12_BATCH_HASH:
        raise RuntimeError(
            "fixed batch differs from v12: "
            f"expected={EXPECTED_V12_BATCH_HASH} actual={batch_fp['batch_hash_v12_compatible']}"
        )
    trainable_names = sorted(name for name, p in unwrapped.named_parameters() if p.requires_grad)
    report = {
        "source_checkpoint": str(SOURCE),
        "source_sha256": source_sha,
        "config": CONFIG_NAME,
        "protocol": batch_fp,
        "trainable_parameter_names": trainable_names,
        "initial_trainable_parameter_hash": _trainable_parameter_hash(unwrapped),
        "optimizer_step_source": 500,
        "milestones": [],
        "snapshots": [],
        "formal_training_started": False,
        "formal_evaluation_started": False,
        "short200_started": False,
    }
    milestone_set = {0, 1, 5, 25, 50, 100}
    # Evaluate the untouched source first, then execute every intervening
    # optimizer step.  This preserves the intended 100-step trajectory while
    # keeping the requested milestone evaluations independent forwards.
    for local_step in range(0, 101):
        if local_step > 0:
            unwrapped.train()
            completed = 500 + local_step
            unwrapped.set_token_eru_step(completed)
            unwrapped.set_token_eru_dino_metric_step(completed)
            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                train_output = model(batch, compute_quality_metrics=False)
            loss = train_output["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at local step {local_step}")
            accelerator.backward(loss)
            pre_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in unwrapped.parameters() if p.requires_grad):
                raise FloatingPointError(f"non-finite parameter at local step {local_step}")
            report.setdefault("training_records", []).append({
                "local_step": local_step,
                "completed_optimizer_step": completed,
                "loss": float(loss.detach()),
                "loss_instance_metric": float(train_output["loss_instance_metric"].detach()),
                "loss_instance_metric_weighted": float(train_output["loss_instance_metric_weighted"].detach()),
                "pre_clip_grad_norm": pre_clip,
                "parameter_hash": _trainable_parameter_hash(unwrapped),
                "finite": True,
            })
        else:
            completed = 500
        if local_step not in milestone_set:
            continue
        rng_before = _rng_state()
        record = _eval_once(model, unwrapped, batch, completed, output, save_tensors=(local_step == 100))
        record["trainable_parameter_hash"] = _trainable_parameter_hash(unwrapped)
        report["milestones"].append(record)
        if local_step in (25, 50, 100):
            metadata = _snapshot(unwrapped, output, local_step, completed, batch_fp, rng_before)
            report["snapshots"].append(metadata)
    report["fixed_batch_reproducible"] = True
    report["source_trainable_parameter_count"] = sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)
    (output / "fixed_batch_learning_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    restore = _restore_check(output, output)
    report["step100_independent_restore"] = restore
    (output / "fixed_batch_learning_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"output": str(output), "step100_independent_restore": restore, "fixed_batch_reproducible": True}, indent=2))


if __name__ == "__main__":
    main()
