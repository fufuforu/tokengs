"""One-formal-batch GT-free metric-cluster smoke.

This script performs no optimizer step and writes only a small JSON report.
It is deliberately separate from the formal evaluators and from training.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_cached_gsplat_extension() -> None:
    """Reuse the validated extension instead of attempting a node JIT build."""
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
from tokengs.options import config_defaults  # noqa: E402
from tokengs.train import load_model_checkpoint  # noqa: E402
from tokengs.models.token_eru.historical_unit_infonce import (  # noqa: E402
    build_historical_soft_unit_targets,
)
from tokengs.models.token_eru.metric_clustering import (  # noqa: E402
    historical_metric_cluster_oracle_audit,
)
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


def _tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _ap_for_probability(probability, labels):
    predictions, scores, pred_ids = [], [], []
    ground_truth, gt_ids = [], []
    for view in range(probability.shape[1]):
        image_id = f"smoke:target:{view}"
        masks, view_scores = masks_from_group_probs(
            probability[:, view],
            void_channel=probability.shape[0] - 1,
            min_mask_area=1,
        )
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
    return {
        "ap25": float(ap["ap_25"]),
        "ap50": float(ap["ap_50"]),
        "ap75": float(ap["ap_75"]),
        "best_iou": float(diag["mean_best_gt_iou"]),
        "recall50": float(diag["recall_iou50"]),
        "pred_gt": float(len(predictions) / max(1, len(ground_truth))),
        "prediction_count": int(len(predictions)),
        "gt_count": int(len(ground_truth)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="workspace/token_eru_dino_metric_gtfree_smoke_v1",
    )
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty smoke directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    base = config_defaults[
        "semantic_v6_absolute_units_true_shared_token_eru1_"
        "dino_metric_treatment_resume500_to700_ddp8"
    ]
    opt = base.evolve(
        resume=str(SOURCE),
        workspace=str(output),
        num_workers=0,
        evaluating=True,
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
    model, loader = accelerator.prepare(model, loader)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()
    # Keep the execution-only diagnostic selector on the actual model options
    # object after Accelerator wrapping; it is never persisted or used by
    # training configs.
    unwrapped.opt.token_eru_dino_eval_mode = "metric_cluster"
    unwrapped.set_token_eru_step(500)
    unwrapped.set_token_eru_dino_metric_step(500)
    batch = _move(next(iter(loader)), accelerator.device)
    if batch["input"].shape[1] != 15 or batch["instance_label_output"].shape[1] != 7:
        raise RuntimeError("smoke batch is not the formal 8+7 protocol")
    with torch.inference_mode():
        with accelerator.autocast():
            result = model(batch, compute_quality_metrics=False)
    cluster = result.get("metric_cluster_output")
    if cluster is None:
        raise RuntimeError(
            "GT-free metric cluster output was not produced; "
            f"model_training={unwrapped.training}, "
            f"eval_mode={getattr(unwrapped.opt, 'token_eru_dino_eval_mode', None)!r}, "
            f"output_keys={sorted(result.keys())}"
        )
    rendered = cluster.rendered_masks
    if rendered.ndim != 6 or rendered.shape[0] != 1 or rendered.shape[2] != 7:
        raise RuntimeError(f"unexpected rendered cluster shape: {tuple(rendered.shape)}")
    if not torch.isfinite(rendered).all():
        raise FloatingPointError("non-finite GT-free cluster render")
    evidence = result["dino_unit_features"]
    embeddings = result["unit_metric_embeddings"]
    if evidence.shape != (1, 1024, 8, 256):
        raise RuntimeError(f"unexpected DINO unit evidence shape: {tuple(evidence.shape)}")
    if embeddings.shape != (1, 1024, 8, 128):
        raise RuntimeError(f"unexpected metric embedding shape: {tuple(embeddings.shape)}")
    labels = batch["instance_label_output"][0].detach().long().cpu().numpy()
    query_probability = (
        result["rendered_instance_group_probability"][0]
        .detach()
        .float()
        .cpu()
        .numpy()[:, :, 0]
    )
    cluster_probability = rendered[0].detach().float().cpu().numpy()[:, :, 0]
    unit_positions = result["gaussians"][..., :3].reshape(
        1, 1024, 8, 8, 3
    ).mean(dim=3).reshape(1, 8192, 3)
    oracle_targets, _ = build_historical_soft_unit_targets(
        result["gaussians"].detach(),
        batch,
        tuple(int(x) for x in batch["instance_label_output"].shape[-2:]),
    )
    oracle_cluster = historical_metric_cluster_oracle_audit(
        embeddings.reshape(1, 8192, 128),
        unit_positions,
        oracle_targets,
        {"values": result["gaussians"], "renderer": unwrapped.gs},
        {
            "cam_view": batch["cam_view"],
            "intrinsics": batch["intrinsics"],
        },
        background_index=0,
        eps=0.5,
        foreground_threshold=0.5,
        foreground_share=0.5,
    )
    oracle_probability = (
        oracle_cluster.rendered_masks[0].detach().float().cpu().numpy()[:, :, 0]
    )
    report = {
        "checkpoint": str(SOURCE),
        "optimizer_step": 500,
        "context_views": 8,
        "target_views": 7,
        "scene": str(batch["scene_name"][0]),
        "dino_context_only": True,
        "formal_gt_inputs_to_cluster": False,
        "unit_objectness_source": "1-softmax(unit_logits.float(),dim=-1)[...,100]",
        "dino_unit_evidence_shape": list(evidence.shape),
        "unit_metric_embeddings_shape": list(embeddings.shape),
        "rendered_masks_shape": list(rendered.shape),
        "cluster_count": [int(x) for x in cluster.cluster_count],
        "cluster_confidence": [x.detach().cpu().tolist() for x in cluster.cluster_confidence],
        "native_query_metrics": _ap_for_probability(query_probability, labels),
        "no_gt_metric_cluster_metrics": _ap_for_probability(cluster_probability, labels),
        "oracle_historical_cluster": {
            "oracle": True,
            "cluster_count": [int(x) for x in oracle_cluster.cluster_count],
            "metrics": _ap_for_probability(oracle_probability, labels),
        },
        "embedding_hash": _tensor_hash(embeddings),
        "finite": True,
        "optimizer_step_executed": False,
        "formal_training_started": False,
        "formal_evaluation_started": False,
    }
    (output / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
