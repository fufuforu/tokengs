"""Single-pair SIU3R reference smoke runner.

This is invoked only by ``run_siu3r_official_reference_smoke.sh`` after it has
verified that the official processed data and checkpoint exist.  The loader is
restricted to official pair index 0 and the Trainer is configured for one
validation batch, one GPU, and no optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def digest(value: object) -> str:
    if isinstance(value, dict) and all(hasattr(item, "detach") for item in value.values()):
        h = hashlib.sha256()
        for key in sorted(value):
            h.update(str(key).encode("utf-8"))
            tensor = value[key].detach().cpu().contiguous()
            h.update(str(tensor.dtype).encode("ascii"))
            h.update(repr(tuple(tensor.shape)).encode("ascii"))
            h.update(tensor.numpy().tobytes())
        return h.hexdigest()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return hashlib.sha256(value.tobytes()).hexdigest()
    return hashlib.sha256(repr(value).encode()).hexdigest()


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _gt_maps(mask_labels, class_labels):
    masks = _numpy(mask_labels[0])
    classes = _numpy(class_labels[0]).astype(np.int64)
    n_views, height, width = masks.shape[1:]
    instance = np.zeros((n_views, height, width), dtype=np.int64)
    semantic = np.zeros_like(instance)
    for index, (mask, cls) in enumerate(zip(masks, classes), start=1):
        instance[mask == 1] = index
        semantic[mask == 1] = int(cls) + 1
    return semantic, instance


def _save_prediction_bundle(path: Path, batch, render_output, context_semantic_ids,
                            context_instance_ids, target_semantic_ids,
                            target_instance_ids, seg_infos):
    context_semantic_gt, context_instance_gt = _gt_maps(
        batch["context_mask_labels"], batch["context_class_labels"]
    )
    target_semantic_gt, target_instance_gt = _gt_maps(
        batch["target_mask_labels"], batch["target_class_labels"]
    )
    infos = seg_infos[0]
    labels = np.asarray([int(item["label_id"]) + 1 for item in infos], dtype=np.int64)
    scores = np.asarray([float(item["score"]) for item in infos], dtype=np.float32)
    payload = {
        "context_semantic_pred": _numpy(context_semantic_ids[0]).astype(np.int64),
        "context_instance_pred": _numpy(context_instance_ids[0]).astype(np.int64),
        "context_semantic_gt": context_semantic_gt,
        "context_instance_gt": context_instance_gt,
        "target_semantic_pred": _numpy(target_semantic_ids[0]).astype(np.int64),
        "target_instance_pred": _numpy(target_instance_ids[0]).astype(np.int64),
        "target_semantic_gt": target_semantic_gt,
        "target_instance_gt": target_instance_gt,
        "target_rgb_pred": _numpy(render_output["render_color"][0]).astype(np.float32),
        "target_rgb_gt": _numpy(batch["target_views_images"][0]).astype(np.float32),
        "target_depth_pred": _numpy(render_output["render_depth"][0]).astype(np.float32),
        "target_depth_gt": _numpy(batch["target_views_depths"][0]).astype(np.float32),
        "context_pred_labels": labels,
        "context_pred_scores": scores,
        "target_pred_labels": labels,
        "target_pred_scores": scores,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **payload)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", default="3090")
    parser.add_argument("--official-repo", type=Path, default=Path("/space/mawb/SIU3R"))
    parser.add_argument("--prediction-bundle", type=Path, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    official_repo = args.official_repo.resolve()
    if not (official_repo / "src" / "pipeline.py").is_file():
        raise SystemExit(f"official repository source is missing: {official_repo}")
    sys.path.insert(0, str(official_repo))
    import torch
    import lightning as L
    from lightning import Trainer
    from omegaconf import OmegaConf
    from src.config import (
        PANOPTIC_SEMANTIC2NAME,
        STUFF_CLASSES,
        THING_CLASSES,
        RootCfg,
        load_typed_config,
    )
    from src.data.datamodules.scannet_datamodule import ScanNetDataModule
    from src.pipeline import Pipeline

    if not torch.cuda.is_available():
        raise SystemExit("official reference smoke requires one CUDA device")
    cfg = OmegaConf.load(official_repo / "configs/main.yaml")
    cfg.mode = "val"
    cfg.ckpt_path = str(args.checkpoint)
    cfg.trainer.devices = 1
    cfg.trainer.strategy = "ddp_find_unused_parameters_true"
    cfg.datamodule.dataset_cfg.data_dir = str(args.data_dir)
    cfg.datamodule.dataset_cfg.val_pair_json = str(args.pairs)
    cfg.datamodule.val_loader_cfg.batch_size = 1
    cfg.datamodule.val_loader_cfg.num_workers = 0
    cfg.pipeline.evaluator.device = "cuda"
    cfg.pipeline.evaluator.eval_path = None
    # The official bind_cfg obtains output_path from HydraConfig, which is
    # only initialized by the hydra-decorated src/run.py entrypoint.  This
    # isolated pair runner uses the same bindings explicitly so it can cap the
    # dataloader at pair 0 without modifying official source.
    cfg.output_path = str(args.output.parent)
    cfg.pipeline.model.image_size = (
        cfg.datamodule.dataset_cfg.image_height,
        cfg.datamodule.dataset_cfg.image_width,
    )
    cfg.pipeline.model.pretrained_weights_path = cfg.pipeline.pretrained_weights_path
    cfg.pipeline.visualizer.write_to = cfg.output_path
    cfg.pipeline.visualizer.dataset_name = cfg.datamodule.dataset_cfg.name
    cfg.pipeline.evaluator.dataset_name = cfg.datamodule.dataset_cfg.name
    cfg.datamodule.dataset_cfg.num_extra_target_views = 4
    cfg.pipeline.model.mask2former.id2label = PANOPTIC_SEMANTIC2NAME
    cfg.pipeline.model.mask2former.label_ids_to_fuse = STUFF_CLASSES
    cfg.pipeline.evaluator.id2label = PANOPTIC_SEMANTIC2NAME
    cfg.pipeline.evaluator.stuffs = STUFF_CLASSES
    cfg.pipeline.evaluator.things = THING_CLASSES
    typed = load_typed_config(cfg, RootCfg)
    datamodule = ScanNetDataModule(
        train_loader_cfg=typed.datamodule.train_loader_cfg,
        val_loader_cfg=typed.datamodule.val_loader_cfg,
        test_loader_cfg=typed.datamodule.test_loader_cfg,
        dataset_cfg=typed.datamodule.dataset_cfg,
    )
    pipeline = Pipeline(typed)
    if args.prediction_bundle is not None:
        original_add = pipeline.visualizer.add

        def capture_add(*add_args, **add_kwargs):
            _save_prediction_bundle(
                args.prediction_bundle,
                add_kwargs["batch"],
                add_kwargs["render_output"],
                add_kwargs["context_semantic_ids"],
                add_kwargs["context_instance_ids"],
                add_kwargs["target_semantic_ids"],
                add_kwargs["target_instance_ids"],
                add_kwargs["target_seg_infos"],
            )
            return original_add(*add_args, **add_kwargs)

        pipeline.visualizer.add = capture_add
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        strategy="ddp_find_unused_parameters_true",
        max_epochs=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        default_root_dir=str(args.output.parent),
    )
    # Load the already SHA-verified official checkpoint before taking the
    # baseline hash.  Passing ckpt_path to Trainer.validate would load it
    # after this baseline and make an otherwise correct inference look like a
    # parameter mutation.
    # The verified official Lightning checkpoint contains the RootCfg object
    # in addition to state_dict, so its trusted official load path requires
    # weights_only=False.  The closure wrapper verifies the fixed SHA256
    # before this line is reached.
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise RuntimeError("official checkpoint has no state_dict")
    pipeline.load_state_dict(checkpoint["state_dict"], strict=True)
    pipeline = pipeline.cuda()
    before = digest({key: value for key, value in pipeline.named_parameters()})
    started = time.time()
    trainer.validate(pipeline, datamodule=datamodule, ckpt_path=None)
    after = digest({key: value for key, value in pipeline.named_parameters()})
    payload = {
        "protocol": "siu3r_official_reference_pair0_smoke",
        "partition": args.partition,
        "pair_index": 0,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "val_pair_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        "parameter_hash_before": before,
        "parameter_hash_after": after,
        "parameter_hash_unchanged": before == after,
        "optimizer_step_executed": False,
        "ttt_started": False,
        "elapsed_seconds": time.time() - started,
        "note": "official Pipeline validation output is retained under the delivery workspace",
    }
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
