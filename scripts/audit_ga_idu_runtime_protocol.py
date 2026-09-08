"""Read-only comparison of the formal Both and GA-IDU data pipelines."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from accelerate import Accelerator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workspace/tsh_ga_idu_runtime_protocol_audit_v2"
BOTH = "semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8"
GA = "semantic_v6_absolute_units_true_shared_ga_idu1_ddp8"

sys.path.insert(0, str(ROOT))
from tokengs.data import get_multi_dataloader
from tokengs.options import config_defaults

DATA_FIELDS = (
    "data_mode", "dataset_kwargs", "num_views", "num_input_views",
    "prompt_mode", "prompt_same_scene_query_min_gap",
    "prompt_same_scene_query_ratio", "prompt_image_probability",
    "query_image_size", "num_workers", "batch_size", "seed", "img_size",
    "random_reflect", "camera_scale_method", "camera_normalization_method",
    "pointmap_trim_lo", "pointmap_trim_hi", "use_interp_target",
    "wide_target_subsample", "max_iters_per_epoch", "gradient_accumulation_steps",
)


def jsonable(x):
    if dataclasses.is_dataclass(x):
        return jsonable(dataclasses.asdict(x))
    if isinstance(x, (tuple, list)):
        return [jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, Path):
        return str(x)
    return x


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def batch_summary(data):
    scene = data["scene_name"]
    if isinstance(scene, (list, tuple)):
        scenes = [str(v) for v in scene]
    else:
        scenes = [str(scene)]
    frame_ids = data["frame_ids"]
    if frame_ids.ndim == 1:
        frame_ids = frame_ids.unsqueeze(0)
    context = frame_ids[:, :8]
    target = frame_ids[:, 8:]
    target_gt = data["instance_label_output"]
    gt_counts = []
    for row in target_gt:
        gt_counts.append(int(sum(int(v > 0) for v in torch.unique(row).tolist())))
    return {
        "scene_ids": scenes,
        "context_image_shape": list(data["images_input"].shape),
        "target_image_shape": list(data["images_output"].shape),
        "context_frame_ids": context.tolist(),
        "target_frame_ids": target.tolist(),
        "all_frame_ids": frame_ids.tolist(),
        "context_target_same_scene": len(set(scenes)) == 1,
        "frame_count": int(frame_ids.shape[1]),
        "context_count": int(data["images_input"].shape[1]),
        "target_count": int(data["images_output"].shape[1]),
        "target_gt_nonzero_instance_count_per_batch": gt_counts,
        "target_gt_shape": list(target_gt.shape),
        "target_gt_view_count": int(target_gt.shape[1]),
        "target_gt_valid": bool(all(v > 0 for v in gt_counts)),
        "target_leakage_check": {
            "encoder_images_are_context_only": list(data["images_input"].shape)[1] == 8,
            "target_images_are_separate": list(data["images_output"].shape)[1] == 7,
        },
    }


def load_batch(config_name: str):
    opt = copy.deepcopy(config_defaults[config_name])
    opt.prompt_overfit_single_batch = True
    opt.prompt_overfit_sample_index = 0
    opt.num_workers = 0
    opt.batch_size = 1
    # Do not override dataset_kwargs, manifest, or view counts. These are the
    # exact resolved formal configuration fields under audit.
    acc = Accelerator(cpu=True)
    loader, _, train_dataset, _ = get_multi_dataloader(opt, acc)
    data = next(iter(loader))
    return opt, train_dataset, data


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    both = config_defaults[BOTH]
    ga = config_defaults[GA]
    diff = {}
    for key in DATA_FIELDS:
        lhs, rhs = jsonable(getattr(both, key, None)), jsonable(getattr(ga, key, None))
        if lhs != rhs:
            diff[key] = {"both": lhs, "ga_idu1": rhs}
    batches = {}
    dataset_info = {}
    for name, cfg in (("Both", BOTH), ("GA-IDU-1", GA)):
        opt, dataset, data = load_batch(cfg)
        batches[name] = batch_summary(data)
        dataset_info[name] = {
            "dataset_class": type(dataset.datasets[0].dataset).__name__,
            "dataset_len": len(dataset),
            "config": {key: jsonable(getattr(opt, key, None)) for key in DATA_FIELDS},
            "sample_index": 0,
            "sample": jsonable(dataset.datasets[0].dataset.sample_list[0].__dict__),
        }
    report = {
        "both_config": BOTH,
        "ga_idu_config": GA,
        "data_field_diff": diff,
        "dataset_info": dataset_info,
        "batches": batches,
        "formal_assertions": {
            "both_context_8": batches["Both"]["context_count"] == 8,
            "both_target_7": batches["Both"]["target_count"] == 7,
            "ga_context_8": batches["GA-IDU-1"]["context_count"] == 8,
            "ga_target_7": batches["GA-IDU-1"]["target_count"] == 7,
            "both_ga_batch_organization_equal": batches["Both"] == batches["GA-IDU-1"],
            "both_same_scene": batches["Both"]["context_target_same_scene"],
            "ga_same_scene": batches["GA-IDU-1"]["context_target_same_scene"],
            "both_no_target_leakage": batches["Both"]["target_leakage_check"]["encoder_images_are_context_only"],
            "ga_no_target_leakage": batches["GA-IDU-1"]["target_leakage_check"]["encoder_images_are_context_only"],
        },
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "manifest_sha256": {
            str(path): sha256(path)
            for path in (
                ROOT / "data/scannet_prompt/scannet_prompt_small_64_8.json",
                ROOT / "data/scannet_prompt/scannet_prompt_full_wide_8x7.json",
            )
        },
    }
    (OUT / "runtime_protocol_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
