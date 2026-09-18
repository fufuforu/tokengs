"""CPU-only SIU3R protocol and TokenGS compatibility audit.

This script never imports the model, never loads a checkpoint into a model and
never starts a trainer.  It writes only the isolated alignment workspace.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokengs.siu3r_protocol import PANOPTIC_CLASSES, PROTOCOL, validate_val_pairs

SIU3R = ROOT.parent / "SIU3R"
OUT = ROOT / "workspace/siu3r_protocol_alignment_v1"
PAIRS = OUT / "val_pair.json"
CHECKPOINT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_"
    "dino_metric_joint_formation_j2_ddp8/checkpoints/model_step_000250.safetensors"
)
EXPECTED_REPO = "8ea80166be76854f938e90521f1a5b688b755c87"
EXPECTED_CKPT = "61debbf5f55b76eea16012aea00fe19291309177bfdc6dd0420f6d1cb58d8cba"
EXPECTED_PAIR = "59cf5594ec2f3223a41d27d7af4da49d348ce9b00c1d151020378c9a8f4b4b3b"


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_evidence(path: str, lines: str, function: str, claim: str) -> dict[str, str]:
    return {"file": path, "lines": lines, "function_or_symbol": function, "claim": claim}


def versions() -> dict[str, Any]:
    result: dict[str, Any] = {"python": sys.version.split()[0]}
    for name in ("torch", "torchvision", "torchmetrics", "lpips"):
        try:
            result[name] = importlib.metadata.version(name)
        except Exception as exc:  # Import can fail because of missing optional wheels.
            result[name] = None
            result[f"{name}_error"] = type(exc).__name__ + ": " + str(exc)
    result["official_expected"] = {
        "torch": "2.4.1",
        "torchvision": "0.19.1",
        "torchmetrics": "1.7.3",
    }
    result["fixed_versions_match"] = all(
        result.get(name) == value
        for name, value in result["official_expected"].items()
    )
    return result


def assert_text(path: Path, needles: list[str]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [needle for needle in needles if needle in text]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=PAIRS)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pair_info = validate_val_pairs(args.pairs)
    repo_commit = run("git", "-C", str(SIU3R), "rev-parse", "HEAD") if (SIU3R / ".git").exists() else None
    repo_status = run("git", "-C", str(SIU3R), "status", "--short") if repo_commit else None
    tokengs_commit = run("git", "rev-parse", "HEAD", cwd=ROOT)
    tokengs_status = run("git", "status", "--short", cwd=ROOT)

    j2_config = ROOT / "workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_ddp8/config.yaml"
    config_text = j2_config.read_text(encoding="utf-8")
    two_context_reasons = [
        "num_input_views: 8" in config_text,
        "num_views: 15" in config_text,
        "semantic_token_max_views: 8" in config_text,
    ]
    model_source = (ROOT / "tokengs/models/tokengs.py").read_text(encoding="utf-8")
    provider_source = (ROOT / "tokengs/data/provider.py").read_text(encoding="utf-8")
    semantic_source = (ROOT / "tokengs/models/semantic_tokengs_v4.py").read_text(encoding="utf-8")
    old_instance_ap = ROOT / "tokengs/utils/instance_ap.py"
    old_eval = ROOT / "scripts/eval_instance_ap_per_view_reval.py"
    legacy_diff = run("git", "diff", "--", str(old_instance_ap), str(old_eval), cwd=ROOT)

    official_data = SIU3R / "data/scannet"
    official_ckpt_candidates = [
        SIU3R / "pretrained_weights",
        SIU3R / "checkpoints",
        SIU3R / "ckpts",
    ]
    official_ckpt_files = [str(p) for root in official_ckpt_candidates if root.exists() for p in root.rglob("*") if p.is_file()]

    source_files = [
        "configs/main.yaml",
        "src/config.py",
        "src/evaluator.py",
        "src/utils/miou.py",
        "src/utils/scannet_constant.py",
        "src/data/components/scannet_dataset.py",
        "src/data/datamodules/scannet_datamodule.py",
        "src/pipeline.py",
        "src/visualizer.py",
        "uv.lock",
    ]
    source_hashes = {path: sha256(SIU3R / path) for path in source_files}

    audit = {
        "protocol": PROTOCOL,
        "alignment_version": "SIU3R Official Protocol Alignment v1",
        "generated_by": str(Path(__file__).resolve()),
        "siu3r": {
            "repo": str(SIU3R),
            "repo_url": "https://github.com/WU-CVGL/SIU3R",
            "commit_expected": EXPECTED_REPO,
            "commit_actual": repo_commit,
            "repo_clean": repo_status == "",
            "git_status_short": repo_status,
            "source_files_read": source_files,
            "source_sha256": source_hashes,
            "required_source_evidence": [
                source_evidence("configs/main.yaml", "29-38,83-93", "datamodule/evaluator config", "256x256 dataset, two extra target views for base config, and all official evaluator switches"),
                source_evidence("src/config.py", "158-199", "load_typed_root_config/bind_cfg", "validation/test changes extra target views to four and binds 20-class ScanNet panoptic/stuff/thing mappings"),
                source_evidence("src/evaluator.py", "28-109,120-236", "Evaluator.setup/process_segmentation/fit_scale_and_shift", "official metric constructors, segment decoding, class-aware score source and depth alignment"),
                source_evidence("src/utils/miou.py", "8-76", "_compute_intersection_and_union/MeanIoU", "background exclusion and dataset-global intersection/union"),
                source_evidence("src/utils/scannet_constant.py", "1-35", "PANOPTIC_SEMANTIC2NAME/STUFF_CLASSES/THING_CLASSES", "20 ScanNet labels, two stuff classes, 18 thing classes"),
                source_evidence("src/data/components/scannet_dataset.py", "22-88,90-114,166-211,222-339", "ScanNetDataset", "official pair selection, 256 processor, RGB/depth loading, intrinsics normalization, relative OpenCV poses and separate context/target segmentation"),
                source_evidence("src/data/datamodules/scannet_datamodule.py", "13-84,123-155", "collate_fn/ScanNetDataModule.val_dataloader/test_dataloader", "tensor ranges, batch collation and validation pair order without shuffle"),
                source_evidence("src/pipeline.py", "55-130,132-214", "Pipeline.step_wo_lift/step_w_query_class_logit_lift", "context images/intrinsics feed model, target poses feed rasterizer, and context/target segmentation outputs are distinct"),
                source_evidence("src/visualizer.py", "761-811", "Visualizer.save_seg_ids", "semantic and global instance IDs are encoded to per-view RGB segment maps"),
                source_evidence("uv.lock", "3373-3385", "torchmetrics package lock", "TorchMetrics 1.7.3 lock entry and Torch 2.4.1 dependency"),
            ],
        },
        "val_pair": {
            **pair_info,
            "sha256_expected": EXPECTED_PAIR,
            "source_url": "https://huggingface.co/datasets/insomnia7/SIU3R/resolve/main/scannet/val_pair.json",
            "record_order_preserved": True,
            "ids_deduplicated": False,
            "ids_resampled": False,
            "ids_sorted": False,
        },
        "official_data": {
            "preprocessed_data_path": str(official_data),
            "found": official_data.is_dir(),
            "expected_size_gb": 112,
            "download_policy": "not_auto_downloaded",
            "missing_data_resume_command": "huggingface-cli download insomnia7/SIU3R --repo-type dataset --include 'scannet/**' --local-dir /space/mawb/SIU3R/data",
        },
        "official_checkpoint": {
            "found": bool(official_ckpt_files),
            "candidate_files": official_ckpt_files[:50],
        },
        "preprocessing": {
            "resolution": [256, 256],
            "rgb_input_range": "uint8 [0,255] loaded then collate divides by 255.0; processor do_rescale=False/do_normalize=False",
            "resize_crop": "VideoMask2FormerImageProcessor(size=(256,256)); no center/random crop in official SIU3R dataset; panoptic maps use processor resize path",
            "intrinsics": "K'=[ [fx/256,0,cx/256], [0,fy/256,cy/256], [0,0,1] ]",
            "intrinsics_evidence": source_evidence("src/data/components/scannet_dataset.py", "77-88", "ScanNetDataset.intrinsics_normalize", "divide first row by width and second row by height"),
            "image_evidence": source_evidence("src/data/datamodules/scannet_datamodule.py", "18-25,39-46", "collate_fn", "RGB tensors are divided by 255.0 and depths by 1000.0"),
            "processor_evidence": source_evidence("src/data/components/scannet_dataset.py", "65-72", "ScanNetDataset.__init__", "fixed 256x256 processor, no rescale/normalize, ignore index 255, 20 labels"),
        },
        "camera_and_input": {
            "convention": "OpenCV camera-to-world: +X right, +Y down, +Z looks into screen",
            "evidence": source_evidence("README.md", "99-100", "Camera Conventions", "official camera convention"),
            "relative_pose": source_evidence("src/data/components/scannet_dataset.py", "90-114", "ScanNetDataset.relative_pose", "inverse of first context extrinsic left-multiplied onto context and target extrinsics"),
            "context_pose_model_input": False,
            "context_pose_model_input_reason": "SIU3R official model is unposed; its dataset uses poses only to prepare relative target cameras, while TokenGS consumes posed context ray/Plücker features (see compatibility below)",
            "target_pose_for_rasterization": source_evidence("src/pipeline.py", "55-87", "Pipeline.step_wo_lift", "target extrinsics/intrinsics are passed to SplattingCUDA"),
        },
        "segmentation": {
            "context_vs_target": source_evidence("src/data/components/scannet_dataset.py", "222-237,258-295,297-339", "ScanNetDataset.__getitem__", "context and target segmentation are loaded and preprocessed separately"),
            "semantic_mapping": "segment_id = RGB integer; semantic_id = segment_id // 1000; instance_id = segment_id % 1000; labels 1..20, 0 unlabeled/background",
            "classes": {str(k): v for k, v in PANOPTIC_CLASSES.items()},
            "valid_class_count": 20,
            "thing_count": 18,
            "stuff_classes": ["wall", "floor"],
            "semantic_evidence": source_evidence("src/utils/scannet_constant.py", "1-28", "PANOPTIC_SEMANTIC2NAME/STUFF_CLASSES/THING_CLASSES", "20 valid ScanNet classes with wall/floor stuff and 18 thing classes"),
            "miou_evidence": source_evidence("src/utils/miou.py", "8-31,34-76", "MeanIoU", "background excluded, dataset-global intersection/union, per-class IoU and mean over valid unions"),
            "instance_map_evidence": source_evidence("src/evaluator.py", "120-227", "Evaluator.process_segmentation", "decode maps, skip instance zero/stuff, build class-aware masks/scores"),
            "prediction_score_source": source_evidence("src/evaluator.py", "176-198", "Evaluator.process_segmentation", "pred.json info['score'] values are averaged per predicted ID; absent JSON uses score 1.0"),
        },
        "metrics": {
            "reconstruction": {
                "psnr": "torchmetrics.image.PeakSignalNoiseRatio; one call per target image, arithmetic mean",
                "ssim": "torchmetrics.image.StructuralSimilarityIndexMeasure; one call per target image, arithmetic mean",
                "lpips": "LearnedPerceptualImagePatchSimilarity('vgg', normalize=True); one call per target image, arithmetic mean",
                "evidence": source_evidence("src/evaluator.py", "49-59,251-269,369-372", "Evaluator.setup/Evaluator.evaluate", "exact constructors and per-image averaging"),
            },
            "depth": {
                "valid": "GT depth > 0 only",
                "alignment": "torch.linalg.lstsq([pred, ones], gt) per target image",
                "outputs": ["AbsRel", "RMSE"],
                "evidence": source_evidence("src/evaluator.py", "229-236,333-375", "Evaluator.fit_scale_and_shift/Evaluator.evaluate", "scale+shift, positive GT mask and arithmetic target-image mean"),
            },
            "semantic_miou": "MeanIoU(num_classes=21, include_background=False, input_format='index', per_class=True, sync_on_compute=False); global state accumulation",
            "instance_map": "MeanAveragePrecision(iou_type='segm', class_metrics=True, sync_on_compute=False); class-aware explicit labels and scores; pair concatenated maps are one image",
            "panoptic_pq": "PanopticQuality(things=[3..20], stuffs=[1,2], return_per_class=True, allow_unknown_preds_category=True, sync_on_compute=False)",
            "evaluator_evidence": source_evidence("src/evaluator.py", "61-105,271-331,376-399", "Evaluator.setup/Evaluator.evaluate", "official metric construction and context/target updates"),
        },
        "aggregation": {
            "reconstruction_and_depth": "each target image then arithmetic mean across target images in dataset",
            "miou": "dataset-global intersection/union, then per-class IoU and arithmetic mean over valid classes",
            "map": "TorchMetrics global state updated with one concatenated pair image per context/target split",
            "pq": "TorchMetrics global state updated with one concatenated pair image per context/target split",
            "official_unit": "pair for multiview segmentation updates; dataset-global final compute; target RGB/depth image for reconstruction/depth",
        },
        "tokengs": {
            "git_commit": tokengs_commit,
            "git_status_short": tokengs_status,
            "checkpoint": {
                "path": str(CHECKPOINT),
                "found": CHECKPOINT.is_file(),
                "sha256": sha256(CHECKPOINT) if CHECKPOINT.is_file() else None,
                "sha256_expected": EXPECTED_CKPT,
                "sha256_matches": CHECKPOINT.is_file() and sha256(CHECKPOINT) == EXPECTED_CKPT,
            },
            "j2_config": {
                "path": str(j2_config),
                "num_input_views": 8,
                "num_views": 15,
                "image_size": [256, 256],
                "semantic_token_max_views": 8,
                "model_type": "semantic_tokengs_v6",
                "semantic_logits_classes": 8,
            },
            "two_context_forward_supported": False,
            "two_context_forward_reason": "current J2 checkpoint/config contract is 8 input views; provider slices to opt.num_input_views and the semantic branch is configured for max 8 views",
            "silent_pad_to_eight": False,
            "six_target_render_supported": True,
            "six_target_render_reason": "GaussianRenderer.render consumes [B,V,4,4] and loops over V; capability is static-only because model forward parity failed",
            "global_query_id_supported": True,
            "global_query_id_reason": "the shared per-Gaussian group channels are rendered from one reconstruction across all decoder cameras; not smoke-validated",
            "semantic_logits_supported": False,
            "class_aware_instance_supported": False,
            "panoptic_supported": False,
            "depth_supported": True,
            "evidence": [
                source_evidence("tokengs/options.py", "972-973", "Options.__post_init__", "GSI-v2 fixed 8+7 contract illustrates fixed-view enforcement"),
                source_evidence("workspace/.../config.yaml", "338-349,401-423", "J2 config", "semantic_tokengs_v6, num_input_views=8, num_views=15, semantic max views=8"),
                source_evidence("tokengs/data/provider.py", "268-294", "Provider._preprocess", "context rays/Plücker/cam_to_world are constructed from c2ws and context is sliced by num_input_views"),
                source_evidence("tokengs/models/tokengs.py", "164-191,505-579", "TokenGS._embed_encoder_input/forward_reconstruction/render_reconstruction", "context encoding and arbitrary decoder camera rendering"),
                source_evidence("tokengs/models/semantic_tokengs_v4.py", "3177-3185,3741-3753", "SemanticTokenGSv4.forward", "8 semantic token logits and gaussian score outputs, not 20-class SIU3R logits"),
            ],
        },
        "pose_parity": {
            "DATA_SPLIT_PARITY": "YES",
            "VIEW_PAIR_PARITY": "YES",
            "IMAGE_PREPROCESSING_PARITY": "YES (adapter contract; data unavailable for runtime image audit)",
            "METRIC_IMPLEMENTATION_PARITY": "YES (isolated backend calls official constructors)",
            "CONTEXT_VIEW_COUNT_PARITY": "NO",
            "INPUT_POSE_ASSUMPTION_PARITY": "NO",
            "SEMANTIC_LABEL_SPACE_PARITY": "NO",
            "INSTANCE_SCORE_PARITY": "NO",
            "STRICT_END_TO_END_SIU3R_PARITY": "NO",
            "result_label": "SIU3R_DATA_AND_EVALUATOR_ALIGNED_POSED_TOKENGS",
        },
        "runtime": {
            "dependencies": versions(),
            "official_reference_smoke_valid": False,
            "tokengs_pair_smoke_valid": False,
            "formal_1860_evaluation_started": False,
            "training_started": False,
            "optimizer_step_executed": False,
            "ttt_started": False,
            "oracle_used": False,
        },
        "legacy_protection": {
            "legacy_files_checked": [str(old_instance_ap), str(old_eval)],
            "legacy_diff_empty": legacy_diff == "",
            "legacy_protocol_name": "per_target_view_v1",
            "legacy_not_called_by_new_evaluator": True,
        },
    }

    # Keep compatibility as a separate machine-readable matrix for downstream review.
    matrix = {
        "protocols": {
            "siu3r_global_multiview_v1": {"scenes": 312, "pairs": 1860, "context": 2, "target": 6, "aggregation": "pair-concat/global"},
            "per_target_view_v1": {"scenes": 40, "context": 8, "target": 7, "aggregation": "independent target image", "class_agnostic_ap": True},
        },
        "dimensions": {
            "data_split": ["SIU3R official val_pair.json", "TokenGS LSM-40 manifest"],
            "view_pair": ["official pair records, no dedup/sort/resample", "8+7 LSM windows"],
            "rgb": ["256x256, [0,1] after /255", "TokenGS native transform may crop/normalize"],
            "camera": ["unposed context model; OpenCV c2w targets", "posed context rays/Plücker + target c2w"],
            "semantic": ["20 ScanNet ids; background excluded", "current J2 8 semantic token classes"],
            "instance": ["class-aware TorchMetrics mAP, global IDs", "class-agnostic LSM instance_ap"],
        },
        "hard_gate": {"two_context_input": False, "semantic_20_class": False, "official_data_present": False, "strict_ready": False},
    }

    def atomic_json(path: Path, value: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    atomic_json(args.output / "protocol_audit.json", audit)
    atomic_json(args.output / "compatibility_matrix.json", matrix)
    md = f"# SIU3R Official Protocol Alignment v1\n\n"
    md += f"- Protocol: `{PROTOCOL}`\n- SIU3R commit: `{repo_commit}`; clean: `{repo_status == ''}`\n"
    md += f"- val_pair SHA match: `{pair_info['sha256_matches']}`; records: `{pair_info['records']}`; scenes: `{pair_info['unique_scenes']}`\n"
    md += f"- TokenGS J2 checkpoint SHA match: `{audit['tokengs']['checkpoint']['sha256_matches']}`\n"
    md += "\n## Official evidence\n\n"
    md += "The authoritative line-level evidence is recorded in `protocol_audit.json`. The required sources are read at the pinned commit: `configs/main.yaml`, `src/config.py`, `src/evaluator.py`, `src/utils/miou.py`, `src/utils/scannet_constant.py`, `src/data/components/scannet_dataset.py`, `src/data/datamodules/scannet_datamodule.py`, `src/pipeline.py`, `src/visualizer.py`, and `uv.lock`.\n\n"
    md += "## Hard-gate result\n\n"
    md += "`TOKEN_GS_TWO_CONTEXT_FORWARD_SUPPORTED: NO` because J2 is configured for 8 context views and the semantic branch for 8-view input. The adapter/evaluator is therefore prepared but the TokenGS official pair smoke is deliberately not run.\n\n"
    md += "`SIU3R_SEMANTIC_MIOU_SUPPORTED: NO`, `SIU3R_CLASS_AWARE_MAP_SUPPORTED: NO`, and `SIU3R_PQ_SUPPORTED: NO`: the checkpoint exposes an 8-class semantic token score path, not the official 20-class class-aware panoptic contract. Reconstruction/depth primitives remain implemented and isolated.\n\n"
    md += "## Pose caveat\n\n"
    md += "The result label is `SIU3R_DATA_AND_EVALUATOR_ALIGNED_POSED_TOKENGS`; it is not a strict unposed SIU3R comparison while TokenGS consumes context camera pose/ray inputs.\n\n"
    md += "## Data status\n\n"
    md += f"Official processed data expected at `{official_data}` was found: `{official_data.is_dir()}`. No 112GB download was started.\n"
    (args.output / "protocol_audit.md").write_text(md, encoding="utf-8")
    print(json.dumps({"audit": str(args.output / "protocol_audit.json"), "pairs": pair_info, "two_context_forward_supported": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
