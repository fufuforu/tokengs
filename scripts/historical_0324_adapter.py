"""Isolated loader/evaluator adapter for the transferred historical 0.324 run.

This module deliberately does not alter the current model or LSM evaluator.
It supplies the historical three-artifact composition to the existing
``eval_instance_lsm_protocol`` entry point in a subprocess-local registry:
RE10K initializes TokenGS, wide7l supplies the frozen reconstruction backbone,
and the 3000-step checkpoint supplies the DINO/unit instance branch.

The historical checkpoint was written by an older code version which
registered DINOv2 inside the instance branch.  Current code keeps DINO as an
external, frozen service, so its keys are strict-checked against the local
cache and are never inserted into the parent model state dict.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT / "workspace/historical_0324_recovery_v1/transfer_bundle"
PRIMARY = BUNDLE_ROOT / "checkpoints/model_step_003000.safetensors"
WIDE7L = BUNDLE_ROOT / "checkpoints/model_step_008000_wide7l.safetensors"
RE10K = BUNDLE_ROOT / "checkpoints/tokengs_re10k.safetensors"
LSM_MANIFEST = BUNDLE_ROOT / "manifests/lsm_instance_eval_manifest.json"
TRAIN_MANIFEST = BUNDLE_ROOT / "manifests/scannet_prompt_full_wide_8x7.json"
LOCAL_TRAIN_MANIFEST = REPO_ROOT / "data/scannet_prompt/scannet_c3g8_train_provisional.json"
LOCAL_QUERY_BANK = REPO_ROOT / "data/scannet_prompt/scannet_c3g8_query_bank.json"
HIST_CONFIG = BUNDLE_ROOT / "evidence/config.yaml"
CLIP_PATH = REPO_ROOT / "checkpoints/clip-vit-large-patch14"
DINO_REPO = Path("/space/mawb/.cache/torch/hub/facebookresearch_dinov2_main")
DINO_WEIGHT = Path("/space/mawb/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth")
DINO_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
DINO_PREFIX = "instance_branch._dino_model."
LSM_MANIFEST_SHA256 = "823744228c249e0fd713813e7eb707e798e7c4e112603444e1dc3aac9417035d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_lsm_scenes() -> list[str]:
    manifest = json.loads(LSM_MANIFEST.read_text(encoding="utf-8"))
    scenes = manifest.get("scenes")
    if isinstance(scenes, dict):
        normalized = [str(scene) for scene in scenes]
        for scene, entry in scenes.items():
            if not isinstance(entry, dict):
                raise RuntimeError(f"LSM scene entry is not an object: {scene}")
            if len(entry.get("context_raw_frame_ids", [])) != 8:
                raise RuntimeError(f"LSM scene {scene} does not have 8 context frames")
            if len(entry.get("test_raw_frame_ids", [])) != 7:
                raise RuntimeError(f"LSM scene {scene} does not have 7 target frames")
    elif isinstance(scenes, list):
        normalized = [str(scene) for scene in scenes]
    else:
        raise RuntimeError(
            f"LSM manifest must contain 40 scenes, got {type(scenes).__name__}"
        )
    if len(normalized) != 40 or len(set(normalized)) != 40:
        raise RuntimeError("LSM manifest contains duplicate scene IDs")
    if sha256_file(LSM_MANIFEST) != LSM_MANIFEST_SHA256:
        raise RuntimeError("LSM manifest SHA256 mismatch")
    return sorted(normalized)


def _assert_evaluation_complete(
    result_path: Path,
    *,
    requested_scene_count: int,
) -> dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("Evaluator result has no protocol object")
    expected_scenes = _expected_lsm_scenes()
    expected_count = 40 if requested_scene_count == 40 else requested_scene_count
    per_scene = payload.get("per_scene")
    if not isinstance(per_scene, dict):
        raise RuntimeError("Evaluator result has no per_scene mapping")
    actual_scenes = sorted(str(scene) for scene in per_scene)
    if payload.get("num_scenes") != len(per_scene):
        raise RuntimeError(
            f"num_scenes mismatch: field={payload.get('num_scenes')} "
            f"per_scene={len(per_scene)}"
        )
    if len(per_scene) != expected_count:
        raise RuntimeError(
            f"Scene cardinality mismatch: requested={expected_count} "
            f"actual={len(per_scene)}"
        )
    expected_subset = expected_scenes[:expected_count]
    if actual_scenes != expected_subset:
        raise RuntimeError(
            f"Scene set mismatch: expected={expected_subset} actual={actual_scenes}"
        )
    manifest_info = protocol.get("manifest")
    if not isinstance(manifest_info, dict):
        raise RuntimeError("Evaluator result has no manifest fingerprint")
    if int(manifest_info.get("scene_count", -1)) != 40:
        raise RuntimeError("Evaluator protocol manifest scene_count is not 40")
    if manifest_info.get("sha256") != LSM_MANIFEST_SHA256:
        raise RuntimeError("Evaluator protocol manifest SHA256 mismatch")
    if int(protocol.get("context_views", -1)) != 8:
        raise RuntimeError("Evaluator protocol context_views is not 8")
    if int(protocol.get("target_views", -1)) != 7:
        raise RuntimeError("Evaluator protocol target_views is not 7")

    def assert_finite(value: Any, path: str = "result") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                assert_finite(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                assert_finite(item, f"{path}[{index}]")
        elif isinstance(value, float):
            if not torch.isfinite(torch.tensor(value)):
                raise RuntimeError(f"Non-finite evaluator output at {path}: {value}")

    assert_finite(payload)
    return {
        "expected_scene_count": expected_count,
        "actual_scene_count": len(per_scene),
        "expected_scenes": expected_subset,
        "actual_scenes": actual_scenes,
        "manifest_sha256": LSM_MANIFEST_SHA256,
        "context_views": 8,
        "target_views": 7,
        "all_outputs_finite": True,
    }


def _bundle_relative(destination: str) -> Path:
    marker = "/transfer_bundle/"
    if marker not in destination:
        raise RuntimeError(f"Manifest destination is outside bundle: {destination}")
    return Path(destination.split(marker, 1)[1])


def validate_bundle() -> dict[str, Any]:
    """Validate the 22 manifest entries plus the manifest itself (23 files)."""
    manifest_path = BUNDLE_ROOT / "transfer_bundle_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing bundle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 22:
        raise RuntimeError(
            "Expected 22 manifest entries plus transfer_bundle_manifest.json "
            f"(23 files total), got {len(entries) if isinstance(entries, list) else entries!r}"
        )
    listed: set[Path] = set()
    files: list[dict[str, Any]] = []
    for entry in entries:
        rel = _bundle_relative(str(entry["destination"]))
        path = BUNDLE_ROOT / rel
        listed.add(path.resolve())
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Bundle entry is not a regular file: {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(entry["size_bytes"]):
            raise RuntimeError(f"Bundle size mismatch: {path}")
        if digest != entry["sha256_destination"]:
            raise RuntimeError(f"Bundle hash mismatch: {path}")
        files.append(
            {
                "path": str(path.resolve()),
                "category": entry.get("category"),
                "size_bytes": size,
                "sha256": digest,
                "source": entry.get("source"),
                "manifest_entry": True,
            }
        )
    actual = {p.resolve() for p in BUNDLE_ROOT.rglob("*") if p.is_file()}
    manifest_resolved = manifest_path.resolve()
    extras = sorted(str(p) for p in actual - listed - {manifest_resolved})
    missing = sorted(str(p) for p in listed - actual)
    if extras or missing or len(actual) != 23:
        raise RuntimeError(
            f"Bundle file topology mismatch: actual={len(actual)} extras={extras} missing={missing}"
        )
    manifest_hash = sha256_file(manifest_path)
    files.append(
        {
            "path": str(manifest_resolved),
            "category": "bundle_manifest",
            "size_bytes": manifest_path.stat().st_size,
            "sha256": manifest_hash,
            "manifest_entry": False,
        }
    )
    return {
        "bundle_hash_valid": True,
        "manifest_entries": len(entries),
        "validated_file_count": len(files),
        "manifest_sha256": manifest_hash,
        "files": files,
    }


def _load_history_yaml() -> dict[str, Any]:
    class Loader(yaml.UnsafeLoader):
        pass

    Loader.add_multi_constructor(
        "!dataclass:",
        lambda loader, _suffix, node: loader.construct_mapping(node, deep=True),
    )
    value = yaml.load(HIST_CONFIG.read_text(encoding="utf-8"), Loader=Loader)
    if not isinstance(value, dict):
        raise RuntimeError("Historical config did not decode to a mapping")
    return value


def build_historical_options(workspace: Path):
    from tokengs.options import config_defaults

    raw = _load_history_yaml()
    opt = copy.deepcopy(config_defaults["semantic_v6_unit_shaping_img_dino_train"])
    for key, value in raw.items():
        if hasattr(opt, key):
            setattr(opt, key, value)
    opt.model_type = "semantic_tokengs_v6"
    # The historical training config names its prompt-small training data.
    # The current LSM evaluator constructs its test loader from opt.data_mode;
    # using the historical value would expose 24 prompt samples from only 8
    # scenes and per_scene aggregation would overwrite repeated scene keys.
    # Route only this isolated evaluator adapter through the registered,
    # manifest-backed 40-scene LSM dataset.
    opt.data_mode = (("scannet_lsm_instance_eval", 1),)
    opt.evaluating = True
    opt.prompt_tokengs_checkpoint = str(RE10K.resolve())
    opt.backbone_resume = str(WIDE7L.resolve())
    # The primary checkpoint contains the complete instance branch.  Do not
    # ask the current constructor to read an unavailable secondary unit fork.
    opt.instance_branch_unit_resume = ""
    opt.prompt_clip_model_path = str(CLIP_PATH.resolve())
    opt.lseg_checkpoint_path = str(
        getattr(opt, "lseg_checkpoint_path", "") or ""
    ).replace("/space0/mawb/tokengs", str(REPO_ROOT))
    opt.mixed_precision = "no"  # current LSM evaluator is eager FP32
    opt.instance_group_num_groups = 100
    opt.instance_branch_num_groups = 100
    opt.instance_branch_cluster_eps = 0.5
    opt.dataset_kwargs = {
        **dict(getattr(opt, "dataset_kwargs", {}) or {}),
        # The current evaluator constructs both dataset sides even though
        # only the test loader is consumed.  Keep the historical full-wide
        # manifest for the test-side SmallPromptDataset, while satisfying the
        # prompt-train superclass with the repository's local C3G8 train
        # manifest and query bank.  This is evaluator plumbing only; no train
        # loader is consumed by the smoke/eval path.
        "train_manifest_path": str(LOCAL_TRAIN_MANIFEST.resolve()),
        "query_bank_path": str(LOCAL_QUERY_BANK.resolve()),
        "small_manifest_path": str(TRAIN_MANIFEST.resolve()),
        "lsm_manifest_path": str(LSM_MANIFEST.resolve()),
    }
    opt.workspace = str(workspace.resolve())
    opt.resume = str(PRIMARY.resolve())
    # Explicit metadata for the isolated adapter; these are not model fields.
    opt.historical_dino_repo_path = str(DINO_REPO.resolve())
    opt.historical_dino_weight_path = str(DINO_WEIGHT.resolve())
    return opt


def _load_local_dino(primary_state: dict[str, torch.Tensor]):
    if not DINO_REPO.is_dir() or not (DINO_REPO / "hubconf.py").is_file():
        raise RuntimeError(f"Local DINO repo is unavailable: {DINO_REPO.resolve()}")
    if not DINO_WEIGHT.is_file() or DINO_WEIGHT.is_symlink():
        raise RuntimeError(f"Local DINO weight is unavailable: {DINO_WEIGHT.resolve()}")
    digest = sha256_file(DINO_WEIGHT)
    if digest != DINO_SHA256:
        raise RuntimeError(f"DINO SHA256 mismatch: {digest} != {DINO_SHA256}")
    model = torch.hub.load(
        str(DINO_REPO.resolve()),
        "dinov2_vitb14",
        source="local",
        pretrained=False,
    )
    state = torch.load(str(DINO_WEIGHT), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Local DINO strict load mismatch: {result}")
    checkpoint_dino = {
        key[len(DINO_PREFIX) :]: value
        for key, value in primary_state.items()
        if key.startswith(DINO_PREFIX)
    }
    if not checkpoint_dino:
        raise RuntimeError("Historical primary checkpoint has no DINO namespace")
    dino_state = model.state_dict()
    if set(checkpoint_dino) != set(dino_state):
        raise RuntimeError(
            "Historical DINO namespace mismatch: "
            f"checkpoint_only={sorted(set(checkpoint_dino)-set(dino_state))[:5]} "
            f"local_only={sorted(set(dino_state)-set(checkpoint_dino))[:5]}"
        )
    for key, value in checkpoint_dino.items():
        if tuple(value.shape) != tuple(dino_state[key].shape) or not torch.equal(
            value, dino_state[key]
        ):
            raise RuntimeError(f"Historical DINO tensor mismatch: {key}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "source": "local",
        "repo_path": str(DINO_REPO.resolve()),
        "weight_path": str(DINO_WEIGHT.resolve()),
        "sha256": digest,
        "strict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_dino_keys": len(checkpoint_dino),
    }


def _historical_mapping(model, primary: dict[str, torch.Tensor]):
    native = torch.nn.Module.state_dict(model)
    mapped: dict[str, torch.Tensor] = {}
    source_to_target: dict[str, str] = {}
    skipped_dino = []
    for key, value in primary.items():
        if key.startswith(DINO_PREFIX):
            skipped_dino.append(key)
            continue
        target = key
        if target not in native:
            candidate = "prompt_matcher." + key
            if candidate in native:
                target = candidate
        if target not in native:
            raise RuntimeError(f"Historical registered key has no target: {key}")
        if tuple(native[target].shape) != tuple(value.shape):
            raise RuntimeError(
                f"Historical shape mismatch for {key}: {tuple(value.shape)} != {tuple(native[target].shape)}"
            )
        mapped[target] = value
        source_to_target[key] = target
    if len(mapped) != len(primary) - len(skipped_dino):
        raise RuntimeError("Historical registered key mapping was not one-to-one")
    return mapped, source_to_target, skipped_dino


def _load_re10k_strict(opt) -> dict[str, Any]:
    from tokengs.models.tokengs import TokenGS

    base = TokenGS(opt)
    state = load_file(str(RE10K), device="cpu")
    result = torch.nn.Module.load_state_dict(base, state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"RE10K strict load mismatch: {result}")
    return {
        "path": str(RE10K.resolve()),
        "sha256": sha256_file(RE10K),
        "keys": len(state),
        "missing_keys": [],
        "unexpected_keys": [],
    }


def make_historical_model_class(primary_state, dino_model, load_report):
    from tokengs.models.semantic_tokengs_v6 import SemanticTokenGSv6

    class HistoricalSemanticTokenGSv6(SemanticTokenGSv6):
        def __init__(self, opt):
            super().__init__(opt)
            if self.instance_branch is None:
                raise RuntimeError("Historical primary requires instance_branch")
            self.instance_branch.__dict__["_dino_model"] = dino_model
            native = torch.nn.Module.state_dict(self)
            wide = load_file(str(WIDE7L), device="cpu")
            prefixes = (
                "patch_embed.",
                "patch_plucker_embed.",
                "enc_dec_backbone.",
                "activation_head.",
                "anchor_pos_encoder.",
            )
            base = {
                key: value
                for key, value in wide.items()
                if key == "gs_tokens" or key.startswith(prefixes)
            }
            missing_base = [
                key
                for key in base
                if key not in native or tuple(native[key].shape) != tuple(base[key].shape)
            ]
            if missing_base:
                raise RuntimeError(f"wide7l strict base mismatch: {missing_base[:8]}")
            torch.nn.Module.load_state_dict(self, base, strict=False)
            mapped, source_to_target, skipped_dino = _historical_mapping(
                self, primary_state
            )
            torch.nn.Module.load_state_dict(self, mapped, strict=False)
            load_report.update(
                {
                    "wide7l": {
                        "path": str(WIDE7L.resolve()),
                        "sha256": sha256_file(WIDE7L),
                        "base_keys_loaded": len(base),
                        "strict_shape_check": True,
                        "excluded_nonbackbone_keys": len(wide) - len(base),
                    },
                    "primary": {
                        "path": str(PRIMARY.resolve()),
                        "sha256": sha256_file(PRIMARY),
                        "total_keys": len(primary_state),
                        "registered_keys_loaded": len(mapped),
                        "dino_keys_external": len(skipped_dino),
                        "mapped_prompt_keys": sum(
                            target.startswith("prompt_matcher.")
                            for target in source_to_target.values()
                        ),
                        "missing_keys": [],
                        "unexpected_keys": [],
                        "fresh_reset": False,
                    },
                    "state_dict_dino_registered": any(
                        key.startswith("instance_branch._dino_model.")
                        for key in torch.nn.Module.state_dict(self)
                    ),
                }
            )

    return HistoricalSemanticTokenGSv6


def run_current_evaluator(
    *,
    output_workspace: Path,
    max_scenes: int,
) -> dict[str, Any]:
    if output_workspace.exists():
        existing = [
            path.name
            for path in output_workspace.iterdir()
            if path.name != "eval.log"
        ]
        if existing:
            raise RuntimeError(
                f"Refusing to overwrite non-empty evaluation workspace: "
                f"{output_workspace} contains {existing}"
            )
    output_workspace.mkdir(parents=True, exist_ok=True)
    bundle = validate_bundle()
    primary_state = load_file(str(PRIMARY), device="cpu")
    opt = build_historical_options(output_workspace)
    re10k_report = _load_re10k_strict(opt)
    dino_model, dino_report = _load_local_dino(primary_state)
    load_report: dict[str, Any] = {"re10k": re10k_report, "dino": dino_report}
    model_class = make_historical_model_class(primary_state, dino_model, load_report)

    import scripts.eval_instance_lsm_protocol as evaluator

    evaluator.model_registry["semantic_tokengs_v6"] = model_class
    evaluator.config_defaults["eval_scannet_lsm_instance"] = opt
    sys.argv = [
        "eval_instance_lsm_protocol.py",
        "--resume",
        str(PRIMARY),
        "--workspace",
        str(output_workspace),
        "--label",
        "historical_0324_dino_3000",
        "--model_type",
        "semantic_tokengs_v6",
        "--num_groups",
        "100",
        "--min_pred_pixels",
        "1",
        "--min_gt_pixels",
        "1",
        "--max_predictions_per_image",
        "100",
        "--backbone-resume",
        str(WIDE7L),
        "--lsm_manifest",
        str(LSM_MANIFEST),
        "--num_input_views",
        "8",
        "--num_views",
        "15",
        "--max_scenes",
        str(max_scenes),
        "--ttt_steps",
        "0",
    ]
    evaluator.main()
    result_path = output_workspace / "instance_ap.json"
    if not result_path.is_file():
        raise RuntimeError(f"Evaluator did not produce {result_path}")
    cardinality = _assert_evaluation_complete(
        result_path, requested_scene_count=max_scenes
    )
    report_path = output_workspace / "historical_load_report.json"
    report_path.write_text(
        json.dumps(
            {"bundle": bundle, "load": load_report, "evaluation": cardinality},
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"bundle": bundle, "load": load_report, "output": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "smoke", "eval"), default="audit")
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--max-scenes", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "audit":
        bundle = validate_bundle()
        print(json.dumps({"bundle": bundle}, indent=2))
        return
    if args.workspace is None:
        raise SystemExit("--workspace is required for smoke/eval")
    requested_scene_count = args.max_scenes if args.mode == "smoke" else 40
    if args.mode == "smoke" and requested_scene_count not in (1, 2):
        raise SystemExit("smoke max-scenes must be 1 or 2")
    result = run_current_evaluator(
        output_workspace=args.workspace,
        max_scenes=requested_scene_count,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
