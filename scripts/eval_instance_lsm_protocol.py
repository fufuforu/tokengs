"""LSM ScanNet class-agnostic novel-view instance segmentation evaluation.

Implements the publicly specified parts of the InstOk3D Table 2 protocol:
40 ScanNet test scenes, stride-10 frame sampling, 8 context views interleaved
with 7 held-out test views, and AP / AP50 / AP25 on target-view instance
masks. Predictions are matched in confidence order only to GT masks from the
same image. Both scene-macro and dataset-pooled aggregation are reported
because the paper does not specify the final aggregation implementation.

The public paper uses scene-level COLMAP cameras, while this repository's
dataset currently reads ScanNet ``.sens`` poses. The output JSON records this
known difference instead of claiming exact official-code parity.

This is not the InstanceSplat / IGGT temporal tracking protocol, which uses a
different benchmark and reports T-mIoU and T-SR rather than AP.

Expected model output: ``rendered_instance_group_probability`` with shape
[B, G+1, V, 1, H, W] (softmaxed over G groups plus a final void channel).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch._dynamo.config.disable = True  # eager fallback; avoids inductor compile hangs in eval

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _240_path_remap import remap_path, remap_opt  # 240-only path adapter
from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.input_types import split_data
from tokengs.options import config_defaults
from tokengs.utils.instance_ap import (
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)


class _LocalAccelerator:
    is_main_process = True


def _audit_lsm_manifest(path: str) -> dict:
    """Validate the fixed 40-scene, stride-10, interleaved 8+7 split."""
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    scenes = manifest.get("scenes") if isinstance(manifest, dict) else None
    if not isinstance(scenes, dict):
        raise ValueError("LSM manifest must contain a 'scenes' object")
    if len(scenes) != 40:
        raise ValueError(
            f"InstOk3D Table 2 requires 40 scenes, got {len(scenes)}"
        )

    for scene_name, entry in scenes.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid manifest entry for {scene_name}")
        context = [int(value) for value in entry.get("context_raw_frame_ids", [])]
        target = [int(value) for value in entry.get("test_raw_frame_ids", [])]
        if len(context) != 8 or len(target) != 7:
            raise ValueError(
                f"{scene_name}: expected 8 context and 7 test frames, "
                f"got {len(context)} and {len(target)}"
            )
        interleaved = []
        for context_id, target_id in zip(context, target):
            interleaved.extend((context_id, target_id))
        interleaved.append(context[-1])
        gaps = [
            right - left
            for left, right in zip(interleaved, interleaved[1:])
        ]
        if any(gap != 10 for gap in gaps):
            raise ValueError(
                f"{scene_name}: frames are not a stride-10 interleaved split: "
                f"{interleaved}"
            )

    return {
        "path": str(manifest_path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "version": manifest.get("version"),
        "scene_count": len(scenes),
        "frame_stride": 10,
        "split": "interleaved_8_context_7_target",
    }


def _prune_inactive_groups(
    probs: np.ndarray,
    active: int,
    void_channel: int,
) -> np.ndarray:
    """Keep only the top-``active`` most-used groups; dump the rest to void.

    Scene-adaptive budget: ``probs`` is [G+1,H,W], the last channel is
    void. The count head predicted ``active`` instances, so the least-used
    groups are zeroed and their mass is moved to the void channel before
    re-normalization, preventing over-segmentation on sparse scenes.
    """
    num_groups = int(void_channel)
    if active >= num_groups:
        return probs
    probs = np.asarray(probs).copy()
    usage = probs[:num_groups].mean(axis=(1, 2))
    active_ids = np.argsort(-usage)[: int(active)]
    keep = np.zeros(num_groups, dtype=bool)
    keep[active_ids] = True
    inactive_sum = probs[:num_groups][~keep].sum(axis=0)
    probs[:num_groups][~keep] = 0.0
    probs[void_channel] = probs[void_channel] + inactive_sum
    total = probs.sum(axis=0)
    return probs / np.maximum(total, 1e-6)


def _mask_diagnostics(
    pred_masks, gt_masks, pred_image_ids, gt_image_ids,
    thresholds=(0.25, 0.5, 0.75),
) -> dict:
    """Report best-GT IoU/recall without changing AP matching or filtering."""
    if not gt_masks:
        return {
            "mean_best_gt_iou": 0.0,
            **{f"recall_iou{int(t * 100)}": 0.0 for t in thresholds},
        }
    best = []
    for gt, image_id in zip(gt_masks, gt_image_ids):
        gt = np.asarray(gt, dtype=np.float32)
        gt_area = float(gt.sum())
        best_iou = 0.0
        if gt_area > 0:
            for pred, pred_id in zip(pred_masks, pred_image_ids):
                if pred_id != image_id:
                    continue
                pred = np.asarray(pred, dtype=np.float32)
                inter = float((pred * gt).sum())
                union = float(pred.sum()) + gt_area - inter
                best_iou = max(best_iou, inter / max(union, 1e-8))
        best.append(best_iou)
    values = np.asarray(best, dtype=np.float32)
    return {
        "mean_best_gt_iou": float(values.mean()),
        **{
            f"recall_iou{int(t * 100)}": float((values >= t).mean())
            for t in thresholds
        },
    }


def _ranking_audit_candidates(
    probs: np.ndarray, void_channel: int, min_mask_area: int
) -> list[dict]:
    """Export the exact pre-AP group candidates without changing evaluation."""
    group_ids = np.argmax(probs, axis=0)
    pixel_max = np.max(probs, axis=0)
    candidates = []
    for group_id in range(int(void_channel)):
        mask = group_ids == group_id
        area = int(mask.sum())
        if area < max(1, int(min_mask_area)):
            continue
        values = probs[group_id][mask]
        candidates.append({
            "query_id": int(group_id),
            "raw_confidence": float(pixel_max[mask].mean()),
            "mean_query_probability_on_mask": float(values.mean()),
            "mean_void_probability": float(probs[void_channel].mean()),
            "mean_void_probability_on_mask": float(probs[void_channel][mask].mean()),
            "mean_top1_top2_margin": float(
                (np.sort(probs, axis=0)[-1] - np.sort(probs, axis=0)[-2])[mask].mean()
            ),
            "mask_area": area,
            "mask": mask,
        })
    candidates.sort(key=lambda item: -item["raw_confidence"])
    return candidates


def _load_checkpoint_arch(args, opt) -> None:
    checkpoint_path = Path(args.resume)
    meta = {}
    metadata_path = checkpoint_path.parent / (
        "metadata_step_"
        + checkpoint_path.stem.replace("model_step_", "")
        + ".json"
    )
    if not metadata_path.is_file():
        metadata_path = checkpoint_path.parent / "metadata_best.json"
    if not metadata_path.is_file():
        metadata_path = checkpoint_path.parent / "metadata.json"
    if metadata_path.is_file():
        meta = json.load(open(metadata_path, encoding="utf-8"))
    config_yaml = checkpoint_path.parent / "config.yaml"
    if not config_yaml.is_file():
        config_yaml = checkpoint_path.parent.parent / "config.yaml"
    if config_yaml.is_file():
        try:
            import yaml

            def _options_ctor(loader, tag_suffix, node):
                return loader.construct_mapping(node, deep=True)

            yaml.add_multi_constructor(
                "!dataclass:",
                _options_ctor,
                Loader=yaml.UnsafeLoader,
            )
            yaml_cfg = yaml.load(
                config_yaml.read_text(encoding="utf-8"),
                Loader=yaml.UnsafeLoader,
            )
            for key, value in yaml_cfg.items():
                meta.setdefault(key, value)
        except Exception as exc:  # pragma: no cover - fallback only
            print(f"[lsm-eval] config.yaml parse failed: {exc}")
    if not meta:
        return
    if "prompt_clip_model_path" in meta:
        opt.prompt_clip_model_path = str(meta["prompt_clip_model_path"])
    if "semantic_v4_feature_dim" in meta:
        value = meta["semantic_v4_feature_dim"]
        if value is not None:
            opt.semantic_v4_feature_dim = int(value)
    if meta.get("semantic_v4_use_geometry") is not None:
        opt.semantic_v4_use_geometry = bool(
            meta["semantic_v4_use_geometry"]
        )
    if meta.get("semantic_v4_teacher_projection") is not None:
        opt.semantic_v4_teacher_projection = str(
            meta["semantic_v4_teacher_projection"]
        )
    if meta.get("prompt_unfreeze_tokengs") is not None:
        opt.prompt_unfreeze_tokengs = bool(meta["prompt_unfreeze_tokengs"])
    if (
        meta.get("instance_group_num_groups") is not None
        and int(args.num_groups) == 64
    ):
        opt.instance_group_num_groups = int(meta["instance_group_num_groups"])
    if meta.get("instance_group_conditioned_gaussians") is not None:
        opt.instance_group_conditioned_gaussians = bool(
            meta["instance_group_conditioned_gaussians"]
        )
    if meta.get("instance_group_condition_dim") is not None:
        opt.instance_group_condition_dim = int(
            meta["instance_group_condition_dim"]
        )
    if meta.get("instance_group_condition_heads") is not None:
        opt.instance_group_condition_heads = int(
            meta["instance_group_condition_heads"]
        )
    if meta.get("instance_group_condition_layers") is not None:
        opt.instance_group_condition_layers = int(
            meta["instance_group_condition_layers"]
        )
    if meta.get("instance_group_condition_residual_scale") is not None:
        opt.instance_group_condition_residual_scale = float(
            meta["instance_group_condition_residual_scale"]
        )
    if meta.get("instance_group_condition_assignment_temperature") is not None:
        opt.instance_group_condition_assignment_temperature = float(
            meta["instance_group_condition_assignment_temperature"]
        )
    if meta.get("instance_group_condition_gaussian_blend") is not None:
        opt.instance_group_condition_gaussian_blend = float(
            meta["instance_group_condition_gaussian_blend"]
        )
    if meta.get("instance_group_condition_per_gaussian") is not None:
        opt.instance_group_condition_per_gaussian = bool(
            meta["instance_group_condition_per_gaussian"]
        )
    if (
        meta.get("instance_group_condition_per_gaussian_opacity_scale")
        is not None
    ):
        opt.instance_group_condition_per_gaussian_opacity_scale = float(
            meta["instance_group_condition_per_gaussian_opacity_scale"]
        )
    if meta.get("instance_group_decoder") is not None:
        opt.instance_group_decoder = bool(meta["instance_group_decoder"])
    if meta.get("instance_group_decoder_layers") is not None:
        opt.instance_group_decoder_layers = int(
            meta["instance_group_decoder_layers"]
        )
    if meta.get("instance_group_group_token_refine") is not None:
        opt.instance_group_group_token_refine = bool(
            meta["instance_group_group_token_refine"]
        )
    if meta.get("instance_group_group_token_refine_scale_init") is not None:
        opt.instance_group_group_token_refine_scale_init = float(
            meta["instance_group_group_token_refine_scale_init"]
        )
    if meta.get("instance_group_group_token_refine_scale_max") is not None:
        opt.instance_group_group_token_refine_scale_max = float(
            meta["instance_group_group_token_refine_scale_max"]
        )
    if meta.get("instance_group_use_anchor_pos") is not None:
        opt.instance_group_use_anchor_pos = bool(
            meta["instance_group_use_anchor_pos"]
        )
    if meta.get("instance_group_per_gaussian") is not None:
        opt.instance_group_per_gaussian = bool(
            meta["instance_group_per_gaussian"]
        )
    if meta.get("instance_group_feature_dim") is not None:
        opt.instance_group_feature_dim = int(
            meta["instance_group_feature_dim"]
        )
    if meta.get("instance_group_residual_head") is not None:
        opt.instance_group_residual_head = bool(
            meta["instance_group_residual_head"]
        )
    if meta.get("instance_group_residual_scale") is not None:
        opt.instance_group_residual_scale = float(
            meta["instance_group_residual_scale"]
        )
    if meta.get("instance_group_adaptive_count") is not None:
        opt.instance_group_adaptive_count = bool(
            meta["instance_group_adaptive_count"]
        )
    if meta.get("instance_group_count_head") is not None:
        opt.instance_group_count_head = bool(meta["instance_group_count_head"])
    if meta.get("instance_group_count_hidden") is not None:
        opt.instance_group_count_hidden = int(
            meta["instance_group_count_hidden"]
        )
    if meta.get("lambda_instance_group_count") is not None:
        opt.lambda_instance_group_count = float(
            meta["lambda_instance_group_count"]
        )
    if meta.get("instance_group_pos_attn_layers") is not None:
        opt.instance_group_pos_attn_layers = int(
            meta["instance_group_pos_attn_layers"]
        )
    if meta.get("instance_group_pos_attn_scale") is not None:
        opt.instance_group_pos_attn_scale = float(
            meta["instance_group_pos_attn_scale"]
        )
    if meta.get("instance_group_dense_decoder") is not None:
        opt.instance_group_dense_decoder = bool(
            meta["instance_group_dense_decoder"]
        )
    if meta.get("instance_group_dense_feature_dim") is not None:
        opt.instance_group_dense_feature_dim = int(
            meta["instance_group_dense_feature_dim"]
        )
    if meta.get("instance_group_dense_upsample") is not None:
        opt.instance_group_dense_upsample = int(
            meta["instance_group_dense_upsample"]
        )
    if meta.get("instance_group_dense_multiscale") is not None:
        opt.instance_group_dense_multiscale = bool(
            meta["instance_group_dense_multiscale"]
        )
    if meta.get("instance_group_dense_scale") is not None:
        opt.instance_group_dense_scale = float(
            meta["instance_group_dense_scale"]
        )
    if meta.get("instance_group_dense_gate") is not None:
        opt.instance_group_dense_gate = bool(
            meta["instance_group_dense_gate"]
        )
    backbone_resume = meta.get("backbone_resume") or meta.get("resume")
    if backbone_resume:
        # Training resume path: frozen-backbone recipes save prompt-only
        # checkpoints (no TokenGS backbone), so the eval must load the
        # backbone that was actually used during training from this path.
        opt.backbone_resume = str(backbone_resume)
    if meta.get("instance_group_render_scale") is not None:
        opt.instance_group_render_scale = float(
            meta["instance_group_render_scale"]
        )
    if meta.get("num_gs_tokens") is not None:
        opt.num_gs_tokens = int(meta["num_gs_tokens"])
    if meta.get("num_dynamic_gs_tokens") is not None:
        opt.num_dynamic_gs_tokens = int(meta["num_dynamic_gs_tokens"])
    if meta.get("gs_token_init") is not None:
        opt.gs_token_init = str(meta["gs_token_init"])
    if meta.get("anchor_3d_query_source") is not None:
        opt.anchor_3d_query_source = str(meta["anchor_3d_query_source"])
    if meta.get("anchor_3d_pos_scale") is not None:
        opt.anchor_3d_pos_scale = float(meta["anchor_3d_pos_scale"])
    if meta.get("anchor_3d_pos_freqs") is not None:
        opt.anchor_3d_pos_freqs = int(meta["anchor_3d_pos_freqs"])
    if meta.get("anchor_3d_pass_decoder") is not None:
        opt.anchor_3d_pass_decoder = bool(meta["anchor_3d_pass_decoder"])
    if meta.get("instance_branch_independent") is not None:
        opt.instance_branch_independent = bool(
            meta["instance_branch_independent"]
        )
    if meta.get("instance_branch_num_groups") is not None:
        opt.instance_branch_num_groups = int(meta["instance_branch_num_groups"])
    if meta.get("instance_branch_anchor_dim") is not None:
        opt.instance_branch_anchor_dim = int(meta["instance_branch_anchor_dim"])
    if meta.get("instance_branch_num_heads") is not None:
        opt.instance_branch_num_heads = int(meta["instance_branch_num_heads"])
    if meta.get("instance_branch_num_layers") is not None:
        opt.instance_branch_num_layers = int(meta["instance_branch_num_layers"])
    if meta.get("instance_branch_gaussians_per_anchor") is not None:
        opt.instance_branch_gaussians_per_anchor = int(
            meta["instance_branch_gaussians_per_anchor"]
        )
    if meta.get("instance_branch_pos_offset_scale") is not None:
        opt.instance_branch_pos_offset_scale = float(
            meta["instance_branch_pos_offset_scale"]
        )
    if meta.get("instance_branch_scale_delta_amp") is not None:
        opt.instance_branch_scale_delta_amp = float(
            meta["instance_branch_scale_delta_amp"]
        )
    if meta.get("instance_branch_opacity_delta_amp") is not None:
        opt.instance_branch_opacity_delta_amp = float(
            meta["instance_branch_opacity_delta_amp"]
        )
    if meta.get("instance_branch_rgb_delta_amp") is not None:
        opt.instance_branch_rgb_delta_amp = float(
            meta["instance_branch_rgb_delta_amp"]
        )
    if meta.get("instance_branch_token_units") is not None:
        opt.instance_branch_token_units = bool(
            meta["instance_branch_token_units"]
        )
    if meta.get("instance_branch_units_per_token") is not None:
        opt.instance_branch_units_per_token = int(
            meta["instance_branch_units_per_token"]
        )
    if meta.get("instance_branch_embed_sample") is not None:
        opt.instance_branch_embed_sample = int(
            meta["instance_branch_embed_sample"]
        )
    if meta.get("instance_branch_unit_feat_dim") is not None:
        opt.instance_branch_unit_feat_dim = int(
            meta["instance_branch_unit_feat_dim"]
        )
    if meta.get("instance_branch_unit_layers") is not None:
        opt.instance_branch_unit_layers = int(
            meta["instance_branch_unit_layers"]
        )
    if meta.get("instance_branch_unit_temp") is not None:
        opt.instance_branch_unit_temp = float(meta["instance_branch_unit_temp"])
    if meta.get("instance_branch_unit_entropy") is not None:
        opt.instance_branch_unit_entropy = float(
            meta["instance_branch_unit_entropy"]
        )
    if meta.get("instance_branch_unit_compactness") is not None:
        opt.instance_branch_unit_compactness = float(
            meta["instance_branch_unit_compactness"]
        )
    if meta.get("instance_branch_unit_purity") is not None:
        opt.instance_branch_unit_purity = bool(
            meta["instance_branch_unit_purity"]
        )
    if meta.get("instance_branch_unit_purity_weight") is not None:
        opt.instance_branch_unit_purity_weight = float(
            meta["instance_branch_unit_purity_weight"]
        )
    if meta.get("instance_branch_gs_refine") is not None:
        opt.instance_branch_gs_refine = bool(meta["instance_branch_gs_refine"])
    if meta.get("instance_branch_gs_refine_scale") is not None:
        opt.instance_branch_gs_refine_scale = float(
            meta["instance_branch_gs_refine_scale"]
        )
    if meta.get("instance_branch_gs_refine_dim") is not None:
        opt.instance_branch_gs_refine_dim = int(
            meta["instance_branch_gs_refine_dim"]
        )
    if meta.get("instance_branch_scene_prototypes") is not None:
        opt.instance_branch_scene_prototypes = bool(
            meta["instance_branch_scene_prototypes"]
        )
    if meta.get("instance_branch_num_slots") is not None:
        opt.instance_branch_num_slots = int(meta["instance_branch_num_slots"])
    if meta.get("instance_branch_slot_dim") is not None:
        opt.instance_branch_slot_dim = int(meta["instance_branch_slot_dim"])
    if meta.get("instance_branch_slot_iterations") is not None:
        opt.instance_branch_slot_iterations = int(
            meta["instance_branch_slot_iterations"]
        )
    if meta.get("instance_branch_slot_temp") is not None:
        opt.instance_branch_slot_temp = float(
            meta["instance_branch_slot_temp"]
        )
    if meta.get("instance_branch_unit_embedding") is not None:
        opt.instance_branch_unit_embedding = bool(
            meta["instance_branch_unit_embedding"]
        )
    if meta.get("instance_branch_unit_encoder") is not None:
        opt.instance_branch_unit_encoder = bool(
            meta["instance_branch_unit_encoder"]
        )
    if meta.get("instance_branch_embed_dim") is not None:
        opt.instance_branch_embed_dim = int(meta["instance_branch_embed_dim"])
    if meta.get("instance_branch_embed_temp") is not None:
        opt.instance_branch_embed_temp = float(
            meta["instance_branch_embed_temp"]
        )
    if meta.get("instance_branch_embed_loss") is not None:
        opt.instance_branch_embed_loss = float(
            meta["instance_branch_embed_loss"]
        )
    if meta.get("instance_branch_embed_loss_mode") is not None:
        opt.instance_branch_embed_loss_mode = str(
            meta["instance_branch_embed_loss_mode"]
        )
    for _field, _cast in (
        ("instance_branch_embed_proto_temp", float),
        ("instance_branch_embed_margin", float),
        ("instance_branch_embed_margin_weight", float),
    ):
        if meta.get(_field) is not None:
            setattr(opt, _field, _cast(meta[_field]))
    for _field, _cast in (
        ("instance_branch_embed_center_push_margin", float),
        ("instance_branch_embed_center_push_weight", float),
    ):
        if meta.get(_field) is not None:
            setattr(opt, _field, _cast(meta[_field]))
    if meta.get("instance_branch_unit_image") is not None:
        opt.instance_branch_unit_image = bool(
            meta["instance_branch_unit_image"]
        )
    if meta.get("instance_branch_unit_image_dim") is not None:
        opt.instance_branch_unit_image_dim = int(
            meta["instance_branch_unit_image_dim"]
        )
    if meta.get("instance_branch_unit_dino") is not None:
        opt.instance_branch_unit_dino = bool(
            meta["instance_branch_unit_dino"]
        )
    if meta.get("instance_branch_unit_dino_dim") is not None:
        opt.instance_branch_unit_dino_dim = int(
            meta["instance_branch_unit_dino_dim"]
        )
    if meta.get("instance_branch_grounding") is not None:
        opt.instance_branch_grounding = bool(
            meta["instance_branch_grounding"]
        )
    if meta.get("instance_branch_grounding_dim") is not None:
        opt.instance_branch_grounding_dim = int(
            meta["instance_branch_grounding_dim"]
        )
    for key in (
        "instance_branch_grounding_3d_pull",
        "instance_branch_grounding_3d_push",
        "instance_branch_grounding_3d_margin",
    ):
        if meta.get(key) is not None:
            setattr(opt, key, float(meta[key]))
    for key in (
        "instance_branch_dynamic_queries",
        "instance_branch_num_queries",
        "instance_branch_query_dim",
        "instance_branch_query_layers",
    ):
        if meta.get(key) is not None:
            setattr(opt, key, meta[key])
    for key in (
        "instance_branch_center_offset",
        "instance_branch_center_offset_hidden",
    ):
        if meta.get(key) is not None:
            setattr(opt, key, meta[key])
    for _field, _cast in (
        ("instance_branch_dpg", bool),
        ("instance_branch_dpg_proto_dim", int),
        ("instance_branch_dpg_heads", int),
        ("instance_branch_dpg_layers", int),
        ("instance_branch_dpg_proto_weight", float),
    ):
        if meta.get(_field) is not None:
            setattr(opt, _field, _cast(meta[_field]))
    for _field, _cast in (
        ("instance_branch_render_space", bool),
        ("instance_branch_render_space_pull", float),
        ("instance_branch_render_space_push", float),
        ("instance_branch_render_space_cross", float),
        ("instance_branch_render_space_margin_push", float),
        ("instance_branch_render_space_margin_cross", float),
        ("instance_branch_render_space_info_nce", float),
        ("instance_branch_render_space_info_temp", float),
        ("instance_branch_render_space_info_samples", int),
    ):
        if meta.get(_field) is not None:
            setattr(opt, _field, _cast(meta[_field]))
    for _field, _cast in (
        ("instance_branch_direct_gs", bool),
        ("instance_branch_direct_gs_embed_dim", int),
        ("instance_branch_direct_gs_hidden", int),
        ("instance_branch_direct_gs_dino", bool),
        ("instance_branch_direct_gs_pull", float),
        ("instance_branch_direct_gs_push", float),
        ("instance_branch_direct_gs_cross", float),
        ("instance_branch_direct_gs_margin_push", float),
        ("instance_branch_direct_gs_margin_cross", float),
        ("instance_branch_direct_gs_info_nce", float),
        ("instance_branch_direct_gs_info_temp", float),
        ("instance_branch_direct_gs_info_samples", int),
    ):
        if meta.get(_field) is not None:
            setattr(opt, _field, _cast(meta[_field]))
    if meta.get("instance_branch_pseudo_conf") is not None:
        opt.instance_branch_pseudo_conf = float(
            meta["instance_branch_pseudo_conf"]
        )
    if meta.get("instance_branch_pseudo_min_views") is not None:
        opt.instance_branch_pseudo_min_views = int(
            meta["instance_branch_pseudo_min_views"]
        )
    if meta.get("instance_branch_pseudo_unit_min_mass") is not None:
        opt.instance_branch_pseudo_unit_min_mass = float(
            meta["instance_branch_pseudo_unit_min_mass"]
        )
    if meta.get("instance_branch_scene_assignment") is not None:
        opt.instance_branch_scene_assignment = bool(
            meta["instance_branch_scene_assignment"]
        )
    for _field, _cast in (
        ("instance_branch_sic_units", bool),
        ("instance_branch_sic_queries", int),
        ("instance_branch_sic_dim", int),
        ("instance_branch_sic_heads", int),
        ("instance_branch_sic_layers", int),
        ("instance_branch_sic_usage", float),
    ):
        if meta.get(_field) is not None:
            setattr(opt, _field, _cast(meta[_field]))
    for _field, _cast in (
        ("instance_branch_abs_units", bool),
        ("abs_true_shared_units", bool),
        ("tsh_instance_warmup_steps", int),
        ("tsh_instance_ramp_end_steps", int),
        ("tsh_lambda_instance", float),
        ("tsh_unit_gradient_multiplier_max", float),
        ("tsh_num_groups", int),
        ("tsh_num_heads", int),
        ("tsh_num_layers", int),
        ("abs_bootstrap_steps", int),
        ("abs_teacher_decay_steps", int),
        ("abs_instance_warmup_steps", int),
        ("abs_teacher_gs_weight", float),
        ("abs_teacher_rgb_weight", float),
    ):
        if meta.get(_field) is not None:
            setattr(opt, _field, _cast(meta[_field]))
    for _field, _cast in (
        ("instance_branch_scene_slots", int),
        ("instance_branch_scene_slot_iters", int),
        ("instance_branch_scene_slot_temp", float),
        ("instance_branch_scene_slot_pos_weight", float),
        ("instance_branch_scene_slot_entropy", float),
        ("instance_branch_scene_slot_void", float),
        ("lambda_scene_assignment_unit", float),
    ):
        if meta.get(_field) is not None:
            setattr(opt, _field, _cast(meta[_field]))
    if meta.get("instance_branch_cluster_pos_weight") is not None:
        opt.instance_branch_cluster_pos_weight = float(
            meta["instance_branch_cluster_pos_weight"]
        )
    if meta.get("instance_branch_cluster_eps") is not None:
        opt.instance_branch_cluster_eps = float(
            meta["instance_branch_cluster_eps"]
        )
    if meta.get("instance_branch_void_fg_share") is not None:
        opt.instance_branch_void_fg_share = float(
            meta["instance_branch_void_fg_share"]
        )
    if (
        getattr(opt, "instance_branch_independent", False)
        or getattr(opt, "instance_branch_token_units", False)
    ):
        # The eval's group-count / void-channel bookkeeping must match the
        # independent branch's own group count.
        opt.instance_group_num_groups = int(
            getattr(opt, "instance_branch_num_groups", 100)
        )
    if getattr(opt, "instance_branch_scene_assignment", False):
        opt.instance_group_num_groups = int(
            getattr(opt, "instance_branch_scene_slots", 100)
        )
    # Backbone architecture fields (needed for the latent-bottleneck
    # backbone, which has a different encoder/decoder/latent config than the
    # old 1024-token re10k backbone).
    for _field, _cast in (
        ("enc_depth", int),
        ("dec_depth", int),
        ("dec_patch_size", int),
        ("enc_embed_dim", int),
        ("enc_num_heads", int),
        ("mlp_ratio", float),
        ("patch_size", int),
        ("dec_init_values", float),
        ("clip_head_z_init", float),
        ("clip_head_readout_std", float),
        ("gaussian_z_offset", float),
        ("opacity_bias", float),
        ("gs_token_std", float),
        ("use_multiscale_encoder", bool),
        ("use_latent_bottleneck", bool),
        ("num_latents", int),
        ("latent_cross_attn_depth", int),
        ("camera_normalization_method", str),
        ("camera_scale_method", str),
        ("img_size", tuple),
        ("num_views", int),
        ("num_input_views", int),
    ):
        if meta.get(_field) is not None:
            setattr(opt, _field, _cast(meta[_field]))
    if meta.get("multiscale_encoder_layers") is not None:
        opt.multiscale_encoder_layers = tuple(
            int(value) for value in meta["multiscale_encoder_layers"]
        )
    if meta.get("prompt_tokengs_checkpoint") is not None:
        opt.prompt_tokengs_checkpoint = str(
            meta["prompt_tokengs_checkpoint"]
        )
    if meta.get("lambda_semantic_feature") is not None:
        opt.lambda_semantic_feature = float(meta["lambda_semantic_feature"])
    if meta.get("semantic_residual_scale") is not None:
        opt.semantic_residual_scale = float(meta["semantic_residual_scale"])
    # Query-memory probe checkpoints carry their architecture in config.yaml;
    # propagate these fields before constructing the model so the refiner is
    # restored rather than silently evaluated with the legacy head.
    for _field, _cast in (
        ("tsh_query_memory_refine", bool),
        ("tsh_query_memory_refine_rounds", int),
        ("tsh_query_memory_refine_gate_steps", int),
        ("tsh_query_memory_refine_probe", bool),
    ):
        if meta.get(_field) is not None:
            setattr(opt, _field, _cast(meta[_field]))
    print(
        f"[lsm-eval] arch from "
        f"{metadata_path.name if metadata_path.is_file() else 'config.yaml'}: "
        f"clip={Path(str(opt.prompt_clip_model_path)).name} "
        f"feature_dim={opt.semantic_v4_feature_dim} "
        f"unfreeze={opt.prompt_unfreeze_tokengs}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--model_type", default="semantic_tokengs_v6"
    )
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument(
        "--min_pred_pixels",
        type=int,
        default=1,
        help="Discard predicted masks smaller than this many pixels.",
    )
    parser.add_argument(
        "--min_gt_pixels",
        type=int,
        default=1,
        help="Discard GT instances smaller than this many pixels.",
    )
    parser.add_argument(
        "--max_predictions_per_image",
        type=int,
        default=100,
        help=(
            "Keep at most this many highest-confidence masks per image "
            "(100 matches the standard COCO max-detections setting)."
        ),
    )
    parser.add_argument(
        "--backbone-resume",
        default="",
        help=(
            "Full checkpoint whose frozen TokenGS backbone should be loaded "
            "for a prompt-only (frozen-backbone) resume checkpoint. "
            "Overrides the path recovered from metadata/config."
        ),
    )
    parser.add_argument(
        "--lsm_manifest",
        default=(
            "/space0/mawb/tokengs/data/scannet_prompt/"
            "lsm_instance_eval_manifest.json"
        ),
    )
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=15)
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=0,
        help="Limit evaluation to the first N scenes (0 = all).",
    )
    parser.add_argument(
        "--audit_teacher",
        action="store_true",
        help=(
            "Read-only teacher/student audit: also render the frozen "
            "base8k teacher once per scene and report teacher_psnr / "
            "student_psnr / psnr_delta per scene.  Student results and the "
            "old_gs_head_calls audit below still refer to the student-only "
            "path; teacher calls are counted separately."
        ),
    )
    parser.add_argument(
        "--instance-branch-cluster-eps",
        type=float,
        default=None,
        help="Override the unit-embedding clustering distance threshold.",
    )
    parser.add_argument(
        "--instance-branch-cluster-pos-weight",
        type=float,
        default=None,
        help="Override the 3D position weight in unit clustering.",
    )
    parser.add_argument(
        "--instance-branch-void-fg-share",
        type=float,
        default=None,
        help="Override the cluster foreground-share void threshold.",
    )
    parser.add_argument(
        "--prune-top-k",
        type=int,
        default=0,
        help=(
            "Keep only the top-K most-used groups per view and dump the "
            "rest into the void channel (0 = no pruning). Useful for "
            "scene-assignment heads with a fixed slot budget."
        ),
    )
    parser.add_argument(
        "--ttt_steps",
        type=int,
        default=0,
        help=(
            "Test-time refine the instance group head on the 8 context "
            "views' GT instance masks before evaluating (0 = off)."
        ),
    )
    parser.add_argument("--ttt_lr", type=float, default=1e-3)
    parser.add_argument(
        "--ranking_audit_dir",
        default="",
        help="Optional directory for exact pre-AP prediction/GT raw cache.",
    )
    args = parser.parse_args()
    args.resume = remap_path(args.resume)
    args.workspace = remap_path(args.workspace)
    args.lsm_manifest = remap_path(args.lsm_manifest)

    if args.num_input_views != 8 or args.num_views != 15:
        raise ValueError(
            "InstOk3D Table 2 evaluation requires --num_input_views 8 "
            "and --num_views 15"
        )
    if args.max_predictions_per_image <= 0:
        raise ValueError("--max_predictions_per_image must be positive")
    manifest_audit = _audit_lsm_manifest(args.lsm_manifest)
    print(
        "[lsm-eval] audited InstOk3D split: "
        f"{manifest_audit['scene_count']} scenes, "
        f"{manifest_audit['split']}, sha256={manifest_audit['sha256'][:12]}"
    )
    print(
        "[lsm-eval] PROTOCOL NOTE: this dataset uses ScanNet .sens poses; "
        "InstOk3D reports scene-level COLMAP cameras. AP output is therefore "
        "auditable but not exact official-protocol parity."
    )

    opt = config_defaults["eval_scannet_lsm_instance"]
    opt.model_type = args.model_type
    opt.resume = args.resume
    opt.workspace = args.workspace
    opt.experiment_name = Path(args.workspace).name
    opt.instance_group_num_groups = int(args.num_groups)
    opt.num_input_views = int(args.num_input_views)
    opt.num_views = int(args.num_views)
    opt.dataset_kwargs = {
        **dict(opt.dataset_kwargs or {}),
        "lsm_manifest_path": args.lsm_manifest,
    }
    _load_checkpoint_arch(args, opt)
    if args.instance_branch_cluster_eps is not None:
        opt.instance_branch_cluster_eps = args.instance_branch_cluster_eps
    if args.instance_branch_cluster_pos_weight is not None:
        opt.instance_branch_cluster_pos_weight = (
            args.instance_branch_cluster_pos_weight
        )
    if args.instance_branch_void_fg_share is not None:
        opt.instance_branch_void_fg_share = args.instance_branch_void_fg_share
    if args.backbone_resume:
        opt.backbone_resume = args.backbone_resume
    # 240-only: rewrite any remaining /space0/mawb/tokengs runtime paths that
    # were copied verbatim from the 108 config/metadata (keeps migrated
    # config.yaml/metadata byte-identical; this is the single remap point).
    remap_opt(opt)
    print(
        "[lsm-eval][240] effective prompt_tokengs_checkpoint="
        f"{opt.prompt_tokengs_checkpoint}"
    )
    print(
        "[lsm-eval][240] effective prompt_clip_model_path="
        f"{opt.prompt_clip_model_path}"
    )
    print(f"[lsm-eval][240] effective backbone_resume={opt.backbone_resume}")
    print(f"[lsm-eval][240] effective lsm_manifest={args.lsm_manifest}")

    Path(args.workspace).mkdir(parents=True, exist_ok=True)
    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())

    from safetensors.torch import load_file

    model = model_registry[opt.model_type](opt)
    # Tolerant load: joint checkpoints store only the trainable heads (the
    # geometry backbone is loaded by the model constructor); wide7l-style
    # checkpoints include the backbone and load fully here as well.
    resume_ckpt = load_file(args.resume, device="cpu")
    torch.nn.Module.load_state_dict(
        model, resume_ckpt, strict=False
    )
    if bool(getattr(opt, "abs_true_shared_units", False)):
        abs_native = {
            name: param
            for name, param in model.absolute_gs_head.named_parameters()
        }
        tsh_native = {
            name: param
            for name, param in model.tsh_instance_head.named_parameters()
        }
        abs_loaded = sum(
            1
            for name in abs_native
            if (
                f"absolute_gs_head.{name}" in resume_ckpt
                and resume_ckpt[f"absolute_gs_head.{name}"].shape
                == abs_native[name].shape
            )
        )
        tsh_loaded = sum(
            1
            for name in tsh_native
            if (
                f"tsh_instance_head.{name}" in resume_ckpt
                and resume_ckpt[f"tsh_instance_head.{name}"].shape
                == tsh_native[name].shape
            )
        )
        print(
            f"[lsm-eval] absolute_gs_head loaded {abs_loaded}/24 "
            f"tsh_instance_head loaded {tsh_loaded}/{len(tsh_native)}"
        )
        assert abs_loaded == 24
        assert tsh_loaded == len(tsh_native)
    if not any(key.startswith("enc_dec_backbone.") for key in resume_ckpt):
        # Frozen-backbone recipe: the checkpoint intentionally excludes the
        # TokenGS backbone; load the backbone that training actually used.
        backbone_path = str(getattr(opt, "backbone_resume", "") or "")
        if backbone_path and Path(backbone_path).is_file():
            backbone_ckpt = load_file(backbone_path, device="cpu")
            frozen_prefixes = (
                "enc_dec_backbone.",
                "patch_embed.",
                "patch_plucker_embed.",
                "activation_head.",
                "anchor_pos_encoder.",
            )
            native_state = torch.nn.Module.state_dict(model)
            loadable = {
                key: value
                for key, value in backbone_ckpt.items()
                if (key.startswith(frozen_prefixes) or key == "gs_tokens")
                and key in native_state
                and native_state[key].shape == value.shape
            }
            torch.nn.Module.load_state_dict(model, loadable, strict=False)
            print(
                f"[lsm-eval] frozen-backbone eval: loaded "
                f"{len(loadable)} backbone keys from {backbone_path}"
            )
        else:
            print(
                "[lsm-eval] WARNING: prompt-only checkpoint without a "
                "backbone_resume; using the constructor's default TokenGS "
                "backbone (results will not match the trained model)."
            )
    model.eval()
    model = model.cuda()
    num_groups = int(opt.instance_group_num_groups)
    # The void channel is always the last rendered channel; for models with
    # scene-adaptive cluster counts (e.g. unit-embedding clustering) the
    # channel count varies per scene, so derive it from the rendered output.
    void_channel = None  # set per scene from probability.shape[1]-1

    def _save_instance_head(model):
        return {
            key: value.detach().clone()
            for key, value in model.instance_group_head.state_dict().items()
        }

    def _run_instance_ttt(model, data) -> None:
        """Fine-tune instance_group_head on context-view GT masks."""
        model_input, _ = split_data(data, opt)
        with torch.no_grad():
            reconstruction, gs_token_hidden, _ = (
                model._forward_prompt_reconstruction(model_input)
            )
            gaussians = reconstruction.gaussians
        saved = {}
        for name, parameter in model.named_parameters():
            saved[name] = parameter.requires_grad
            parameter.requires_grad_(False)
        for parameter in model.instance_group_head.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(
            [
                parameter
                for parameter in model.instance_group_head.parameters()
                if parameter.requires_grad
            ],
            lr=args.ttt_lr,
        )
        cam_view = data["cam_view_input"]
        intrinsics = data["intrinsics_input"]
        labels = data["instance_label_input"]
        for _ in range(args.ttt_steps):
            optimizer.zero_grad()
            loss, _ = model.instance_group_loss_on_views(
                gs_token_hidden, gaussians, cam_view, intrinsics, labels
            )
            loss.backward()
            optimizer.step()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(saved[name])

    gaussians_source = (
        "old_gs_head"
        if not (
            bool(getattr(opt, "instance_branch_abs_units", False))
            and hasattr(model, "absolute_gs_head")
        )
        else "absolute_student"
    )
    old_head_calls = 0
    if gaussians_source == "absolute_student":
        native = {
            name: param
            for name, param in model.absolute_gs_head.named_parameters()
        }
        loaded = 0
        missing = []
        norm = 0.0
        for name, param in native.items():
            key = f"absolute_gs_head.{name}"
            if key in resume_ckpt and resume_ckpt[key].shape == param.shape:
                loaded += 1
                norm += float(param.detach().float().abs().sum())
            else:
                missing.append(name)
        print(
            "[lsm-eval] gaussians_source=absolute_student "
            f"abs_loaded={loaded}/{len(native)} abs_norm={norm:.6f} "
            f"missing={missing[:5]}"
        )
        _orig_activation = model.activation_head.forward

        def _counting_activation(*args, **kwargs):
            nonlocal old_head_calls
            old_head_calls += 1
            return _orig_activation(*args, **kwargs)

        model.activation_head.forward = _counting_activation
    else:
        print(
            "[lsm-eval] gaussians_source=old_gs_head "
            "(abs mode off or no abs decoder in eval arch)"
        )

    per_scene = {}
    pooled_pred_masks = []
    pooled_pred_scores = []
    pooled_pred_image_ids = []
    pooled_gt_masks = []
    pooled_gt_image_ids = []
    teacher_audit_calls = 0
    ranking_audit_root = (
        Path(args.ranking_audit_dir) if args.ranking_audit_dir else None
    )
    if ranking_audit_root is not None:
        ranking_audit_root.mkdir(parents=True, exist_ok=True)
    for i, data in enumerate(test_loader):
        if args.max_scenes > 0 and i >= args.max_scenes:
            break
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        saved_head = None
        if args.ttt_steps > 0:
            saved_head = _save_instance_head(model)
            _run_instance_ttt(model, data)
        with torch.inference_mode():
            out = model(data, compute_quality_metrics=True)
        if saved_head is not None:
            # Restore the original head weights so each scene's TTT starts
            # fresh (the eval above already used the TTT-adapted head).
            model.instance_group_head.load_state_dict(saved_head)
        scene_entry = {
            "psnr": float(out["psnr"].detach()),
            "ssim": float(out["ssim"].detach()),
            "lpips": float(out["lpips"].detach()),
        }
        if args.audit_teacher:
            model_input, supervision = split_data(data, opt)
            calls_before = old_head_calls
            with torch.no_grad():
                teacher_recon, _teacher_hidden, teacher_rgb_res = (
                    model._forward_prompt_reconstruction(model_input)
                )
            teacher_calls_this_scene = old_head_calls - calls_before
            teacher_audit_calls += teacher_calls_this_scene
            t_pred = teacher_rgb_res["images_pred"].clamp(0, 1)
            t_gt = supervision.images_output.clamp(0, 1)
            t_mse = (t_pred - t_gt).square().mean(dim=(2, 3, 4))
            teacher_psnr = float(
                (-10.0 * torch.log10(t_mse.clamp_min(1e-10))).mean()
            )
            student_psnr = float(out["psnr"].detach())
            scene_entry["teacher_psnr"] = teacher_psnr
            scene_entry["student_psnr"] = student_psnr
            scene_entry["psnr_delta"] = float(
                teacher_psnr - student_psnr
            )
            scene_entry["teacher_old_head_calls"] = teacher_calls_this_scene
        probability = out["rendered_instance_group_probability"]  # [B,G+1,V,1,H,W]
        void_channel = probability.shape[1] - 1
        instance_labels = data["instance_label_output"].long()
        adaptive = bool(getattr(opt, "instance_group_adaptive_count", False))
        predicted_counts = out.get("predicted_instance_count")
        if adaptive and predicted_counts is None:
            print(
                "[lsm-eval] WARNING: adaptive count enabled but the "
                "checkpoint has no count head; keeping all groups."
            )
            adaptive = False
        scene_name = data["scene_name"][0]
        batch_size, _, view_count, _, height, width = probability.shape
        pred_masks = []
        pred_scores = []
        pred_image_ids = []
        gt_masks = []
        gt_image_ids = []
        audit_predictions = []
        audit_gt = []
        audit_pred_masks = []
        audit_gt_masks = []
        for b in range(batch_size):
            for v in range(view_count):
                # LSM / InstOk3D protocol: per-scene AP. All target views of
                # a scene share one matching pool, so a predicted mask can
                # match the same instance's mask in any test view and the
                # per-scene Hungarian/confidence matching is scene-level.
                image_id = f"{scene_name}:b{b}"
                probs = probability[b, :, v, 0].float().cpu().numpy()
                if adaptive:
                    active = int(predicted_counts[b].item())
                    probs = _prune_inactive_groups(
                        probs, active, void_channel
                    )
                elif args.prune_top_k > 0:
                    probs = _prune_inactive_groups(
                        probs, args.prune_top_k, void_channel
                    )
                masks, scores = masks_from_group_probs(
                    probs,
                    void_channel=void_channel,
                    min_mask_area=args.min_pred_pixels,
                )
                masks = masks[: args.max_predictions_per_image]
                scores = scores[: args.max_predictions_per_image]
                gt_map = instance_labels[b, v].cpu().numpy()
                gts = gt_masks_from_instance_map(
                    gt_map, min_mask_area=args.min_gt_pixels
                )
                pred_masks.extend(masks)
                pred_scores.extend(scores)
                pred_image_ids.extend([image_id] * len(masks))
                gt_masks.extend(gts)
                gt_image_ids.extend([image_id] * len(gts))
                if ranking_audit_root is not None:
                    candidates = _ranking_audit_candidates(
                        probs, void_channel, args.min_pred_pixels
                    )
                    kept_count = min(
                        len(candidates), args.max_predictions_per_image
                    )
                    for rank, candidate in enumerate(candidates):
                        mask_index = len(audit_pred_masks)
                        audit_pred_masks.append(candidate.pop("mask"))
                        candidate.update({
                            "prediction_id": f"{scene_name}:b{b}:v{v}:q{candidate['query_id']}",
                            "image_id": image_id,
                            "view_index": int(v),
                            "candidate_rank": int(rank),
                            "kept_by_evaluator": bool(rank < kept_count),
                            "filtered_by": (
                                None if rank < kept_count else "max_predictions_per_image"
                            ),
                            "mask_index": int(mask_index),
                        })
                        audit_predictions.append(candidate)
                    for gt_index, gt_mask in enumerate(gts):
                        gt_mask_index = len(audit_gt_masks)
                        audit_gt_masks.append(gt_mask)
                        audit_gt.append({
                            "gt_id": f"{scene_name}:b{b}:v{v}:gt{gt_index}",
                            "image_id": image_id,
                            "view_index": int(v),
                            "instance_index": int(gt_index),
                            "mask_area": int(gt_mask.sum()),
                            "mask_index": int(gt_mask_index),
                        })
        pooled_pred_masks.extend(pred_masks)
        pooled_pred_scores.extend(pred_scores)
        pooled_pred_image_ids.extend(pred_image_ids)
        pooled_gt_masks.extend(gt_masks)
        pooled_gt_image_ids.extend(gt_image_ids)
        mask_diag = _mask_diagnostics(
            pred_masks, gt_masks, pred_image_ids, gt_image_ids
        )
        active_query_threshold = 0.01
        query_usage = probability[:, :-1].float().mean(dim=(0, 2, 3, 4, 5))
        active_queries = int((query_usage > active_query_threshold).sum().item())
        void_ratio = float(probability[:, -1].float().mean().detach())
        results = instance_ap(
            pred_masks,
            pred_scores,
            gt_masks,
            thresholds=(0.25, 0.5, 0.75),
            vectorized=True,
            pred_image_ids=pred_image_ids,
            gt_image_ids=gt_image_ids,
        )
        coco_ap = instance_ap(
            pred_masks,
            pred_scores,
            gt_masks,
            thresholds=tuple(t / 100 for t in range(50, 100, 5)),
            vectorized=True,
            pred_image_ids=pred_image_ids,
            gt_image_ids=gt_image_ids,
        )
        per_scene[str(scene_name)] = {
            **results,
            "ap": coco_ap["ap_mean"],
            "num_gt_instances": len(gt_masks),
            "num_pred_instances": len(pred_masks),
            "num_images": view_count * batch_size,
            **mask_diag,
            "active_queries": active_queries,
            "void_ratio": void_ratio,
            "pred_gt_ratio": len(pred_masks) / max(1, len(gt_masks)),
            **scene_entry,
        }
        if ranking_audit_root is not None:
            scene_dir = ranking_audit_root / str(scene_name)
            scene_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                scene_dir / "masks.npz",
                pred_masks=np.asarray(audit_pred_masks, dtype=np.uint8),
                gt_masks=np.asarray(audit_gt_masks, dtype=np.uint8),
            )
            with (scene_dir / "records.json").open("w", encoding="utf-8") as handle:
                json.dump({
                    "scene": str(scene_name),
                    "predictions": audit_predictions,
                    "ground_truth": audit_gt,
                    "evaluator": {
                        "min_pred_pixels": int(args.min_pred_pixels),
                        "min_gt_pixels": int(args.min_gt_pixels),
                        "max_predictions_per_image": int(args.max_predictions_per_image),
                        "score_threshold": None,
                        "score_definition": "mean max(nonvoid group probability) over predicted mask",
                        "void_definition": "last rendered instance-group probability channel",
                    },
                }, handle, indent=2)
        print(
            f"[lsm-eval] {scene_name}: "
            f"AP={coco_ap['ap_mean']:.4f} AP50={results['ap_50']:.4f} "
            f"AP25={results['ap_25']:.4f} gt={len(gt_masks)} "
            f"pred={len(pred_masks)} psnr={float(out['psnr']):.2f}"
        )

    def _mean(key: str) -> float:
        values = [entry[key] for entry in per_scene.values()]
        return float(sum(values) / max(1, len(values)))

    pooled_ap = instance_ap(
        pooled_pred_masks,
        pooled_pred_scores,
        pooled_gt_masks,
        thresholds=(0.25, 0.5, 0.75),
        vectorized=True,
        pred_image_ids=pooled_pred_image_ids,
        gt_image_ids=pooled_gt_image_ids,
    )
    pooled_coco_ap = instance_ap(
        pooled_pred_masks,
        pooled_pred_scores,
        pooled_gt_masks,
        thresholds=tuple(t / 100 for t in range(50, 100, 5)),
        vectorized=True,
        pred_image_ids=pooled_pred_image_ids,
        gt_image_ids=pooled_gt_image_ids,
    )
    payload = {
        "protocol": {
            "name": "lsm_scannet_class_agnostic_2d_instance",
            "version": 3,
            "public_reference": "InstOk3D Table 2 and supplementary setup",
            "context_views": int(args.num_input_views),
            "target_views": int(args.num_views - args.num_input_views),
            "expected_scene_count": 40,
            "evaluated_scene_count": len(per_scene),
            "partial_evaluation": len(per_scene) != 40,
            "manifest": manifest_audit,
            "camera_source": "scannet_sens_rgbd_pose",
            "reference_camera_source": "scene_level_colmap",
            "matching": "confidence_ordered_greedy_same_image",
            "ap_interpolation": "101_point",
            "ap_thresholds": [round(t / 100, 2) for t in range(50, 100, 5)],
            "ap25_threshold": 0.25,
            "ap50_threshold": 0.5,
            "prediction_score": "mean_rendered_group_probability_over_mask",
            "max_predictions_per_image": int(args.max_predictions_per_image),
            "min_pred_pixels": int(args.min_pred_pixels),
            "min_gt_pixels": int(args.min_gt_pixels),
            "ignore_instance_ids": [0, 255, -1],
            "scene_macro_aggregation": "mean_over_scenes",
            "pooled_aggregation": "all_images_with_image_local_matching",
            "test_time_training_steps": int(args.ttt_steps),
            "official_code_parity": False,
            "comparison_eligibility": {
                "instok3d_table_2": "partial_protocol_alignment",
                "instancesplat_iggt_tracking": "not_applicable_different_benchmark",
            },
            "known_parity_gaps": [
                "ScanNet .sens poses are used instead of the paper's shared COLMAP cameras",
                "the paper does not specify scene-macro versus dataset-pooled AP aggregation",
                "the official InstOk3D evaluation implementation is not public",
            ],
            "different_protocol_note": (
                "InstanceSplat reports T-mIoU and T-SR on the IGGT tracking "
                "benchmark; those metrics must be evaluated separately."
            ),
        },
        "label": args.label,
        "checkpoint": args.resume,
        "gaussians_source": gaussians_source,
        "old_gs_head_calls": old_head_calls,
        "num_scenes": len(per_scene),
        "mean_ap": _mean("ap"),
        "mean_ap25": _mean("ap_25"),
        "mean_ap50": _mean("ap_50"),
        "mean_ap75": _mean("ap_75"),
        "pooled_ap": pooled_coco_ap["ap_mean"],
        "pooled_ap25": pooled_ap["ap_25"],
        "pooled_ap50": pooled_ap["ap_50"],
        "pooled_ap75": pooled_ap["ap_75"],
        "mean_psnr": _mean("psnr"),
        "mean_ssim": _mean("ssim"),
        "mean_lpips": _mean("lpips"),
        "mean_best_gt_iou": _mean("mean_best_gt_iou"),
        "recall_iou25": _mean("recall_iou25"),
        "recall_iou50": _mean("recall_iou50"),
        "recall_iou75": _mean("recall_iou75"),
        "mean_active_queries": _mean("active_queries"),
        "mean_void_ratio": _mean("void_ratio"),
        "mean_pred_gt_ratio": _mean("pred_gt_ratio"),
        "mean_teacher_psnr": (
            _mean("teacher_psnr") if args.audit_teacher else None
        ),
        "mean_psnr_delta": (
            _mean("psnr_delta") if args.audit_teacher else None
        ),
        "audit_teacher_total_old_head_calls": (
            teacher_audit_calls if args.audit_teacher else 0
        ),
        "student_only_old_gs_head_calls": (
            old_head_calls - teacher_audit_calls
        ),
        "per_scene": per_scene,
    }
    output_path = Path(args.workspace) / "instance_ap.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[lsm-eval] scenes={len(per_scene)} "
        f"mean AP={payload['mean_ap']:.4f} AP50={payload['mean_ap50']:.4f} "
        f"AP25={payload['mean_ap25']:.4f} "
        f"pooled AP={payload['pooled_ap']:.4f} AP50={payload['pooled_ap50']:.4f} "
        f"AP25={payload['pooled_ap25']:.4f} "
        f"PSNR={payload['mean_psnr']:.2f} SSIM={payload['mean_ssim']:.4f} "
        f"LPIPS={payload['mean_lpips']:.4f}"
    )
    if args.audit_teacher:
        print(
            "[lsm-eval][audit-teacher] "
            f"teacher_psnr={payload['mean_teacher_psnr']:.2f} "
            f"student_psnr={payload['mean_psnr']:.2f} "
            f"delta={payload['mean_psnr_delta']:.2f} "
            f"teacher_old_head_calls={payload['audit_teacher_total_old_head_calls']} "
            f"student_only_old_gs_head_calls={payload['student_only_old_gs_head_calls']}"
        )
    print(f"[lsm-eval] wrote {output_path}")
    if gaussians_source == "absolute_student":
        print(
            f"[lsm-eval] gaussians_source=absolute_student "
            f"old_gs_head_calls={payload['student_only_old_gs_head_calls']}"
        )


if __name__ == "__main__":
    main()
