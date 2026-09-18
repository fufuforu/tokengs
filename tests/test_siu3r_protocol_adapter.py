from __future__ import annotations

import inspect
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tokengs.siu3r_protocol import (
    PAIR_SHA256,
    build_instance_masks,
    concat_views_height,
    depth_metrics,
    global_semantic_iou,
    normalize_intrinsics,
    validate_val_pairs,
)
from tokengs.siu3r_adapter import build_input, validate_model_contract


ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "workspace/siu3r_protocol_alignment_v1/val_pair.json"


def _official_torch_runtime() -> bool:
    expected = {"torch": "2.4.1", "torchvision": "0.19.0", "torchmetrics": "1.7.3"}
    # torchvision reports a local build suffix in some official CUDA envs;
    # only the major/minor API family matters for the dynamic smoke guard.
    try:
        return (
            importlib.metadata.version("torch").startswith("2.4.1")
            and importlib.metadata.version("torchvision").startswith("0.19")
            and importlib.metadata.version("torchmetrics") == expected["torchmetrics"]
        )
    except importlib.metadata.PackageNotFoundError:
        return False


def test_official_manifest_sha_cardinality_and_schema():
    audit = validate_val_pairs(PAIRS)
    assert audit["sha256"] == PAIR_SHA256
    assert audit["records"] == 1860
    assert audit["unique_scenes"] == 312
    assert audit["sha256_matches"] and audit["cardinality_matches"]


def test_official_manifest_preserves_each_2_plus_6_record():
    records = json.loads(PAIRS.read_text())
    assert all(len(row["context_ids"]) == 2 and len(row["target_ids"]) == 6 for row in records)
    assert all(
        len(set(row["context_ids"])) == 2
        and len(set(row["target_ids"])) == 6
        and len(set(row["target_ids"]) - set(row["context_ids"])) == 4
        for row in records
    )
    assert records[0]["target_ids"] == [1727, 1729, 1732, 1738, 1739, 1744]
    assert records[0]["context_ids"] == [1727, 1744]


def test_256_preprocessing_and_intrinsics_normalization():
    K = np.array([[512.0, 0.0, 320.0], [0.0, 480.0, 240.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(normalize_intrinsics(K), [[2.0, 0.0, 1.25], [0.0, 1.875, 0.9375], [0.0, 0.0, 1.0]])
    data = build_input(
        context_images=np.zeros((2, 3, 256, 256), dtype=np.float32),
        context_intrinsics=np.stack([normalize_intrinsics(K)] * 2),
        target_cam_to_world=np.stack([np.eye(4)] * 6),
        target_intrinsics=np.stack([normalize_intrinsics(K)] * 6),
        context_ids=[1727, 1744],
        target_ids=[1727, 1729, 1732, 1738, 1739, 1744],
    )
    assert data.context_images.shape == (2, 3, 256, 256)


def test_multiview_concat_occupies_non_overlapping_height_regions():
    views = [np.full((2, 3), index, dtype=np.int64) for index in range(6)]
    joined = concat_views_height(views)
    assert joined.shape == (12, 3)
    for index in range(6):
        np.testing.assert_array_equal(joined[index * 2 : (index + 1) * 2], index)


def test_same_global_instance_id_forms_one_joint_mask():
    semantic = np.array([[3, 0], [3, 0]])
    instance_a = np.array([[7, 0], [0, 0]])
    instance_b = np.array([[0, 0], [7, 0]])
    joined_semantic = concat_views_height([semantic, semantic])
    joined_instance = concat_views_height([instance_a, instance_b])
    masks = build_instance_masks(joined_semantic, joined_instance, prediction=False)
    assert len(masks["masks"]) == 1
    assert int(masks["masks"][0].sum()) == 2


def test_background_excluded_from_miou_and_official_twenty_class_mean():
    prediction = np.array([[0, 3], [3, 0]])
    target = np.array([[0, 3], [4, 0]])
    per_class, mean = global_semantic_iou([prediction], [target])
    assert per_class[0] == pytest.approx(0.0)  # absent semantic id 1 is zero, not background
    # Only semantic ids 3 and 4 are present; background id 0 never contributes.
    # Official MeanIoU returns all 20 foreground class entries and evaluator.py
    # averages that complete vector, so the 0-valued absent classes remain in
    # the denominator.
    assert per_class[2] == pytest.approx(0.5)
    assert per_class[3] == pytest.approx(0.0)
    assert mean == pytest.approx(0.025)


def test_wall_floor_are_excluded_from_instance_map_inputs():
    semantic = np.array([[1, 2, 3], [3, 0, 4]])
    instance = np.array([[11, 12, 13], [13, 0, 14]])
    output = build_instance_masks(semantic, instance, prediction=False)
    assert output["labels"] == [2, 3]


def test_class_aware_prediction_metadata_is_explicit():
    semantic = np.full((2, 2), 3)
    instance = np.array([[1, 1], [2, 2]])
    with pytest.raises(ValueError, match="explicit class-aware"):
        build_instance_masks(semantic, instance, prediction=True)
    result = build_instance_masks(semantic, instance, prediction=True, labels={1: 2, 2: 2}, scores={1: 0.9, 2: 0.8})
    assert result["labels"] == [2, 2]
    assert result["scores"] == [0.9, 0.8]


def test_depth_scale_shift_matches_official_lstsq_definition():
    target = np.array([[1.0, 2.0], [0.0, 4.0]])
    prediction = (target - 0.5) / 2.0
    result = depth_metrics(prediction, target)
    assert result["scale"] == pytest.approx(2.0)
    assert result["shift"] == pytest.approx(0.5)
    assert result["absrel"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)


def test_current_j2_hard_gate_rejects_eight_view_substitution():
    report = validate_model_contract({"num_input_views": 8, "num_views": 15, "img_size": (256, 256), "semantic_class_count": 8})
    assert report["two_context_forward_supported"] is False
    assert report["semantic_20_class_supported"] is False
    assert "STRICT_SIU3R_INPUT_VIEW_PARITY: NO" in report["failure_messages"]


def test_adapter_has_no_model_training_or_gt_input_hook():
    source = inspect.getsource(__import__("tokengs.siu3r_adapter", fromlist=["build_input"]))
    assert "forward_ttt" not in source
    assert "torch.optim" not in source
    assert ".step(" not in source
    assert "oracle" not in source.lower()
    assert "p_u" not in source
    assert "target_rgb" not in source
    assert "target_depth" not in source
    assert "target_labels" not in source


def test_legacy_lsm_sources_are_unchanged():
    subprocess.run(["git", "diff", "--quiet", "--", "tokengs/utils/instance_ap.py", "scripts/eval_instance_ap_per_view_reval.py"], cwd=ROOT, check=True)


@pytest.mark.skipif(not _official_torch_runtime(), reason="dynamic metric test requires official torch 2.4.1/torchvision 0.19.x/torchmetrics 1.7.3 runtime")
def test_global_id_shuffle_does_not_improve_official_map_or_pq():
    torch = pytest.importorskip("torch")
    from torchmetrics.detection import MeanAveragePrecision, PanopticQuality

    target_sem = torch.tensor([[[3, 3, 0, 0], [3, 3, 0, 0]]])
    target_ins = torch.tensor([[[7, 7, 0, 0], [7, 7, 0, 0]]])
    good_pred = {"masks": torch.tensor([[[1, 1, 0, 0]]], dtype=torch.bool), "labels": torch.tensor([2]), "scores": torch.tensor([0.9])}
    gt = {"masks": torch.tensor([[[1, 1, 0, 0]]], dtype=torch.bool), "labels": torch.tensor([2])}
    metric = MeanAveragePrecision(iou_type="segm", class_metrics=True, sync_on_compute=False)
    metric.update([good_pred], [gt])
    good_map = float(metric.compute()["map"])
    bad_pred = {"masks": torch.tensor([[[0, 0, 1, 1]]], dtype=torch.bool), "labels": torch.tensor([2]), "scores": torch.tensor([0.9])}
    metric = MeanAveragePrecision(iou_type="segm", class_metrics=True, sync_on_compute=False)
    metric.update([bad_pred], [gt])
    bad_map = float(metric.compute()["map"])
    assert bad_map <= good_map

    pq_good = PanopticQuality(things=list(range(3, 21)), stuffs=[1, 2], return_per_class=True, allow_unknown_preds_category=True, sync_on_compute=False)
    pq_bad = PanopticQuality(things=list(range(3, 21)), stuffs=[1, 2], return_per_class=True, allow_unknown_preds_category=True, sync_on_compute=False)
    pred_good = torch.stack([target_sem[0], target_ins[0]], dim=-1).unsqueeze(0)
    pred_bad = torch.stack([target_sem[0], torch.tensor([[[8, 8, 0, 0], [8, 8, 0, 0]]])[0]], dim=-1).unsqueeze(0)
    truth = torch.stack([target_sem[0], target_ins[0]], dim=-1).unsqueeze(0)
    pq_good.update(pred_good, truth)
    pq_bad.update(pred_bad, truth)
    assert float(pq_bad.compute().mean()) <= float(pq_good.compute().mean())
