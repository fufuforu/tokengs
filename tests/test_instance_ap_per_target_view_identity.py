"""Regression tests for image-local per-target-view AP bookkeeping."""

from __future__ import annotations

import numpy as np

from tokengs.utils.instance_ap import instance_ap, make_target_view_image_id


def _mask() -> np.ndarray:
    value = np.zeros((4, 4), dtype=bool)
    value[1:3, 1:3] = True
    return value


def test_same_view_perfect_match() -> None:
    result = instance_ap(
        [_mask()], [1.0], [_mask()], thresholds=(0.5,),
        pred_image_ids=["view0"], gt_image_ids=["view0"],
    )
    assert np.isclose(result["ap_50"], 1.0)


def test_cross_view_same_shape_must_not_match() -> None:
    result = instance_ap(
        [_mask()], [1.0], [_mask()], thresholds=(0.5,),
        pred_image_ids=["view0"], gt_image_ids=["view1"],
    )
    assert result["ap_50"] == 0.0


def test_legacy_shared_id_demonstrates_false_positive() -> None:
    legacy = instance_ap(
        [_mask()], [1.0], [_mask()], thresholds=(0.5,),
        pred_image_ids=["scene:b0"], gt_image_ids=["scene:b0"],
    )
    corrected = instance_ap(
        [_mask()], [1.0], [_mask()], thresholds=(0.5,),
        pred_image_ids=["scene:b0:v0"], gt_image_ids=["scene:b0:v1"],
    )
    assert np.isclose(legacy["ap_50"], 1.0)
    assert np.isclose(corrected["ap_50"], 0.0)


def test_repeated_instance_id_across_views_is_local_to_image() -> None:
    view0 = make_target_view_image_id(
        scene_id="scene", context_frame_ids=[0, 20],
        target_frame_ids=[10, 30], target_view_index=0,
    )
    view1 = make_target_view_image_id(
        scene_id="scene", context_frame_ids=[0, 20],
        target_frame_ids=[10, 30], target_view_index=1,
    )
    assert view0 != view1
    result = instance_ap(
        [_mask(), _mask()], [1.0, 0.9], [_mask(), _mask()], thresholds=(0.5,),
        pred_image_ids=[view0, view1], gt_image_ids=[view0, view1],
    )
    assert np.isclose(result["ap_50"], 1.0)


def test_single_view_equivalence() -> None:
    mask = _mask()
    legacy = instance_ap([mask], [0.8], [mask], thresholds=(0.5,))
    corrected = instance_ap(
        [mask], [0.8], [mask], thresholds=(0.5,),
        pred_image_ids=[make_target_view_image_id(
            scene_id="scene", context_frame_ids=[0],
            target_frame_ids=[10], target_view_index=0,
        )],
        gt_image_ids=[make_target_view_image_id(
            scene_id="scene", context_frame_ids=[0],
            target_frame_ids=[10], target_view_index=0,
        )],
    )
    assert corrected == legacy


def test_seven_view_ids_are_unique_and_shared_by_pred_gt() -> None:
    ids = [make_target_view_image_id(
        scene_id="scene", context_frame_ids=[0, 20, 40, 60, 80, 100, 120, 140],
        target_frame_ids=[10, 30, 50, 70, 90, 110, 130], target_view_index=view,
    ) for view in range(7)]
    assert len(set(ids)) == 7
    assert all(pred_id == gt_id for pred_id, gt_id in zip(ids, ids))


def test_multi_window_same_scene_ids_are_distinct() -> None:
    first = make_target_view_image_id(
        scene_id="scene", context_frame_ids=[0, 20],
        target_frame_ids=[10, 30], target_view_index=0,
    )
    second = make_target_view_image_id(
        scene_id="scene", context_frame_ids=[100, 120],
        target_frame_ids=[110, 130], target_view_index=0,
    )
    assert first != second


def test_invalid_target_view_is_rejected() -> None:
    try:
        make_target_view_image_id(
            scene_id="scene", context_frame_ids=[0],
            target_frame_ids=[10], target_view_index=1,
        )
    except ValueError:
        return
    raise AssertionError("out-of-range target view was accepted")
