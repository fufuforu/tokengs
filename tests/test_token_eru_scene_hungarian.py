from types import SimpleNamespace

import torch

from tokengs.models.token_eru.scene_hungarian_loss import (
    build_scene_gt_masks,
    compute_scene_pairwise_cost,
    scene_hungarian_instance_group_loss,
    solve_scene_hungarian,
)


def _cfg(**overrides):
    values = dict(
        instance_group_match_topk=1,
        lambda_instance_group_mask=1.0,
        lambda_instance_group_dice=1.0,
        lambda_instance_group_void=0.1,
        lambda_instance_group_unmatched=0.1,
        lambda_instance_group_ce=1.0,
        instance_group_usage_entropy=0.05,
        instance_group_min_instance_pixels=1,
        _scene_void_probability=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _two_object_maps(views=7):
    maps = torch.zeros(views, 4, 4, dtype=torch.long)
    maps[:, :2, :2] = 11
    maps[:, 2:, 2:] = 22
    return maps


def test_scene_gt_union_uses_original_instance_ids():
    ids, masks, visibility = build_scene_gt_masks(
        _two_object_maps(), min_visible_pixels=1
    )
    assert ids.tolist() == [11, 22]
    assert masks.shape == (7, 2, 4, 4)
    assert visibility.tolist() == [[True] * 7, [True] * 7]


def test_visibility_excludes_absent_views():
    maps = _two_object_maps()
    maps[:, 2:, 2:] = 0
    ids, _, visibility = build_scene_gt_masks(maps, min_visible_pixels=1)
    assert ids.tolist() == [11]
    assert visibility.shape == (1, 7)


def test_scene_cost_averages_only_visible_views():
    gt = torch.zeros(2, 1, 2, 2)
    gt[:, 0, 0, 0] = 1
    pred = gt.clone()
    pred[1, 0] = 0.01
    visibility = torch.tensor([[True, False]])
    cost = compute_scene_pairwise_cost(
        pred, gt, visibility, bce_weight=1.0, dice_weight=1.0, eps=1e-6
    )
    expected = compute_scene_pairwise_cost(
        pred[:1], gt[:1], visibility[:, :1], bce_weight=1.0,
        dice_weight=1.0, eps=1e-6
    )
    assert torch.allclose(cost, expected, atol=1e-6, rtol=0)


def test_hungarian_called_once_for_seven_views():
    cost = torch.zeros(2, 2)
    assignment = solve_scene_hungarian(
        cost, torch.tensor([11, 22]), torch.ones(2, 7, dtype=torch.bool)
    )
    assert assignment.query_indices.numel() == 2
    assert assignment.pairwise_cost.dtype == torch.float32


def test_same_assignment_reused_for_all_views():
    maps = _two_object_maps()
    pred = torch.full((1, 7, 2, 4, 4), 0.01)
    pred[:, :, 0, :2, :2] = 0.99
    pred[:, :, 1, 2:, 2:] = 0.99
    void = torch.full((1, 7, 4, 4), 0.01)
    cfg = _cfg(_scene_void_probability=void)
    _, stats = scene_hungarian_instance_group_loss(
        pred, maps.unsqueeze(0), existing_loss_config=cfg, return_diagnostics=True
    )
    assert stats["instance_group_hungarian_calls"] == 1
    assert stats["instance_group_same_assignment_all_views"] is True


def test_nonvisible_object_view_not_treated_as_empty_mask():
    maps = _two_object_maps()
    maps[1, 2:, 2:] = 0
    ids, _, visibility = build_scene_gt_masks(maps, min_visible_pixels=1)
    assert ids.tolist() == [11, 22]
    assert visibility[1, 1].item() is False


def test_query_and_gt_matching_are_one_to_one():
    assignment = solve_scene_hungarian(
        torch.tensor([[0.0, 2.0], [2.0, 0.0]]),
        torch.tensor([11, 22]),
        torch.ones(2, 7, dtype=torch.bool),
    )
    assert len(set(assignment.query_indices.tolist())) == 2
    assert len(set(assignment.gt_columns.tolist())) == 2


def test_empty_scene_is_finite():
    pred = torch.full((1, 7, 2, 4, 4), 0.5, requires_grad=True)
    labels = torch.zeros(1, 7, 4, 4, dtype=torch.long)
    cfg = _cfg(_scene_void_probability=torch.full((1, 7, 4, 4), 0.5))
    loss, stats = scene_hungarian_instance_group_loss(
        pred, labels, existing_loss_config=cfg, return_diagnostics=True
    )
    assert torch.isfinite(loss)
    assert stats["instance_group_hungarian_calls"] == 1


def test_single_object_single_view_visibility():
    maps = torch.zeros(1, 3, 3, dtype=torch.long)
    maps[0, 0, 0] = 7
    ids, masks, visibility = build_scene_gt_masks(maps, min_visible_pixels=1)
    assert ids.tolist() == [7]
    assert masks.shape == (1, 1, 3, 3)
    assert visibility.tolist() == [[True]]


def test_void_channel_excluded_from_hungarian():
    assignment = solve_scene_hungarian(
        torch.tensor([[0.0], [0.1]]), torch.tensor([11]), torch.ones(1, 7, dtype=torch.bool)
    )
    assert assignment.query_indices.tolist() == [0]


def test_per_view_legacy_path_unchanged():
    # A one-view scene uses exactly the same detached BCE+Dice assignment and
    # all legacy mask/CE/void/unmatched/entropy terms.
    from tokengs.models.instance_group_loss import hungarian_instance_group_loss

    maps = torch.zeros(1, 4, 4, dtype=torch.long)
    maps[:, :2, :2] = 11
    pred = torch.full((1, 1, 3, 4, 4), 0.01)
    pred[:, :, 0, :2, :2] = 0.99
    void = torch.full((1, 1, 4, 4), 0.01)
    cfg = _cfg(_scene_void_probability=void)
    scene_loss, _ = scene_hungarian_instance_group_loss(
        pred, maps.unsqueeze(0), existing_loss_config=cfg
    )
    rendered = torch.cat([pred, void.unsqueeze(2)], dim=2).permute(
        0, 2, 1, 3, 4
    ).unsqueeze(3)
    legacy_loss, _ = hungarian_instance_group_loss(
        rendered,
        maps.unsqueeze(0),
        num_groups=3,
        min_instance_pixels=1,
        dice_weight=1.0,
        mask_weight=1.0,
        void_weight=0.1,
        unmatched_weight=0.1,
        ce_weight=1.0,
        usage_entropy_weight=0.05,
    )
    assert torch.allclose(scene_loss, legacy_loss, atol=1e-6, rtol=0)


def test_scene_loss_backward_finite():
    labels = _two_object_maps().unsqueeze(0)
    pred = torch.full((1, 7, 3, 4, 4), 0.2, requires_grad=True)
    cfg = _cfg(_scene_void_probability=torch.full((1, 7, 4, 4), 0.2))
    loss, _ = scene_hungarian_instance_group_loss(pred, labels, existing_loss_config=cfg)
    loss.backward()
    assert torch.isfinite(loss)
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_scene_instance_ids_not_shared_across_batch():
    first = _two_object_maps()
    second = _two_object_maps().clone()
    second[second == 11] = 101
    second[second == 22] = 202
    pred = torch.full((2, 7, 3, 4, 4), 0.2)
    cfg = _cfg(_scene_void_probability=torch.full((2, 7, 4, 4), 0.2))
    loss, stats = scene_hungarian_instance_group_loss(
        pred, torch.stack([first, second]), existing_loss_config=cfg
    )
    assert torch.isfinite(loss)
    assert stats["instance_group_hungarian_calls"] == 2


def test_bf16_input_cost_computed_in_fp32():
    if not hasattr(torch, "bfloat16"):
        return
    gt = torch.zeros(2, 1, 2, 2)
    gt[:, 0, 0, 0] = 1
    pred = gt.to(torch.bfloat16)
    visibility = torch.ones(1, 2, dtype=torch.bool)
    cost = compute_scene_pairwise_cost(
        pred, gt, visibility, bce_weight=1.0, dice_weight=1.0, eps=1e-6
    )
    assert cost.dtype == torch.float32
    assert torch.isfinite(cost).all()
