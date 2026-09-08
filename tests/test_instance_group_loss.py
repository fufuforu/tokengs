import unittest

import torch

from tokengs.models.instance_group_loss import hungarian_instance_group_loss


def _probabilities(swapped_second_view: bool) -> torch.Tensor:
    labels = torch.tensor(
        [
            [[1, 1, 2, 2], [1, 1, 2, 2]],
            [[1, 1, 2, 2], [1, 1, 2, 2]],
        ]
    )
    views = []
    for view_index in range(2):
        first = (labels[view_index] == 1).float()
        second = (labels[view_index] == 2).float()
        if swapped_second_view and view_index == 1:
            first, second = second, first
        void = torch.full_like(first, 1e-3)
        view = torch.stack([first, second, void]).clamp(1e-3, 1.0 - 1e-3)
        views.append(view)
    return torch.stack(views, dim=1).unsqueeze(0).unsqueeze(3)


class TestInstanceGroupLoss(unittest.TestCase):
    def setUp(self):
        self.labels = torch.tensor(
            [
                [
                    [[1, 1, 2, 2], [1, 1, 2, 2]],
                    [[1, 1, 2, 2], [1, 1, 2, 2]],
                ]
            ]
        )

    def _loss(self, probabilities, scene_level_matching):
        loss, _ = hungarian_instance_group_loss(
            probabilities,
            self.labels,
            num_groups=2,
            min_instance_pixels=1,
            dice_weight=1.0,
            mask_weight=1.0,
            void_weight=0.0,
            unmatched_weight=0.0,
            ce_weight=1.0,
            scene_level_matching=scene_level_matching,
        )
        return loss

    def test_scene_matching_penalizes_cross_view_slot_swaps(self):
        consistent = self._loss(_probabilities(False), True)
        swapped = self._loss(_probabilities(True), True)

        self.assertLess(float(consistent), 0.02)
        self.assertGreater(float(swapped), float(consistent) + 1.0)

    def test_legacy_per_view_matching_accepts_slot_swaps(self):
        consistent = self._loss(_probabilities(False), False)
        swapped = self._loss(_probabilities(True), False)

        torch.testing.assert_close(consistent, swapped)

    def test_scene_matching_backpropagates(self):
        probabilities = _probabilities(False).requires_grad_(True)
        loss = self._loss(probabilities, True)
        loss.backward()

        self.assertIsNotNone(probabilities.grad)
        self.assertTrue(torch.isfinite(probabilities.grad).all())

    def test_scene_matching_supports_fifteen_views(self):
        probabilities = _probabilities(False).repeat(1, 1, 8, 1, 1, 1)
        probabilities = probabilities[:, :, :15].requires_grad_(True)
        labels = self.labels.repeat(1, 8, 1, 1)[:, :15]
        loss, stats = hungarian_instance_group_loss(
            probabilities,
            labels,
            num_groups=2,
            min_instance_pixels=1,
            void_weight=0.0,
            unmatched_weight=0.0,
            ce_weight=1.0,
            scene_level_matching=True,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(stats["instance_group_matched_count"]), 2.0)
        self.assertTrue(torch.isfinite(probabilities.grad).all())


if __name__ == "__main__":
    unittest.main()
