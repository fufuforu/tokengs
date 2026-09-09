import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tokengs.models.globalsplat_instance_v2.scene_instance_loss import (
    collect_scene_instances, scene_global_hungarian_instance_loss,
)


class SceneLossTest(unittest.TestCase):
    def test_scene_hungarian_once_and_cross_view_mapping(self):
        labels = torch.zeros(1, 7, 8, 8, dtype=torch.long)
        labels[:, :, :4, :4] = 7
        labels[:, :, 4:, 4:] = 11
        probs = torch.full((1, 101, 7, 1, 8, 8), 0.01)
        probs[:, 0, :, 0, :4, :4] = 0.9
        probs[:, 1, :, 0, 4:, 4:] = 0.9
        probs[:, 100] = 0.01
        probs.requires_grad_()
        loss, stats, assignments = scene_global_hungarian_instance_loss(probs, labels, min_visible_pixels=4)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(int(stats["gsi_v2_hungarian_calls"]), 1)
        self.assertEqual(len(assignments[0].query_to_instance_id), 2)
        loss.backward()
        self.assertTrue(torch.isfinite(probs.grad).all())

    def test_small_positive_is_not_object(self):
        labels = torch.zeros(7, 8, 8, dtype=torch.long)
        labels[0, 0, 0] = 9
        ids, *_ = collect_scene_instances(labels, min_visible_pixels=4)
        self.assertEqual(ids, [])


if __name__ == "__main__":
    unittest.main()
