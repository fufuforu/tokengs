import unittest

import torch

from tokengs.models.token_eru.unit_3d_anchor import Unit3DAnchor


class Unit3DAnchorAlignmentTest(unittest.TestCase):
    def test_contiguous_child_blocks_are_unit_aligned(self):
        means = torch.arange(1 * 1024 * 8 * 8 * 3, dtype=torch.float32).reshape(1, 65536, 3)
        opacity = torch.ones(1, 65536, 1)
        anchor = Unit3DAnchor()
        output = anchor(torch.zeros(1, 1024, 8, 256), means, opacity)
        expected = means.reshape(1, 1024, 8, 8, 3).mean(dim=-2)
        self.assertTrue(torch.equal(output.unit_centers_world, expected))

    def test_anchor_has_only_position_path_parameters(self):
        anchor = Unit3DAnchor()
        names = [name for name, _ in anchor.named_parameters()]
        self.assertEqual(names, [
            "position_mlp.0.weight",
            "position_mlp.0.bias",
            "position_mlp.2.weight",
            "position_mlp.2.bias",
        ])
        self.assertNotIn("position_encoding.frequencies", dict(anchor.named_parameters()))


if __name__ == "__main__":
    unittest.main()
