import unittest

import torch

from tokengs.models.token_eru.unit_3d_anchor import Unit3DAnchor


class Unit3DAnchorCheckpointTest(unittest.TestCase):
    def test_state_dict_contains_anchor_only_and_roundtrips(self):
        source = Unit3DAnchor()
        restored = Unit3DAnchor()
        restored.load_state_dict(source.state_dict(), strict=True)
        self.assertEqual(set(source.state_dict()), set(restored.state_dict()))
        for key in source.state_dict():
            self.assertTrue(torch.equal(source.state_dict()[key], restored.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
