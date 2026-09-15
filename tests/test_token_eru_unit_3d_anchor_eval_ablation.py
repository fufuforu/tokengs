import unittest

import torch

from tokengs.models.token_eru.unit_3d_anchor import Unit3DAnchor


class Unit3DAnchorEvalAblationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.anchor = Unit3DAnchor()
        self.units = torch.randn(1, 4, 8, 256)
        self.means = torch.randn(1, 4 * 8 * 8, 3)
        self.opacity = torch.rand(1, 4 * 8 * 8, 1)
        self.anchor.eval()

    def test_full_matches_default_and_is_not_persistent(self):
        self.anchor.set_eval_ablation("full", permutation_seed=42)
        full = self.anchor(self.units, self.means, self.opacity)
        self.anchor.set_eval_ablation("full", permutation_seed=99)
        default = self.anchor(self.units, self.means, self.opacity)
        self.assertTrue(torch.equal(full.anchored_units, default.anchored_units))
        self.assertNotIn("_eval_ablation_mode", self.anchor.state_dict())
        self.assertNotIn("_eval_ablation_permutation_seed", self.anchor.state_dict())

    def test_off_returns_original_units(self):
        self.anchor.set_eval_ablation("off")
        output = self.anchor(self.units, self.means, self.opacity)
        self.assertTrue(torch.equal(output.anchored_units, self.units))

    def test_shuffle_is_deterministic_and_preserves_center_values(self):
        self.anchor.set_eval_ablation("full")
        original = self.anchor(self.units, self.means, self.opacity)
        self.anchor.set_eval_ablation("shuffle", permutation_seed=42)
        shuffled_a = self.anchor(self.units, self.means, self.opacity)
        self.anchor.set_eval_ablation("shuffle", permutation_seed=42)
        shuffled_b = self.anchor(self.units, self.means, self.opacity)
        self.assertTrue(torch.equal(shuffled_a.unit_centers_normalized, shuffled_b.unit_centers_normalized))
        self.assertTrue(torch.equal(
            torch.sort(original.unit_centers_normalized.reshape(-1, 3), dim=0).values,
            torch.sort(shuffled_a.unit_centers_normalized.reshape(-1, 3), dim=0).values,
        ))
        self.anchor.set_eval_ablation("shuffle", permutation_seed=43)
        shuffled_c = self.anchor(self.units, self.means, self.opacity)
        self.assertFalse(torch.equal(shuffled_a.unit_centers_normalized, shuffled_c.unit_centers_normalized))

    def test_zero_is_after_normalization(self):
        self.anchor.set_eval_ablation("zero")
        output = self.anchor(self.units, self.means, self.opacity)
        self.assertTrue(torch.equal(output.unit_centers_normalized, torch.zeros_like(output.unit_centers_normalized)))

    def test_non_full_mode_is_forbidden_in_training(self):
        self.anchor.train()
        with self.assertRaises(RuntimeError):
            self.anchor.set_eval_ablation("off")
        self.anchor.eval()


if __name__ == "__main__":
    unittest.main()
