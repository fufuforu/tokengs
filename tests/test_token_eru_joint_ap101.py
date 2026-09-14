import math
import unittest

from scripts.eval_token_eru_dino_joint_formation import ap101_upper_bound


class AP101BoundTest(unittest.TestCase):
    def test_ap101_upper_bound_half(self):
        self.assertAlmostEqual(ap101_upper_bound(0.5), 51.0 / 101.0)

    def test_ap101_upper_bound_one_sixth(self):
        self.assertAlmostEqual(ap101_upper_bound(1.0 / 6.0), 17.0 / 101.0)

    def test_ap101_upper_bound_one(self):
        self.assertEqual(ap101_upper_bound(1.0), 1.0)

    def test_joint_step100_target2_boundary_passes(self):
        ap50, recall = 0.504950, 0.5
        self.assertLessEqual(ap50, ap101_upper_bound(recall) + 1e-8)

    def test_joint_step100_target6_boundary_passes(self):
        # 0.168317 is the rounded display value; use the exact 17/101
        # boundary because the production tolerance is fixed at 1e-8.
        ap50, recall = 17.0 / 101.0, 1.0 / 6.0
        self.assertLessEqual(ap50, ap101_upper_bound(recall) + 1e-8)

    def test_ap_over_discrete_bound_fails(self):
        ap50 = ap101_upper_bound(0.5) + 1e-6
        self.assertGreater(ap50, ap101_upper_bound(0.5) + 1e-8)

    def test_invalid_recall_fails(self):
        for value in (math.nan, math.inf, -math.inf, -1e-3, 1.001):
            with self.assertRaises(ValueError):
                ap101_upper_bound(value)


if __name__ == "__main__":
    unittest.main()
