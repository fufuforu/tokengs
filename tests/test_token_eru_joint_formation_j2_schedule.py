import math
import unittest

from tokengs.models.token_eru.j2_schedule import joint_formation_j2_lr_scale


class StageJ2ScheduleTest(unittest.TestCase):
    def test_fixed_points(self):
        self.assertAlmostEqual(joint_formation_j2_lr_scale(1), 0.02)
        self.assertAlmostEqual(joint_formation_j2_lr_scale(50), 1.0)
        self.assertAlmostEqual(joint_formation_j2_lr_scale(710), 0.1)

    def test_cosine_is_finite(self):
        for step in (1, 25, 50, 51, 355, 710):
            self.assertTrue(math.isfinite(joint_formation_j2_lr_scale(step)))

    def test_invalid_inputs(self):
        for kwargs in (
            {"optimizer_step": 0},
            {"optimizer_step": 711},
            {"optimizer_step": 1, "warmup_steps": 0},
            {"optimizer_step": 1, "warmup_steps": 710},
            {"optimizer_step": 1, "min_ratio": -0.1},
            {"optimizer_step": 1, "min_ratio": 1.1},
        ):
            with self.assertRaises(ValueError):
                joint_formation_j2_lr_scale(**kwargs)


if __name__ == "__main__":
    unittest.main()

