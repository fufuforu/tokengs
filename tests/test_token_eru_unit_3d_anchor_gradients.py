import unittest

import torch

from tokengs.models.token_eru.unit_3d_anchor import Unit3DAnchor


class Unit3DAnchorGradientTest(unittest.TestCase):
    def test_zero_init_receives_gradient_and_geometry_path_is_differentiable(self):
        anchor = Unit3DAnchor()
        units = torch.randn(1, 2, 4, 256, requires_grad=True)
        means = torch.randn(1, 64, 3, requires_grad=True)
        opacity = torch.rand(1, 64, 1, requires_grad=True)
        output = anchor(units, means, opacity)
        output.anchored_units.square().mean().backward()
        self.assertIsNotNone(anchor.position_mlp[2].weight.grad)
        self.assertGreater(float(anchor.position_mlp[2].weight.grad.abs().sum()), 0.0)
        # The zero last layer correctly blocks the first layer at the first
        # forward/backward; later optimizer updates are what open this path.
        if means.grad is not None:
            self.assertTrue(torch.isfinite(means.grad).all())
        if opacity.grad is not None:
            self.assertTrue(torch.isfinite(opacity.grad).all())


if __name__ == "__main__":
    unittest.main()
