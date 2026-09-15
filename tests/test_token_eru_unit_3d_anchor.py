import unittest

import torch

from tokengs.models.token_eru.unit_3d_anchor import (
    FixedFourierPositionEncoding,
    Unit3DAnchor,
    compute_opacity_weighted_unit_centers,
    normalize_unit_centers,
    reshape_child_gaussian_attributes,
)


class Unit3DAnchorTest(unittest.TestCase):
    def test_child_layout_and_opacity_activation(self):
        means = torch.arange(2 * 64 * 3, dtype=torch.float32).reshape(2, 64, 3)
        opacity = torch.linspace(0.0, 1.0, 2 * 64).reshape(2, 64)
        child_means, child_opacity = reshape_child_gaussian_attributes(
            means, opacity, token_count=2, units_per_token=4, children_per_unit=8
        )
        self.assertEqual(tuple(child_means.shape), (2, 2, 4, 8, 3))
        self.assertEqual(tuple(child_opacity.shape), (2, 2, 4, 8, 1))
        self.assertTrue(torch.equal(child_means.reshape_as(means), means))
        self.assertTrue(torch.equal(child_opacity.reshape_as(opacity), opacity))

    def test_weighted_center_and_fallback(self):
        means = torch.tensor([[[[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]]])
        opacity = torch.tensor([[[[[0.25], [0.75]]]]])
        centers, mass, fallback = compute_opacity_weighted_unit_centers(means, opacity)
        self.assertAlmostEqual(float(centers[0, 0, 0, 0]), 2.5)
        self.assertAlmostEqual(float(mass[0, 0, 0, 0]), 1.0)
        self.assertFalse(bool(fallback.any()))
        zero = torch.zeros_like(opacity)
        centers, _, fallback = compute_opacity_weighted_unit_centers(means, zero)
        self.assertTrue(bool(fallback.all()))
        self.assertTrue(torch.equal(centers, means.mean(dim=-2)))

    def test_normalization_is_finite_and_statistics_detached(self):
        centers = torch.randn(2, 2, 4, 3, requires_grad=True)
        normalized, scene_center, scene_scale = normalize_unit_centers(centers)
        self.assertEqual(tuple(normalized.shape), tuple(centers.shape))
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertFalse(scene_center.requires_grad)
        self.assertFalse(scene_scale.requires_grad)
        self.assertGreater(float(normalized.std()), 0.0)

    def test_fourier_dimension_and_frequency_buffer(self):
        encoding = FixedFourierPositionEncoding(6, True)
        self.assertEqual(encoding.output_dim, 39)
        self.assertEqual(sum(parameter.numel() for parameter in encoding.parameters()), 0)
        self.assertEqual(tuple(encoding(torch.zeros(1, 2, 3)).shape), (1, 2, 39))

    def test_zero_initialized_anchor_is_identity(self):
        module = Unit3DAnchor()
        units = torch.randn(1, 2, 4, 256)
        means = torch.randn(1, 64, 3)
        opacity = torch.rand(1, 64, 1)
        output = module(units, means, opacity)
        self.assertEqual(tuple(output.unit_centers_world.shape), (1, 2, 4, 3))
        self.assertEqual(tuple(output.position_features.shape), (1, 2, 4, 39))
        self.assertEqual(tuple(output.anchored_units.shape), tuple(units.shape))
        self.assertEqual(float(output.anchor_delta.abs().max()), 0.0)
        self.assertTrue(torch.equal(output.anchored_units, units))

    def test_invalid_logit_like_opacity_is_rejected(self):
        means = torch.zeros(1, 64, 3)
        with self.assertRaises(ValueError):
            reshape_child_gaussian_attributes(means, torch.full((1, 64), 2.0))


if __name__ == "__main__":
    unittest.main()
