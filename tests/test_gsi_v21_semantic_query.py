import dataclasses
import unittest

import torch

from tokengs.models.globalsplat_instance_v2.instance_head import SceneGlobalInstanceHead
from tokengs.models.globalsplat_instance_v2.model import render_query_semantic_probabilities
from tokengs.models.globalsplat_instance_v2.scene_instance_loss import (
    derive_scene_object_semantic_targets,
    scene_global_hungarian_instance_loss,
    semantic_class_matching_cost,
    semantic_query_classification_loss,
)
from tokengs.models.globalsplat_instance_v2.semantic_query_head import SemanticQueryHead
from tokengs.models.globalsplat_instance_v2.types import CandidateLayout, InstanceDecode


def _layout(batch: int = 1, points: int = 2048, candidates: int = 16) -> CandidateLayout:
    logits = torch.zeros(batch, points, candidates, 1)
    weights = torch.softmax(logits.view(batch, points, 1, candidates, 1), dim=3)
    return CandidateLayout(
        stage=0,
        mix=0.0,
        active_per_slot=1,
        gate_logits_full=logits,
        weights_current=weights,
        weights_previous=None,
    )


class InstanceDecodeQueryFeatureTest(unittest.TestCase):
    def test_dataclass_is_frozen_and_query_features_are_runtime_only(self):
        features = torch.randn(1, 100, 64, requires_grad=True)
        output = InstanceDecode(
            gaussian_embeddings=torch.randn(1, 2048, 64),
            gaussian_objectness_logits=torch.randn(1, 2048, 1),
            object_queries=features,
            assignment_logits=torch.randn(1, 2048, 101),
            assignment_probabilities=torch.randn(1, 2048, 101),
            query_features=features,
        )
        self.assertTrue(dataclasses.is_dataclass(output))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            output.query_features = torch.zeros_like(features)
        self.assertEqual(tuple(output.query_features.shape), (1, 100, 64))
        self.assertTrue(output.query_features.requires_grad)
        self.assertIs(output.query_features, output.object_queries)

    def test_forward_reuses_final_query_tensor_and_backpropagates(self):
        torch.manual_seed(3407)
        head = SceneGlobalInstanceHead(query_layers=1)
        scene_ins = torch.randn(1, 2048, 512, requires_grad=True)
        output = head(scene_ins, _layout(), gate_gradient_scale=0.0)

        self.assertEqual(tuple(output.query_features.shape), (1, 100, 64))
        self.assertTrue(output.query_features.requires_grad)
        self.assertIs(output.query_features, output.object_queries)
        query_loss = output.query_features.square().mean()
        query_loss.backward()
        self.assertIsNotNone(head.query_decoder.object_queries.grad)
        self.assertGreater(float(head.query_decoder.object_queries.grad.abs().sum()), 0.0)

    def test_forward_does_not_add_state_dict_keys_or_runtime_tensor(self):
        head = SceneGlobalInstanceHead(query_layers=1)
        keys_before = tuple(head.state_dict().keys())
        output = head(torch.randn(1, 2048, 512), _layout(), gate_gradient_scale=0.0)
        self.assertEqual(keys_before, tuple(head.state_dict().keys()))
        self.assertNotIn("query_features", head.state_dict())
        self.assertNotIn("query_features", head.state_dict().keys())
        self.assertIs(output.query_features, output.object_queries)

    def test_state_dict_round_trip_excludes_query_features(self):
        head = SceneGlobalInstanceHead(query_layers=1)
        state = head.state_dict()
        restored = SceneGlobalInstanceHead(query_layers=1)
        restored.load_state_dict(state, strict=True)
        self.assertNotIn("query_features", state)
        self.assertNotIn("query_features", restored.state_dict())


class SemanticQueryTest(unittest.TestCase):
    def test_head_shape_and_validation(self):
        head = SemanticQueryHead(64, 8)
        features = torch.randn(2, 100, 64)
        self.assertEqual(tuple(head(features).shape), (2, 100, 9))
        with self.assertRaises(ValueError):
            head(torch.randn(2, 99, 64))
        with self.assertRaises(ValueError):
            head(torch.randn(2, 100, 32))
        with self.assertRaises(FloatingPointError):
            head(torch.full((1, 100, 64), float("nan")))

    def test_head_initialization_preserves_cpu_rng(self):
        torch.manual_seed(1234)
        before = torch.random.get_rng_state()
        SemanticQueryHead(64, 8, init_seed=3407)
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_head_initialization_preserves_cuda_rng(self):
        torch.cuda.manual_seed_all(1234)
        before = torch.cuda.get_rng_state_all()
        SemanticQueryHead(64, 8, init_seed=3407)
        after = torch.cuda.get_rng_state_all()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))

    def test_targets_preserve_object_order_and_purity(self):
        instances = torch.tensor([
            [[10, 10, 0], [10, 0, 20]],
            [[10, 10, 0], [10, 0, 20]],
        ])
        semantics = torch.tensor([
            [[4, 4, 0], [4, 0, 4]],
            [[4, 4, 0], [4, 0, 5]],
        ])
        classes, valid, purity = derive_scene_object_semantic_targets(
            instances, semantics, torch.tensor([20, 10]), {0: 0, 4: 4, 5: 5},
            ignore_ids=(0, 255, -1), min_purity=0.95,
        )
        self.assertEqual(classes.tolist(), [4, 4])
        self.assertEqual(valid.tolist(), [False, True])
        self.assertAlmostEqual(float(purity[0]), 0.5)
        self.assertAlmostEqual(float(purity[1]), 1.0)

    def test_invalid_matching_columns_are_zero(self):
        logits = torch.randn(100, 9)
        cost = semantic_class_matching_cost(
            logits, torch.tensor([4, -1]), torch.tensor([True, False])
        )
        self.assertTrue(torch.equal(cost[:, 1], torch.zeros(100)))
        self.assertTrue(torch.isfinite(cost).all())

    def test_semantic_ce_valid_unmatched_and_invalid(self):
        logits = torch.zeros(3, 9, requires_grad=True)
        loss = semantic_query_classification_loss(
            logits,
            torch.tensor([0, 1]), torch.tensor([0, 1]),
            torch.tensor([4, -1]), torch.tensor([True, False]),
            no_object_index=8, eos_coef=0.1,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_rendered_semantic_probability_shape_and_normalization(self):
        assignment = torch.zeros(1, 101, 7, 1, 2, 2)
        assignment[:, 0] = 0.5
        assignment[:, 100] = 0.5
        logits = torch.zeros(1, 100, 9)
        probabilities = render_query_semantic_probabilities(assignment, logits)
        self.assertEqual(tuple(probabilities.shape), (1, 7, 9, 2, 2))
        torch.testing.assert_close(
            probabilities.sum(dim=2), torch.ones(1, 7, 2, 2), atol=1e-5, rtol=1e-5
        )

    def test_disabled_hungarian_path_is_unchanged_and_single_assignment(self):
        torch.manual_seed(3407)
        rendered = torch.softmax(torch.randn(1, 101, 7, 1, 8, 8), dim=1)
        labels = torch.zeros(1, 7, 8, 8, dtype=torch.long)
        labels[:, :, :4, :4] = 1
        first = scene_global_hungarian_instance_loss(rendered, labels, min_visible_pixels=1)
        second = scene_global_hungarian_instance_loss(rendered, labels, min_visible_pixels=1)
        torch.testing.assert_close(first[0], second[0])
        self.assertEqual(float(first[1]["gsi_v2_hungarian_calls"]), 1.0)

    def test_semantic_head_round_trip_has_only_parameters(self):
        head = SemanticQueryHead(64, 8)
        state = head.state_dict()
        restored = SemanticQueryHead(64, 8)
        restored.load_state_dict(state, strict=True)
        self.assertEqual(tuple(state.keys()), ("linear.weight", "linear.bias"))
        torch.testing.assert_close(head.linear.weight, restored.linear.weight)


if __name__ == "__main__":
    unittest.main()
