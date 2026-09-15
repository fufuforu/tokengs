import unittest
from types import SimpleNamespace

import torch

from tokengs.models.token_eru.query_metric_coupling import QueryMetricCoupling
from tokengs.models.semantic_tokengs_v6 import SemanticTokenGSv6
from tokengs.options import config_defaults


class QueryMetricCouplingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.module = QueryMetricCoupling()
        self.embeddings = torch.randn(1, 1024, 8, 128)
        self.queries = torch.randn(1, 100, 256)
        self.base = torch.randn(1, 1024, 8, 101)

    def test_shapes_and_gate_zero_identity(self):
        out = self.module(self.embeddings, self.queries, self.base, gate=0.0)
        self.assertEqual(tuple(out.final_unit_logits.shape), (1, 1024, 8, 101))
        self.assertEqual(tuple(out.metric_group_logits.shape), (1, 1024, 8, 100))
        self.assertTrue(torch.equal(out.final_unit_logits, self.base))
        self.assertTrue(torch.all(out.residual_logits[..., -1] == 0))
        self.assertTrue(getattr(self.module.log_temperature, "_no_weight_decay", False))

    def test_centering_and_nonzero_gate(self):
        out = self.module(self.embeddings, self.queries, self.base, gate=0.25)
        self.assertEqual(tuple(out.query_metric_embeddings.shape), (1, 100, 128))
        self.assertLessEqual(
            float(out.centered_metric_group_logits.mean(dim=-1).abs().max()), 1e-6
        )
        self.assertGreater(float((out.final_unit_logits - self.base).abs().max()), 0.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.module(self.embeddings[:, :1], self.queries, self.base, gate=0.1)
        with self.assertRaises(ValueError):
            self.module(self.embeddings, self.queries[:, :99], self.base, gate=0.1)
        with self.assertRaises(ValueError):
            self.module(self.embeddings, self.queries, self.base[..., :100], gate=0.1)
        with self.assertRaises(ValueError):
            self.module(self.embeddings, self.queries, self.base, gate=0.3)
        bad = self.embeddings.clone()
        bad[0, 0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            self.module(bad, self.queries, self.base, gate=0.1)

    def test_instance_gradient_reaches_projection_and_temperature(self):
        self.module.zero_grad(set_to_none=True)
        out = self.module(self.embeddings, self.queries, self.base, gate=0.01)
        loss = out.final_unit_logits[..., :100].square().mean()
        loss.backward()
        self.assertIsNotNone(self.module.query_projection.weight.grad)
        self.assertIsNotNone(self.module.log_temperature.grad)
        self.assertGreater(float(self.module.query_projection.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(self.module.log_temperature.grad.abs().sum()), 0.0)

    def test_local_gate_schedule(self):
        opt = SimpleNamespace(
            token_eru_query_metric_gate_ramp_steps=25,
            token_eru_query_metric_max_gate=0.25,
        )
        self.assertEqual(SemanticTokenGSv6.token_eru_query_metric_gate(0, opt), 0.0)
        self.assertAlmostEqual(
            SemanticTokenGSv6.token_eru_query_metric_gate(1, opt), 0.01
        )
        self.assertAlmostEqual(
            SemanticTokenGSv6.token_eru_query_metric_gate(25, opt), 0.25
        )
        self.assertAlmostEqual(
            SemanticTokenGSv6.token_eru_query_metric_gate(100, opt), 0.25
        )

    def test_legacy_defaults_do_not_enable_qmc(self):
        control = config_defaults[
            "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_local250_control_short200_ddp8"
        ]
        treatment = config_defaults[
            "semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_local250_query_metric_v1_short200_ddp8"
        ]
        self.assertFalse(control.token_eru_query_metric_enabled)
        self.assertTrue(treatment.token_eru_query_metric_enabled)


if __name__ == "__main__":
    unittest.main()
