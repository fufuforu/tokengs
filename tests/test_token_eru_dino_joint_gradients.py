import unittest

import torch
from torch import nn

from tokengs.models.token_eru.dino_metric import DINOUnitFusion, MetricEmbeddingHead
from tokengs.models.semantic_tokengs_v6 import SemanticTokenGSv6


class JointFormationGradientTest(unittest.TestCase):
    def test_metric_gradient_reaches_understanding_and_pair_path(self):
        r_units = torch.randn(1, 2, 8, 4, requires_grad=True)
        u_units = torch.randn(1, 2, 8, 4, requires_grad=True)
        adapter = nn.Linear(4, 4)
        fusion = DINOUnitFusion(4, 4)
        head = MetricEmbeddingHead(4, 4, 2)
        dino = torch.randn_like(u_units)
        fused = fusion(u_units, dino, 1.0)
        # The adapter is the explicit U->R coupling edge in the production
        # graph; this compact test checks that it remains differentiable.
        coupled = r_units + adapter(fused)
        loss = head(coupled).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(u_units.grad)
        self.assertGreater(float(u_units.grad.abs().sum()), 0.0)
        self.assertIsNotNone(r_units.grad)
        self.assertGreater(float(r_units.grad.abs().sum()), 0.0)
        self.assertGreater(
            sum(float(p.grad.abs().sum()) for p in adapter.parameters() if p.grad is not None),
            0.0,
        )

    def test_gate_zero_is_exact_input(self):
        units = torch.randn(1, 1024, 8, 256)
        evidence = torch.randn_like(units)
        fusion = DINOUnitFusion()
        result = fusion(units, evidence, 0.0)
        self.assertTrue(torch.equal(result, units))

    def test_gate_schedule_joint_formation(self):
        class Opt:
            token_eru_dino_metric_joint_formation = True
            token_eru_dino_gate_start_step = 0
            token_eru_dino_gate_end_step = 50
            token_eru_dino_metric_loss_weight = 1.0

        self.assertEqual(SemanticTokenGSv6.token_eru_dino_schedule(0, Opt()), (0.0, 0.0))
        self.assertEqual(SemanticTokenGSv6.token_eru_dino_schedule(25, Opt()), (0.5, 0.5))
        self.assertEqual(SemanticTokenGSv6.token_eru_dino_schedule(50, Opt()), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
