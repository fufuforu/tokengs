import unittest

import torch
from torch import nn

from tokengs.models.token_eru.query_metric_coupling import QueryMetricCoupling


class QueryMetricGradientTest(unittest.TestCase):
    def test_gate_zero_has_no_qmc_gradient_and_gate_positive_does(self):
        module = QueryMetricCoupling()
        embeddings = torch.randn(1, 1024, 8, 128, requires_grad=True)
        queries = torch.randn(1, 100, 256)
        base = torch.randn(1, 1024, 8, 101, requires_grad=True)
        out = module(embeddings, queries, base, gate=0.0)
        out.final_unit_logits.sum().backward()
        self.assertIsNone(module.query_projection.weight.grad)
        module.zero_grad(set_to_none=True)
        out = module(embeddings, queries, base, gate=0.01)
        out.final_unit_logits[..., :100].sum().backward()
        self.assertIsNotNone(module.query_projection.weight.grad)

    def test_rgb_only_path_does_not_use_qmc(self):
        module = QueryMetricCoupling()
        rgb_only = nn.Parameter(torch.ones(()))
        (rgb_only * 2.0).backward()
        self.assertIsNone(module.query_projection.weight.grad)
        self.assertIsNone(module.log_temperature.grad)


if __name__ == "__main__":
    unittest.main()
