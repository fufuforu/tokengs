import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tokengs.models.globalsplat_instance_v2.tri_stream import TriStreamAdapter, grad_scale


class TriStreamTest(unittest.TestCase):
    def test_grad_scale(self):
        x = torch.randn(4, requires_grad=True)
        y = grad_scale(x, 0.25)
        self.assertTrue(torch.equal(x, y))
        y.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.full_like(x, 0.25)))

    def test_gate_identity_and_instance_gradient(self):
        torch.manual_seed(1)
        adapter = TriStreamAdapter(dim=64, hidden_dim=64)
        geo, tex, ins = [torch.randn(2, 16, 64, requires_grad=True) for _ in range(3)]
        out = adapter(geo, tex, ins, recon_to_instance_grad_scale=0.1, instance_to_reconstruction_gate=0.0)
        self.assertTrue(torch.equal(out[0], geo))
        self.assertTrue(torch.equal(out[1], tex))
        out[2].square().mean().backward()
        self.assertGreater(float(geo.grad.abs().sum()), 0.0)
        self.assertGreater(float(tex.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
