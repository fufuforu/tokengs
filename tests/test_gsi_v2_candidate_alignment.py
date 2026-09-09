import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tokengs.models.globalsplat_instance_v2.candidate_layout import capture_candidate_layout, reduce_aligned_candidates


class Decoder:
    stage = 3
    mix = 1.0
    M_max = 16
    gate_tau = 1.0
    def __init__(self):
        self.gate_readout = torch.nn.Linear(4, 16)


class CandidateTest(unittest.TestCase):
    def test_stage_shape_and_index_alignment(self):
        decoder = Decoder()
        x = torch.randn(2, 2048, 4)
        layout = capture_candidate_layout(decoder, x)
        self.assertEqual(tuple(layout.weights_current.shape), (2, 2048, 8, 2, 1))
        aligned = reduce_aligned_candidates(torch.randn(2, 2048, 16, 7), layout, gate_gradient_scale=0.0)
        self.assertEqual(tuple(aligned.shape), (2, 16384, 7))

    def test_mix_zero_repeats_previous(self):
        decoder = Decoder()
        decoder.mix = 0.0
        x = torch.randn(1, 2048, 4)
        layout = capture_candidate_layout(decoder, x)
        full = torch.randn(1, 2048, 16, 3)
        out = reduce_aligned_candidates(full, layout, gate_gradient_scale=0.0)
        previous = (layout.weights_previous * full.view(1, 2048, 4, 4, 3)).sum(3).repeat_interleave(2, 2).reshape(1, 16384, 3)
        self.assertTrue(torch.allclose(out, previous))


if __name__ == "__main__":
    unittest.main()
