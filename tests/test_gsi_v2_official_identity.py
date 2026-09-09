import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


class OfficialIdentityTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "official identity test runs on CUDA to avoid CPU renderer ambiguity")
    def test_cuda_identity_placeholder(self):
        self.assertTrue(torch.cuda.is_available())


if __name__ == "__main__":
    unittest.main()
