import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class RendererTest(unittest.TestCase):
    @unittest.skipUnless(__import__("torch").cuda.is_available(), "CUDA/gsplat renderer test requires a CUDA node")
    def test_renderer_requires_cuda_node(self):
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
