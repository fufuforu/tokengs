import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tokengs.models.globalsplat_instance_v2.camera_adapter import (
    intrinsics_vec_to_matrix, make_official_context_input, make_official_target_meta,
)


class CameraAdapterTest(unittest.TestCase):
    def test_matrix_preserves_device_dtype_and_values(self):
        vector = torch.tensor([[[2.0, 3.0, 4.0, 5.0]]], dtype=torch.float64)
        matrix = intrinsics_vec_to_matrix(vector)
        self.assertEqual(matrix.dtype, vector.dtype)
        self.assertTrue(torch.equal(matrix[0, 0], torch.tensor([[2., 0., 4.], [0., 3., 5.], [0., 0., 1.]], dtype=vector.dtype)))

    def test_context_has_no_target_and_target_transposes_once(self):
        enc = SimpleNamespace(images_rgb_unnormalized=torch.zeros(1, 8, 3, 4, 5), intrinsics_input=torch.ones(1, 8, 4), cam_to_world_input=torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 8, 1, 1))
        cam = torch.arange(1 * 7 * 4 * 4, dtype=torch.float32).reshape(1, 7, 4, 4)
        dec = SimpleNamespace(cam_view=cam, intrinsics=torch.ones(1, 7, 4))
        context = make_official_context_input(SimpleNamespace(encoder=enc))
        target = make_official_target_meta(SimpleNamespace(decoder=dec), (4, 5))
        self.assertNotIn("target", context)
        self.assertTrue(torch.equal(target["extrinsic"], cam.transpose(-1, -2)))
        self.assertEqual(float(target["images"].sum()), 0.0)

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            intrinsics_vec_to_matrix(torch.tensor([[[0., 1., 0., 0.]]]))
        with self.assertRaises(ValueError):
            intrinsics_vec_to_matrix(torch.full((1, 8, 4), float("nan")))


if __name__ == "__main__":
    unittest.main()
