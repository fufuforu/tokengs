import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tokengs.models.globalsplat_instance_v2.dependency import load_globalsplat_symbols, load_official_state_strict


class CheckpointTest(unittest.TestCase):
    def test_official_lineage_strict(self):
        root = "/space/mawb/globalsplat"
        symbols = load_globalsplat_symbols(root, "feb3fd7f7a6a8a9fafcb0ede5c314cd995cdf55b")
        model = symbols.GlobalSplat(sh_degree=3, static_only=True, use_camera_diff_as_input=False,
            patch_size=8, latent_rep_token_amount=2048, dim_latents=512, dim_rays=256,
            dim_rgb_feat=512, rounds=4, slot_calib_layers_per_round=2, num_heads=8, M_max=16)
        report = load_official_state_strict(model, "/space/mawb/globalsplat/ckpts/globalsplat-re10k-16k-noopacity.ckpt", "7069e1e2c72c2d08ecdf68a544ea024d24bd33461c39fa389eb94bc7b34d047a")
        self.assertEqual(report.loaded_tensor_count, 454)
        self.assertFalse(any(key.startswith("tokengs") or key.startswith("ta_riu") for key in model.state_dict()))


if __name__ == "__main__":
    unittest.main()
