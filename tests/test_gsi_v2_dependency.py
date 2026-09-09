import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from tokengs.models.globalsplat_instance_v2.dependency import (
    extract_official_model_state, load_globalsplat_symbols, load_official_state_strict,
)


ROOT = Path("/space/mawb/globalsplat")
COMMIT = "feb3fd7f7a6a8a9fafcb0ede5c314cd995cdf55b"
CKPT = ROOT / "ckpts/globalsplat-re10k-16k-noopacity.ckpt"
SHA = "7069e1e2c72c2d08ecdf68a544ea024d24bd33461c39fa389eb94bc7b34d047a"


class DependencyTest(unittest.TestCase):
    def test_lazy_import_does_not_load_checkpoint(self):
        self.assertNotIn("globalsplat.model.globalsplat", sys.modules)
        import tokengs.models  # noqa: F401
        self.assertNotIn("globalsplat.model.globalsplat", sys.modules)

    def test_wrong_commit_rejected(self):
        with self.assertRaises(RuntimeError):
            load_globalsplat_symbols(str(ROOT), "0" * 40)

    def test_checkpoint_report(self):
        state, report = extract_official_model_state(CKPT, SHA)
        self.assertEqual(len(state), 454)
        self.assertEqual(report.checkpoint_state_numel, 84321123)

    def test_official_strict_restore(self):
        symbols = load_globalsplat_symbols(str(ROOT), COMMIT)
        model = symbols.GlobalSplat(sh_degree=3, static_only=True, use_camera_diff_as_input=False,
            patch_size=8, latent_rep_token_amount=2048, dim_latents=512, dim_rays=256,
            dim_rgb_feat=512, rounds=4, slot_calib_layers_per_round=2, num_heads=8, M_max=16)
        report = load_official_state_strict(model, CKPT, SHA)
        self.assertEqual((report.loaded_tensor_count, report.checkpoint_tensor_count), (454, 454))
        self.assertEqual(report.loaded_state_numel, 84321123)


if __name__ == "__main__":
    unittest.main()
