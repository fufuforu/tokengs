import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file


class _NoDinoStateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.metric_head = nn.Linear(4, 2)
        # A production FrozenDINOv2Extractor keeps the external model in
        # __dict__, so this test models the required non-registered behavior.
        self.__dict__["_dino_model"] = nn.Linear(4, 4)


class JointFormationCheckpointTest(unittest.TestCase):
    def test_external_dino_is_not_saved(self):
        model = _NoDinoStateModel()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(
                {key: value.detach().cpu() for key, value in model.state_dict().items()},
                str(path),
            )
            state = load_file(str(path), device="cpu")
            self.assertTrue(all("dino_model" not in key for key in state))
            self.assertIn("metric_head.weight", state)

    def test_runtime_outputs_are_not_state_dict_entries(self):
        model = _NoDinoStateModel()
        model._runtime_unit_embeddings = torch.randn(1, 8192, 128)
        self.assertNotIn("_runtime_unit_embeddings", model.state_dict())


if __name__ == "__main__":
    unittest.main()
