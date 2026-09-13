from pathlib import Path

import torch
from torch import nn

from tokengs.models.token_eru.dino_metric import HistoricalDINOUnitEncoder


class MockExtractor(nn.Module):
    def forward(self, images):
        b, v = images.shape[:2]
        return torch.ones(b, v, 768, 18, 18, device=images.device)


def test_dino_encoder_constructor_is_explicit_and_mockable(tmp_path):
    repo = Path(tmp_path) / "repo"
    repo.mkdir()
    (repo / "hubconf.py").write_text("def dinov2_vitb14(pretrained=False): pass\n")
    weight = Path(tmp_path) / "weights.pth"
    torch.save({}, weight)
    encoder = HistoricalDINOUnitEncoder(str(repo), str(weight))
    encoder.dino_extractor = MockExtractor()
    assert encoder.dino_extractor is not None
    assert all("_dino_model" not in key for key in encoder.state_dict())


def test_target_view_tensor_is_not_part_of_dino_interface():
    import inspect

    signature = inspect.signature(HistoricalDINOUnitEncoder.forward)
    assert list(signature.parameters) == [
        "self",
        "context_images",
        "historical_alignment_inputs",
    ]
