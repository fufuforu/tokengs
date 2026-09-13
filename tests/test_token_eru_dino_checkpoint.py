from pathlib import Path

import torch

from tokengs.models.token_eru.dino_metric import HistoricalDINOUnitEncoder


def test_external_dino_is_not_saved_in_parent_state_dict(tmp_path):
    repo = Path(tmp_path) / "repo"
    repo.mkdir()
    (repo / "hubconf.py").write_text("def dinov2_vitb14(pretrained=False): pass\n")
    weight = Path(tmp_path) / "weights.pth"
    torch.save({}, weight)
    module = HistoricalDINOUnitEncoder(str(repo), str(weight))
    keys = set(module.state_dict())
    assert not any(key.startswith("dino_extractor._dino_model") for key in keys)
    assert any(key.startswith("unit_projector.") for key in keys)
