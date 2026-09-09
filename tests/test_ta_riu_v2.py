import torch
from torch import nn

from tokengs.models.ta_riu_v2 import GeometryAlignedDINOUnitEncoder


class MockExtractor(nn.Module):
    def forward(self, images):
        b = images.shape[0]
        return torch.ones(b, 8, 768, 18, 18, device=images.device)


def _inputs():
    q = torch.randn(1, 2, 2, 4)
    xyz = torch.zeros(1, 8, 14)
    xyz[..., 2] = 2.0
    xyz[..., 3] = 0.5
    xyz[..., 4:7] = 0.01
    xyz[..., 7] = 1.0
    xyz[..., 11:] = 0.5
    images = torch.rand(1, 8, 3, 32, 32)
    c2w = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 8, 1, 1)
    intr = torch.tensor([10.0, 10.0, 16.0, 16.0]).reshape(1, 1, 4).repeat(1, 8, 1)
    return q, xyz, images, c2w, intr


def test_ta_riu_v2_gate_zero_is_exact_identity_and_has_no_dino_state():
    model = GeometryAlignedDINOUnitEncoder(
        unit_dim=4, num_tokens=2, units_per_token=2, gaussians_per_unit=2,
        dino_extractor=MockExtractor(),
    )
    q, xyz, images, c2w, intr = _inputs()
    out = model(q, xyz, images, c2w, intr, (32, 32), gate=0.0)
    assert torch.equal(out["z_inst"], q)
    assert not any("dino_extractor" in key or "_dino_model" in key for key in model.state_dict())


def test_ta_riu_v2_output_shapes_and_finite():
    model = GeometryAlignedDINOUnitEncoder(
        unit_dim=4, num_tokens=2, units_per_token=2, gaussians_per_unit=2,
        dino_extractor=MockExtractor(),
    )
    q, xyz, images, c2w, intr = _inputs()
    out = model(q, xyz, images, c2w, intr, (32, 32), gate=1.0)
    assert out["unit_embedding"].shape == (1, 4, 64)
    assert out["dino_unit"].shape == (1, 4, 768)
    assert all(torch.isfinite(value).all() for value in out.values())
