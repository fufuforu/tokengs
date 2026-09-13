import torch

from tokengs.models.token_eru.dino_metric import (
    DINOUnitFusion,
    MetricEmbeddingHead,
)
from tokengs.models.token_eru.historical_unit_infonce import (
    historical_soft_unit_infonce,
)


def test_metric_shapes_and_gate_zero_identity():
    units = torch.randn(2, 1024, 8, 256)
    evidence = torch.randn_like(units)
    fusion = DINOUnitFusion()
    fused = fusion(units, evidence, 0.0)
    assert fused is units
    embedding = MetricEmbeddingHead()(fused)
    assert embedding.shape == (2, 1024, 8, 128)
    assert torch.allclose(embedding.norm(dim=-1), torch.ones_like(embedding[..., 0]), atol=1e-5)


def test_metric_fusion_changes_with_nonzero_gate():
    units = torch.randn(1, 4, 8, 256)
    evidence = torch.randn_like(units)
    fusion = DINOUnitFusion()
    assert not torch.equal(fusion(units, evidence, 1.0), units)


def test_historical_soft_infonce_is_finite_and_differentiable():
    embeddings = torch.randn(1, 12, 128, requires_grad=True)
    targets = torch.zeros(1, 12, 3)
    targets[:, :4, 0] = 1
    targets[:, 4:8, 1] = 1
    targets[:, 8:, 2] = 1
    valid = torch.ones(1, 12, dtype=torch.bool)
    result = historical_soft_unit_infonce(
        embeddings, targets, valid, temperature=0.1
    )
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_metric_input_validation():
    head = MetricEmbeddingHead()
    try:
        head(torch.zeros(1, 10, 256))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid unit layout was accepted")
