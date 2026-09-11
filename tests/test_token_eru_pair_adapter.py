import torch

from tokengs.models.token_eru import ZeroInitPairAdapter


def test_pair_adapter_shape_and_zero_initial_output():
    torch.manual_seed(7)
    module = ZeroInitPairAdapter(16, 5)
    source = torch.randn(2, 9, 16)
    output = module(source)
    assert output.shape == source.shape
    assert torch.equal(output, torch.zeros_like(output))


def test_pair_adapter_can_receive_gradient():
    module = ZeroInitPairAdapter(8, 4)
    source = torch.randn(1, 3, 8, requires_grad=True)
    module(source).square().sum().backward()
    assert module.up.weight.grad is not None
    assert torch.isfinite(module.up.weight.grad).all()
