import torch

from tokengs.models.token_eru.early_query_codecoder import EarlyObjectQueryAdapter


def test_gate_open_gradients_reach_both_adapter_directions():
    torch.manual_seed(3)
    adapter = EarlyObjectQueryAdapter()
    u = torch.randn(1, 1024, 1024, requires_grad=True)
    q = torch.randn(1, 100, 256, requires_grad=True)
    out = adapter(u, q, 1.0)
    (out.understanding_hidden.square().mean() + out.query_state.square().mean()).backward()
    assert adapter.query_output.weight.grad is not None
    assert adapter.understanding_output.weight.grad is not None
    assert torch.isfinite(adapter.query_output.weight.grad).all()
    assert torch.isfinite(adapter.understanding_output.weight.grad).all()


def test_all_four_output_projections_are_zero_initialized():
    adapter = EarlyObjectQueryAdapter()
    for module in (adapter.query_output, adapter.understanding_output, adapter.query_ffn[-1], adapter.understanding_ffn[-1]):
        assert torch.equal(module.weight, torch.zeros_like(module.weight))
        assert torch.equal(module.bias, torch.zeros_like(module.bias))
