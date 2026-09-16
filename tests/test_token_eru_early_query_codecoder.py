import torch

from tokengs.models.token_eru.early_query_codecoder import EarlyObjectQueryAdapter


def test_shape_and_zero_gate_identity():
    torch.manual_seed(0)
    module = EarlyObjectQueryAdapter()
    u = torch.randn(2, 1024, 1024)
    q = torch.randn(2, 100, 256)
    out = module(u, q, 0.0)
    assert torch.equal(out.understanding_hidden, u)
    assert torch.equal(out.query_state, q)


def test_nonzero_gate_is_finite_and_changes_after_training_signal():
    torch.manual_seed(1)
    module = EarlyObjectQueryAdapter()
    u = torch.randn(1, 1024, 1024)
    q = torch.randn(1, 100, 256)
    out = module(u, q, 1.0)
    assert out.understanding_hidden.shape == u.shape
    assert out.query_state.shape == q.shape
    assert torch.isfinite(out.understanding_hidden).all()
    assert torch.isfinite(out.query_state).all()
    loss = out.query_state.square().mean() + out.understanding_hidden.square().mean()
    loss.backward()
    assert module.query_output.weight.grad is not None
    assert module.query_ffn[-1].weight.grad is not None


def test_invalid_shapes_and_gate_are_rejected():
    module = EarlyObjectQueryAdapter()
    with torch.no_grad():
        u = torch.zeros(1, 1024, 1024)
        q = torch.zeros(1, 100, 256)
    for bad_u in (torch.zeros(1, 1023, 1024), torch.zeros(1, 1024, 1023)):
        try:
            module(bad_u, q, 0.0)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid U shape was accepted")
    for bad_gate in (-0.1, 1.1, float("nan")):
        try:
            module(u, q, bad_gate)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid gate was accepted")


def test_eval_path_switches_query_and_u_writes_without_parameters():
    torch.manual_seed(11)
    module = EarlyObjectQueryAdapter()
    # Make the diagnostic branches observable without changing the production
    # zero-init contract: this is an isolated unit-test perturbation.
    with torch.no_grad():
        module.query_output.weight.fill_(0.01)
        module.understanding_output.weight.fill_(0.01)
        module.query_ffn[-1].weight.fill_(0.01)
        module.understanding_ffn[-1].weight.fill_(0.01)
    u = torch.randn(1, 1024, 1024)
    q = torch.randn(1, 100, 256)
    full = module(u, q, 1.0)
    no_u = module(u, q, 1.0, u_write=False)
    no_q = module(u, q, 1.0, query_update=False)
    assert torch.equal(full.query_state, no_u.query_state)
    assert torch.equal(no_u.understanding_hidden, u)
    assert torch.equal(no_q.query_state, q)
    assert not torch.equal(full.understanding_hidden, no_q.understanding_hidden)


def test_eval_controls_are_not_state_dict_entries():
    module = EarlyObjectQueryAdapter()
    keys_before = tuple(module.state_dict())
    module.eval()
    _ = module(
        torch.zeros(1, 1024, 1024),
        torch.zeros(1, 100, 256),
        0.0,
        query_update=False,
        u_write=False,
    )
    assert tuple(module.state_dict()) == keys_before
