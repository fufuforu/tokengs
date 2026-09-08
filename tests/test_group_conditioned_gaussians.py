import unittest

import torch

from tokengs.models.instance_group_head import GroupConditionedGaussianAdapter


def test_group_conditioner_preserves_warm_start_at_initialization():
    torch.manual_seed(0)
    module = GroupConditionedGaussianAdapter(
        token_dim=32,
        num_groups=5,
        condition_dim=16,
        num_heads=4,
        num_layers=1,
    )
    hidden = torch.randn(2, 7, 32)
    proposal = torch.randn(2, 7 * 64, 14)
    conditioned, assignment, logits, queries, positions = module(
        hidden, proposal
    )

    assert conditioned.shape == hidden.shape
    assert assignment.shape == (2, 7, 6)
    assert logits.shape == (2, 7, 6)
    assert queries.shape == (2, 5, 16)
    assert positions.shape == (2, 7, 3)
    assert torch.allclose(conditioned, hidden, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        assignment.sum(dim=-1), torch.ones(2, 7), atol=1e-6, rtol=1e-6
    )


def test_group_conditioner_receives_gradient_through_residual():
    torch.manual_seed(1)
    module = GroupConditionedGaussianAdapter(
        token_dim=16,
        num_groups=4,
        condition_dim=8,
        num_heads=2,
        num_layers=1,
    )
    hidden = torch.randn(1, 3, 16, requires_grad=True)
    proposal = torch.randn(1, 3 * 64, 14)
    conditioned, _, _, _, _ = module(hidden, proposal)
    loss = conditioned.square().mean()
    loss.backward()

    assert module.residual[-1].weight.grad is not None
    assert module.residual[-1].weight.grad.abs().sum() > 0


def test_assignment_supervision_reaches_shared_group_queries():
    torch.manual_seed(2)
    module = GroupConditionedGaussianAdapter(
        token_dim=16,
        num_groups=4,
        condition_dim=8,
        num_heads=2,
        num_layers=1,
    )
    hidden = torch.randn(1, 6, 16)
    proposal = torch.randn(1, 6 * 64, 14)
    _, probabilities, _, _, _ = module(hidden, proposal)
    target_groups = torch.tensor([[0, 1, 2, 3, 0, 1]])
    loss = torch.nn.functional.nll_loss(
        probabilities[..., :4].clamp_min(1e-8).log().reshape(-1, 4),
        target_groups.reshape(-1),
    )
    loss.backward()

    assert module.group_tokens.grad is not None
    assert module.group_tokens.grad.abs().sum() > 0
    assert module.position_in[0].weight.grad is not None
    assert module.position_in[0].weight.grad.abs().sum() > 0


def test_proposal_positions_change_group_assignment():
    torch.manual_seed(3)
    module = GroupConditionedGaussianAdapter(
        token_dim=16,
        num_groups=4,
        condition_dim=8,
        num_heads=2,
        num_layers=1,
    ).eval()
    hidden = torch.randn(1, 4, 16)
    proposal_a = torch.randn(1, 4 * 64, 14)
    proposal_b = proposal_a.clone()
    proposal_b[:, :64, :3] += 5.0

    with torch.no_grad():
        _, probabilities_a, _, _, _ = module(hidden, proposal_a)
        _, probabilities_b, _, _, _ = module(hidden, proposal_b)

    assert not torch.allclose(probabilities_a, probabilities_b)


def test_per_gaussian_conditioner_returns_local_assignments_and_opacity_delta():
    torch.manual_seed(4)
    module = GroupConditionedGaussianAdapter(
        token_dim=16,
        num_groups=4,
        condition_dim=8,
        num_heads=2,
        num_layers=1,
        per_gaussian_assignment=True,
    )
    hidden = torch.randn(1, 3, 16)
    proposal = torch.randn(1, 3 * 64, 14)
    conditioned, token_probs, _, _, _ = module(hidden, proposal)

    assert conditioned.shape == hidden.shape
    assert token_probs.shape == (1, 3, 5)
    assert module.last_gaussian_probabilities.shape == (1, 192, 5)
    assert module.last_gaussian_logits.shape == (1, 192, 5)
    assert module.last_gaussian_opacity_delta.shape == (1, 192, 1)
    assert torch.allclose(
        module.last_gaussian_probabilities.sum(dim=-1),
        torch.ones(1, 192),
        atol=1e-6,
        rtol=1e-6,
    )
    # The local residual is zero initialized, so warm-start geometry is exact.
    assert torch.allclose(
        module.last_gaussian_opacity_delta,
        torch.zeros_like(module.last_gaussian_opacity_delta),
        atol=1e-6,
        rtol=1e-6,
    )


class TestGroupConditionedGaussians(unittest.TestCase):
    def test_warm_start_identity(self):
        test_group_conditioner_preserves_warm_start_at_initialization()

    def test_residual_gradient(self):
        test_group_conditioner_receives_gradient_through_residual()

    def test_shared_query_gradient(self):
        test_assignment_supervision_reaches_shared_group_queries()

    def test_proposal_position_conditioning(self):
        test_proposal_positions_change_group_assignment()

    def test_per_gaussian_assignment(self):
        test_per_gaussian_conditioner_returns_local_assignments_and_opacity_delta()
