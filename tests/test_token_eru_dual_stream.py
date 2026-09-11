import copy

import torch
from torch import nn

from tokengs.models.token_eru import TokenGSEarlyDualStreamDecoder


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, gs_tokens, keys, values):
        del keys, values
        return torch.tanh(self.proj(gs_tokens))


class _Memory:
    def __init__(self, dim):
        self.keys = torch.zeros(1, 1, 1, dim)
        self.values = torch.zeros(1, 1, 1, dim)


def test_understanding_copy_is_strict_and_not_shared():
    blocks = nn.ModuleList([_Block(8), _Block(8)])
    decoder = TokenGSEarlyDualStreamDecoder(
        blocks, decoder_dim=8, num_blocks=2, adapter_bottleneck_dim=3
    )
    decoder.initialize_understanding_from_reconstruction()
    for source, target in zip(blocks, decoder.understanding_decoder_blocks):
        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters()
        ):
            assert torch.equal(source_parameter, target_parameter)
            assert source_parameter is not target_parameter
            assert source_parameter.data_ptr() != target_parameter.data_ptr()


def test_step_zero_stream_identity_and_gate_schedule():
    blocks = nn.ModuleList([_Block(8), _Block(8)])
    decoder = TokenGSEarlyDualStreamDecoder(
        blocks, decoder_dim=8, num_blocks=2, adapter_bottleneck_dim=3
    )
    decoder.initialize_understanding_from_reconstruction()
    decoder.set_gates(
        reconstruction_to_understanding=0.0,
        understanding_to_reconstruction=0.0,
    )
    query = torch.randn(2, 5, 8)
    reconstruction, understanding = decoder(query, _Memory(8))
    assert torch.equal(reconstruction, understanding)

    decoder.set_gates(
        reconstruction_to_understanding=1.0,
        understanding_to_reconstruction=0.1,
    )
    r2, u2 = decoder(query, _Memory(8))
    assert torch.isfinite(r2).all() and torch.isfinite(u2).all()
