import torch
from torch import nn

from tokengs.models.token_eru import TokenGSEarlyDualStreamDecoder


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, gs_tokens, keys, values):
        del keys, values
        return self.linear(gs_tokens)


def test_save_reload_new_modules_strictly_equal():
    source = nn.ModuleList([_Block(4), _Block(4)])
    first = TokenGSEarlyDualStreamDecoder(
        source, decoder_dim=4, num_blocks=2, adapter_bottleneck_dim=2
    )
    first.initialize_understanding_from_reconstruction()
    state = first.state_dict()
    second = TokenGSEarlyDualStreamDecoder(
        source, decoder_dim=4, num_blocks=2, adapter_bottleneck_dim=2
    )
    result = second.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    for key, value in state.items():
        assert torch.equal(value, second.state_dict()[key])
    assert all("reconstruction_decoder" not in key for key in state)
