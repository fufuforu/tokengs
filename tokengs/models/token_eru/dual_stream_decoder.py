from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .pair_adapter import ZeroInitPairAdapter


class TokenGSEarlyDualStreamDecoder(nn.Module):
    """Run the existing TokenGS decoder and a strict, aligned copy in parallel.

    The reconstruction ModuleList is intentionally kept as a non-registered
    reference: the original checkpoint namespace remains under
    ``enc_dec_backbone.decoder_blocks``.  The copied understanding blocks and
    all adapters are registered here and are therefore saved in new ERU
    checkpoints.
    """

    def __init__(
        self,
        reconstruction_decoder: nn.Module,
        *,
        decoder_dim: int,
        num_blocks: int,
        adapter_bottleneck_dim: int = 128,
    ) -> None:
        super().__init__()
        if len(reconstruction_decoder) != int(num_blocks):
            raise ValueError(
                f"ERU decoder block count mismatch: expected {num_blocks}, "
                f"got {len(reconstruction_decoder)}"
            )
        self.decoder_dim = int(decoder_dim)
        self.num_blocks = int(num_blocks)
        # Do not register a second reference to reconstruction blocks: this
        # preserves the old checkpoint keys and avoids duplicate state_dict
        # namespaces.
        self.__dict__["_reconstruction_decoder"] = reconstruction_decoder
        self.understanding_decoder_blocks = nn.ModuleList(
            [copy.deepcopy(block) for block in reconstruction_decoder]
        )
        self.reconstruction_to_understanding = nn.ModuleList(
            [
                ZeroInitPairAdapter(self.decoder_dim, adapter_bottleneck_dim)
                for _ in range(self.num_blocks)
            ]
        )
        self.understanding_to_reconstruction = nn.ModuleList(
            [
                ZeroInitPairAdapter(self.decoder_dim, adapter_bottleneck_dim)
                for _ in range(self.num_blocks)
            ]
        )
        self.reconstruction_to_understanding_gate = 0.0
        self.understanding_to_reconstruction_gate = 0.0

    @property
    def reconstruction_decoder(self) -> nn.Module:
        return self.__dict__["_reconstruction_decoder"]

    def initialize_understanding_from_reconstruction(self) -> None:
        for index, (source, target) in enumerate(
            zip(self.reconstruction_decoder, self.understanding_decoder_blocks)
        ):
            result = target.load_state_dict(source.state_dict(), strict=True)
            if result.missing_keys or result.unexpected_keys:
                raise RuntimeError(
                    f"ERU strict decoder copy failed at block {index}: "
                    f"missing={result.missing_keys}, "
                    f"unexpected={result.unexpected_keys}"
                )

    def set_gates(
        self,
        *,
        reconstruction_to_understanding: float,
        understanding_to_reconstruction: float,
    ) -> None:
        values = (
            reconstruction_to_understanding,
            understanding_to_reconstruction,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("ERU gates must be finite")
        if any(float(value) < 0.0 for value in values):
            raise ValueError("ERU gates must be non-negative")
        if float(reconstruction_to_understanding) > 1.0:
            raise ValueError("reconstruction_to_understanding gate must be <= 1")
        if float(understanding_to_reconstruction) > 0.1:
            raise ValueError("understanding_to_reconstruction gate must be <= 0.1")
        self.reconstruction_to_understanding_gate = float(
            reconstruction_to_understanding
        )
        self.understanding_to_reconstruction_gate = float(
            understanding_to_reconstruction
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        encoder_memory: object,
        **existing_decoder_inputs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys = getattr(encoder_memory, "keys", None)
        values = getattr(encoder_memory, "values", None)
        if keys is None or values is None:
            if isinstance(encoder_memory, (tuple, list)) and len(encoder_memory) == 2:
                keys, values = encoder_memory
            else:
                raise TypeError("ERU encoder_memory must expose keys and values")

        r = query_tokens
        # clone preserves the query index layout and autograd connection while
        # ensuring the two streams have independent subsequent activations.
        u = query_tokens.clone()
        for r_block, u_block, r2u, u2r in zip(
            self.reconstruction_decoder,
            self.understanding_decoder_blocks,
            self.reconstruction_to_understanding,
            self.understanding_to_reconstruction,
        ):
            r_hat = r_block(
                gs_tokens=r,
                keys=keys,
                values=values,
                **existing_decoder_inputs,
            )
            u_hat = u_block(
                gs_tokens=u,
                keys=keys,
                values=values,
                **existing_decoder_inputs,
            )
            r = r_hat + self.understanding_to_reconstruction_gate * u2r(u_hat)
            u = u_hat + self.reconstruction_to_understanding_gate * r2u(r_hat)
        return r, u
