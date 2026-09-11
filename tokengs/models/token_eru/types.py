from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenERUOutput:
    reconstruction_tokens: torch.Tensor
    understanding_tokens: torch.Tensor
    reconstruction_units: torch.Tensor
    understanding_units: torch.Tensor
    base_instance_logits: torch.Tensor
    final_instance_logits: torch.Tensor
    reconstruction_to_understanding_gate: float
    understanding_to_reconstruction_gate: float
