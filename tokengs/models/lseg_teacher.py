"""Frozen LSeg teacher wrapper around C3G's LSegFeatureExtractor.

The extractor is held by a plain object (never registered as an nn.Module
submodule), so it is invisible to ``state_dict``, ``to()`` and
``accelerator.prepare``. It stays frozen in fp32 and is moved to the input
device lazily on first use.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# The timm ViT-L backbone is already cached locally; force offline so the
# LSeg model construction never tries to reach huggingface.co (the server
# has no internet access).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_C3G_ROOT = Path("/space0/mawb/C3G")
_DEFAULT_LSEG_CKPT = Path(
    "/space0/mawb/tokengs/checkpoints/demo_e200.ckpt"
)


class LSegTeacher:
    def __init__(self, checkpoint: str | Path = _DEFAULT_LSEG_CKPT):
        ckpt = Path(checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"LSeg checkpoint is unavailable: {ckpt}"
            )
        if str(_C3G_ROOT) not in sys.path:
            sys.path.insert(0, str(_C3G_ROOT))
        from src.model.lseg import LSegFeatureExtractor

        self.extractor = LSegFeatureExtractor.from_pretrained(
            str(ckpt), half_res=True
        )
        self.extractor.eval()
        for parameter in self.extractor.parameters():
            parameter.requires_grad_(False)
        self._device = None

    def _to_device(self, device: torch.device) -> None:
        if self._device != device:
            self.extractor = self.extractor.to(device)
            self._device = device

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Images [B,3,H,W] in [0,1] -> LSeg features [B,512,H/2,W/2]."""
        self._to_device(images.device)
        x = images.float().clamp(0.0, 1.0) * 2.0 - 1.0
        with torch.autocast(
            device_type=x.device.type, enabled=False
        ):
            return self.extractor.extract_features(x).float()
