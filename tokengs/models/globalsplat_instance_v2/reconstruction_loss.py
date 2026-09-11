from __future__ import annotations

from pathlib import Path

import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dependency import GlobalSplatSymbols


class ExplicitMatVGG19PerceptualLoss(nn.Module):
    def __init__(self, weight_path: str | Path) -> None:
        super().__init__()
        path = Path(weight_path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"VGG MatConvNet weight file missing: {path}")
        from torchvision.models import vgg19
        self.vgg = vgg19(weights=None)
        for index, layer in enumerate(self.vgg.features):
            if isinstance(layer, nn.MaxPool2d):
                self.vgg.features[index] = nn.AvgPool2d(kernel_size=2, stride=2)
        data = scipy.io.loadmat(path)
        layers = data["layers"][0]
        layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        filter_sizes = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        with torch.no_grad():
            for i, layer_index in enumerate(layer_indices):
                weights = torch.from_numpy(layers[layer_index][0][0][2][0][0]).permute(3, 2, 0, 1)
                biases = torch.from_numpy(layers[layer_index][0][0][2][0][1]).view(filter_sizes[i])
                self.vgg.features[layer_index].weight = nn.Parameter(weights, requires_grad=False)
                self.vgg.features[layer_index].bias = nn.Parameter(biases, requires_grad=False)
        self.blocks = nn.ModuleList([nn.Sequential(*list(self.vgg.features[a:b])) for a, b in ((0, 4), (4, 9), (9, 14), (14, 23), (23, 32))])
        self.register_buffer("mean", torch.tensor([123.6800, 116.7790, 103.9390]).view(1, 3, 1, 1), persistent=False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        return image * 255.0 - self.mean.to(device=image.device, dtype=image.dtype)

    def extract_target(self, target: torch.Tensor):
        target_p = self._preprocess(target)
        x = target_p
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return target_p, features

    def forward(self, pred: torch.Tensor, target: torch.Tensor, target_cache=None) -> torch.Tensor:
        target_p, target_features = self.extract_target(target) if target_cache is None else target_cache
        pred_p = self._preprocess(pred)
        x = pred_p
        pred_features = []
        for block in self.blocks:
            x = block(x)
            pred_features.append(x)
        errors = [F.l1_loss(pred_p, target_p), F.l1_loss(pred_features[0], target_features[0]) / 2.6,
                  F.l1_loss(pred_features[1], target_features[1]) / 4.8,
                  F.l1_loss(pred_features[2], target_features[2]) / 3.7,
                  F.l1_loss(pred_features[3], target_features[3]) / 5.6,
                  F.l1_loss(pred_features[4], target_features[4]) * 10.0 / 1.5]
        return sum(errors) / 255.0


class GSIReconstructionLoss(nn.Module):
    def __init__(self, *, symbols: GlobalSplatSymbols, mode: str, vgg_weight_path: str,
                 rgb_outer_weight: float = 2.0, mse_weight: float = 1.0,
                 perceptual_weight: float = 0.5, inview_weight: float = 1e-2) -> None:
        super().__init__()
        self.symbols = symbols
        self.mode = str(mode)
        self.rgb_outer_weight = float(rgb_outer_weight)
        self.mse_weight = float(mse_weight)
        self.perceptual_weight = float(perceptual_weight)
        self.inview_weight = float(inview_weight)
        if self.mode == "official_vgg":
            self.perceptual_loss = ExplicitMatVGG19PerceptualLoss(vgg_weight_path)
        elif self.mode == "mse_smoke":
            self.perceptual_loss = None
        else:
            raise ValueError(f"unknown reconstruction loss mode {mode}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.perceptual_loss is not None:
            self.perceptual_loss.eval()
        return self

    def forward(self, gaussians, rendered_rgb: torch.Tensor, target_rgb: torch.Tensor,
                context_K: torch.Tensor, context_w2c: torch.Tensor, *, target_cache=None):
        mse = F.mse_loss(rendered_rgb.float(), target_rgb.float())
        perc = rendered_rgb.new_zeros(())
        cache = target_cache
        if self.perceptual_loss is not None:
            if cache is None:
                cache = self.perceptual_loss.extract_target(target_rgb.flatten(0, 1))
            perc = self.perceptual_loss(rendered_rgb.flatten(0, 1), target_rgb.flatten(0, 1), target_cache=cache)
        render_loss = self.rgb_outer_weight * (self.mse_weight * mse + self.perceptual_weight * perc)
        inview = self.symbols.frustum_soft_loss_w2c(
            means=gaussians.means, intrinsics=context_K, extrinsics=context_w2c,
            H=rendered_rgb.shape[-2], W=rendered_rgb.shape[-1], max_depth=125.0,
        ) * self.inview_weight
        total = render_loss + inview + gaussians.reg
        return total, {"loss_rgb": render_loss.detach(), "loss_mse": mse.detach(), "loss_perceptual": perc.detach(), "loss_inview": inview.detach(), "loss_decoder_reg": gaussians.reg.detach()}, cache
