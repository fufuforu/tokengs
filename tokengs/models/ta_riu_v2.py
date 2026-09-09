"""TA-RIU-v2: geometry-aligned DINO evidence for shared instance units.

The DINO backbone is intentionally *not* an ``nn.Module`` child.  It is a
frozen, locally loaded inference service so its 86M parameters never enter a
TokenGS state dict or optimizer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from tokengs.models.instance_group_head import _project_dense_features
from tokengs.models.instance_group_loss import _project_gs_to_views


DINO_INPUT_HW = (252, 252)
DINO_PATCH_DIM = 768
DINO_PATCH_GRID = (18, 18)
DINO_EXPECTED_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
DINO_DEFAULT_REPO = "/space/mawb/.cache/torch/hub/facebookresearch_dinov2_main"
DINO_DEFAULT_WEIGHT = "/space/mawb/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenDINOv2Extractor(nn.Module):
    """Offline local-cache DINOv2 patch extractor."""

    def __init__(
        self,
        repo_path: str,
        weight_path: str,
        model_name: str = "dinov2_vitb14",
    ) -> None:
        # Do not register the external backbone.  ``nn.Module.__setattr__``
        # would otherwise put all DINO parameters into the parent module.
        super().__init__()
        self.repo_path = str(Path(repo_path).expanduser().resolve())
        self.weight_path = str(Path(weight_path).expanduser().resolve())
        self.model_name = str(model_name)
        repo = Path(self.repo_path)
        weight = Path(self.weight_path)
        if not repo.is_dir():
            raise RuntimeError(f"DINO repo_path must be a directory: {self.repo_path}")
        if not (repo / "hubconf.py").is_file():
            raise RuntimeError(f"DINO repo missing hubconf.py: {repo / 'hubconf.py'}")
        if not weight.is_file():
            raise RuntimeError(f"DINO weight_path must be a regular file: {self.weight_path}")
        self.__dict__["_dino_model"] = None
        self.__dict__["_dino_device"] = None
        self.last_strict_load = None
        self.last_patch_shape = None

    def _ensure_model(self, device: torch.device) -> nn.Module:
        model = self.__dict__.get("_dino_model")
        current = self.__dict__.get("_dino_device")
        if model is None:
            # This is deliberately the only supported loader path.
            model = torch.hub.load(
                self.repo_path,
                "dinov2_vitb14",
                source="local",
                pretrained=False,
            )
            state = torch.load(
                self.weight_path,
                map_location="cpu",
                weights_only=True,
            )
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            result = model.load_state_dict(state, strict=True)
            if result.missing_keys != [] or result.unexpected_keys != []:
                raise RuntimeError(
                    "DINO strict load mismatch: "
                    f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
                )
            self.last_strict_load = {
                "missing_keys": list(result.missing_keys),
                "unexpected_keys": list(result.unexpected_keys),
            }
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.__dict__["_dino_model"] = model
            current = None
        if current != device:
            model.to(device)
            self.__dict__["_dino_device"] = device
        model.eval()
        return model

    def forward(self, images_input: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(images_input) or images_input.ndim != 5:
            raise ValueError(
                f"DINO input must be [B,8,3,H,W], got {getattr(images_input, 'shape', None)}"
            )
        if images_input.shape[1] != 8 or images_input.shape[2] != 3:
            raise ValueError(f"DINO input must have V=8,C=3, got {tuple(images_input.shape)}")
        if not torch.isfinite(images_input).all():
            raise ValueError("DINO input contains non-finite values")
        if float(images_input.detach().amin()) < -1e-4 or float(images_input.detach().amax()) > 1.0001:
            raise ValueError("DINO input must be RGB in [0,1]")
        b, v = images_input.shape[:2]
        x = images_input.reshape(b * v, 3, *images_input.shape[-2:]).float()
        x = F.interpolate(x, size=DINO_INPUT_HW, mode="bilinear", align_corners=False)
        mean = x.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = x.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        x = (x - mean) / std
        model = self._ensure_model(x.device)
        model.eval()
        with torch.inference_mode():
            tokens = model.get_intermediate_layers(
                x, n=1, return_class_token=False
            )[0]
            if tokens.ndim != 3 or tokens.shape[1:] != (324, DINO_PATCH_DIM):
                raise RuntimeError(f"unexpected DINO patch output: {tuple(tokens.shape)}")
            tokens = F.normalize(tokens.float(), dim=-1)
        out = tokens.detach().reshape(b, v, 18, 18, DINO_PATCH_DIM).permute(0, 1, 4, 2, 3)
        self.last_patch_shape = list(out.shape)
        return out.detach()


class GeometryAlignedDINOUnitEncoder(nn.Module):
    """Fuse frozen-view DINO evidence into the existing absolute units."""

    def __init__(
        self,
        unit_dim: int = 256,
        dino_dim: int = 768,
        dino_proj_dim: int = 128,
        position_dim: int = 32,
        embedding_dim: int = 64,
        num_tokens: int = 1024,
        units_per_token: int = 8,
        gaussians_per_unit: int = 8,
        dino_extractor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.unit_dim = int(unit_dim)
        self.dino_dim = int(dino_dim)
        self.num_tokens = int(num_tokens)
        self.units_per_token = int(units_per_token)
        self.gaussians_per_unit = int(gaussians_per_unit)
        self.dino_extractor = dino_extractor or FrozenDINOv2Extractor(
            DINO_DEFAULT_REPO, DINO_DEFAULT_WEIGHT
        )
        self.dino_norm = nn.LayerNorm(self.dino_dim)
        self.dino_proj = nn.Sequential(
            nn.Linear(self.dino_dim, int(dino_proj_dim)), nn.GELU(),
            nn.Linear(int(dino_proj_dim), int(dino_proj_dim)),
        )
        self.position_proj = nn.Sequential(
            nn.Linear(3, int(position_dim)), nn.GELU(),
            nn.Linear(int(position_dim), int(position_dim)),
        )
        fusion_dim = self.unit_dim + int(dino_proj_dim) + int(position_dim) + 1
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.unit_dim), nn.GELU(), nn.Linear(self.unit_dim, self.unit_dim)
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(self.unit_dim), nn.Linear(self.unit_dim, 128), nn.GELU(),
            nn.Linear(128, int(embedding_dim)),
        )

    @property
    def unit_count(self) -> int:
        return self.num_tokens * self.units_per_token

    def forward(
        self,
        q_abs: torch.Tensor,
        base_gaussians: torch.Tensor,
        images_input: torch.Tensor,
        cam_to_world_input: torch.Tensor,
        intrinsics_input: torch.Tensor,
        image_hw: tuple[int, int],
        gate: float,
    ) -> dict[str, torch.Tensor]:
        expected_units = self.unit_count
        if q_abs.shape != (q_abs.shape[0], self.num_tokens, self.units_per_token, self.unit_dim):
            raise ValueError(f"q_abs shape must be [B,{self.num_tokens},{self.units_per_token},{self.unit_dim}], got {tuple(q_abs.shape)}")
        if base_gaussians.ndim != 3 or base_gaussians.shape[1] != expected_units * self.gaussians_per_unit or base_gaussians.shape[-1] != 14:
            raise ValueError(f"base_gaussians must be [B,{expected_units*self.gaussians_per_unit},14], got {tuple(base_gaussians.shape)}")
        if images_input.ndim != 5 or images_input.shape[1:3] != (8, 3):
            raise ValueError(f"images_input must be [B,8,3,H,W], got {tuple(images_input.shape)}")
        dino = self.dino_extractor(images_input)
        if dino.shape[1:] != (8, self.dino_dim, 18, 18):
            raise RuntimeError(f"DINO feature map must be [B,8,{self.dino_dim},18,18], got {tuple(dino.shape)}")
        h, w = int(image_hw[0]), int(image_hw[1])
        scale = dino.new_tensor([252.0 / w, 252.0 / h, 252.0 / w, 252.0 / h])
        scaled_intrinsics = intrinsics_input.float() * scale.view(1, 1, 4)
        fused, per_gs_has_source = _project_dense_features(
            xyz_world=base_gaussians[..., :3].detach(),
            features=dino,
            source_c2w=cam_to_world_input,
            source_intrinsics=scaled_intrinsics,
            image_hw=DINO_INPUT_HW,
        )
        fused = fused.reshape(-1, expected_units, self.gaussians_per_unit, self.dino_dim)
        valid = per_gs_has_source.reshape(-1, expected_units, self.gaussians_per_unit, 1).float()
        dino_unit = (fused * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
        unit_has_source = valid.any(dim=2).squeeze(-1)
        dino_unit = dino_unit * unit_has_source.unsqueeze(-1).float()
        unit_xyz = base_gaussians[..., :3].detach().reshape(-1, expected_units, self.gaussians_per_unit, 3).mean(dim=2)
        center = unit_xyz.mean(dim=1, keepdim=True)
        rms = (unit_xyz - center).square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-3)
        position_normalized = ((unit_xyz - center) / rms).reshape_as(q_abs[..., :3])
        q_flat = q_abs.reshape(-1, expected_units, self.unit_dim)
        fusion_input = torch.cat(
            [F.layer_norm(q_flat, (self.unit_dim,)), self.dino_proj(self.dino_norm(dino_unit)), self.position_proj(position_normalized.reshape(-1, expected_units, 3)), unit_has_source.unsqueeze(-1).float()],
            dim=-1,
        )
        delta = self.fusion(fusion_input).reshape_as(q_abs)
        z_inst = q_abs if float(gate) == 0.0 else q_abs + float(gate) * delta
        embedding = F.normalize(self.embedding_head(z_inst.reshape(-1, expected_units, self.unit_dim)), dim=-1)
        return {
            "z_inst": z_inst,
            "delta": delta,
            "dino_unit": dino_unit.reshape(-1, expected_units, self.dino_dim),
            "unit_embedding": embedding,
            "unit_xyz": unit_xyz,
            "position_normalized": position_normalized,
            "unit_has_source": unit_has_source,
            "per_gs_has_source": per_gs_has_source.reshape(-1, expected_units),
        }


@torch.no_grad()
def build_unit_soft_instance_targets(
    base_gaussians: torch.Tensor,
    data: dict,
    image_size: tuple[int, int],
    num_tokens: int = 1024,
    units_per_token: int = 8,
    gaussians_per_unit: int = 8,
    min_valid_votes: int = 8,
    min_foreground_fraction: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    views = torch.cat([data["cam_view_input"], data["cam_view"]], dim=1)
    intrinsics = torch.cat([data["intrinsics_input"], data["intrinsics"]], dim=1)
    labels = torch.cat([data["instance_label_input"], data["instance_label_output"]], dim=1).long()
    b, n, _ = base_gaussians.shape
    projected = [
        _project_gs_to_views(
            base_gaussians[bi, ..., :3], views[bi], intrinsics[bi], labels[bi], image_size
        )
        for bi in range(b)
    ]
    ids = torch.stack([item[0] for item in projected], dim=0)
    valid = torch.stack([item[1] for item in projected], dim=0)
    ids = ids.reshape(b, 15, num_tokens, units_per_token, gaussians_per_unit)
    valid = valid.reshape_as(ids)
    targets, masks = [], []
    stats = {"target_valid_unit_share": [], "target_mean_foreground_fraction": [], "target_instance_count": [], "target_mean_valid_votes": []}
    for bi in range(b):
        scene_ids = ids[bi]
        scene_valid = valid[bi]
        unique = torch.unique(scene_ids[scene_valid])
        unique = unique[(unique > 0) & (unique != 255)]
        instance_count = max(1, int(unique.numel()))
        if unique.numel():
            # Vectorized over all units; the previous per-unit Python loop
            # caused a device synchronization for every one of 8192 units.
            votes_by_instance = (
                scene_ids.unsqueeze(-1).eq(unique.view(1, 1, 1, 1, -1))
                & scene_valid.unsqueeze(-1)
            ).sum(dim=(0, 3)).float()  # [T,K,I]
            total_votes = scene_valid.sum(dim=(0, 3)).float()  # [T,K]
            foreground_votes = votes_by_instance.sum(dim=-1)
            foreground_fraction = foreground_votes / total_votes.clamp_min(1.0)
            target = (
                votes_by_instance / foreground_votes.unsqueeze(-1).clamp_min(1.0)
            ).reshape(num_tokens * units_per_token, instance_count)
            unit_valid = (
                (total_votes >= float(min_valid_votes))
                & (foreground_fraction >= float(min_foreground_fraction))
                & (foreground_votes > 0)
            ).reshape(-1)
            target = target * unit_valid.unsqueeze(-1).float()
            valid_fracs = foreground_fraction.reshape(-1)[unit_valid]
            valid_votes = total_votes.reshape(-1)[unit_valid]
        else:
            target = base_gaussians.new_zeros(
                (num_tokens * units_per_token, instance_count), dtype=torch.float32
            )
            unit_valid = torch.zeros(
                num_tokens * units_per_token,
                device=base_gaussians.device,
                dtype=torch.bool,
            )
            valid_fracs = target.new_zeros(0)
            valid_votes = target.new_zeros(0)
        targets.append(target); masks.append(unit_valid)
        stats["target_valid_unit_share"].append(float(unit_valid.float().mean()))
        stats["target_mean_foreground_fraction"].append(float(valid_fracs.mean()) if valid_fracs.numel() else 0.0)
        stats["target_instance_count"].append(int(unique.numel()))
        stats["target_mean_valid_votes"].append(float(valid_votes.mean()) if valid_votes.numel() else 0.0)
    target_width = max(x.shape[1] for x in targets)
    targets = [F.pad(x, (0, target_width - x.shape[1])) for x in targets]
    return torch.stack(targets), torch.stack(masks), {k: torch.tensor(v, device=base_gaussians.device) for k, v in stats.items()}


def soft_unit_info_nce(embeddings, targets_per_batch, unit_valid, temperature=.1, max_units=2048):
    losses, same, different, anchors, sampled = [], [], [], [], []
    for emb, target, valid in zip(embeddings, targets_per_batch, unit_valid):
        idx = torch.where(valid)[0]
        if idx.numel() > int(max_units):
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[: int(max_units)]]
        z = F.normalize(emb[idx].float(), dim=-1)
        p = F.normalize(target[idx].float(), dim=-1)
        sim = z @ z.T / float(temperature)
        weights = p @ p.T
        eye = torch.eye(idx.numel(), device=z.device, dtype=torch.bool)
        pos = weights.masked_fill(eye, 0.0)
        denom = sim.masked_fill(eye, float("-inf"))
        has = pos.sum(dim=1) > 0
        if has.any():
            logp = F.log_softmax(denom, dim=1)
            # The diagonal has zero positive weight and -inf log-probability;
            # avoid the IEEE 0 * (-inf) NaN before the weighted reduction.
            logp = logp.masked_fill(eye, 0.0)
            losses.append(-(pos[has] / pos[has].sum(dim=1, keepdim=True).clamp_min(1e-6) * logp[has]).sum(dim=1).mean())
            same.append((z @ z.T).masked_select(pos > 0).mean())
            different.append((z @ z.T).masked_select((pos == 0) & ~eye).mean() if ((pos == 0) & ~eye).any() else z.new_zeros(()))
            anchors.append(has.float().sum()); sampled.append(torch.tensor(float(idx.numel()), device=z.device))
    if losses:
        loss = torch.stack(losses).mean()
        stats = {"unit_info_nce_loss": loss.detach(), "unit_same_cosine": torch.stack(same).mean().detach(), "unit_different_cosine": torch.stack(different).mean().detach(), "unit_embedding_collapse": torch.stack(same).mean().detach(), "unit_valid_anchor_count": torch.stack(anchors).sum().detach(), "unit_sampled_count": torch.stack(sampled).sum().detach()}
        return loss, stats
    zero = embeddings.sum() * 0.0
    return zero, {"unit_info_nce_loss": zero.detach(), "unit_same_cosine": zero.detach(), "unit_different_cosine": zero.detach(), "unit_embedding_collapse": zero.detach(), "unit_valid_anchor_count": zero.detach(), "unit_sampled_count": zero.detach()}
