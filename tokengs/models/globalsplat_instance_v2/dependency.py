"""Lazy, fixed-commit access to the external official GlobalSplat code."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn


EXPECTED_CHECKPOINT_TENSORS = 454
EXPECTED_CHECKPOINT_NUMEL = 84_321_123


@dataclass(frozen=True)
class GlobalSplatSymbols:
    GlobalSplat: type[nn.Module]
    DualStreamSlotEncoder: type[nn.Module]
    StreamRound: type[nn.Module]
    PairAdapter: type[nn.Module]
    LayerScale: type[nn.Module]
    Gaussians: type
    render_static_batched: Callable
    rotation_6d_to_matrix: Callable
    matrix_to_quaternion: Callable
    frustum_soft_loss_w2c: Callable


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_globalsplat_checkout(repo_path: str | Path, expected_commit: str) -> Path:
    root = Path(repo_path).expanduser().resolve()
    if not (root / "globalsplat" / "model" / "globalsplat.py").is_file():
        raise RuntimeError(f"GlobalSplat source missing: {root / 'globalsplat/model/globalsplat.py'}")
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        raise RuntimeError(f"Cannot inspect GlobalSplat checkout {root}: {exc}") from exc
    if actual != expected_commit:
        raise RuntimeError(f"GlobalSplat commit mismatch at {root}: {actual} != {expected_commit}")
    return root


def _ensure_inside(path: str | None, root: Path, label: str) -> None:
    if path is None:
        raise RuntimeError(f"{label} has no __file__")
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} imported outside {root}: {resolved}") from exc


@lru_cache(maxsize=4)
def load_globalsplat_symbols(repo_path: str, expected_commit: str) -> GlobalSplatSymbols:
    root = verify_globalsplat_checkout(repo_path, expected_commit)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        gs = importlib.import_module("globalsplat.model.globalsplat")
        dual = importlib.import_module("globalsplat.model.encoder.dual_stream")
        types = importlib.import_module("globalsplat.model.types")
        rendering = importlib.import_module("globalsplat.model.rendering")
        geometry = importlib.import_module("globalsplat.misc.geometry")
        frustum = importlib.import_module("globalsplat.loss.frustum_loss")
        symbols = GlobalSplatSymbols(
            GlobalSplat=gs.GlobalSplat,
            DualStreamSlotEncoder=dual.DualStreamSlotEncoder,
            StreamRound=dual._StreamRound,
            PairAdapter=dual.PairAdapter,
            LayerScale=dual.LayerScale,
            Gaussians=types.Gaussians,
            render_static_batched=rendering.render_static_batched,
            rotation_6d_to_matrix=geometry.rotation_6d_to_matrix,
            matrix_to_quaternion=geometry.matrix_to_quaternion,
            frustum_soft_loss_w2c=frustum.frustum_soft_loss_w2c,
        )
        modules = (gs, dual, types, rendering, geometry, frustum)
        for module in modules:
            _ensure_inside(getattr(module, "__file__", None), root, module.__name__)
        return symbols
    except Exception as exc:
        raise RuntimeError(f"GlobalSplat import failed for fixed repo {root}: {exc}") from exc


@dataclass(frozen=True)
class OfficialCheckpointReport:
    path: str
    sha256: str
    checkpoint_tensor_count: int
    checkpoint_state_numel: int
    loaded_tensor_count: int
    loaded_state_numel: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]


def extract_official_model_state(
    checkpoint_path: str | Path, expected_sha256: str
) -> tuple[dict[str, torch.Tensor], OfficialCheckpointReport]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Official checkpoint missing: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise RuntimeError(f"Official checkpoint SHA mismatch at {path}: {actual_sha} != {expected_sha256}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if set(checkpoint) != {"state_dict", "pytorch-lightning_version"}:
        raise RuntimeError(f"Official checkpoint top-level fields are not exact: {sorted(checkpoint)}")
    source = checkpoint["state_dict"]
    if not isinstance(source, dict) or not source or any(not k.startswith("model.") for k in source):
        raise RuntimeError("Official checkpoint state_dict keys must all start with model.")
    state = {key[len("model."):]: value for key, value in source.items()}
    count = len(state)
    numel = sum(value.numel() for value in state.values() if torch.is_tensor(value))
    if count != EXPECTED_CHECKPOINT_TENSORS or numel != EXPECTED_CHECKPOINT_NUMEL:
        raise RuntimeError(f"Official checkpoint size mismatch: {count} tensors/{numel} numel")
    report = OfficialCheckpointReport(
        path=str(path), sha256=actual_sha, checkpoint_tensor_count=count,
        checkpoint_state_numel=numel, loaded_tensor_count=0, loaded_state_numel=0,
        missing_keys=(), unexpected_keys=(), shape_mismatches=(),
    )
    return state, report


def load_official_state_strict(
    model: nn.Module, checkpoint_path: str | Path, expected_sha256: str
) -> OfficialCheckpointReport:
    state, base = extract_official_model_state(checkpoint_path, expected_sha256)
    model_state = model.state_dict()
    shape_mismatches = tuple(
        key for key, value in state.items()
        if key in model_state and model_state[key].shape != value.shape
    )
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys or shape_mismatches:
        raise RuntimeError(
            f"Official strict restore failed: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys} shape={shape_mismatches}"
        )
    if any(not torch.isfinite(value).all() for value in model.parameters()):
        raise RuntimeError("Official restore contains non-finite model parameters")
    return OfficialCheckpointReport(
        **{**base.__dict__, "loaded_tensor_count": len(state), "loaded_state_numel": base.checkpoint_state_numel}
    )
