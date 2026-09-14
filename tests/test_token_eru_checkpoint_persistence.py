"""CPU tests for the ERU-DINO milestone persistence policy."""

from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from tokengs.train import (
    _checkpoint_space_guard,
    _token_eru_full_state_due,
    save_token_eru_intra_epoch_checkpoint_synchronized,
    save_token_eru_model_only_checkpoint_synchronized,
)


class _Accelerator:
    is_main_process = True
    process_index = 0
    num_processes = 1
    device = torch.device("cpu")

    @staticmethod
    def wait_for_everyone():
        return None

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def print(*args, **kwargs):
        del kwargs
        print(*args)


def _opt(path, full_steps=()):
    return SimpleNamespace(
        workspace=str(path),
        batch_size=1,
        resume="/protected/eru500/model_step_000500.safetensors",
        tsh_fork_continue_step=500,
        token_eru_enabled=True,
        token_eru_matching_mode="scene",
        abs_ckpt_full_state=True,
        abs_ckpt_full_state_steps=tuple(full_steps),
        token_eru_dino_metric_enabled=True,
        token_eru_dino_repo_path="/local/dino/repo",
        token_eru_dino_weight_path="/local/dino/weight.pth",
        token_eru_dino_metric_loss_weight=1.0,
        token_eru_dino_gate_start_step=500,
        token_eru_dino_gate_end_step=525,
        token_eru_dino_embedding_dim=128,
        token_eru_dino_cluster_eps=0.5,
    )


def test_full_state_step_selection():
    opt = SimpleNamespace(abs_ckpt_full_state=True, abs_ckpt_full_state_steps=(600, 700))
    assert not _token_eru_full_state_due(opt, 525)
    assert not _token_eru_full_state_due(opt, 550)
    assert _token_eru_full_state_due(opt, 600)
    assert not _token_eru_full_state_due(opt, 650)
    assert _token_eru_full_state_due(opt, 700)


def test_space_guard_rejects_insufficient_space(tmp_path, monkeypatch):
    class _Stats:
        f_bavail = 1
        f_frsize = 4096

    monkeypatch.setattr("tokengs.train.os.statvfs", lambda _: _Stats())
    try:
        _checkpoint_space_guard(str(tmp_path), _Accelerator(), 600, "full-state")
    except RuntimeError as exc:
        assert "requires at least" in str(exc)
    else:
        raise AssertionError("space guard accepted insufficient space")


def test_model_only_and_full_state_restore(tmp_path):
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    accelerator = _Accelerator()

    model_only_opt = _opt(tmp_path, full_steps=(600, 700))
    save_token_eru_model_only_checkpoint_synchronized(
        model_only_opt, accelerator, model, epoch=1, completed_step=525
    )
    model_path = tmp_path / "checkpoints/model_step_000525.safetensors"
    assert model_path.is_file()
    assert (tmp_path / "checkpoints/step_000525.complete").is_file()
    assert not (tmp_path / "checkpoints/optimizer_step_000525.pth").exists()
    restored = load_file(str(model_path), device="cpu")
    assert set(restored) == set(model.state_dict())

    full_opt = _opt(tmp_path, full_steps=(600, 700))
    save_token_eru_intra_epoch_checkpoint_synchronized(
        full_opt, accelerator, model, optimizer, scheduler, epoch=1, completed_step=600
    )
    full_dir = tmp_path / "checkpoints"
    assert (full_dir / "model_step_000600.safetensors").is_file()
    assert (full_dir / "optimizer_step_000600.pth").is_file()
    assert (full_dir / "scheduler_step_000600.pth").is_file()
    assert (full_dir / "rng_step_000600_rank00.pth").is_file()
    assert (full_dir / "step_000600.complete").is_file()
    assert not list(full_dir.glob("*.tmp.*"))


def test_stage_helper_does_not_copy_parent_state():
    source = Path(__file__).parents[1] / "scripts/stage_token_eru_dino_metric_short200_identity.py"
    text = source.read_text(encoding="utf-8")
    assert "shutil.copy2" not in text
    assert "parent_checkpoint.json" in text
    assert "step500_identity_verified_in_memory" in text
