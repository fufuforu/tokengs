"""Small CPU/mock tests for the TA-RIU-v3 tensor contracts."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokengs.models.absolute_unit_decoder import AbsoluteUnitDecoder
from tokengs.models import ta_riu_v3 as v3


class MockDINO(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, views = images.shape[:2]
        return torch.zeros(b, views, 768, 18, 18)


def test_absolute_forward_equals_split() -> None:
    torch.manual_seed(1)
    decoder = AbsoluteUnitDecoder(
        token_dim=16, units_per_token=2, gaussians_per_unit=2, feat_dim=8
    )
    x = torch.randn(2, 4, 16)
    old_gs, old_q, old_c = decoder(x)
    q = decoder.form_units(x)
    new_gs, new_c = decoder.decode_units(q)
    assert torch.equal(old_q, q)
    assert torch.equal(old_c, new_c)
    assert torch.equal(old_gs, new_gs)


def test_flatten_unflatten_is_view_major() -> None:
    x = torch.arange(2 * 3 * 4 * 1).reshape(2, 3, 4, 1)
    adapter = v3.ReconstructionUnitAdapter(
        feature_dim=1, context_views=4, units_per_view=3
    )
    flat = adapter(x)
    assert flat[0, :, 0].tolist() == [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    assert torch.equal(v3.unflatten_reconstruction_units(flat, 4, 3), x)


def test_restore_encoder_memory_order() -> None:
    x = torch.arange(1 * 2 * 6 * 512).reshape(1, 2, 6, 512).float()
    out = v3.restore_encoder_memory(x, num_views=2, patches_per_view=3)
    expected = x.permute(0, 2, 1, 3).contiguous().reshape(1, 6, 1024)
    assert torch.equal(out, expected)


def test_query_embedding_and_former_shapes() -> None:
    q = v3.AlignedInstanceQueryEmbedding(context_views=2, units_per_view=3, dim=8)
    queries = q(4, torch.device("cpu"), torch.float32)
    assert queries.shape == (4, 6, 8)
    former = v3.AlignedInstanceUnitFormer(
        dim=8, num_heads=2, depth=2, context_views=2,
        units_per_view=3, dino_tokens_per_view=5,
    )
    rec = torch.randn(4, 6, 8)
    dino = torch.randn(4, 2, 5, 8)
    out = former(queries, rec, dino)
    assert out.shape == (4, 6, 8)
    assert torch.isfinite(out).all()
    for block in former.blocks:
        assert block.dino_cross_attn.batch_first
        assert block.dino_cross_attn.embed_dim == 8


def test_context_dino_is_external_and_context_only() -> None:
    original = v3.FrozenDINOv2Extractor
    try:
        v3.FrozenDINOv2Extractor = lambda *args, **kwargs: MockDINO()
        encoder = v3.ContextDINOEncoder("/mock/repo", "/mock/weight")
        out = encoder(torch.rand(1, 8, 3, 32, 32))
        assert out.shape == (1, 8, 324, 256)
        assert not any("dino_model" in key for key in encoder.state_dict())
        assert "target_images" not in inspect.signature(encoder.forward).parameters
        try:
            encoder(torch.rand(1, 8, 3, 32, 32), target_images=None)
        except TypeError:
            pass
        else:
            raise AssertionError("target_images must not be accepted")
    finally:
        v3.FrozenDINOv2Extractor = original


def test_pair_mixer_gate_and_upstream_gradient() -> None:
    torch.manual_seed(2)
    mixer = v3.PairUnitMixer(dim=8, hidden_dim=16)
    rec = torch.randn(2, 6, 8, requires_grad=True)
    ins = torch.randn(2, 6, 8, requires_grad=True)
    jr, ji, dr, di = mixer(rec, ins, gate=0.0)
    assert torch.equal(jr, rec) and torch.equal(ji, ins)
    assert torch.equal(dr, torch.zeros_like(dr))
    assert torch.equal(di, torch.zeros_like(di))
    try:
        mixer(rec, ins, gate=1.1)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range gate must fail")
    optimizer = torch.optim.Adam(mixer.parameters(), lr=1e-2)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        jr, ji, _, _ = mixer(rec, ins, gate=1.0)
        (jr.square().mean() + ji.square().mean()).backward()
        assert all(torch.isfinite(p.grad).all() for p in mixer.parameters() if p.grad is not None)
        if step == 0:
            assert mixer.rec_out.weight.grad is not None
            assert mixer.ins_out.weight.grad is not None
        optimizer.step()
    assert mixer.trunk[0].weight.grad is not None
    assert mixer.trunk[0].weight.grad.abs().sum() > 0


def test_shape_mismatch_and_no_large_attention() -> None:
    mixer = v3.PairUnitMixer(dim=8, hidden_dim=16)
    a = torch.randn(1, 6, 8)
    try:
        mixer(a, torch.randn(1, 5, 8), 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must fail")
    former = v3.AlignedInstanceUnitFormer(
        dim=8, num_heads=2, depth=1, context_views=2,
        units_per_view=3, dino_tokens_per_view=5,
    )
    assert all(block.dino_cross_attn.embed_dim == 8 for block in former.blocks)


def test_real_dino_cache_integration() -> None:
    """Opt-in integration test for the immutable local DINO cache."""
    if os.environ.get("TA_RIU_V2_RUN_DINO_INTEGRATION") != "1":
        print("test_real_dino_cache_integration: SKIP")
        return
    repo = "/space/mawb/.cache/torch/hub/facebookresearch_dinov2_main"
    weight = "/space/mawb/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"
    extractor = v3.FrozenDINOv2Extractor(repo, weight)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = torch.rand(1, 8, 3, 256, 256, device=device)
    raw = extractor(images)
    assert raw.shape == (1, 8, 768, 18, 18)
    patch_tokens = raw.permute(0, 1, 3, 4, 2).reshape(1, 8, 324, 768)
    assert patch_tokens.shape == (1, 8, 324, 768)
    assert torch.isfinite(patch_tokens).all()
    dino_model = extractor.__dict__["_dino_model"]
    assert sum(parameter.numel() for parameter in dino_model.parameters()) == 86580480
    assert extractor.last_strict_load == {"missing_keys": [], "unexpected_keys": []}
    assert not any("_dino_model" in key for key in extractor.state_dict())


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: PASS")


if __name__ == "__main__":
    main()
