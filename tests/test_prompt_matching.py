import os
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from copy import deepcopy
from safetensors.torch import load_file

from tokengs.models.input_types import EncoderLatent, ModelInputDecoder
from tokengs.models.prompt_matching import (
    ConditionalQueryDecoder,
    DEFAULT_CLIP_MODEL_PATH,
    PromptEncoder,
    PromptConditionedTokenMatcher,
    PromptGaussianDecoder,
)
from tokengs.models.enc_dec import DecoderBlock
from tokengs.models.prompt_tokengs import PromptTokenGS
from tokengs.models.tokengs import TokenGS
from tokengs.options import Options


class _DummyDecoderLayer(nn.Module):
    def forward(self, gs_tokens, keys, values):
        del keys, values
        return gs_tokens + 0.125


class _DummyGaussianHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1024, 14)

    def forward(self, tokens):
        return self.projection(tokens)


class _DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([_DummyDecoderLayer()])


class _FakePromptEncoder:
    output_dim = 4

    def encode_text(self, text_query):
        output = torch.zeros(len(text_query), 1, self.output_dim)
        output[..., 0] = 1
        return output

    def encode_image(self, query_image, query_mask):
        del query_mask
        output = torch.zeros(query_image.shape[0], 1, self.output_dim)
        output[..., 1] = 1
        return output


class TestManifestPromptDispatch(unittest.TestCase):
    def test_semantic_fork_hidden_keeps_gradient_only_when_enabled(self):
        hidden = torch.randn(1, 4, 8, requires_grad=True)
        frozen_host = SimpleNamespace(semantic_last_decoder=None)
        adapted_host = SimpleNamespace(semantic_last_decoder=object())
        frozen = PromptTokenGS._matching_hidden(frozen_host, hidden)
        adapted = PromptTokenGS._matching_hidden(adapted_host, hidden)
        self.assertFalse(frozen.requires_grad)
        self.assertTrue(adapted.requires_grad)
        adapted.square().mean().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_semantic_fork_trains_only_cross_attention_and_matcher(self):
        model = PromptTokenGS.__new__(PromptTokenGS)
        nn.Module.__init__(model)
        matcher = nn.Module()
        matcher.prompt_encoder = nn.Linear(8, 8)
        matcher.matching_decoder = nn.Linear(32, 1)
        model.prompt_matcher = matcher
        model.semantic_last_decoder = DecoderBlock(
            dim=32,
            num_heads=4,
            mlp_ratio=2.0,
            qkv_bias=True,
            ffn_bias=True,
            qk_norm=True,
            init_values=1e-3,
        )
        model._freeze_for_prompt_training()

        hidden = model.semantic_last_decoder(
            gs_tokens=torch.randn(1, 9, 32),
            keys=torch.randn(1, 4, 7, 8),
            values=torch.randn(1, 4, 7, 8),
        )
        matcher.matching_decoder(hidden).square().mean().backward()
        groups = model.prompt_trainable_groups()
        self.assertTrue(groups["last_cross_attention"])
        self.assertTrue(
            any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in groups["last_cross_attention"]
            )
        )
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in groups["matching_decoder"]
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.semantic_last_decoder.parameters()
                if not parameter.requires_grad
            )
        )
        matcher.trainable_state_dict = lambda: OrderedDict(
            (
                ("matching_decoder.weight", matcher.matching_decoder.weight),
                ("matching_decoder.bias", matcher.matching_decoder.bias),
            )
        )
        state = PromptTokenGS.state_dict(model)
        self.assertTrue(any(key.startswith("semantic_last_decoder.") for key in state))
        self.assertTrue(all("prompt_encoder" not in key for key in state))
        self.assertTrue(all("semantic_last_decoder.mlp" not in key for key in state))
        incompatibilities = PromptTokenGS.load_state_dict(model, state, strict=True)
        self.assertEqual(incompatibilities.missing_keys, [])
        self.assertEqual(incompatibilities.unexpected_keys, [])

    def test_text_image_and_mixed_rows_use_manifest_modes(self):
        host = SimpleNamespace(
            opt=SimpleNamespace(prompt_mode="manifest", prompt_mixed_text_weight=0.5),
            prompt_matcher=SimpleNamespace(prompt_encoder=_FakePromptEncoder()),
        )
        data = {
            "prompt_mode": ["text_only", "image_only", "text_image_mixed"],
            "positive_text_prompt": ["wall", "chair", "table"],
            "query_image": torch.rand(3, 3, 8, 8),
            "query_mask": torch.ones(3, 1, 8, 8),
            "has_image_query": torch.tensor([False, True, True]),
        }
        output = PromptTokenGS._encode_prompt_batch(host, data)
        self.assertEqual(output.shape, (3, 1, 4))
        torch.testing.assert_close(output[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(output[1, 0], torch.tensor([0.0, 1.0, 0.0, 0.0]))
        expected_mixed = torch.tensor([2**-0.5, 2**-0.5, 0.0, 0.0])
        torch.testing.assert_close(output[2, 0], expected_mixed)

    def test_conditional_query_addition_shapes_and_gradients(self):
        torch.manual_seed(23)
        geometry_block = DecoderBlock(
            dim=32,
            num_heads=4,
            mlp_ratio=2.0,
            qkv_bias=True,
            ffn_bias=True,
            qk_norm=True,
            init_values=1e-3,
        )
        geometry_block.requires_grad_(False)
        decoder = ConditionalQueryDecoder(
            deepcopy(geometry_block), token_dim=32, prompt_dim=16
        )
        decoder.set_trainable()
        tokens = torch.randn(2, 11, 32)
        keys = torch.randn(2, 4, 7, 8)
        values = torch.randn(2, 4, 7, 8)
        prompts = torch.randn(2, 3, 16)
        logits, hidden = decoder(tokens, keys, values, prompts)
        self.assertEqual(logits.shape, (2, 3, 11))
        self.assertEqual(hidden.shape, (2, 3, 11, 32))
        logits.square().mean().backward()
        trainable = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in trainable
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in geometry_block.parameters()))

    def test_conditional_query_can_freeze_copied_decoder(self):
        torch.manual_seed(29)
        decoder = ConditionalQueryDecoder(
            DecoderBlock(
                dim=32,
                num_heads=4,
                mlp_ratio=2.0,
                qkv_bias=True,
                ffn_bias=True,
                qk_norm=True,
                init_values=1e-3,
            ),
            token_dim=32,
            prompt_dim=16,
        )
        decoder.set_trainable(tune_last_cross_attention=False)
        logits, _ = decoder(
            torch.randn(1, 11, 32),
            torch.randn(1, 4, 7, 8),
            torch.randn(1, 4, 7, 8),
            torch.randn(1, 2, 16),
        )
        logits.square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in decoder.prompt_projection.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in decoder.matching_head.parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad and parameter.grad is None
                for parameter in decoder.semantic_last_decoder.parameters()
            )
        )


@unittest.skipUnless(DEFAULT_CLIP_MODEL_PATH.is_dir(), "Local CLIP checkpoint is missing")
class TestPromptMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.matcher = PromptConditionedTokenMatcher(DEFAULT_CLIP_MODEL_PATH)
        cls.query_image = torch.rand(1, 3, 224, 224)
        cls.query_mask = torch.zeros(1, 1, 224, 224)
        cls.query_mask[:, :, 32:192, 48:176] = 1

    def setUp(self):
        self.matcher.zero_grad(set_to_none=True)

    def test_text_image_and_mixed_forward(self):
        hidden = torch.randn(1, 16, 1024)
        common = {
            "gs_token_hidden": hidden,
            "text_query": ["chair"],
            "query_image": self.query_image,
            "query_mask": self.query_mask,
        }
        for mode in ("text_only", "image_only", "text_image_mixed"):
            output = self.matcher(mode=mode, **common)
            self.assertEqual(output["prompt_embedding"].shape, (1, 1, 512))
            self.assertEqual(output["token_logits"].shape, (1, 1, 16))
            torch.testing.assert_close(
                output["prompt_embedding"].norm(dim=-1), torch.ones(1, 1)
            )

    def test_masked_input_cls_removes_outside_mask_pixels(self):
        encoder = self.matcher.prompt_encoder
        original_mode = encoder.image_pooling
        image_a = self.query_image.clone()
        image_b = image_a.clone()
        outside = self.query_mask.expand_as(image_b) == 0
        image_b[outside] = 1.0 - image_b[outside]
        try:
            encoder.image_pooling = "masked_input_cls"
            embedding_a = encoder.encode_image(image_a, self.query_mask)
            embedding_b = encoder.encode_image(image_b, self.query_mask)
        finally:
            encoder.image_pooling = original_mode
        torch.testing.assert_close(embedding_a, embedding_b, rtol=0, atol=1e-6)

    def test_rejects_unknown_image_pooling(self):
        with self.assertRaisesRegex(ValueError, "Unsupported CLIP image pooling"):
            PromptEncoder(DEFAULT_CLIP_MODEL_PATH, image_pooling="unknown")

    def test_1024_and_4096_token_shapes(self):
        decoder = PromptGaussianDecoder()
        prompts = torch.randn(1, 2, 512)
        for token_count in (1024, 4096):
            logits = decoder(torch.randn(1, token_count, 1024), prompts)
            self.assertEqual(logits.shape, (1, 2, token_count))

    def test_only_matching_decoder_receives_gradients(self):
        tokengs_stub = nn.Sequential(nn.Linear(32, 1024), nn.LayerNorm(1024))
        gaussian_head = nn.Linear(1024, 14)
        frozen_backbone = nn.ModuleDict(
            {"tokengs": tokengs_stub, "gaussian_head": gaussian_head}
        )
        self.matcher.freeze_backbones(frozen_backbone)
        self.matcher.train()

        hidden = tokengs_stub(torch.randn(1, 12, 32))
        output = self.matcher(
            hidden,
            mode="text_only",
            text_query=["table"],
        )
        output["token_logits"].mean().backward()

        self.assertTrue(all(not p.requires_grad for p in frozen_backbone.parameters()))
        self.assertTrue(all(p.grad is None for p in frozen_backbone.parameters()))
        self.assertTrue(all(not p.requires_grad for p in self.matcher.prompt_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in self.matcher.prompt_encoder.parameters()))
        trainable = list(self.matcher.matching_decoder.parameters())
        self.assertTrue(trainable)
        self.assertTrue(all(p.requires_grad and p.grad is not None for p in trainable))
        self.assertFalse(self.matcher.prompt_encoder.training)

    def test_trainable_checkpoint_excludes_clip(self):
        state = self.matcher.trainable_state_dict()
        self.assertTrue(state)
        self.assertTrue(all(key.startswith("matching_decoder.") for key in state))
        self.assertFalse(any("clip_model" in key for key in state))
        self.assertEqual(set(state), set(self.matcher.state_dict()))

        restored = PromptConditionedTokenMatcher(DEFAULT_CLIP_MODEL_PATH)
        incompatibilities = restored.load_state_dict(state, strict=True)
        self.assertEqual(incompatibilities.missing_keys, [])
        self.assertEqual(incompatibilities.unexpected_keys, [])

    def test_optional_decoder_hidden_preserves_gaussian_rgb(self):
        model = TokenGS.__new__(TokenGS)
        nn.Module.__init__(model)
        model.opt = SimpleNamespace(
            time_embedding=False,
            num_dynamic_gs_tokens=0,
            num_gs_tokens=8,
            gaussian_z_offset=1.0,
        )
        model.gs_tokens = nn.Parameter(torch.randn(8, 1024))
        model.enc_dec_backbone = _DummyBackbone()
        model.activation_head = _DummyGaussianHead()
        latent = EncoderLatent(
            keys=torch.randn(2, 1, 3, 4),
            values=torch.randn(2, 1, 3, 4),
        )
        decoder_input = ModelInputDecoder()

        default_gaussians = model.forward_decoder(latent, decoder_input)
        optional_gaussians, hidden = model.forward_decoder(
            latent, decoder_input, return_gs_token_hidden=True
        )
        self.assertEqual(hidden.shape, (2, 8, 1024))
        torch.testing.assert_close(default_gaussians, optional_gaussians, rtol=0, atol=0)
        torch.testing.assert_close(
            default_gaussians[..., 11:14],
            optional_gaussians[..., 11:14],
            rtol=0,
            atol=0,
        )


@unittest.skipUnless(
    os.environ.get("TOKENGS_RUN_ORIGINAL_CHECKPOINT_TEST") == "1",
    "Set TOKENGS_RUN_ORIGINAL_CHECKPOINT_TEST=1 for the 847 MiB integration test",
)
class TestOriginalCheckpointCompatibility(unittest.TestCase):
    def test_original_checkpoint_loads_strictly(self):
        checkpoint = Path("/space0/mawb/TokenGS/ckpts/dl3dv_6v.safetensors")
        self.assertTrue(checkpoint.is_file())
        model = TokenGS(
            Options(
                img_size=(256, 448),
                num_gs_tokens=4096,
                num_input_views=6,
            )
        )
        incompatibilities = model.load_state_dict(load_file(checkpoint), strict=True)
        self.assertEqual(incompatibilities.missing_keys, [])
        self.assertEqual(incompatibilities.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main()
