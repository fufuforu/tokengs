import unittest
from copy import deepcopy
from pathlib import Path

import torch

from tokengs.models.semantic_adapter_v2 import (
    C3G8_CLASS_NAMES,
    PromptSemanticAdapter,
    SemanticMatcherV2,
    SemanticTokenAdapter,
    compute_semantic_v2_metrics,
    semantic_token_probabilities,
)
from tokengs.models.prompt_training import compute_prompt_mask_loss
from tokengs.models.enc_dec import DecoderBlock
from tokengs.options import config_defaults


CLIP_PATH = Path("/space0/mawb/tokengs/checkpoints/clip-vit-base-patch32")


class TestSemanticAdapterV2Utilities(unittest.TestCase):
    def test_token_and_prompt_adapters_are_normalized(self):
        token_adapter = SemanticTokenAdapter(input_dim=32, semantic_dim=16)
        prompt_adapter = PromptSemanticAdapter(input_dim=24, semantic_dim=16)
        tokens = token_adapter(torch.randn(2, 11, 32))
        prompts = prompt_adapter(torch.randn(2, 8, 24))
        self.assertEqual(tokens.shape, (2, 11, 16))
        self.assertEqual(prompts.shape, (2, 8, 16))
        torch.testing.assert_close(tokens.norm(dim=-1), torch.ones(2, 11))
        torch.testing.assert_close(prompts.norm(dim=-1), torch.ones(2, 8))

    def test_multiclass_metrics_are_macro_and_ignore_zero(self):
        target = torch.zeros(1, 8, 1, 1, 2, 2)
        target[:, 0, :, :, 0, 0] = 1
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[..., 1, 1] = False
        probability = target * 0.9 + (1.0 - target) * 0.1
        metrics = compute_semantic_v2_metrics(probability, target, valid)
        self.assertEqual(metrics["per_class_iou"].shape, (8,))
        self.assertAlmostEqual(float(metrics["per_class_iou"][0]), 1.0)
        self.assertAlmostEqual(float(metrics["per_class_recall"][0]), 1.0)
        self.assertAlmostEqual(float(metrics["macro_miou"]), 0.125)
        self.assertEqual(metrics["argmax_confusion"].shape, (8, 8))

    def test_balanced_bce_upweights_rare_positive_pixels(self):
        probability = torch.full((1, 8, 1, 1, 4, 4), 0.1, requires_grad=True)
        target = torch.zeros_like(probability)
        target[:, :, :, :, 0, 0] = 1.0
        valid = torch.ones_like(probability, dtype=torch.bool)
        regular = compute_prompt_mask_loss(
            probability, target, valid, lambda_dice=0.0
        )["loss_bce"]
        balanced = compute_prompt_mask_loss(
            probability,
            target,
            valid,
            lambda_dice=0.0,
            balance_classes=True,
        )["loss_bce"]
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            balanced_bf16 = compute_prompt_mask_loss(
                probability,
                target,
                valid,
                lambda_dice=0.0,
                balance_classes=True,
            )["loss_bce"]
        self.assertGreater(float(balanced), float(regular))
        torch.testing.assert_close(balanced_bf16, balanced)
        balanced.backward()
        self.assertTrue(torch.isfinite(probability.grad).all())

    def test_softmax_scores_are_mutually_exclusive_and_differentiable(self):
        logits = torch.randn(2, 8, 1024, requires_grad=True)
        probability = semantic_token_probabilities(logits, "softmax")
        torch.testing.assert_close(
            probability.sum(dim=1), torch.ones(2, 1024)
        )
        probability.square().mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(
            semantic_token_probabilities(logits.detach(), "sigmoid"),
            logits.detach().float().sigmoid(),
        )

    def test_v2_presets_keep_the_control_variables_fixed(self):
        smoke = config_defaults["semantic_v2_scannet_smoke"]
        train = config_defaults["semantic_v2_scannet_train"]
        evaluate = config_defaults["semantic_v2_scannet_eval"]
        self.assertEqual(smoke.num_epochs * smoke.max_iters_per_epoch, 20)
        self.assertEqual(train.num_epochs * train.max_iters_per_epoch, 2000)
        self.assertEqual(train.model_type, "semantic_tokengs_v2")
        self.assertEqual(train.data_mode, (("scannet_semantic_small", 1),))
        self.assertEqual(train.semantic_v2_dim, 256)
        self.assertEqual(train.num_gs_tokens, 1024)
        self.assertEqual(train.batch_size, 1)
        self.assertEqual(train.lr, 1e-4)
        self.assertEqual(train.prompt_lambda_bce, 1.0)
        self.assertEqual(train.prompt_lambda_dice, 1.0)
        self.assertTrue(train.prompt_save_validation_checkpoints)
        self.assertTrue(evaluate.evaluating)
        self.assertTrue(evaluate.resume.endswith("model_best.safetensors"))
        balanced_smoke = config_defaults["semantic_v2_balanced_bce_scannet_smoke"]
        balanced_train = config_defaults["semantic_v2_balanced_bce_scannet_train"]
        balanced_eval = config_defaults["semantic_v2_balanced_bce_scannet_eval"]
        self.assertTrue(balanced_smoke.semantic_v2_balanced_bce)
        self.assertTrue(balanced_train.semantic_v2_balanced_bce)
        self.assertEqual(
            balanced_train.num_epochs * balanced_train.max_iters_per_epoch, 2000
        )
        self.assertTrue(balanced_eval.resume.endswith("model_best.safetensors"))
        competition_smoke = config_defaults[
            "semantic_v2_balanced_softmax_scannet_smoke"
        ]
        competition_train = config_defaults[
            "semantic_v2_balanced_softmax_scannet_train"
        ]
        competition_eval = config_defaults[
            "semantic_v2_balanced_softmax_scannet_eval"
        ]
        self.assertEqual(competition_smoke.semantic_v2_score_mode, "softmax")
        self.assertTrue(competition_train.semantic_v2_balanced_bce)
        self.assertEqual(competition_train.semantic_v2_score_mode, "softmax")
        self.assertEqual(
            competition_train.num_epochs * competition_train.max_iters_per_epoch,
            2000,
        )
        self.assertTrue(competition_eval.resume.endswith("model_best.safetensors"))
        partial_smoke = config_defaults[
            "semantic_v2_last_cross_attn_scannet_smoke"
        ]
        partial_train = config_defaults[
            "semantic_v2_last_cross_attn_scannet_train"
        ]
        partial_eval = config_defaults[
            "semantic_v2_last_cross_attn_scannet_eval"
        ]
        self.assertTrue(partial_smoke.semantic_v2_tune_last_cross_attention)
        self.assertTrue(partial_train.semantic_v2_tune_last_cross_attention)
        self.assertEqual(partial_train.semantic_v2_score_mode, "softmax")
        self.assertTrue(partial_train.semantic_v2_balanced_bce)
        self.assertEqual(partial_train.num_gs_tokens, 1024)
        self.assertEqual(
            partial_train.num_epochs * partial_train.max_iters_per_epoch, 2000
        )
        self.assertTrue(partial_eval.resume.endswith("model_best.safetensors"))

    def test_semantic_decoder_fork_preserves_geometry_and_backpropagates(self):
        torch.manual_seed(17)
        geometry_block = DecoderBlock(
            dim=32,
            num_heads=4,
            mlp_ratio=2.0,
            qkv_bias=True,
            ffn_bias=True,
            qk_norm=True,
            init_values=1e-3,
        )
        semantic_block = deepcopy(geometry_block)
        geometry_block.requires_grad_(False)
        semantic_block.requires_grad_(False)
        semantic_block.gs_cross_attn.requires_grad_(True)
        semantic_block.gs_cross_attn_scale.requires_grad_(True)
        tokens = torch.randn(2, 11, 32)
        keys = torch.randn(2, 4, 7, 8)
        values = torch.randn(2, 4, 7, 8)
        geometry_before = geometry_block(tokens, keys, values).detach()
        semantic_before = semantic_block(tokens, keys, values)
        torch.testing.assert_close(semantic_before, geometry_before)
        semantic_before.square().mean().backward()
        trainable = [p for p in semantic_block.parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(
            all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        )
        self.assertTrue(all(p.grad is None for p in geometry_block.parameters()))
        geometry_after = geometry_block(tokens, keys, values).detach()
        torch.testing.assert_close(geometry_after, geometry_before)


@unittest.skipUnless(CLIP_PATH.is_dir(), "Local CLIP checkpoint is missing")
class TestSemanticMatcherV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(11)
        cls.matcher = SemanticMatcherV2(CLIP_PATH, token_dim=32, semantic_dim=16)

    def test_cosine_forward_and_temperature_gradient(self):
        output = self.matcher(torch.randn(2, 13, 32))
        self.assertEqual(output["semantic_tokens"].shape, (2, 13, 16))
        self.assertEqual(output["semantic_prompts"].shape, (2, 8, 16))
        self.assertEqual(output["token_logits"].shape, (2, 8, 13))
        output["token_logits"].mean().backward()
        self.assertIsNotNone(self.matcher.log_temperature.grad)
        self.assertTrue(torch.isfinite(self.matcher.log_temperature.grad))
        for module in (
            self.matcher.semantic_token_adapter,
            self.matcher.prompt_semantic_adapter,
        ):
            self.assertTrue(
                all(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    for parameter in module.parameters()
                )
            )
        self.assertTrue(all(not p.requires_grad for p in self.matcher.text_encoder.parameters()))

    def test_balanced_bce_backward_only_updates_semantic_matcher(self):
        self.matcher.zero_grad(set_to_none=True)
        output = self.matcher(torch.randn(1, 16, 32))
        probability = semantic_token_probabilities(
            output["token_logits"], "softmax"
        ).reshape(1, 8, 1, 1, 4, 4)
        target = torch.zeros_like(probability)
        for class_index in range(8):
            target[:, class_index, :, :, class_index // 4, class_index % 4] = 1
        valid = torch.ones_like(probability, dtype=torch.bool)
        loss = compute_prompt_mask_loss(
            probability, target, valid, balance_classes=True
        )["loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        trainable = (
            list(self.matcher.semantic_token_adapter.parameters())
            + list(self.matcher.prompt_semantic_adapter.parameters())
            + [self.matcher.log_temperature]
        )
        self.assertTrue(
            all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        )
        self.assertTrue(
            all(p.grad is None for p in self.matcher.text_encoder.parameters())
        )

    def test_trainable_checkpoint_excludes_clip(self):
        state = self.matcher.trainable_state_dict()
        self.assertIn("log_temperature", state)
        self.assertTrue(
            all(
                key == "log_temperature"
                or key.startswith("semantic_token_adapter.")
                or key.startswith("prompt_semantic_adapter.")
                for key in state
            )
        )
        self.assertFalse(any("clip" in key for key in state))
        restored = SemanticMatcherV2(CLIP_PATH, token_dim=32, semantic_dim=16)
        incompatibilities = restored.load_trainable_state_dict(state, strict=True)
        self.assertEqual(incompatibilities.missing_keys, [])
        self.assertEqual(incompatibilities.unexpected_keys, [])

    def test_fixed_class_order(self):
        self.assertEqual(
            C3G8_CLASS_NAMES,
            ("wall", "floor", "ceiling", "chair", "table", "sofa", "bed", "other"),
        )


if __name__ == "__main__":
    unittest.main()
