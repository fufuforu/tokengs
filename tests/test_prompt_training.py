import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from tokengs.models.prompt_matching import PromptGaussianDecoder

from tokengs.models.prompt_training import (
    compute_prompt_mask_loss,
    compute_prompt_mask_metrics,
    token_scores_to_gaussians,
)
from tokengs.options import config_defaults
from tokengs.rendering.gs import GaussianRenderer


def _fake_rasterization(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    width,
    height,
    **_kwargs,
):
    del quats, scales, Ks
    views = viewmats.shape[0]
    alpha_value = opacities.mean()
    rgb_value = colors.mean(dim=0) * alpha_value
    rgb = rgb_value.view(1, 1, 1, 3).expand(views, height, width, 3)
    depth = torch.zeros(views, height, width, 1, device=colors.device)
    alpha = alpha_value.view(1, 1, 1, 1).expand(views, height, width, 1)
    means2d = torch.zeros(views, means.shape[0], 2, device=colors.device)
    return torch.cat((rgb, depth), dim=-1), alpha, {"means2d": means2d}


class TestPromptTrainingUtilities(unittest.TestCase):
    def test_contiguous_token_to_gaussian_mapping(self):
        logits = torch.tensor([[[0.0, torch.log(torch.tensor(3.0))]]])
        scores = token_scores_to_gaussians(logits, num_gaussians_per_token=64)
        self.assertEqual(scores.shape, (1, 1, 128))
        torch.testing.assert_close(scores[..., :64], torch.full((1, 1, 64), 0.5))
        torch.testing.assert_close(scores[..., 64:], torch.full((1, 1, 64), 0.75))

    def test_mask_loss_and_metrics_backward(self):
        probability = torch.tensor(
            [[[[[[0.8, 0.2], [0.6, 0.1]]]]]], requires_grad=True
        )
        target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        valid = torch.ones_like(probability, dtype=torch.bool)
        losses = compute_prompt_mask_loss(probability, target, valid)
        metrics = compute_prompt_mask_metrics(probability, target, valid)
        losses["loss"].backward()
        self.assertIsNotNone(probability.grad)
        self.assertGreater(float(losses["loss_bce"]), 0)
        self.assertGreater(float(losses["loss_dice"]), 0)
        self.assertEqual(float(metrics["mask_iou"]), 1.0)

    def test_mask_bce_receives_probability_not_logits(self):
        probability = torch.tensor([[[[[[0.8, 0.2]]]]]], requires_grad=True)
        target = torch.tensor([[[[1.0, 0.0]]]])
        valid = torch.ones_like(probability, dtype=torch.bool)
        with patch(
            "tokengs.models.prompt_training.F.binary_cross_entropy",
            wraps=F.binary_cross_entropy,
        ) as bce:
            compute_prompt_mask_loss(probability, target, valid)
        bce_input = bce.call_args.args[0]
        self.assertGreaterEqual(float(bce_input.min()), 0.0)
        self.assertLessEqual(float(bce_input.max()), 1.0)
        self.assertEqual(bce_input.dtype, torch.float32)

    def test_mask_loss_autocast_fp16_bf16_matches_fp32(self):
        raw = torch.tensor(
            [[[[[[1.2, -0.8], [0.4, -1.6]]]]]], requires_grad=True
        )
        target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        valid = torch.ones_like(raw, dtype=torch.bool)
        reference = compute_prompt_mask_loss(raw.sigmoid(), target, valid)

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                test_raw = raw.detach().clone().requires_grad_(True)
                with torch.autocast(device_type="cpu", dtype=dtype):
                    probability = test_raw.to(dtype).sigmoid()
                    losses = compute_prompt_mask_loss(probability, target, valid)
                    metrics = compute_prompt_mask_metrics(probability, target, valid)
                self.assertEqual(losses["loss"].dtype, torch.float32)
                self.assertTrue(
                    all(value.dtype == torch.float32 for value in metrics.values())
                )
                losses["loss"].backward()
                self.assertIsNotNone(test_raw.grad)
                self.assertTrue(torch.isfinite(test_raw.grad).all())
                torch.testing.assert_close(
                    losses["loss"], reference["loss"], rtol=5e-3, atol=5e-3
                )

    def test_prompt_decoder_gradients_under_fp16_bf16_autocast(self):
        frozen_tokengs = torch.nn.Linear(8, 16)
        frozen_tokengs.requires_grad_(False)
        token_input = torch.randn(1, 6, 8)
        with torch.no_grad():
            frozen_tokens = frozen_tokengs(token_input)
        target = torch.tensor([[[[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]]])

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                decoder = PromptGaussianDecoder(
                    token_dim=16, prompt_dim=12, hidden_dim=8, num_heads=2
                )
                prompt = torch.randn(1, 1, 12)
                with torch.autocast(device_type="cpu", dtype=dtype):
                    logits = decoder(frozen_tokens, prompt)
                    probability = logits.sigmoid()[:, :, None, None, None, :]
                    valid = torch.ones_like(probability, dtype=torch.bool)
                    losses = compute_prompt_mask_loss(probability, target, valid)
                losses["loss"].backward()
                decoder_grads = [
                    parameter.grad
                    for parameter in decoder.parameters()
                    if parameter.requires_grad
                ]
                self.assertTrue(decoder_grads)
                self.assertTrue(
                    all(
                        gradient is not None and torch.isfinite(gradient).all()
                        for gradient in decoder_grads
                    )
                )
                self.assertTrue(
                    all(parameter.grad is None for parameter in frozen_tokengs.parameters())
                )

    def test_overfit_preset_is_fixed_500_step_constant_lr(self):
        opt = config_defaults["prompt_scannet_overfit"]
        self.assertEqual(opt.prompt_overfit_sample_index, 12)
        self.assertEqual(opt.num_epochs, 500)
        self.assertEqual(opt.max_iters_per_epoch, 1)
        self.assertEqual(opt.lr, 1e-4)
        self.assertEqual(opt.lr_scheduler, "constant")
        self.assertEqual(opt.print_freq, 10)
        self.assertEqual(opt.prompt_visualization_steps, (0, 10, 50, 100, 200, 500))

    def test_small_prompt_training_presets(self):
        smoke = config_defaults["prompt_scannet_small_smoke"]
        train = config_defaults["prompt_scannet_small_train"]
        evaluate = config_defaults["prompt_scannet_small_eval"]
        self.assertEqual(smoke.num_epochs * smoke.max_iters_per_epoch, 20)
        self.assertEqual(train.num_epochs * train.max_iters_per_epoch, 2000)
        self.assertEqual(train.max_iters_per_epoch, 200)
        self.assertEqual(train.lr, 1e-4)
        self.assertEqual(train.lr_scheduler, "constant")
        self.assertEqual(train.num_gs_tokens, 1024)
        self.assertEqual(train.prompt_mode, "manifest")
        self.assertEqual(train.max_eval_iters, 24)
        self.assertFalse(train.eval_before_training)
        self.assertTrue(evaluate.evaluating)
        self.assertEqual(evaluate.max_eval_iters, 24)

        diverse_smoke = config_defaults["prompt_scannet_query_diverse_smoke"]
        diverse_train = config_defaults["prompt_scannet_query_diverse_train"]
        diverse_eval = config_defaults["prompt_scannet_query_diverse_eval"]
        self.assertEqual(
            diverse_smoke.num_epochs * diverse_smoke.max_iters_per_epoch, 20
        )
        self.assertEqual(
            diverse_train.num_epochs * diverse_train.max_iters_per_epoch, 2000
        )
        self.assertEqual(diverse_train.batch_size, train.batch_size)
        self.assertEqual(diverse_train.prompt_lambda_bce, train.prompt_lambda_bce)
        self.assertEqual(diverse_train.prompt_lambda_dice, train.prompt_lambda_dice)
        self.assertIn(
            "query_diverse",
            diverse_train.dataset_kwargs["small_manifest_path"],
        )
        self.assertTrue(diverse_eval.evaluating)
        self.assertEqual(diverse_eval.eval_n_media_dumps, 24)

        masked_smoke = config_defaults["prompt_scannet_masked_cls_smoke"]
        masked_train = config_defaults["prompt_scannet_masked_cls_train"]
        masked_eval = config_defaults["prompt_scannet_masked_cls_eval"]
        self.assertEqual(masked_smoke.num_epochs * masked_smoke.max_iters_per_epoch, 20)
        self.assertEqual(masked_train.num_epochs * masked_train.max_iters_per_epoch, 2000)
        self.assertEqual(masked_train.prompt_image_pooling, "masked_input_cls")
        self.assertTrue(masked_train.prompt_save_validation_checkpoints)
        self.assertEqual(masked_train.batch_size, diverse_train.batch_size)
        self.assertEqual(masked_train.prompt_lambda_bce, diverse_train.prompt_lambda_bce)
        self.assertEqual(masked_train.prompt_lambda_dice, diverse_train.prompt_lambda_dice)
        self.assertTrue(masked_eval.evaluating)
        self.assertTrue(masked_eval.resume.endswith("model_best.safetensors"))

        conditional_smoke = config_defaults["conditional_v3_scannet_smoke"]
        conditional_train = config_defaults["conditional_v3_scannet_train"]
        conditional_eval = config_defaults["conditional_v3_scannet_eval"]
        self.assertEqual(
            conditional_train.model_type, "conditional_prompt_tokengs"
        )
        self.assertEqual(conditional_train.prompt_mode, "manifest")
        self.assertEqual(conditional_train.num_gs_tokens, 1024)
        self.assertEqual(
            conditional_train.num_epochs * conditional_train.max_iters_per_epoch,
            2000,
        )
        self.assertEqual(
            conditional_smoke.num_epochs * conditional_smoke.max_iters_per_epoch,
            20,
        )
        self.assertTrue(conditional_eval.resume.endswith("model_best.safetensors"))

        frozen_smoke = config_defaults[
            "conditional_v3_frozen_decoder_scannet_smoke"
        ]
        frozen_train = config_defaults[
            "conditional_v3_frozen_decoder_scannet_train"
        ]
        frozen_eval = config_defaults[
            "conditional_v3_frozen_decoder_scannet_eval"
        ]
        self.assertFalse(frozen_train.conditional_v3_tune_last_cross_attention)
        self.assertEqual(
            frozen_train.num_epochs * frozen_train.max_iters_per_epoch, 2000
        )
        self.assertEqual(
            frozen_smoke.num_epochs * frozen_smoke.max_iters_per_epoch, 20
        )
        self.assertEqual(frozen_train.lr, conditional_train.lr)
        self.assertEqual(
            frozen_train.dataset_kwargs, conditional_train.dataset_kwargs
        )
        self.assertTrue(frozen_eval.evaluating)
        self.assertTrue(frozen_eval.resume.endswith("model_best.safetensors"))

        target_diverse_smoke = config_defaults["prompt_scannet_target_diverse_smoke"]
        target_diverse_train = config_defaults["prompt_scannet_target_diverse_train"]
        target_diverse_eval = config_defaults["prompt_scannet_target_diverse_eval"]
        self.assertEqual(target_diverse_train.model_type, "prompt_tokengs")
        self.assertEqual(target_diverse_train.num_gs_tokens, 1024)
        self.assertEqual(
            target_diverse_train.num_epochs
            * target_diverse_train.max_iters_per_epoch,
            2000,
        )
        self.assertEqual(
            target_diverse_smoke.num_epochs
            * target_diverse_smoke.max_iters_per_epoch,
            20,
        )
        self.assertEqual(target_diverse_train.max_eval_iters, 24)
        self.assertEqual(target_diverse_train.lr, 1e-4)
        self.assertTrue(target_diverse_train.prompt_save_validation_checkpoints)
        self.assertIn(
            "target_diverse",
            target_diverse_train.dataset_kwargs["small_manifest_path"],
        )
        self.assertTrue(target_diverse_eval.evaluating)
        self.assertTrue(target_diverse_eval.resume.endswith("model_best.safetensors"))

        last_cross_smoke = config_defaults[
            "prompt_scannet_target_diverse_last_cross_smoke"
        ]
        last_cross_train = config_defaults[
            "prompt_scannet_target_diverse_last_cross_train"
        ]
        last_cross_eval = config_defaults[
            "prompt_scannet_target_diverse_last_cross_eval"
        ]
        self.assertTrue(last_cross_smoke.prompt_tune_last_cross_attention)
        self.assertTrue(last_cross_train.prompt_tune_last_cross_attention)
        self.assertEqual(last_cross_train.model_type, "prompt_tokengs")
        self.assertEqual(last_cross_train.num_gs_tokens, 1024)
        self.assertEqual(
            last_cross_train.num_epochs * last_cross_train.max_iters_per_epoch,
            2000,
        )
        self.assertEqual(
            last_cross_train.dataset_kwargs,
            target_diverse_train.dataset_kwargs,
        )
        self.assertTrue(last_cross_eval.evaluating)
        self.assertTrue(last_cross_eval.resume.endswith("model_best.safetensors"))

    def test_prompt_renderer_forward_and_backward(self):
        renderer = GaussianRenderer(
            SimpleNamespace(
                img_size=(4, 5),
                znear=0.01,
                zfar=10.0,
                deferred_bp=False,
            )
        )
        gaussians = torch.zeros(1, 128, 14)
        gaussians[..., 2] = 2.0
        gaussians[..., 3] = 0.8
        gaussians[..., 4:7] = 0.1
        gaussians[..., 7] = 1.0
        original_gaussians = gaussians.clone()
        logits = torch.randn(1, 1, 2, requires_grad=True)
        scores = token_scores_to_gaussians(logits, 64)
        cam_view = torch.eye(4).view(1, 1, 4, 4)
        intrinsics = torch.tensor([[[100.0, 100.0, 2.5, 2.0]]])
        with patch("tokengs.rendering.gs.rasterization", _fake_rasterization):
            output = renderer.render_prompt_scores(
                gaussians, scores, cam_view, intrinsics=intrinsics
            )
        self.assertEqual(output["rendered_prompt_probability"].shape, (1, 1, 1, 1, 4, 5))
        self.assertEqual(output["rendered_alpha"].shape, (1, 1, 1, 1, 4, 5))
        torch.testing.assert_close(gaussians, original_gaussians, rtol=0, atol=0)
        output["rendered_prompt_probability"].mean().backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0)


@unittest.skipUnless(
    os.environ.get("TOKENGS_RUN_RE10K_CHECKPOINT_TEST") == "1",
    "Set TOKENGS_RUN_RE10K_CHECKPOINT_TEST=1 for the full checkpoint load",
)
class TestPromptTokenGSCheckpoint(unittest.TestCase):
    def test_re10k_checkpoint_and_freezing(self):
        from tokengs.models.prompt_tokengs import PromptTokenGS

        model = PromptTokenGS(config_defaults["prompt_scannet_smoke"])
        self.assertEqual(tuple(model.gs_tokens.shape), (1024, 1024))
        self.assertEqual(model.num_gaussians_per_token, 64)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if not name.startswith("prompt_matcher.matching_decoder.")
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.prompt_matcher.matching_decoder.parameters()
            )
        )
        state = model.state_dict()
        self.assertTrue(state)
        self.assertFalse(any("clip" in key or "gs_tokens" in key for key in state))


if __name__ == "__main__":
    unittest.main()
