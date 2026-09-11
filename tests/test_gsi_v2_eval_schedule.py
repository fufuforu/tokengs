import copy
import math
import unittest
from types import SimpleNamespace

import torch

from tokengs.models.globalsplat_instance_v2.model import GlobalSplatInstanceV2
from tokengs.options import config_defaults


def _dummy_model() -> GlobalSplatInstanceV2:
    model = object.__new__(GlobalSplatInstanceV2)
    torch.nn.Module.__init__(model)
    model.phase = "joint"
    model.opt = config_defaults["gsi_v2_joint_scannet_short355_ddp8"]
    model.reconstruction = SimpleNamespace(set_stage=lambda stage, mix: None)
    model.test_parameter = torch.nn.Parameter(torch.tensor([1.25]))
    model._train_step = 0
    model._gate = 0.0
    model._gate_gradient_scale = 0.0
    model._instance_weight = 0.0
    return model


class EvalScheduleTest(unittest.TestCase):
    def test_invalid_inputs(self):
        model = _dummy_model()
        with self.assertRaises(ValueError):
            model.set_eval_schedule(-1)
        with self.assertRaises(TypeError):
            model.set_eval_schedule(1.5)
        with self.assertRaises(ValueError):
            model.set_eval_schedule(1, gate_override=math.nan)
        with self.assertRaises(ValueError):
            model.set_eval_schedule(1, gate_override=-0.1)
        with self.assertRaises(ValueError):
            model.set_eval_schedule(1, gate_override=1.1)

    def test_short355_native_schedule(self):
        model = _dummy_model()
        expected = {
            0: (0.0, 0.0),
            25: (0.5, 0.0),
            50: (1.0, 0.0),
            100: (1.0, 1.0),
            200: (1.0, 1.0),
            355: (1.0, 1.0),
        }
        for step, (weight, gate) in expected.items():
            result = model.set_eval_schedule(step)
            self.assertEqual(result["completed_optimizer_step"], float(step))
            self.assertAlmostEqual(result["instance_weight"], weight)
            self.assertAlmostEqual(result["gate"], gate)
            self.assertAlmostEqual(result["gate_gradient_scale"], 0.1 * gate)

    def test_override_preserves_instance_weight(self):
        model = _dummy_model()
        result = model.set_eval_schedule(200, gate_override=0.0)
        self.assertEqual(result["gate"], 0.0)
        self.assertEqual(result["gate_gradient_scale"], 0.0)
        self.assertEqual(result["instance_weight"], 1.0)

    def test_eval_schedule_does_not_change_parameters_or_keys(self):
        model = _dummy_model()
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        keys_before = tuple(model.state_dict().keys())
        model.set_eval_schedule(355)
        self.assertEqual(keys_before, tuple(model.state_dict().keys()))
        for key, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[key])

    def test_train_and_eval_schedule_match(self):
        train_model = _dummy_model()
        eval_model = _dummy_model()
        for step in (0, 25, 50, 100, 200, 355):
            train_model.set_train_step(step)
            result = eval_model.set_eval_schedule(step)
            self.assertAlmostEqual(train_model._instance_weight, result["instance_weight"])
            self.assertAlmostEqual(train_model._gate, result["gate"])
            self.assertAlmostEqual(train_model._gate_gradient_scale, result["gate_gradient_scale"])

    def test_eval_stage_remains_callable(self):
        model = _dummy_model()
        model.set_eval_stage()


if __name__ == "__main__":
    unittest.main()
