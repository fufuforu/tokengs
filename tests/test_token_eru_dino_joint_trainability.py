import unittest

import torch
from torch import nn

from tokengs.train import configure_joint_formation_trainability


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])


class _ERUDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.understanding_decoder_blocks = nn.ModuleList(
            [nn.Linear(4, 4) for _ in range(12)]
        )
        self.reconstruction_to_understanding = nn.ModuleList(
            [nn.Linear(4, 4) for _ in range(12)]
        )
        self.understanding_to_reconstruction = nn.ModuleList(
            [nn.Linear(4, 4) for _ in range(12)]
        )


class _AbsoluteUnitDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_norm = nn.LayerNorm(4)
        self.tok_proj = nn.Linear(4, 4)
        self.unit_queries = nn.Parameter(torch.zeros(8, 4))
        self.unit_readout = nn.Linear(8, 4)
        self.slot_emb = nn.Parameter(torch.zeros(8, 2))
        self.gs_decoder = nn.Linear(10, 14)
        self.center_mlp = nn.Linear(4, 3)


class _DinoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.unit_projector = nn.Linear(4, 4)
        self.dino_extractor = nn.Module()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_dec_backbone = _Decoder()
        self.token_eru_decoder = _ERUDecoder()
        self.token_eru_unit_formation = _AbsoluteUnitDecoder()
        self.absolute_gs_head = _AbsoluteUnitDecoder()
        self.tsh_instance_head = nn.Linear(4, 4)
        self.token_eru_dino_encoder = _DinoEncoder()
        self.token_eru_dino_fusion = nn.Linear(4, 4)
        self.token_eru_metric_head = nn.Linear(4, 4)
        self.gs_tokens = nn.Parameter(torch.zeros(4, 4))


class _Opt:
    token_eru_dino_metric_joint_formation = True


class JointFormationTrainabilityTest(unittest.TestCase):
    def test_complete_disjoint_classification(self):
        model = _Model()
        categories = configure_joint_formation_trainability(model, _Opt())
        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        categorized = {
            name
            for names in categories.values()
            for name in names
        }
        self.assertEqual(categorized, {name for name, _ in model.named_parameters()})
        self.assertEqual(
            trainable,
            {
                name
                for category, names in categories.items()
                if category not in {"frozen_backbone", "frozen_other"}
                for name in names
            },
        )
        self.assertIn("enc_dec_backbone.decoder_blocks.0.weight", trainable)
        self.assertIn("absolute_gs_head.gs_decoder.weight", trainable)
        self.assertIn("absolute_gs_head.tok_proj.weight", trainable)
        self.assertNotIn("enc_dec_backbone.decoder_blocks", categories["frozen_backbone"])

    def test_unrelated_parameters_are_frozen(self):
        model = _Model()
        model.extra_frozen = nn.Linear(4, 4)
        configure_joint_formation_trainability(model, _Opt())
        self.assertTrue(all(not p.requires_grad for p in model.extra_frozen.parameters()))


if __name__ == "__main__":
    unittest.main()
