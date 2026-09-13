from pathlib import Path

import torch

from scripts.historical_0324_adapter import (
    BUNDLE_ROOT,
    PRIMARY,
    validate_bundle,
    _historical_mapping,
)


def test_transfer_bundle_has_22_entries_plus_manifest():
    report = validate_bundle()
    assert report["bundle_hash_valid"] is True
    assert report["manifest_entries"] == 22
    assert report["validated_file_count"] == 23


def test_primary_and_expected_artifacts_exist():
    assert BUNDLE_ROOT.is_dir()
    assert PRIMARY.is_file()


def test_historical_mapping_maps_prompt_namespace_without_dino():
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.instance_branch = torch.nn.Linear(2, 2, bias=False)
            self.prompt_matcher = torch.nn.Module()
            self.prompt_matcher.semantic_token_adapter = torch.nn.Linear(
                2, 2, bias=False
            )

    dummy = Dummy()
    # Exercise the pure mapping contract without constructing the 670M model.
    primary = {
        "instance_branch.weight": torch.zeros(2, 2),
        "semantic_token_adapter.weight": torch.zeros(2, 2),
        "instance_branch._dino_model.blocks.0.weight": torch.zeros(1),
    }
    mapped, source_to_target, skipped = _historical_mapping(dummy, primary)
    assert set(mapped) == {
        "instance_branch.weight",
        "prompt_matcher.semantic_token_adapter.weight",
    }
    assert source_to_target["semantic_token_adapter.weight"].startswith("prompt_matcher.")
    assert len(skipped) == 1
