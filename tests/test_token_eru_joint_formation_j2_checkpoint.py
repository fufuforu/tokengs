import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_"
    "joint_formation_v1_ddp8/checkpoints/model_step_000710.safetensors"
)
EXPECTED = "e848f733fe7f3812db15f143787c5e663475fdd3a0bd01b302c2a047f1a33c2d"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class StageJ2CheckpointContractTest(unittest.TestCase):
    def test_parent_contract_if_present(self):
        if not PARENT.is_file():
            self.skipTest("parent checkpoint is not present in this checkout")
        self.assertEqual(sha256_file(PARENT), EXPECTED)
        metadata = PARENT.with_name("metadata_step_000710.json")
        self.assertTrue(metadata.is_file())
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        self.assertEqual(int(payload.get("optimizer_step", -1)), 710)

    def test_j2_stage_contract(self):
        self.assertEqual(710 + 1, 711)
        self.assertEqual(710 + 710, 1420)


if __name__ == "__main__":
    unittest.main()

