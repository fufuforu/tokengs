import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = REPO_ROOT / "data" / "scannet_prompt"
TRAIN_MANIFEST = PROMPT_ROOT / "scannet_c3g8_train_provisional.json"
QUERY_BANK = PROMPT_ROOT / "scannet_c3g8_query_bank.json"


@unittest.skipUnless(
    TRAIN_MANIFEST.is_file() and QUERY_BANK.is_file(),
    "ScanNet prompt metadata has not been built",
)
class TestScanNetPromptIntegration(unittest.TestCase):
    def test_manifest_and_bank_do_not_leak_eval_scenes(self):
        manifest = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
        bank = json.loads(QUERY_BANK.read_text(encoding="utf-8"))
        train_scenes = set(manifest["scenes"])
        eval_scenes = set(manifest["excluded_eval_scenes"])
        self.assertTrue(manifest["provisional"])
        self.assertFalse(train_scenes & eval_scenes)
        self.assertEqual(eval_scenes, set(bank["excluded_eval_scenes"]))

        distribution = Counter()
        scenes_by_class = defaultdict(set)
        for entry in bank["entries"]:
            required = {
                "scene",
                "raw_frame_id",
                "class_id",
                "bbox",
                "mask_area",
                "image_size",
                "instance_id",
                "source_type",
            }
            self.assertTrue(required <= set(entry))
            self.assertIn(entry["scene"], train_scenes)
            self.assertNotIn(entry["scene"], eval_scenes)
            self.assertIn(entry["class_id"], range(1, 8))
            self.assertGreater(entry["mask_area"], 0)
            x0, y0, x1, y1 = entry["bbox"]
            width, height = entry["image_size"]
            self.assertTrue(0 <= x0 < x1 <= width)
            self.assertTrue(0 <= y0 < y1 <= height)
            if entry["class_id"] in (4, 5, 6, 7):
                self.assertEqual(entry["source_type"], "instance")
                self.assertIsNotNone(entry["instance_id"])
            else:
                self.assertEqual(entry["source_type"], "semantic")
                self.assertIsNone(entry["instance_id"])
            distribution[entry["class_id"]] += 1
            scenes_by_class[entry["class_id"]].add(entry["scene"])

        self.assertEqual(set(distribution), set(range(1, 8)))
        self.assertTrue(all(len(scenes_by_class[class_id]) >= 2 for class_id in range(1, 8)))


if __name__ == "__main__":
    unittest.main()
