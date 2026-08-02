import json
import unittest
from copy import deepcopy
from collections import Counter
from pathlib import Path

from scripts.build_scannet_prompt_small_split import _attach_queries


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    REPO_ROOT / "data" / "scannet_prompt" / "scannet_prompt_small_64_8.json"
)
DIVERSE_MANIFEST = (
    REPO_ROOT
    / "data"
    / "scannet_prompt"
    / "scannet_prompt_small_64_8_query_diverse.json"
)
TARGET_DIVERSE_MANIFEST = (
    REPO_ROOT
    / "data"
    / "scannet_prompt"
    / "scannet_prompt_target_diverse_64_8.json"
)


class TestQueryAssignment(unittest.TestCase):
    def test_rotates_across_entries_and_scenes(self):
        samples = [
            {
                "scene": f"target_{index}",
                "class_id": 4,
                "prompt_mode": "image_only",
            }
            for index in range(6)
        ]
        bank_entries = [
            {
                "scene": scene,
                "raw_frame_id": frame_id,
                "class_id": 4,
                "bbox": [0, 0, 32, 32],
                "mask_area": 1024,
                "image_size": [64, 64],
                "instance_id": frame_id,
                "source_type": "instance",
            }
            for frame_id in range(3)
            for scene in ("query_a", "query_b", "query_c")
        ]

        assigned = deepcopy(samples)
        _attach_queries(
            assigned,
            bank_entries,
            allowed_query_scenes={"query_a", "query_b", "query_c"},
        )

        query_keys = {
            (
                sample["query"]["scene"],
                sample["query"]["raw_frame_id"],
                sample["query"]["instance_id"],
            )
            for sample in assigned
        }
        self.assertEqual(len(query_keys), len(samples))
        self.assertEqual(
            {sample["query"]["scene"] for sample in assigned},
            {"query_a", "query_b", "query_c"},
        )


@unittest.skipUnless(
    MANIFEST.is_file() and DIVERSE_MANIFEST.is_file(),
    "Query-diverse ScanNet prompt manifest is unavailable",
)
class TestQueryDiverseManifest(unittest.TestCase):
    def test_changes_only_query_assignment(self):
        baseline = json.loads(MANIFEST.read_text(encoding="utf-8"))
        diverse = json.loads(DIVERSE_MANIFEST.read_text(encoding="utf-8"))

        def without_query(sample):
            return {key: value for key, value in sample.items() if key != "query"}

        for split in ("train_samples", "validation_samples"):
            self.assertEqual(
                [without_query(sample) for sample in baseline[split]],
                [without_query(sample) for sample in diverse[split]],
            )
        self.assertEqual(
            baseline["validation_samples"], diverse["validation_samples"]
        )

    def test_training_queries_are_diverse_and_cross_scene(self):
        manifest = json.loads(DIVERSE_MANIFEST.read_text(encoding="utf-8"))
        for class_id in range(1, 8):
            samples = [
                sample
                for sample in manifest["train_samples"]
                if int(sample["class_id"]) == class_id and sample["query"] is not None
            ]
            unique_queries = {
                (
                    sample["query"]["scene"],
                    int(sample["query"]["raw_frame_id"]),
                    sample["query"].get("instance_id"),
                )
                for sample in samples
            }
            minimum_unique = 14 if class_id == 6 else len(samples)
            self.assertGreaterEqual(len(unique_queries), minimum_unique)
            self.assertTrue(
                all(sample["query"]["scene"] != sample["scene"] for sample in samples)
            )


@unittest.skipUnless(MANIFEST.is_file(), "Small ScanNet prompt manifest is unavailable")
class TestScanNetPromptSmallManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_scene_split_and_eval_exclusion(self):
        train = set(self.manifest["train_scenes"])
        validation = set(self.manifest["validation_scenes"])
        excluded = set(self.manifest["excluded_eval_scenes"])
        self.assertEqual(len(train), 64)
        self.assertEqual(len(validation), 8)
        self.assertFalse(train & validation)
        self.assertFalse((train | validation) & excluded)

    def test_balanced_classes_and_prompt_modes(self):
        expected = {class_id: 40 for class_id in range(1, 9)}
        distribution = Counter(
            int(sample["class_id"]) for sample in self.manifest["train_samples"]
        )
        self.assertEqual(dict(distribution), expected)
        validation_distribution = Counter(
            int(sample["class_id"])
            for sample in self.manifest["validation_samples"]
        )
        self.assertEqual(
            dict(validation_distribution),
            {class_id: 3 for class_id in range(1, 9)},
        )
        image_class_modes = Counter(
            sample["prompt_mode"]
            for sample in self.manifest["train_samples"]
            if int(sample["class_id"]) != 8
        )
        self.assertEqual(
            image_class_modes,
            Counter(
                {
                    "text_only": 88,
                    "image_only": 96,
                    "text_image_mixed": 96,
                }
            ),
        )

    def test_frames_queries_and_other_constraints(self):
        train_scenes = set(self.manifest["train_scenes"])
        for split in ("train_samples", "validation_samples"):
            for sample in self.manifest[split]:
                inputs = sample["input_frame_ids"]
                self.assertEqual(len(set(inputs)), 2)
                self.assertNotIn(sample["target_frame_id"], inputs)
                if int(sample["class_id"]) == 8:
                    self.assertEqual(sample["prompt_mode"], "text_only")
                    self.assertIsNone(sample["query"])
                elif sample["prompt_mode"] == "text_only":
                    self.assertIsNone(sample["query"])
                else:
                    query = sample["query"]
                    self.assertIn(query["scene"], train_scenes)
                    self.assertNotEqual(query["scene"], sample["scene"])
                    self.assertEqual(query["class_id"], sample["class_id"])


@unittest.skipUnless(
    TARGET_DIVERSE_MANIFEST.is_file(),
    "Target-diverse ScanNet prompt manifest is unavailable",
)
class TestTargetDiverseManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(TARGET_DIVERSE_MANIFEST.read_text(encoding="utf-8"))

    def test_has_dense_unique_targets_and_reuses_scene_split(self):
        self.assertEqual(len(self.manifest["train_scenes"]), 64)
        self.assertEqual(len(self.manifest["validation_scenes"]), 8)
        self.assertEqual(self.manifest["target_frames_per_scene"], 32)
        self.assertEqual(self.manifest["min_target_foreground_ratio"], 0.005)
        for split, expected_per_class in (("train_samples", 96), ("validation_samples", 3)):
            samples = self.manifest[split]
            self.assertEqual(len(samples), expected_per_class * 8)
            for class_id in range(1, 9):
                class_samples = [s for s in samples if int(s["class_id"]) == class_id]
                keys = {(s["scene"], int(s["target_frame_id"])) for s in class_samples}
                if split == "train_samples":
                    self.assertEqual(len(keys), expected_per_class)
                    self.assertTrue(
                        all(float(s["foreground_ratio"]) >= 0.005 for s in class_samples)
                    )

    def test_validation_rows_match_query_diverse_baseline(self):
        baseline = json.loads(DIVERSE_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            self.manifest["validation_samples"], baseline["validation_samples"]
        )

    def test_target_diverse_constraints(self):
        train_scenes = set(self.manifest["train_scenes"])
        excluded = set(self.manifest["excluded_eval_scenes"])
        self.assertFalse((set(self.manifest["validation_scenes"]) | train_scenes) & excluded)
        for sample in self.manifest["train_samples"] + self.manifest["validation_samples"]:
            self.assertEqual(len(set(sample["input_frame_ids"])), 2)
            self.assertNotIn(sample["target_frame_id"], sample["input_frame_ids"])
            if int(sample["class_id"]) == 8 or sample["prompt_mode"] == "text_only":
                self.assertIsNone(sample["query"])
            else:
                query = sample["query"]
                self.assertIn(query["scene"], train_scenes)
                self.assertNotEqual(query["scene"], sample["scene"])
                self.assertEqual(query["class_id"], sample["class_id"])


if __name__ == "__main__":
    unittest.main()
