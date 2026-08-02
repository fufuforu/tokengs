import csv
import json
import unittest
from pathlib import Path

import numpy as np

from tokengs.data.datafield import DF_IMAGE_RGB, DF_SEMANTIC_LABEL
from tokengs.data.static.scannet import ScanNetC3G8Eval


REPO_ROOT = Path(__file__).resolve().parents[1]
C3G_ROOT = REPO_ROOT.parent / "C3G" / "datasets" / "scannet_test"
RAW_ROOT = REPO_ROOT / "data" / "ScanNet" / "scans"
LABEL_ROOT = REPO_ROOT / "data" / "scannet2d_labels"
MANIFEST_PATH = C3G_ROOT / "selected_seqs_test.json"


@unittest.skipUnless(MANIFEST_PATH.is_file(), "C3G ScanNet test data is unavailable")
class TestScanNetC3G8Integration(unittest.TestCase):
    def test_mapping_matches_c3g_tsv_for_every_raw_id(self):
        dataset = ScanNetC3G8Eval(
            root_path=str(RAW_ROOT),
            label_root=str(LABEL_ROOT),
            manifest_path=str(MANIFEST_PATH),
        )
        with (C3G_ROOT / "scannetv2-labels.combined.tsv").open(
            newline="", encoding="utf-8"
        ) as handle:
            raw_to_class = {
                int(row["id"]): row["nyu40class"].lower()
                for row in csv.DictReader(handle, delimiter="\t")
            }
        class_to_train_id = {
            name: train_id
            for train_id, name in enumerate(dataset.semantic_class_names, start=1)
        }
        raw_ids = np.asarray([0, *sorted(raw_to_class), 65535], dtype=np.int64)
        expected = np.asarray(
            [
                0
                if raw_id == 0
                else class_to_train_id.get(raw_to_class.get(raw_id), 8)
                for raw_id in raw_ids
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(dataset._map_labels(raw_ids), expected)

    def test_uploaded_manifest_and_raw_frames(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest), 40)
        self.assertTrue(all(len(frame_ids) == 30 for frame_ids in manifest.values()))

        for scene_name, frame_ids in manifest.items():
            for frame_id in frame_ids:
                for directory, suffix in (
                    ("images", ".jpg"),
                    ("images", ".npz"),
                    ("depths", ".png"),
                    ("labels", ".png"),
                ):
                    self.assertTrue(
                        (
                            C3G_ROOT
                            / scene_name
                            / directory
                            / f"{frame_id}{suffix}"
                        ).is_file()
                    )

        dataset = ScanNetC3G8Eval(
            root_path=str(RAW_ROOT),
            label_root=str(LABEL_ROOT),
            manifest_path=str(MANIFEST_PATH),
        )
        report = dataset.get_manifest_report(validate_frames=True)
        self.assertEqual(report["effective_scenes"], 40)
        self.assertEqual(report["samples"], 320)
        self.assertEqual(report["missing_labels"], [])
        self.assertEqual(report["invalid_or_missing_pose_frames"], [])

        context, target = dataset.get_context_target_frames(0)
        data = dataset.get_data(
            0,
            data_fields=[DF_IMAGE_RGB, DF_SEMANTIC_LABEL],
            frame_indices=context + target,
        )
        self.assertEqual(data["scene_name"], "scene0686_01")
        self.assertEqual(data["frame_ids"].tolist(), [0, 20, 10])
        self.assertEqual(
            data[DF_IMAGE_RGB].shape[-2:], data[DF_SEMANTIC_LABEL].shape[-2:]
        )


if __name__ == "__main__":
    unittest.main()
