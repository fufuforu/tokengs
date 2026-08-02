import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tokengs.data.datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_FRAME_IDS,
    DF_IMAGE_RGB,
    DF_SEMANTIC_LABEL,
)
from tokengs.data.provider import Provider
from tokengs.data.registry import dataset_registry
from tokengs.data.static.scannet import ScanNet, ScanNetC3G8Eval
from tokengs.data.static.scannet_prompt import ScanNetPromptTrain
from tokengs.options import Options


def _jpeg_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=100)
    return buffer.getvalue()


def _write_sens(path: Path, poses: list[np.ndarray], height: int, width: int) -> None:
    intrinsic = np.eye(4, dtype=np.float32)
    intrinsic[0, 0] = 100.0
    intrinsic[1, 1] = 101.0
    intrinsic[0, 2] = width / 2
    intrinsic[1, 2] = height / 2
    identity = np.eye(4, dtype=np.float32)
    sensor_name = b"synthetic"
    with path.open("wb") as handle:
        handle.write(struct.pack("<I", 4))
        handle.write(struct.pack("<Q", len(sensor_name)))
        handle.write(sensor_name)
        for matrix in (intrinsic, identity, intrinsic, identity):
            handle.write(matrix.astype("<f4").tobytes())
        handle.write(struct.pack("<ii", 2, 1))
        handle.write(struct.pack("<IIII", width, height, width, height))
        handle.write(struct.pack("<f", 1000.0))
        handle.write(struct.pack("<Q", len(poses)))
        for frame_id, pose in enumerate(poses):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[..., 0] = 40 + frame_id * 20
            color = _jpeg_bytes(image)
            handle.write(pose.astype("<f4").tobytes())
            handle.write(struct.pack("<QQQQ", frame_id, frame_id, len(color), 0))
            handle.write(color)


class TestScanNetDataset(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        base = Path(self.tempdir.name)
        self.scan_root = base / "ScanNet" / "scans"
        self.label_root = base / "labels"
        self.scene = "scene0000_00"
        scene_dir = self.scan_root / self.scene
        label_dir = self.label_root / self.scene / "label-filt"
        scene_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)

        poses = [np.eye(4, dtype=np.float32) for _ in range(4)]
        poses[1][0, 0] = np.inf
        poses[2][0, 3] = 0.1
        poses[3][0, 3] = 0.2
        _write_sens(scene_dir / f"{self.scene}.sens", poses, height=6, width=8)
        for frame_id in range(4):
            label = np.ones((6, 8), dtype=np.uint16) * (frame_id + 1)
            label[:, 0] = 99
            label[:, -1] = 99
            Image.fromarray(label).save(label_dir / f"{frame_id}.png")

    def tearDown(self):
        self.tempdir.cleanup()

    def _dataset(self):
        return ScanNet(
            root_path=str(self.scan_root),
            label_root=str(self.label_root),
            subset=self.scene,
            frame_stride=1,
        )

    def test_filters_invalid_pose_and_returns_raw_frame_ids(self):
        dataset = self._dataset()
        self.assertEqual(dataset.count_frames(0), 3)
        self.assertEqual(dataset.get_reader_stats(0)["invalid_pose_frame_ids"], [1])
        data = dataset.get_data(
            0,
            data_fields=[
                DF_IMAGE_RGB,
                DF_CAMERA_C2W_TRANSFORM,
                DF_CAMERA_INTRINSICS,
                DF_SEMANTIC_LABEL,
            ],
            frame_indices=[0, 1, 2],
        )
        self.assertEqual(data[DF_IMAGE_RGB].shape, (3, 3, 6, 8))
        self.assertEqual(data[DF_SEMANTIC_LABEL].shape, (3, 6, 8))
        self.assertEqual(data[DF_FRAME_IDS].tolist(), [0, 2, 3])
        self.assertEqual(data["scene_name"], self.scene)

    def test_provider_uses_shared_crop_and_nearest_label_resize(self):
        key = "_test_scannet"
        dataset_registry[key] = {
            "cls": ScanNet,
            "kwargs": {
                "root_path": str(self.scan_root),
                "label_root": str(self.label_root),
                "subset": self.scene,
                "frame_stride": 1,
            },
            "scene_scale": 1.0,
            "min_gap": 2,
            "max_gap": 2,
        }
        try:
            opt = Options(
                evaluating=True,
                img_size=(4, 4),
                num_input_views=2,
                num_views=3,
                batch_size=1,
                num_workers=0,
                random_reflect=True,
            )
            provider = Provider(key, opt, training=True)
            item = provider.get_item(0)
        finally:
            dataset_registry.pop(key)

        self.assertFalse(provider.random_reflect)
        self.assertEqual(item["semantic_label_all"].shape, (3, 4, 4))
        self.assertEqual(item["semantic_label_input"].shape, (2, 4, 4))
        self.assertEqual(item["semantic_label_output"].shape, (1, 4, 4))
        self.assertNotIn(99, torch.unique(item["semantic_label_all"]).tolist())
        self.assertEqual(item["frame_ids"].shape, (3,))
        self.assertEqual(item["scene_name"], self.scene)

    def test_c3g8_raw_id_mapping(self):
        dataset = ScanNet(
            root_path=str(self.scan_root),
            label_root=str(self.label_root),
            subset=self.scene,
            frame_stride=1,
            label_mapping="c3g8",
        )
        raw = np.asarray(
            [[0, 1, 3, 41, 2, 4, 6, 11, 99, 65535]], dtype=np.int64
        )
        mapped = dataset._map_labels(raw)
        self.assertEqual(mapped.tolist(), [[0, 1, 2, 3, 4, 5, 6, 7, 8, 8]])
        self.assertGreaterEqual(int(mapped.min()), 0)
        self.assertLessEqual(int(mapped.max()), 8)

    def test_c3g_manifest_context_target_and_provider_batch(self):
        manifest_path = Path(self.tempdir.name) / "selected_seqs_test.json"
        manifest_path.write_text(
            json.dumps({self.scene: ["0", "2", "3"]}), encoding="utf-8"
        )
        dataset = ScanNetC3G8Eval(
            root_path=str(self.scan_root),
            label_root=str(self.label_root),
            manifest_path=str(manifest_path),
            excluded_scenes=(),
            llff_hold=8,
            test_ids=(1,),
        )
        self.assertEqual(len(dataset), 1)
        sample = dataset.sample_list[0]
        self.assertEqual(sample.context_raw_frame_ids, (0, 3))
        self.assertEqual(sample.target_raw_frame_id, 2)
        context, target = dataset.get_context_target_frames(0)
        data = dataset.get_data(
            0,
            data_fields=[DF_IMAGE_RGB, DF_SEMANTIC_LABEL],
            frame_indices=context + target,
        )
        self.assertEqual(data[DF_FRAME_IDS].tolist(), [0, 3, 2])
        self.assertEqual(
            data[DF_IMAGE_RGB].shape[-2:], data[DF_SEMANTIC_LABEL].shape[-2:]
        )

        key = "_test_scannet_c3g8_eval"
        dataset_registry[key] = {
            "cls": ScanNetC3G8Eval,
            "kwargs": {
                "root_path": str(self.scan_root),
                "label_root": str(self.label_root),
                "manifest_path": str(manifest_path),
                "excluded_scenes": (),
                "llff_hold": 8,
                "test_ids": (1,),
            },
            "scene_scale": 1.0,
            "min_gap": 0,
            "max_gap": 0,
        }
        try:
            provider = Provider(
                key,
                Options(
                    evaluating=True,
                    img_size=(4, 4),
                    num_input_views=2,
                    num_views=3,
                    batch_size=1,
                    num_workers=0,
                    random_reflect=False,
                ),
                training=False,
            )
            item = provider.get_item(0)
        finally:
            dataset_registry.pop(key)
        self.assertEqual(item["frame_ids"].tolist(), [0, 3, 2])
        self.assertEqual(item["semantic_label_all"].shape, (3, 4, 4))
        self.assertTrue(
            set(torch.unique(item["semantic_label_all"]).tolist()) <= set(range(9))
        )

    def test_prompt_modes_and_cross_scene_query(self):
        other_scene = "scene0001_00"
        other_scene_dir = self.scan_root / other_scene
        other_label_root = self.label_root / other_scene
        other_label_dir = other_label_root / "label-filt"
        other_instance_dir = other_label_root / "instance-filt"
        other_scene_dir.mkdir(parents=True)
        other_label_dir.mkdir(parents=True)
        other_instance_dir.mkdir(parents=True)
        poses = [np.eye(4, dtype=np.float32) for _ in range(4)]
        _write_sens(other_scene_dir / f"{other_scene}.sens", poses, height=6, width=8)
        raw_ids = [1, 2, 3, 4]
        for frame_id, raw_id in enumerate(raw_ids):
            label = np.full((6, 8), raw_id, dtype=np.uint16)
            label[:, 0] = 99
            label[:, -1] = 99
            instance = np.ones((6, 8), dtype=np.uint8)
            instance[:, 0] = 0
            instance[:, -1] = 0
            Image.fromarray(label).save(other_label_dir / f"{frame_id}.png")
            Image.fromarray(instance).save(other_instance_dir / f"{frame_id}.png")

        manifest_path = Path(self.tempdir.name) / "prompt_train.json"
        bank_path = Path(self.tempdir.name) / "query_bank.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "provisional": True,
                    "source": "synthetic",
                    "excluded_eval_scenes": ["scene9999_00"],
                    "scenes": [self.scene, other_scene],
                }
            ),
            encoding="utf-8",
        )
        class_to_frame = {1: 0, 4: 1, 2: 2, 5: 3}
        entries = []
        for class_id, frame_id in class_to_frame.items():
            entries.append(
                {
                    "scene": other_scene,
                    "raw_frame_id": frame_id,
                    "class_id": class_id,
                    "bbox": [1, 0, 7, 6],
                    "mask_area": 36,
                    "image_size": [8, 6],
                    "instance_id": 1 if class_id in (4, 5) else None,
                    "source_type": "instance" if class_id in (4, 5) else "semantic",
                }
            )
        bank_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "metadata_only": True,
                    "excluded_eval_scenes": ["scene9999_00"],
                    "entries": entries,
                }
            ),
            encoding="utf-8",
        )

        common_kwargs = {
            "root_path": str(self.scan_root),
            "label_root": str(self.label_root),
            "train_manifest_path": str(manifest_path),
            "query_bank_path": str(bank_path),
            "frame_stride": 1,
            "query_image_size": (8, 8),
            "prompt_min_target_pixels": 1,
        }
        image_dataset = ScanNetPromptTrain(prompt_mode="image_only", **common_kwargs)
        image_prompt = image_dataset.make_prompt_sample(
            semantic_label_output=torch.ones((1, 4, 4), dtype=torch.long),
            target_scene_name=self.scene,
            used_frame_ids=torch.tensor([0, 2, 3]),
            rng=np.random.default_rng(1),
        )
        self.assertTrue(image_prompt["has_image_query"])
        self.assertEqual(image_prompt["prompt_class_id"].item(), 1)
        self.assertEqual(image_prompt["query_class_id"].item(), 1)
        self.assertEqual(image_prompt["query_scene_name"], other_scene)
        self.assertEqual(image_prompt["query_image"].shape, (3, 8, 8))
        self.assertEqual(image_prompt["query_mask"].shape, (1, 8, 8))

        text_dataset = ScanNetPromptTrain(prompt_mode="text_only", **common_kwargs)
        text_prompt = text_dataset.make_prompt_sample(
            semantic_label_output=torch.full((1, 4, 4), 8, dtype=torch.long),
            target_scene_name=self.scene,
            used_frame_ids=torch.tensor([0, 2, 3]),
            rng=np.random.default_rng(1),
        )
        self.assertFalse(text_prompt["has_image_query"])
        self.assertEqual(text_prompt["prompt_class_id"].item(), 8)
        self.assertEqual(text_prompt["positive_text_prompt"], "other")

        with self.assertRaisesRegex(RuntimeError, "no non-background C3G8 pixels"):
            text_dataset.make_prompt_sample(
                semantic_label_output=torch.zeros((1, 4, 4), dtype=torch.long),
                target_scene_name=self.scene,
                used_frame_ids=torch.tensor([0, 2, 3]),
                rng=np.random.default_rng(1),
            )

        mixed_dataset = ScanNetPromptTrain(
            prompt_mode="text_image_mixed",
            prompt_image_probability=1.0,
            **common_kwargs,
        )
        mixed_prompt = mixed_dataset.make_prompt_sample(
            semantic_label_output=torch.ones((1, 4, 4), dtype=torch.long),
            target_scene_name=self.scene,
            used_frame_ids=torch.tensor([0, 2, 3]),
            rng=np.random.default_rng(1),
        )
        self.assertEqual(mixed_prompt["prompt_type"], "image")


if __name__ == "__main__":
    unittest.main()
