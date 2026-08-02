# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt-conditioned ScanNet training samples and lazy image-query loading."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from tokengs.data.static.scannet import (
    DEFAULT_C3G8_PROTOCOL_PATH,
    ScanNet,
    ScanNetSensReader,
)


PromptMode = Literal["text_only", "image_only", "text_image_mixed"]


@dataclass(frozen=True)
class QueryBankEntry:
    scene: str
    raw_frame_id: int
    class_id: int
    bbox: tuple[int, int, int, int]
    mask_area: int
    image_size: tuple[int, int]
    instance_id: Optional[int]
    source_type: str

    @classmethod
    def from_dict(cls, item: dict) -> "QueryBankEntry":
        return cls(
            scene=str(item["scene"]),
            raw_frame_id=int(item["raw_frame_id"]),
            class_id=int(item["class_id"]),
            bbox=tuple(int(value) for value in item["bbox"]),
            mask_area=int(item["mask_area"]),
            image_size=tuple(int(value) for value in item["image_size"]),
            instance_id=(
                None if item.get("instance_id") is None else int(item["instance_id"])
            ),
            source_type=str(item["source_type"]),
        )


@dataclass(frozen=True)
class SmallPromptSample:
    sample_id: str
    scene: str
    input_frame_ids: tuple[int, int]
    target_frame_id: int
    class_id: int
    prompt_mode: PromptMode
    query: Optional[QueryBankEntry]

    @classmethod
    def from_dict(cls, item: dict) -> "SmallPromptSample":
        query = item.get("query")
        return cls(
            sample_id=str(item["sample_id"]),
            scene=str(item["scene"]),
            input_frame_ids=tuple(int(value) for value in item["input_frame_ids"]),
            target_frame_id=int(item["target_frame_id"]),
            class_id=int(item["class_id"]),
            prompt_mode=str(item["prompt_mode"]),
            query=None if query is None else QueryBankEntry.from_dict(query),
        )


class ScanNetPromptTrain(ScanNet):
    """Provisional non-eval ScanNet split with text and cross-scene image queries."""

    has_prompt_samples = True

    def __init__(
        self,
        root_path: str,
        label_root: str,
        train_manifest_path: Optional[str] = None,
        query_bank_path: Optional[str] = None,
        semantic_protocol_path: Optional[str] = None,
        prompt_mode: PromptMode = "text_image_mixed",
        query_image_size: Sequence[int] = (224, 224),
        prompt_image_probability: float = 0.5,
        prompt_min_target_pixels: int = 64,
        frame_stride: int = 10,
        skip_bad: bool = False,
        **kwargs,
    ):
        protocol_path = Path(semantic_protocol_path or DEFAULT_C3G8_PROTOCOL_PATH)
        with protocol_path.open(encoding="utf-8") as handle:
            protocol = yaml.safe_load(handle)
        training_cfg = protocol["training"]

        self.train_manifest_path = Path(
            train_manifest_path or training_cfg["train_manifest"]
        )
        self.query_bank_path = Path(query_bank_path or training_cfg["query_bank"])
        self.prompt_mode = prompt_mode
        if prompt_mode not in ("text_only", "image_only", "text_image_mixed"):
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
        self.query_image_size = tuple(int(value) for value in query_image_size)
        if len(self.query_image_size) != 2 or min(self.query_image_size) <= 0:
            raise ValueError("query_image_size must contain two positive integers")
        self.prompt_image_probability = float(prompt_image_probability)
        if not 0.0 <= self.prompt_image_probability <= 1.0:
            raise ValueError("prompt_image_probability must be in [0, 1]")
        self.prompt_min_target_pixels = int(prompt_min_target_pixels)

        with self.train_manifest_path.open(encoding="utf-8") as handle:
            train_manifest = json.load(handle)
        if not isinstance(train_manifest.get("provisional"), bool):
            raise ValueError("Prompt training manifest must declare provisional as a boolean")
        self.train_split_is_provisional = train_manifest["provisional"]
        self.train_split_source = str(train_manifest["source"])
        train_scenes = [str(scene) for scene in train_manifest["scenes"]]
        self.eval_scene_names = set(str(scene) for scene in train_manifest["excluded_eval_scenes"])
        leaked_scenes = sorted(set(train_scenes) & self.eval_scene_names)
        if leaked_scenes:
            raise ValueError(f"Evaluation scenes leaked into prompt training: {leaked_scenes}")

        super().__init__(
            root_path=root_path,
            label_root=label_root,
            subset=train_scenes,
            frame_stride=frame_stride,
            label_mapping="c3g8",
            semantic_protocol_path=str(protocol_path),
            skip_bad=skip_bad,
            **kwargs,
        )
        self.train_scene_names = {path.name for path in self.sample_list}
        if self.train_scene_names != set(train_scenes):
            missing = sorted(set(train_scenes) - self.train_scene_names)
            raise FileNotFoundError(f"Prompt training scenes are unavailable: {missing}")

        with self.query_bank_path.open(encoding="utf-8") as handle:
            bank = json.load(handle)
        bank_eval_scenes = set(str(scene) for scene in bank["excluded_eval_scenes"])
        if bank_eval_scenes != self.eval_scene_names:
            raise ValueError("Query bank and training manifest disagree on eval exclusions")
        entries = [QueryBankEntry.from_dict(item) for item in bank["entries"]]
        bank_leaks = sorted({entry.scene for entry in entries} & self.eval_scene_names)
        if bank_leaks:
            raise ValueError(f"Evaluation scenes leaked into query bank: {bank_leaks}")
        foreign_scenes = sorted({entry.scene for entry in entries} - self.train_scene_names)
        if foreign_scenes:
            raise ValueError(f"Query bank contains non-training scenes: {foreign_scenes}")
        if any(entry.class_id == 8 for entry in entries):
            raise ValueError("C3G8 'other' must not have image-query entries")

        self.query_entries = entries
        self.query_entries_by_class_scene: dict[int, dict[str, list[QueryBankEntry]]] = {}
        for entry in entries:
            self.query_entries_by_class_scene.setdefault(entry.class_id, {}).setdefault(
                entry.scene, []
            ).append(entry)
        self._query_reader_scene: Optional[str] = None
        self._query_reader: Optional[ScanNetSensReader] = None

    def _query_candidates_exist(self, class_id: int, target_scene: str) -> bool:
        return any(
            scene != target_scene
            for scene in self.query_entries_by_class_scene.get(class_id, {})
        )

    def _sample_query_entry(
        self, class_id: int, target_scene: str, rng: np.random.Generator
    ) -> QueryBankEntry:
        by_scene = self.query_entries_by_class_scene.get(class_id, {})
        scenes = sorted(scene for scene in by_scene if scene != target_scene)
        if not scenes:
            raise RuntimeError(
                f"No cross-scene image query for class {class_id}, target {target_scene}"
            )
        query_scene = scenes[int(rng.integers(0, len(scenes)))]
        scene_entries = by_scene[query_scene]
        return scene_entries[int(rng.integers(0, len(scene_entries)))]

    def _get_query_reader(self, scene_name: str) -> ScanNetSensReader:
        if self._query_reader_scene != scene_name or self._query_reader is None:
            scene_dir = self.root_path / scene_name
            self._query_reader = ScanNetSensReader(
                scene_dir / f"{scene_name}.sens", frame_stride=1
            )
            self._query_reader_scene = scene_name
        return self._query_reader

    @staticmethod
    def _letterbox_query(
        rgb: np.ndarray, mask: np.ndarray, output_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_h, output_w = output_size
        crop_h, crop_w = mask.shape
        scale = min(output_w / crop_w, output_h / crop_h)
        resized_w = max(1, int(round(crop_w * scale)))
        resized_h = max(1, int(round(crop_h * scale)))
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask.astype(np.float32))[None]
        rgb_tensor = F.interpolate(
            rgb_tensor[None], size=(resized_h, resized_w), mode="bicubic", align_corners=False
        )[0].clamp(0, 1)
        mask_tensor = F.interpolate(
            mask_tensor[None], size=(resized_h, resized_w), mode="nearest"
        )[0]
        rgb_output = torch.zeros((3, output_h, output_w), dtype=torch.float32)
        mask_output = torch.zeros((1, output_h, output_w), dtype=torch.float32)
        top = (output_h - resized_h) // 2
        left = (output_w - resized_w) // 2
        rgb_output[:, top : top + resized_h, left : left + resized_w] = rgb_tensor
        mask_output[:, top : top + resized_h, left : left + resized_w] = mask_tensor
        return rgb_output, mask_output

    def _load_query(self, entry: QueryBankEntry) -> tuple[torch.Tensor, torch.Tensor]:
        reader = self._get_query_reader(entry.scene)
        rgb = np.asarray(reader.read_color(entry.raw_frame_id), dtype=np.uint8).copy()
        semantic_path = (
            self.label_root / entry.scene / "label-filt" / f"{entry.raw_frame_id}.png"
        )
        semantic_raw = np.asarray(Image.open(semantic_path)).copy()
        semantic = self._map_labels(semantic_raw)
        if entry.instance_id is None:
            mask = semantic == entry.class_id
        else:
            instance_path = (
                self.label_root
                / entry.scene
                / "instance-filt"
                / f"{entry.raw_frame_id}.png"
            )
            instances = np.asarray(Image.open(instance_path))
            mask = (instances == entry.instance_id) & (semantic == entry.class_id)
        if rgb.shape[:2] != mask.shape:
            raise ValueError(
                f"Query RGB/mask mismatch for {entry.scene}/{entry.raw_frame_id}: "
                f"{rgb.shape[:2]} vs {mask.shape}"
            )
        if (rgb.shape[1], rgb.shape[0]) != entry.image_size:
            raise ValueError(
                f"Query image size disagrees with bank metadata for "
                f"{entry.scene}/{entry.raw_frame_id}"
            )
        x0, y0, x1, y1 = entry.bbox
        rgb_crop = rgb[y0:y1, x0:x1]
        mask_crop = mask[y0:y1, x0:x1]
        if not mask_crop.any():
            raise ValueError(f"Empty query mask for bank entry: {entry}")
        return self._letterbox_query(rgb_crop, mask_crop, self.query_image_size)

    def make_prompt_sample(
        self,
        semantic_label_output: torch.Tensor,
        target_scene_name: str,
        used_frame_ids: torch.Tensor,
        rng: np.random.Generator,
    ) -> dict:
        if target_scene_name in self.eval_scene_names:
            raise AssertionError(f"Evaluation scene used for training: {target_scene_name}")
        counts = torch.bincount(semantic_label_output.flatten(), minlength=9)
        if int(counts[1:9].sum()) == 0:
            raise RuntimeError(
                f"Target sample {target_scene_name} contains no non-background C3G8 pixels"
            )
        text_classes = [
            class_id
            for class_id in range(1, 9)
            if int(counts[class_id]) >= self.prompt_min_target_pixels
        ]
        if not text_classes:
            text_classes = [int(torch.argmax(counts[1:]).item()) + 1]
        image_classes = [
            class_id
            for class_id in text_classes
            if class_id != 8
            and self._query_candidates_exist(class_id, target_scene_name)
        ]

        image_attempted = self.prompt_mode == "image_only" or (
            self.prompt_mode == "text_image_mixed"
            and rng.random() < self.prompt_image_probability
        )
        use_image = image_attempted and bool(image_classes)
        if self.prompt_mode == "image_only" and not image_classes:
            raise RuntimeError(
                f"No eligible image-query class in target sample {target_scene_name}"
            )
        eligible_classes = image_classes if use_image else text_classes
        prompt_class_id = eligible_classes[int(rng.integers(0, len(eligible_classes)))]
        negative_ids = [class_id for class_id in range(1, 9) if class_id != prompt_class_id]
        negative_class_id = negative_ids[int(rng.integers(0, len(negative_ids)))]
        class_names = self.semantic_class_names
        output = {
            "prompt_mode": self.prompt_mode,
            "prompt_type": "image" if use_image else "text",
            "prompt_class_id": torch.tensor(prompt_class_id, dtype=torch.long),
            "positive_text_prompt": class_names[prompt_class_id - 1],
            "negative_text_prompt": class_names[negative_class_id - 1],
            "binary_mask_output": (semantic_label_output == prompt_class_id).float(),
            "image_query_attempted": torch.tensor(image_attempted, dtype=torch.bool),
            "has_image_query": torch.tensor(use_image, dtype=torch.bool),
            "query_image": torch.zeros((3, *self.query_image_size), dtype=torch.float32),
            "query_mask": torch.zeros((1, *self.query_image_size), dtype=torch.float32),
            "query_scene_name": "",
            "query_frame_id": torch.tensor(-1, dtype=torch.long),
            "query_class_id": torch.tensor(prompt_class_id, dtype=torch.long),
        }
        if not use_image:
            return output

        entry = self._sample_query_entry(prompt_class_id, target_scene_name, rng)
        if entry.scene == target_scene_name:
            raise AssertionError("Image query must come from a different scene")
        if entry.scene in self.eval_scene_names:
            raise AssertionError("Image query must not come from an eval scene")
        if entry.class_id != prompt_class_id:
            raise AssertionError("Image-query class must equal prompt class")
        if entry.scene == target_scene_name and entry.raw_frame_id in used_frame_ids.tolist():
            raise AssertionError("Image query reused an input/target frame")
        query_image, query_mask = self._load_query(entry)
        output.update(
            {
                "query_image": query_image,
                "query_mask": query_mask,
                "query_scene_name": entry.scene,
                "query_frame_id": torch.tensor(entry.raw_frame_id, dtype=torch.long),
                "query_class_id": torch.tensor(entry.class_id, dtype=torch.long),
            }
        )
        return output


class ScanNetPromptSmall(ScanNetPromptTrain):
    """Fixed, balanced 64/8-scene prompt split driven entirely by a manifest."""

    has_explicit_split = True

    def __init__(
        self,
        root_path: str,
        label_root: str,
        small_manifest_path: str,
        split: Literal["train", "validation"] = "train",
        prompt_mode: str = "manifest",
        frame_stride: int = 1,
        **kwargs,
    ):
        if prompt_mode != "manifest":
            raise ValueError("ScanNetPromptSmall requires prompt_mode='manifest'")
        if split not in ("train", "validation"):
            raise ValueError("ScanNetPromptSmall split must be train or validation")
        if int(frame_stride) != 1:
            raise ValueError("ScanNetPromptSmall requires frame_stride=1")

        super().__init__(
            root_path=root_path,
            label_root=label_root,
            prompt_mode="text_image_mixed",
            frame_stride=1,
            **kwargs,
        )
        self.prompt_mode = "manifest"
        self.small_manifest_path = Path(small_manifest_path)
        self.split = split
        manifest = json.loads(self.small_manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("provisional", False):
            raise ValueError("Small ScanNet prompt manifest must be marked provisional")
        if set(manifest["excluded_eval_scenes"]) != self.eval_scene_names:
            raise ValueError("Small manifest and provisional split disagree on eval scenes")

        train_scenes = set(str(value) for value in manifest["train_scenes"])
        validation_scenes = set(str(value) for value in manifest["validation_scenes"])
        if train_scenes & validation_scenes:
            raise ValueError("Small train and validation scenes overlap")
        if (train_scenes | validation_scenes) & self.eval_scene_names:
            raise ValueError("C3G8 eval scene leaked into small prompt split")
        selected_scenes = train_scenes if split == "train" else validation_scenes

        all_scene_dirs = {path.name: path for path in self.sample_list}
        missing_scenes = sorted(selected_scenes - set(all_scene_dirs))
        if missing_scenes:
            raise FileNotFoundError(f"Small prompt scenes are unavailable: {missing_scenes}")
        self.scene_dirs = {scene: all_scene_dirs[scene] for scene in selected_scenes}
        self.split_scene_names = selected_scenes
        sample_key = "train_samples" if split == "train" else "validation_samples"
        samples = [SmallPromptSample.from_dict(item) for item in manifest[sample_key]]
        self._validate_samples(samples, train_scenes)
        if {sample.scene for sample in samples} != selected_scenes:
            raise ValueError(f"Not every {split} scene is represented by a sample")
        self.sample_list = samples
        self._cached_scene_name: Optional[str] = None
        self._active_sample: Optional[SmallPromptSample] = None

    def _validate_samples(
        self, samples: list[SmallPromptSample], train_scenes: set[str]
    ) -> None:
        for sample in samples:
            if sample.scene not in self.split_scene_names:
                raise ValueError(f"Sample scene is outside {self.split}: {sample.sample_id}")
            if len(set(sample.input_frame_ids)) != 2:
                raise ValueError(f"Input frames must be distinct: {sample.sample_id}")
            if sample.target_frame_id in sample.input_frame_ids:
                raise ValueError(f"Input and target frames overlap: {sample.sample_id}")
            if sample.class_id not in range(1, 9):
                raise ValueError(f"Invalid C3G8 class: {sample.sample_id}")
            if sample.prompt_mode not in (
                "text_only",
                "image_only",
                "text_image_mixed",
            ):
                raise ValueError(f"Invalid prompt mode: {sample.sample_id}")
            if sample.class_id == 8 and sample.prompt_mode != "text_only":
                raise ValueError(f"Other must use text_only: {sample.sample_id}")
            needs_query = sample.prompt_mode != "text_only"
            if needs_query != (sample.query is not None):
                raise ValueError(f"Prompt/query mismatch: {sample.sample_id}")
            if sample.query is not None:
                if sample.query.scene == sample.scene:
                    raise ValueError(f"Query scene equals target scene: {sample.sample_id}")
                if sample.query.scene not in train_scenes:
                    raise ValueError(f"Query must come from the small train split: {sample.sample_id}")
                if sample.query.scene in self.eval_scene_names:
                    raise ValueError(f"Eval scene used as query: {sample.sample_id}")
                if sample.query.class_id != sample.class_id:
                    raise ValueError(f"Query class mismatch: {sample.sample_id}")

    def _get_reader(self, idx: int) -> ScanNetSensReader:
        sample = self.sample_list[idx]
        if self._cached_scene_name != sample.scene or self._cached_reader is None:
            scene_dir = self.scene_dirs[sample.scene]
            self._cached_reader = ScanNetSensReader(
                scene_dir / f"{sample.scene}.sens", frame_stride=1
            )
            self._cached_scene_name = sample.scene
        self._active_sample = sample
        return self._cached_reader

    def get_context_target_frames(self, idx: int) -> tuple[list[int], list[int]]:
        sample = self.sample_list[idx]
        reader = self._get_reader(idx)
        raw_to_logical = {
            raw_frame_id: logical_id
            for logical_id, raw_frame_id in enumerate(reader.frame_ids)
        }
        requested = (*sample.input_frame_ids, sample.target_frame_id)
        unavailable = [value for value in requested if value not in raw_to_logical]
        if unavailable:
            raise ValueError(
                f"Small prompt frames have invalid poses for {sample.sample_id}: {unavailable}"
            )
        return (
            [raw_to_logical[value] for value in sample.input_frame_ids],
            [raw_to_logical[sample.target_frame_id]],
        )

    def make_prompt_sample(
        self,
        semantic_label_output: torch.Tensor,
        target_scene_name: str,
        used_frame_ids: torch.Tensor,
        rng: np.random.Generator,
    ) -> dict:
        del rng
        sample = self._active_sample
        if sample is None or sample.scene != target_scene_name:
            raise RuntimeError("Small prompt sample state is inconsistent")
        expected_frames = (*sample.input_frame_ids, sample.target_frame_id)
        if tuple(int(value) for value in used_frame_ids.tolist()) != expected_frames:
            raise RuntimeError(f"Manifest frame mismatch for {sample.sample_id}")
        class_names = self.semantic_class_names
        negative_class_id = sample.class_id % 8 + 1
        output = {
            "sample_id": sample.sample_id,
            "prompt_mode": sample.prompt_mode,
            "prompt_type": {
                "text_only": "text",
                "image_only": "image",
                "text_image_mixed": "mixed",
            }[sample.prompt_mode],
            "prompt_class_id": torch.tensor(sample.class_id, dtype=torch.long),
            "positive_text_prompt": class_names[sample.class_id - 1],
            "negative_text_prompt": class_names[negative_class_id - 1],
            "binary_mask_output": (semantic_label_output == sample.class_id).float(),
            "image_query_attempted": torch.tensor(
                sample.query is not None, dtype=torch.bool
            ),
            "has_image_query": torch.tensor(sample.query is not None, dtype=torch.bool),
            "query_image": torch.zeros((3, *self.query_image_size), dtype=torch.float32),
            "query_mask": torch.zeros((1, *self.query_image_size), dtype=torch.float32),
            "query_scene_name": "",
            "query_frame_id": torch.tensor(-1, dtype=torch.long),
            "query_class_id": torch.tensor(sample.class_id, dtype=torch.long),
        }
        if sample.query is None:
            return output
        query_image, query_mask = self._load_query(sample.query)
        output.update(
            {
                "query_image": query_image,
                "query_mask": query_mask,
                "query_scene_name": sample.query.scene,
                "query_frame_id": torch.tensor(
                    sample.query.raw_frame_id, dtype=torch.long
                ),
                "query_class_id": torch.tensor(
                    sample.query.class_id, dtype=torch.long
                ),
            }
        )
        return output


class ScanNetSemanticSmall(ScanNet):
    """Fixed 64/8-scene semantic split without prompt or query sampling.

    The manifest rows are intentionally preserved so V2 sees exactly the same
    scene/frame distribution as the prompt baselines. Every row exposes the
    complete C3G8 target label; class_id, prompt_mode, and query metadata are
    ignored.
    """

    has_explicit_split = True
    has_prompt_samples = False

    def __init__(
        self,
        root_path: str,
        label_root: str,
        small_manifest_path: str,
        semantic_protocol_path: Optional[str] = None,
        split: Literal["train", "validation"] = "train",
        frame_stride: int = 1,
        **kwargs,
    ):
        if split not in ("train", "validation"):
            raise ValueError("ScanNetSemanticSmall split must be train or validation")
        if int(frame_stride) != 1:
            raise ValueError("ScanNetSemanticSmall requires frame_stride=1")

        self.small_manifest_path = Path(small_manifest_path)
        self.split = split
        manifest = json.loads(self.small_manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("provisional", False):
            raise ValueError("Small ScanNet semantic manifest must be provisional")

        self.eval_scene_names = set(str(value) for value in manifest["excluded_eval_scenes"])
        train_scenes = set(str(value) for value in manifest["train_scenes"])
        validation_scenes = set(str(value) for value in manifest["validation_scenes"])
        if train_scenes & validation_scenes:
            raise ValueError("Small semantic train and validation scenes overlap")
        if (train_scenes | validation_scenes) & self.eval_scene_names:
            raise ValueError("C3G8 eval scene leaked into the small semantic split")
        selected_scenes = train_scenes if split == "train" else validation_scenes

        protocol_path = Path(semantic_protocol_path or DEFAULT_C3G8_PROTOCOL_PATH)
        super().__init__(
            root_path=root_path,
            label_root=label_root,
            subset=sorted(selected_scenes),
            frame_stride=1,
            label_mapping="c3g8",
            semantic_protocol_path=str(protocol_path),
            **kwargs,
        )
        all_scene_dirs = {path.name: path for path in self.sample_list}
        missing_scenes = sorted(selected_scenes - set(all_scene_dirs))
        if missing_scenes:
            raise FileNotFoundError(f"Small semantic scenes are unavailable: {missing_scenes}")
        self.scene_dirs = {scene: all_scene_dirs[scene] for scene in selected_scenes}
        self.split_scene_names = selected_scenes

        sample_key = "train_samples" if split == "train" else "validation_samples"
        samples = [SmallPromptSample.from_dict(item) for item in manifest[sample_key]]
        for sample in samples:
            if sample.scene not in selected_scenes:
                raise ValueError(f"Sample scene is outside {split}: {sample.sample_id}")
            if len(set(sample.input_frame_ids)) != 2:
                raise ValueError(f"Input frames must be distinct: {sample.sample_id}")
            if sample.target_frame_id in sample.input_frame_ids:
                raise ValueError(f"Input and target frames overlap: {sample.sample_id}")
        if {sample.scene for sample in samples} != selected_scenes:
            raise ValueError(f"Not every {split} scene is represented by a sample")

        self.sample_list = samples
        self._cached_scene_name: Optional[str] = None

    def _get_reader(self, idx: int) -> ScanNetSensReader:
        sample = self.sample_list[idx]
        if self._cached_scene_name != sample.scene or self._cached_reader is None:
            scene_dir = self.scene_dirs[sample.scene]
            self._cached_reader = ScanNetSensReader(
                scene_dir / f"{sample.scene}.sens", frame_stride=1
            )
            self._cached_scene_name = sample.scene
        return self._cached_reader

    def get_context_target_frames(self, idx: int) -> tuple[list[int], list[int]]:
        sample = self.sample_list[idx]
        reader = self._get_reader(idx)
        raw_to_logical = {
            raw_frame_id: logical_id
            for logical_id, raw_frame_id in enumerate(reader.frame_ids)
        }
        requested = (*sample.input_frame_ids, sample.target_frame_id)
        unavailable = [value for value in requested if value not in raw_to_logical]
        if unavailable:
            raise ValueError(
                f"Small semantic frames have invalid poses for {sample.sample_id}: "
                f"{unavailable}"
            )
        return (
            [raw_to_logical[value] for value in sample.input_frame_ids],
            [raw_to_logical[sample.target_frame_id]],
        )
