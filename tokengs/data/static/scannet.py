# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ScanNet RGB, camera, and 2D semantic-label loading from raw .sens files."""

import csv
import json
import struct
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import yaml
from PIL import Image

from tokengs.data.datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_FRAME_IDS,
    DF_IMAGE_RGB,
    DF_SCENE_NAME,
    DF_SEMANTIC_LABEL,
    DF_INSTANCE_LABEL,
)


@dataclass
class SensFrameMeta:
    camera_to_world: np.ndarray
    color_size_bytes: int
    color_offset: int


@dataclass(frozen=True)
class C3GEvalSample:
    scene_name: str
    context_raw_frame_ids: tuple[int, int]
    target_raw_frame_id: int
    target_manifest_index: int


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_C3G8_PROTOCOL_PATH = (
    _REPO_ROOT / "configs" / "semantic" / "scannet_c3g8.yaml"
)


class ScanNetSensReader:
    """Index a .sens file once and lazily decode selected JPEG frames."""

    def __init__(self, sens_path: str | Path, frame_stride: int = 1):
        self.sens_path = Path(sens_path)
        self.scene_name = self.sens_path.stem
        self.frame_stride = int(frame_stride)
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")

        self.frames_meta: List[SensFrameMeta] = []
        self.invalid_pose_frame_ids: List[int] = []
        self.frame_ids: List[int] = []
        self.intrinsic_color: np.ndarray
        self.num_raw_frames = 0
        self.color_width = 0
        self.color_height = 0
        self._parse_metadata()

    @staticmethod
    def _read(fmt: str, handle):
        size = struct.calcsize(fmt)
        data = handle.read(size)
        if len(data) != size:
            raise EOFError("Unexpected end of .sens file")
        return struct.unpack(fmt, data)

    @classmethod
    def _read_mat4(cls, handle) -> np.ndarray:
        return np.asarray(cls._read("<16f", handle), dtype=np.float32).reshape(4, 4)

    @staticmethod
    def _pose_is_valid(pose: np.ndarray) -> bool:
        if not np.isfinite(pose).all():
            return False
        try:
            return abs(float(np.linalg.det(pose))) >= 1e-8
        except np.linalg.LinAlgError:
            return False

    def _parse_metadata(self) -> None:
        with self.sens_path.open("rb") as handle:
            (version,) = self._read("<I", handle)
            if version != 4:
                raise ValueError(f"Unsupported .sens version {version}: {self.sens_path}")

            (name_length,) = self._read("<Q", handle)
            handle.seek(name_length, 1)
            self.intrinsic_color = self._read_mat4(handle)
            self._read_mat4(handle)  # extrinsic_color
            self._read_mat4(handle)  # intrinsic_depth
            self._read_mat4(handle)  # extrinsic_depth
            color_compression, _depth_compression = self._read("<ii", handle)
            if color_compression not in (1, 2):
                raise ValueError(
                    f"Unsupported ScanNet color compression {color_compression}: {self.sens_path}"
                )
            (
                self.color_width,
                self.color_height,
                _depth_width,
                _depth_height,
            ) = self._read("<IIII", handle)
            self._read("<f", handle)  # depth_shift
            (self.num_raw_frames,) = self._read("<Q", handle)

            for raw_frame_id in range(self.num_raw_frames):
                camera_to_world = self._read_mat4(handle)
                self._read("<QQ", handle)  # color/depth timestamps
                color_size, depth_size = self._read("<QQ", handle)
                color_offset = handle.tell()
                handle.seek(color_size + depth_size, 1)
                self.frames_meta.append(
                    SensFrameMeta(camera_to_world, int(color_size), color_offset)
                )
                if not self._pose_is_valid(camera_to_world):
                    self.invalid_pose_frame_ids.append(raw_frame_id)

        invalid = set(self.invalid_pose_frame_ids)
        valid_ids = [i for i in range(self.num_raw_frames) if i not in invalid]
        self.frame_ids = valid_ids[:: self.frame_stride]
        if not self.frame_ids:
            raise RuntimeError(f"No valid frames found in {self.sens_path}")

    def __len__(self) -> int:
        return len(self.frame_ids)

    def read_color(self, raw_frame_id: int) -> Image.Image:
        meta = self.frames_meta[raw_frame_id]
        with self.sens_path.open("rb") as handle:
            handle.seek(meta.color_offset)
            payload = handle.read(meta.color_size_bytes)
        with Image.open(BytesIO(payload)) as image:
            return image.convert("RGB")

    def get_c2w(self, raw_frame_id: int) -> torch.Tensor:
        return torch.from_numpy(
            self.frames_meta[raw_frame_id].camera_to_world.copy()
        ).float()

    @property
    def intrinsics(self) -> torch.Tensor:
        matrix = self.intrinsic_color
        return torch.tensor(
            [matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]],
            dtype=torch.float32,
        )

    def stats(self) -> dict:
        return {
            "scene_name": self.scene_name,
            "raw_frames": self.num_raw_frames,
            "invalid_pose_frames": len(self.invalid_pose_frame_ids),
            "invalid_pose_frame_ids": list(self.invalid_pose_frame_ids),
            "valid_frames_before_stride": self.num_raw_frames
            - len(self.invalid_pose_frame_ids),
            "usable_frames": len(self.frame_ids),
            "frame_stride": self.frame_stride,
        }


class ScanNet:
    """DL3DV-style static dataset backed by ScanNet .sens scenes."""

    is_static = True
    has_semantic_labels = True
    has_instance_labels = False
    disable_random_reflect = True

    def __init__(
        self,
        root_path: str,
        label_root: str,
        subset: Optional[Sequence[str] | str] = None,
        frame_stride: int = 10,
        label_mapping: str = "raw",
        label_mapping_tsv: Optional[str] = None,
        semantic_protocol_path: Optional[str] = None,
        skip_bad: bool = True,
        **_kwargs,
    ):
        root = Path(root_path)
        self.root_path = root / "scans" if (root / "scans").is_dir() else root
        self.label_root = Path(label_root)
        self.frame_stride = int(frame_stride)
        self.label_mapping = label_mapping
        self.skip_bad = bool(skip_bad)
        self.sample_list = self._collect_scenes(subset)
        self._cached_idx: Optional[int] = None
        self._cached_reader: Optional[ScanNetSensReader] = None

        if label_mapping not in ("raw", "nyu40", "c3g8"):
            raise ValueError("label_mapping must be 'raw', 'nyu40', or 'c3g8'")
        self._label_lut = None
        self._unknown_label_id = 0
        self.semantic_class_names = None
        if label_mapping == "nyu40":
            mapping_path = (
                Path(label_mapping_tsv)
                if label_mapping_tsv is not None
                else self.root_path.parent / "scannetv2-labels.combined.tsv"
            )
            self._label_lut = self._load_nyu40_lut(mapping_path)
        elif label_mapping == "c3g8":
            protocol_path = (
                Path(semantic_protocol_path)
                if semantic_protocol_path is not None
                else DEFAULT_C3G8_PROTOCOL_PATH
            )
            (
                self._label_lut,
                self._unknown_label_id,
                self.semantic_class_names,
            ) = self._load_c3g8_protocol(protocol_path)

    def _collect_scenes(self, subset: Optional[Sequence[str] | str]) -> list[Path]:
        if subset is None or subset == "all":
            scene_names = sorted(
                path.name
                for path in self.root_path.iterdir()
                if path.is_dir() and path.name.startswith("scene")
            )
        elif isinstance(subset, str):
            scene_names = [name.strip() for name in subset.split(",") if name.strip()]
        else:
            scene_names = list(subset)

        scenes = []
        for scene_name in scene_names:
            scene_dir = self.root_path / scene_name
            sens_path = scene_dir / f"{scene_name}.sens"
            semantic_dir = self.label_root / scene_name / "label-filt"
            if sens_path.is_file() and semantic_dir.is_dir():
                scenes.append(scene_dir)
            elif not self.skip_bad:
                raise FileNotFoundError(
                    f"Missing .sens or semantic labels for ScanNet scene {scene_name}"
                )
        if not scenes:
            raise RuntimeError(f"No labeled ScanNet scenes found under {self.root_path}")
        return scenes

    @staticmethod
    def _load_nyu40_lut(path: Path) -> np.ndarray:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        max_id = max(int(row["id"]) for row in rows)
        lut = np.zeros(max_id + 1, dtype=np.int64)
        for row in rows:
            lut[int(row["id"])] = int(row["nyu40id"])
        return lut

    @staticmethod
    def _load_c3g8_protocol(path: Path) -> tuple[np.ndarray, int, tuple[str, ...]]:
        with path.open(encoding="utf-8") as handle:
            protocol = yaml.safe_load(handle)

        classes = sorted(protocol["classes"], key=lambda item: int(item["train_id"]))
        train_ids = [int(item["train_id"]) for item in classes]
        if train_ids != list(range(1, 9)):
            raise ValueError(f"C3G8 train IDs must be 1..8: {path}")
        if [item["name"] for item in classes] != list(protocol["prompts"]):
            raise ValueError(f"C3G8 class order and prompts disagree: {path}")
        if int(protocol["ignore_index"]) != 0:
            raise ValueError(f"C3G8 requires ignore_index=0: {path}")

        fallback = int(protocol["unknown_nonzero_train_id"])
        raw_assignments = {}
        for item in classes:
            train_id = int(item["train_id"])
            for raw_id_value in item.get("raw_ids", []):
                raw_id = int(raw_id_value)
                previous = raw_assignments.setdefault(raw_id, train_id)
                if previous != train_id:
                    raise ValueError(f"Raw ID {raw_id} has multiple C3G8 mappings: {path}")

        max_raw_id = max(raw_assignments, default=0)
        lut = np.full(max_raw_id + 1, fallback, dtype=np.int64)
        lut[0] = 0
        for raw_id, train_id in raw_assignments.items():
            lut[raw_id] = train_id
        return lut, fallback, tuple(item["name"] for item in classes)

    def __len__(self) -> int:
        return len(self.sample_list)

    def _get_reader(self, idx: int) -> ScanNetSensReader:
        if self._cached_idx != idx or self._cached_reader is None:
            scene_dir = self.sample_list[idx]
            self._cached_reader = ScanNetSensReader(
                scene_dir / f"{scene_dir.name}.sens", frame_stride=self.frame_stride
            )
            self._cached_idx = idx
        return self._cached_reader

    def count_frames(self, idx: int) -> int:
        return len(self._get_reader(idx))

    def count_cameras(self, _idx: int) -> int:
        return 1

    def get_reader_stats(self, idx: int) -> dict:
        return self._get_reader(idx).stats()

    def _map_labels(self, labels: np.ndarray) -> np.ndarray:
        if self._label_lut is None:
            return labels.astype(np.int64, copy=False)
        mapped = np.full(labels.shape, self._unknown_label_id, dtype=np.int64)
        valid = (labels >= 0) & (labels < len(self._label_lut))
        mapped[valid] = self._label_lut[labels[valid]]
        return mapped

    def get_data(
        self,
        idx: int,
        data_fields: List[str],
        frame_indices=None,
        view_indices=None,
        camera_convention: str = "opencv",
    ) -> dict:
        del view_indices
        if camera_convention != "opencv":
            raise ValueError("ScanNet exposes raw OpenCV-style c2w poses")

        reader = self._get_reader(idx)
        logical_ids = range(len(reader)) if frame_indices is None else frame_indices
        logical_ids = [int(i) for i in logical_ids]
        if not logical_ids or any(i < 0 or i >= len(reader) for i in logical_ids):
            raise IndexError(f"Invalid logical frame indices for {reader.scene_name}: {logical_ids}")
        raw_frame_ids = [reader.frame_ids[i] for i in logical_ids]

        images = []
        c2ws = []
        semantic_labels = []
        instance_labels = []
        for raw_frame_id in raw_frame_ids:
            image = np.asarray(reader.read_color(raw_frame_id), dtype=np.uint8).copy()
            label_path = (
                self.label_root
                / reader.scene_name
                / "label-filt"
                / f"{raw_frame_id}.png"
            )
            with Image.open(label_path) as label_image:
                label = np.asarray(label_image).copy()
            if label.ndim == 3:
                label = label[..., 0]
            if label.shape != image.shape[:2]:
                raise ValueError(
                    f"RGB/label shape mismatch at {reader.scene_name}/{raw_frame_id}: "
                    f"{image.shape[:2]} vs {label.shape}"
                )
            if DF_INSTANCE_LABEL in data_fields:
                instance_path = (
                    self.label_root
                    / reader.scene_name
                    / "instance-filt"
                    / f"{raw_frame_id}.png"
                )
                with Image.open(instance_path) as instance_image:
                    instance = np.asarray(instance_image).copy()
                if instance.ndim == 3:
                    instance = instance[..., 0]
                if instance.shape != image.shape[:2]:
                    raise ValueError(
                        f"RGB/instance shape mismatch at "
                        f"{reader.scene_name}/{raw_frame_id}: "
                        f"{image.shape[:2]} vs {instance.shape}"
                    )
                instance_labels.append(torch.from_numpy(instance).long())
            images.append(
                torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
            )
            c2ws.append(reader.get_c2w(raw_frame_id))
            semantic_labels.append(torch.from_numpy(self._map_labels(label)).long())

        output = {
            "__key__": reader.scene_name,
            DF_SCENE_NAME: reader.scene_name,
            DF_FRAME_IDS: torch.tensor(raw_frame_ids, dtype=torch.long),
        }
        if DF_IMAGE_RGB in data_fields:
            output[DF_IMAGE_RGB] = torch.stack(images).contiguous()
        if DF_CAMERA_C2W_TRANSFORM in data_fields:
            output[DF_CAMERA_C2W_TRANSFORM] = torch.stack(c2ws).contiguous()
        if DF_CAMERA_INTRINSICS in data_fields:
            output[DF_CAMERA_INTRINSICS] = reader.intrinsics.repeat(
                len(raw_frame_ids), 1
            )
        if DF_SEMANTIC_LABEL in data_fields:
            output[DF_SEMANTIC_LABEL] = torch.stack(semantic_labels).contiguous()
        if DF_INSTANCE_LABEL in data_fields:
            output[DF_INSTANCE_LABEL] = torch.stack(instance_labels).contiguous()
        return output


class ScanNetC3G8Eval(ScanNet):
    """C3G/LSM manifest evaluation protocol backed by raw ScanNet .sens files."""

    def __init__(
        self,
        root_path: str,
        label_root: str,
        manifest_path: Optional[str] = None,
        semantic_protocol_path: Optional[str] = None,
        excluded_scenes: Optional[Sequence[str]] = None,
        llff_hold: Optional[int] = None,
        test_ids: Optional[Sequence[int]] = None,
        skip_bad: bool = False,
        **kwargs,
    ):
        if int(kwargs.pop("frame_stride", 1)) != 1:
            raise ValueError("ScanNetC3G8Eval requires frame_stride=1")
        if "label_mapping" in kwargs and kwargs.pop("label_mapping") != "c3g8":
            raise ValueError("ScanNetC3G8Eval requires label_mapping='c3g8'")

        protocol_path = (
            Path(semantic_protocol_path)
            if semantic_protocol_path is not None
            else DEFAULT_C3G8_PROTOCOL_PATH
        )
        with protocol_path.open(encoding="utf-8") as handle:
            protocol = yaml.safe_load(handle)
        evaluation = protocol["evaluation"]
        if tuple(evaluation["context_offsets"]) != (-1, 1):
            raise ValueError("ScanNetC3G8Eval requires context_offsets=[-1, 1]")

        self.manifest_path = Path(manifest_path or evaluation["manifest"])
        excluded_scenes = (
            tuple(excluded_scenes)
            if excluded_scenes is not None
            else tuple(evaluation["excluded_scenes"])
        )
        self.configured_excluded_scene_names = list(excluded_scenes)
        llff_hold = int(
            llff_hold if llff_hold is not None else evaluation["llff_hold"]
        )
        test_ids = (
            tuple(test_ids)
            if test_ids is not None
            else tuple(evaluation["target_residues"])
        )
        with self.manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError(
                f"C3G manifest must be a scene-to-frame-list object: {self.manifest_path}"
            )

        self.manifest_scene_names = list(manifest)
        self.excluded_scene_names = [
            scene_name for scene_name in excluded_scenes if scene_name in manifest
        ]
        excluded = set(excluded_scenes)
        self.manifest_frames = {
            scene_name: tuple(int(frame_id) for frame_id in sorted(frame_ids))
            for scene_name, frame_ids in manifest.items()
            if frame_ids and scene_name not in excluded
        }
        if not self.manifest_frames:
            raise RuntimeError(f"No usable scenes in C3G manifest: {self.manifest_path}")

        super().__init__(
            root_path=root_path,
            label_root=label_root,
            subset=list(self.manifest_frames),
            frame_stride=1,
            label_mapping="c3g8",
            semantic_protocol_path=str(protocol_path),
            skip_bad=skip_bad,
            **kwargs,
        )
        available_scene_dirs = {path.name: path for path in self.sample_list}
        missing_scenes = sorted(set(self.manifest_frames) - set(available_scene_dirs))
        if missing_scenes:
            raise FileNotFoundError(
                f"C3G manifest scenes are missing raw .sens or labels: {missing_scenes}"
            )

        self.llff_hold = llff_hold
        self.test_ids = tuple(int(value) for value in test_ids)
        if self.llff_hold <= 0:
            raise ValueError("llff_hold must be positive")

        eval_samples = []
        for scene_name, frame_ids in self.manifest_frames.items():
            for target_index in range(len(frame_ids)):
                if target_index % self.llff_hold not in self.test_ids:
                    continue
                left_index = max(target_index - 1, 0)
                right_index = min(target_index + 1, len(frame_ids) - 1)
                eval_samples.append(
                    C3GEvalSample(
                        scene_name=scene_name,
                        context_raw_frame_ids=(
                            frame_ids[left_index],
                            frame_ids[right_index],
                        ),
                        target_raw_frame_id=frame_ids[target_index],
                        target_manifest_index=target_index,
                    )
                )
        self.scene_dirs = available_scene_dirs
        self.sample_list = eval_samples
        self._cached_scene_name: Optional[str] = None

    def _get_reader(self, idx: int) -> ScanNetSensReader:
        sample = self.sample_list[idx]
        if (
            self._cached_scene_name != sample.scene_name
            or self._cached_reader is None
        ):
            scene_dir = self.scene_dirs[sample.scene_name]
            self._cached_reader = ScanNetSensReader(
                scene_dir / f"{sample.scene_name}.sens", frame_stride=1
            )
            self._cached_scene_name = sample.scene_name
        return self._cached_reader

    def get_context_target_frames(self, idx: int) -> tuple[list[int], list[int]]:
        sample = self.sample_list[idx]
        reader = self._get_reader(idx)
        raw_to_logical = {
            raw_frame_id: logical_id
            for logical_id, raw_frame_id in enumerate(reader.frame_ids)
        }
        requested = (*sample.context_raw_frame_ids, sample.target_raw_frame_id)
        unavailable = [frame_id for frame_id in requested if frame_id not in raw_to_logical]
        if unavailable:
            raise ValueError(
                f"C3G frames have invalid poses or are outside the .sens stream for "
                f"{sample.scene_name}: {unavailable}"
            )
        context = [raw_to_logical[frame_id] for frame_id in sample.context_raw_frame_ids]
        target = [raw_to_logical[sample.target_raw_frame_id]]
        return context, target

    def get_manifest_report(self, validate_frames: bool = False) -> dict:
        preprocessed_scene_names = sorted(
            path.name
            for path in self.manifest_path.parent.glob("scene*")
            if path.is_dir()
        )
        report = {
            "manifest_path": str(self.manifest_path),
            "manifest_scenes": len(self.manifest_scene_names),
            "manifest_frames": sum(len(ids) for ids in self.manifest_frames.values()),
            "effective_scenes": len(self.manifest_frames),
            "configured_excluded_scenes": list(self.configured_excluded_scene_names),
            "excluded_scenes_present": list(self.excluded_scene_names),
            "preprocessed_scenes_not_in_manifest": sorted(
                set(preprocessed_scene_names) - set(self.manifest_scene_names)
            ),
            "manifest_scenes_missing_preprocessed_data": sorted(
                set(self.manifest_scene_names) - set(preprocessed_scene_names)
            ),
            "samples": len(self.sample_list),
            "frames_per_scene": sorted({len(ids) for ids in self.manifest_frames.values()}),
            "llff_hold": self.llff_hold,
            "test_ids": list(self.test_ids),
        }
        if validate_frames:
            report.update(self.validate_manifest_frames())
        return report

    def validate_manifest_frames(self) -> dict:
        missing_labels = []
        invalid_or_missing_pose_frames = []
        for scene_name, frame_ids in self.manifest_frames.items():
            sample_index = next(
                i for i, sample in enumerate(self.sample_list) if sample.scene_name == scene_name
            )
            reader = self._get_reader(sample_index)
            valid_frame_ids = set(reader.frame_ids)
            for raw_frame_id in frame_ids:
                if not (
                    self.label_root / scene_name / "label-filt" / f"{raw_frame_id}.png"
                ).is_file():
                    missing_labels.append((scene_name, raw_frame_id))
                if raw_frame_id not in valid_frame_ids:
                    invalid_or_missing_pose_frames.append((scene_name, raw_frame_id))
        return {
            "missing_labels": missing_labels,
            "invalid_or_missing_pose_frames": invalid_or_missing_pose_frames,
        }


@dataclass
class C3GPromptEvalSample:
    scene_name: str
    context_raw_frame_ids: tuple[int, int]
    target_raw_frame_id: int
    class_id: int


class ScanNetC3G8PromptEval(ScanNetC3G8Eval):
    """Eight-class text-only prompt evaluation over the held-out C3G split.

    Each (scene, target frame) from the C3G/LSM evaluation protocol is expanded
    into eight text-only prompt samples, one per C3G8 class. Reconstruction,
    masks, and per-class segmentation metrics are all evaluated at the target
    frame with the frozen TokenGS reconstruction.
    """

    has_prompt_samples = True
    has_explicit_split = True

    def __init__(
        self,
        root_path: str,
        label_root: str,
        manifest_path: Optional[str] = None,
        semantic_protocol_path: Optional[str] = None,
        excluded_scenes: Optional[Sequence[str]] = None,
        llff_hold: Optional[int] = None,
        test_ids: Optional[Sequence[int]] = None,
        skip_bad: bool = False,
        **kwargs,
    ):
        # Provider injects prompt-related kwargs for has_prompt_samples datasets.
        for key in (
            "prompt_mode",
            "query_image_size",
            "prompt_image_probability",
            "prompt_min_target_pixels",
        ):
            kwargs.pop(key, None)
        super().__init__(
            root_path=root_path,
            label_root=label_root,
            manifest_path=manifest_path,
            semantic_protocol_path=semantic_protocol_path,
            excluded_scenes=excluded_scenes,
            llff_hold=llff_hold,
            test_ids=test_ids,
            skip_bad=skip_bad,
            **kwargs,
        )
        self.query_image_size = (224, 224)
        prompt_samples = []
        for sample in self.sample_list:
            for class_id in range(1, 9):
                prompt_samples.append(
                    C3GPromptEvalSample(
                        scene_name=sample.scene_name,
                        context_raw_frame_ids=sample.context_raw_frame_ids,
                        target_raw_frame_id=sample.target_raw_frame_id,
                        class_id=class_id,
                    )
                )
        self.sample_list = prompt_samples
        self._active_prompt_sample: Optional[C3GPromptEvalSample] = None

    def _get_reader(self, idx: int) -> ScanNetSensReader:
        sample = self.sample_list[idx]
        if (
            self._cached_scene_name != sample.scene_name
            or self._cached_reader is None
        ):
            scene_dir = self.scene_dirs[sample.scene_name]
            self._cached_reader = ScanNetSensReader(
                scene_dir / f"{sample.scene_name}.sens", frame_stride=1
            )
            self._cached_scene_name = sample.scene_name
        self._active_prompt_sample = sample
        return self._cached_reader

    def get_context_target_frames(self, idx: int) -> tuple[list[int], list[int]]:
        sample = self.sample_list[idx]
        reader = self._get_reader(idx)
        raw_to_logical = {
            raw_frame_id: logical_id
            for logical_id, raw_frame_id in enumerate(reader.frame_ids)
        }
        requested = (*sample.context_raw_frame_ids, sample.target_raw_frame_id)
        unavailable = [value for value in requested if value not in raw_to_logical]
        if unavailable:
            raise ValueError(
                f"C3G prompt eval frames have invalid poses for "
                f"{sample.scene_name}/{sample.target_raw_frame_id}: {unavailable}"
            )
        return (
            [raw_to_logical[value] for value in sample.context_raw_frame_ids],
            [raw_to_logical[sample.target_raw_frame_id]],
        )

    def make_prompt_sample(
        self,
        semantic_label_output: torch.Tensor,
        target_scene_name: str,
        used_frame_ids: torch.Tensor,
        rng: np.random.Generator,
    ) -> dict:
        del rng
        sample = self._active_prompt_sample
        if sample is None or sample.scene_name != target_scene_name:
            raise RuntimeError("C3G prompt eval sample state is inconsistent")
        expected_frames = (*sample.context_raw_frame_ids, sample.target_raw_frame_id)
        if tuple(int(value) for value in used_frame_ids.tolist()) != expected_frames:
            raise RuntimeError(
                f"C3G prompt eval frame mismatch for {sample.scene_name}/"
                f"{sample.target_raw_frame_id}"
            )
        class_names = self.semantic_class_names
        negative_class_id = sample.class_id % 8 + 1
        return {
            "sample_id": (
                f"c3g8_{sample.scene_name}_{sample.target_raw_frame_id}_"
                f"class{sample.class_id}"
            ),
            "prompt_mode": "text_only",
            "prompt_type": "text",
            "prompt_class_id": torch.tensor(sample.class_id, dtype=torch.long),
            "positive_text_prompt": class_names[sample.class_id - 1],
            "negative_text_prompt": class_names[negative_class_id - 1],
            "binary_mask_output": (
                semantic_label_output == sample.class_id
            ).float(),
            "image_query_attempted": torch.tensor(False, dtype=torch.bool),
            "has_image_query": torch.tensor(False, dtype=torch.bool),
            "query_image": torch.zeros(
                (3, *self.query_image_size), dtype=torch.float32
            ),
            "query_mask": torch.zeros(
                (1, *self.query_image_size), dtype=torch.float32
            ),
            "query_scene_name": "",
            "query_frame_id": torch.tensor(-1, dtype=torch.long),
            "query_class_id": torch.tensor(sample.class_id, dtype=torch.long),
        }
