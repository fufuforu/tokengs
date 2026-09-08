# SPDX-License-Identifier: Apache-2.0
"""C3G ScanNet-test dataloader adapted to TokenGS, with 8-class labels.

Expected directory:
    data/scannet_test/
        selected_seqs_test.json
        scannetv2-labels.combined.tsv
        sceneXXXX_XX/
            images/<frame_id>.jpg
            images/<frame_id>.npz
            depths/<frame_id>.png
            labels/<frame_id>.png

Each dataset item is ordered as:
    [source_left, source_right, target]

Semantic IDs returned to Provider:
    0: ignore/void
    1: wall
    2: floor
    3: ceiling
    4: chair
    5: table
    6: sofa
    7: bed
    8: other

Ported from tokengs_c3g (the exact loader used for the ~0.536 C3G8 result).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from tokengs.data.datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_DEPTH,
    DF_FOREGROUND_MASK,
    DF_IMAGE_RGB,
    DF_INSTANCE_LABEL,
    DF_SCENE_NAME,
    DF_SEMANTIC_LABEL,
)


SCANNET_8_CLASSES = (
    "wall",
    "floor",
    "ceiling",
    "chair",
    "table",
    "sofa",
    "bed",
    "other",
)


def _parse_resolution(resolution: Union[str, Sequence[int]]) -> tuple[int, int]:
    """Return (height, width)."""
    if isinstance(resolution, str):
        text = resolution.lower().strip()
        if "x" not in text:
            raise ValueError(
                f"Resolution must look like '256x256', got {resolution!r}"
            )
        width, height = text.split("x", maxsplit=1)
        return int(height), int(width)

    if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
        return int(resolution[0]), int(resolution[1])

    raise ValueError(f"Unsupported resolution: {resolution!r}")


def _ensure_c2w_4x4(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3] = pose
        return out
    raise ValueError(f"camera_pose must be [4,4] or [3,4], got {pose.shape}")


def _ensure_k_3x3(intrinsics: np.ndarray) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.shape == (3, 3):
        return intrinsics.copy()
    if intrinsics.shape == (4, 4):
        return intrinsics[:3, :3].copy()
    raise ValueError(
        f"camera_intrinsics must be [3,3] or [4,4], got {intrinsics.shape}"
    )


def _resolve_optional_path(root: Path, path: Union[str, Path]) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def _build_scannet_8class_lut(tsv_path: Path) -> np.ndarray:
    """Build raw ScanNet category-id -> C3G 8-class id lookup table."""
    if not tsv_path.exists():
        raise FileNotFoundError(
            "Missing ScanNet label mapping TSV: "
            f"{tsv_path}\n"
            "Pass label_tsv=<absolute path to scannetv2-labels.combined.tsv>."
        )

    class_to_id = {
        name: index + 1
        for index, name in enumerate(SCANNET_8_CLASSES)
    }
    other_id = class_to_id["other"]

    rows: list[tuple[int, int]] = []
    max_raw_id = 0

    with tsv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required = {"id", "nyu40class"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"TSV must contain columns {sorted(required)}, "
                f"got {reader.fieldnames}"
            )

        for row in reader:
            raw_text = row.get("id", "")
            if raw_text is None or raw_text == "":
                continue

            raw_id = int(raw_text)
            nyu40_class = str(row.get("nyu40class", "")).strip().lower()
            mapped_id = class_to_id.get(nyu40_class, other_id)

            rows.append((raw_id, mapped_id))
            max_raw_id = max(max_raw_id, raw_id)

    # Unknown non-zero IDs map to "other"; zero remains ignored.
    lut = np.full(max_raw_id + 1, other_id, dtype=np.int64)
    lut[0] = 0
    for raw_id, mapped_id in rows:
        lut[raw_id] = mapped_id
    return lut


def _map_raw_label(raw_label: np.ndarray, lut: np.ndarray) -> np.ndarray:
    raw_label = np.asarray(raw_label)
    other_id = SCANNET_8_CLASSES.index("other") + 1

    mapped = np.full(raw_label.shape, other_id, dtype=np.int64)
    mapped[raw_label == 0] = 0

    valid = (raw_label >= 0) & (raw_label < len(lut))
    mapped[valid] = lut[raw_label[valid].astype(np.int64)]
    return mapped


def _c3g_crop_resize_rgb_label(
    image: Image.Image,
    label: Optional[Image.Image],
    intrinsics: np.ndarray,
    output_hw: tuple[int, int],
) -> tuple[Image.Image, Optional[Image.Image], np.ndarray]:
    """Apply exactly the same crop/resize geometry to RGB and label images."""
    image = image.convert("RGB")
    k = intrinsics.astype(np.float32).copy()

    target_h, target_w = output_hw
    width, height = image.size

    if label is not None and label.size != image.size:
        raise ValueError(
            f"RGB/label size mismatch: rgb={image.size}, label={label.size}"
        )

    cx = int(np.rint(k[0, 2]))
    cy = int(np.rint(k[1, 2]))

    margin_x = min(cx, width - cx)
    margin_y = min(cy, height - cy)
    if margin_x <= 0 or margin_y <= 0:
        raise ValueError(
            f"Invalid principal point ({k[0,2]:.3f}, {k[1,2]:.3f}) "
            f"for image size {(width, height)}"
        )

    left = cx - margin_x
    top = cy - margin_y
    right = cx + margin_x
    bottom = cy + margin_y

    crop_box = (left, top, right, bottom)
    image = image.crop(crop_box)
    if label is not None:
        label = label.crop(crop_box)

    k[0, 2] -= float(left)
    k[1, 2] -= float(top)

    crop_w, crop_h = image.size

    if crop_h > 1.1 * crop_w:
        target_h, target_w = target_w, target_h

    resize_scale = max(target_w / crop_w, target_h / crop_h)
    new_w = max(target_w, int(np.rint(crop_w * resize_scale)))
    new_h = max(target_h, int(np.rint(crop_h * resize_scale)))

    image = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    if label is not None:
        label = label.resize((new_w, new_h), resample=Image.Resampling.NEAREST)

    scale_x = new_w / crop_w
    scale_y = new_h / crop_h
    k[0, :] *= scale_x
    k[1, :] *= scale_y

    crop_left = int(np.rint((new_w - target_w) * 0.5))
    crop_top = int(np.rint((new_h - target_h) * 0.5))
    crop_left = min(max(crop_left, 0), new_w - target_w)
    crop_top = min(max(crop_top, 0), new_h - target_h)

    final_box = (
        crop_left,
        crop_top,
        crop_left + target_w,
        crop_top + target_h,
    )
    image = image.crop(final_box)
    if label is not None:
        label = label.crop(final_box)

    k[0, 2] -= float(crop_left)
    k[1, 2] -= float(crop_top)

    if image.size != (target_w, target_h):
        raise RuntimeError(
            f"Unexpected final image size {image.size}; "
            f"expected {(target_w, target_h)}"
        )
    if label is not None and label.size != image.size:
        raise RuntimeError(
            f"Unexpected final label size {label.size}; expected {image.size}"
        )

    return image, label, k


def _pil_to_float_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class ScanNetC3GSemanticEval:
    """Map-style TokenGS dataset for C3G's preprocessed ScanNet test split."""

    is_static = True
    use_2d_labels = True
    has_semantic_labels = True
    already_preprocessed = True

    def __init__(
        self,
        root_path: Union[str, Path],
        resolution: Union[str, Sequence[int]] = "256x256",
        selected_json: str = "selected_seqs_test.json",
        label_tsv: Union[str, Path] = "scannetv2-labels.combined.tsv",
        llff_hold: int = 8,
        test_ids: Sequence[int] = (1, 4),
        subset: Optional[Sequence[str]] = None,
        ignored_scenes: Sequence[str] = ("scene0696_02",),
        validate_files: bool = True,
        require_labels: bool = True,
        debug: bool = False,
        **_: object,
    ) -> None:
        self.root_path = Path(root_path)
        self.resolution = _parse_resolution(resolution)
        self.llff_hold = int(llff_hold)
        self.test_ids = {int(x) for x in test_ids}
        self.require_labels = bool(require_labels)
        self.debug = bool(debug)

        self.label_tsv_path = _resolve_optional_path(self.root_path, label_tsv)
        self.label_lut = _build_scannet_8class_lut(self.label_tsv_path)

        json_path = self.root_path / selected_json
        if not json_path.exists():
            raise FileNotFoundError(f"Missing C3G split file: {json_path}")

        with json_path.open("r", encoding="utf-8") as file:
            scenes = json.load(file)

        subset_set = set(subset) if subset is not None else None
        ignored_set = set(ignored_scenes)

        self.scenes: dict[str, list] = {}
        for scene, frame_ids in scenes.items():
            if not frame_ids or scene in ignored_set:
                continue
            if subset_set is not None and scene not in subset_set:
                continue
            self.scenes[scene] = sorted(frame_ids)

        self.sample_list: list[dict] = []
        for scene, frame_ids in self.scenes.items():
            selected_targets = [
                position
                for position in range(len(frame_ids))
                if position % self.llff_hold in self.test_ids
            ]

            for target_position in selected_targets:
                left_position = max(target_position - 1, 0)
                right_position = min(target_position + 1, len(frame_ids) - 1)
                logical_positions = [left_position, right_position, target_position]
                selected_frame_ids = [
                    frame_ids[position]
                    for position in logical_positions
                ]

                sample = {
                    "scene": scene,
                    "logical_positions": logical_positions,
                    "frame_ids": selected_frame_ids,
                }

                if validate_files and not self._sample_is_valid(sample):
                    continue
                self.sample_list.append(sample)

        if not self.sample_list:
            raise RuntimeError(
                f"No valid C3G ScanNet samples found under {self.root_path}"
            )

    def __repr__(self) -> str:
        return (
            f"ScanNetC3GEval(root={self.root_path}, "
            f"samples={len(self.sample_list)}, resolution={self.resolution})"
        )

    def _frame_paths(
        self,
        scene: str,
        frame_id: object,
    ) -> tuple[Path, Path, Path]:
        stem = str(frame_id)
        image_path = self.root_path / scene / "images" / f"{stem}.jpg"
        metadata_path = self.root_path / scene / "images" / f"{stem}.npz"
        label_path = self.root_path / scene / "labels" / f"{stem}.png"
        return image_path, metadata_path, label_path

    def _sample_is_valid(self, sample: dict) -> bool:
        for frame_id in sample["frame_ids"]:
            image_path, metadata_path, label_path = self._frame_paths(
                sample["scene"], frame_id
            )
            if not image_path.exists() or not metadata_path.exists():
                return False
            if self.require_labels and not label_path.exists():
                return False
            try:
                with np.load(metadata_path) as metadata:
                    pose = _ensure_c2w_4x4(metadata["camera_pose"])
                    _ = _ensure_k_3x3(metadata["camera_intrinsics"])
                if not np.isfinite(pose).all():
                    return False
            except Exception:
                return False
        return True

    def __len__(self) -> int:
        return len(self.sample_list)

    def count_frames(self, idx: int) -> int:
        del idx
        return 3

    def count_cameras(self, idx: int) -> int:
        del idx
        return 1

    def get_context_target_frames(self, idx: int) -> tuple[list[int], list[int]]:
        del idx
        return [0, 1], [2]

    def load_video_reader(self, idx: int):
        sample = self.sample_list[idx]
        key = f"{sample['scene']}_{sample['frame_ids'][2]}"
        return None, key, 3

    def get_data(
        self,
        idx: int,
        data_fields: List[str],
        frame_indices: Optional[Sequence[int]] = None,
        view_indices: Optional[Sequence[int]] = None,
        camera_convention: str = "opencv",
    ) -> dict:
        del view_indices
        if camera_convention != "opencv":
            raise ValueError(
                "ScanNetC3GEval only supports OpenCV convention, "
                f"got {camera_convention}"
            )

        sample = self.sample_list[idx]
        local_indices = (
            [0, 1, 2]
            if frame_indices is None
            else [int(x) for x in frame_indices]
        )

        images = []
        c2ws = []
        intrinsics = []
        semantic_labels = []
        instance_labels = []

        for local_index in local_indices:
            if local_index < 0 or local_index >= 3:
                raise IndexError(f"Local frame index out of range: {local_index}")

            frame_id = sample["frame_ids"][local_index]
            image_path, metadata_path, label_path = self._frame_paths(
                sample["scene"], frame_id
            )

            image = Image.open(image_path).convert("RGB")
            raw_label_pil: Optional[Image.Image]
            if label_path.exists():
                raw_label_pil = Image.open(label_path)
            elif self.require_labels:
                raise FileNotFoundError(f"Missing semantic label: {label_path}")
            else:
                raw_label_pil = None

            with np.load(metadata_path) as metadata:
                c2w = _ensure_c2w_4x4(metadata["camera_pose"])
                k = _ensure_k_3x3(metadata["camera_intrinsics"])

            if not np.isfinite(c2w).all():
                raise ValueError(f"Non-finite pose in {metadata_path}")

            image, raw_label_pil, k = _c3g_crop_resize_rgb_label(
                image=image,
                label=raw_label_pil,
                intrinsics=k,
                output_hw=self.resolution,
            )

            if raw_label_pil is None:
                mapped_label = np.zeros(self.resolution, dtype=np.int64)
            else:
                raw_label = np.asarray(raw_label_pil)
                mapped_label = _map_raw_label(raw_label, self.label_lut)

            images.append(_pil_to_float_tensor(image))
            c2ws.append(torch.from_numpy(c2w))
            intrinsics.append(
                torch.tensor(
                    [k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
                    dtype=torch.float32,
                )
            )
            semantic_labels.append(torch.from_numpy(mapped_label).long())
            instance_labels.append(torch.zeros_like(semantic_labels[-1]))

        images_t = torch.stack(images, dim=0).float().contiguous()
        c2ws_t = torch.stack(c2ws, dim=0).float().contiguous()
        intrinsics_t = torch.stack(intrinsics, dim=0).float().contiguous()
        semantic_t = torch.stack(semantic_labels, dim=0).long().contiguous()
        instance_t = torch.stack(instance_labels, dim=0).long().contiguous()

        masks_t = torch.ones_like(images_t[:, :1])
        depth_maps_t = torch.zeros_like(images_t[:, :1])

        output = {
            "__key__": f"{sample['scene']}_{sample['frame_ids'][2]}",
            DF_SCENE_NAME: sample["scene"],
            "scene": sample["scene"],
            "frame_ids": list(sample["frame_ids"]),
        }

        for field in data_fields:
            if field == DF_IMAGE_RGB:
                output[field] = images_t
            elif field == DF_CAMERA_C2W_TRANSFORM:
                output[field] = c2ws_t
            elif field == DF_CAMERA_INTRINSICS:
                output[field] = intrinsics_t
            elif field == DF_FOREGROUND_MASK:
                output[field] = masks_t
            elif field == DF_DEPTH:
                output[field] = depth_maps_t
            elif field == DF_SEMANTIC_LABEL:
                output[field] = semantic_t
            elif field == DF_INSTANCE_LABEL:
                output[field] = instance_t

        return output
