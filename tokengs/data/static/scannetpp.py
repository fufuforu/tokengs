"""ScanNet++ instance-mask training dataset.

Reads the processed ScanNet++ scenes produced by
``scripts/build_scannetpp_instance_masks.py``: per-scene ``images/`` +
``masks/`` (per-frame instance id PNGs) + ``manifest.json`` (c2w / K per
frame).  Provides LSM-style 8-context / 7-target interleaved windows with
per-frame instance masks (no semantic labels).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from tokengs.data.datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_DEPTH,
    DF_FOREGROUND_MASK,
    DF_FRAME_IDS,
    DF_IMAGE_RGB,
    DF_INSTANCE_LABEL,
    DF_SCENE_NAME,
    DF_SEMANTIC_LABEL,
)


class ScanNetPPInstanceTrain:
    """LSM-style 8+7 windows over processed ScanNet++ scenes."""

    is_static = True
    has_semantic_labels = False
    has_instance_labels = True
    disable_random_reflect = True

    def __init__(
        self,
        root_path: str,
        scenes_file: Optional[str] = None,
        windows_per_scene: int = 8,
        frame_stride: int = 10,
        seed: int = 0,
        **_kwargs,
    ) -> None:
        root = Path(root_path)
        self.root = root
        self.frame_stride = int(frame_stride)
        self.windows_per_scene = int(windows_per_scene)
        self._seed = int(seed)
        self.sample_list: List[dict] = []
        if scenes_file and Path(scenes_file).is_file():
            scene_names = [
                line.strip()
                for line in Path(scenes_file).read_text().splitlines()
                if line.strip()
            ]
        else:
            scene_names = sorted(
                p.name
                for p in root.iterdir()
                if p.is_dir() and (p / "manifest.json").is_file()
            )
        self.scene_manifests: dict[str, dict] = {}
        for scene in scene_names:
            manifest_path = root / scene / "manifest.json"
            masks_dir = root / scene / "masks"
            if not manifest_path.is_file() or not masks_dir.is_dir():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            n = int(manifest.get("num_frames", len(manifest.get("frames", []))))
            if n < 60:
                continue
            self.scene_manifests[scene] = manifest
            for w in range(self.windows_per_scene):
                self.sample_list.append(
                    {"scene": scene, "window_index": w}
                )
        if not self.sample_list:
            raise RuntimeError(
                f"No usable ScanNet++ scenes under {root} "
                f"(need manifest.json + masks/)"
            )
        self.scene_names = list(self.scene_manifests.keys())
        print(
            f"[ScanNetPPInstanceTrain] scenes={len(self.scene_names)} "
            f"windows={len(self.sample_list)}"
        )

    def _manifest(self, scene: str) -> dict:
        return self.scene_manifests[scene]

    def __len__(self) -> int:
        return len(self.sample_list)

    def get_context_target_frames(
        self, idx: int
    ) -> Tuple[List[int], List[int]]:
        sample = self.sample_list[idx]
        manifest = self._manifest(sample["scene"])
        frames = manifest["frames"]
        n = len(frames)
        rng = np.random.default_rng(
            self._seed + abs(hash(sample["scene"])) % (2**31)
            + sample["window_index"]
        )
        stride = self.frame_stride
        max_start = max(0, n - (7 * stride + 7) - 1)
        for _ in range(5000):
            start = int(rng.integers(0, max_start + 1))
            ctx = [start + 2 * k * stride for k in range(8)]
            tgt = [start + (2 * k + 1) * stride for k in range(7)]
            if max(ctx + tgt) < n:
                return ctx, tgt
        raise RuntimeError(
            f"Scene {sample['scene']} too sparse for LSM-style window"
        )

    def get_data(
        self,
        idx: int,
        data_fields: List[str],
        frame_indices=None,
        view_indices=None,
        camera_convention: str = "opencv",
        **kwargs,
    ) -> dict:
        del kwargs
        if camera_convention != "opencv":
            raise ValueError("ScanNet++ exposes raw OpenCV-style c2w poses")
        sample = self.sample_list[idx]
        scene = sample["scene"]
        manifest = self._manifest(scene)
        frames = manifest["frames"]
        logical_ids = (
            range(len(frames)) if frame_indices is None else frame_indices
        )
        logical_ids = [int(i) for i in logical_ids]

        images = []
        c2ws = []
        intrinsics = []
        instance_labels = []
        for logical_id in logical_ids:
            entry = frames[logical_id]
            image_path = self.root / scene / entry["image"]
            with Image.open(image_path) as img:
                image = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
            c2w = np.asarray(entry["c2w"], dtype=np.float32)  # [4,4]
            # Images/masks are stored at render_scale * full resolution, so
            # the pinhole intrinsics must be the half-resolution K_render,
            # NOT the full-resolution K (using K would double focal length /
            # principal point and misalign the whole scene geometry).
            k = np.asarray(entry["K_render"], dtype=np.float32)  # [3,3]
            images.append(
                torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            )
            c2ws.append(torch.from_numpy(c2w))
            intrinsics.append(
                torch.tensor(
                    [k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
                    dtype=torch.float32,
                )
            )
            if DF_INSTANCE_LABEL in data_fields:
                mask_path = self.root / scene / entry["mask"]
                with Image.open(mask_path) as msk:
                    inst = np.asarray(msk, dtype=np.int64).copy()
                if inst.ndim == 3:
                    inst = inst[..., 0]
                instance_labels.append(torch.from_numpy(inst).long())

        output = {
            "__key__": scene,
            DF_SCENE_NAME: scene,
            DF_FRAME_IDS: torch.tensor(logical_ids, dtype=torch.long),
            DF_IMAGE_RGB: torch.stack(images).contiguous(),
            DF_CAMERA_C2W_TRANSFORM: torch.stack(c2ws).contiguous(),
            DF_CAMERA_INTRINSICS: torch.stack(intrinsics).contiguous(),
            DF_FOREGROUND_MASK: torch.ones(
                len(logical_ids), 1, images[0].shape[1], images[0].shape[2]
            ),
            DF_DEPTH: torch.ones(
                len(logical_ids), 1, images[0].shape[1], images[0].shape[2]
            ),
        }
        if DF_INSTANCE_LABEL in data_fields:
            output[DF_INSTANCE_LABEL] = torch.stack(instance_labels).long()
        return output
