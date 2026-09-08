"""LSM-style ScanNet training dataset (8 context + 7 test, interleaved).

Mirrors the LSM instance-evaluation protocol used by
``scannet_lsm_instance_eval``: per window, context frames sit at raw-frame
offsets [s, s+20, ..., s+140] and test frames at [s+10, s+30, ..., s+130].
Each scene contributes ``windows_per_scene`` randomly placed windows, so
training sees the same view structure as evaluation while covering the full
1473-scene ScanNet train split (eval scenes excluded by the manifest).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from tokengs.data.static.scannet import ScanNet, ScanNetSensReader


def _load_train_scenes(train_manifest_path: str) -> list[str]:
    manifest = json.loads(Path(train_manifest_path).read_text(encoding="utf-8"))
    scenes = manifest["scenes"]
    return sorted(str(scene) for scene in scenes)


class ScanNetLSMStyleTrain(ScanNet):
    """Fixed 8-context / 7-test interleaved windows over the ScanNet train set."""

    is_static = True
    has_semantic_labels = True
    has_instance_labels = True
    # Not an explicit split: the Provider retries a fresh window when
    # ``get_context_target_frames`` rejects a sample (e.g. scenes too sparse
    # to host a full 8+7 window), instead of crashing the run.

    def __init__(
        self,
        root_path: str,
        label_root: str,
        train_manifest_path: str,
        semantic_protocol_path: Optional[str] = None,
        windows_per_scene: int = 16,
        seed: int = 0,
        **_kwargs,
    ) -> None:
        scene_names = _load_train_scenes(train_manifest_path)
        super().__init__(
            root_path=root_path,
            label_root=label_root,
            subset=scene_names,
            frame_stride=1,
            label_mapping="c3g8",
            semantic_protocol_path=semantic_protocol_path,
            skip_bad=False,
        )
        self.windows_per_scene = int(windows_per_scene)
        self._cached_scene_name: Optional[str] = None
        self._cached_reader: Optional[ScanNetSensReader] = None
        self.scene_dirs = list(self.sample_list)

        self._seed = int(seed)
        self.windows: list[dict] = []
        for scene_dir in self.scene_dirs:
            for _ in range(self.windows_per_scene):
                self.windows.append(
                    {
                        "scene": scene_dir.name,
                        "window_index": len(self.windows),
                    }
                )

        self.sample_list = self.windows
        if not self.sample_list:
            raise RuntimeError(
                f"No LSM-style training windows under {self.root_path}"
            )
        print(
            f"[ScanNetLSMStyleTrain] scenes={len(scene_names)} "
            f"windows={len(self.sample_list)}"
        )

    def _get_reader(self, idx: int) -> ScanNetSensReader:
        sample = self.sample_list[idx]
        if (
            self._cached_scene_name != sample["scene"]
            or self._cached_reader is None
        ):
            scene_dir = next(
                path for path in self.scene_dirs
                if path.name == sample["scene"]
            )
            self._cached_reader = ScanNetSensReader(
                scene_dir / f"{sample['scene']}.sens", frame_stride=1
            )
            self._cached_scene_name = sample["scene"]
        return self._cached_reader

    def get_context_target_frames(self, idx: int) -> tuple[list[int], list[int]]:
        sample = self.sample_list[idx]
        reader = self._get_reader(idx)
        # Deterministic per (scene, window): the same windows are reused
        # every epoch without re-scanning every scene's .sens at build time.
        rng = np.random.default_rng(
            self._seed + hash(sample["scene"]) % (2**31) + sample["window_index"]
        )
        num_raw = len(reader)
        valid_raw = list(reader.frame_ids)
        valid_set = set(valid_raw)
        raw_to_logical = {
            raw_frame_id: logical_id
            for logical_id, raw_frame_id in enumerate(valid_raw)
        }
        max_raw = max(valid_set) if valid_set else 0
        max_start = max(0, max_raw - 150)
        if num_raw < 200:
            # Too sparse to reliably host an interleaved 8+7 window; let the
            # Provider skip this sample via retry.
            raise RuntimeError(
                f"Scene {sample['scene']} too sparse for LSM-style window "
                f"(valid frames={num_raw})"
            )
        # ScanNet sometimes has long bursts of invalid poses; sample random
        # starts instead of walking consecutive ones so sparse scenes still
        # find a fully-valid window.
        context = test = None
        for _ in range(5000):
            candidate = int(rng.integers(0, max_start + 1))
            ctx = [candidate + 20 * k for k in range(8)]
            tgt = [candidate + 10 + 20 * k for k in range(7)]
            if valid_set.issuperset(ctx + tgt):
                context, test = ctx, tgt
                break
        if context is None:
            raise RuntimeError(
                f"No valid LSM-style window found for "
                f"{sample['scene']} (valid frames={num_raw})"
            )
        context_logical = [raw_to_logical[value] for value in context]
        target_logical = [
            raw_to_logical[value] for value in test
        ]
        return context_logical, target_logical

    def count_frames(self, idx: int) -> int:
        return len(self._get_reader(idx))

    def count_cameras(self, idx: int) -> int:
        return 1

    def load_video_reader(self, idx: int):
        sample = self.sample_list[idx]
        key = f"{sample['scene']}_{idx}"
        return None, key, len(sample["test_raw_frame_ids"])
