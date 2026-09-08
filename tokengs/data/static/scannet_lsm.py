"""LSM ScanNet instance-segmentation evaluation dataset (40 scenes)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from tokengs.data.static.scannet import ScanNet


class ScanNetLSMInstanceEval(ScanNet):
    """Fixed 8-context / 7-test view evaluation per scene.

    Frame ids come from ``lsm_instance_eval_manifest.json`` (stride-10
    sampling of the C3G test scenes, interleaved split), and instance
    supervision is available through the standard ScanNet ``instance-filt``
    labels. Evaluation renders the 7 test views after feeding 8 context
    views to the model, matching the protocol of LSM / Gaussian Grouping /
    ObjectGS / IGGT / InstOk3D.
    """

    is_static = True
    has_semantic_labels = True
    has_instance_labels = True
    has_explicit_split = True

    def __init__(
        self,
        root_path: str,
        label_root: str,
        lsm_manifest_path: str,
        semantic_protocol_path: Optional[str] = None,
        **_kwargs,
    ):
        manifest = json.loads(
            Path(lsm_manifest_path).read_text(encoding="utf-8")
        )
        scene_names = sorted(manifest["scenes"])
        super().__init__(
            root_path=root_path,
            label_root=label_root,
            subset=scene_names,
            frame_stride=1,
            label_mapping="c3g8",
            semantic_protocol_path=semantic_protocol_path,
            skip_bad=False,
        )
        self.scene_entries = manifest["scenes"]
        self.split_scene_names = [
            Path(path).name for path in self.sample_list
        ]
        missing = sorted(set(scene_names) - set(self.split_scene_names))
        if missing:
            raise FileNotFoundError(
                f"LSM eval scenes unavailable: {missing}"
            )

    def get_context_target_frames(
        self, idx: int
    ) -> tuple[list[int], list[int]]:
        scene_name = self.split_scene_names[idx]
        reader = self._get_reader(idx)
        raw_to_logical = {
            raw_frame_id: logical_id
            for logical_id, raw_frame_id in enumerate(reader.frame_ids)
        }
        entry = self.scene_entries[scene_name]
        context = [
            raw_to_logical[value]
            for value in entry["context_raw_frame_ids"]
        ]
        target = [
            raw_to_logical[value] for value in entry["test_raw_frame_ids"]
        ]
        return context, target
