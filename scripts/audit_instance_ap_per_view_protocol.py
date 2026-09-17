"""Write the read-only audit artifacts for per-target-view AP identity."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokengs.utils.instance_ap import make_target_view_image_id


OUT = ROOT / "workspace/instance_ap_per_view_protocol_audit_v1"
MANIFEST = ROOT / "data/scannet_prompt/lsm_instance_eval_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    scenes = payload.get("scenes")
    if not isinstance(scenes, dict) or len(scenes) != 40:
        raise RuntimeError("LSM manifest must contain exactly 40 scenes")
    image_ids = []
    scene_evidence = []
    for scene_id, entry in scenes.items():
        context = entry["context_raw_frame_ids"]
        target = entry["test_raw_frame_ids"]
        if len(context) != 8 or len(target) != 7:
            raise RuntimeError(f"{scene_id} is not an 8+7 entry")
        ids = [make_target_view_image_id(
            scene_id=scene_id,
            context_frame_ids=context,
            target_frame_ids=target,
            target_view_index=view,
        ) for view in range(7)]
        image_ids.extend(ids)
        scene_evidence.append({
            "scene_id": scene_id,
            "context_frame_ids": [int(value) for value in context],
            "target_frame_ids": [int(value) for value in target],
            "target_view_image_ids": ids,
            "stride_10": all(
                right - left == 10
                for left, right in zip(
                    sum(([int(c), int(t)] for c, t in zip(context, target)), [])
                    + [int(context[-1])],
                    (sum(([int(c), int(t)] for c, t in zip(context, target)), [])
                     + [int(context[-1])])[1:],
                )
            ),
        })
    if len(set(image_ids)) != 280:
        raise RuntimeError("LSM target-view image IDs are not unique")
    OUT.mkdir(parents=True, exist_ok=True)
    callsites = [
        {
            "file": "scripts/eval_instance_lsm_protocol.py",
            "function": "main evaluation loop",
            "current_image_id_construction": "make_target_view_image_id(scene, manifest context/target, view)",
            "cross_target_view_matching_possible": False,
            "needs_modification": True,
            "training_loss": False,
            "formal_evaluation": True,
        },
        {
            "file": "scripts/audit_gsi_v2_lsm40_paired.py",
            "function": "_instance_metrics via _evaluate_model",
            "current_image_id_construction": "make_target_view_image_id(scene, manifest context/target, view)",
            "cross_target_view_matching_possible": False,
            "needs_modification": True,
            "training_loss": False,
            "formal_evaluation": True,
            "downstream": ["J2 LSM-40", "QMC LSM-40", "Gaussian Grouping adapter"],
        },
        {
            "file": "scripts/audit_token_eru_dino_joint_formation_24window_paired.py",
            "function": "_native_window_metrics",
            "current_image_id_construction": "make_target_view_image_id(scene, frame lists, view)",
            "cross_target_view_matching_possible": False,
            "needs_modification": True,
            "training_loss": False,
            "formal_evaluation": True,
            "downstream": ["EQC 24-window evaluator"],
        },
        {
            "file": "scripts/eval_token_eru_dino_joint_formation_lsm40_paired.py",
            "function": "_evaluate_model delegation",
            "current_image_id_construction": "delegates to corrected audit_gsi_v2_lsm40_paired",
            "cross_target_view_matching_possible": False,
            "needs_modification": False,
            "training_loss": False,
            "formal_evaluation": True,
        },
        {
            "file": "scripts/eval_token_eru_query_metric_paired.py",
            "function": "_evaluate_model delegation",
            "current_image_id_construction": "delegates to corrected audit_gsi_v2_lsm40_paired",
            "cross_target_view_matching_possible": False,
            "needs_modification": False,
            "training_loss": False,
            "formal_evaluation": True,
        },
        {
            "file": "scripts/eval_token_eru_early_query_codecoder_paired.py",
            "function": "_evaluate_model delegation",
            "current_image_id_construction": "delegates to corrected 24-window evaluator",
            "cross_target_view_matching_possible": False,
            "needs_modification": False,
            "training_loss": False,
            "formal_evaluation": True,
        },
        {
            "file": "scripts/historical_0324_adapter.py",
            "function": "run_current_evaluator",
            "current_image_id_construction": "delegates to corrected eval_instance_lsm_protocol",
            "cross_target_view_matching_possible": False,
            "needs_modification": False,
            "training_loss": False,
            "formal_evaluation": True,
        },
    ]
    _write(OUT / "callsite_inventory.json", {
        "protocol": "TokenGS posed ScanNet-LSM40 per-target-view AP protocol v1",
        "entries": callsites,
        "instance_ap_formula_changed": False,
        "training_scene_hungarian_changed": False,
    })
    _write(OUT / "lsm40_manifest_evidence.json", {
        "manifest_path": str(MANIFEST.resolve()),
        "manifest_sha256": _sha256(MANIFEST),
        "source_split_path": payload.get("source_split_path"),
        "source_split_sha256": payload.get("source_split_sha256"),
        "scene_count": len(scenes),
        "scene_ids_in_manifest_order": list(scenes),
        "scenes": scene_evidence,
        "target_view_count": len(image_ids),
        "unique_target_view_image_id_count": len(set(image_ids)),
        "context_views": 8,
        "target_views": 7,
        "train_eval_intersection_checked": False,
        "camera_source": "ScanNet .sens intrinsics/poses",
        "shared_colmap_cameras": False,
        "official_instok3d_evaluator_parity": False,
    })


if __name__ == "__main__":
    main()
