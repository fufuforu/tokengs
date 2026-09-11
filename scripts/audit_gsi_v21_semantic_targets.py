"""Read-only audit of ScanNet semantic targets on the formal 8+7 data path."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml

# Make direct ``python scripts/...py`` invocation resolve this checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokengs.data import get_multi_dataloader
from tokengs.models.globalsplat_instance_v2.scene_instance_loss import (
    collect_scene_instances,
)
from tokengs.options import config_defaults


PROTOCOL_PATH = REPO_ROOT / "configs" / "semantic" / "scannet_c3g8.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "workspace" / "gsi_v21_semantic_target_audit_v1"
IGNORE_IDS = (0, 255, -1)


class _AuditAccelerator:
    is_main_process = True

    @staticmethod
    def print(*args: Any, **kwargs: Any) -> None:
        print(*args, **kwargs)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _unique_output(path: Path) -> Path:
    if not path.exists() or not any(path.iterdir()):
        path.mkdir(parents=True, exist_ok=True)
        return path
    stem = path.name
    parent = path.parent
    version = 2
    while True:
        candidate = parent / f"{stem.rsplit('_v', 1)[0]}_v{version}"
        if not candidate.exists() or not any(candidate.iterdir()):
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        version += 1


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _scene_name(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise RuntimeError(f"Expected batch size 1 scene_name, got {value!r}")
        value = value[0]
    return str(value)


def _frame_ids(value: Any) -> list[int]:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 2:
        if tensor.shape[0] != 1:
            raise RuntimeError(f"Expected batch size 1 frame_ids, got {tuple(tensor.shape)}")
        tensor = tensor[0]
    return [int(item) for item in tensor.reshape(-1).tolist()]


def _protocol_metadata(dataset_provider: Any) -> dict[str, Any]:
    dataset = dataset_provider.dataset
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    classes = sorted(protocol["classes"], key=lambda item: int(item["train_id"]))
    raw_to_contiguous: dict[int, int] = {0: 0}
    duplicate_raw_ids: dict[int, list[int]] = {}
    raw_ids_by_class: dict[str, list[int] | str] = {}
    for item in classes:
        train_id = int(item["train_id"])
        name = str(item["name"])
        if "raw_ids" in item:
            raw_ids = [int(raw_id) for raw_id in item["raw_ids"]]
            raw_ids_by_class[name] = raw_ids
            for raw_id in raw_ids:
                if raw_id in raw_to_contiguous and raw_to_contiguous[raw_id] != train_id:
                    duplicate_raw_ids.setdefault(raw_id, []).append(train_id)
                raw_to_contiguous[raw_id] = train_id
        else:
            raw_ids_by_class[name] = "remaining_nonzero"
    lut = getattr(dataset, "_label_lut", None)
    if lut is None:
        raise RuntimeError("Formal ScanNet provider has no label LUT")
    names = tuple(str(item) for item in getattr(dataset, "semantic_class_names", ()))
    if names != tuple(str(item["name"]) for item in classes):
        raise RuntimeError(
            f"Provider class names disagree with {PROTOCOL_PATH}: {names!r}"
        )
    contiguous_lut = {str(raw_id): int(label) for raw_id, label in enumerate(lut.tolist())}
    return {
        "semantic_label_space": str(protocol["name"]),
        "source_label_space": str(protocol["source_label_space"]),
        "num_semantic_classes": len(names),
        "class_names": list(names),
        "raw_ids_by_class": raw_ids_by_class,
        "raw_to_contiguous_explicit": {str(k): v for k, v in sorted(raw_to_contiguous.items())},
        "provider_label_lut": contiguous_lut,
        "unknown_nonzero_train_id": int(getattr(dataset, "_unknown_label_id", -1)),
        "ignore_ids": list(IGNORE_IDS),
        "ignore_index": int(protocol["ignore_index"]),
        "thing_classes": [
            int(item) for item in protocol["training"]["instance_query_class_ids"]
        ],
        "stuff_classes": [
            int(item)
            for item in range(1, len(names) + 1)
            if int(item) not in protocol["training"]["instance_query_class_ids"]
        ],
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "duplicate_explicit_raw_ids": duplicate_raw_ids,
    }


def _audit_sample(batch: dict[str, Any], split: str, index: int, protocol: dict[str, Any]) -> dict[str, Any]:
    required = (
        "instance_label_output",
        "semantic_label_output",
        "scene_name",
        "frame_ids",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise RuntimeError(f"{split} batch is missing required fields: {missing}")

    instance_labels = batch["instance_label_output"]
    semantic_labels = batch["semantic_label_output"]
    if instance_labels.ndim != 4 or semantic_labels.ndim != 4:
        raise RuntimeError(
            f"Expected batched [B,7,H,W] labels, got instance={tuple(instance_labels.shape)} "
            f"semantic={tuple(semantic_labels.shape)}"
        )
    if instance_labels.shape[0] != 1 or instance_labels.shape[1] != 7:
        raise RuntimeError(f"Expected one sample with 7 target views, got {tuple(instance_labels.shape)}")
    if semantic_labels.shape != instance_labels.shape:
        raise RuntimeError("Instance and semantic target shapes differ")

    scene = _scene_name(batch["scene_name"])
    frames = _frame_ids(batch["frame_ids"])
    if len(frames) != 15:
        raise RuntimeError(f"Expected 8+7 frame IDs for {scene}, got {frames}")
    if len(set(frames)) != 15:
        raise RuntimeError(f"Duplicate context/target frame IDs for {scene}: {frames}")

    ids = instance_labels[0].long().cpu()
    sem = semantic_labels[0].long().cpu()
    object_ids, masks, visible, _ambiguous, dropped = collect_scene_instances(ids)
    classes = []
    valid_count = 0
    no_legal_count = 0
    low_purity_count = 0
    conflict_count = 0
    for object_id in object_ids:
        total = 0
        counts: dict[int, int] = {}
        per_view_modes: list[int] = []
        for view in range(ids.shape[0]):
            pixels = masks[object_id][view]
            legal = pixels & ~torch.isin(sem[view], torch.tensor(list(IGNORE_IDS)))
            legal &= (sem[view] >= 1) & (sem[view] <= protocol["num_semantic_classes"])
            labels = sem[view][legal]
            if labels.numel():
                unique, counts_view = torch.unique(labels, return_counts=True)
                mode_index = int(torch.argmax(counts_view).item())
                per_view_modes.append(int(unique[mode_index].item()))
                for label, count in zip(unique.tolist(), counts_view.tolist()):
                    counts[int(label)] = counts.get(int(label), 0) + int(count)
                total += int(labels.numel())
        if total == 0:
            semantic_class = -1
            purity = 0.0
            semantic_valid = False
            no_legal_count += 1
        else:
            semantic_class = max(counts, key=counts.get)
            purity = float(counts[semantic_class] / total)
            semantic_valid = purity >= 0.95
            if semantic_valid:
                valid_count += 1
            else:
                low_purity_count += 1
        if len(set(per_view_modes)) > 1:
            conflict_count += 1
        classes.append(
            {
                "instance_id": int(object_id),
                "semantic_class": int(semantic_class),
                "semantic_class_name": (
                    protocol["class_names"][semantic_class - 1]
                    if 1 <= semantic_class <= protocol["num_semantic_classes"]
                    else None
                ),
                "semantic_valid": bool(semantic_valid),
                "semantic_purity": purity,
                "visible_views": list(visible[object_id]),
                "per_view_modal_classes": per_view_modes,
                "pixel_counts": {str(key): int(value) for key, value in sorted(counts.items())},
            }
        )
    return {
        "split": split,
        "sample_index": index,
        "scene_id": scene,
        "context_frame_ids": frames[:8],
        "target_frame_ids": frames[8:],
        "instance_count": len(object_ids),
        "dropped_instance_ids": list(dropped),
        "instances": classes,
        "semantic_valid_count": valid_count,
        "semantic_invalid_count": len(object_ids) - valid_count,
        "no_legal_semantic_count": no_legal_count,
        "low_purity_count": low_purity_count,
        "cross_view_class_conflict_count": conflict_count,
    }


def _collect(loader: Any, split: str, protocol: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, batch in enumerate(loader):
        record = _audit_sample(batch, split, index, protocol)
        if record["scene_id"] in {item["scene_id"] for item in records}:
            continue
        records.append(record)
        if len(records) >= limit:
            break
    if len(records) < limit:
        raise RuntimeError(
            f"Only collected {len(records)} unique {split} scenes, need {limit}"
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples_per_split", type=int, default=8)
    args = parser.parse_args()
    output_dir = _unique_output(args.output_dir)

    base = config_defaults["gsi_v2_joint_scannet_short355_ddp8"]
    audit_opt = base.evolve(
        num_workers=0,
        batch_size=1,
        use_instance_labels=True,
        evaluating=False,
    )
    train_loader, test_loader, train_dataset, test_dataset = get_multi_dataloader(
        audit_opt, _AuditAccelerator()
    )
    if len(train_dataset.datasets) != 1 or len(test_dataset.datasets) != 1:
        raise RuntimeError("Expected exactly one formal ScanNet dataset in each split")
    protocol = _protocol_metadata(train_dataset.datasets[0])
    train_records = _collect(train_loader, "train", protocol, args.samples_per_split)
    val_records = _collect(test_loader, "validation", protocol, args.samples_per_split)

    all_records = train_records + val_records
    total_instances = sum(item["instance_count"] for item in all_records)
    total_valid = sum(item["semantic_valid_count"] for item in all_records)
    total_conflicts = sum(item["cross_view_class_conflict_count"] for item in all_records)
    conflict_denominator = total_instances
    valid_ratio = total_valid / total_instances if total_instances else 0.0
    conflict_ratio = total_conflicts / conflict_denominator if conflict_denominator else 0.0
    audit_valid = bool(
        total_instances > 0
        and valid_ratio >= 0.99
        and conflict_ratio < 0.01
        and not protocol["duplicate_explicit_raw_ids"]
    )
    result = {
        "semantic_target_audit_valid": audit_valid,
        "implementation_started": False,
        "data_config_name": "gsi_v2_joint_scannet_short355_ddp8",
        "data_mode": _jsonable(audit_opt.data_mode),
        "num_input_views": int(audit_opt.num_input_views),
        "num_views": int(audit_opt.num_views),
        "formal_dataloader": "tokengs.data.get_multi_dataloader",
        "protocol": protocol,
        "counts": {
            "train_samples": len(train_records),
            "validation_samples": len(val_records),
            "total_instances": total_instances,
            "semantic_valid_instances": total_valid,
            "semantic_valid_ratio": valid_ratio,
            "cross_view_conflict_instances": total_conflicts,
            "cross_view_conflict_ratio": conflict_ratio,
            "no_legal_semantic_instances": sum(item["no_legal_semantic_count"] for item in all_records),
            "low_purity_instances": sum(item["low_purity_count"] for item in all_records),
        },
        "train": train_records,
        "validation": val_records,
    }
    (output_dir / "semantic_target_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "semantic_target_audit_valid": audit_valid,
        "semantic_valid_ratio": valid_ratio,
        "cross_view_conflict_ratio": conflict_ratio,
        "total_instances": total_instances,
    }, indent=2))
    if not audit_valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
