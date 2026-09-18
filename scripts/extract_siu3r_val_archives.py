#!/usr/bin/env python3
"""Safely extract verified SIU3R validation archives into official layout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


REQUIRED_DIRS = ("color", "depth", "extrinsic", "instance", "panoptic", "semantic")
REQUIRED_FILES = ("intrinsic.txt", "iou.pt", "iou.png")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_members(tar: tarfile.TarFile, scene: str, temp: Path) -> None:
    root = PurePosixPath(scene)
    for member in tar.getmembers():
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts:
            raise RuntimeError(f"archive path escape in {scene}: {member.name}")
        if not name.parts or name.parts[0] != scene:
            raise RuntimeError(f"archive member outside scene root {scene}: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"symlink/hardlink rejected in {scene}: {member.name}")
        candidate = (temp / Path(*name.parts)).resolve()
        if temp.resolve() not in candidate.parents and candidate != temp.resolve():
            raise RuntimeError(f"resolved archive path escapes temp dir: {member.name}")


def validate_scene(scene_dir: Path) -> dict:
    missing_dirs = [name for name in REQUIRED_DIRS if not (scene_dir / name).is_dir()]
    missing_files = [name for name in REQUIRED_FILES if not (scene_dir / name).is_file()]
    file_count = sum(1 for path in scene_dir.rglob("*") if path.is_file()) if scene_dir.is_dir() else 0
    return {
        "file_count": file_count,
        "required_directories": list(REQUIRED_DIRS),
        "missing_directories": missing_dirs,
        "missing_files": missing_files,
        "integrity_status": "complete" if not missing_dirs and not missing_files else "missing_required_content",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.result / "download_manifest.json").read_text())
    records = {item["relative_path"]: item for item in manifest["records"]}
    archives = sorted(path for path in (args.stage / "scannet/val").glob("scene*.tar.gz") if path.is_file())
    if len(archives) != 312:
        raise SystemExit(f"refusing extraction: expected 312 val archives, found {len(archives)}")
    val_root = args.data_root / "val"
    val_root.mkdir(parents=True, exist_ok=True)
    output = []
    for archive in archives:
        scene = archive.name.removesuffix(".tar.gz")
        relative = f"scannet/val/{archive.name}"
        record = records.get(relative)
        if record is None or record.get("expected_sha256") is None:
            raise SystemExit(f"missing immutable source digest for {relative}")
        archive_sha = sha256(archive)
        if archive.stat().st_size != record["expected_size"] or archive_sha != record["expected_sha256"]:
            raise SystemExit(f"archive validation failed before extraction: {archive}")
        destination = val_root / scene
        marker = destination / ".siu3r_scene_complete.json"
        if destination.exists():
            if not marker.is_file():
                raise SystemExit(f"refusing to overwrite existing unmarked scene: {destination}")
            old = json.loads(marker.read_text())
            if old.get("archive_sha256") != archive_sha or old.get("integrity_status") != "complete":
                raise SystemExit(f"existing scene marker does not match source; refusing overwrite: {destination}")
            check = validate_scene(destination)
            if check["integrity_status"] != "complete":
                raise SystemExit(f"existing marked scene is incomplete: {destination}")
            output.append({"archive_path": str(archive), "archive_sha256": archive_sha, "extracted_path": str(destination), **check, "complete_marker": str(marker), "action": "reused"})
            continue
        temp = Path(tempfile.mkdtemp(prefix=f".{scene}.extract-", dir=val_root))
        try:
            with tarfile.open(archive, "r:gz") as tar:
                validate_members(tar, scene, temp)
                tar.extractall(temp)
            extracted = temp / scene
            check = validate_scene(extracted)
            if check["integrity_status"] != "complete":
                raise RuntimeError(f"required scene content missing after extraction: {check}")
            marker = extracted / ".siu3r_scene_complete.json"
            marker.write_text(json.dumps({"archive_sha256": archive_sha, "integrity_status": "complete", "created_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
            check = validate_scene(extracted)
            if destination.exists():
                raise RuntimeError(f"destination appeared during extraction: {destination}")
            extracted.rename(destination)
            temp.rmdir()
            output.append({"archive_path": str(archive), "archive_sha256": archive_sha, "extracted_path": str(destination), **check, "complete_marker": str(destination / marker.name), "action": "extracted"})
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "data_root": str(args.data_root), "scene_count": len(output), "scenes": output, "all_complete": len(output) == 312 and all(item["integrity_status"] == "complete" for item in output)}
    target = args.result / "extraction_manifest.json"
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(target)
    print(json.dumps({"scene_count": len(output), "all_complete": payload["all_complete"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
