#!/usr/bin/env python3
"""Parallel wrapper for the isolated, safe SIU3R val archive extractor."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from extract_siu3r_val_archives import sha256, validate_members, validate_scene


def extract_one(archive: Path, records: dict[str, dict], val_root: Path) -> dict:
    scene = archive.name.removesuffix(".tar.gz")
    relative = f"scannet/val/{archive.name}"
    record = records.get(relative)
    if record is None or record.get("expected_sha256") is None:
        raise RuntimeError(f"missing immutable source digest for {relative}")
    archive_sha = sha256(archive)
    if archive.stat().st_size != record["expected_size"] or archive_sha != record["expected_sha256"]:
        raise RuntimeError(f"archive validation failed before extraction: {archive}")
    destination = val_root / scene
    marker = destination / ".siu3r_scene_complete.json"
    if destination.exists():
        if not marker.is_file():
            raise RuntimeError(f"refusing to overwrite existing unmarked scene: {destination}")
        old = json.loads(marker.read_text())
        if old.get("archive_sha256") != archive_sha or old.get("integrity_status") != "complete":
            raise RuntimeError(f"existing scene marker does not match source: {destination}")
        check = validate_scene(destination)
        if check["integrity_status"] != "complete":
            raise RuntimeError(f"existing marked scene is incomplete: {destination}")
        return {"archive_path": str(archive), "archive_sha256": archive_sha, "extracted_path": str(destination), **check, "complete_marker": str(marker), "action": "reused"}
    temp = Path(tempfile.mkdtemp(prefix=f".{scene}.extract-", dir=val_root))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            validate_members(tar, scene, temp)
            tar.extractall(temp)
        extracted = temp / scene
        check = validate_scene(extracted)
        if check["integrity_status"] != "complete":
            raise RuntimeError(f"required scene content missing: {check}")
        marker = extracted / ".siu3r_scene_complete.json"
        marker.write_text(json.dumps({"archive_sha256": archive_sha, "integrity_status": "complete", "created_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
        check = validate_scene(extracted)
        if destination.exists():
            raise RuntimeError(f"destination appeared during extraction: {destination}")
        extracted.rename(destination)
        temp.rmdir()
        return {"archive_path": str(archive), "archive_sha256": archive_sha, "extracted_path": str(destination), **check, "complete_marker": str(destination / marker.name), "action": "extracted"}
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads((args.result / "download_manifest.json").read_text())
    records = {item["relative_path"]: item for item in manifest["records"]}
    archives = sorted(path for path in (args.stage / "scannet/val").glob("scene*.tar.gz") if path.is_file())
    if len(archives) != 312:
        raise SystemExit(f"refusing extraction: expected 312 val archives, found {len(archives)}")
    val_root = args.data_root / "val"
    val_root.mkdir(parents=True, exist_ok=True)
    output = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {pool.submit(extract_one, archive, records, val_root): archive for archive in archives}
        for future in as_completed(future_map):
            output.append(future.result())
    output.sort(key=lambda item: item["extracted_path"])
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "data_root": str(args.data_root), "scene_count": len(output), "scenes": output, "all_complete": len(output) == 312 and all(item["integrity_status"] == "complete" for item in output)}
    target = args.result / "extraction_manifest.json"
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(target)
    print(json.dumps({"scene_count": len(output), "all_complete": payload["all_complete"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
