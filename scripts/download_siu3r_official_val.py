#!/usr/bin/env python3
"""Resumable, whitelist-only SIU3R validation downloader.

The script uses only the official Hugging Face resolve endpoint pinned to an
immutable revision.  It never requests scannet/train or the COCO checkpoint.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.parse
import urllib.request


REPO = "insomnia7/SIU3R"
WHITELIST = {
    "siu3r_epoch100.ckpt",
    "scannet/README.md",
    "scannet/val_pair.json",
    "scannet/val_refer_pair.json",
    "scannet/val_refer_seg_data.json",
}
REVISION = "65ad169493bfd26081f99ba26a5dc964aae40139"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def api_entries(cache_dir: Path | None = None) -> list[dict]:
    if cache_dir is not None:
        cached = [cache_dir / "hf_root_tree.json", cache_dir / "hf_scannet_tree.json", cache_dir / "hf_val_tree.json"]
        if all(path.is_file() for path in cached):
            entries: list[dict] = []
            for path in cached:
                entries.extend(json.loads(path.read_text(encoding="utf-8")))
            return entries
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true&expand=true"
    entries: list[dict] = []
    while url:
        request = urllib.request.Request(url, headers={"User-Agent": "siu3r-official-closure-v1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            page = json.load(response)
            link = response.headers.get("Link", "")
        entries.extend(page)
        url = None
        for item in link.split(","):
            if 'rel="next"' in item:
                url = item.split(";", 1)[0].strip().strip("<>")
                break
    return entries


def selected_entries(entries: list[dict]) -> list[dict]:
    selected = []
    for entry in entries:
        path = entry.get("path", "")
        if path in WHITELIST or path.startswith("scannet/val/"):
            if entry.get("type") == "file":
                selected.append(entry)
    selected.sort(key=lambda item: item["path"])
    return selected


def status_for(entry: dict, local: Path) -> dict:
    lfs = entry.get("lfs") or {}
    expected_size = int(lfs.get("size", entry.get("size", -1)))
    expected_sha = lfs.get("oid")
    record = {
        "repo_id": REPO,
        "hf_revision": REVISION,
        "relative_path": entry["path"],
        "expected_size": expected_size,
        "expected_sha256": expected_sha,
        "local_path": str(local),
        "size": local.stat().st_size if local.is_file() else None,
        "sha256": sha256(local) if local.is_file() and local.stat().st_size == expected_size else None,
        "download_status": "missing",
    }
    if local.is_file() and record["size"] == expected_size:
        record["download_status"] = "complete_sha_match" if expected_sha is None or record["sha256"] == expected_sha else "sha_mismatch"
    elif local.is_file():
        record["download_status"] = "size_mismatch"
    return record


def fetch(entry: dict, stage: Path, log_dir: Path) -> dict:
    relative = entry["path"]
    local = stage / relative
    local.parent.mkdir(parents=True, exist_ok=True)
    initial = status_for(entry, local)
    if initial["download_status"] in {"complete_sha_match", "complete_size_only"}:
        return initial
    if initial["download_status"] in {"sha_mismatch", "size_mismatch"}:
        raise RuntimeError(f"existing file fails source validation; refusing overwrite: {local}")

    part = local.with_name(local.name + ".part")
    if part.exists():
        existing = status_for(entry, part)
        # A previous HTTP response can leave an exact-length but corrupted
        # body.  Range=expected_size then returns HTTP 416 forever, so keep
        # the bad artifact for audit and restart this object from byte zero.
        if existing["size"] >= existing["expected_size"] and (
            not existing["expected_sha256"] or existing["sha256"] != existing["expected_sha256"]
        ):
            bad = part.with_name(part.name + f".invalid.{int(time.time())}")
            os.replace(part, bad)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{urllib.parse.quote(relative, safe='/')}?download=true"
    log = log_dir / (relative.replace("/", "__") + ".log")
    command = [
        "curl", "-fL", "--retry", "2", "--retry-connrefused", "--retry-delay", "3",
        "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "90", "-C", "-", "-o", str(part), url,
    ]
    result = None
    with log.open("ab") as handle:
        # Keep the official-host request rate low.  HF may return transient
        # 429/5xx responses for the val archive fan-out; the same .part file
        # remains resumable across invocations.
        for attempt in range(1, 7):
            handle.write((datetime.now(timezone.utc).isoformat() + f" attempt={attempt}\n" + " ".join(command) + "\n").encode())
            result = subprocess.run(command, stdout=handle, stderr=handle, check=False)
            if result.returncode == 0:
                break
    if result is None or result.returncode != 0:
        # curl can return 33/416 when a previous attempt already filled the
        # Range file exactly.  Validate that case before treating it as a
        # failure; never promote an unverified partial.
        complete = status_for(entry, part)
        if complete["size"] == complete["expected_size"] and (not complete["expected_sha256"] or complete["sha256"] == complete["expected_sha256"]):
            os.replace(part, local)
            return status_for(entry, local) | {"download_status": "downloaded_sha_match" if complete["expected_sha256"] else "downloaded_size_match"}
        raise RuntimeError(f"curl failed ({result.returncode}) for {relative}; resume with the same command")
    completed = status_for(entry, part)
    if completed["size"] != completed["expected_size"] or (completed["expected_sha256"] and completed["sha256"] != completed["expected_sha256"]):
        raise RuntimeError(f"download validation failed for {relative}: {completed}")
    os.replace(part, local)
    return status_for(entry, local) | {"download_status": "downloaded_sha_match" if completed["expected_sha256"] else "downloaded_size_match"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.stage.mkdir(parents=True, exist_ok=True)
    log_dir = args.result / "logs" / "download_files"
    log_dir.mkdir(parents=True, exist_ok=True)
    entries = selected_entries(api_entries(args.stage))
    if len(entries) != 317 or sum(e["path"].endswith(".tar.gz") for e in entries) != 312:
        raise SystemExit(f"unexpected whitelist inventory: files={len(entries)} archives={sum(e['path'].endswith('.tar.gz') for e in entries)}")
    started = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {pool.submit(fetch, entry, args.stage, log_dir): entry for entry in entries}
        for future in as_completed(future_map):
            entry = future_map[future]
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append({"relative_path": entry["path"], "error": repr(exc)})
    records.sort(key=lambda item: item["relative_path"])
    payload = {
        "repo_id": REPO,
        "hf_revision": REVISION,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "whitelist": sorted(WHITELIST) + ["scannet/val/**"],
        "archive_count": sum(item["relative_path"].endswith(".tar.gz") for item in records),
        "records": records,
        "failures": failures,
        "download_complete": not failures and all(item["download_status"].startswith(("complete", "downloaded")) for item in records),
    }
    args.result.mkdir(parents=True, exist_ok=True)
    tmp = args.result / "download_manifest.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, args.result / "download_manifest.json")
    print(json.dumps({"records": len(records), "failures": failures, "download_complete": payload["download_complete"]}, indent=2))
    return 0 if payload["download_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
