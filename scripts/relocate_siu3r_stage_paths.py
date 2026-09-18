#!/usr/bin/env python3
"""Atomically update generated closure manifests after stage relocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def replace(value, old: str, new: str):
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [replace(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: replace(item, old, new) for key, item in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    args = parser.parse_args()
    for name in ("download_manifest.json", "extraction_manifest.json"):
        path = args.result / name
        data = json.loads(path.read_text())
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(replace(data, args.old, args.new), indent=2, ensure_ascii=False) + "\n")
        tmp.replace(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
