#!/usr/bin/env python3
"""Atomically link verified small metadata files into SIU3R's data layout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


FILES = ("README.md", "val_pair.json", "val_refer_pair.json", "val_refer_seg_data.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        source = args.stage / "scannet" / name
        destination = args.data_root / name
        if not source.is_file():
            raise SystemExit(f"missing verified staged metadata: {source}")
        if destination.exists() or destination.is_symlink():
            if destination.is_file() and os.path.samefile(source, destination):
                continue
            raise SystemExit(f"refusing to overwrite existing data metadata: {destination}")
        temp = destination.with_name(destination.name + ".tmp")
        os.link(source, temp)
        os.replace(temp, destination)
    print("materialized=" + str(args.data_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
