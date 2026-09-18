#!/usr/bin/env python3
"""Install only the SHA-verified official checkpoint via same-filesystem link."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


EXPECTED_SIZE = 5464307091
EXPECTED_SHA256 = "0c6b3e6eac8a44a864ec98c07b417aabcce4a12e510bed5f46c77afed74e46a0"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    if not args.stage.is_file():
        raise SystemExit(f"missing staged checkpoint: {args.stage}")
    actual_sha = sha256(args.stage)
    if args.stage.stat().st_size != EXPECTED_SIZE or actual_sha != EXPECTED_SHA256:
        raise SystemExit(f"checkpoint SHA/size mismatch; refusing load: size={args.stage.stat().st_size} sha256={actual_sha}")
    args.target.parent.mkdir(parents=True, exist_ok=True)
    if args.target.exists() or args.target.is_symlink():
        if args.target.is_file() and args.target.stat().st_size == EXPECTED_SIZE and sha256(args.target) == EXPECTED_SHA256:
            print("already_verified=" + str(args.target))
            return 0
        raise SystemExit(f"refusing to overwrite existing checkpoint: {args.target}")
    temp = args.target.with_name(args.target.name + ".tmp")
    os.link(args.stage, temp)
    os.replace(temp, args.target)
    print("installed_verified=" + str(args.target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
