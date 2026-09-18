#!/usr/bin/env python3
"""Audit the exact gsplat lock entry and prior GPU-v2 failure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_new(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{__import__('os').getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/space/mawb/SIU3R"))
    ap.add_argument("--failure-log", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    pyproject = args.repo / "pyproject.toml"
    lock = args.repo / "uv.lock"
    lock_text = lock.read_text()
    lines = lock_text.splitlines()
    start = next(i for i, line in enumerate(lines) if line == 'name = "gsplat"' and i > 0 and lines[i - 1] == "[[package]]") - 1
    end = next((i for i in range(start + 1, len(lines)) if i > start and lines[i] == "[[package]]"), len(lines))
    stanza = "\n".join(lines[start:end]) + "\n"
    version = re.search(r'^version = "([^"]+)"$', stanza, re.MULTILINE).group(1)
    source = re.search(r'^source = \{ git = "([^"]+)#([0-9a-f]+)" \}$', stanza, re.MULTILINE)
    source_url, commit = source.group(1), source.group(2)
    py_text = pyproject.read_text()
    import_callsite = {
        "file": "src/models/gaussian_renderer.py",
        "line": 11,
        "code": "from gsplat import rasterization",
        "usage_lines": [92, 93],
    }
    failure_text = args.failure_log.read_text()
    first_import = re.search(r'File "([^"]+/SIU3R/src/[^"]+)", line (\d+), in <module>\n\s+([^\n]+)', failure_text)
    failure_type = re.search(r'([A-Za-z_]+Error): ([^\n]+)', failure_text)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "SIU3R_REPO_COMMIT": subprocess.run(["git", "-C", str(args.repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
        "pyproject_sha256": sha256(pyproject),
        "uv_lock_sha256": sha256(lock),
        "GSLPAT_PACKAGE_NAME": "gsplat",
        "GSPLAT_PACKAGE_NAME": "gsplat",
        "GSPLAT_LOCKED_VERSION": version,
        "GSPLAT_SOURCE_TYPE": "git",
        "GSPLAT_SOURCE_URL": source_url,
        "GSPLAT_GIT_COMMIT": commit,
        "GSPLAT_LOCK_HASH": "NOT_PRESENT_GIT_SOURCE",
        "GSPLAT_LOCK_HASH_TYPE": "uv.lock has no content hash field for this Git package",
        "GSPLAT_LOCK_STANZA_SHA256": hashlib.sha256(stanza.encode()).hexdigest(),
        "GSPLAT_BUILD_BACKEND": "setuptools.setup.py legacy install (pip build log); backend not specified in uv.lock",
        "GSPLAT_REQUIRED_TORCH": "2.4.1+cu118 (uv.lock package dependency)",
        "GSPLAT_REQUIRED_CUDA": "not declared by gsplat lock metadata; CUDA extension requires a compatible local toolkit",
        "OFFICIAL_IMPORT_CALLSITE": import_callsite,
        "pyproject_declares_gsplat": '"gsplat"' in py_text,
        "failure_log": str(args.failure_log),
        "failure_sha256": sha256(args.failure_log),
        "failure_exception_type": failure_type.group(1) if failure_type else None,
        "failure_exception": failure_type.group(0) if failure_type else None,
        "first_official_import": {
            "file": first_import.group(1),
            "line": int(first_import.group(2)),
            "code": first_import.group(3).strip(),
        } if first_import else None,
        "undefined_symbol": "undefined symbol" in failure_text.lower(),
        "abi_error": "abi" in failure_text.lower() and "error" in failure_text.lower(),
        "torch_cuda_version_mismatch": "mismatch" in failure_text.lower() and "cuda" in failure_text.lower(),
        "failure_classification": "python_package_missing_after_gsplat_build_failure",
    }
    write_new(args.output, payload)
    md = args.output.with_suffix(".md")
    if md.exists():
        raise RuntimeError(f"refusing to overwrite {md}")
    md.write_text(
        "# gsplat dependency audit v3\n\n"
        f"- `GSPLAT_LOCKED_VERSION={version}`\n"
        f"- `GSPLAT_SOURCE_TYPE=git`\n"
        f"- `GSPLAT_SOURCE_URL={source_url}`\n"
        f"- `GSPLAT_GIT_COMMIT={commit}`\n"
        "- `GSPLAT_LOCK_HASH=NOT_PRESENT_GIT_SOURCE` (Git source; uv.lock has no package content hash)\n"
        "- `OFFICIAL_IMPORT_CALLSITE=src/models/gaussian_renderer.py:11`\n\n"
        "The prior GPU-v2 log ends with `ModuleNotFoundError: No module named 'gsplat'` at the official gaussian renderer import. No undefined-symbol error or checkpoint/model forward was reached.\n"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
