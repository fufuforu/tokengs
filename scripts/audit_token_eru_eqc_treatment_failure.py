"""Read-only audit of the failed EQC E1 job 53619.

This script only parses Slurm artifacts and the failed workspace.  It never
loads a model, constructs an optimizer, or mutates the failed experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


JOB = "53619"
WORKSPACE = Path(
    "/space/mawb/tokengs/workspace/"
    "semantic_v6_j2_local250_eqc_v1_treatment_short200_ddp8"
)
PARENT = Path(
    "/space/mawb/tokengs/workspace/"
    "semantic_v6_absolute_units_true_shared_token_eru1_"
    "dino_metric_joint_formation_j2_ddp8/checkpoints/"
    "model_step_000250.safetensors"
)


def run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        return getattr(exc, "output", "") or f"command failed: {exc!r}"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_steps(text: str) -> dict[str, object]:
    values = [int(item) for item in re.findall(r"\[abs-train\] step=(\d+)", text)]
    samples = re.findall(r"\[abs-train\] step=(\d+) sample=([^ ]+)", text)
    return {
        "steps": sorted(set(values)),
        "last_step": max(values) if values else None,
        "last_sample": samples[-1][1] if samples else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError(f"refusing to overwrite audit report: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    stdout_path = WORKSPACE / "logs" / "slurm_53619.out"
    stderr_path = WORKSPACE / "logs" / "slurm_53619.err"
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
    combined = stdout + "\n" + stderr
    status = run(["sacct", "-j", JOB, "--format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList", "-P"])
    active_raw = run(["squeue", "-h", "-j", JOB, "-o", "%i %T %N %R"]).strip()
    active = "" if active_raw.startswith("slurm_load_jobs error:") else active_raw
    steps = parse_steps(stdout)
    rank_lines = re.findall(
        r"\[eqc-diagnostic\].*?\"rank\":\s*(\d+).*?\"step\":\s*(\d+).*?\"scene\":\s*\"([^\"]*)\"",
        stdout,
    )
    evidence = {
        "job_id": JOB,
        "workspace": str(WORKSPACE),
        "workspace_preserved": WORKSPACE.is_dir(),
        "job_accounting": status,
        "active_job_output": active_raw,
        "job_active": bool(active),
        "node": "3dimage-12",
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
        "last_logged_training_progress": steps,
        "rank_progress_records": [
            {"rank": int(rank), "step": int(step), "scene": scene or None}
            for rank, step, scene in rank_lines
        ],
        "first_abnormal_rank": None,
        "rank_progress_limitation": "The failed launcher did not enable rank-local trace; rank-specific last-step cannot be recovered from the combined stdout.",
        "watchdog": {
            "present": "watchdog got stuck" in combined,
            "sequence_number": None,
            "collective": None,
            "text": [line for line in stderr.splitlines() if "watchdog" in line or "ProcessGroupNCCL" in line],
        },
        "cuda_illegal_memory_access": bool(re.search(r"illegal memory|device-side assert|CUDA error", combined, re.I)),
        "dataloader_worker_failure": bool(re.search(r"DataLoader worker|worker.*exited|BrokenPipeError", combined, re.I)),
        "filesystem_checkpoint_failure": bool(re.search(r"checkpoint.*(failed|error)|No space left|I/O error", combined, re.I)),
        "unused_parameter_failure": bool(re.search(r"unused parameter|Expected to have finished reduction", combined, re.I)),
        "checkpoint_complete_files": sorted(path.name for path in (WORKSPACE / "checkpoints").glob("*.complete")) if (WORKSPACE / "checkpoints").is_dir() else [],
        "optimizer_step_files": sorted(path.name for path in (WORKSPACE / "checkpoints").glob("optimizer*") if path.is_file()) if (WORKSPACE / "checkpoints").is_dir() else [],
        "parent_checkpoint": {"path": str(PARENT), "sha256": sha256(PARENT)},
        "classification": "F_UNKNOWN",
        "classification_reason": "NCCL watchdog termination is proven, but the combined log has no rank-local heartbeat or collective trace to distinguish a rank stall from a transient NCCL/CUDA runtime hang.",
        "formal_training_started": False,
        "optimizer_step_executed_in_retry": False,
    }
    out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
