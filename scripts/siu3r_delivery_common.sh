#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${TOKENGS_PYTHON:-/space/mawb/anaconda3/envs/tokengs/bin/python}"
PARTITION="${SIU3R_PARTITION:-3090}"
NODE_NAME="${SIU3R_NODE:-}"
DELIVERY_WORKSPACE=""
DELIVERY_POSITIONAL=()

parse_delivery_args() {
  DELIVERY_POSITIONAL=()
  while (($#)); do
    case "$1" in
      --partition) PARTITION="$2"; shift 2 ;;
      --node) NODE_NAME="$2"; shift 2 ;;
      --workspace) DELIVERY_WORKSPACE="$2"; shift 2 ;;
      --python) PYTHON_BIN="$2"; shift 2 ;;
      *) DELIVERY_POSITIONAL+=("$1"); shift ;;
    esac
  done
  export PARTITION NODE_NAME PYTHON_BIN DELIVERY_WORKSPACE
  export PYTHONPATH="$ROOT${PYTHONPATH:+:${PYTHONPATH}}"
}

ensure_siu3r_gpu_allocation() {
  # Delivery scripts may be invoked from a login node.  Re-exec exactly once
  # inside Slurm; an existing allocation is reused to avoid nested srun.
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    return 0
  fi
  local script="$1"
  local qscript qroot qpartition qpython qnode qworkspace
  printf -v qscript '%q' "$script"
  printf -v qroot '%q' "$ROOT"
  printf -v qpartition '%q' "$PARTITION"
  printf -v qpython '%q' "$PYTHON_BIN"
  printf -v qnode '%q' "$NODE_NAME"
  printf -v qworkspace '%q' "$DELIVERY_WORKSPACE"
  local inner="cd $qroot && exec $qscript --partition $qpartition --python $qpython"
  if [[ -n "$NODE_NAME" ]]; then inner+=" --node $qnode"; fi
  if [[ -n "$DELIVERY_WORKSPACE" ]]; then inner+=" --workspace $qworkspace"; fi
  echo "requesting Slurm GPU allocation: partition=$PARTITION node=${NODE_NAME:-auto}" >&2
  local cmd=(srun -p "$PARTITION")
  [[ -n "$NODE_NAME" ]] && cmd+=(-w "$NODE_NAME")
  cmd+=(--gpus-per-task=1 --ntasks=1 --cpus-per-task=8 --mem=64G bash -lc "$inner")
  exec "${cmd[@]}"
}

write_siu3r_gpu_probe() {
  local probe_log="$DELIVERY_WORKSPACE/gpu_preflight.log"
  {
    echo "hostname=$(hostname)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    nvidia-smi -L
    "$PYTHON_BIN" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"device_name={torch.cuda.get_device_name(0)}")
    x = torch.randn(16, 16, device="cuda")
    print(f"cuda_tensor_finite={bool(torch.isfinite(x).all())}")
PY
  } 2>&1 | tee "$probe_log"
}

prepare_workspace() {
  local requested="${DELIVERY_WORKSPACE:-$ROOT/workspace/siu3r_delivery/$1}"
  if [[ -e "$requested" ]]; then
    echo "refusing to overwrite existing delivery workspace: $requested" >&2
    exit 2
  fi
  mkdir -p "$requested"
  DELIVERY_WORKSPACE="$requested"
  export DELIVERY_WORKSPACE
}

write_metadata() {
  local metadata="$DELIVERY_WORKSPACE/run_metadata.txt"
  local tmp="${metadata}.tmp"
  {
    echo "partition=$PARTITION"
    echo "tokengs_commit=$(git -C "$ROOT" rev-parse HEAD)"
    echo "tokengs_status=$(git -C "$ROOT" status --short | tr '\n' ';')"
    echo "checkpoint=$ROOT/workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_ddp8/checkpoints/model_step_000250.safetensors"
    sha256sum "$ROOT/workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_ddp8/checkpoints/model_step_000250.safetensors" 2>/dev/null || true
    echo "val_pair=$ROOT/workspace/siu3r_protocol_alignment_v1/val_pair.json"
    sha256sum "$ROOT/workspace/siu3r_protocol_alignment_v1/val_pair.json" 2>/dev/null || true
    "$PYTHON_BIN" -c 'import importlib.metadata as m,sys; print("python="+sys.version.split()[0]); d={x.metadata.get("Name","").lower():x.version for x in m.distributions()}; [print(n+"="+d.get(n,"missing")) for n in ("torch","torchvision","torchmetrics","lpips")]' 2>&1 || true
  } > "$tmp"
  mv "$tmp" "$metadata"
}

trap 'status=$?; echo "failure_status=$status" >> "$DELIVERY_WORKSPACE/run_metadata.txt" 2>/dev/null || true; exit "$status"' EXIT
