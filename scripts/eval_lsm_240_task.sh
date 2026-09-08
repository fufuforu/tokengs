#!/usr/bin/env bash
# 240-only per-task body of the parallel LSM eval launcher.
# Run by srun with one GPU per task; SLURM_PROCID maps to a checkpoint.
set -uo pipefail

rank="${SLURM_PROCID:?SLURM_PROCID is required}"
eval_mode="${EVAL_MODE:?EVAL_MODE is required}"
repo="${REPO_ROOT:?REPO_ROOT is required}"
py_bin="${TOKENG_PYTHON:?TOKENG_PYTHON is required}"
log_root="${EVAL_LOG_ROOT:?EVAL_LOG_ROOT is required}"
ws_root="${EVAL_WS_ROOT:?EVAL_WS_ROOT is required}"
manifest="${LSM_MANIFEST:?LSM_MANIFEST is required}"

IFS='|' read -r -a ckpts <<<"${EVAL_CKPT_LIST:?EVAL_CKPT_LIST is required}"
IFS=',' read -r -a step_gpus <<<"${SLURM_STEP_GPUS:-}"
ckpt="${ckpts[$rank]}"
gpu="${step_gpus[$rank]:-$rank}"
export CUDA_VISIBLE_DEVICES="${gpu}"

tag="${ckpt#model_step_}"
tag="${EVAL_LABEL_PREFIX:-recon_full3}_${tag}"
out_ws="${ws_root}/${tag}"
log_file="${log_root}/task${rank}_${tag}.log"
resume_dir="${EVAL_RESUME_DIR:?EVAL_RESUME_DIR is required}"
if [[ "${ckpt}" == "model_best" ]]; then
  resume_path="${resume_dir}/${ckpt}.safetensors"
else
  resume_path="${resume_dir}/checkpoints/${ckpt}.safetensors"
fi
mkdir -p "$out_ws" "$log_root"

{
  echo "[task] rank=${rank} slurm_gpus=${SLURM_STEP_GPUS:-} cuda_visible=${CUDA_VISIBLE_DEVICES}"
  echo "[task] checkpoint=${repo}/${resume_path}"
  echo "[task] workspace=${out_ws}"
  echo "[task] label=${tag} max_scenes=${EVAL_MAX_SCENES}"
  echo "[task] manifest=${manifest}"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  "${py_bin}" - <<'PY'
import torch
print(
    "[task] python-import-check torch=", torch.__version__,
    "cuda=", torch.version.cuda,
    "device_count=", torch.cuda.device_count(),
    "device=", torch.cuda.get_device_name(0),
    flush=True,
)
PY
} | tee "${log_file}"

cd "${repo}"
exec >>"${log_file}" 2>&1
exec "${py_bin}" -u scripts/eval_instance_lsm_protocol.py \
  --resume "${resume_path}" \
  --workspace "${out_ws}" \
  --label "${tag}" \
  --lsm_manifest "${manifest}" \
  --max_scenes "${EVAL_MAX_SCENES}"
