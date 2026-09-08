#!/usr/bin/env bash
# 240-only launcher: parallel LSM recon eval for the migrated recon_full3
# checkpoints on one 3090 node, one GPU per srun task.
#
#   bash scripts/eval_lsm_240_parallel.sh smoke   # 2 tasks x 1 scene
#   bash scripts/eval_lsm_240_parallel.sh full    # 5 tasks x 40 scenes
#
# Mapping (task rank -> checkpoint):
#   0 model_step_001000 | 1 model_step_005680 | 2 model_step_011360
#   3 model_step_017040 | 4 model_best
set -euo pipefail

mode="${1:-full}"
repo="/space/mawb/tokengs"
py_bin="/space/mawb/anaconda3/envs/tokengs/bin/python"

if [[ "${mode}" == "smoke" ]]; then
  tasks=2
  ckpt_list="model_step_001000|model_best"
  max_scenes=1
elif [[ "${mode}" == "full" ]]; then
  tasks=5
  ckpt_list="model_step_001000|model_step_005680|model_step_011360|model_step_017040|model_best"
  max_scenes=0
else
  echo "usage: $0 {smoke|full}" >&2
  exit 2
fi

manifest="${repo}/data/scannet_prompt/lsm_instance_eval_manifest.json"
ws_root="${EVAL_WS_ROOT:-${repo}/workspace/lsm_240_eval/${mode}}"
log_root="${EVAL_LOG_ROOT:-${repo}/workspace/lsm_240_eval/logs/${mode}}"
mkdir -p "${ws_root}" "${log_root}"

export REPO_ROOT="${repo}"
export TOKENG_PYTHON="${py_bin}"
export EVAL_MODE="${mode}"
export EVAL_CKPT_LIST="${ckpt_list}"
if [[ -n "${EVAL_CKPT_LIST_OVERRIDE:-}" ]]; then
  export EVAL_CKPT_LIST="${EVAL_CKPT_LIST_OVERRIDE}"
fi
export EVAL_MAX_SCENES="${max_scenes}"
export EVAL_LOG_ROOT="${log_root}"
export EVAL_WS_ROOT="${ws_root}"
export EVAL_RESUME_DIR="${EVAL_RESUME_DIR:-workspace/semantic_v6_absolute_units_recon_full3}"
export LSM_MANIFEST="${manifest}"
export CUDA_HOME="/space/mawb/.cuda126"
export CC="/space/mawb/anaconda3/envs/tokengs/bin/x86_64-conda-linux-gnu-gcc"
export CXX="/space/mawb/anaconda3/envs/tokengs/bin/x86_64-conda-linux-gnu-g++"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export FAST_COMPILE=1
export MAX_JOBS=8
export PATH="${py_bin%/*}:$PATH"
export PYTHONPATH="${repo}/240_shims:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/space/mawb/anaconda3/envs/tokengs/lib:/space/mawb/anaconda3/envs/tokengs/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/space/mawb/anaconda3/envs/tokengs/lib/python3.11/site-packages/nvidia/cudnn/lib:/space/mawb/anaconda3/envs/tokengs/lib/python3.11/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"

echo "[launcher] mode=${mode} tasks=${tasks} gpus-per-task=1 nodes=1"
echo "[launcher] checkpoint list: ${ckpt_list}"
echo "[launcher] workspace root: ${ws_root}"
echo "[launcher] log root: ${log_root}"

srun -p a6000 \
  --nodes=1 \
  --ntasks="${tasks}" \
  --gpus-per-task=1 \
  --gpu-bind=closest \
  --exclude=3dimage-11,3dimage-12 \
  --output="${log_root}/srun_task_%t.log" \
  --error="${log_root}/srun_task_%t.log" \
  bash "${repo}/scripts/eval_lsm_240_task.sh"
