#!/usr/bin/env bash
# Evaluate the v6-wide 8000-step continuation checkpoints on the LSM
# 40-scene instance protocol (8 context / 7 test views).
set -euo pipefail

export PATH="/space0/mawb/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.4/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=8

cd /space0/mawb/tokengs
CKPT_DIR=workspace/semantic_v6_open_vocab_full_ce2_wide_train_8000/checkpoints

for STEP in "$@"; do
  python -u scripts/eval_instance_lsm_protocol.py \
    --resume "$CKPT_DIR/model_step_${STEP}.safetensors" \
    --workspace "workspace/lsm_instance_v6wide8000_${STEP}" \
    --label "v6wide8000_step${STEP}" \
    --num_groups 128
done
