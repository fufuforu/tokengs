#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/space0/mawb/anaconda3/envs/tokengs/bin/python}"
WORKSPACE="workspace/scannet_joint_v7"
STEP="${1:-12000}"

echo "[run_joint_v7] training workspace=${WORKSPACE} steps=${STEP}"
"$PYTHON_BIN" scripts/train_scannet_joint.py \
  --workspace "$WORKSPACE" \
  --num_steps "$STEP" \
  --data_mode scannet_lsm_style_train \
  --windows_per_scene 4 \
  --num_input_views 8 \
  --num_views 15 \
  --num_groups 128 \
  --lr_instance 1e-4 \
  --lr_semantic 1e-5 \
  --lr_geometry 3e-5 \
  --warmup_steps 1000 \
  --lambda_rgb 1.0 \
  --lambda_boundary_rgb 0.1 \
  --lambda_instance_3d 1.0 \
  --unfreeze_mode decoder \
  --ckpt_freq 1000 \
  --log_freq 10

echo "[run_joint_v7] instance eval"
"$PYTHON_BIN" scripts/eval_instance_lsm_protocol.py \
  --resume "${WORKSPACE}/checkpoints/model_step_${STEP}.safetensors" \
  --workspace "${WORKSPACE}_instance_eval_${STEP}" \
  --label "v7_step${STEP}" \
  --model_type semantic_tokengs_v6 \
  --num_groups 128

echo "[run_joint_v7] semantic eval"
"$PYTHON_BIN" scripts/eval_c3g_semantic_v3.py \
  --resume "${WORKSPACE}/checkpoints/model_step_${STEP}.safetensors" \
  --workspace "${WORKSPACE}_semantic_eval_${STEP}" \
  --label "v7_step${STEP}"
