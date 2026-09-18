#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/siu3r_delivery_common.sh"
parse_delivery_args "$@"
prepare_workspace "tokengs_j2_pair0_smoke"
write_metadata

PAIR_FILE="${SIU3R_VAL_PAIR:-$ROOT/workspace/siu3r_protocol_alignment_v1/val_pair.json}"
DATA_DIR="${SIU3R_DATA_DIR:-/space/mawb/SIU3R/data/scannet}"
CHECKPOINT="${TOKEN_GS_CHECKPOINT:-$ROOT/workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_ddp8/checkpoints/model_step_000250.safetensors}"
CONFIG="$ROOT/workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_ddp8/config.yaml"

if [[ ! -f "$PAIR_FILE" || ! -d "$DATA_DIR" ]]; then
  echo "ONE_PAIR_SMOKE_VALID=NO" | tee "$DELIVERY_WORKSPACE/status.txt"
  echo "reason=official processed ScanNet validation data is unavailable; no 112GB download started" | tee -a "$DELIVERY_WORKSPACE/status.txt"
  exit 3
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing TokenGS checkpoint: $CHECKPOINT" >&2
  exit 3
fi

INPUT_VIEWS="$($PYTHON_BIN - "$CONFIG" <<'PY'
import re, sys
text = open(sys.argv[1], encoding='utf-8').read()
match = re.search(r'^num_input_views:\s*(\d+)', text, re.M)
print(match.group(1) if match else '-1')
PY
)"
if [[ "$INPUT_VIEWS" != "2" ]]; then
  {
    echo "STRICT_SIU3R_INPUT_VIEW_PARITY: NO"
    echo "RETRAINING_OR_VARIABLE_VIEW_SUPPORT_REQUIRED: YES"
    echo "TOKEN_GS_TWO_CONTEXT_FORWARD_SUPPORTED=NO"
    echo "smoke=NOT_STARTED"
  } | tee "$DELIVERY_WORKSPACE/status.txt"
  exit 4
fi

exec "$PYTHON_BIN" "$ROOT/scripts/run_tokengs_siu3r_pair0.py" \
  --pairs "$PAIR_FILE" --data-dir "$DATA_DIR" --checkpoint "$CHECKPOINT" \
  --output "$DELIVERY_WORKSPACE/result.json" --partition "$PARTITION"
