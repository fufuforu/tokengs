#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/siu3r_delivery_common.sh"
parse_delivery_args "$@"
prepare_workspace "tokengs_j2_local250_full1860"
write_metadata

PAIR_FILE="${SIU3R_VAL_PAIR:-$ROOT/workspace/siu3r_protocol_alignment_v1/val_pair.json}"
DATA_DIR="${SIU3R_DATA_DIR:-/space/mawb/SIU3R/data/scannet}"
CHECKPOINT="${TOKEN_GS_CHECKPOINT:-$ROOT/workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_ddp8/checkpoints/model_step_000250.safetensors}"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "missing official processed ScanNet data at $DATA_DIR; no 112GB download started" >&2
  exit 3
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing TokenGS checkpoint: $CHECKPOINT" >&2
  exit 3
fi
INPUT_VIEWS="$($PYTHON_BIN - "$ROOT/workspace/semantic_v6_absolute_units_true_shared_token_eru1_dino_metric_joint_formation_j2_ddp8/config.yaml" <<'PY'
import re, sys
text = open(sys.argv[1], encoding='utf-8').read()
match = re.search(r'^num_input_views:\s*(\d+)', text, re.M)
print(match.group(1) if match else '-1')
PY
)"
if [[ "$INPUT_VIEWS" != "2" ]]; then
  echo "STRICT_SIU3R_INPUT_VIEW_PARITY: NO" >&2
  echo "RETRAINING_OR_VARIABLE_VIEW_SUPPORT_REQUIRED: YES" >&2
  echo "formal 1860-pair evaluation not started" >&2
  exit 4
fi
PREDICTIONS_DIR="${SIU3R_TOKENGS_PREDICTIONS_DIR:-$DELIVERY_WORKSPACE/predictions}"
exec "$PYTHON_BIN" "$ROOT/scripts/eval_tokengs_siu3r_protocol.py" \
  --pairs "$PAIR_FILE" --predictions-dir "$PREDICTIONS_DIR" \
  --output "$DELIVERY_WORKSPACE/results.json" --device "${SIU3R_DEVICE:-cuda}" \
  --allow-formal-1860
