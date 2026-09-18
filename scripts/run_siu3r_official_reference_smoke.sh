#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/siu3r_delivery_common.sh"
parse_delivery_args "$@"
ensure_siu3r_gpu_allocation "$0"
prepare_workspace "official_reference_pair0_smoke"
write_siu3r_gpu_probe
write_metadata

PAIR_FILE="${SIU3R_VAL_PAIR:-/space/mawb/SIU3R/data/scannet/val_pair.json}"
DATA_DIR="${SIU3R_DATA_DIR:-/space/mawb/SIU3R/data/scannet}"
CHECKPOINT="${SIU3R_OFFICIAL_CHECKPOINT:-/space/mawb/SIU3R/pretrained_weights/siu3r_epoch100.ckpt}"

if [[ ! -d "$DATA_DIR" ]]; then
  {
    echo "OFFICIAL_PREPROCESSED_DATA_FOUND=NO"
    echo "missing_path=$DATA_DIR"
    echo "expected_size=approximately 112GB (per requested protocol handoff)"
    echo "resume_download=huggingface-cli download insomnia7/SIU3R --repo-type dataset --include 'scannet/**' --local-dir /space/mawb/SIU3R/data"
    echo "smoke=NOT_STARTED"
  } | tee "$DELIVERY_WORKSPACE/status.txt"
  exit 3
fi
if [[ ! -f "$PAIR_FILE" ]]; then
  echo "missing official val_pair.json: $PAIR_FILE" >&2
  exit 3
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "OFFICIAL_SIU3R_CHECKPOINT_FOUND=NO" | tee "$DELIVERY_WORKSPACE/status.txt"
  echo "set SIU3R_OFFICIAL_CHECKPOINT to a local official checkpoint; smoke not started" >&2
  exit 3
fi

exec "$PYTHON_BIN" "$ROOT/scripts/run_siu3r_official_pair0_closure.py" \
  --data-dir "$DATA_DIR" --pairs "$PAIR_FILE" --checkpoint "$CHECKPOINT" \
  --output "$DELIVERY_WORKSPACE/result.json" --log "$DELIVERY_WORKSPACE/official_pair0.log" --partition "$PARTITION"
