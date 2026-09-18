#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/siu3r_delivery_common.sh"
parse_delivery_args "$@"
ensure_siu3r_gpu_allocation "$0"
prepare_workspace "official_full1860"
write_siu3r_gpu_probe
write_metadata

PAIR_FILE="${SIU3R_VAL_PAIR:-/space/mawb/SIU3R/data/scannet/val_pair.json}"
DATA_DIR="${SIU3R_DATA_DIR:-/space/mawb/SIU3R/data/scannet}"
CHECKPOINT="${SIU3R_OFFICIAL_CHECKPOINT:-/space/mawb/SIU3R/pretrained_weights/siu3r_epoch100.ckpt}"
OFFICIAL_PYTHON="${SIU3R_OFFICIAL_PYTHON:-${PYTHON_BIN:-/space/mawb/SIU3R/.venv/bin/python}}"
[[ -d "$DATA_DIR" && -f "$PAIR_FILE" && -f "$CHECKPOINT" ]] || { echo "missing official data, pair file, or checkpoint" >&2; exit 3; }
[[ "$(stat -c '%s' "$CHECKPOINT")" == "5464307091" ]] || { echo "checkpoint size mismatch" >&2; exit 3; }
[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "0c6b3e6eac8a44a864ec98c07b417aabcce4a12e510bed5f46c77afed74e46a0" ]] || { echo "checkpoint SHA256 mismatch" >&2; exit 3; }
[[ "$(sha256sum "$PAIR_FILE" | awk '{print $1}')" == "59cf5594ec2f3223a41d27d7af4da49d348ce9b00c1d151020378c9a8f4b4b3b" ]] || { echo "val_pair SHA256 mismatch" >&2; exit 3; }
exec "$OFFICIAL_PYTHON" "$ROOT/scripts/run_siu3r_official_full1860.py" \
  --data-dir "$DATA_DIR" --pairs "$PAIR_FILE" --checkpoint "$CHECKPOINT" \
  --output "$DELIVERY_WORKSPACE/official_metrics_raw.json" --log "$DELIVERY_WORKSPACE/official_full1860.log" \
  --partition "$PARTITION" --pair0-gate "${SIU3R_PAIR0_GATE:-/space/mawb/tokengs/workspace/siu3r_official_val_reference_v1/pair0/pair0_numerical_parity.json}"
