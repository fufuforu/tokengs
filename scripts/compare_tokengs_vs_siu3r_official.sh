#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/siu3r_delivery_common.sh"

parse_delivery_args "$@"
prepare_workspace "tokengs_vs_official_compare"
write_metadata

if [[ "${#DELIVERY_POSITIONAL[@]}" -lt 2 ]]; then
  echo "usage: $0 [--partition 3090|4090] [--workspace DIR] OFFICIAL_RESULTS.json TOKEN_GS_RESULTS.json" >&2
  exit 2
fi
OFFICIAL="${DELIVERY_POSITIONAL[0]}"
TOKENGS="${DELIVERY_POSITIONAL[1]}"
[[ -f "$OFFICIAL" ]] || { echo "missing official results: $OFFICIAL" >&2; exit 3; }
[[ -f "$TOKENGS" ]] || { echo "missing TokenGS results: $TOKENGS" >&2; exit 3; }
"$PYTHON_BIN" - "$OFFICIAL" "$TOKENGS" "$DELIVERY_WORKSPACE/comparison.json" <<'PY'
import json, os, sys
official, tokengs, output = sys.argv[1:]
left, right = json.load(open(official)), json.load(open(tokengs))
if left.get("protocol") != "siu3r_global_multiview_v1" or right.get("protocol") != "siu3r_global_multiview_v1":
    raise SystemExit("comparison requires SIU3R global multiview results on both sides")
payload = {
    "protocol": "siu3r_global_multiview_v1",
    "comparison_label": "SIU3R_DATA_AND_EVALUATOR_ALIGNED_POSED_TOKENGS",
    "official": left,
    "tokengs": right,
    "strict_end_to_end_siu3r_parity": False,
    "known_non_comparable_factors": ["TokenGS uses posed context camera inputs", "current J2 is 8-context, not official 2-context", "current J2 lacks native 20-class class-aware logits"],
}
tmp = output + ".tmp"
open(tmp, "w").write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
os.replace(tmp, output)
PY
