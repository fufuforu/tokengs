#!/usr/bin/env bash
# Create m0/m4 causal-fork workspaces from a shared step-125 warm-up state.
#
#   bash scripts/fork_true_shared_from_warmup.sh SOURCE M4_DEST M0_DEST
#
# Existing destination directories are never overwritten.
set -euo pipefail
repo="/space/mawb/tokengs"
if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 SOURCE M4_DEST M0_DEST" >&2
  exit 2
fi
warm="$1"
m4ws="$repo/$2"
m0ws="$repo/$3"
for d in "${m4ws}" "${m0ws}"; do
  if [[ -e "${d}" ]]; then
    echo "refusing to overwrite existing fork workspace: ${d}" >&2
    exit 2
  fi
done
cp -a "${warm}" "${m4ws}"
cp -a "${warm}" "${m0ws}"
echo "forked from ${warm} ->"
echo "  m4: ${m4ws}"
echo "  m0: ${m0ws}"
