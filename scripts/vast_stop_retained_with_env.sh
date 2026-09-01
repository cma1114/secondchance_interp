#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(48055971|48128594)$ ]]; then
  echo "usage: $0 {48055971|48128594}" >&2
  exit 2
fi

instance_id="$1"
set -a
source .env
set +a
export XDG_STATE_HOME=/tmp/secondchance-vast-state
exec .venv/bin/vastai --api-key "$SPAR_SPRING_EXTENSION_2026_VAST_KEY" stop instance "$instance_id"
