#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 INSTANCE_ID" >&2
  exit 2
fi
instance_id="$1"
set -a
source .env
set +a
.venv/bin/python scripts/vast_fleet_guard.py prestart --intended-instance "$instance_id"
exec .venv/bin/vastai --api-key "$SPAR_SPRING_EXTENSION_2026_VAST_KEY" --raw start instance "$instance_id"
