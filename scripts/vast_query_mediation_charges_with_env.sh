#!/usr/bin/env bash
set -euo pipefail

set -a
source .env
set +a
export XDG_STATE_HOME=/tmp/secondchance-vast-state
exec .venv/bin/vastai --api-key "$SPAR_SPRING_EXTENSION_2026_VAST_KEY" --raw show invoices-v1 \
  --charges \
  --charge-type instance \
  --start-date 2026-08-20 \
  --end-date 2026-08-21 \
  --limit 100
