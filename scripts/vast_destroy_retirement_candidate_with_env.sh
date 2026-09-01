#!/usr/bin/env bash
set -euo pipefail

set -a
source .env
set +a
export XDG_STATE_HOME=/tmp/secondchance-vast-state
exec .venv/bin/vastai --api-key "$SPAR_SPRING_EXTENSION_2026_VAST_KEY" destroy instance 48128594 --yes
