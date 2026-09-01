#!/usr/bin/env bash
set -euo pipefail

set -a
source .env
set +a
export XDG_STATE_HOME=/tmp/secondchance-vast-state
exec .venv/bin/vastai create instance 39035793 \
  --image pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel \
  --disk 180 \
  --label qwen36-fixed-bcd-replacement \
  --ssh --direct --cancel-unavail \
  --api-key "$SPAR_SPRING_EXTENSION_2026_VAST_KEY"
