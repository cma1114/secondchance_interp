#!/usr/bin/env bash
set -euo pipefail

set -a
source .env
set +a
export XDG_STATE_HOME=/tmp/secondchance-vast-state
exec .venv/bin/vastai search offers \
  'gpu_name in [A100_SXM4,A100_PCIE] gpu_total_ram>=75 num_gpus<=2 reliability>=0.98 disk_space>=180 cuda_vers>=12.8 direct_port_count>=2' \
  --limit 20 --storage 180 --order 'dph' --raw \
  --api-key "$SPAR_SPRING_EXTENSION_2026_VAST_KEY"
