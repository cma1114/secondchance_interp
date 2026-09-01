#!/usr/bin/env bash
set -euo pipefail

cd /root/secondchance_interp
root="outputs/model_replications/mcq_uncertainty_policy"
mkdir -p "$root/logs"

echo "PHASE qwen_intervention START $(date -Iseconds)"
/opt/conda/bin/python -m mechanistic.run_qwen36_mcq_uncertainty_intervention \
  --specs "$root/specs/qwen36_27b.json" \
  2>&1 | tee "$root/logs/qwen_intervention_full.log"
echo "PHASE qwen_intervention COMPLETE $(date -Iseconds)"

echo "PHASE seed_intervention START $(date -Iseconds)"
/opt/conda/bin/python -m mechanistic.run_seed_oss_mcq_uncertainty_intervention \
  --specs "$root/specs/seed_oss_36b.json" \
  2>&1 | tee "$root/logs/seed_intervention_full.log"
echo "PHASE seed_intervention COMPLETE $(date -Iseconds)"

echo "ALL_CAUSAL_PHASES_COMPLETE $(date -Iseconds)"
