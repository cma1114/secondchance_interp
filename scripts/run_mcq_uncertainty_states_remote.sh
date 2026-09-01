#!/usr/bin/env bash
set -euo pipefail

cd /root/secondchance_interp
mkdir -p outputs/model_replications/mcq_uncertainty_policy/logs

python -m mechanistic.collect_qwen36_mcq_uncertainty_states \
  --specs outputs/model_replications/mcq_uncertainty_policy/specs/qwen36_27b.json \
  2>&1 | tee outputs/model_replications/mcq_uncertainty_policy/logs/qwen_states_full.log

python -m mechanistic.fit_mcq_uncertainty_directions \
  --specs outputs/model_replications/mcq_uncertainty_policy/specs/qwen36_27b.json \
  2>&1 | tee outputs/model_replications/mcq_uncertainty_policy/logs/qwen_directions.log

python -m mechanistic.collect_seed_oss_mcq_uncertainty_states \
  --specs outputs/model_replications/mcq_uncertainty_policy/specs/seed_oss_36b.json \
  2>&1 | tee outputs/model_replications/mcq_uncertainty_policy/logs/seed_states_full.log

python -m mechanistic.fit_mcq_uncertainty_directions \
  --specs outputs/model_replications/mcq_uncertainty_policy/specs/seed_oss_36b.json \
  2>&1 | tee outputs/model_replications/mcq_uncertainty_policy/logs/seed_directions.log
