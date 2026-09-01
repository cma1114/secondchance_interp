#!/usr/bin/env bash
set -euo pipefail

cd /root/secondchance_interp

run_logged() {
  local output_dir="$1"
  shift
  mkdir -p "$output_dir"
  "$@" 2>&1 | tee "$output_dir/full.log"
}

run_logged \
  outputs/model_replications/canonical_history_decision_factorial/qwen36_27b/simplemc/run \
  python -m mechanistic.run_qwen36_canonical_history_decision_factorial \
  --config configs/qwen36_27b_simplemc_token_matched_feedback_test.json \
  --trusted-behavior outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/run/simplemc/results.npz \
  --output-dir outputs/model_replications/canonical_history_decision_factorial/qwen36_27b/simplemc/run

run_logged \
  outputs/model_replications/canonical_history_decision_factorial/qwen36_27b/triviamc/run \
  python -m mechanistic.run_qwen36_canonical_history_decision_factorial \
  --config configs/qwen36_27b_triviamc_difficulty_filtered_token_matched.json \
  --trusted-behavior outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/run/triviamc/results.npz \
  --output-dir outputs/model_replications/canonical_history_decision_factorial/qwen36_27b/triviamc/run

run_logged \
  outputs/model_replications/canonical_history_decision_factorial/seed_oss_36b/simplemc/run \
  python -m mechanistic.run_canonical_history_decision_factorial \
  --config configs/seed_oss_36b_simplemc_clean_gate.json \
  --trusted-behavior outputs/model_replications/seed_oss_36b_clean_behavioral_replication/simplemc/run/results.json \
  --output-dir outputs/model_replications/canonical_history_decision_factorial/seed_oss_36b/simplemc/run

run_logged \
  outputs/model_replications/canonical_history_decision_factorial/seed_oss_36b/triviamc/run \
  python -m mechanistic.run_canonical_history_decision_factorial \
  --config configs/seed_oss_36b_triviamc_clean_gate.json \
  --trusted-behavior outputs/model_replications/seed_oss_36b_clean_behavioral_replication/triviamc/run/results.json \
  --output-dir outputs/model_replications/canonical_history_decision_factorial/seed_oss_36b/triviamc/run

run_logged \
  outputs/model_replications/canonical_history_decision_factorial/gemma4_31b/simplemc/run \
  .venv_gemma55/bin/python -m mechanistic.run_gemma4_canonical_history_decision_factorial \
  --config configs/gemma4_31b_simplemc_clean_gate.json \
  --trusted-behavior outputs/model_replications/gemma4_31b_negative_model_comparison/simplemc/behavior/run/results.json \
  --output-dir outputs/model_replications/canonical_history_decision_factorial/gemma4_31b/simplemc/run

run_logged \
  outputs/model_replications/canonical_history_decision_factorial/gemma4_31b/triviamc/run \
  .venv_gemma55/bin/python -m mechanistic.run_gemma4_canonical_history_decision_factorial \
  --config configs/gemma4_31b_triviamc_clean_gate.json \
  --trusted-behavior outputs/model_replications/gemma4_31b_negative_model_comparison/triviamc/behavior/run/results.json \
  --output-dir outputs/model_replications/canonical_history_decision_factorial/gemma4_31b/triviamc/run
