#!/usr/bin/env bash
set -euo pipefail

cd /root/secondchance_interp
export PYTHONPATH=.
export HF_TOKEN="$(< /root/.hf_token)"

fixed_root=outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/fixed_a_donor_receiver_mediation
relay_root=outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/all_candidate_matched_relay_full_range
config=configs/qwen36_27b_simplemc_token_matched_feedback_test.json
cohort=outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_module_factorial/cohort_plan.json

mkdir -p "$fixed_root/discovery" "$fixed_root/confirmation" "$relay_root/run"

python -u -m mechanistic.run_fixed_a_donor_receiver_mediation \
  --config "$config" \
  --cohort "$cohort" \
  --output-dir "$fixed_root/discovery" \
  --split discovery \
  > "$fixed_root/discovery/full.log" 2>&1

python -u -m mechanistic.run_fixed_a_donor_receiver_mediation \
  --config "$config" \
  --cohort "$cohort" \
  --output-dir "$fixed_root/confirmation" \
  --split confirmation \
  > "$fixed_root/confirmation/full.log" 2>&1

python -u -m mechanistic.run_all_candidate_matched_relay_full_range \
  --config "$config" \
  --remapping-plan outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/plan.json \
  --baseline outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/baseline_results.json \
  --trusted-game outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/run/incorrect_again_results.json \
  --trusted-neutral outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/neutral_results.json \
  --output-dir "$relay_root/run" \
  > "$relay_root/run/full.log" 2>&1
