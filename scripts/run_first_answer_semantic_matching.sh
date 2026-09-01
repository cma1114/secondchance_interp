#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/workspace/secondchance_interp}"
cd "$ROOT"

export HF_HOME="${HF_HOME:-/workspace/hf}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

BASE="outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback"
SEMANTIC="$BASE/first_answer_semantic_matching"
SPLIT="$BASE/gdn_source_layer_decomposition"
REMAPPED="outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping"
CONFIG="configs/qwen36_27b_simplemc_token_matched_feedback_test.json"

python -u -m mechanistic.collect_first_presentation_semantic_residuals \
  --config "$CONFIG" \
  --question-plan "$SPLIT/discovery_plan.json" \
  --output "$SEMANTIC/discovery"

python -u -m mechanistic.collect_first_presentation_semantic_residuals \
  --config "$CONFIG" \
  --question-plan "$SPLIT/confirmation_plan.json" \
  --output "$SEMANTIC/confirmation"

python -u -m mechanistic.collect_first_presentation_semantic_residuals \
  --config "$CONFIG" \
  --question-plan "$SPLIT/confirmation_plan.json" \
  --remapping-plan "$REMAPPED/plan.json" \
  --output "$SEMANTIC/confirmation_remapped"

python -u -m mechanistic.analyze_first_answer_semantic_matching \
  --discovery "$SEMANTIC/discovery" \
  --confirmation "$SEMANTIC/confirmation" \
  --remapped-confirmation "$SEMANTIC/confirmation_remapped" \
  --baseline-results "$BASE/baseline_results.json" \
  --remapped-baseline-results "$REMAPPED/remapped_baseline_results.json" \
  --output "$SEMANTIC/analysis" \
  --seed 20260810
