#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/secondchance_interp}"
cd "$ROOT"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

BASE="outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback"
OUT="$BASE/contextual_option_representations"
SPLIT="$BASE/gdn_source_layer_decomposition"
REMAPPED="outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping"
CONFIG="configs/qwen36_27b_simplemc_token_matched_feedback_test.json"

python -u -m mechanistic.collect_contextual_option_representations \
  --config "$CONFIG" \
  --question-plan "$SPLIT/discovery_plan.json" \
  --output "$OUT/discovery_original" \
  --batch-size 4

python -u -m mechanistic.collect_contextual_option_representations \
  --config "$CONFIG" \
  --question-plan "$SPLIT/confirmation_plan.json" \
  --output "$OUT/confirmation_original" \
  --batch-size 4

python -u -m mechanistic.collect_contextual_option_representations \
  --config "$CONFIG" \
  --question-plan "$SPLIT/confirmation_plan.json" \
  --remapping-plan "$REMAPPED/plan.json" \
  --output "$OUT/confirmation_remapped" \
  --batch-size 4

python -u -m mechanistic.analyze_contextual_option_cross_mapping \
  --discovery-original "$OUT/discovery_original" \
  --confirmation-original "$OUT/confirmation_original" \
  --confirmation-remapped "$OUT/confirmation_remapped" \
  --baseline-results "$BASE/baseline_results.json" \
  --remapped-baseline-results "$REMAPPED/remapped_baseline_results.json" \
  --output "$OUT/analysis" \
  --seed 20260810

echo complete > "$OUT/cross_mapping_complete.txt"
