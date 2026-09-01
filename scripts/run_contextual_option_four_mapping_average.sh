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
AVG="$OUT/four_mapping_average"
SPLIT="$BASE/gdn_source_layer_decomposition"
REMAPPED="outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping"
CONFIG="configs/qwen36_27b_simplemc_token_matched_feedback_test.json"

python -u -m mechanistic.prepare_contextual_option_four_mapping_plans \
  --existing-plan "$REMAPPED/plan.json" \
  --output-two "$AVG/mapping_2_plan.json" \
  --output-three "$AVG/mapping_3_plan.json"

# The original mapping and map 1 confirmation residuals are already complete.
# Collect map 1 discovery and the two complementary mappings on both splits.
python -u -m mechanistic.collect_contextual_option_representations \
  --config "$CONFIG" --question-plan "$SPLIT/discovery_plan.json" \
  --remapping-plan "$REMAPPED/plan.json" --output "$AVG/discovery_map1" --batch-size 4

for INDEX in 2 3; do
  python -u -m mechanistic.collect_contextual_option_representations \
    --config "$CONFIG" --question-plan "$SPLIT/discovery_plan.json" \
    --remapping-plan "$AVG/mapping_${INDEX}_plan.json" \
    --output "$AVG/discovery_map${INDEX}" --batch-size 4
  python -u -m mechanistic.collect_contextual_option_representations \
    --config "$CONFIG" --question-plan "$SPLIT/confirmation_plan.json" \
    --remapping-plan "$AVG/mapping_${INDEX}_plan.json" \
    --output "$AVG/confirmation_map${INDEX}" --batch-size 4
done

python -u -m mechanistic.analyze_contextual_option_four_mapping_average \
  --discovery-roots \
    "$OUT/discovery_original" "$AVG/discovery_map1" "$AVG/discovery_map2" "$AVG/discovery_map3" \
  --confirmation-roots \
    "$OUT/confirmation_original" "$OUT/confirmation_remapped" \
    "$AVG/confirmation_map2" "$AVG/confirmation_map3" \
  --baseline-results "$BASE/baseline_results.json" \
  --output "$OUT/analysis" --seed 20260810

echo complete > "$AVG/complete.txt"
