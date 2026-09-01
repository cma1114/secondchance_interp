#!/usr/bin/env bash
set -euo pipefail

cd /root/secondchance_interp

base="outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback"
runner=(python -m mechanistic.run_decision_semantic_ablation
  --config "configs/qwen36_27b_simplemc_token_matched_feedback_routes.json"
  --baseline-logits "$base/baseline_results.json"
  --first-readout 24
  --last-readout 55
  --anchor line_end)

PYTHONPATH=. "${runner[@]}" \
  --plan "$base/gdn_source_layer_decomposition/discovery_plan.json" \
  --option-roots \
    "$base/contextual_option_representations/discovery_original" \
    "$base/contextual_option_representations/four_mapping_average/discovery_map1" \
    "$base/contextual_option_representations/four_mapping_average/discovery_map2" \
    "$base/contextual_option_representations/four_mapping_average/discovery_map3" \
  --output "$base/decision_semantic_ablation/discovery" \
  2>&1 | tee "$base/decision_semantic_ablation/discovery.log"

PYTHONPATH=. "${runner[@]}" \
  --plan "$base/gdn_source_layer_decomposition/confirmation_plan.json" \
  --option-roots \
    "$base/contextual_option_representations/confirmation_original" \
    "$base/contextual_option_representations/confirmation_remapped" \
    "$base/contextual_option_representations/four_mapping_average/confirmation_map2" \
    "$base/contextual_option_representations/four_mapping_average/confirmation_map3" \
  --output "$base/decision_semantic_ablation/confirmation" \
  2>&1 | tee "$base/decision_semantic_ablation/confirmation.log"

PYTHONPATH=. python -m mechanistic.analyze_decision_semantic_ablation \
  --results \
    "$base/decision_semantic_ablation/discovery" \
    "$base/decision_semantic_ablation/confirmation" \
  --baseline-results "$base/baseline_results.json" \
  --output "$base/decision_semantic_ablation/analysis" \
  2>&1 | tee "$base/decision_semantic_ablation/analysis.log"
