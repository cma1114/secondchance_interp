#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/sublayer_qwen36_simplemc.json}"
NATURAL_ROOT="${2:-outputs/mechanistic/qwen36_27b_simplemc_sublayers}"
OLD_SPLIT_PLAN="${3:-outputs/causal/qwen36_27b_simplemc_components/screen/component_patch_plan.json}"
RUN_ROOT="${4:-outputs/causal/qwen36_27b_simplemc_causal_sweep}"

DISCOVERY_PLAN="$RUN_ROOT/plans/discovery_plan.json"
SMOKE_PLAN="$RUN_ROOT/plans/smoke_plan.json"
SMOKE_PATCHES="$RUN_ROOT/smoke/patches"
SMOKE_ANALYSIS="$RUN_ROOT/smoke/analysis"
DISCOVERY_PATCHES="$RUN_ROOT/discovery/patches"
DISCOVERY_ANALYSIS="$RUN_ROOT/discovery/analysis"
CONFIRMATION_PLAN="$RUN_ROOT/plans/confirmation_plan.json"
CONFIRMATION_PATCHES="$RUN_ROOT/confirmation/patches"
CONFIRMATION_ANALYSIS="$RUN_ROOT/confirmation/analysis"

nvidia-smi

python -m mechanistic.prepare_component_causal_sweep \
  --split-plan "$OLD_SPLIT_PLAN" \
  --natural-root "$NATURAL_ROOT" \
  --output "$SMOKE_PLAN" \
  --max-questions 4 \
  --max-components 4

python -m mechanistic.run_component_patching \
  --config "$CONFIG_PATH" \
  --plan "$SMOKE_PLAN" \
  --output "$SMOKE_PATCHES"

python -m mechanistic.analyze_component_causal_sweep \
  --natural-root "$NATURAL_ROOT" \
  --patch-root "$SMOKE_PATCHES" \
  --plan "$SMOKE_PLAN" \
  --output "$SMOKE_ANALYSIS" \
  --bootstrap-samples 500

python -m mechanistic.prepare_component_causal_sweep \
  --split-plan "$OLD_SPLIT_PLAN" \
  --natural-root "$NATURAL_ROOT" \
  --output "$DISCOVERY_PLAN"

python -m mechanistic.run_component_patching \
  --config "$CONFIG_PATH" \
  --plan "$DISCOVERY_PLAN" \
  --output "$DISCOVERY_PATCHES"

python -m mechanistic.analyze_component_causal_sweep \
  --natural-root "$NATURAL_ROOT" \
  --patch-root "$DISCOVERY_PATCHES" \
  --plan "$DISCOVERY_PLAN" \
  --output "$DISCOVERY_ANALYSIS"

python -m mechanistic.select_component_causal_candidates \
  --effects "$DISCOVERY_ANALYSIS/component_causal_effects.csv" \
  --discovery-plan "$DISCOVERY_PLAN" \
  --split-plan "$OLD_SPLIT_PLAN" \
  --output "$CONFIRMATION_PLAN" \
  --max-candidates 8

python -m mechanistic.run_component_patching \
  --config "$CONFIG_PATH" \
  --plan "$CONFIRMATION_PLAN" \
  --output "$CONFIRMATION_PATCHES"

python -m mechanistic.analyze_component_causal_sweep \
  --natural-root "$NATURAL_ROOT" \
  --patch-root "$CONFIRMATION_PATCHES" \
  --plan "$CONFIRMATION_PLAN" \
  --output "$CONFIRMATION_ANALYSIS"
