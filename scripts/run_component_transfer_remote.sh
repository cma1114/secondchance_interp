#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/sublayer_qwen36_triviamc.json}"
SOURCE_PLAN="${2:-outputs/causal/qwen36_27b_simplemc_causal_sweep/plans/confirmation_plan.json}"
MANIFEST="${3:-outputs/reproduction/triviamc_qwen36_27b/stimulus_manifest.json}"
RUN_ROOT="${4:-outputs/causal/qwen36_27b_triviamc_component_transfer}"

SMOKE_PLAN="$RUN_ROOT/plans/smoke_plan.json"
TRANSFER_PLAN="$RUN_ROOT/plans/transfer_plan.json"
SMOKE_PATCHES="$RUN_ROOT/smoke/patches"
SMOKE_ANALYSIS="$RUN_ROOT/smoke/analysis"
TRANSFER_PATCHES="$RUN_ROOT/transfer/patches"
TRANSFER_ANALYSIS="$RUN_ROOT/transfer/analysis"

nvidia-smi

python -m mechanistic.prepare_component_transfer \
  --source-plan "$SOURCE_PLAN" \
  --manifest "$MANIFEST" \
  --output "$SMOKE_PLAN" \
  --max-questions 4

python -m mechanistic.run_component_patching \
  --config "$CONFIG_PATH" \
  --plan "$SMOKE_PLAN" \
  --output "$SMOKE_PATCHES"

python -m mechanistic.analyze_component_causal_sweep \
  --natural-root "$SMOKE_PATCHES" \
  --patch-root "$SMOKE_PATCHES" \
  --plan "$SMOKE_PLAN" \
  --output "$SMOKE_ANALYSIS" \
  --bootstrap-samples 500

python -m mechanistic.prepare_component_transfer \
  --source-plan "$SOURCE_PLAN" \
  --manifest "$MANIFEST" \
  --output "$TRANSFER_PLAN"

python -m mechanistic.run_component_patching \
  --config "$CONFIG_PATH" \
  --plan "$TRANSFER_PLAN" \
  --output "$TRANSFER_PATCHES"

python -m mechanistic.analyze_component_causal_sweep \
  --natural-root "$TRANSFER_PATCHES" \
  --patch-root "$TRANSFER_PATCHES" \
  --plan "$TRANSFER_PLAN" \
  --output "$TRANSFER_ANALYSIS"
