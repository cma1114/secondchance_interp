#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/sublayer_qwen36_simplemc.json}"
NATURAL_ROOT="${2:-outputs/mechanistic/qwen36_27b_simplemc_sublayers}"
SCREEN_DIR="${3:-outputs/causal/qwen36_27b_simplemc_components/screen}"
PATCH_ROOT="${4:-outputs/causal/qwen36_27b_simplemc_components/patches}"
ANALYSIS_DIR="${5:-outputs/causal/qwen36_27b_simplemc_components/analysis}"

nvidia-smi
python -m mechanistic.collect_sublayers --config "$CONFIG_PATH"
python -m mechanistic.analyze_sublayers --config "$CONFIG_PATH" --output "$SCREEN_DIR"
python -m mechanistic.run_component_patching --config "$CONFIG_PATH" --plan "$SCREEN_DIR/component_patch_plan.json" --output "$PATCH_ROOT"
python -m mechanistic.analyze_component_patching --natural-root "$NATURAL_ROOT" --patch-root "$PATCH_ROOT" --plan "$SCREEN_DIR/component_patch_plan.json" --output "$ANALYSIS_DIR"
