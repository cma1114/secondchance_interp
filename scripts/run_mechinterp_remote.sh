#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/mechinterp_qwen36_simplemc.json}"
ANALYSIS_DIR="${2:-}"

nvidia-smi
python -m mechanistic.prompt_audit --config "$CONFIG_PATH"
python -m mechanistic.collect --config "$CONFIG_PATH"

ANALYSIS_ARGS=(--config "$CONFIG_PATH")
if [[ -n "$ANALYSIS_DIR" ]]; then ANALYSIS_ARGS+=(--output "$ANALYSIS_DIR"); fi
if [[ "${RUN_PROBES:-0}" == "1" ]]; then ANALYSIS_ARGS+=(--run-probes); fi
python -m mechanistic.run_analysis "${ANALYSIS_ARGS[@]}"
