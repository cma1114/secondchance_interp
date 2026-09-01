#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/attention_ablation_qwen36_simplemc.json}"
ANALYSIS_DIR="${2:-outputs/causal/qwen36_27b_simplemc_attention_edge/analysis}"

nvidia-smi
python -m mechanistic.run_attention_ablation --config "$CONFIG_PATH"
python -m mechanistic.analyze_attention_ablation --config "$CONFIG_PATH" --output-dir "$ANALYSIS_DIR"

