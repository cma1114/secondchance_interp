#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/causal_feedback_steering_qwen36_simplemc.json}"
SMOKE_QUESTIONS="${2:-2}"

if [[ ! -x .venv-mechinterp/bin/python ]]; then
  echo "Missing .venv-mechinterp; run scripts/bootstrap_mechinterp_host.sh first." >&2
  exit 1
fi

nvidia-smi
.venv-mechinterp/bin/python -m mechanistic.prompt_audit \
  --config configs/mechinterp_qwen36_simplemc.json
.venv-mechinterp/bin/python -m mechanistic.run_steering \
  --config "$CONFIG_PATH" \
  --smoke-questions "$SMOKE_QUESTIONS"
