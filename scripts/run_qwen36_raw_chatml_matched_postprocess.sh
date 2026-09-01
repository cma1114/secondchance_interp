#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

residual_root="outputs/mechanistic/qwen36_27b_simplemc_raw_chatml_matched"
jlens_root="outputs/mechanistic/qwen36_27b_jlens_raw_chatml_matched"
content_root="outputs/mechanistic/qwen36_27b_jlens_raw_chatml_matched_answer_content"
analysis_root="$jlens_root/analysis"
trajectory_root="$residual_root/analysis/trajectories"
probe_score_file="$trajectory_root/cross_fitted_candidate_probe_scores.npz"

PYTHONPATH=. python -m mechanistic.analyze_baseline_matched \
  --residual-root "$residual_root" \
  --output "$residual_root/analysis"

# Rebuild the native-lens and cross-fitted Baseline-probe figures from the new
# raw-serialized residuals.  In particular, do not reuse probe scores from any
# earlier HF-template run.
PYTHONPATH=. python -m mechanistic.all_trial_figures \
  --input "$residual_root" \
  --output "$trajectory_root/all_trials"

PYTHONPATH=. python -m mechanistic.condition_switch_figures \
  --input "$residual_root" \
  --output "$trajectory_root" \
  --pdf-output "$trajectory_root/pdfs" \
  --layer-step 2 \
  --folds 5 \
  --seed 42

PYTHONPATH=. python -m mechanistic.probe_mechanism_trajectories \
  --scores "$probe_score_file" \
  --input "$residual_root" \
  --output "$trajectory_root/probe_mechanism"

PYTHONPATH=. python -m mechanistic.prior_answer_probe_trajectories \
  --scores "$probe_score_file" \
  --input "$residual_root" \
  --output "$trajectory_root/prior_answer_probe"

PYTHONPATH=. python -m mechanistic.jlens_collect \
  --config configs/jlens_qwen36_simplemc_raw_chatml_matched.json \
  --residual-root "$residual_root" \
  --output "$jlens_root" \
  --position-sample 128 \
  --top-k 20 \
  --seed 42

PYTHONPATH=. python -m mechanistic.analyze_jlens \
  --jlens-root "$jlens_root" \
  --residual-root "$residual_root" \
  --output "$analysis_root"

PYTHONPATH=. python -m mechanistic.analyze_jlens_system_ablation \
  --jlens-root "$jlens_root" \
  --residual-root "$residual_root" \
  --output "$analysis_root"

PYTHONPATH=. python -m mechanistic.build_jlens_token_explorer \
  --source "$jlens_root/top_tokens.json" \
  --output "$analysis_root/jlens_unrestricted_token_explorer.html" \
  --exclude-system

PYTHONPATH=. python -m mechanistic.jlens_answer_content \
  --config configs/jlens_qwen36_simplemc_raw_chatml_matched.json \
  --residual-root "$residual_root" \
  --jlens-root "$jlens_root" \
  --output "$content_root" \
  --batch-size 16

PYTHONPATH=. python -m mechanistic.analyze_jlens_answer_content \
  --jlens-root "$jlens_root" \
  --content-root "$content_root" \
  --residual-root "$residual_root" \
  --output "$content_root/analysis"

PYTHONPATH=. python -m mechanistic.build_jlens_answer_content_explorer \
  --source "$content_root/analysis/answer_representation_trajectories.json" \
  --output "$content_root/analysis/jlens_answer_representation_explorer.html"

PYTHONPATH=. python -m mechanistic.analyze_jlens_baseline_contrasts \
  --residual-root "$residual_root" \
  --jlens-root "$jlens_root" \
  --output "$analysis_root/rank_contrasts"

PYTHONPATH=. python -m mechanistic.build_jlens_baseline_contrasts \
  --source "$analysis_root/rank_contrasts/jlens_baseline_contrasts.json" \
  --output "$analysis_root/rank_contrasts/jlens_fixed_rank_contrasts.html"
