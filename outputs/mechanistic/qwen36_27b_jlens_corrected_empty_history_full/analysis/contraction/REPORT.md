# Scalar contraction test on corrected-prompt JLens evidence

Questions: 500. Fits are balanced across the generated Baseline answer letter and evaluated with 5-fold cross-fitting.

At each layer, the centered four-answer JLens vector in Game or Neutral is modeled as `condition = alpha * Baseline + residual`. Alpha below one is literal contraction. The cross-validated transformation fraction asks how much of the condition-minus-Baseline change is predicted by that one scalar; the residual is everything not predicted by contraction.

| Layer | Game alpha | Neutral alpha | Game change explained | Neutral change explained | Game inverse-rank slope explained |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.720 | 0.929 | 27.4% | 6.1% | -489.3% |
| 40 | 1.176 | 1.379 | 6.0% | 41.8% | 65.6% |
| 48 | 0.962 | 0.924 | 0.2% | 1.2% | 10.5% |
| 52 | 0.418 | 0.643 | 49.2% | 39.8% | 90.0% |
| 56 | 0.537 | 0.836 | 62.8% | 19.5% | 92.9% |
| 60 | 0.491 | 0.830 | 70.0% | 20.4% | 92.8% |
| 64 | 0.496 | 0.888 | 70.4% | 11.2% | 116.1% |

Interpretation rule: high change-explained values and a small residual rank slope support global gain reduction. A substantial residual inverse-rank slope means the transformation is more specifically rank-structured than scalar contraction predicts. Switching necessarily depends on the residual because positive scalar contraction alone cannot change the answer ordering.

## Full-residual variance control

We tested whether the apparent A–D contraction could instead be an incidental consequence of lower variance throughout the entire residual stream. At every layer, the analysis removes each condition's across-question mean and measures the weighted across-question variance relative to Baseline. Questions are weighted so that the four generated Baseline answer letters contribute equally. Uncertainty is estimated using 1,000 paired, answer-letter-stratified question-cluster bootstrap samples.

Four quantities are compared:

1. variance of the raw residual stream;
2. variance of the complete residual stream after transport through JLens and the model's final RMS normalization;
3. variance within the fixed three-dimensional A–D contrast subspace defined by the model's output rows; and
4. variance in the orthogonal complement of that A–D subspace.

Ratios below one indicate less across-question variance than Baseline.

| Layer | Game normalized full | Game A–D | Game complement | Neutral normalized full | Neutral A–D | Neutral complement |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1.245 | 1.070 | 1.245 | 0.719 | 0.695 | 0.719 |
| 40 | 0.555 | 0.617 | 0.555 | 0.699 | 0.702 | 0.699 |
| 48 | 0.407 | 0.340 | 0.407 | 0.548 | 0.456 | 0.548 |
| 52 | 0.458 | 0.267 | 0.458 | 0.605 | 0.518 | 0.605 |
| 56 | 0.517 | 0.366 | 0.520 | 0.719 | 0.779 | 0.718 |
| 60 | 0.651 | 0.323 | 0.658 | 0.808 | 0.821 | 0.807 |
| 64 | 1.300 | 0.336 | 1.314 | 1.322 | 0.892 | 1.328 |

The proposed global-variance explanation is partly correct: Game produces broad full-stream contraction across much of the late-middle model, and this contraction is generally stronger than Neutral's from approximately layers 40–60. But it is not sufficient. From the early 50s onward, Game compresses variance in the A–D contrast subspace substantially more than variance in the rest of the residual stream. The clearest dissociation is at layer 64: Game's normalized full-stream and orthogonal-complement variances are **higher** than Baseline (1.300 and 1.314), while its A–D variance remains only 0.336 of Baseline. Neutral shows much weaker answer-specific contraction at the same layer (0.892).

Thus, the best-supported description is a broad representational transformation/contraction during roughly layers 40–60, superimposed with a disproportionately strong and persistent contraction of answer-choice contrasts. The A–D result is not merely an artifact of globally reduced residual-stream variance.

Artifacts:

- Figure: `figures/qwen36_27b_simplemc_corrected/full_residual_variance.png`
- Compact estimates and confidence intervals: `outputs/mechanistic/qwen36_27b_jlens_corrected_empty_history_full/analysis/full_residual_variance/full_residual_variance_summary.json`
- Numerical arrays: `outputs/mechanistic/qwen36_27b_jlens_corrected_empty_history_full/analysis/full_residual_variance/full_residual_variance.npz`
- Reproduction script: `mechanistic/analyze_full_residual_variance.py`
