# Revision-semantic strength and switching

## Bottom line

Only the block-47 retry/alternative-action family (`again`, `retry`, `another`, `second`, and related tokens) shows a reproducible unadjusted relationship: questions with a stronger Evaluation-minus-Neutral GLA-47 retry write are more likely to avoid W1. The effect appears in both historical splits and both lenses, but is modest and its confidence interval includes no association after controlling for the fresh remapped-Baseline W1 margin. Block-42 incorrectness and block-43 replacement strength do not reliably distinguish switched from repeated trials.

## Frozen confirmation estimates

The score is the question-level contextual GLA change: Evaluation's after-minus-before semantic-family score minus Matched Neutral's corresponding change. Positive values mean the GLA expresses more of that semantic family in Evaluation.

| Lens | Readout | Switched minus repeated [95% CI] | AUC | Baseline-adjusted OR per SD [95% CI] |
|---|---|---:|---:|---:|
| J-lens | L42 evaluation | -0.019 [-0.045, +0.007] | 0.428 | 0.90 [0.67, 1.19] |
| J-lens | L43 replacement | +0.005 [-0.011, +0.022] | 0.528 | 1.10 [0.83, 1.46] |
| J-lens | L47 retry | +0.048 [+0.023, +0.075] | 0.625 | 1.34 [0.97, 1.84] |
| R-lens | L42 evaluation | -0.010 [-0.034, +0.015] | 0.458 | 1.02 [0.76, 1.36] |
| R-lens | L43 replacement | +0.004 [-0.010, +0.018] | 0.524 | 1.08 [0.81, 1.42] |
| R-lens | L47 retry | +0.055 [+0.021, +0.091] | 0.597 | 1.25 [0.91, 1.72] |

![Semantic strength association](/Users/christopherackerman/repos/secondchance_interp/figures/qwen36_revision_semantics_switch_association.png)

## Interpretation constraints

The token families and blocks were frozen from the aggregate vocabulary explorer before this question-level association was inspected. Discovery and confirmation use the historical 251/249 split. This remains observational: a positive association can show that the readout tracks revision behavior, but not that the English-token direction itself causes switching.

## Artifacts

- [Tidy association table](associations.csv)
- [Numerical summary](summary.json)
- [Question-level semantic scores](../run/results.npz)
