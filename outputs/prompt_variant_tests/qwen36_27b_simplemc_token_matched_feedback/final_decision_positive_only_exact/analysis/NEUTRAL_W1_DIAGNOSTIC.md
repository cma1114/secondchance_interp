# Why did Neutral sometimes choose W1 more after W1 ablation?

## Answer

Most of the apparent paradox came from calling signed projection-zeroing an ablation. The original intervention set `h · v_W1` to zero. When that projection was negative, it *added* the W1-defined direction rather than removing it. The positive-only rerun leaves negative projections untouched.

After that correction, there is no stable Neutral W1 increase. Across all 273 W1 != W2 questions, positive-only removal changed Neutral W1 selection by +0.7 pp (95% CI -2.6 to +4.0): 13 questions entered W1 and 11 left it. Discovery moved away from W1 (-2.2 pp (95% CI -7.3 to +2.9)); confirmation moved toward it (+3.7 pp (95% CI -0.7 to +8.1)).

The original signed intervention produced a larger pooled increase of +3.3 pp (95% CI -1.1 to +7.7). Because that increase shrinks when the only rule change is leaving negative projections untouched, it depends substantially on also moving negative projections toward zero. The two nonlinear interventions should not be treated as an additive decomposition.

## What happened to the logits?

Positive-only removal did not raise W1 evidence on average in Neutral. Pooled centered A-D logit changes were:

- W1: -0.037
- W2: -0.055
- Mean of the other two options: +0.046

Thus W1 rose on a few individual boundary cases, but the average W1 score fell. W2 fell somewhat more, while ranks 3-4 rose. The five-net-question confirmation increase is therefore a heterogeneous redistribution near decision boundaries, not a general causal boost to W1.

## Interpretation

The semantic reference vector is not a monotonic W1-evidence axis. It is a layer-specific direction constructed from contextual option-newline states. Removing it at every layer changes downstream computation nonlinearly and remains heterogeneous by original answer letter. It is causally involved in answer computation, but neither signed zeroing nor positive-only removal can be interpreted as simply deleting the model's memory of W1.

The most defensible conclusion is therefore: **Neutral choosing W1 more is not a replicated mechanism requiring a special explanation. It is mainly a signed-intervention artifact plus a small, split-unstable set of boundary crossings.**

## Files

- Figure: `figures/qwen36_27b_simplemc_corrected/neutral_w1_ablation_diagnostic.png`
- Numerical diagnostic: `neutral_w1_diagnostic.json`
- Per-question transitions: `../data/per_question_condition.csv`
