# Complete feedback-end residual replacement — canonical prompt

Held-out questions: **249**. At the period ending the feedback sentence, the complete post-block residual was replaced with the paired same-question residual from the other condition. All prompts use the canonical explicit empty-history raw ChatML format.

Natural switching: Game **42.2%**; Neutral **28.1%**.

## Causal effects

| Direction | Window | Winner advantage | A–D spread | Entropy | Switch rate | Answers changed |
|---|---|---:|---:|---:|---:|---:|
| Neutral into Game | L1–16 | +0.036 [+0.017, +0.056] | +0.030 [+0.020, +0.039] | -0.015 [-0.020, -0.010] | +1.205 [-2.410, +4.819] pp | 26 |
| Neutral into Game | L17–32 | +0.061 [+0.035, +0.088] | +0.043 [+0.031, +0.054] | -0.021 [-0.029, -0.014] | -0.402 [-4.418, +3.614] pp | 27 |
| Neutral into Game | L33–40 | +0.099 [+0.084, +0.115] | +0.056 [+0.049, +0.062] | -0.026 [-0.030, -0.021] | -0.803 [-3.213, +1.606] pp | 12 |
| Neutral into Game | L41–48 | +0.031 [+0.019, +0.042] | +0.023 [+0.018, +0.028] | -0.011 [-0.014, -0.008] | +1.205 [-1.205, +3.614] pp | 11 |
| Neutral into Game | L49–64 | +0.036 [+0.027, +0.045] | +0.023 [+0.019, +0.027] | -0.010 [-0.012, -0.007] | -0.402 [-2.811, +2.008] pp | 12 |
| Neutral into Game | All | +0.108 [+0.076, +0.140] | +0.074 [+0.060, +0.088] | -0.036 [-0.044, -0.027] | -1.205 [-5.221, +2.811] pp | 30 |
| Game into Neutral | L1–16 | -0.006 [-0.020, +0.008] | -0.000 [-0.006, +0.006] | +0.000 [-0.003, +0.004] | +0.000 [-2.410, +2.410] pp | 10 |
| Game into Neutral | L17–32 | -0.005 [-0.018, +0.008] | -0.001 [-0.008, +0.005] | +0.001 [-0.002, +0.004] | +0.000 [-2.811, +2.811] pp | 15 |
| Game into Neutral | L33–40 | -0.052 [-0.064, -0.039] | -0.024 [-0.029, -0.019] | +0.007 [+0.005, +0.010] | +0.402 [-1.205, +2.410] pp | 7 |
| Game into Neutral | L41–48 | -0.019 [-0.028, -0.009] | -0.007 [-0.011, -0.003] | +0.003 [+0.000, +0.005] | -1.205 [-3.213, +0.803] pp | 8 |
| Game into Neutral | L49–64 | -0.023 [-0.031, -0.014] | -0.009 [-0.013, -0.006] | +0.003 [+0.001, +0.005] | +0.803 [-1.205, +3.213] pp | 9 |
| Game into Neutral | All | -0.030 [-0.048, -0.013] | -0.017 [-0.025, -0.009] | +0.007 [+0.003, +0.011] | -0.803 [-3.614, +2.008] pp | 15 |

## All-layer rank redistribution

| Direction | Rank 1 | Rank 2 | Rank 3 | Rank 4 |
|---|---:|---:|---:|---:|
| Neutral into Game | +0.081 | -0.018 | +0.005 | -0.067 |
| Game into Neutral | -0.023 | -0.002 | +0.009 | +0.015 |

## Interpretation

The complete feedback-end residual is a real causal carrier of continuous answer compression, but not of net switching. Replacing all Game feedback-end readouts with Neutral restores 13.6% of the natural winner-advantage gap, 23.4% of the spread gap, and 26.0% of the entropy gap. It mediates only 8.6% of the switch-rate gap, with a confidence interval spanning zero.

The effect is asymmetric. Inserting the Game feedback-end state into Neutral moves only 3.8% of the winner-advantage gap, 5.4% of the spread gap, and 5.1% of the entropy gap; switching moves slightly in the wrong direction. The strongest isolated source window is L33–40. This is more consistent with the Neutral state supplying sharpening that Game lacks than with a portable feedback-end compression command.

Figure: `figures/qwen36_27b_simplemc_corrected/feedback_end_residual_replacement.png`
