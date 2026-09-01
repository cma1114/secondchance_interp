# Remapped feedback-clause 2x2 factorial

## Design

The second presentation uses the same frozen option remapping in every condition. Only the evaluation word (`incorrect`/`lost`) and action instruction (`different`/`again`) vary. Unrestricted A-D output is primary.

| Condition | Exact feedback |
|---|---|
| `incorrect_different` | `Your answer was incorrect. Choose a different answer.` |
| `incorrect_again` | `Your answer was incorrect. Choose the answer again.` |
| `lost_different` | `Your answer was lost. Choose a different answer.` |
| `lost_again` | `Your answer was lost. Choose the answer again.` |

## Bottom line

The incorrectness evaluation is behaviorally sufficient for most of the semantic W1-avoidance effect even when the action clause merely says to answer again. Relative to `lost_again`, `incorrect_again` increases W1 avoidance by +15.2 [+11.0, +19.4] percentage points and lowers the centered W1 logit by -0.516 [-0.578, -0.455]. The full standard-Game contrast is +18.8 [+14.6, +23.2] percentage points, so the evaluation-only manipulation reproduces about 80.9% of that behavioral difference.

However, evaluation-only also raises A-D entropy by +0.136 [+0.111, +0.163] bits. The result therefore shows active semantic revision, but not entropy-free revision. The action-only comparison raises W1 avoidance by only +3.4 [+0.2, +6.6] points while raising entropy by +0.135 [+0.118, +0.154] bits. Thus the evaluation clause is the main driver of which semantic answer loses, whereas both clauses contribute to uncertainty.

On the 273 W1 != W2 questions, evaluation-only is almost behaviorally indistinguishable from standard Game on the key semantic endpoint: W1 is selected on 21.6% of evaluation-only trials versus 20.9% of standard-Game trials, compared with 39.2% under Neutral.

## All 500 questions

| Condition | W1 avoidance / content switch | W1 selection | Entropy (bits) | W1 centered logit |
|---|---:|---:|---:|---:|
| `incorrect_different` | 65.6 [61.6, 69.4]% | 34.4 [30.6, 38.4]% | 1.642 [1.612, 1.670] | +0.323 [+0.241, +0.405] |
| `incorrect_again` | 62.0 [58.0, 66.0]% | 38.0 [34.0, 42.0]% | 1.583 [1.549, 1.616] | +0.439 [+0.348, +0.530] |
| `lost_different` | 50.2 [45.8, 54.4]% | 49.8 [45.6, 54.0]% | 1.582 [1.548, 1.615] | +0.665 [+0.574, +0.754] |
| `lost_again` | 46.8 [42.6, 51.2]% | 53.2 [49.0, 57.6]% | 1.446 [1.405, 1.487] | +0.956 [+0.840, +1.076] |

## Decisive held-action comparisons

Effects are the first condition minus the second on the same questions.

| Contrast | W1 avoidance | Entropy | W1 centered logit |
|---|---:|---:|---:|
| `incorrect_again - lost_again` | +15.2 [+11.0, +19.4] pp | +0.136 [+0.111, +0.163] | -0.516 [-0.578, -0.455] |
| `lost_different - lost_again` | +3.4 [+0.2, +6.6] pp | +0.135 [+0.118, +0.154] | -0.291 [-0.339, -0.245] |

## W1 != W2 conflict questions (n=273)

| Contrast | W1 avoidance | W1 selection | W2 selection | Entropy |
|---|---:|---:|---:|---:|
| `evaluation_effect_when_action_again` | +17.6 [+12.1, +23.1] pp | -17.6 [-23.1, -12.1] pp | +8.4 [+1.8, +15.0] pp | +0.077 [+0.043, +0.110] |
| `action_effect_when_evaluation_lost` | +5.1 [+1.1, +9.2] pp | -5.1 [-9.2, -1.1] pp | +3.7 [-1.1, +8.4] pp | +0.117 [+0.097, +0.139] |
| `evaluation_effect_when_action_different` | +13.2 [+7.7, +18.3] pp | -13.2 [-18.7, -8.1] pp | +0.7 [-4.8, +6.2] pp | +0.015 [-0.013, +0.043] |
| `action_effect_when_evaluation_incorrect` | +0.7 [-3.3, +4.8] pp | -0.7 [-4.8, +3.3] pp | -4.0 [-8.8, +0.7] pp | +0.055 [+0.037, +0.074] |

## Interpretation rule

`incorrect_again - lost_again` directly tests whether the incorrectness evaluation is behaviorally sufficient without the different-answer instruction. It is large for W1 avoidance, but it also increases entropy; the experiment therefore supports independent semantic revision without establishing a purely entropy-neutral route.

Machine-readable results: `summary.json`; question-level data: `trial_table.csv`.

## Preferred follow-up

Use `incorrect_again` versus `lost_again` for the next mechanistic experiment.
At the period closing the evaluation clause, bidirectionally transplant the GLA
recurrent state updates—not merely the token residual—while keeping the shared
`Choose the answer again.` clause and repeated question fixed. First establish
an all-GLA positive control, then localize with eight-block bands and targeted
individual/leave-one-out tests. The decisive endpoints are transfer and rescue
of semantic W1 avoidance, with entropy reported as a separate outcome.

## Completed mechanistic follow-up

The planned GLA-update transplant succeeded. On all 273 W1 != W2 questions,
copying the Evaluation condition's all-GLA recurrent update at the
evaluation-closing period into Matched Neutral transferred 15.8 percentage
points of aggregated-A-D-argmax W1 avoidance (95% CI 11.0–20.5), or 86.0% of
the natural 18.3-point difference on that mechanistic endpoint. It transferred
0.428 logits (0.332–0.523) of the natural
0.469-logit W1-minus-W2 margin difference and 0.068 bits (0.040–0.097) of the
natural 0.077-bit entropy difference. The reverse transplant produced a
smaller but positive held-out margin rescue.

Blocks 17–24 were the only isolated band to replicate. No constituent GLA was
sufficient alone or a single-block bottleneck under leave-one-out testing.
See the [consolidated causal report](../evaluation_update_transplant/analysis/REPORT.md).
