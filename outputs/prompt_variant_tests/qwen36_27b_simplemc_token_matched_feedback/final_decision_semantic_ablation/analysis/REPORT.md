# Continuous W1-semantic ablation at the final decision position

W1 is the semantic answer selected in the original Baseline presentation. W2 is the semantic answer selected by a fresh Baseline solution of the remapped second presentation. Primary analyses use questions where W1 and W2 differ.

Discrete answers in this report resolve exact ties in displayed A-D order before mapping the winning letter back to semantic content. The report was regenerated after the remapped-answer tie audit; all continuous quantities are invariant to that correction.

At every post-block readout 1–64, the experiment measured the final-position projection onto the question- and layer-specific four-mapping W1 semantic vector. In the intervention pass it immediately subtracted that projection. Natural and intervention passes used the same exact historical four-question cohorts.

## Primary causal results

| Split | Natural Neutral–Game W1 gap | Ablated gap | Gap removed | Natural targeting contrast | Ablated contrast | Contrast removed |
|---|---:|---:|---:|---:|---:|---:|
| Discovery | +14.599 [+6.569, +22.628] pp | +13.139 [+6.569, +20.438] pp | +1.460 [-5.109, +8.029] pp | +0.235 [+0.073, +0.396] | +0.181 [+0.029, +0.337] | +0.054 [-0.035, +0.145] |
| Confirmation | +20.588 [+12.500, +28.676] pp | +18.382 [+10.294, +26.471] pp | +2.206 [-5.147, +9.559] pp | +0.587 [+0.397, +0.772] | +0.381 [+0.190, +0.565] | +0.206 [+0.140, +0.275] |
| Pooled | +17.582 [+11.722, +23.443] pp | +15.751 [+10.623, +21.245] pp | +1.832 [-3.297, +6.960] pp | +0.410 [+0.283, +0.534] | +0.281 [+0.156, +0.404] | +0.130 [+0.071, +0.189] |

Positive `gap removed` means the ablation eliminates part of the behavioral evidence for semantic recollection and suppression. Positive `contrast removed` means it eliminates part of the corresponding W1-versus-W2 logit effect.

## Direct effects on the discordant questions

| Split | Game W1 selection | Neutral W1 selection | Game W1–W2 margin | Neutral W1–W2 margin | Behavioral fraction explained | Logit fraction explained |
|---|---:|---:|---:|---:|---:|---:|
| Discovery | +3.650 [-1.460, +9.489] pp | +2.190 [-4.380, +8.759] pp | +0.284 [+0.162, +0.405] | +0.230 [+0.110, +0.356] | 10.0% | 22.9% |
| Confirmation | +6.618 [+0.735, +12.500] pp | +4.412 [-1.471, +10.294] pp | +0.404 [+0.284, +0.530] | +0.198 [+0.088, +0.307] | 10.7% | 35.1% |
| Pooled | +5.128 [+1.099, +9.158] pp | +3.297 [-1.099, +7.692] pp | +0.344 [+0.257, +0.432] | +0.214 [+0.132, +0.300] | 10.4% | 31.6% |

## How much activation was removed?

- **Game:** natural mean absolute W1 projection peaks at readout 64 with 22.1791 residual units, equal to 0.630949% of squared residual magnitude. During continuous ablation, regenerated activation removed at each layer peaks at readout 64 with 20.0823 units. Maximum residual projection after subtraction was 0.162565.
- **Neutral:** natural mean absolute W1 projection peaks at readout 64 with 23.2391 residual units, equal to 0.710769% of squared residual magnitude. During continuous ablation, regenerated activation removed at each layer peaks at readout 64 with 21.4005 units. Maximum residual projection after subtraction was 0.158337.

## Positive and negative activation strengths

The canonical sign-resolved figure is
[`figures/qwen36_27b_simplemc_corrected/final_decision_semantic_sign_strength.png`](../../../../../figures/qwen36_27b_simplemc_corrected/final_decision_semantic_sign_strength.png).
It uses the untouched confirmation-set executions and separates both factors that
the earlier signed/absolute figure combined: Game versus Neutral, and conflict
(`W1 != W2`) versus non-conflict (`W1 = W2`). Positive strength is
`max(projection, 0)` and negative strength is `max(-projection, 0)`, so both are
nonnegative magnitudes. Curves give every W1 letter equal weight; shading is a
95% within-letter bootstrap confidence interval.

The dominant result is similarity, not condition separation. In both conflict
and non-conflict trials, positive and negative strengths follow nearly the same
layerwise trajectory in Game and Neutral. Positive activation becomes much
larger than negative activation late in the model, but this happens in both
conditions. Conflict and non-conflict trajectories also differ much less than
the causal effects of projection removal do. Thus the behavioral interaction is
not explained by a large natural-execution difference in the gross amount of
positive or negative W1-semantic activation. It must depend on how the
representation is used, on finer structure not captured by this one-dimensional
projection, or both.

## Standard outcome checks

| Scenario | Condition | W1 selection | W2 selection | Accuracy | A–D entropy |
|---|---|---:|---:|---:|---:|
| Natural | Game | 19.1% | 45.6% | 28.9% | 1.643 bits |
| Natural | Neutral | 39.7% | 38.2% | 34.9% | 1.467 bits |
| Ablated | Game | 25.7% | 30.9% | 31.3% | 1.643 bits |
| Ablated | Neutral | 44.1% | 27.9% | 39.0% | 1.496 bits |

## Agreement-dependent reversal in the bidirectional intervention

The original intervention zeros the signed W1 projection at every readout: it removes positive W1-aligned activation and moves negative W1-aligned activation back to zero. Its effect on W1 choice reverses according to whether the semantic answer selected on the first presentation (W1) agrees with the answer selected by a fresh solution of the remapped second presentation (W2).

When **W1 differs from W2**, bidirectional projection-zeroing increases W1 choice in both conditions and both frozen splits:

| Split | Game W1 choice change | Neutral W1 choice change |
|---|---:|---:|
| Discovery (n=137) | +3.6 pp | +2.2 pp |
| Confirmation (n=136) | +6.6 pp | +4.4 pp |
| Pooled (n=273) | +5.1 pp | +3.3 pp |

When **W1 equals W2**, the same intervention decreases W1 choice in both conditions and both splits:

| Split | Game W1 choice change | Neutral W1 choice change |
|---|---:|---:|
| Discovery (n=114) | -3.5 pp | -5.3 pp |
| Confirmation (n=113) | -5.3 pp | -12.4 pp |
| Pooled (n=227) | -4.4 pp | -8.8 pp |

This replicated interaction rules out interpreting the constructed direction as context-independent positive evidence for W1. Its causal effect depends on whether W1 agrees with the answer supported by a fresh solution of the second presentation. The positive-only follow-up produces a weaker version of the W1=W2 decrease, but does not replicate the W1-choice increase on W1!=W2 trials; see the linked positive-only report.

## Historical-run validation

- Discovery: natural logits exactly matched the saved historical logits on 251/251 Game and 251/251 Neutral questions; maximum absolute difference was 0.
- Confirmation: natural logits exactly matched the saved historical logits on 249/249 Game and 249/249 Neutral questions; maximum absolute difference was 0.
