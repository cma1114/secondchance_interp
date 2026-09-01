# Joint candidate-score and first-decision-letter lesion

## Bottom line

The redundancy hypothesis is **not supported**. Removing the candidate-score coordinate and the first-decision letter-identity subspace together did not produce a larger or more reliable recovery of W1 than either lesion alone.

On held-out conflict trials, decision-letter removal produced the predicted discrete interaction (+2.941 [+0.735, +5.882] percentage points), driven by Game choosing W1 more often. But the discovery interaction was +0.730 [-3.650, +5.109], and the held-out continuous W1-minus-W2 interaction was only +0.006 [-0.022, +0.035] logits. Thus the attractive held-out choice effect does not replicate across the frozen splits or in the continuous margin.

For scale, natural held-out W1 choice was 17.6% in Game and 40.4% in Neutral, a -22.8-point gap. The letter-only intervention closes only +2.941 [+0.735, +5.882] points of that gap (about 17% at the point estimate).

The joint lesion was weaker, not stronger: its held-out conflict-trial Game-minus-Neutral W1-choice effect was +2.941 [-0.735, +7.353] points, and Game itself changed by +0.735 [-1.471, +3.676] points. The Game factorial interaction was antagonistic (-2.941 [-7.353, +0.735] points), rather than the positive synergy expected if the two coordinates were redundant routes whose joint removal exposed the mechanism.

The experiment therefore leaves the core binding problem unresolved. A one-dimensional option-value coordinate is readable at the option newline, and centered A-D identity is present at the first-decision position, but jointly removing those two decoded coordinates does not causally account for preferential Game revision.

Effects are measured against the exact zero-delta identity-K/V path. The primary endpoint is held-out conflict-trial recovery of W1 in Game relative to Neutral.

## Validation

- Natural-versus-identity maximum A–D logit difference: 0.
- Natural-versus-identity choice changes: 0.
- Natural trusted-choice agreement: 98.80%.
- First-decision Baseline-choice agreement: 98.40%.
- Maximum residual candidate score after score removal: 5.67734e-06.
- Maximum residual A–D norm after decision-letter removal: 5.95686e-05.
- Same-host prior natural maximum A–D logit difference: 0; choice changes: 0.
- Same-host prior score-only maximum A–D logit difference: 0; choice changes: 0.

The 98.8% trusted-choice figure compares against an older run from another host and reflects known BF16 host drift. The matched same-host natural and score-only controls reproduced exactly and are the relevant numerical validation for the causal contrasts.

Excluding the eight questions whose current first decision differed from the older cross-host Baseline does not change the conclusion. Among 133 held-out matched conflict questions, the letter-only W1-choice interaction is +3.008 [+0.752, +6.015] points, versus +0.746 [-3.731, +5.224] in discovery.

## Held-out confirmation

Conflict questions: **136**; no-conflict questions: **113**.

| Intervention | Game W1 choice | Neutral W1 choice | Game−Neutral W1 choice | Game−Neutral W1−W2 margin |
|---|---:|---:|---:|---:|
| Score only | +1.471 [-2.206, +5.147] | -0.735 [-2.941, +1.471] | +2.206 [-2.206, +6.618] | +0.023 [-0.004, +0.049] |
| Decision letter only | +2.206 [+0.000, +5.147] | -0.735 [-2.206, +0.000] | +2.941 [+0.735, +5.882] | +0.006 [-0.022, +0.035] |
| Both | +0.735 [-1.471, +3.676] | -2.206 [-5.882, +0.735] | +2.941 [-0.735, +7.353] | +0.017 [-0.012, +0.047] |

The complete machine-readable summary contains all/conflict/no-conflict results for switching, W1 and W2 choice, probabilities, margins, entropy, spread, and the factorial interaction on both frozen splits.

Canonical figure: `figures/qwen36_joint_option_score_decision_letter.png`.
