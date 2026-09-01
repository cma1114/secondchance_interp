# First-boundary accumulated GLA-state transplant

The visible recipient prompt and second presentation are unchanged. At all 48 GLA layers, the accumulated recurrent matrix state immediately after the first-answer boundary is replaced. Effects are measured relative to reinserting the recipient's own state, because splitting the kernel itself is not numerically identical to the unsplit pass.

Discrete answers in this report resolve exact ties in displayed A-D order before mapping the winning letter back to semantic content. The report was regenerated after the remapped-answer tie audit; all continuous quantities are invariant to that correction.

## Primary semantic-transfer result

Positive values mean transplantation makes the model suppress the donor's semantic winner relative to the recipient's original first answer.

| Split | Game | Neutral | Game minus Neutral |
|---|---:|---:|---:|
| Discovery | +0.024 [-0.042, +0.090] | +0.029 [-0.076, +0.131] | -0.005 [-0.083, +0.077] |
| Confirmation | +0.038 [-0.039, +0.128] | +0.017 [-0.069, +0.101] | +0.020 [-0.068, +0.109] |

## Interpretation

A selective semantic-memory transfer requires a replicated positive Game effect that exceeds Neutral and the same-winner mapping control. If this pattern is absent, the full accumulated GLA state at this boundary is not a clean, transplantable representation of the first semantic answer, even if disrupting GLA writes there affects later suppression.

## Numerical control

Discovery identity reinsertion versus unsplit maximum logit difference: 0.621. Confirmation: 0.500. Accordingly, unsplit natural logits are validation references only, not the causal counterfactual used for the reported effect.
