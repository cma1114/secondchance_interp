# First-answer-boundary GLA semantic-memory rewrite

At the exact first-answer-boundary span and all 48 GLA layers, replace key, value, decay gate g, and beta with same-question tensors from an alternative first presentation that produced a different semantic answer.

Discrete answers in this report resolve exact ties in displayed A-D order before mapping the winning letter back to semantic content. The report was regenerated after the remapped-answer tie audit; all continuous quantities are invariant to that correction.

Positive W1-minus-donor margin change means the transplanted donor answer became relatively less favored. The key prediction is positive in Game, negative in Neutral, and therefore a positive Game-minus-Neutral policy divergence.

## Primary causal result

| Split | Primary pairs | Game W1−donor change | Neutral W1−donor change | Game−Neutral policy divergence | Behavioral divergence |
|---|---:|---:|---:|---:|---:|
| Discovery | 96 | -0.008 [-0.050, +0.033] | +0.019 [-0.023, +0.061] | -0.027 [-0.066, +0.012] | -3.1 [-14.6, +8.3] pp |
| Confirmation | 104 | +0.005 [-0.036, +0.046] | +0.020 [-0.016, +0.055] | -0.014 [-0.055, +0.027] | +5.8 [+0.0, +12.5] pp |
| Pooled | — | -0.001 [-0.030, +0.028] | +0.019 [-0.008, +0.046] | -0.020 [-0.048, +0.008] | +1.5 [-5.0, +8.0] pp |

## Content and letter specificity

| Split | Condition | Semantic margin change | Literal-letter-control change | Semantic minus literal | Donor selection change | W1 selection change |
|---|---|---:|---:|---:|---:|---:|
| Discovery | Game | -0.008 [-0.050, +0.033] | -0.043 [-0.081, -0.004] | +0.034 [-0.002, +0.073] | +0.0 [-5.2, +5.2] pp | -1.0 [-6.2, +4.2] pp |
| Discovery | Neutral | +0.019 [-0.023, +0.061] | -0.054 [-0.100, -0.012] | +0.073 [+0.029, +0.117] | -2.1 [-5.2, +0.0] pp | +0.0 [-4.2, +4.2] pp |
| Confirmation | Game | +0.005 [-0.036, +0.046] | -0.044 [-0.082, -0.009] | +0.050 [+0.008, +0.092] | -4.8 [-9.6, -1.0] pp | +1.9 [+0.0, +4.8] pp |
| Confirmation | Neutral | +0.020 [-0.016, +0.055] | -0.031 [-0.078, +0.009] | +0.051 [+0.014, +0.095] | +0.0 [-2.9, +2.9] pp | +1.0 [-1.9, +4.8] pp |

## Standard behavioral checks on confirmation

| Scenario | Game accuracy | Game change | Neutral change | Normalized lift | Game AccIncor | Game second choice | Game entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Natural | 28.9% | 65.5% | 45.0% | 0.372 | 35/103 = 34.0% | 65/163 = 39.9% | 1.643 bits |
| Donor memory patch | 29.3% | 63.9% | 44.6% | 0.348 | 32/103 = 31.1% | 62/159 = 39.0% | 1.641 bits |
| Identity patch | 28.9% | 65.5% | 45.0% | 0.372 | 35/103 = 34.0% | 65/163 = 39.9% | 1.643 bits |

## Numerical validation

- Discovery: natural logits exactly match history on 251/251 Game and 251/251 Neutral rows; identity patches exactly match on 502/502 condition-rows.
- Confirmation: natural logits exactly match history on 249/249 Game and 249/249 Neutral rows; identity patches exactly match on 498/498 condition-rows.
