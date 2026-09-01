# Fixed-A first-decision semantic-state transfer

The literal first decision is `A` in both histories, but A names different semantic answers X and Y. The complete first-decision residual is exchanged between those histories while feedback and the second presentation remain fixed.

## Bottom line

The natural fixed-A semantic-history effect reproduced, but the first-decision
residual swap did **not** transfer its semantic target. Discovery found no
positive Game transfer at any of the eleven frozen readouts. The prespecified
rule selected readout 52 from a small Game-minus-Neutral contrast driven mostly
by Neutral. On held-out confirmation, the margin transfer was -0.010 logits in
Game and -0.013 in Neutral; their difference was +0.002 logits and all three
confidence intervals included zero. The discrete Game estimate was +3.4
selection points but was imprecise and also included zero.

Thus, a complete post-block residual vector at one first-decision readout is
not a portable representation of the semantic answer in this test. This is a
cleaner null than the earlier cross-order patches because literal `A` and the
entire second presentation are fixed.

The null does not exclude semantic information that has already been written
into same-layer recurrent GLA state before the post-block hook, or information
distributed across the first option positions and decision boundary. A
post-block patch can affect higher layers, but it cannot retroactively change
the current or lower layers' recurrent updates to later prompt positions.

## Result

Discovery selected post-block readout **52** by the frozen Game-minus-Neutral semantic-transfer rule.

### Confirmation

| Outcome | Estimate [95% CI] |
|---|---:|
| Game semantic-target margin transfer | -0.010 [-0.035, +0.017] logits |
| Neutral semantic-target margin transfer | -0.013 [-0.038, +0.013] logits |
| Game minus Neutral transfer | +0.002 [-0.017, +0.021] logits |
| Game answer-selection transfer | +3.425 [-2.055, +8.904] pp |
| Neutral answer-selection transfer | +0.000 [-2.055, +2.055] pp |

Positive Game transfer means exchanging the first-decision state moved later suppression toward the donor semantic answer. Negative Neutral transfer means it moved repetition toward the donor semantic answer.

## Validation

- Discovery natural semantic-targeting interaction: 0.876 [0.712, 1.060] logits.
- Maximum identity-patch A-D logit error: 0.00000000.
- Maximum identity source residual error: 0.00000000.
- Confirmation natural semantic-targeting interaction: 1.016 [0.853, 1.205] logits.
- Mean donor-versus-recipient residual distance at readout 52: 16.895 residual units.

Canonical figure: `figures/qwen36_fixed_a_first_decision_transfer.png`.
