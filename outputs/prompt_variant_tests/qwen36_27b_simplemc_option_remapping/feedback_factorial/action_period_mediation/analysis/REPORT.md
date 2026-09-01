# Action-closing-period causal mediation

Positive transfer means that replacing the recipient's action-period state moves it toward the donor condition. The table uses the W1-minus-W2 margin on conflict trials; percentages are fractions of the natural Evaluation-versus-Matched-Neutral gap.

| Intervention | Direction | Conflict margin transfer | Fraction of natural gap | No-conflict W1-selection transfer |
|---|---|---:|---:|---:|
| `residual_trajectory` | Neutral → Evaluation | +0.014 [-0.013, +0.042] | +3.0 [-3.1, +8.6]% | -0.4 [-3.5, +2.6] pp |
| `residual_trajectory` | Evaluation → Neutral | +0.011 [-0.012, +0.035] | +2.4 [-2.9, +7.0]% | -0.9 [-3.1, +1.3] pp |
| `gla_state` | Neutral → Evaluation | +0.240 [+0.179, +0.303] | +51.2 [+42.5, +60.8]% | +10.1 [+4.8, +15.4] pp |
| `gla_state` | Evaluation → Neutral | +0.325 [+0.253, +0.396] | +69.2 [+59.2, +81.2]% | +6.2 [+1.3, +11.0] pp |
| `joint` | Neutral → Evaluation | +0.258 [+0.184, +0.335] | +55.0 [+45.6, +64.3]% | +8.4 [+3.1, +13.7] pp |
| `joint` | Evaluation → Neutral | +0.393 [+0.310, +0.476] | +83.7 [+75.4, +93.9]% | +10.1 [+4.8, +15.9] pp |

## Validation

- Questions: 500 (273 conflict; 227 no conflict).
- Maximum same-batch natural deviation from trusted logits: 0.000000 logits.
- Maximum recipient-state identity deviation from same-batch natural: 0.624397 logits. State effects are corrected against this identity pass.

Full condition, subset, entropy, spread, compression, targeting, and magnitude results are in `summary.json`.
