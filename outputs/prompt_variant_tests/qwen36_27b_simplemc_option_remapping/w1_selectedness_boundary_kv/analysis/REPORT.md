# First-decision selectedness K/V transplant

## Bottom line

This experiment asks whether ordinary-attention K/V at the final token of the empty first-answer scaffold carries the missing ‘A was selected’ signal. Semantic A and its displayed position are identical in donor and recipient; only the ordering of B-D changes whether A wins.

Exact-regime eligibility retained 38/41 discovery and 32/36 confirmation questions.

The strict opposite-direction selectedness signature did not pass on both frozen splits.

## Natural donor-minus-recipient effect on semantic A

Positive Neutral-minus-Game values mean that making A the first-pass winner favors its semantic content more under `lost` than under `incorrect`.

| Split | Game centered A | Neutral centered A | Neutral minus Game |
|---|---:|---:|---:|
| Discovery | -0.079 [-0.369, +0.188] | -0.028 [-0.364, +0.287] | +0.052 [-0.084, +0.176] |
| Confirmation | +0.114 [-0.124, +0.333] | +0.264 [-0.037, +0.543] | +0.150 [+0.013, +0.283] |

## Causal effect of importing donor boundary K/V into recipient

| Split | Game centered A | Neutral centered A | Neutral minus Game | Interaction fraction |
|---|---:|---:|---:|---:|
| Discovery | -0.033 [-0.080, +0.018] | -0.043 [-0.091, +0.007] | -0.010 [-0.039, +0.017] | -20.2% |
| Confirmation | +0.015 [-0.039, +0.068] | +0.019 [-0.023, +0.062] | +0.005 [-0.029, +0.038] | 3.1% |

### Target-versus-recipient-winner margin

| Split | Game | Neutral | Neutral minus Game |
|---|---:|---:|---:|
| Discovery | +0.030 [-0.022, +0.084] | -0.030 [-0.090, +0.022] | -0.060 [-0.111, -0.015] |
| Confirmation | +0.073 [+0.016, +0.133] | +0.085 [+0.033, +0.135] | +0.012 [-0.046, +0.069] |

### Semantic-A choice rate

| Split | Game | Neutral | Neutral minus Game |
|---|---:|---:|---:|
| Discovery | -2.632 [-7.895, +0.000] pp | +0.000 [-7.895, +7.895] pp | +2.632 [-5.263, +13.158] pp |
| Confirmation | +0.000 [+0.000, +0.000] pp | +0.000 [+0.000, +0.000] pp | +0.000 [+0.000, +0.000] pp |

## Validation

- Discovery complete-cache donor maximum A-D error: 0.
- Discovery untouched donor-row maximum A-D error: 0.
- Discovery mean relative donor-recipient boundary K/V difference: 0.0797.
- Confirmation complete-cache donor maximum A-D error: 0.
- Confirmation untouched donor-row maximum A-D error: 0.
- Confirmation mean relative donor-recipient boundary K/V difference: 0.0729.

Canonical figure: `figures/qwen36_w1_selectedness_boundary_kv.png`.
