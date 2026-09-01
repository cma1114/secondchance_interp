# Canonical remapped final-query W1-line attention-edge ablation

## Question

Does preferential semantic switching require the final pre-answer query to read the complete first-presentation option line containing W1 through ordinary attention? Earlier queries and all other source tokens remain untouched.

Every reported effect is **intervened minus natural within the named condition**. Conflict means W1 differs from the answer a fresh Baseline would choose under the remapped second presentation (W2).

## Bottom line

No. The natural behavioral phenomenon is large on conflict trials: Game chooses W1 on 20.1% of trials versus 38.5% in Neutral, a 18.3-percentage-point Game-specific avoidance difference. But preventing the final decision query from directly reading W1's original option line does not undo it.

Across conflict trials, blocking the selected line at block 44, blocks 36/40/44/48, or every ordinary-attention block from 4 through 48 changes Game W1 choice by -0.37, +0.00, -0.37 percentage points, respectively. The selected-line-minus-matched-control effects are +0.73, +0.37, -0.73 points. The complete confidence intervals are reported below; the frozen discovery and confirmation halves and the no-conflict trials determine whether the substantive null replicates.

The all-block conflict intervention changes Game's W1-minus-W2 logit margin by -0.01 [-0.02, -0.00] overall, -0.02 [-0.03, -0.00] in discovery, and -0.01 [-0.02, +0.00] in confirmation. Its interpretation is based on these generated values rather than hard-coded prose.

Therefore, the earlier finding that first-presentation option-line K/V is causally important must not be interpreted as a direct final-token lookup. Its information must be read by earlier downstream queries and propagated through intermediate residual/recurrent states before the final decision. This experiment rules out the clean direct-edge mechanism; it does not localize the intervening relay.

A predecessor full pass was invalid because a Boolean SDPA mask treated the attempted block as allowed. The corrected runner writes `False` for Boolean masks, passes Boolean/additive/implicit-causal regression tests, produces nonzero logit changes, and reproduces all trusted natural logits exactly. Only the corrected results below are scientific results.

## conflict W1 not equal W2 (n=273)

| Intervention | Condition | Δ W1 choice | Δ switch away from W1 | Δ W1 centered advantage | Δ entropy |
|---|---|---:|---:|---:|---:|
| `block_44_selected` | Game | -0.37 [-1.10, +0.00] pp | +0.37 [+0.00, +1.10] pp | -0.00 [-0.01, +0.00] | +0.00 [-0.00, +0.00] |
| `block_44_selected` | Neutral | +0.37 [-0.73, +1.83] pp | -0.37 [-1.83, +0.73] pp | -0.01 [-0.01, -0.00] | -0.00 [-0.00, +0.00] |
| `block_44_matched_control` | Game | -1.10 [-2.56, +0.00] pp | +1.10 [+0.00, +2.56] pp | +0.00 [-0.01, +0.01] | +0.00 [-0.00, +0.00] |
| `block_44_matched_control` | Neutral | +0.00 [-1.47, +1.47] pp | +0.00 [-1.47, +1.47] pp | -0.00 [-0.01, +0.00] | +0.00 [-0.00, +0.00] |
| `band_36_48_selected` | Game | +0.00 [-1.10, +1.10] pp | +0.00 [-1.10, +1.10] pp | -0.00 [-0.01, +0.01] | +0.00 [-0.00, +0.00] |
| `band_36_48_selected` | Neutral | +0.37 [-1.10, +1.83] pp | -0.37 [-2.20, +1.10] pp | -0.01 [-0.01, +0.00] | +0.00 [-0.00, +0.00] |
| `band_36_48_matched_control` | Game | -0.37 [-1.83, +0.73] pp | +0.37 [-0.73, +1.83] pp | -0.01 [-0.01, +0.00] | -0.00 [-0.00, +0.00] |
| `band_36_48_matched_control` | Neutral | -0.37 [-1.83, +0.73] pp | +0.37 [-0.73, +1.47] pp | +0.00 [-0.01, +0.01] | +0.00 [-0.00, +0.00] |
| `all_04_48_selected` | Game | -0.37 [-1.47, +0.73] pp | +0.37 [-0.73, +1.47] pp | -0.01 [-0.01, +0.00] | -0.00 [-0.00, +0.00] |
| `all_04_48_selected` | Neutral | -0.37 [-1.83, +1.10] pp | +0.37 [-1.10, +1.83] pp | -0.01 [-0.01, +0.00] | +0.00 [-0.00, +0.00] |
| `all_04_48_matched_control` | Game | +0.37 [-0.73, +1.47] pp | -0.37 [-1.83, +0.73] pp | -0.01 [-0.01, +0.00] | +0.00 [-0.00, +0.00] |
| `all_04_48_matched_control` | Neutral | +0.00 [-1.47, +1.47] pp | +0.00 [-1.47, +1.47] pp | +0.00 [-0.01, +0.01] | +0.00 [-0.00, +0.00] |

Selected W1-line effect minus matched unselected-line effect:

| Block set | Condition | Δ W1 choice | Δ switch away from W1 | Δ W1 centered advantage |
|---|---|---:|---:|---:|
| `block_44` | Game | +0.73 [-0.73, +2.20] pp | -0.73 [-2.20, +0.73] pp | -0.00 [-0.01, +0.00] |
| `block_44` | Neutral | +0.37 [-1.10, +1.83] pp | -0.37 [-1.83, +1.10] pp | -0.00 [-0.01, +0.00] |
| `band_36_48` | Game | +0.37 [-0.73, +1.47] pp | -0.37 [-1.47, +0.73] pp | +0.01 [-0.00, +0.01] |
| `band_36_48` | Neutral | +0.73 [-1.10, +2.93] pp | -0.73 [-2.93, +1.10] pp | -0.01 [-0.01, -0.00] |
| `all_04_48` | Game | -0.73 [-2.20, +0.73] pp | +0.73 [-0.73, +2.20] pp | +0.00 [-0.01, +0.01] |
| `all_04_48` | Neutral | -0.37 [-2.20, +1.47] pp | +0.37 [-1.47, +2.20] pp | -0.01 [-0.01, -0.00] |

## no conflict W1 equal W2 (n=227)

| Intervention | Condition | Δ W1 choice | Δ switch away from W1 | Δ W1 centered advantage | Δ entropy |
|---|---|---:|---:|---:|---:|
| `block_44_selected` | Game | -1.32 [-3.52, +0.44] pp | +1.32 [-0.44, +3.52] pp | -0.00 [-0.01, +0.01] | +0.00 [-0.00, +0.00] |
| `block_44_selected` | Neutral | +0.88 [+0.00, +2.20] pp | -0.88 [-2.20, +0.00] pp | +0.01 [-0.00, +0.02] | -0.00 [-0.00, +0.00] |
| `block_44_matched_control` | Game | -0.44 [-1.32, +0.00] pp | +0.44 [+0.00, +1.32] pp | -0.01 [-0.01, +0.00] | +0.00 [-0.00, +0.00] |
| `block_44_matched_control` | Neutral | +0.88 [+0.00, +2.20] pp | -0.88 [-2.20, +0.00] pp | +0.01 [-0.00, +0.01] | -0.00 [-0.00, +0.00] |
| `band_36_48_selected` | Game | -0.88 [-2.64, +0.88] pp | +0.88 [-0.88, +2.64] pp | -0.01 [-0.01, +0.00] | +0.00 [-0.00, +0.00] |
| `band_36_48_selected` | Neutral | +0.88 [-0.88, +2.64] pp | -0.88 [-2.64, +0.88] pp | -0.00 [-0.01, +0.00] | +0.00 [-0.00, +0.00] |
| `band_36_48_matched_control` | Game | -1.32 [-3.52, +0.44] pp | +1.32 [-0.44, +3.52] pp | +0.00 [-0.01, +0.01] | +0.00 [-0.00, +0.00] |
| `band_36_48_matched_control` | Neutral | +1.32 [+0.00, +3.08] pp | -1.32 [-3.08, +0.00] pp | +0.00 [-0.00, +0.01] | +0.00 [-0.00, +0.00] |
| `all_04_48_selected` | Game | +0.00 [-1.76, +1.76] pp | +0.00 [-1.76, +1.76] pp | -0.00 [-0.01, +0.00] | -0.00 [-0.00, +0.00] |
| `all_04_48_selected` | Neutral | +0.44 [+0.00, +1.32] pp | -0.44 [-1.32, +0.00] pp | +0.00 [-0.01, +0.01] | +0.00 [-0.00, +0.00] |
| `all_04_48_matched_control` | Game | -1.32 [-3.08, +0.00] pp | +1.32 [+0.00, +3.08] pp | +0.00 [-0.01, +0.01] | +0.00 [-0.00, +0.00] |
| `all_04_48_matched_control` | Neutral | +1.32 [+0.00, +3.08] pp | -1.32 [-3.08, +0.00] pp | +0.00 [-0.00, +0.01] | +0.00 [-0.00, +0.00] |

Selected W1-line effect minus matched unselected-line effect:

| Block set | Condition | Δ W1 choice | Δ switch away from W1 | Δ W1 centered advantage |
|---|---|---:|---:|---:|
| `block_44` | Game | -0.88 [-2.64, +0.88] pp | +0.88 [-0.88, +2.64] pp | +0.00 [-0.00, +0.01] |
| `block_44` | Neutral | +0.00 [+0.00, +0.00] pp | +0.00 [+0.00, +0.00] pp | +0.00 [-0.01, +0.01] |
| `band_36_48` | Game | +0.44 [-2.20, +3.08] pp | -0.44 [-3.08, +2.20] pp | -0.01 [-0.02, +0.00] |
| `band_36_48` | Neutral | -0.44 [-1.76, +0.88] pp | +0.44 [-0.88, +2.20] pp | -0.01 [-0.02, +0.00] |
| `all_04_48` | Game | +1.32 [-0.44, +3.52] pp | -1.32 [-3.52, +0.44] pp | -0.00 [-0.01, +0.00] |
| `all_04_48` | Neutral | -0.88 [-2.20, +0.00] pp | +0.88 [+0.00, +2.20] pp | -0.00 [-0.01, +0.01] |

## Validation

Natural logits reproduced the trusted canonical remapped run with maximum absolute error `0.0`.

Canonical figure: `figures/qwen36_remapped_final_query_edge_ablation.png`.
