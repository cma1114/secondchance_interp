# Fixed-A final-query selected-option attention-edge ablation

## Bottom line

This report tests whether the final answer decision must directly read the semantic option line selected on the first presentation. Only final-query ordinary-attention edges are removed; every earlier query remains intact.

The predicted result did not occur. Across the prespecified block sets, removing the selected-line edges did not reliably increase Game's W1 choices, reduce its preferential avoidance of W1, or change the W1-versus-counterfactual margin. Selected-line effects were also not reliably different from matched unselected-line controls. Thus, although the selected option's K/V history is causally important in the earlier cache-transplant experiment, a clean direct read from the final query is not the route by which that information affects preferential semantic switching.

## Discovery (n=57 exact-regime questions)

| Intervention | Δ Game W1 choice | Δ Neutral W1 choice | Δ preferential W1 avoidance | Δ Game W1 margin |
|---|---:|---:|---:|---:|
| `block_44_selected` | +0.00 [+0.00, +0.00] pp | -1.75 [-4.39, +0.00] pp | -1.75 [-4.39, +0.00] pp | +0.01 [-0.01, +0.02] |
| `block_44_matched_control` | +0.88 [+0.00, +2.63] pp | +0.00 [+0.00, +0.00] pp | -0.88 [-2.63, +0.00] pp | +0.01 [-0.01, +0.03] |
| `band_36_48_selected` | +0.00 [+0.00, +0.00] pp | -0.88 [-2.63, +0.00] pp | -0.88 [-2.63, +0.00] pp | +0.00 [-0.01, +0.01] |
| `band_36_48_matched_control` | +0.00 [+0.00, +0.00] pp | -1.75 [-4.39, +0.00] pp | -1.75 [-4.39, +0.00] pp | +0.01 [-0.00, +0.03] |
| `all_04_48_selected` | +0.00 [+0.00, +0.00] pp | +0.00 [+0.00, +0.00] pp | +0.00 [+0.00, +0.00] pp | +0.01 [-0.01, +0.02] |
| `all_04_48_matched_control` | +0.00 [+0.00, +0.00] pp | +0.88 [+0.00, +2.63] pp | +0.88 [+0.00, +2.63] pp | +0.01 [-0.01, +0.02] |

Selected-line effect minus matched-control effect:

| Block set | Δ Game W1 choice | Δ preferential W1 avoidance | Δ Game W1 margin |
|---|---:|---:|---:|
| `block_44` | -0.88 [-2.63, +0.00] pp | -0.88 [-3.51, +1.75] pp | -0.00 [-0.02, +0.01] |
| `band_36_48` | +0.00 [+0.00, +0.00] pp | +0.88 [+0.00, +2.63] pp | -0.01 [-0.03, +0.00] |
| `all_04_48` | +0.00 [+0.00, +0.00] pp | -0.88 [-2.63, +0.00] pp | +0.00 [-0.01, +0.02] |

## Confirmation (n=64 exact-regime questions)

| Intervention | Δ Game W1 choice | Δ Neutral W1 choice | Δ preferential W1 avoidance | Δ Game W1 margin |
|---|---:|---:|---:|---:|
| `block_44_selected` | -2.34 [-6.25, +0.78] pp | +1.56 [+0.00, +3.91] pp | +3.91 [+0.78, +7.81] pp | +0.01 [-0.01, +0.02] |
| `block_44_matched_control` | -1.56 [-3.91, +0.00] pp | +0.78 [+0.00, +2.34] pp | +2.34 [+0.00, +5.47] pp | +0.01 [-0.01, +0.02] |
| `band_36_48_selected` | -0.78 [-4.69, +2.34] pp | +0.78 [+0.00, +2.34] pp | +1.56 [-2.34, +5.47] pp | +0.01 [-0.00, +0.02] |
| `band_36_48_matched_control` | +0.00 [-3.91, +3.91] pp | +0.78 [+0.00, +2.34] pp | +0.78 [-3.12, +4.69] pp | +0.00 [-0.01, +0.02] |
| `all_04_48_selected` | +0.00 [-3.12, +3.12] pp | +0.78 [+0.00, +2.34] pp | +0.78 [-2.34, +3.91] pp | -0.00 [-0.01, +0.01] |
| `all_04_48_matched_control` | -1.56 [-4.69, +1.56] pp | +0.00 [+0.00, +0.00] pp | +1.56 [-1.56, +4.69] pp | -0.00 [-0.01, +0.01] |

Selected-line effect minus matched-control effect:

| Block set | Δ Game W1 choice | Δ preferential W1 avoidance | Δ Game W1 margin |
|---|---:|---:|---:|
| `block_44` | -0.78 [-3.91, +1.56] pp | +1.56 [-1.56, +4.69] pp | -0.00 [-0.01, +0.01] |
| `band_36_48` | -0.78 [-3.91, +1.56] pp | +0.78 [-3.12, +5.47] pp | +0.01 [-0.01, +0.02] |
| `all_04_48` | +1.56 [-1.56, +4.69] pp | -0.78 [-3.91, +2.34] pp | +0.00 [-0.01, +0.01] |

## Definitions

- **W1** is the semantic content chosen as literal `A` on the first presentation. X and Y histories have the same second presentation but different W1 content.
- **Preferential W1 avoidance** is Neutral's W1-choice rate minus Game's W1-choice rate. A negative intervention change means the lesion erased part of Game's preferential avoidance.
- **Game W1 margin** is W1's A-D logit minus the counterfactual first-answer content's logit, averaged symmetrically over X and Y histories.
- **Matched control** is the unselected first-presentation option line with the nearest token count to the selected A line.

Canonical figure: `figures/qwen36_fixed_a_final_query_edge_ablation.png`.
