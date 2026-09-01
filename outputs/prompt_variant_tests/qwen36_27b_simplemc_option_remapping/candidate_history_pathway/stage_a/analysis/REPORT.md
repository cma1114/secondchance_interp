# Candidate-history entry factorial

## Design

All 32 availability subsets of the five disjoint 2P option-line token classes were tested across ordinary-attention layers L4–L64. Every non-natural matching-source lesion has an equal receiver-cell lesion aimed at a balanced wrong 1P line.

## Validation

- 500 questions: 251 discovery, 249 confirmation; 273 1P/2P conflicts.
- Natural maximum A–D error: 0.00000000.
- All-open identity maximum A–D error: 0.00000000.
- Maximum intervention change: 7.468 logits.

## Confirmation complete-block effect

Values are matching-source minus balanced-wrong-source changes in candidate-centered logits.

| Rank | Game | Neutral |
|---|---:|---:|
| R1 | +0.513 [+0.335, +0.699] | -0.120 [-0.315, +0.074] |
| R2 | +0.087 [-0.065, +0.244] | +0.198 [+0.036, +0.364] |
| R3 | -0.143 [-0.271, -0.011] | +0.029 [-0.112, +0.174] |
| R4 | -0.457 [-0.549, -0.366] | -0.107 [-0.221, +0.003] |

## Primary token-class finding

The semantic wordpieces are the dominant entry route. The table compares the full matching-edge blockade with blocking only semantic-wordpiece receivers and with opening only semantic-wordpiece receivers from the all-closed state. The allow-only sign is reversed because reopening a successful route restores the natural state.

| Task | Rank | Complete block | Block semantic only | Open semantic only |
|---|---|---:|---:|---:|
| Game | R1 | +0.513 [+0.335, +0.699] | +0.536 [+0.344, +0.733] | -0.426 [-0.596, -0.247] |
| Game | R2 | +0.087 [-0.065, +0.244] | +0.075 [-0.075, +0.235] | -0.090 [-0.241, +0.056] |
| Game | R3 | -0.143 [-0.271, -0.011] | -0.136 [-0.272, -0.002] | +0.105 [-0.030, +0.236] |
| Game | R4 | -0.457 [-0.549, -0.366] | -0.474 [-0.569, -0.380] | +0.411 [+0.314, +0.507] |
| Neutral | R1 | -0.120 [-0.315, +0.074] | -0.078 [-0.286, +0.114] | +0.140 [-0.050, +0.338] |
| Neutral | R2 | +0.198 [+0.036, +0.364] | +0.200 [+0.032, +0.367] | -0.216 [-0.382, -0.052] |
| Neutral | R3 | +0.029 [-0.112, +0.174] | +0.020 [-0.122, +0.168] | -0.046 [-0.194, +0.099] |
| Neutral | R4 | -0.107 [-0.221, +0.003] | -0.143 [-0.257, -0.027] | +0.122 [+0.011, +0.232] |

On the confirmation split, blocking semantic-wordpiece reads while leaving all other routes open nearly reproduces the complete-block rank vector in both tasks. Conversely, semantic reads alone recover most of the route from the all-closed state. Leading spaces, option letters, and colons are individually small. Newlines carry a smaller secondary Game signal, especially R1/R4, but are neither the main necessary route nor sufficient for the full effect.

The discovery split independently shows the same route-level conclusion: semantic wordpieces dominate necessity, sufficiency, and Shapley allocation. The exact Neutral redistribution across R1-R4 varies between splits, so the robust claim is about the semantic receiver route, not one frozen Neutral rank profile.

## Confirmation choice effects of complete blockade

These are matching-source minus balanced-wrong-source changes in the probability of each semantic rank being the top A-D choice.

| Rank | Game | Neutral |
|---|---:|---:|
| R1 | +0.124 [+0.056, +0.193] | -0.092 [-0.161, -0.024] |
| R2 | +0.020 [-0.044, +0.084] | +0.076 [+0.012, +0.141] |
| R3 | -0.080 [-0.141, -0.024] | +0.020 [-0.032, +0.068] |
| R4 | -0.064 [-0.108, -0.024] | -0.004 [-0.044, +0.036] |

## Reading the token-class results

- Block-one necessity asks whether removing a class hurts when every other class remains open.
- Allow-only sufficiency asks whether that class alone recovers the matching-history effect from the all-closed state.
- Shapley values average each class's marginal contribution across every possible background of the other four classes, retaining redundancy and synergy in the measured 32-cell response surface.
- The figure shows confirmation results. Machine-readable discovery and confirmation results, separately for Game and Neutral and R1–R4, are in `summary.json` and `token_class_effects.csv`.
