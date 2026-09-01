# Negative-only W1 semantic projection ablation

This experiment removes only negative projection onto the layer-specific semantic vector for W1 at the final decision position. At every readout, negative `h · v_W1` is moved to zero; positive projection is untouched. The frozen primary subset is conflict trials (`W1 != W2`). Agreement trials (`W1 = W2`) are the prespecified control.

## Bottom line

On the 273 pooled conflict trials, removing negative W1 projection increased W1 choices in **both** conditions:

- Game: 20.9% to 26.4%, a **+5.49 percentage-point** change (95% CI +2.20 to +8.79).
- Neutral: 38.5% to 42.9%, a **+4.40-point** change (95% CI +1.10 to +8.06).
- The Game-specific difference was only **+1.10 points** (95% CI -2.93 to +5.13).

This establishes that the negative coefficient has functional `not-W1` content: removing it moves behavior and logits toward W1. But it is active in both Game and Neutral and therefore does **not** explain why Game naturally avoids W1 much more than Neutral does. The intervention's Game-specific logit effect is small but reproducible: the W1-versus-W2 margin moved 0.10 logits more in Game than Neutral (95% CI +0.06 to +0.14), without a reliable condition-specific answer change.

On the 227 agreement trials, where W1 and W2 are the same semantic answer, the intervention did not raise W1 choices: Game changed by -1.32 points (95% CI -5.73 to +3.08) and Neutral by -4.41 points (95% CI -9.25 to +0.44). That sign reversal confirms that this repeated nonlinear intervention is not equivalent to simply deleting a stable memory of W1.

## Discovery

| Subset | Condition | Natural W1 | Negative-only W1 | W1 change (95% CI) | Natural W2 | Negative-only W2 | Centered-W1 logit change (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|
| Conflict (n=137) | Game | 22.6% | 29.2% | +6.57 [+2.19, +11.68] pp | 45.3% | 32.8% | +0.08 [+0.03, +0.12] |
| Conflict (n=137) | Neutral | 37.2% | 40.9% | +3.65 [-1.46, +9.49] pp | 46.0% | 35.8% | +0.04 [-0.01, +0.08] |
| Agreement (n=114) | Game | 50.9% | 50.9% | +0.00 [-6.14, +6.14] pp | 50.9% | 50.9% | -0.01 [-0.08, +0.05] |
| Agreement (n=114) | Neutral | 67.5% | 66.7% | -0.88 [-7.04, +5.26] pp | 67.5% | 66.7% | -0.08 [-0.17, -0.00] |
| All (n=251) | Game | 35.5% | 39.0% | +3.59 [+0.00, +7.57] pp | 47.8% | 41.0% | +0.04 [-0.00, +0.08] |
| All (n=251) | Neutral | 51.0% | 52.6% | +1.59 [-2.79, +5.58] pp | 55.8% | 49.8% | -0.02 [-0.07, +0.03] |

The Game-minus-Neutral entry below is not the primary outcome; it only reports whether the two conditions respond differently. The within-Game change above is the direct test of whether removing negative W1 projection reduces or increases Game's W1 choices.

| Subset | Game-minus-Neutral W1-selection change (95% CI) | Game-minus-Neutral centered-W1 change (95% CI) |
|---|---:|---:|
| Conflict (n=137) | +2.92 [-2.92, +8.76] pp | +0.04 [+0.01, +0.08] |
| Agreement (n=114) | +0.88 [-6.14, +7.89] pp | +0.07 [+0.02, +0.13] |
| All (n=251) | +1.99 [-2.79, +6.77] pp | +0.06 [+0.02, +0.09] |

## Confirmation

| Subset | Condition | Natural W1 | Negative-only W1 | W1 change (95% CI) | Natural W2 | Negative-only W2 | Centered-W1 logit change (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|
| Conflict (n=136) | Game | 19.1% | 23.5% | +4.41 [+0.00, +9.56] pp | 45.6% | 38.2% | +0.12 [+0.07, +0.18] |
| Conflict (n=136) | Neutral | 39.7% | 44.9% | +5.15 [+1.47, +9.56] pp | 38.2% | 32.4% | +0.07 [+0.01, +0.13] |
| Agreement (n=113) | Game | 53.1% | 50.4% | -2.65 [-8.85, +3.54] pp | 53.1% | 50.4% | -0.02 [-0.11, +0.07] |
| Agreement (n=113) | Neutral | 73.5% | 65.5% | -7.96 [-15.04, -0.88] pp | 73.5% | 65.5% | -0.05 [-0.16, +0.05] |
| All (n=249) | Game | 34.5% | 35.7% | +1.20 [-2.81, +5.22] pp | 49.0% | 43.8% | +0.06 [+0.01, +0.11] |
| All (n=249) | Neutral | 55.0% | 54.2% | -0.80 [-4.82, +3.21] pp | 54.2% | 47.4% | +0.01 [-0.05, +0.07] |

The Game-minus-Neutral entry below is not the primary outcome; it only reports whether the two conditions respond differently. The within-Game change above is the direct test of whether removing negative W1 projection reduces or increases Game's W1 choices.

| Subset | Game-minus-Neutral W1-selection change (95% CI) | Game-minus-Neutral centered-W1 change (95% CI) |
|---|---:|---:|
| Conflict (n=136) | -0.74 [-6.62, +5.15] pp | +0.06 [+0.02, +0.10] |
| Agreement (n=113) | +5.31 [-1.77, +12.39] pp | +0.03 [-0.03, +0.10] |
| All (n=249) | +2.01 [-2.41, +6.43] pp | +0.05 [+0.01, +0.08] |

## Interpretation rule

On conflict trials, evidence for a causal `not-W1` signal would require negative-only removal to increase W1 selection in Game reliably, with a corresponding movement of W1 logits toward W2. A similar effect in Neutral would show that the representation is not Game-specific. Null or unstable effects across the frozen split rule out this particular one-dimensional negative-projection account; they do not rule out a distributed semantic representation.

## Reproducibility

The runner preserved the historical physical batch-of-four cohorts and SDPA kernels. Natural logits, projections, and residual norms were reused from the bit-exact 500-question positive-only companion. Semantic directions were reconstructed from the same four counterbalanced option mappings and cached per cohort as float32 arrays. `summary.json` contains the complete layerwise negative-projection dose and all reported confidence intervals.
