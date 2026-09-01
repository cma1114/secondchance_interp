# Mixer 56 across Baseline, Game, and Neutral

Confirmation questions: **249**. Condition-specific mean outputs were estimated on the disjoint 251-question discovery set, equal-weighting the natural Baseline winner letters. All prompts use the canonical explicit empty-history ChatML format.

## Immediate JLens write

| Condition | Baseline rank 1 | Rank 2 | Rank 3 | Rank 4 |
|---|---:|---:|---:|---:|
| Baseline | +0.407 | +0.629 | -0.017 | -1.018 |
| Game | +0.523 | +0.304 | -0.117 | -0.709 |
| Neutral | +0.977 | +0.490 | -0.271 | -1.196 |

## Within-condition causal mean-ablation

| Condition | Answers changed | Spread change | Winner-advantage change | Entropy change | Switch-rate change |
|---|---:|---:|---:|---:|---:|
| Baseline | +3.614 [+1.606, +6.024] pp | -0.111 [-0.125, -0.098] | -0.134 [-0.152, -0.117] | +0.019 [+0.016, +0.023] | +3.614 [+1.606, +6.024] pp |
| Game | +4.819 [+2.410, +7.631] pp | -0.037 [-0.043, -0.030] | -0.043 [-0.053, -0.032] | +0.010 [+0.008, +0.013] | +0.402 [-2.008, +2.811] pp |
| Neutral | +1.606 [+0.402, +3.213] pp | -0.100 [-0.112, -0.087] | -0.132 [-0.149, -0.114] | +0.022 [+0.019, +0.024] | +0.000 [-1.606, +1.606] pp |

## Interpretation

Mixer 56 is not a sign-reversing component. Removing its question-specific output flattens the A–D distribution in all three conditions, so its natural causal function is sharpening in Baseline, Game, and Neutral. The sharpening is substantially weaker in Game; Neutral is approximately Baseline-like.

| Paired contrast in natural Mixer-56 sharpening | Spread | Winner advantage |
|---|---:|---:|
| Baseline minus Game | +0.075 [+0.064, +0.086] | +0.092 [+0.075, +0.109] |
| Neutral minus Game | +0.063 [+0.054, +0.072] | +0.089 [+0.074, +0.105] |
| Baseline minus Neutral | +0.012 [+0.004, +0.020] | +0.003 [-0.012, +0.017] |

Panel A is an observational JLens finite-difference attribution. Panel B is causal: it replaces the question-specific Mixer-56 output with that condition's discovery-set mean.

Figure: `figures/qwen36_27b_simplemc_corrected/mixer56_across_conditions.png`
