# Fixed-A complete causal-cache transplant

The first-decision causal cache is decomposed into conventional-attention K/V, GLA causal-convolution state, and GLA delta-rule recurrent matrices. All eight factorial combinations are compared with recipient-cache continuation.

## Bottom line

The causal influence of which semantic answer was selected on the first presentation is transmitted primarily through the conventional-attention K/V cache. In both splits, transplanting K/V alone reproduces the complete-cache semantic-history transfer, while neither GLA convolutional state nor GLA recurrent matrices transfer the donor answer's semantic identity without K/V. K/V alone is sufficient, and the continuous semantic-margin transfer without K/V is negligible. This localizes the persistent-state family, but not a token position: the K/V transplant includes every prefix position through the first-decision boundary and could contain either an explicit previous-answer record or distributed first-presentation information. It also does not yet identify the later computation that makes Game discount that memory more strongly than Neutral.

## Metric definition

For an X-history recipient, the intervention inserts state from the Y-history donor and measures the change in the X-minus-Y final logit margin; the symmetric Y-history comparison is averaged with it. **Negative transfer therefore means that the final answer moved toward the donor history's previous semantic answer.** A positive Game-minus-Neutral value means donor-history dependence is weaker in Game than in Neutral. The selection metric is the analogous change in which semantic answer wins the A-D argmax.

This symmetric X↔Y crossover is not a test of whether deleting memory reduces overall switching. The complete-cache intervention exactly exchanges the two histories' continuations, so their aggregate switch rate is unchanged by construction. It replaces recipient answer X with donor answer Y rather than removing the existence of a previous answer.

## Validation

- Discovery complete-cache donor reproduction maximum error: 0 logits.
- Discovery exact-regime fixed-A sample: 56/64 questions (8 screened out before intervention).
- Discovery non-A first decisions in the analyzed sample: 0.
- Discovery cached identity versus unsplit natural answer differences: 14.
- Discovery cached versus unsplit difference in the Neutral-minus-Game prior-answer margin gap: -0.016 logits.
- Confirmation complete-cache donor reproduction maximum error: 0 logits.
- Confirmation exact-regime fixed-A sample: 63/73 questions (10 screened out before intervention).
- Confirmation non-A first decisions in the analyzed sample: 0.
- Confirmation cached identity versus unsplit natural answer differences: 10.
- Confirmation cached versus unsplit difference in the Neutral-minus-Game prior-answer margin gap: +0.005 logits.

## Factorial semantic transfer

### Discovery

| Cache families transplanted | Game | Neutral | Game − Neutral | Game selection | Neutral selection |
|---|---:|---:|---:|---:|---:|
| Identity | +0.000 [+0.000, +0.000] logits | +0.000 [+0.000, +0.000] logits | +0.000 [+0.000, +0.000] logits | +0.000 [+0.000, +0.000] pp | +0.000 [+0.000, +0.000] pp |
| Attention K/V | -0.630 [-0.949, -0.315] logits | -2.474 [-3.056, -1.911] logits | +1.844 [+1.500, +2.247] logits | -14.286 [-26.786, -1.786] pp | -65.179 [-86.607, -43.750] pp |
| GLA convolution | -0.038 [-0.064, -0.013] logits | -0.015 [-0.035, +0.006] logits | -0.023 [-0.054, +0.007] logits | -2.679 [-7.143, +0.000] pp | +0.893 [-1.786, +4.464] pp |
| K/V + convolution | -0.637 [-0.956, -0.320] logits | -2.483 [-3.066, -1.919] logits | +1.846 [+1.497, +2.254] logits | -11.607 [-24.107, +0.893] pp | -66.964 [-88.393, -45.536] pp |
| GLA recurrent matrix | +0.010 [-0.049, +0.068] logits | +0.043 [-0.024, +0.107] logits | -0.033 [-0.080, +0.015] logits | +2.679 [-2.679, +8.929] pp | -2.679 [-8.036, +2.679] pp |
| K/V + recurrent | -0.589 [-0.939, -0.238] logits | -2.426 [-3.043, -1.831] logits | +1.837 [+1.482, +2.250] logits | -6.250 [-20.536, +8.036] pp | -70.536 [-91.987, -48.214] pp |
| Convolution + recurrent | +0.003 [-0.069, +0.073] logits | +0.034 [-0.040, +0.103] logits | -0.031 [-0.077, +0.014] logits | +5.357 [+0.000, +10.714] pp | -4.464 [-9.821, +0.893] pp |
| Complete causal cache | -0.627 [-0.986, -0.270] logits | -2.440 [-3.057, -1.844] logits | +1.813 [+1.466, +2.221] logits | -8.929 [-23.214, +5.357] pp | -69.643 [-92.857, -46.429] pp |

### Confirmation

| Cache families transplanted | Game | Neutral | Game − Neutral | Game selection | Neutral selection |
|---|---:|---:|---:|---:|---:|
| Identity | +0.000 [+0.000, +0.000] logits | +0.000 [+0.000, +0.000] logits | +0.000 [+0.000, +0.000] logits | +0.000 [+0.000, +0.000] pp | +0.000 [+0.000, +0.000] pp |
| Attention K/V | -0.663 [-0.997, -0.334] logits | -2.797 [-3.360, -2.276] logits | +2.135 [+1.774, +2.549] logits | -16.667 [-34.127, -0.794] pp | -88.095 [-107.937, -68.254] pp |
| GLA convolution | +0.010 [-0.014, +0.034] logits | -0.006 [-0.027, +0.015] logits | +0.016 [-0.014, +0.046] logits | -0.794 [-5.556, +3.175] pp | +0.794 [-3.175, +5.556] pp |
| K/V + convolution | -0.641 [-0.982, -0.307] logits | -2.811 [-3.375, -2.290] logits | +2.170 [+1.811, +2.580] logits | -13.492 [-31.746, +3.968] pp | -89.683 [-110.317, -69.048] pp |
| GLA recurrent matrix | +0.027 [-0.044, +0.096] logits | +0.050 [-0.029, +0.127] logits | -0.023 [-0.071, +0.025] logits | +5.556 [-0.794, +12.698] pp | +3.968 [-2.381, +10.317] pp |
| K/V + recurrent | -0.624 [-1.002, -0.248] logits | -2.755 [-3.357, -2.196] logits | +2.131 [+1.762, +2.548] logits | -7.143 [-27.778, +12.698] pp | -86.508 [-107.143, -65.873] pp |
| Convolution + recurrent | +0.049 [-0.026, +0.122] logits | +0.036 [-0.051, +0.118] logits | +0.013 [-0.033, +0.058] logits | +8.730 [+0.794, +17.460] pp | +2.381 [-3.968, +9.524] pp |
| Complete causal cache | -0.614 [-0.995, -0.230] logits | -2.762 [-3.365, -2.203] logits | +2.148 [+1.782, +2.568] logits | -7.937 [-28.571, +11.151] pp | -85.714 [-107.937, -63.492] pp |

## Cached-regime natural dependence on the prior semantic answer

These values are measured before transplantation. Positive margins and selection rates mean the final decision favors the semantic answer selected on the first presentation over the paired alternative.

| Split | Game margin | Neutral margin | Game selection | Neutral selection |
|---|---:|---:|---:|---:|
| Discovery | +0.313 [+0.135, +0.493] logits | +1.220 [+0.922, +1.529] logits | +14.286 [+8.929, +20.536] pp | +43.750 [+33.929, +53.571] pp |
| Confirmation | +0.307 [+0.115, +0.497] logits | +1.381 [+1.102, +1.682] logits | +19.841 [+12.698, +27.778] pp | +50.794 [+41.270, +60.317] pp |

## Shapley allocation of the complete-cache effect

### Discovery

| State family | Game | Neutral | Game − Neutral |
|---|---:|---:|---:|
| attention_kv | -0.620 [-0.937, -0.306] logits | -2.472 [-3.055, -1.910] logits | +1.853 [+1.505, +2.259] logits |
| gla_conv | -0.028 [-0.049, -0.008] logits | -0.013 [-0.029, +0.004] logits | -0.015 [-0.036, +0.006] logits |
| gla_recurrent | +0.020 [-0.040, +0.079] logits | +0.044 [-0.022, +0.107] logits | -0.024 [-0.067, +0.020] logits |

### Confirmation

| State family | Game | Neutral | Game − Neutral |
|---|---:|---:|---:|
| attention_kv | -0.659 [-0.994, -0.330] logits | -2.800 [-3.363, -2.279] logits | +2.141 [+1.781, +2.553] logits |
| gla_conv | +0.014 [-0.004, +0.032] logits | -0.009 [-0.025, +0.008] logits | +0.023 [+0.001, +0.044] logits |
| gla_recurrent | +0.031 [-0.039, +0.097] logits | +0.047 [-0.034, +0.126] logits | -0.016 [-0.060, +0.027] logits |

## Interpretation discipline

The complete-cache cell is an implementation positive control and a localization upper bound, not by itself a mechanistic discovery. The family-only cells and their interactions determine whether semantic history is carried primarily by ordinary attention memory, GLA convolutional history, GLA recurrent matrices, or a distributed combination.
