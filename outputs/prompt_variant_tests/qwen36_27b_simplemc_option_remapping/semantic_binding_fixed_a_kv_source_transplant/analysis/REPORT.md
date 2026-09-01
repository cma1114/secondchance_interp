# Fixed-A conventional-attention K/V source localization

## Bottom line

The semantic-history carrier is sharply localized to the **first-presentation
option line whose literal label is `A`**—the option the model selected. On the
held-out confirmation questions, transplanting only that line's K/V moved the
final donor-versus-recipient semantic margin by -0.558 logits in Game and
-3.161 logits in Matched Neutral. Their +2.603-logit difference replicated the
discovery result (+2.187) and was larger than the +2.135-logit difference from
transplanting every conventional-attention K/V entry. The selected-line patch
increased selection of the donor semantic answer by 15.1 points in Game versus
43.7 points in Neutral.

The rest of the first question weakly opposed this differential effect. The
entire first question nevertheless reproduced essentially all of the all-K/V
contrast (+2.124 versus +2.135 logits). In contrast, the first-decision boundary
and the post-question cue/assistant-header span were both near zero. Adding the
boundary to the selected option did not change the selected-option result.

This supplies the missing causal link: the semantic content corresponding to
the first answer remains available in conventional-attention K/V at its option
line, and the same transplanted semantic memory affects later choice far more
under `lost` than under `incorrect`. It is not a generic entropy artifact: the
selected-line patch reduced A-D entropy similarly in Game (-0.110 bits) and
Neutral (-0.129 bits), while its semantic-margin and donor-selection effects
differed greatly. The sign is also informative: Game still moves somewhat
toward the transplanted semantic answer, so this supports **strongly attenuated
reinstatement in Game**, not a simple sign-reversed inhibitory read.

Because every analyzed first decision was literal `A`, this establishes
content-specific memory and condition-dependent use within the fixed-A cohort;
it does not yet establish letter-general operation for B-D. The justified next
step is to localize which of the 16 conventional-attention layers read the
selected-option K/V.

## Metric

The visible recipient prompt is unchanged. For each fixed-A X/Y pair, the intervention replaces conventional-attention K/V entries from a specified first-presentation token region with entries from the opposite semantic history. Negative margin transfer means the final answer moves toward the donor history's previous semantic answer. Positive Game-minus-Neutral means that donor-answer reinstatement is weaker in Game than in Neutral.

This candidate-specific crossover is the primary endpoint. Aggregate switching is not a valid primary endpoint because the symmetric X↔Y complete-cache positive control permutes histories by construction.

## Validation

- Discovery exact-regime sample: 56/64 (8 excluded before feedback).
- Complete causal-cache donor reproduction maximum A-D error: 0 logits.
- Informative-prefix versus all-attention-K/V maximum A-D error: 0 logits.
- Cached identity versus unsplit natural answer differences: 14.
- Natural semantic targeting: +0.923 [+0.747, +1.127] logits.
- Cached-identity semantic targeting: +0.907 [+0.737, +1.108] logits.
- Confirmation exact-regime sample: 63/73 (10 excluded before feedback).
- Complete causal-cache donor reproduction maximum A-D error: 0 logits.
- Informative-prefix versus all-attention-K/V maximum A-D error: 0 logits.
- Cached identity versus unsplit natural answer differences: 10.
- Natural semantic targeting: +1.068 [+0.883, +1.274] logits.
- Cached-identity semantic targeting: +1.074 [+0.888, +1.282] logits.

## Discovery source-region transfer

| K/V source transplanted | Positions | Game margin | Neutral margin | Game − Neutral | Game donor chosen | Neutral donor chosen | Game entropy | Neutral entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Identity | 0.0 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] pp | +0.000 [+0.000, +0.000] pp | +0.000 [+0.000, +0.000] bits | +0.000 [+0.000, +0.000] bits |
| Selected A option line | 10.1 | -0.665 [-0.981, -0.357] | -2.852 [-3.223, -2.490] | +2.187 [+1.943, +2.446] | +12.500 [+5.357, +19.643] pp | +42.857 [+33.036, +52.679] pp | -0.094 [-0.139, -0.047] bits | -0.096 [-0.152, -0.036] bits |
| Other first-question tokens | 62.1 | +0.064 [-0.202, +0.322] | +0.424 [+0.037, +0.754] | -0.361 [-0.574, -0.115] | +5.357 [-2.679, +12.500] pp | -4.464 [-10.714, +2.679] pp | -0.094 [-0.145, -0.043] bits | -0.099 [-0.154, -0.039] bits |
| Entire first question | 72.2 | -0.561 [-0.835, -0.282] | -2.371 [-2.901, -1.873] | +1.810 [+1.488, +2.188] | +4.464 [-1.786, +10.714] pp | +30.357 [+19.643, +41.071] pp | +0.010 [-0.007, +0.027] bits | +0.013 [-0.008, +0.035] bits |
| First-decision boundary | 1.0 | -0.037 [-0.080, +0.006] | -0.047 [-0.087, -0.004] | +0.010 [-0.018, +0.036] | -0.893 [-3.571, +1.786] pp | +0.000 [-2.679, +2.679] pp | +0.008 [+0.001, +0.016] bits | +0.011 [+0.005, +0.017] bits |
| Post-question cue/header | 22.0 | -0.057 [-0.137, +0.026] | -0.068 [-0.154, +0.021] | +0.010 [-0.031, +0.052] | -0.893 [-5.357, +2.679] pp | +1.786 [+0.000, +4.464] pp | +0.012 [+0.001, +0.023] bits | +0.014 [+0.001, +0.028] bits |
| Selected option + boundary | 11.1 | -0.671 [-0.988, -0.359] | -2.844 [-3.209, -2.483] | +2.173 [+1.932, +2.425] | +12.500 [+4.464, +20.536] pp | +40.179 [+30.357, +49.107] pp | -0.087 [-0.132, -0.041] bits | -0.090 [-0.144, -0.032] bits |
| Entire informative prefix | 95.2 | -0.630 [-0.942, -0.305] | -2.474 [-3.060, -1.912] | +1.844 [+1.505, +2.243] | +4.464 [-2.679, +11.607] pp | +29.464 [+17.857, +40.179] pp | +0.008 [-0.004, +0.020] bits | +0.004 [-0.008, +0.015] bits |
| All conventional-attention K/V | 163.2 | -0.630 [-0.942, -0.305] | -2.474 [-3.060, -1.912] | +1.844 [+1.505, +2.243] | +4.464 [-2.679, +11.607] pp | +29.464 [+17.857, +40.179] pp | +0.008 [-0.004, +0.020] bits | +0.004 [-0.008, +0.015] bits |
| Complete causal cache | 163.2 | -0.627 [-0.976, -0.267] | -2.440 [-3.061, -1.843] | +1.813 [+1.474, +2.217] | +4.464 [-2.679, +11.607] pp | +34.821 [+23.214, +46.429] pp | +0.000 [+0.000, +0.000] bits | +0.000 [+0.000, +0.000] bits |

## Confirmation source-region transfer

| K/V source transplanted | Positions | Game margin | Neutral margin | Game − Neutral | Game donor chosen | Neutral donor chosen | Game entropy | Neutral entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Identity | 0.0 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] pp | +0.000 [+0.000, +0.000] pp | +0.000 [+0.000, +0.000] bits | +0.000 [+0.000, +0.000] bits |
| Selected A option line | 10.2 | -0.558 [-0.809, -0.320] | -3.161 [-3.632, -2.747] | +2.603 [+2.207, +3.056] | +15.079 [+6.349, +23.810] pp | +43.651 [+34.127, +53.175] pp | -0.110 [-0.162, -0.056] bits | -0.129 [-0.184, -0.075] bits |
| Other first-question tokens | 63.4 | -0.066 [-0.389, +0.239] | +0.452 [+0.176, +0.728] | -0.519 [-0.705, -0.334] | +1.587 [-7.937, +11.111] pp | -5.556 [-11.111, +0.000] pp | -0.115 [-0.166, -0.064] bits | -0.145 [-0.194, -0.095] bits |
| Entire first question | 73.7 | -0.614 [-0.874, -0.358] | -2.738 [-3.263, -2.253] | +2.124 [+1.762, +2.528] | +1.587 [-7.143, +10.317] pp | +42.857 [+33.333, +52.381] pp | +0.005 [-0.014, +0.026] bits | -0.001 [-0.030, +0.028] bits |
| First-decision boundary | 1.0 | -0.012 [-0.060, +0.040] | -0.015 [-0.054, +0.025] | +0.004 [-0.028, +0.035] | -4.762 [-9.524, -0.794] pp | -0.794 [-2.381, +0.000] pp | +0.006 [-0.000, +0.011] bits | +0.000 [-0.006, +0.007] bits |
| Post-question cue/header | 22.0 | -0.029 [-0.131, +0.076] | -0.018 [-0.106, +0.073] | -0.011 [-0.056, +0.033] | -2.381 [-7.143, +2.381] pp | -0.794 [-3.968, +1.587] pp | +0.006 [-0.006, +0.017] bits | -0.001 [-0.020, +0.015] bits |
| Selected option + boundary | 11.2 | -0.566 [-0.823, -0.324] | -3.172 [-3.642, -2.749] | +2.606 [+2.209, +3.062] | +18.254 [+10.317, +26.984] pp | +41.270 [+31.746, +50.794] pp | -0.114 [-0.168, -0.059] bits | -0.133 [-0.186, -0.081] bits |
| Entire informative prefix | 96.7 | -0.663 [-1.002, -0.329] | -2.797 [-3.361, -2.271] | +2.135 [+1.774, +2.545] | +4.762 [-3.968, +14.286] pp | +43.651 [+33.333, +53.968] pp | +0.007 [-0.005, +0.021] bits | -0.002 [-0.019, +0.015] bits |
| All conventional-attention K/V | 164.7 | -0.663 [-1.002, -0.329] | -2.797 [-3.361, -2.271] | +2.135 [+1.774, +2.545] | +4.762 [-3.968, +14.286] pp | +43.651 [+33.333, +53.968] pp | +0.007 [-0.005, +0.021] bits | -0.002 [-0.019, +0.015] bits |
| Complete causal cache | 164.7 | -0.614 [-1.005, -0.228] | -2.762 [-3.372, -2.197] | +2.148 [+1.777, +2.564] | +3.968 [-5.556, +14.286] pp | +42.857 [+31.746, +53.968] pp | +0.000 [+0.000, +0.000] bits | +0.000 [+0.000, +0.000] bits |

## Interpretation status

The table is intentionally source-localization-first. A region should be treated as carrying semantic history only if its signed effect replicates across discovery and confirmation, approaches the complete-K/V control, and cannot be explained by entropy alone. Layer-band localization should be attempted only for such a region.

Canonical figure: `figures/qwen36_fixed_a_kv_source_localization.png`.
