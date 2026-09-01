# Baseline function of Qwen3.6-27B Mixers 56 and 63

Confirmation questions: **249**. The equal-letter mean-ablation source was estimated on the disjoint 251-question discovery set. All prompts use the canonical `baseline_matched_empty_history` explicit ChatML format.

## Natural behavior

- Baseline accuracy: 37.3%
- Game accuracy: 32.5%
- Natural Game switch rate relative to Baseline: 42.2%

## Causal effects

Mean ablation replaces a Baseline mixer output with the answer-letter-balanced discovery-set mean. Baseline-into-Game insertion replaces a Game output with the paired same-question natural Baseline output. Values below are paired changes from the natural target condition.

| Intervention | Baseline answer changed | Baseline accuracy | Baseline spread | Baseline winner advantage | Game switch rate | Game spread | Game winner advantage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mixer 56 | +3.614 pp [+1.606, +6.024] | +0.402 pp [-1.606, +2.410] | -0.111 [-0.125, -0.098] | -0.134 [-0.152, -0.117] | +0.000 pp [-2.811, +2.811] | +0.010 [+0.004, +0.016] | +0.030 [+0.019, +0.041] |
| Mixer 63 | +6.426 pp [+3.614, +9.639] | +0.000 pp [-2.410, +2.410] | +0.032 [+0.019, +0.046] | +0.031 [+0.001, +0.062] | +4.819 pp [-0.402, +10.040] | -0.076 [-0.095, -0.059] | -0.098 [-0.146, -0.052] |
| Both | +6.827 pp [+4.016, +10.040] | +1.606 pp [-0.803, +4.016] | -0.124 [-0.142, -0.107] | -0.159 [-0.189, -0.128] | -1.606 pp [-6.827, +3.614] | -0.047 [-0.065, -0.029] | -0.023 [-0.073, +0.025] |

## Interpretation

Panel A is a JLens finite-difference direct attribution, not a causal intervention: it asks what answer-aligned write is immediately present in each natural Baseline mixer output while downstream computation is held fixed. Panels B and C are causal final-logit effects.

### Mixer 56

Mixer 56 is an ordinary Baseline discrimination/sharpening component, although it is not specifically a winner booster. Its immediate JLens-aligned write is +0.407 to Baseline rank 1, +0.629 to rank 2, approximately zero to rank 3, and -1.018 to rank 4. Replacing its question-specific Baseline output with the discovery-set mean produces the inverse causal pattern: ranks 1 and 2 fall and rank 4 rises. Baseline spread decreases by 0.111 and original-winner advantage by 0.134; 3.6% of Baseline answers change, with no reliable accuracy effect.

Putting the paired Baseline Mixer-56 output into Game modestly restores sharpening: spread rises by 0.010 and original-winner advantage by 0.030. It does not, however, change the Game switch rate (0.0 percentage points, 95% CI -2.8 to +2.8).

### Mixer 63

Mixer 63 performs the opposite operation even during ordinary Baseline answering. Its immediate write is -0.316 to rank 1, -0.151 to rank 2, +0.090 to rank 3, and +0.377 to rank 4. It is therefore a late rank-opposed flattening/rebalancing component, not a normal winner-sharpening component. Mean-ablation removes part of that operation: Baseline spread increases by 0.032 and winner advantage by 0.031; 6.4% of answers change, again with no reliable accuracy effect.

Inserting the paired Baseline Mixer-63 output into Game makes Game **more**, not less, compressed: spread falls by 0.076 and winner advantage by 0.098. Switching rises by 4.8 percentage points, although its interval narrowly includes zero (-0.4 to +10.0).

### Consequence for the eight-output mechanism

Jointly inserting the Baseline outputs of Mixers 56 and 63 into Game does not restore ordinary behavior: the switch-rate change is -1.6 points (95% CI -6.8 to +3.6), spread falls by 0.047, and winner advantage does not change reliably. Their opposing natural roles largely cancel.

Thus the prior eight-output mediation result should not be described as Game simply suppressing ordinary late answer sharpening. Mixer 56 supplies normal question-specific discrimination, while Mixer 63 supplies normal late flattening. Their importance to the Game–Neutral intervention is contextual and depends on the coordinated sequence of late mixer states. The upstream mechanism and the relevant interacting features remain unidentified.

Figure: `figures/qwen36_27b_simplemc_corrected/baseline_mixer_function.png`

Numerical results: `baseline_mixer_function_summary.json` and `baseline_mixer_function_results.npz`.
