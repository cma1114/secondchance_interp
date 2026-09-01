# Output-preserved evaluation-period GLA update: consolidated result

This is the canonical corrected analysis. The intervention copies only the evaluation-period GLA recurrent-memory update; it restores the source token's local output to the target-natural value, preventing donor information from also escaping through the period token's residual, ordinary-attention K/V, MLP, or short convolutional path.

Primary analysis uses all 273 W1 != W2 questions across the frozen discovery and confirmation splits, with the two frozen splits also reported separately.

## Portable state

The natural Evaluation-minus-Matched-Neutral W1-minus-W2 margin difference is +0.469 [+0.362, +0.575]. Copying the Evaluation period's recurrent update into Neutral transfers +0.097 [+0.056, +0.139]; the reverse Neutral-into-Evaluation transplant transfers +0.091 [+0.054, +0.127] in the opposite causal direction.
The corresponding displayed-answer W1-selection transfers are +3.7 [+0.0, +7.3] and +5.1 [+1.8, +8.4] percentage points. The Evaluation-to-Neutral estimate is +20.0 [+0.0, +38.2]% of the natural W1-selection difference, but its interval is necessarily wide because answer selection is discrete.
The Evaluation update also transfers +0.035 [+0.023, +0.048] bits of A-D entropy; this route is therefore not a pure scalar W1-suppression channel.

The split asymmetry is material: the all-GLA bidirectional margin transfer is +0.047 [+0.007, +0.086] in discovery and +0.141 [+0.094, +0.188] in confirmation. Both frozen splits pass the prespecified joint gate, but only confirmation has both directional intervals above zero. The route is replicated, while its magnitude is heterogeneous.

## Localization

Only `blocks_25_32` passed the frozen discovery screen and replicated on confirmation. It contains the tested GLA blocks 25, 26, 27, 29, 30, 31. Its bidirectional margin transfer is +0.037 [+0.017, +0.057] in discovery and +0.073 [+0.049, +0.099] in confirmation. Pooled, it is +58.4 [+42.5, +82.5]% of the all-GLA bidirectional transfer. The band is thus a substantial sufficient carrier, but not the whole recurrent route.

| GLA block | Alone: bidirectional margin transfer | Deleting it from all-GLA: loss of transfer |
|---:|---:|---:|
| 25 | -0.002 [-0.011, +0.007] | +0.006 [-0.007, +0.019] |
| 26 | +0.012 [+0.003, +0.021] | +0.013 [-0.000, +0.026] |
| 27 | +0.015 [+0.005, +0.024] | +0.014 [+0.000, +0.028] |
| 29 | +0.002 [-0.006, +0.010] | +0.006 [-0.009, +0.020] |
| 30 | +0.005 [-0.005, +0.014] | +0.015 [+0.001, +0.029] |
| 31 | +0.000 [-0.009, +0.009] | +0.006 [-0.009, +0.020] |

Blocks 26 and 27 each show small positive block-alone effects after pooling, but neither is independently positive on both frozen splits. More importantly, every leave-one-block-out intervention retains nearly the complete all-GLA transfer. No individual block is a necessary bottleneck. The portable update is distributed and redundant at single-block resolution, with its clearest jointly sufficient carrier in blocks 25–32.

## What the correction changes

The historical transplant changed both the recurrent memory update and the period token's own downstream-visible residual, producing much larger headline effects. The corrected output-preserved result establishes a real persistent GLA-memory route, but not that this route alone explains most of the behavioral task difference. The remaining corrected source-trace and relay controls test where the evaluation information travels outside this route.
