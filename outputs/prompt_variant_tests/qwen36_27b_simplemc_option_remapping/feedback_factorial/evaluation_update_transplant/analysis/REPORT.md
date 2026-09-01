# Action-matched evaluation-period GLA update: consolidated result

> **Historical non-output-preserved intervention; superseded for route-specific
> attribution.** This run also changed the source period's own downstream-visible
> output. Use the [corrected output-preserved report](../output_preserved/analysis/REPORT.md),
> which finds 0.097/0.091-logit reciprocal transfer and localizes 58.4% of the
> smaller recurrent route to blocks 25--32.

Primary analysis uses all 273 W1 != W2 questions across the frozen discovery and confirmation splits.

## Portable state

Under aggregated A-D-logit argmax, the natural Evaluation-minus-Matched-Neutral W1-avoidance difference is +18.3 [+13.2, +23.8] percentage points.
Putting the Evaluation period update into Neutral transfers +15.4 [+10.6, +20.1] A-D-argmax points, or +84.0 [+65.1, +106.4]% of that natural difference.
The corresponding W1-minus-W2 margin transfer is +0.428 [+0.332, +0.523]; the natural margin difference is +0.469 [+0.363, +0.575].
The reverse Neutral-into-Evaluation transplant restores W1 by +4.4 [+0.7, +8.4] points and shifts the margin by +0.057 [+0.005, +0.108].
The Evaluation update also transfers +0.068 [+0.040, +0.097] bits of A-D entropy; the natural entropy difference is +0.077 [+0.045, +0.110] bits.

## Localization

Only blocks 17–24 replicated as an isolated eight-block band. That band contains GLA blocks 17, 18, 19, 21, 22, and 23. Its bidirectional margin transfer is only +6.6 [+2.4, +11.3]% of the natural Evaluation-to-Neutral margin gap (and +14.4 [+8.7, +21.8]% of the all-GLA bidirectional transfer), so the isolated band is a small sufficient fragment rather than the whole mechanism.

| GLA block | Alone: bidirectional margin transfer | Deleting it from all-GLA: loss of transfer |
|---:|---:|---:|
| 17 | -0.001 [-0.011, +0.008] | +0.007 [-0.006, +0.021] |
| 18 | +0.002 [-0.009, +0.012] | -0.012 [-0.026, +0.003] |
| 19 | +0.002 [-0.007, +0.012] | -0.021 [-0.037, -0.006] |
| 21 | +0.004 [-0.004, +0.013] | -0.018 [-0.032, -0.004] |
| 22 | -0.007 [-0.016, +0.002] | -0.020 [-0.033, -0.006] |
| 23 | +0.002 [-0.008, +0.011] | -0.007 [-0.020, +0.007] |

No individual block is sufficient on the primary margin, and no individual deletion causes a replicated positive loss of the joint effect. The portable update is therefore distributed and synergistic at single-block resolution.

## Next decisive intervention

The reverse direction is also split-asymmetric: its discovery margin estimate is +0.010 [-0.060, +0.079], while the pooled estimate is carried by confirmation. 
The period update is strongly sufficient in the Evaluation-to-Neutral direction but the reverse transplant is smaller and less stable. The clean next test is the same all-GLA bidirectional transplant at (a) the aligned `incorrect`/`lost` word alone and (b) that word plus the evaluation-closing period together. Evaluation remains present at its native word in the current reverse-period intervention, so this test distinguishes an update first written by the evaluation word from a state consolidated at the period. Only if the two-token gate succeeds should it be localized further.
