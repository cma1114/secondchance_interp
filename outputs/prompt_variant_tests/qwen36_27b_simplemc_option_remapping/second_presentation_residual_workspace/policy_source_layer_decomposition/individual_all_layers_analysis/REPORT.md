# Individual-layer feedback-policy transmission

## Question and intervention

This is the complete L1--L64 refinement of the feedback-suffix crossover. The aligned source is the whole policy-bearing suffix `incorrect/lost . Choose the answer again .`. Its residual states remain those of the recipient prompt. We cross only the downstream writes made from those source tokens between paired Game and Neutral runs.

For each layer, **sufficiency** crosses the suffix writes at that layer alone. **Necessity loss** compares the complete all-layer crossover with a crossover at every layer except that layer. The outcome is the fraction of the paired donor-task A--D logit vector transferred into the recipient. Thus the Game rows ask how Neutral suffix writes change a Game run, while the Neutral rows ask how Game suffix writes change a Neutral run. The tasks are never pooled.

All 500 canonical questions were completed. The frozen discovery and confirmation splits contain 251 and 249 questions. Every output was finite; the corrected and same-batch natural controls both reproduced with maximum absolute error 0.0.

## Findings

The complete all-layer crossover transfers nearly the entire donor-task state on confirmation: **0.925 [0.909, 0.940]** into Game and **0.941 [0.927, 0.955]** into Neutral. No individual layer explains most of this effect. The policy is transmitted by a distributed sequence of writes.

It also transfers the expected discrete and rank-shaped policy. Game's switch rate moves from **62.7%** naturally to **47.0%** when it receives Neutral suffix writes, close to Neutral's natural **45.0%**. Neutral's switch rate moves in the reciprocal direction, from **45.0%** to **60.6%**, close to Game's natural rate. The all-layer crossover changes the bivalent R4-minus-mean(R1,R2) score by **-0.451** in Game and **+0.455** in Neutral, carrying the opposite rank policies rather than merely a generic task difference.

For **Game**, layer 36 is dominant. Crossing it alone transfers **0.130 [0.111, 0.151]**; omitting it loses **0.120 [0.100, 0.143]**. Layer 45 is second on both tests: **0.070 [0.055, 0.084]** alone and **0.069 [0.055, 0.083]** lost when omitted. Other practical effects form a broad L28--50 cluster, including ordinary-attention layers 32, 40, 44, and 48 and GLA layers 33--35, 45--47, and 50.

The rank-shaped result is especially clean at the two peaks. On confirmation, L36 alone changes the bivalent score by **-0.087** in Game and **+0.056** in Neutral; L45 alone changes it by **-0.036** and **+0.032**. All four directions and both corresponding necessity effects replicate on discovery. Individual winner changes are thresholded and less stable across splits, so the continuous rank/logit effects are the primary layer-localized result.

For **Neutral**, layer 36 is again dominant: **0.119 [0.100, 0.143]** alone and **0.148 [0.126, 0.170]** lost when omitted. Layer 45 is again second: **0.068 [0.055, 0.082]** alone and **0.078 [0.061, 0.095]** lost when omitted. The remaining practical effects again span L28--50 and both carrier families.

The same two peaks and the same mid-layer concentration appear on discovery. The precise magnitude of some smaller layer effects varies between splits, so the defensible result is the replicated structure—not a claim that every small confidence interval marks a distinct mechanism.

The carrier-family controls agree with the individual-layer map. On confirmation, ordinary-attention writes alone transfer **0.349 [0.316, 0.378]** into Game and **0.450 [0.420, 0.484]** into Neutral. GLA writes alone transfer **0.445 [0.410, 0.478]** into Game and **0.553 [0.520, 0.588]** into Neutral. Neither family alone reproduces complete transfer; the dominant peaks alternate between ordinary-attention L36 and GLA L45.

## Bottom line

The evaluation feedback does not travel through one late policy layer. Its causal transmission is distributed, becomes practically concentrated in the middle-to-late stack, and has two reproducible individual maxima: **L36** and **L45**, in that order, in both Game and Neutral. Weak effects occur outside the central window, but almost all large single-layer effects lie between L28 and L50.

Confidence intervals are question-bootstrap 95% intervals.

Canonical figure: [qwen36_feedback_policy_individual_layers.png](/Users/christopherackerman/repos/secondchance_interp/figures/qwen36_feedback_policy_individual_layers.png)

Exact estimates: `individual_layer_estimates.csv`; machine-readable controls and definitions: `summary.json`.
