# Exact source attribution for old and fresh scores inside 2P semantics

## Question and method

This analysis asks which exact model writes supply first-presentation score
information and fresh second-presentation score information to the semantic
tokens of each second-presentation option line.

It reuses the validated 500-question residual workspace. The unique old-score
and fresh-score directions were fit on the frozen 251-question discovery split
after jointly controlling the two scores and both display positions, then
applied without refitting to the 249-question confirmation split.

For every semantic option token, the analysis measures:

1. the exact ordinary-attention output attributable to each exhaustive prompt
   region at all 16 ordinary-attention layers; and
2. the complete mixer and MLP write at every layer 1--64.

Each write is divided by that token's natural residual RMS and projected onto
the frozen score direction. A source correlation asks whether the variation in
that exact write across questions and candidates tracks the target score. A
normalized write is a signed residual contribution, not an answer logit.

The complete attention partition reconstructs the mixer with relative RMSE
0.00162 across 256,000 cells. Natural cached activations were not recomputed.

## First-presentation score: Game

The matching first-presentation option line is the dominant ordinary-attention
source of old score in the second-presentation semantic residual. Its held-out
content-mean correlation with unique old score is:

| Layer | Correlation |
|---:|---:|
| 32 | 0.265 |
| 36 | 0.342 |
| 44 | 0.370 |
| 48 | 0.401 |

At layer 32, its exact normalized write is +0.131 for R1, -0.039 for R2,
-0.224 for R3, and -0.400 for R4. Thus this route transfers a graded record of
the old candidate ranking, not merely the identity of W1.

The other three first-presentation lines also contribute, but less strongly:
their correlation peaks at 0.244 at layer 44. The first-answer boundary is not
a comparably large carrier.

## First-presentation score: Neutral

Neutral uses essentially the same retrieval route. The matching-line
correlations are 0.267, 0.336, 0.371, and 0.399 at layers 32, 36, 44, and 48.
At layer 32, the exact rank writes are +0.131, -0.040, -0.224, and -0.396.

The near identity of the Game and Neutral layer-32 writes is important: the
initial retrieval of old rank is shared. The task distinction is created by
what later computation does with that retrieved ranking, not by Game failing
to retrieve it.

## Fresh second-presentation score in each task

Fresh score does not have one comparably dominant ordinary-attention source.
No held-out source correlation exceeds 0.137. Contributions are distributed
over the current question, other second-presentation candidates, and earlier
context.

The clearest component-level fresh-score writes are MLP computations around
layers 29--31. At the content mean, MLP 30 correlates 0.182 with fresh score in
Game and 0.189 in Neutral; MLP 31 correlates 0.188 and 0.182. Positive fresh-
score writes continue through the 40s, including MLP 47.

The defensible interpretation is that fresh score is computed from the current
presentation by a distributed comparison, rather than copied from one prompt
position. This analysis does not isolate a single fresh-score source.

## Where the two task policies first diverge strongly

The strongest discovery-selected and held-out-replicating task-dependent
old-rank write into the final semantic token is **MLP 49**.

MLP 49's absolute normalized old-score writes are:

| First-pass rank | Game | Neutral | Game minus Neutral |
|---|---:|---:|---:|
| R1 | +1.177 | +1.386 | -0.209 |
| R2 | +0.639 | +0.685 | -0.046 |
| R3 | +0.410 | +0.292 | +0.118 |
| R4 | -0.263 | -0.432 | +0.169 |

Both tasks therefore still receive an old-rank-preserving MLP write: R1 is
favored most and R4 least. Neutral's write is substantially more top-heavy.
Game reduces the R1/R2 reinstatement and shifts relative support toward R3/R4.

The bivalent difference, R4 minus the mean of R1 and R2, is +0.330
`[+0.254,+0.407]` on discovery and +0.296 `[+0.223,+0.371]` on confirmation.
The same MLP also produces a held-out bivalent difference of +0.126
`[+0.082,+0.170]` along the distinct fresh-score direction.

This corrects an overly simple verbal account in which Game necessarily writes
a negative W1 value at this point. MLP 49 is pro-R1 in both tasks; the policy
difference is primarily **stronger Neutral reinstatement plus relative Game
redistribution**, not an absolute sign reversal in this component.

At ordinary-attention layer 52, the current second-presentation line continues
the same transformation at its final semantic token. Its old-score write is
`[+0.545,+0.405,+0.336,+0.250]` in Game and
`[+0.612,+0.452,+0.348,+0.244]` in Neutral. The bivalent task difference is
+0.063 `[+0.040,+0.087]`. This is a later continuation, not the source of the
earlier MLP-49 write.

## What is established

1. Old rank reaches the matching second-presentation semantic line through
   ordinary attention to the matching first-presentation line, most clearly
   across layers 32--48.
2. This initial semantic-rank retrieval is nearly identical in Game and
   Neutral.
3. Fresh score is built by distributed current-presentation computation, with
   prominent MLP contributions around layers 29--31 and continuing through the
   40s.
4. MLP 49 writes the strongest replicating candidate-specific task divergence
   into the final semantic token: Neutral reinstates old rank more strongly,
   while Game redistributes relative support away from R1/R2 toward R3/R4.

## Remaining causal gap

These are exact additive-write and source-attribution results, not lesions.
They identify MLP 49 as the principal observed writer of the bivalent policy
state, but do not yet establish that this MLP write is necessary for final
preferential switching. The clean causal follow-up is a narrow intervention on
the MLP-49 task-difference write at the second-presentation semantic tokens,
with Game and Neutral effects reported separately.

## Artifacts

- [Canonical figure](../../../../../figures/qwen36_second_presentation_score_source_attribution.png)
- [Compact summary](score_source_attribution.json)
- [Compact per-question arrays](score_source_attribution_arrays.npz)
