# Nonmatching first-presentation history factorial

## Bottom line

The other three first-presentation lines are causally used, but **not to create
the Game-versus-Neutral rank policy**. Blocking all three while preserving each
candidate's semantic match shifts evidence away from R1 and toward R3/R4 almost
identically in Game and Neutral:

- Game within-task rank-slope change: **+0.093**
  [+0.066, +0.120].
- Neutral within-task rank-slope change: **+0.089**
  [+0.060, +0.118].

Thus intact nonmatching-line reads support the old high-ranked candidates in
both tasks. They are a shared ranking/evidence route, not the source of the
distinctive revision policy.

## The task-specific result

- Natural Game-minus-Neutral rank slope: **+0.243**
  [+0.207, +0.279].
- After blocking all three nonmatching lines while preserving the semantic
  match: **+0.247**
  [+0.213, +0.280].
- Change caused by that intervention: **+0.004**
  [-0.021, +0.029].

The null task interaction replicates across trial types:

- conflict: **-0.017**
  [-0.051, +0.017];
- no conflict: **+0.029**
  [-0.009, +0.066].

The behavioral endpoint says the same thing. Blocking the three nonmatches
changes held-out W1 choice by **-6.4**
points in Game and **-6.0**
points in Neutral; their interaction is **-0.4**
[-6.6,
+5.8] points.

In contrast, blocking only the semantic matches collapses the held-out
Game-minus-Neutral rank slope to **+0.025**
[-0.006, +0.056],
and blocking all four lines gives **+0.007**
[-0.017, +0.031].
Removing the nonmatching lines after the match is already absent adds no
reliable overall policy-slope effect.

The narrow conclusion is therefore: direct access to the other three original
lines contributes shared rank evidence, but the condition-specific mapping of
first-pass rank into retention versus revision is carried through the matching
line or another preserved input already bound to it. This experiment does not
by itself distinguish a contextualized rank code inside that matching line
from a rank/policy signal supplied through the preserved answer boundary, GLA
state, or feedback pathway.

See `rank_results.csv` for R1--R4 effects and `aggregate_results.csv` for all,
conflict, and no-conflict endpoints in discovery and confirmation.

## Exact intervention

At every ordinary-attention layer 4--64 and for all four repeated option lines
simultaneously, the primary lesion blocks attention to the other three complete
first-presentation option lines while preserving the complete semantically
matching line. Matching-only and all-four blockades complete the causal
factorial. No GLA state, feedback token, question token, answer boundary, or
later decision query is directly edited.

## Validation

- Natural A-D-logit maximum error versus trusted results:
  **0.000000**.
- Natural answer agreement: **100.0%**.
- Prompt hashes match the prior exhaustive source run:
  **True**.
- Rank/content alignment matches the prior run:
  **True**.
- Matching-only maximum A-D-logit error versus the prior independently run
  layers-4--64 joint matching lesion:
  **0.000000**.
- Matching and nonmatching token sets are disjoint and exactly partition all
  four first-presentation option lines.

## Artifacts

- Canonical figure: `figures/qwen36_nonmatching_history_factorial.png`
- Machine-readable summary: `summary.json`
- Rank-level table: `rank_results.csv`
- Aggregate behavioral/rank table: `aggregate_results.csv`
