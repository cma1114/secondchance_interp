# Seed-OSS 36B SimpleMC matching-history blockade

## What was tested

For every frozen SimpleMC question and in both Game and Neutral, every token of each complete 2P option line was prevented from attending to every token of its semantically matching complete 1P option line at all 64 Seed attention layers. The cyclic control edited the same receivers and layers but used the next old-rank source line. This is a whole-line attention-edge intervention, not a residual replacement or token-localization result.

## Frozen confirmation results

Matching blockade minus cyclic control, candidate-centered A-D logits:

| Task | W1 | W2 | W3 | W4 |
|---|---:|---:|---:|---:|
| Game | +1.197 [+0.702, +1.700] | -0.045 [-0.555, +0.455] | -0.527 [-1.017, -0.060] | -0.625 [-1.041, -0.219] |
| Neutral | +0.119 [-0.513, +0.737] | +0.065 [-0.530, +0.653] | -0.152 [-0.696, +0.387] | -0.032 [-0.476, +0.387] |
| Game minus Neutral | +1.078 [+0.777, +1.385] | -0.110 [-0.392, +0.178] | -0.375 [-0.619, -0.135] | -0.593 [-0.790, -0.388] |

Old-semantic-W1 choice rates:

| Scenario | Game | Neutral | Game minus Neutral |
|---|---:|---:|---:|
| Natural | +36.5 [+30.9, +42.2] | +45.8 [+40.2, +51.8] | -9.2 [-13.7, -5.2] |
| Matching blockade | +49.0 [+43.0, +55.0] | +47.8 [+41.8, +53.8] | +1.2 [-3.6, +6.0] |
| Cyclic wrong-line blockade | +39.8 [+34.1, +45.4] | +45.8 [+40.2, +51.8] | -6.0 [-10.0, -2.4] |

Primary matching-minus-cyclic change in the Game-minus-Neutral W1-choice gap: **+7.2 [+1.6, +12.9] percentage points**.

## Interpretation

The central Game mechanism replicates across architectures. Relative to the
cyclic wrong-line control, blocking Seed's true matching 1P→2P option-line
reads raises old-W1 evidence by **1.197 logits** and lowers W3/W4 evidence by
**0.527/0.625 logits** on the untouched confirmation half. Therefore, with
the route intact, matching candidate history is doing the opposite: it
selectively reduces the old winner and supports weaker old candidates in
Game. That rank-specific result is incompatible with equal answer noise.

The choice result makes the causal role concrete. Naturally, Seed chooses its
old semantic winner on 36.5% of Game trials and 45.8% of Neutral trials, a
-9.2-point Game-minus-Neutral gap. After blocking all four true matching
routes, the rates become 49.0% and 47.8%, eliminating the point-estimate gap.
The equal-structure cyclic lesion retains a -6.0-point gap. Thus the primary
matching-minus-cyclic change is +7.2 points `[+1.6,+12.9]`. The independent
discovery half agrees: +8.8 points `[+2.0,+15.5]`, with the same rank-shaped
Game logit effect.

The stronger Qwen-SimpleMC claim about Neutral does **not** reproduce here.
Seed's confirmation Neutral matching-minus-cyclic W1--W4 estimates are all
uncertain and include zero. This is the same asymmetry later observed when the
Qwen mechanism was extended to TriviaMC: the policy-dependent Game route is
stable, while natural Neutral's use of this exact direct matching route can
vary across datasets or models.

This completes only the recollection part of the cross-architecture test. It
does not yet show that Seed's `incorrect`/`lost` feedback state is causally
transferable or that it changes the matching route. Those are the gated policy
crossover and policy-by-history experiments.

## Validation and scope

- Natural reproduction maximum absolute error: 0.00000000.
- Natural displayed-choice agreement: 100.00%.
- All outputs were finite and every executed source and receiver span was nonempty.
- The intervention covered all 64 grouped-query attention layers and no GLA or recurrent state exists in Seed.
- Correctness is not an endpoint.
