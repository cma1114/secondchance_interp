# Seed-OSS 36B TriviaMC matching-history blockade

## What was tested

For every frozen TriviaMC question and in both Game and Neutral, every token of each complete 2P option line was prevented from attending to every token of its semantically matching complete 1P option line at all 64 Seed attention layers. The cyclic control edited the same receivers and layers but used the next old-rank source line. This is a whole-line attention-edge intervention, not a residual replacement or token-localization result.

## Frozen confirmation results

Matching blockade minus cyclic control, candidate-centered A-D logits:

| Task | W1 | W2 | W3 | W4 |
|---|---:|---:|---:|---:|
| Game | +1.607 [+1.056, +2.153] | -0.327 [-0.780, +0.108] | -0.603 [-0.989, -0.202] | -0.677 [-1.065, -0.294] |
| Neutral | +0.330 [-0.259, +0.925] | +0.173 [-0.343, +0.671] | -0.393 [-0.822, +0.029] | -0.110 [-0.490, +0.256] |
| Game minus Neutral | +1.277 [+0.942, +1.618] | -0.500 [-0.798, -0.201] | -0.210 [-0.434, +0.013] | -0.567 [-0.778, -0.361] |

Old-semantic-W1 choice rates:

| Scenario | Game | Neutral | Game minus Neutral |
|---|---:|---:|---:|
| Natural | +68.4 [+62.8, +74.0] | +75.6 [+70.4, +80.8] | -7.2 [-11.6, -3.2] |
| Matching blockade | +76.8 [+71.6, +82.0] | +76.8 [+71.6, +81.6] | +0.0 [-3.2, +3.2] |
| Cyclic wrong-line blockade | +68.0 [+62.4, +73.6] | +78.0 [+72.8, +82.8] | -10.0 [-14.0, -6.0] |

Primary matching-minus-cyclic change in the Game-minus-Neutral W1-choice gap: **+10.0 [+5.2, +14.8] percentage points**.

## Interpretation

The frozen replication gate passed on both prespecified endpoints. In Game,
blocking each 2P option line's access to its matching 1P line raised old-W1
evidence and reduced evidence for W3/W4 relative to an equal-structure cyclic
wrong-line lesion. The corresponding W1 task interaction was +1.277
[+0.942, +1.618]. At the displayed-choice level, natural Game chose old W1
7.2 percentage points less often than Neutral; the matching blockade removed
that difference exactly at the point estimate (76.8% versus 76.8%), whereas
the cyclic control retained a -10.0-point gap. Thus the Seed replication is not
only behavioral: semantically matching 1P-to-2P access is causally required for
Seed's preferential old-W1 avoidance in Game on TriviaMC as well as SimpleMC.

Neutral again does not show a stable matching-specific W1 profile: its
confirmation W1 estimate is +0.330 with a confidence interval spanning zero.
The cross-model replicated claim is therefore the policy-dependent Game use of
matching recollected history, not an identical Neutral route in Seed.

## Validation and scope

- Natural reproduction maximum absolute error: 0.00000000.
- Natural displayed-choice agreement: 100.00%.
- All outputs were finite and every executed source and receiver span was nonempty.
- The intervention covered all 64 grouped-query attention layers and no GLA or recurrent state exists in Seed.
- Correctness is not an endpoint.
