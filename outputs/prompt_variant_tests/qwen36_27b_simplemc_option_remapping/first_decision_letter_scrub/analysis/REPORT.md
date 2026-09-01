# Continuous first-decision letter scrub

## Bottom line

This is a clean negative result: continuously removing the directly readable A–D identity at the first-decision token after every readout 48–63 does **not** causally explain Game's preferential avoidance of semantic W1.

Natural held-out conflict-trial W1 choice is 17.6% in Game and 40.4% in Neutral, a 22.8-point preferential-avoidance gap. Continuous letter scrubbing changes Game W1 choice by +0.735 [+0.000, +2.206] points, Neutral by +0.735 [+0.000, +2.206] points, and the primary Game-minus-Neutral interaction by +0.000 [-2.206, +2.206] points. It therefore explains 0% of that held-out gap at the point estimate.

The corresponding held-out continuous W1-minus-W2 interaction is +0.001 [-0.015, +0.017] logits. Discovery gives +4.380 [+0.000, +8.759] points but only +0.010 [-0.007, +0.028] logits. The discovery choice movement does not replicate in confirmation or in the continuous margin.

The edit does have a small, almost perfectly shared effect on conflict trials: W1-minus-W2 rises by +0.019 [+0.004, +0.033] logits in Game and +0.018 [+0.004, +0.031] in Neutral. A–D spread falls by -0.035 [-0.052, -0.018] and -0.040 [-0.060, -0.022] logits, while entropy rises slightly in both. Thus this late A–D coordinate participates in generic candidate geometry/flattening, not the condition-specific semantic binding that makes Game avoid W1.

The mechanistic implication is narrow but important: the explicit late answer-letter state at the first-decision token is not the route that carries the remembered winner into Game-specific suppression. This does not exclude an earlier relay before readout 48, or winner information encoded in other dimensions or positions.

## Validation

- Natural-versus-identity maximum A–D logit error: 0; choice changes: 0.
- Same-batch natural trusted-choice agreement: 98.80%.
- Maximum A–D logit error versus the preceding validated same-host natural run: 0; choice changes: 0.
- First-decision Baseline-choice agreement: 98.40%.
- Maximum post-projection A–D coefficient norm: 3.33338e-05.
- Mean removed A–D component norm across targeted readouts: 2.0125.

## No-conflict trials

Held-out no-conflict Game W1-choice effect: +1.770 [+0.000, +4.425] points; Neutral: +0.000 [-2.655, +2.655] points; interaction: +1.770 [-1.770, +5.310] points.

The machine-readable summary contains all/conflict/no-conflict effects for W1 and W2 choice, switching, probabilities, margins, entropy, and A–D spread on both frozen splits.

## Letter-bias and matched-decision checks

On held-out conflict trials with W1=A (n=94), the W1-choice interaction is +1.064 [+0.000, +3.191] points. With W1 in B–D (n=42), it is -2.381 [-7.143, +0.000] points.

Restricting to the 133 held-out conflict questions whose current first decision matches W1 gives +0.000 [-2.256, +2.256] points and +0.002 [-0.015, +0.019] logits.

Canonical figure: `figures/qwen36_first_decision_letter_scrub.png`.
