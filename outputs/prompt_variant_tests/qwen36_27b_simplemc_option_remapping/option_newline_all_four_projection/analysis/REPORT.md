# All-four option-newline candidate-value K/V-carrier projection

> **Invalid causal intervention.** The fitted score was defined after
> subtracting a displayed-letter mean, but this run projected the raw residual
> without that centering. It therefore removed large static A/B/C/D components
> in addition to the intended candidate-value deviation. It also lacked a
> same-path identity K/V-replacement control. The numerical outputs below are
> preserved for audit only and must not be interpreted as a candidate-value
> ablation.

## Design

Project the candidate-value direction out of all four first-presentation option-newline ordinary-attention K/V carrier states at input readouts 35, 39, 43, 47, 51, and 55.
Natural and projected executions used all 500 canonical remapped questions, the frozen 251/249 split, and exact historical physical batches.

## Held-out conflict result

On 136 confirmation conflict questions, W1 choice changed by +2.2 [-1.5, +5.9] points in Game and +5.9 [+0.7, +11.0] in Neutral. The Game-minus-Neutral interaction was -3.7 [-10.3, +2.9] points.

The W1-minus-W2 margin changed by +0.053 [+0.007, +0.098] logits in Game and +0.059 [+0.001, +0.114] in Neutral; interaction -0.006 [-0.052, +0.040].

Switching changed by -2.2 [-5.9, +1.5] points in Game and -5.9 [-11.0, -0.7] in Neutral. A--D entropy changed by +0.011 in Game and +0.008 in Neutral.

## Interpretation

The fitted direction is causally used in ordinary candidate scoring, but it is
not the Game-specific revision signal. Across all 249 held-out questions,
projecting it out changed the final answer on 9.6% of Game trials and 10.8% of
Neutral trials, and significantly reduced A--D spread in both conditions. On
conflict trials it restored the W1-minus-W2 margin by nearly the same amount in
Game and Neutral (+0.053 versus +0.059 logits).

The prespecified Game-minus-Neutral W1-choice interaction was null and pointed
away from selective Game restoration: +2.2 points in Game versus +5.9 points in
Neutral, interaction -3.7 [-10.3, +2.9]. Discovery did not reproduce even the
held-out Game W1 increase (Game -1.5 points, Neutral 0.0). Thus removing this
one-dimensional value coordinate from all four option carriers perturbs a
shared scoring computation; it does not explain why Game preferentially avoids
W1.

This conclusion is deliberately limited to the decoded one-dimensional
coordinate carried through ordinary-attention K/V at the six tested readouts.
It does not rule out a nonlinear or multidimensional selectedness code in the
option states.

## Validation and scope

The maximum held-out mean residual dose at any readout was 7.86% of residual L2. Mean absolute post-projection probe score was 0.0000; the maximum absolute post-score was 0.0000.

Discovery replication, no-conflict results, natural rates, and all secondary metrics are in `summary.json` and `effects.csv`.

Canonical figure: `figures/qwen36_option_newline_all_four_projection.png`.
