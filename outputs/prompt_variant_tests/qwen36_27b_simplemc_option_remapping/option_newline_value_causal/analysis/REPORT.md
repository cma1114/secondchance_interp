# Causal test of the option-newline candidate-value coordinate

## Bottom line

The candidate-value coordinate is genuinely decodable at the first-presentation option newline, but this intervention does **not** establish that the coordinate causally controls Game-specific revision. The small predicted-direction margin interaction seen in discovery remained positive but was uncertain on held-out confirmation, while the prespecified opposite-sign control failed to replicate. Discrete answer changes were likewise unstable across splits.

## Design

At the first-presentation W1 option newline only, clamp the frozen candidate-value probe coordinate over readouts 33-56 to its matched same-content/same-letter unchosen-presentation score.
The run uses natural, exact zero-dose chosen-value sham, devaluation, and equal/opposite controls in both Game and Neutral, preserving historical physical batches. The causal sample excludes W1=A because the W1 newline is token-for-token identical before later distractors appear; it contains 74 discovery and 71 confirmation W1=B/C/D questions.

## Primary held-out result

On 38 confirmation conflict questions, devaluation changed the W1-minus-W2 margin by -0.013 [-0.062, +0.036] logits in Game and -0.053 [-0.106, +0.000] in Neutral. The Game-minus-Neutral interaction was +0.040 [-0.028, +0.103] logits.

W1 choice changed by +2.6 [+0.0, +7.9] points in Game and +0.0 [-7.9, +7.9] in Neutral. The interaction was +2.6 [-7.9, +15.8] points.

Switching away from W1 changed by -2.6 [-7.9, +0.0] points in Game and +0.0 [-7.9, +7.9] in Neutral; the interaction was -2.6 [-15.8, +7.9] points.

## Replication and controls

The discovery conflict interaction in W1-minus-W2 margin was +0.057 [+0.005, +0.116] logits, compared with +0.040 [-0.028, +0.103] on confirmation. The equal/opposite edit gave -0.080 [-0.130, -0.030] in discovery but only -0.011 [-0.078, +0.060] on confirmation, so the sign-reversal evidence did not replicate.

The chosen-sham edit had exactly zero dose and zero behavioral/logit effect. The devaluation edit changed at most 2.89% of residual L2 on average at any tested readout, and the post-clamp probe-score error averaged about 0.016 score units. Natural A-D logits reproduced the trusted run exactly.

The intervention also did not produce a stable entropy or A-D-spread effect across splits. Full conflict/no-conflict, letter-stratified, and secondary-metric results are in `summary.json` and `effects.csv`.

## Interpretation

This is an informative causal null. A linear decoder can read a context-dependent candidate-value/selectedness correlate from the semantic option state, but moving that one fitted coordinate from its chosen-presentation value to its matched unchosen-presentation value is not sufficient to reproducibly alter the later Game-versus-Neutral policy. The operative binding may be nonlinear, multidimensional, distributed across option states, or represented in a different feature basis. The decoder result remains valid; the stronger claim that its one-dimensional direction is the mechanism does not.

Canonical figure: `figures/qwen36_option_newline_value_causal.png`.
