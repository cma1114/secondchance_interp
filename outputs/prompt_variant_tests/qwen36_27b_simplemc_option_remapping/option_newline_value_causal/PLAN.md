# Causal test of the option-newline candidate-value coordinate

## Question

Does the linearly decoded candidate-value component at the first-presentation
W1 option newline causally contribute to the later Game--Neutral difference in
semantic W1 avoidance?

## Frozen cohort

Use the 145 same-content/same-letter selectedness-sensitive questions whose W1
is displayed at B, C, or D: 74 discovery and 71 confirmation. W1=A is excluded
from the causal primary analysis because its option-newline residual is exactly
identical across the chosen and unchosen presentations; later distractors
cannot retroactively alter that already-written state. Preserve every target's
historical four-question physical cohort, batch size 4, SDPA implementation,
canonical remapped second presentation, and action-matched `incorrect` versus
`lost` prompts.

The confirmation split contains 38 W1 != W2 conflict questions and 33 W1 = W2
questions. Discovery contains 41 and 33, respectively.

## Coordinate and intervention

At each readout, the frozen probe score for displayed letter `l` is

`score(x,l) = ((x - letter_mean[l]) / scale) dot weight`.

Equivalently, its raw-residual gradient is `g = weight / scale`. At only the
first-presentation W1 option-closing newline, change the residual by the
minimum-L2 update needed to set this score to a target value:

`x' = x + (target_score - live_score) g / ||g||^2`.

Apply this clamp after blocks/readouts 33--56, where the held-out candidate-
value signal emerges and is strongest. Do not alter any other token position
or residual direction.

For every question, obtain the targets from the already-completed six-
permutation cache:

- `natural`: no intervention;
- `chosen-sham`: clamp to the cached identity/chosen W1 score;
- `devalue`: clamp to the cached same-content/same-letter score from the frozen
  presentation where W1 loses;
- `opposite`: move equally far in the other direction, targeting
  `2 * chosen_score - unchosen_score`.

The sham controls numerical drift between the cached Baseline prefix and the
live Game/Neutral execution. The opposite edit supplies a sign-reversal control.

## Predictions and endpoints

The primary held-out endpoint is the conflict-trial interaction in W1-minus-W2
margin:

`(devalue - natural)_Game - (devalue - natural)_Neutral`.

If the coordinate is merely generic W1 evidence, devaluation should move Game
and Neutral similarly. If it helps bind “W1 was my earlier choice” to opposite
policies, devaluation should specifically shrink the Game--Neutral semantic-W1
avoidance gap: W1 should recover in Game relative to Neutral. The opposite edit
should reverse that interaction.

Also report, separately for conflict, no-conflict, and all eligible questions:

- W1 selection and W2 selection;
- switching away from W1;
- W1-minus-W2 margin and W1 centered A--D evidence;
- A--D entropy and spread;
- Game-minus-Neutral effects;
- natural, pre-clamp, and post-clamp probe score;
- raw residual L2 dose and dose as a fraction of residual norm;
- W1-letter strata B, C, and D.

Use paired question bootstrap confidence intervals. Treat the discovery and
confirmation splits as separate replications; do not select a layer band from
the causal outcomes.

## Interpretation

A replicated Game--Neutral interaction with the predicted sign, plus an
opposite-direction sign control and a negligible chosen-sham effect, would
show that the candidate-value coordinate is causally used differently by Game
and Neutral. Similar W1 changes in both conditions would establish only a
generic candidate-evidence role. A null would show that the decoded coordinate
is readable but not itself the operative selectedness-binding channel.

## Storage and presentation

Keep the 7.864 GB residual cache on the retained stopped Vast host. Retrieve
only compact causal outputs. Save one canonical PNG in `figures/`, update the
canonical remapping report, and link the final report from the root README.

## Completed result

The experiment completed on 74 discovery and 71 confirmation questions. The
exact zero-dose sham had zero effect and natural A--D logits reproduced the
trusted run exactly. On the 38 held-out conflict questions, devaluation changed
the W1-minus-W2 margin by -0.013 [-0.062, +0.036] logits in Game and -0.053
[-0.107, 0.000] in Neutral. The prespecified Game-minus-Neutral interaction was
+0.040 [-0.029, +0.102] logits. W1 choice changed by +5.3 [0.0, +13.2]
percentage points in Game and 0.0 [-7.9, +7.9] in Neutral, giving an uncertain
+5.3 [-5.3, +18.4] point interaction.

Discovery's margin interaction was +0.057 [+0.006, +0.115], but its W1-choice
interaction was 0.0 [-9.8, +9.8] points. The equal/opposite sign control also
failed to replicate: -0.080 [-0.130, -0.033] logits in discovery versus -0.011
[-0.080, +0.061] in confirmation. The fitted coordinate is therefore a real
decodable correlate, but this intervention does not establish it as the causal
selectedness-binding mechanism.

- [Full causal report](analysis/REPORT.md)
- [Machine-readable summary](analysis/summary.json)
- [Canonical figure](../../../../figures/qwen36_option_newline_value_causal.png)
