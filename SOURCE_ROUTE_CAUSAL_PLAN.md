# Conditional source attribution for Mixers 56 and 63

## Question

The eight-component intervention causally mediates a substantial part of the
Game-specific ordered answer-rank transformation, but its individual effects
are strongly interaction-dependent. This experiment asks where the two
strongest conditional switch mediators—Mixer 56 and Mixer 63—obtain the
information they contribute.

## Frozen split and prompt

- Model: Qwen3.6-27B at pinned revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Dataset: all 500 SimpleMC questions.
- Discovery: the existing frozen 251-question discovery set.
- Confirmation: the untouched existing 249-question confirmation set.
- Prompt: the corrected `baseline_matched_empty_history` prompt serialized as
  explicit raw Qwen ChatML. Baseline, Game, and Neutral share the same system
  prompt and first presentation. Game and Neutral contain empty historical
  assistant content. Thinking scaffolds are present consistently at both
  assistant boundaries.

## Discovery: exhaustive source-route screen

Every Game and Neutral prompt token is assigned to exactly one of 21 fixed,
interpretable source spans: system text; first instruction; first question
stem; four first option spans; first choice cue; historical assistant turn;
feedback subject, condition, and action; second instruction; repeated question
stem; four repeated option spans; second choice cue; final assistant prefix;
and remaining structural ChatML tokens.

For all 251 discovery questions:

1. At Mixer 56 (ordinary attention, zero-based layer 55), remove every
   source-span × attention-head edge into the final decision query, renormalize
   the remaining attention probabilities exactly, retain Qwen's learned
   per-dimension output gate, and reconstruct the removed route's contribution
   after the output projection.
2. At Mixer 63 (Gated DeltaNet, zero-based layer 62), replay the exact recurrent
   update while setting beta to zero for every source-span × value-head route,
   then reconstruct the route's contribution after normalization and output
   projection.
3. Align each route's immediate A–D write by the same question's Baseline answer
   ranks and compare Game with Neutral.
4. Separately for Mixer 56 and Mixer 63, freeze the eight routes with the
   largest positive Game-minus-Neutral ordered-rank writes. Selection uses no
   confirmation questions.

The full matrices, attention masses, token-level attention weights, recurrent
write strengths, and selected routes are retained. The number eight is only a
fixed confirmation budget, not a claim that exactly eight routes exist.

## Held-out conditional causal confirmation

For all 249 confirmation questions, first reproduce the full eight-component
Game-into-Neutral and Neutral-into-Game interventions. Then, one selected source
route at a time, remove that route from its Mixer 56 or Mixer 63 source output
while leaving the other seven components and every other route unchanged.

The primary estimand is therefore the route's incremental effect *inside the
seven-other-component context*, not its isolated effect. Both intervention
directions are reported:

- insert Game-like computation into Neutral;
- remove Game-like computation from Game.

Primary outcomes are ordered answer-rank redistribution and switching away
from the Baseline winner. Paired bootstrap 95% confidence intervals use only
the frozen confirmation set. Every patched forward includes a same-batch
unpatched control row and is recentered on the corresponding single-trial
natural logits to control Qwen's batch-shape numerical drift.

## Interpretation limits

Mixer 56 route removal is exact for final-query attention edges. Mixer 63 route
removal is an exact beta-write counterfactual for the chosen source tokens, but
recurrent state interactions mean its per-source routes need not add linearly.
The held-out conditional interventions test whether a route matters for the
known eight-component mechanism; they do not by themselves prove that the
route is the only natural causal path from the prompt to the answer.
