# Action-closing-period causal mediation

## Question

The evaluation-closing period is already known to causally transmit most of
the Evaluation-versus-Matched-Neutral W1-avoidance effect. Does that causal
state pass through the shared period ending `Choose the answer again.`?

## Interventions

Use all 500 frozen remapped SimpleMC trials under the canonical action-matched
prompts. Preserve raw ChatML, empty thinking scaffolds, SDPA, and each exact
historical batch of four. In both directions between Evaluation and Matched
Neutral, keep the recipient prompt fixed and intervene only at the shared
action-closing period:

1. replace that token's complete post-block residual trajectory at all 64
   transformer blocks;
2. replace all 48 accumulated GLA recurrent matrix states immediately after
   that token;
3. apply both replacements jointly.

The state interventions use recipient-state reinsertion as the segmented-kernel
identity control. All effects are anchored to the previously validated exact
natural A-D logits. The already-completed evaluation-period write deletion is
the upstream causal benchmark; it is not recomputed.

## Outcomes

Report W1 selection, W2 selection, W1-minus-W2 margin, centered W1 evidence,
A-D entropy, generic compression, and extra W1 targeting. Analyze all trials,
W1 != W2 conflict trials, W1 = W2 non-conflict trials, and initial-W1=A versus
initial-W1=B-D. Report the pre-existing 251/249 split as a stability check.

The decisive quantity is how much of the natural Evaluation-minus-Neutral
effect moves with each action-period transplant, in each direction. A large
joint or GLA-state transfer would identify the action period as a causal
bottleneck. A weak transfer would show that the readable action-period state
is not where most of the upstream causal effect is carried.

## Status

Complete. The GLA-state intervention transferred most of the conflict-trial
effect in both directions; residual-only transfer was negligible. See the
[canonical report](analysis/REPORT.md) and
[machine-readable summary](analysis/summary.json).
