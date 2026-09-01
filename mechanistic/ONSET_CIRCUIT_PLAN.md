# Layers 47--51 onset-circuit experiment

## Question

The ordered Game-minus-Baseline transformation first becomes visible around
JLens readout 48. Four nearby components have held-out reciprocal causal
effects on that transformation: Mixer 47, MLP 49, Mixer 50, and Mixer 51. This
experiment asks whether they form a joint onset circuit and where the two
ordinary-attention mixers obtain their condition and question-specific
information.

All layer numbers are zero-indexed. Mixer 47 is the attention module in the
48th transformer block.

## Stage 1: four-component circuit

On the untouched 249-question confirmation split, replace component outputs at
the final decision position between paired Game and Neutral executions.
Interventions are reciprocal.

- Jointly patch all four components.
- Patch the early pair (Mixer 47 and MLP 49).
- Patch the later pair (Mixer 50 and Mixer 51).
- Patch all four minus each component in turn.

Primary outcomes are the complete four-rank Game--Neutral geometry, ordered
rank slope, original-winner advantage, A--D entropy and spread, and switching.
Joint and leave-one-out effects determine whether the components add,
overlap, or interact. Individual-component estimates come from the already
completed held-out runs and are not rerun merely for convenience.

## Stage 2: attention-head localization

Mixer 47 and Mixer 51 are ordinary 24-head softmax-attention modules. Patch
their pre-output-projection head contexts at the final decision position.

1. On the 251-question discovery split, sweep all 48 heads Neutral into Game.
2. Within each mixer, select the union of the three largest ordered-rank
   mediators and the three largest switching mediators. Selection uses only
   discovery questions.
3. On the untouched 249-question confirmation split, test every selected head
   in both directions, plus the selected heads jointly within each mixer and
   across both mixers.

The head intervention replaces a complete head context, not an A--D
projection or a fitted steering direction.

Because BF16 matrix kernels can change slightly with batch shape, every
batched head intervention is compared with an unpatched forward pass having
the identical prompts, batch size, and row layout. Batch-1 natural logits are
retained for the natural Game--Neutral gap, but are not used as the immediate
reference for a batched causal effect.

## Stage 3: source localization

For heads with held-out reciprocal evidence, intervene on the final-query
attention edges to semantically defined prompt spans:

- user feedback condition (`incorrect` or its Neutral counterpart);
- action phrase (`different answer` or Neutral action phrase);
- complete feedback sentence;
- condition-specific system instruction where present;
- first question and options;
- repeated question and options;
- `[redacted]` turn;
- local final-answer cue as a structural control.

Run the same source intervention in Game and Neutral wherever the span has a
matched counterpart. The primary source claim requires a differential effect
on the confirmed head-mediated rank geometry, not merely high attention
weight. Condition-only spans are reported as Game necessity tests and are not
treated as matched contrasts.

Source-edge interventions use the same batch-matched unpatched control as the
head interventions.

## Interpretation rules

- A component or head is a mediator only if the predicted sign appears on the
  held-out split and the reciprocal intervention is directionally consistent.
- A prompt span is an upstream source only if perturbing that edge changes the
  downstream rank-opposition or switching effect through a confirmed head.
- Attention weights alone are routing evidence, not causality.
- Joint and individual fractions are not added because component effects can
  overlap and interact.
