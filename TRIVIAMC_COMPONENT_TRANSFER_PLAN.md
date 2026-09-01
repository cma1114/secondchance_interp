# Qwen3.6-27B TriviaMC component-transfer plan

## Question

Do the final-position components causally localized on SimpleMC mediate the
same Game-versus-Neutral transformations on TriviaMC?

This is a preregistered transfer test, not a new component search. No TriviaMC
outcome is used to select or redefine a component.

## Fixed model and data

- Checkpoint: `Qwen/Qwen3.6-27B`, revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Dataset: all 500 questions in the frozen Qwen3.6-27B TriviaMC manifest.
- Prompts: the same faithful Baseline, Game, and corrected Neutral prompt
  constructors used for SimpleMC.
- Position: the final prompt position predicting the first answer token.

The run collects exact self-hosted Baseline A-D logits so that the original
winner, causal geometry, switching, and winner advantage are all defined from
this checkpoint rather than from provider-returned probabilities.

## Frozen components and interventions

The eight individual targets and all grouped scenarios are copied verbatim
from the SimpleMC held-out confirmation plan. Each is patched in both
directions using paired same-question component outputs:

- Neutral into Game: removal/necessity test.
- Game into Neutral: addition/sufficiency test.

The fixed groups are:

- flattening: MLP 63 and Mixers 39, 50, 51, 59, 60, 63;
- switching: Mixers 50, 51, 60, 62;
- union: all eight selected components.

## Outcomes

The primary outcomes and sign predictions are unchanged:

- flattening: A-D entropy and centered A-D logit spread;
- winner-changing behavior: original-winner advantage and switch probability.

Dataset-weighted estimates explain the observed TriviaMC sample. Equal-letter
macro estimates and A/B/C/D splits test whether transfer is robust to the
answer-letter imbalance found on SimpleMC. Accuracy and causal logit geometry
remain diagnostics.

The transfer succeeds strongly if the fixed groups move their paired outcomes
toward the source condition in both patch directions. Partial transfer—such as
flattening without letter-general switching—is reported as such rather than
reselecting components on TriviaMC.

## Execution and cost

A four-question smoke test must validate the exact checkpoint, prompt hashes,
baseline collection, and complete analysis before the full run starts. The
full run contains 22 patched scenarios plus Baseline, natural Game, and natural
Neutral: 12,500 forwards total. At the measured SimpleMC throughput, expected
compute is about 75-90 minutes and under $2 on the retained A100, comfortably
below the standing $15 cap. Outputs are resumable. The instance is stopped,
not destroyed, after retrieval and validation.
