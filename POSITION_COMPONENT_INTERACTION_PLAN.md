# Qwen3.6-27B component-interaction diagnostic

## Purpose

Determine why the eight-component joint patch has a much larger held-out
effect than the same components patched individually. The experiment does not
assume that the selected outputs constitute an additive circuit.

## Fixed data and interventions

- Model and prompts: identical to the corrected empty-history SimpleMC
  position-component experiment.
- Questions: the untouched 249-question confirmation split.
- Components, in causal order:
  1. feedback-end MLP 32;
  2. feedback-end MLP 44;
  3. decision Mixer 50;
  4. decision Mixer 52;
  5. decision Mixer 56;
  6. decision Mixer 60;
  7. decision Mixer 61;
  8. decision Mixer 63.
- Both patch directions are tested: Neutral outputs into Game and Game outputs
  into Neutral.

## Diagnostics

### Cumulative patches

Patch the first component, then the first two, continuing through all eight in
causal order. This identifies the step at which the coordinated intervention
acquires its large continuous and behavioral effect.

### Leave-one-out patches

Starting from the complete eight-component intervention, omit each component
in turn. The difference from the complete intervention is that component's
conditional contribution in the seven-component context.

## Outcomes

- ordered redistribution of evidence across each question's Baseline ranks;
- switching away from the Baseline winner.

Effects use paired question-bootstrap 95% confidence intervals. Cumulative and
conditional effects are reported separately for removal from Game and insertion
into Neutral.

## Interpretation boundary

The patch inserts component outputs cached from the natural source condition.
A coordinated patch can therefore transplant a coherent sequence of
condition-specific writes even when isolated writes are ineffective or
off-context. A large joint effect demonstrates a jointly effective output set;
it does not by itself prove that the outputs form a naturally executed circuit.
