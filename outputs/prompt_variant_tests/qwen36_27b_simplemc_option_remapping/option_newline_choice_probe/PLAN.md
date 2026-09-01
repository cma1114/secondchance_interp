# Option-newline selected-answer probe

> **Post-run clarification (2026-08-20):** The original premise that this
> exact newline had already been localized as carrying semantic content was too
> strong. Earlier work used newline residuals to construct content-aligned
> directions, but corrected cross-mapping tests found only modest
> mapping-invariant content geometry around layers 32--43 and strong late
> displayed-letter/position structure. Whole-line K/V transplants causally
> localized semantic history to the selected option line, not specifically to
> its newline. The candidate-value decoding result itself is unchanged.

## Question

Does the contextual residual at the exact option-closing newline position—the
position previously used to construct content-aligned option directions—also
carry a linearly readable candidate-value signal resembling “this seems like
the right answer”?

## Frozen data and position

Reuse the completed six-permutation Baseline design: 500 SimpleMC questions,
six mappings per question, and the historical physical cohorts of four
questions. For every presentation, collect post-block residuals at exactly four
positions: the newline token closing option lines A, B, C, and D. Do not collect
the final content token or the first-answer boundary.

The existing 251/249 question split remains frozen. All six mappings of a
question remain in the same split.

## Probe

At each of the 64 post-block readouts, train one shared linear scoring direction
on the 251 discovery questions. The direction assigns one scalar to each of the
four option-newline residuals; the predicted answer is the option with the
largest score. Before fitting, estimate and remove each displayed letter's mean
residual using discovery questions, then standardize residual dimensions using
discovery data. This prevents a static A-D position signature from constituting
the result.

Fit the direction as the normalized difference between the mean selected-option
representation and the mean of the three rejected-option representations. This
is an isotropic linear discriminant/ranker with no tuned hyperparameters.

## Frozen evaluation

On the 249 held-out questions, report across all six mappings:

- top-1 answer prediction accuracy at every readout, with question-bootstrap
  95% confidence intervals;
- the selected option's probe-score margin over the strongest rejected option;
- a letter-only answer-bias baseline;
- exact agreement of the rerun's natural choices with the completed
  six-permutation screen.

On the already frozen selectedness-sensitive pairs, compare the score attached
to the same W1 semantic content at the same displayed letter when W1 wins versus
when it loses. Stratify by W1 letter. W1=A is a built-in prefix sanity check:
because its complete prefix through the A newline is identical before later
distractors appear, its local residual and score must be identical across the
paired presentations.

## Interpretation

Held-out rank decoding above the static letter baseline establishes a linearly
readable candidate-value signal at the option-closing newline. It remains
correlational. The matched-pair comparison distinguishes a stable absolute
plausibility signal from a context-dependent selectedness signal. No causal
projection intervention is authorized automatically; design one only after the
decoding result is understood.

## Storage

Keep the 7.864 GB float16 residual cache only on the retained Vast
host. Retrieve only compact probe weights, metrics, report, and one canonical
PNG.

## Outcome

The exact collector reproduced all 3,000 prior choices and A--D logits exactly.
On 249 held-out questions, the option-newline ranker reached 64.9% [60.5,
69.1] at descriptive peak readout 53 versus 51.9% [46.7, 56.9] for the
letter-only predictor. The paired gain was 13.0 points [5.3, 20.8].

On 107 held-out same-content/same-letter selectedness-sensitive pairs, the W1
score at readout 53 was 4.36 units higher [3.06, 5.79] when W1 won than when it
lost. The effect was exactly zero for A, whose prefix is identical, and positive
for B--D, whose local states can incorporate earlier competitors. This supports
a context-dependent candidate-value signal at the option-closing newline. The
test is correlational and does not itself establish mapping-invariant semantic
content or causal use of the fitted direction.
