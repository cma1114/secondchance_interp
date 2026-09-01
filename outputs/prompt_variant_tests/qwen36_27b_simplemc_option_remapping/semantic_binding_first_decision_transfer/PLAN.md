# Fixed-A first-decision semantic-state transfer

## Question

Does the complete residual state at the internal first-answer decision position
carry the semantic identity of that answer?

## Frozen cohort

Reuse the established semantic-binding cohort without modification:

- two first-presentation mappings for the same question;
- both make the model's Baseline decision literal `A`;
- `A` denotes semantic answer X in one mapping and Y in the other;
- the feedback and complete second presentation are identical;
- 64 discovery questions and 73 confirmation questions.

This controls the literal answer code. Any transferred suppression or
repetition target must follow semantic content rather than A/B/C/D identity.

## Intervention

Collect the complete residual vector at the first-decision position for all
four natural cells: Evaluation/X, Neutral/X, Evaluation/Y, and Neutral/Y.
At one post-block readout, exchange the X-history and Y-history vectors within
the same feedback condition:

- Evaluation/X receives Evaluation/Y and vice versa;
- Neutral/X receives Neutral/Y and vice versa.

The visible recipient prompt, feedback, and second presentation remain fixed.
Every patched pass retains the exact established four-row, batch-size-4 SDPA
kernel. In each pass the untested condition is explicitly identity-patched and
serves as a numerical control.

Discovery tests the frozen readout grid `1, 8, 16, 24, 32, 40, 48, 52, 56,
60, 63`. The readout maximizing the signed semantic-policy transfer
`Game transfer - Neutral transfer` is selected. Confirmation tests only that
readout.

## Outcomes

For an X-history receiving the Y state, semantic transfer in Game should make X
less suppressed relative to Y; for a Y-history receiving X, it should make Y
less suppressed relative to X. The symmetric mean of those two margin changes
is the **Game semantic-target transfer**. Positive values mean the later Game
suppression target moved toward the donor semantic answer.

The analogous Neutral quantity is expected to be negative if Neutral uses the
same state to repeat the prior semantic answer. The prespecified selection
score and primary cross-condition endpoint are Game minus Neutral.

Also report:

- answer-selection transfer between recipient and donor semantic answers;
- the complete centered A-D redistribution and entropy change;
- natural semantic-targeting replication;
- maximum identity-patch logit error and residual replacement error.

Discovery is used only to select the readout. The frozen confirmation estimate
is decisive. Confidence intervals use paired question bootstrap resampling.

## Interpretation

- Replicated positive Game transfer, especially with negative Neutral transfer,
  establishes that the first-decision residual carries causally usable semantic
  answer identity.
- A null means the semantic history is not portable in the complete residual at
  any tested single readout. It does not exclude distributed storage in earlier
  option positions, recurrent state, or a multi-position binding.
