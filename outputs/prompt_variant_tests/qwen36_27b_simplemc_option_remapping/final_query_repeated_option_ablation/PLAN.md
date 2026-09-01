# Final decision to repeated-option causal edge test

## Question

The previous experiment established that the repeated W1 option line causally
reads the first-presentation W1 option line. Does the final decision query then
read that repeated option line directly?

## Intervention

- Preserve the canonical action-matched remapped SimpleMC prompts, trusted
  natural outputs, batch-of-four SDPA cohorts, and 251/249 split.
- At only the final pre-answer query, block ordinary-attention edges to the
  complete second-presentation option line containing semantic W1.
- Apply the mask across all 16 ordinary-attention blocks 4--64 so the test does
  not assume a relay layer.
- Separately repeat the intervention for each of the other three second-
  presentation option lines. The primary source-specific contrast is the W1-
  line effect minus the per-question mean of these three controls.
- Report conflict and no-conflict trials, Game and Neutral, and discovery and
  held-out confirmation separately.

## Prediction

If the final decision directly reads the repeated W1 line, blocking that edge
should raise W1 choice and W1--W2 margin in Game, lower W1 choice in Neutral,
and reduce the Game--Neutral W1-avoidance gap. If the test is null, the repeated
line's confirmed causal effect must reach the answer through another query or
state before the final decision.

## Forward count

For each four-question cohort and each condition: one exact natural forward and
four source-line interventions (W1 plus three controls). Total: 10 complete
model forwards per cohort, 1,250 over 500 questions.

## Completion and result

Completed on all 500 questions. The final decision does directly read the
repeated W1 option line, but the read is pro-W1 rather than suppressive. On
held-out conflict trials, blocking W1's line relative to the mean of the three
other line controls changed the W1--W2 margin by -0.135 logits in Game and
-0.320 logits in Neutral. The Game-minus-Neutral difference was +0.186 logits,
closely matching discovery (+0.169). The corresponding held-out reduction in
the discrete W1-avoidance gap was 8.3 percentage points.

Therefore this experiment does not validate a direct final-query suppression
edge. It identifies a final pro-W1 reinstatement/readout route that is much
stronger in Neutral. The Game-specific consequence of the earlier validated
semantic receiver path must already have entered intermediate downstream
states before the final query.

Natural choices matched the trusted run on 988/1000 condition-question outputs.
The 0.125-logit maximum discrepancy arose under a different NVIDIA-driver
regime; all causal contrasts use the paired natural companion from the same
batch.
