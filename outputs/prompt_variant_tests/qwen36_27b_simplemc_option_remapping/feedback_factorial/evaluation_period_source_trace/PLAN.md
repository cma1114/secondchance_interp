# Exact source trace of the evaluation-closing GLA write

## Question

The historical causal transplant suggested that the GLA memory update at the
period after `Your answer was incorrect.` versus `Your answer was lost.`
transferred most of the W1-avoidance behavior, but also changed the source
token's own output. What does the isolated persistent recurrent update contribute
when later tokens query GLA memory, and does its contribution contain the
question-specific demotion of W1?

## Exact measurement

Use all 500 frozen SimpleMC remapping trials and the action-matched prompts:

- Evaluation: `Your answer was incorrect. Choose the answer again.`
- Matched Neutral: `Your answer was lost. Choose the answer again.`

Preserve raw ChatML, empty thinking scaffolds, SDPA, and the exact historical
batches of four.

For every one of the 48 GLA blocks, replay its complete gated-delta recurrence
with `beta=0` only at the evaluation-closing period. Return the natural output
to the model, but record natural minus replayed post-normalization,
post-output-projection output at every token from that period through the final
decision. This is the exact within-block deletion effect of that source write,
including later decay and delta-rule interactions. It is deliberately not a
JLens interpretation and is not assumed to be additive across blocks.

Measure:

1. the norm of that source contribution by later structural prompt region and
   at prespecified token anchors;
2. its direct canonical A-D contribution, aligned within question to W1, W2,
   and the other options;
3. Evaluation-minus-Neutral differences separately on W1 != W2 and W1 = W2
   questions.

## Causal companion

In a separate complete forward, set `beta=0` at the evaluation-closing period
in all 48 GLAs simultaneously. Report changes in W1 and W2 selection, W1 logit,
and A-D entropy for Evaluation and Matched Neutral. The already-confirmed
Evaluation-to-Neutral update transplant remains the sufficiency test; this
joint deletion is the corresponding necessity test. The corrected runner
restores the source period's own local GLA output after suppressing the memory
write, so this global arm measures persistent recurrent-memory necessity
without also broadcasting a changed source residual through ordinary
attention, MLPs, or the short causal-convolution path.

No layer is selected from the data. Report the complete 500-question result
and the frozen 251-question discovery and 249-question confirmation splits.
Use W1-letter-stratified paired bootstrap confidence intervals.

## Validation and operations

- Natural A-D logits must reproduce the trusted factorial exactly.
- Benchmark a complete historical cohort through both natural-with-replay and
  joint-ablation paths before extrapolating runtime and cost.
- Retrieve compact shards immediately. Stop but retain the Vast host after the
  complete approved audit-remediation suite is finished.
- Save one canonical PNG with visible confidence intervals and index the report
  and figure from the root README.

## Corrected result

The 500-question output-preserved run reproduces trusted natural logits exactly.
Removing the evaluation-period write from all 48 GLAs increases Game W1 choice
by 5.5 points [2.9, 8.4] and changes Matched Neutral by 0.0 [-2.6, 2.6]. It
therefore removes 5.5 points [1.5, 9.5], or 30.0% [9.5, 50.0%], of the natural
task gap and 0.097 logits [0.059, 0.135] of the W1-minus-W2 margin gap. The
margin effect is positive on both frozen splits. Final-decision retrieval is
largest at GLA blocks 49, 33, and 47, and the cumulative direct trace favors W2
over W1. There is no reliable entropy effect. The canonical report is
[output_preserved_analysis/REPORT.md](output_preserved_analysis/REPORT.md).
