# Action-ending-period source lesion

## Question

Does the period ending the shared action clause `Choose the answer again.` add
causal information beyond the state already created at the period ending
`Your answer was incorrect.` (or `lost.`)?

The preceding accumulated state is held fixed. This experiment removes only
the action-ending period token's own downstream routes:

1. **GLA write:** set the write-strength beta to zero only at that period in all
   48 GLA blocks. Every recurrent state accumulated before the period remains,
   and the source token's own local residual output is restored so the lesion
   isolates the persistent recurrent-memory route.
2. **Attention read:** prevent every later query from attending to that period
   in all 16 conventional-attention blocks. Every other attention edge remains.
3. **Joint:** apply both lesions simultaneously.

Each lesion is run separately in Evaluation (`Your answer was incorrect.`) and
Matched Neutral (`Your answer was lost.`). The visible action clause and source
token are identical across conditions.

## Primary test

On W1 != W2 conflict questions, estimate within each condition how each lesion
changes:

- selection of W1 and W2;
- the W1-minus-W2 logit margin;
- W1's centered A-D advantage;
- A-D entropy and spread;
- switching away from W1.

The action period adds a condition-specific revision effect only if removing
its routes changes Evaluation more than Matched Neutral. Similar effects in
both conditions indicate a generic action-boundary computation. Null effects
indicate that the relevant state was already established upstream.

The same endpoints will be reported on W1 = W2 questions, W1=A versus W1=B-D,
and the frozen 251-question discovery / 249-question confirmation split.

## Execution and validation

- Current action-matched, exact-remapped SimpleMC prompts only.
- Raw Qwen ChatML, thinking disabled with the established empty scaffold.
- Exact historical four-question cohorts and SDPA kernels.
- Eight complete model forwards per cohort: natural plus three lesions in each
  of two conditions (1,000 forwards total for 500 questions).
- Natural logits must match the trusted run exactly.
- Benchmark the complete eight-forward path before the full run.
- Stop, retain, and audit the Vast host immediately after retrieval.

## Corrected result

The historical ordinary-attention arm was invalid because assigning `-inf` to
a Boolean SDPA mask cast to `True` and left the source edge enabled. The fixed
runner asserts that every ordinary-attention and GLA hook fires, and the
canonical rerun additionally preserves the source token's own output.

The corrected attention route is small but real. In Neutral conflict trials,
blocking later reads changes W1-minus-W2 margin by -0.033 logits [-0.051,
-0.014], with discovery and confirmation both at -0.033. In Game no-conflict
trials it raises W1 centered advantage by +0.048 [+0.031, +0.066], with the
same sign on both splits. No pooled W1-selection effect excludes zero and the
joint lesion does not reproduce the main task gap. The action-ending period is
therefore a secondary causal source, not the main policy bottleneck.

The prior accumulated-state transplant is not evidence that this period adds
information; it replaced the entire pre-existing state plus the period update.
This source-specific lesion is the direct test of the period's incremental
contribution.
