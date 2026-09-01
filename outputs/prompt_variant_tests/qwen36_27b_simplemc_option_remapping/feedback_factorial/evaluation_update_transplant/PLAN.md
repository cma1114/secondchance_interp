# Action-matched evaluation-period GLA-update transplant

## Question

Does the recurrent GLA update written at the period after `Your answer was
incorrect.` carry a portable internal revision state, relative to the
token-aligned `Your answer was lost.` control?

Both conditions continue with exactly `Choose the answer again.` and the same
remapped repeated question. The experiment therefore isolates the evaluation
word while holding the action clause fixed.

## Intervention

For every GLA block, copy the source condition's key, value, decay gate `g`,
and write strength `beta` at the evaluation-closing period into the target
condition at that same token. These tensors determine the recurrent memory
update made by the token. The query, residual stream, all other tokens, and all
later computation remain native to the target prompt. In the corrected runner,
the source token's own GLA output is explicitly restored to its native target
value after the donor update is written; this prevents the transplanted update
from also broadcasting through that token's residual, ordinary-attention K/V,
MLP path, or short downstream causal-convolution taps.

Run both directions in the same forward pass:

1. Evaluation update into Matched Neutral.
2. Matched-Neutral update into Evaluation.

Each intervention is corrected against an untouched same-batch target control
and anchored to the previously validated exact natural logits.

## Frozen inference

- Primary subset: questions where first-presentation Baseline winner W1 differs
  from fresh remapped Baseline winner W2.
- Primary outcome: signed transfer of the W1-minus-W2 A-D logit margin toward
  the evaluation-only condition.
- Secondary outcomes: W1 selection, W2 selection, and A-D entropy.
- Confidence intervals: paired bootstrap, stratified by W1 answer letter.

The all-48-GLA transplant is a gate. Localize into prespecified eight-block
bands, then individual and leave-one-block-out tests, only if the
bidirectional-average primary confidence interval is positive and at least one
directional primary confidence interval is positive. This prevents spending on
localization if the update is not portable.

The discovery band screen reports every prespecified band. A band advances to
held-out confirmation only if its bidirectional-average W1-minus-W2 margin
transfer has a strictly positive 95% interval and at least one directional
interval is also strictly positive. Only bands that repeat this criterion on
the frozen confirmation split advance to block localization. No fixed number
of bands or blocks is selected in advance.

Within confirmed bands, report both each GLA block alone and all 48 GLA blocks
except that block. The first measures sufficiency; the second measures whether
removing that block diminishes the already-confirmed joint transplant. Because
the recurrent updates can interact nonlinearly, neither statistic is assumed
to add across blocks.

The historical non-output-preserved localization advanced bands 1–8 and 17–24,
with only 17–24 replicating. That result is superseded because the intervention
also changed the period token's own downstream-visible output.

In the corrected output-preserved run, only blocks 25–32 pass the frozen
discovery screen and replicate on confirmation, leaving GLA blocks 25, 26, 27,
29, 30, and 31 for individual and leave-one-out tests. All six are tested on
both splits; there is no further top-N selection.

## Result

The corrected all-GLA update is portable and causal, but is not most of the
task effect. Evaluation-to-Neutral transplantation transfers 0.097 logits
[0.056, 0.139] and Neutral-to-Evaluation transfers 0.091 [0.054, 0.127], versus
a natural 0.469-logit W1-minus-W2 margin gap. Blocks 25–32 carry 58.4%
[42.5, 82.5]% of this corrected all-GLA route. Blocks 26 and 27 have small
pooled block-alone effects, but no single block is independently sufficient on
both splits and no single deletion removes the joint transfer. This supports a
distributed, redundant recurrent-memory implementation. The canonical result
is the [output-preserved consolidated report](output_preserved/analysis/REPORT.md);
the older [non-output-preserved report](analysis/REPORT.md) is historical and
superseded.

## Compute accounting

The discovery gate contains 251 complete model forward passes. Every pass has
four rows: a Neutral target, an Evaluation target, and one untouched control
for each condition. Model loading and prompt preflight are additional fixed
costs. The exact path must be benchmarked before the complete gate is launched.
