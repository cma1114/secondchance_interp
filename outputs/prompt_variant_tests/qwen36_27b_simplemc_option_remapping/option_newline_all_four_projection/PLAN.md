# All-four option-newline candidate-value projection

## Question

Does the shared candidate-value direction decoded at the four option-closing
newline states form a causally used relative scoring representation, and does
removing it differentially alter Game revision?

## Frozen sample and execution

Use all 500 canonical remapped SimpleMC questions and the existing frozen
251/249 discovery/confirmation split. Preserve the historical physical batches
of four, batch size 4, SDPA implementation, raw Qwen ChatML, empty historical
assistant response, canonical remapped second presentation, and action-matched
`incorrect` versus `lost` prompts. Run both splits in one checkpointed pass and
analyze them separately; no result-dependent layer or cohort selection is
allowed.

## Intervention

At every first-presentation option-closing newline (A, B, C, and D), remove the
candidate-value coordinate from the ordinary-attention K/V carrier states whose
inputs fall in the readout 33--56 band.
The coordinate is freshly fit in the RMS-normalized residual geometry consumed
by Qwen's next block. For RMS-normalized option residual `z` and unit
normalized-space probe direction `u` (the fitted weight divided by feature
scale), apply the literal one-dimensional orthogonal projection

`z' = z - (z · u)u`,

then restore the original residual RMS. This
scale-invariant formulation is required because Qwen's full-prompt historical
token buffers can have arbitrary raw scale while the next block consumes their
RMS-normalized direction. The held-out probe must be refit and validated in
this exact geometry before the causal benchmark.

Concretely, for ordinary-attention blocks 36, 40, 44, 48, 52, and 56 (input
readouts 35, 39, 43, 47, 51, and 55), pass `z'` through the model's own block
input normalization and K/V projections and replace the four corresponding
source-token K/V entries. This is the carrier previously shown to transmit the
first-presentation option information. It deliberately does not claim to edit
the recurrent GLA memory; the infeasible token-by-token benchmark for a literal
whole-residual edit was stopped before full launch.

This sets every option's signed activation along the fitted value direction to
zero at every tested readout while preserving the orthogonal residual. It does
not force the probe's letter-specific affine offset to zero, replace the
complete option state, or directly edit final answer logits.

Compare only:

- `natural`: observation hooks with no residual change;
- `project_all`: project the fitted score out at all four option newlines.

The already validated zero-dose hook control need not be repeated on the full
sample. One exact complete-cohort benchmark must verify prompt hashes, natural
A--D logits, all four newline anchors, post-projection scores, residual dose,
forward count, runtime, and cost before launch.

## Prespecified analyses

Report discovery and confirmation separately, with paired question-bootstrap
95% confidence intervals. For all questions and separately for W1 != W2
conflict and W1 = W2 no-conflict trials, report:

- W1 and W2 choice rates and switching away from W1;
- W1-minus-W2 margin where defined;
- centered W1 evidence;
- A--D entropy and spread;
- any final-answer change;
- Game and Neutral effects separately and their difference-in-differences;
- pre/post probe scores and residual L2 dose at each readout.

The primary held-out endpoint is the Game-minus-Neutral interaction in W1
choice on conflict trials. A generic candidate-scoring role predicts final
choice/evidence changes without necessarily predicting an interaction. A
revision-specific role predicts that removing all four scores preferentially
restores W1 or reduces switching in Game.

## Output discipline

Retrieve only compact results, stop and retain the Vast host immediately after
completion, produce one canonical PNG, update the canonical remapping report
and root README, and record actual cost in the Vast ledger.

## Outcome

**Superseded/invalid.** Code review found that the executed projection omitted
the displayed-letter centering used to define the probe score and that the run
lacked an identity K/V-replacement control. The result below is retained only
as an audit record; a corrected centered run is required.

Completed on all 500 questions. The projection changed roughly 8--14% of final
answers across frozen splits and conditions and consistently compressed the
A--D distribution, establishing a generic causal role in candidate scoring.
It did not produce a replicated Game-specific effect on W1. On held-out
conflict trials, W1 choice rose by 2.2 points in Game and 5.9 in Neutral;
the interaction was -3.7 [-10.3, +2.9] points. Discovery showed -1.5 points in
Game and 0.0 in Neutral. See `analysis/REPORT.md` for the canonical result.
