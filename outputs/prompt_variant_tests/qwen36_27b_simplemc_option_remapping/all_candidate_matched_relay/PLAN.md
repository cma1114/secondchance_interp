# All-candidate matched semantic relay factorial

## Question

Does `incorrect` specifically alter retrieval of the first-pass winner, or does
it change semantic reinstatement for every candidate in proportion to that
candidate's first-pass evidence?

## Frozen design

- Canonical 500-question action-matched remapped SimpleMC paradigm.
- Historical batch-of-four SDPA execution, raw Qwen ChatML, pinned model
  revision, and frozen 251/249 discovery/confirmation split.
- Candidates are ranked W1--W4 from the original first-presentation A--D
  logits, not from second-presentation behavior.
- Causal matching-edge interventions use ordinary-attention blocks 4--48,
  matching the previously validated causal receiver pathway.

Per condition and four-question cohort, run:

1. natural, while collecting compact layerwise matched-edge attention mass,
   pre-gate context norm, mean output-gate strength, and projected write norm;
2. four individual matching lesions: original Wr → repeated Wr;
3. four cyclic nonmatching controls: original W(r+1) → repeated Wr;
4. one joint lesion blocking all four matching edges.

This is 10 complete forwards per condition and 20 per cohort. The exact runner
loop, including the extra SDPA calls used to recover observational attention
weights, must be benchmarked before full launch.

## Prespecified analyses

1. Matching specificity by W1--W4, condition, split, and conflict status:
   matching-lesion effect minus cyclic-control effect on each candidate's
   centered logit advantage and selection.
2. Graded versus categorical selectedness: regress the Game-minus-Neutral
   matching-specific lesion effect on continuous first-pass candidate evidence
   and a separate W1 indicator, with question-centered predictors and paired
   question bootstrap intervals. A held-out W1 coefficient above zero is the
   explicit-winner signature.
3. Joint mediation: determine whether blocking all four matching relays in
   Neutral reduces the natural Game--Neutral W1-choice and W1--W2-margin gaps,
   and compare the joint effect with the sum of individual lesions.
4. Mechanistic decomposition: compare layerwise attention mass, context norm,
   output-gate strength, and projected write norm between Game and Neutral for
   each first-pass rank. These are observational decompositions interpreted
   alongside, not instead of, the causal lesions.

## Interpretation gate

- A broad rank-general Game--Neutral difference supports general semantic
  reinstatement policy.
- A smooth relationship with first-pass evidence supports graded reuse without
  a categorical winner tag.
- The original linear-score W1 discontinuity was only a screening statistic.
  Because W1 is a nonlinear thresholded function of relative evidence, it
  cannot identify explicit winner binding without flexible score-and-gap and
  near-tie controls.

## Result

Completed on all 500 questions with exact natural-logit reproduction. The
original linear-score gate passed (+0.297 [+0.145, +0.447] logits in discovery
and +0.510 [+0.340, +0.675] in held-out confirmation), but that interpretation
is superseded by the frozen nonlinear audit. Flexible candidate-score and
best-competitor-gap controls, together with near-tie tests, do not establish a
separate categorical winner representation.

On conflict trials, jointly blocking all four semantic matching relays reduced
the Game--Neutral W1-choice gap from -15.3 to -1.5 points in discovery and from
-22.1 to a 0.0-point estimate in confirmation; the held-out gap-reduction
interval is +11.8 to +32.4 points. Held-out continuous W1--W2 margin
mediation was partial (0.314 logits, about 49% of the natural gap), and weaker
in discovery.

Canonical report: `analysis/REPORT.md`. Canonical figure:
`figures/qwen36_all_candidate_matched_relay.png` from the repository root.

## Complete absolute-attention trajectory correction

The first absolute-attention figure incorrectly stopped at block 48. That
cutoff was inherited from the causal intervention range, but it was not
justified for the distinct descriptive question of how natural matched-line
attention evolves through the complete network. Qwen3.6-27B has ordinary
attention at blocks 4, 8, ..., 64, so the incomplete figure must be replaced,
not relabeled.

The corrective measurement is intentionally narrow:

- same canonical 500 questions, mappings, historical batches of four, raw
  ChatML, and action-matched Game/Neutral prompts;
- both natural conditions, because their absolute trajectories and difference
  are the scientific object;
- all 16 ordinary-attention blocks 4--64, because a complete layerwise
  trajectory has no question-specific basis for omitting 52--64;
- all four first-pass ranks R1--R4, because the requested comparison is whether
  retrieval depends on which candidate ultimately won;
- the frozen 249-question confirmation split in the presentation figure, to
  preserve the established held-out denominator;
- natural forwards only: the existing causal factorial remains valid and is
  not rerun, because the correction concerns missing observational layers.

The exact path has two complete model forwards per four-question cohort, one
per condition, plus synthetic-value SDPA calls at every one of the 16 ordinary
attention blocks to recover attention weights. A complete cohort must be
benchmarked before launch; runtime and cost are extrapolated from that measured
path rather than from the nominal two forwards.

Validation requires: all 500 questions complete; literal original and repeated
option-line anchors for all four ranks; finite metrics at all 16 blocks; exact
prompt hashes; same-batch natural behavioral reproduction; and agreement of
the remeasured blocks 4--48 with the earlier natural companion before replacing
the canonical absolute-attention PNG and CSV.

### Correction result

Completed on all 500 questions and all 16 ordinary-attention blocks. The
replacement companion used identical question IDs, prompt hashes, and rank
definitions; the remeasured block-4--48 attention mass differed from the prior
A100 companion by 0.00054 on average (99th percentile 0.00377). The fresh-host
natural answer agreed with the prior run on 95.9% of condition-question cells;
late-layer conclusions were unchanged on the 231/249 held-out questions that
reproduced in both conditions.

The completed trajectory peaks around blocks 48--52 and then loses about five
percentage points of matching attention at block 56 across every rank and both
conditions. The previously omitted quarter also reveals rank-dependent
selection: R1 receives more matching attention than mean(R2--R4) from block 52
through block 64 in both conditions, with stronger Neutral than Game R1
selection through block 60. Both canonical figures, the CSV, the report, and
the root README now use the complete block-4--64 companion.
