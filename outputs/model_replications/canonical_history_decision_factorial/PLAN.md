# Canonical non-remapped history-source test

## Objective

Test, on the ordinary non-remapped prompt, whether the second answer depends
on (a) candidate-matched reads from the first option lines, (b) the complete
state at the missing first-answer generation boundary, or (c) redundant use
of both routes.

This is a deliberately reduced reuse of established interventions. It does
not repeat the remapped token-type factorial, relay inventory, or projection
experiments.

## Frozen scope

- Models: pinned Qwen3.6-27B, Seed-OSS 36B, and Gemma 4 31B revisions.
- Datasets: the canonical 500-question SimpleMC and 500-question
  difficulty-filtered TriviaMC sets.
- Conditions: Game (`incorrect`) and Neutral (`lost`).
- Prompt: canonical non-remapped second presentation.
- Layers: every ordinary-attention layer in each architecture (Qwen's 16
  attention layers; all 64 Seed layers; all 60 Gemma layers). Qwen's 48 GLA
  layers are also edited in the first-boundary cells.
- Endpoints: old-W1 displayed choice, W1-minus-W2 logit margin, and
  within-question-centered W1 logit, on full and frozen discovery/confirmation
  splits.

## Five cells per condition

1. Natural.
2. Matching: deny every complete second option line reads from its identical
   complete first option line.
3. Cyclic wrong-line: the same receiver lines and edge count, with a cyclically
   nonmatching first option line.
4. First decision: deny every causally later attention query access to the
   final token of the first assistant-generation prefix. On Qwen, also remove
   that position's write to every GLA layer.
5. Joint: combine cells 2 and 4.

The first answer letter is absent from the stored conversation. Cell 4 tests
the complete residual state that would have generated that answer, not an
emitted letter token and not a fitted A–D coordinate.

## Interpretation

- Matching differs from cyclic wrong-line: the candidate-matched option-line
  route is causally specific on the ordinary prompt.
- First decision differs from natural: the missing-answer boundary is a causal
  source.
- First decision is null but joint exceeds matching: the boundary is a backup
  route revealed only when the line route is unavailable.
- First decision is null and joint equals matching: the boundary is not a
  necessary or redundant source for the measured effect.

Because semantic content, displayed letter, and line identity coincide on a
non-remapped prompt, this experiment cannot establish semantic mapping
invariance by itself. That interpretation must be combined with the prior
remapped experiments.

## Validity and operations

Every natural companion must reproduce the frozen canonical A–D logits
exactly; every output must be finite; every selected hook must fire. The full
design is 10 complete forwards per physical cohort and 9,000 forwards over all
six model/dataset runs. Benchmark every exact path before launch, checkpoint
after each cohort, retrieve compact outputs, and stop the GPU before analysis.

