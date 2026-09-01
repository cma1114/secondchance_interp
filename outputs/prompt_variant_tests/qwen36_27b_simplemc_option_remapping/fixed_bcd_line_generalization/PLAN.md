# Fixed-B/C/D selected-line semantic-transfer generalization

## Question

Does the fixed-A finding generalize when the first-pass selected option is
literal B, C, or D: does the selected first-presentation option line contain a
transferable semantic-answer state, and is that transfer mediated by the
semantically matching line in the repeated presentation?

## Why a new cohort screen is necessary

The identity/canonical-remapping pair yields only 5 fixed-B, 8 fixed-C, and 8
fixed-D candidates before exact-regime rechecking. That is inadequate. The
screen therefore begins with the existing balanced four-mapping set: identity,
the canonical remap, and two complementary remaps. For every question, each
semantic option occupies A, B, C, and D exactly once. It uses only the
first-decision output and cannot select on the later donor-transplant outcome.

A complete 24-mapping screen was initially proposed, then rejected after its
exact one-cohort benchmark: it would take about 5.9 hours and $9.6 merely to
construct the cohort. Further mappings will be added only if the balanced-four
screen does not supply adequate fixed-B/C/D pairs.

## Frozen restrictions and reasons

- **Literal selected positions:** B, C, and D. A is excluded because its direct
  transplant and serial mediation are already complete.
- **Pair definition:** within one question and one literal letter, both first
  mappings must naturally select that same displayed letter, but the content
  placed at that letter must differ.
- **Source span:** the complete selected option line, including label, answer
  text, and closing newline. Prior work did not localize the semantic payload
  to one token.
- **Exact token alignment:** donor and recipient selected lines must occupy the
  identical token indices and have identical token counts. This prevents a
  transplant from spilling into adjacent option lines or confusing semantic
  replacement with positional movement.
- **Persistent-state family:** ordinary-attention K/V only. In fixed A this
  family reproduced the semantic transfer, while GLA-only cache families did
  not.
- **Layers:** every ordinary-attention layer, 4, 8, ..., 64. The question is
  position generalization, so no cutoff is inherited from the A localization.
- **Conditions:** Game (`incorrect`) and Neutral (`lost`) are analyzed
  separately before their difference.
- **Splits:** the established frozen 251-question discovery and 249-question
  confirmation split.
- **Cohort cap:** at most 70 independently hash-sampled eligible pairs per
  letter and split before exact causal-batch rechecking. This limits cost
  without selecting on intervention outcomes.

## Stage 1: full selected-line transplant

For each X/Y pair, keep the recipient prompt and repeated presentation fixed.
Replace only the selected-line ordinary-attention K/V with the paired history
in which the same literal letter selected different semantic content. Measure
movement toward the donor semantic answer in centered logit margin and choice.

The primary gate for a letter is positive pooled Game/Neutral donor-semantic
margin transfer in discovery. Confirmation estimates are held out and reported
regardless of whether the discovery gate passes.

## Stage 2: serial mediation

Only letters passing the discovery gate receive the mediation factorial. Cross
the donor transplant with blocking reads from the transplanted source line by:

1. the repeated line containing the donor semantic answer;
2. a token-count-matched nonmatching repeated line.

The primary mediation endpoint is the reduction in donor-semantic transfer
under the matching blockade after subtracting the nonmatching control.

## Required controls

- exact prompt suffix identity after the first-decision boundary;
- all four natural first decisions equal the intended literal letter;
- cached-recipient output agrees with the same-batch natural output;
- benchmark-only complete-cache donor reproduction;
- source and receiver positions are nonempty and in the intended turns;
- exact forward-pass counts, resumable checkpoints, prompt hashes, runtime,
  and cost recorded before full launch.

## Completed outcome

The complete 24-mapping supplemental screen was required because the balanced
four-mapping screen supplied only five fixed-B pairs per split. It yielded 149
exact-valid discovery pairs and 148 held-out confirmation pairs across B, C,
and D. All benchmark and runtime controls passed on the validated replacement
host.

Stage 1 replicated donor-semantic transfer in Neutral at B, C, and D but not
in Game. Held-out Neutral transfer was +1.927 logits for B,
+1.475 for C, and +1.058 for D; Game estimates were +0.290, +0.009, and
-0.128. All letters passed the prespecified pooled discovery gate because the
Neutral effects were strong, so Stage 2 ran for B, C, and D.

Stage 2 did not support matching-specific mediation. Matching-line
blockade removed a modest component of Neutral transfer, but the nonmatching
control blockade removed more at every letter on both splits. The result is
therefore a policy-specific qualification: counterfactual selected-line
semantics transfer robustly under `lost`, while `incorrect` largely prevents
that substituted history from controlling the final answer. A subsequent
same-pipeline fixed-A calibration reproduced this same pattern at A and showed
that the old two-mapping fixed-A Game effect was design-contingent.
