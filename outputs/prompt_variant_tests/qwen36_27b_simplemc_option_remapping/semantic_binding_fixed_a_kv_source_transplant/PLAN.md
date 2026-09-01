# Fixed-A conventional-attention K/V source localization

## Status

Completed. The selected literal-`A` option line carried the dominant semantic
history effect in both frozen splits; the first-decision boundary and later
cue/header did not. Proceeding to layer-band localization is justified by the
prespecified decision rule.

- [Canonical report](analysis/REPORT.md)
- [Machine-readable summary](analysis/summary.json)
- [Presentation figure](../../../../figures/qwen36_fixed_a_kv_source_localization.png)

## Question

The complete first-boundary conventional-attention K/V cache transfers the
semantic identity of the first answer. Which token region stores the causal
information?

## Design

Reuse the frozen fixed-A crossover cohort. Both first presentations produce
literal `A`, but `A` names semantic answer X in one and Y in the other. The
second presentation is identical. Evaluation and Matched Neutral differ only
in `incorrect` versus `lost`.

At the first-decision boundary, transplant donor K/V entries across all 16
ordinary-attention layers for ten prespecified cells:

1. identity reinsertion;
2. selected `A` option line;
3. the remaining first-question tokens;
4. the entire first question;
5. the first-decision boundary;
6. the post-question cue/header excluding that boundary;
7. selected option plus boundary;
8. the entire informative prefix from question start through boundary; and
9. all conventional-attention K/V through the boundary; and
10. the complete causal cache, including GLA state, as an exact donor-reproduction
    implementation control.

The visible recipient prompt, GLA state, and all unselected K/V entries remain
unchanged. Run the historical batch-of-four SDPA regime, recheck the pre-feedback
fixed-A condition, and preserve the frozen 64-question discovery and 73-question
confirmation split.

## Primary endpoint

For an X-history recipient receiving Y-history K/V, measure the change in the
X-minus-Y final A-D logit margin; average it with the symmetric Y-history
recipient. Negative values mean movement toward the donor history's semantic
answer. Report Evaluation, Matched Neutral, and Evaluation-minus-Neutral.

Also report semantic-answer selection transfer, A-D entropy change, total answer
changes, source-position counts, cached-versus-unsplit validation, and exact
positive-control reproduction.

## Decision rule

Proceed to layer-band localization only if a restricted source region has the
same signed effect in discovery and confirmation, captures a meaningful portion
of the complete-K/V transfer, and is not explained by a generic entropy change.
