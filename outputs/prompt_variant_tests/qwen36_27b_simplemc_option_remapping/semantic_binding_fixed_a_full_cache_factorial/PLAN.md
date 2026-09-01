# Fixed-A complete causal-cache transplant and decomposition

## Status

Completed. The exact-regime screen retained 56 discovery and 63 confirmation
questions. Full-cache donor continuation reproduced with 0.0 maximum A--D
logit error. Conventional-attention K/V carried essentially all continuous
semantic-history transfer; GLA convolutional and recurrent states did not.

- [Canonical report](analysis/REPORT.md)
- [Machine-readable summary](analysis/summary.json)
- [Presentation figure](../../../../figures/qwen36_fixed_a_full_cache_factorial.png)

## Question

Which persistent internal state at the first-decision boundary carries the
semantic-history effect in the fixed-A paradigm?

Both histories predict literal `A` at the first decision, but `A` denotes
semantic answer X in one first presentation and Y in the other. Feedback and
the complete post-boundary token sequence are identical within condition.

## Complete causal state

For future-token computation, Qwen3.6-27B has three persistent cache families:

1. conventional-attention key/value tensors (16 blocks);
2. GLA causal-convolution states (48 blocks); and
3. GLA delta-rule recurrent matrices (48 blocks).

The prior experiment exchanged only family 3 and was incorrectly described as
a complete-state transplant.

## Positive control

Run each four-cell fixed-A cohort as a cached prefix ending at the internal
first-decision position, followed by the unchanged suffix. Exchange X and Y
batch rows in all three state families. Because the X and Y suffix tokens are
identical within Game and within Neutral, a complete-cache transplant must
reproduce the corresponding donor-history continuation. The runner aborts on
the first question if maximum A-D logit error exceeds `1e-4`.

## Factorial decomposition

Using the same prefix cache and suffix, run all eight combinations of the three
state families: identity, each family alone, each pair, and all three. Compare
every intervention with cached identity continuation. Report symmetric X↔Y
semantic-target transfer in Game and Neutral, Game-minus-Neutral transfer,
answer-selection transfer, and a three-factor Shapley allocation including
interactions.

## Frozen samples and validation

- Reuse the 64-question discovery and 73-question held-out confirmation sets.
- Preserve the exact fixed-A prompts, batch-of-four rows, model revision,
  raw-ChatML formatting, and SDPA configuration.
- Recheck the fixed-A eligibility condition under this experiment's exact
  four-cell execution regime. The historical cohort was selected from
  separately batched Baseline runs, and Qwen is numerically batch-sensitive.
  Screen out (and report) any question for which all four prefix decisions are
  not A before running interventions; this pre-feedback screen cannot select on
  an intervention outcome.
- Record the difference between cached identity continuation and the unsplit
  natural forward, but use only within-cached-regime causal contrasts.
- Discovery and confirmation are both descriptive; no layer or intervention is
  selected from discovery.
