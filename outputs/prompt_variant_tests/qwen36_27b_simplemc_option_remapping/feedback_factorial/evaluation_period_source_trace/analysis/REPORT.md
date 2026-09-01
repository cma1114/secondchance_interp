# Exact source trace of the evaluation-closing GLA write

> **Historical non-output-preserved intervention; superseded for route-specific
> attribution.** This run allowed the source period's own residual output to
> change. Use the [corrected output-preserved report](../output_preserved_analysis/REPORT.md),
> which removes 30.0% [9.5%, 50.0%] of the behavioral gap rather than the
> historical 72.0% estimate.

## Bottom line

This experiment asks what the recurrent memory update made by the period after
`Your answer was incorrect.` (or `lost.`) contributes when later tokens query
each GLA. It does **not** apply another vocabulary lens. Each block is replayed
with only that period write removed, and the resulting residual-stream
difference is measured at every later token.

Natural A-D logits reproduce the trusted 500-question run exactly (maximum
absolute error 0.0). The primary subset contains
273 W1 != W2 questions.

The full causal deletion sets the period write to zero in all 48 GLAs at once.
On conflict trials it changes W1 selection by:

- Evaluation: +10.3 pp
  [+5.5,
  +15.0].
- Matched Neutral: -2.9 pp
  [-7.0,
  +0.7].

Consequently, the natural **18.3-point** Neutral-minus-Evaluation W1-selection
gap falls to **5.1
points**. The deletion removes
**13.2 points**
[7.0,
19.4], or
72.0%
[48.8,
90.9] of the
natural gap.
The W1-minus-W2 margin gap shrinks by
0.344 logits
[0.264,
0.423].
This replicates under the frozen split: the W1-selection gap reduction is
10.2 points
[2.2,
19.0] in discovery and
16.2 points
[7.4,
25.0] in confirmation.

The source trace shows where that causal effect is expressed. At the final
decision, the Evaluation-period write is read much more strongly than the
Neutral-period write by several middle/late GLAs, especially blocks 33, 47,
and 49. Its direct answer-aligned contribution begins separating around block
33. By the end, its Evaluation-minus-Neutral direct W1-versus-W2 contribution
is -0.0089
[-0.0134, -0.0041]: it favors W2
over W1. The raw direct contribution is small because it is measured before
downstream amplification; the separate global deletion above establishes the
large final causal effect.

The causal effect is not entropy-free. Removing the period write lowers
Evaluation entropy on conflict trials by
0.024 bits, so
the natural Evaluation write contributes some broad uncertainty as well as the
W1-versus-W2 redistribution.

The four panels separate retrieval strength, the final-decision answer effect
sourced by the period write, downstream token location, and the complete causal
deletion. See the definitions below before interpreting cumulative direct
writes as final logits.

[Canonical PNG](../../../../../../figures/qwen36_evaluation_period_source_trace.png)

## Definitions and limits

- **W1** is the semantic answer chosen on the original first presentation.
- **W2** is the semantic answer chosen by a fresh Baseline under the remapped
  second presentation. It is **not** the runner-up from the first presentation.
- **Other** in Panel B is the mean of the two remaining semantic candidates on
  W1 != W2 trials. Because all four direct A-D contributions are centered,
  `W1 + W2 + 2 * Other = 0`.
- **Panel B** is measured at the final decision, after the remapped second
  presentation has been processed. It therefore shows the later answer-specific
  effect causally sourced by the earlier period write; it does not show which
  answer identities were locally encoded at the moment the write was made.
- **Source trace** is natural GLA output minus a within-block replay with beta
  zeroed only at the evaluation-closing period. It is an exact deletion effect
  inside that recurrent block, including later interactions, but source traces
  from different blocks are not guaranteed to add causally through the whole
  nonlinear model.
- **Direct A-D contribution** unembeds the source-trace residual vector with the
  four canonical answer rows, aligns by semantic identity, and centers across
  options. It is not a final logit.
- **Global ablation** is a separate complete forward in which that period write
  is removed from all 48 GLAs simultaneously.

The block-33 onset in Panel B is an early component-level precursor. The
complete final-decision residual does not show a practically large net W1
demotion until readouts 52--54, followed by its largest step at Mixer 56. These
are different measurements: a small isolated causal trace can be present before
it dominates the net residual state.

Machine-readable intervals, including the frozen 251-question discovery and
249-question confirmation splits, are in [`summary.json`](summary.json).
