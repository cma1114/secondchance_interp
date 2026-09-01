# Exact source trace of the evaluation-closing GLA write

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

- Evaluation: +5.5 pp
  [+2.9,
  +8.4].
- Matched Neutral: +0.0 pp
  [-2.6,
  +2.6].

Consequently, the natural **18.3-point** Neutral-minus-Evaluation W1-selection
gap falls to **12.8
points**. The deletion removes
**5.5 points**
[1.5,
9.5], or
30.0%
[9.5,
50.0] of the
natural gap.
The W1-minus-W2 margin gap shrinks by
0.097 logits
[0.059,
0.135].
This replicates under the frozen split: the W1-selection gap reduction is
4.4 points
[-0.7,
9.5] in discovery and
6.6 points
[0.7,
13.2] in confirmation.

The source trace shows where that causal effect is expressed. At the final
decision, the Evaluation-period write is read much more strongly than the
Neutral-period write in several GLAs. The three largest Evaluation-minus-Neutral
final-decision retrieval-norm differences are at blocks 49 (+1.133), 33 (+0.582), 47 (+0.427).
The complete cumulative answer-aligned trajectory is shown in Panel B rather
than assigning an onset by visual inspection. By the end, its
Evaluation-minus-Neutral direct W1-versus-W2 contribution
is -0.0089
[-0.0134, -0.0041]: it favors W2
over W1. The raw direct contribution is small because it is measured before
downstream amplification; the separate global deletion above establishes the
final causal effect.

The corrected route has no reliable entropy effect. Removing the period write
changes Evaluation entropy on conflict trials by
-0.001 bits
[-0.011,
+0.009]. The
output-preserved persistent-memory route is therefore more specifically tied to
answer redistribution than the historical broad intervention suggested.

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

Earlier complete-residual crossover work localizes decisive task-dependent
final-answer impact much later than many component-level source traces. These
are different measurements: a small isolated causal trace can be present before
it dominates the net residual state.

Machine-readable intervals, including the frozen 251-question discovery and
249-question confirmation splits, are in [`summary.json`](summary.json).
