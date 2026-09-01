# What the action-matched GLAs add to the residual stream

## Bottom line

The 48 GLAs do **not** repeatedly add one small, monotonic “suppress the old
answer” vector. Their natural outputs show two quite different phases.

1. At the period after `Your answer was incorrect.` versus `Your answer was
   lost.`, GLA output vectors become strongly condition-specific early and
   remain so, but this difference is largely not aligned with A–D answer
   evidence.
2. At the final decision position, direct answer-aligned differences remain
   near zero through block 47. A sharp, heterogeneous answer computation begins
   at blocks 49–50. Blocks 50–59 initially reduce W2 more than W1; blocks 61–63
   then pivot toward reducing W1 and restoring W2.

Across the complete GLA stack on the 273 W1 != W2 questions, the cumulative
raw-unembedding contrast is:

| Semantic candidate | Evaluation minus Matched Neutral [95% CI] |
|---|---:|
| Previous semantic answer W1 | **-0.226 [-0.363, -0.087]** |
| Fresh remapped winner W2 | +0.043 [-0.086, +0.173] |
| Mean of the other two options | **+0.092 [+0.014, +0.169]** |

Thus the final net GLA write contains reliable previous-answer demotion and a
smaller redistribution toward otherwise disfavored options. It does not show a
reliable net W2 boost. On the 227 W1 = W2 questions, the cumulative W1 contrast
is only -0.067 [-0.234, +0.105]. The reliable W1 demotion is therefore specific
to trials where the previous and freshly regenerated answers conflict.

This is a clearer computational story than “all GLAs are involved,” but it is
still not a compact circuit. The causal transplant says the evaluation-period
memory update carries most of the behavior; the natural-write analysis says
that this state is expressed as answer-specific residual writes primarily in
the last quarter of the network.

## What was measured

A GLA internally updates a recurrent matrix using its key, value, decay gate,
and write strength. It then queries that matrix, applies its output projection,
and adds the resulting vector to the residual stream. The previous causal
experiment transplanted the **memory-update ingredients**. This experiment
records the resulting **post-output-projection vector actually added to the
residual stream**.

All 500 questions were run under the exact action-matched contrast:

- Evaluation: `Your answer was incorrect. Choose the answer again.`
- Matched Neutral: `Your answer was lost. Choose the answer again.`

For every GLA block, outputs were captured at:

- the period closing the evaluation sentence; and
- the final position immediately before the model answers the repeated
  question.

The exact historical four-question SDPA cohorts were preserved separately for
the two conditions: 125 cohorts × 2 conditions = 250 complete model forwards.
Natural A–D logits reproduce the trusted factorial bit-for-bit (maximum error
0.0).

“Direct A–D write” means the dot product between the GLA output and the
canonical A, B, C, and D unembedding rows, centered across those four letters.
Because the unembedding is linear, these raw contributions can be summed across
GLAs. They are **not final logits**: the model subsequently applies other
components and a final RMS normalization.

## 1. The evaluation sentence produces an early, high-dimensional difference

At the evaluation-closing period, Evaluation and Neutral GLA outputs diverge
far earlier and more strongly than they do at the final decision:

| GLA block | Period difference norm | Period cosine | Final-position difference norm | Final-position cosine |
|---:|---:|---:|---:|---:|
| 17 | 3.790 | 0.650 | 0.603 | 0.989 |
| 33 | 13.834 | 0.368 | 6.278 | 0.844 |
| 49 | 16.433 | 0.487 | 9.527 | 0.887 |
| 59 | 25.801 | -0.085 | 8.523 | 0.887 |
| 63 | 52.324 | 0.762 | 59.639 | 0.837 |

The tight intervals in the figure reflect a highly consistent prompt-level
contrast across 500 questions. This is mostly the common contextual difference
between `incorrect` and `lost`, not evidence that an answer-specific mechanism
is already present at block 17.

![GLA output geometry](/Users/christopherackerman/repos/secondchance_interp/figures/qwen36_action_matched_gla_output_geometry.png)

## 2. Direct answer computation is late and non-monotonic

The cumulative answer-aligned contrast is essentially zero through block 47.
It then changes sharply:

| Through GLA block | W1 | W2 | Other options |
|---:|---:|---:|---:|
| 49 | -0.005 | +0.023 | -0.009 |
| 50 | +0.057 | -0.128 | +0.035 |
| 51 | +0.101 | -0.232 | +0.066 |
| 53 | +0.025 | -0.306 | +0.140 |
| 59 | +0.016 | -0.269 | +0.126 |
| 61 | -0.147 | -0.219 | +0.183 |
| 62 | -0.177 | -0.173 | +0.175 |
| 63 | **-0.226** | +0.043 | **+0.092** |

So even after the condition state exists, the GLA residual writes do not simply
suppress W1. The middle of the late computation temporarily favors W1 relative
to W2; only blocks 61–63 produce the final net W1 demotion, with block 63 making
a large W2-restoring move. This helps explain why whole-block localization was
unsatisfying: the blocks participate in an interacting sequence with opposing
effects rather than contributing exchangeable copies of one command.

![GLA answer-aligned writes](/Users/christopherackerman/repos/secondchance_interp/figures/qwen36_action_matched_gla_answer_writes.png)

## 3. Vocabulary interpretation is weak

Raw-unembedding and JLens decoding were applied to each block’s mean
Evaluation-minus-Neutral output vector, globally and separately by W1 letter.
Some late final-position writes point toward families such as `Answer` around
block 49, `Respond` around block 51, and `Reflect`/`Reflection` at block 63.
Evaluation-period contrasts include `repeat`- and `question`-related families
at some late blocks. But many top directions are structural fragments or
uninterpretable token pieces, and the W1-letter-stratified readouts are nearly
identical rather than exposing a clean answer-identity feature.

The vocabulary lens therefore does **not** decode a stable “previous answer is
invalid” direction from individual GLA outputs. The complete readable-token
audit is in
[`vocabulary_readouts.json`](../../run/vocabulary_readouts.json).

## Mechanistic interpretation

The combined evidence now supports the following limited account:

1. The negative evaluation produces a large distributed recurrent-state
   difference while the feedback sentence is processed.
2. That state is largely non-answer-aligned in its immediate residual writes.
3. During the final answer computation, late GLAs transform candidate evidence
   through several opposing stages.
4. The net GLA contribution on conflict trials demotes the previous semantic
   answer and redistributes evidence toward the alternatives.

This rules out the simplest “gradually add the same W1-suppression vector”
story. It also shows more than the behavioral result alone: the instruction
difference is present in GLA outputs early, whereas its direct answer-targeted
expression is late, conflict-sensitive, and dynamically constructed.

What remains missing is an interpretable feature-level account of how the
early condition state interacts with W1 identity to control the late sequence.
Individual residual writes are descriptive; the earlier all-GLA state
transplant supplies the causal evidence.

## Compact artifacts

- [`summary.json`](summary.json): definitions, confidence intervals, and every
  layerwise aggregate.
- [`layerwise_metrics.csv`](layerwise_metrics.csv): tidy layerwise table.
- [`results.npz`](../../run/results.npz): compact question-level norms,
  canonical A–D writes, paired cosines, and aggregated mean vectors.
- [`vocabulary_readouts.json`](../../run/vocabulary_readouts.json): raw and
  JLens-readable top directions.
