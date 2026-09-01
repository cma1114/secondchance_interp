# Question-specific JLens decoding of final-position GLA differences

## Bottom line

The final-position GLA output difference between Evaluation and Matched Neutral
does **not** cleanly decode as a question-specific `not W1` semantic vector.
Unlike the previous mean-vector analysis, this test preserves each question's
own difference vector and scores the actual tokens in that question's four
option texts before aligning results by W1, W2, and the remaining options.

Individual GLA directions alternate sharply across blocks. On the 273 W1 != W2
questions, block 63 gives the following centered maximum-token JLens scores:

| Candidate | Mean [95% CI] |
|---|---:|
| W1 | -0.087 [-0.235, +0.067] |
| W2 | +0.043 [-0.104, +0.184] |
| Other options | +0.022 [-0.066, +0.111] |
| W1 minus the mean alternatives | -0.116 [-0.314, +0.095] |

None of these magnitude contrasts excludes zero. Summing the block-specific
JLens-transported differences through block 63 is likewise inconclusive: W1
-0.074 [-0.237, +0.085], W2 -0.053 [-0.185, +0.077], and the other options
+0.063 [-0.015, +0.145]. These intervals are read directly from the current
machine-readable summary so the prose and artifact use the same bootstrap
realization.

One rank statistic looks stronger but is not robust to a reasonable scoring
choice. At block 63, W1 is the lowest of the four **maximum-token** scores on
52.7% of conflict questions versus 37.9% of no-conflict questions, a
conflict-minus-no-conflict difference of +14.9 points [+6.5, +23.7]. But when
each option is represented by the **mean** score of its substantive tokens,
the corresponding difference is only +2.6 points [-5.3, +10.6]. Because taking
the maximum over a variable number of option tokens is sensitive to
tokenization and option length, the maximum-token result is not sufficient
evidence for semantic W1 decoding.

The unrestricted per-question vocabulary audit is also weak: late positive and
negative tokens are usually structural fragments, generic words such as
`reflect` or `act`, multilingual fragments, or uninterpretable pieces. They do
not usually resemble the actual W1 content.

## What was measured

For each question `q` and GLA block `l`, the experiment captured the actual
post-output-projection vector written at the final decision position and formed

`d(q,l) = GLA_output_Evaluation(q,l) - GLA_output_Neutral(q,l)`.

Each `d(q,l)` was passed through block `l`'s learned JLens transport, final RMS
normalization, and the model unembedding. For every option, we retained:

- the maximum vocabulary score over substantive tokenizer tokens in its text;
- the mean vocabulary score over those tokens;
- an unnormalized linear score before final RMS normalization; and
- the same measures after cumulatively summing the layer-specific transported
  differences.

Only after those question-specific scores were calculated were questions
aligned by semantic identity and averaged. Thus “France,” “1812,” and “Michael
Jordan” were never averaged together as residual vectors.

All 500 questions used the exact historical batch-of-four SDPA cohorts and the
action-matched prompts:

- Evaluation: `Your answer was incorrect. Choose the answer again.`
- Matched Neutral: `Your answer was lost. Choose the answer again.`

Natural A-D logits reproduce the trusted run exactly (maximum absolute error
0.0).

![Question-specific GLA JLens results](/Users/christopherackerman/repos/secondchance_interp/figures/qwen36_action_matched_question_specific_gla_jlens.png)

## Panel interpretation

### A. Individual GLA difference

Each curve is the centered JLens score of the relevant option content in one
block's Evaluation-minus-Neutral output. The wide, rapidly alternating curves
show that no stable W1-negative direction is repeated by the GLAs.

### B. Cumulative transported differences

Each block's component difference is first transported into JLens output
coordinates; transported vectors are then cumulatively summed and decoded.
The early positive-W1/negative-W2 pattern changes sign later. By block 63, W1
and W2 are both slightly negative and the lower options slightly positive, but
the intervals overlap zero. This is consistent with broad redistribution, not
a clean semantic W1-only direction.

### C. Conflict-specific W1-last rate

This plots the W1-bottom-ranked rate on conflict questions minus the same rate
on no-conflict questions. The maximum-token definition gives a persistent
positive result, including +14.9 points at block 63. The mean-token definition
does not. The disagreement is the reason the apparent rank effect is treated
as a lexical/tokenization-sensitive clue rather than a robust semantic result.

## Interpretation

This result narrows the mechanistic account. The earlier causal transplant
still shows that evaluation-period GLA state carries behaviorally important
information, and the direct A-D analysis still shows a late net W1 demotion on
conflict trials. But that information is not exposed by JLens as a simple
question-specific semantic direction inside each final-position GLA output.

Possible reasons include:

1. W1 identity is encoded relationally, rather than as the vocabulary direction
   of the option text.
2. It becomes readable only in the complete residual state after combining GLA,
   ordinary-attention, and MLP writes.
3. JLens is reliable for residual states but less reliable when applied to an
   isolated component-output difference, which is off its training
   distribution.

Accordingly, the experiment does not support the strong claim that “the final
GLA difference vector explicitly contains `not France`.” Its main positive
result is a fragile, max-token W1-last signature that requires a better
semantic readout before being interpreted mechanistically.

## Artifacts

- [`summary.json`](summary.json): complete layerwise estimates and confidence
  intervals.
- [`layerwise_metrics.csv`](layerwise_metrics.csv): tidy aggregate table.
- [`results.npz`](../run/results.npz): question-level option scores, norms, and
  natural logits.
- [`sample_top_tokens.json`](../run/sample_top_tokens.json): frozen
  W1-letter-balanced per-question vocabulary audit.
