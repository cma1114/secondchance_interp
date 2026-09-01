# Identical-source selectedness attention-edge test

## Question

How does the model know that a particular semantic option was its first-pass
winner? Specifically, does the causal use of an original option line depend on
whether that option won, even when the option line and all preceding tokens are
held literally identical?

## Frozen cohort

Reuse the 77 W1=A pairs from the six-permutation screen: 41 discovery and 36
held-out confirmation questions. In each pair, semantic A remains at displayed
A. The identity first presentation chooses A, while a frozen permutation of
only B-D does not. The second presentation and feedback text are identical
within condition.

Recheck eligibility under the exact batch-four SDPA execution regime. Retain a
question only when both chosen-history rows select A and both unchosen-history
rows select the same non-A answer before feedback.

## Intervention

Use four physical rows per question:

1. chosen history, Game (`incorrect`);
2. chosen history, Neutral (`lost`);
3. unchosen history, Game;
4. unchosen history, Neutral.

In ordinary-attention blocks 4-48, block every token of the repeated semantic-A
option line from reading every token of one original option line. Test the
original semantic-A line first, then semantic B, C, and D separately as source
controls. All other queries, sources, modules, and layers remain untouched.

Before accepting any result, require:

- byte/token identity through the original A line;
- identical original-A token positions;
- bit-exact ordinary-attention K/V for the original A line across chosen and
  unchosen histories at all 16 conventional-attention layers;
- exact first-pass chosen/unchosen status in both condition rows;
- finite natural and intervention logits.

## Primary endpoint

For semantic A in the repeated presentation, compute the lesion effect
(intervention minus natural), then compare that effect between histories where
A won and histories where the identical A source did not win.

The policy-binding prediction is:

- Game: blocking the A relay restores more A evidence when A had won;
- Neutral: blocking the A relay removes more A evidence when A had won;
- therefore Game minus Neutral is positive.

Require these directions in both splits and require the held-out
Game-minus-Neutral 95% paired-bootstrap interval to exclude zero. Report the
same contrast after subtracting the mean B/C/D-source effect.

Only if this frozen gate passes may the next stage localize selectedness by
transplanting the post-A B-D comparison suffix. Do not launch that stage after
a null or ambiguous prerequisite.

## Result

Completed on 2026-08-19 with 36/41 exact-eligible discovery pairs and 33/36
exact-eligible confirmation pairs. Prefix identity and original-A K/V equality
both passed exactly (maximum error 0.0).

The selectedness interaction was directionally consistent but did not clear
the frozen gate: Game-minus-Neutral was +0.300 [+0.040, +0.552] logits in
discovery and +0.193 [-0.020, +0.414] in confirmation. The underlying absolute
effect was a large pro-A original→repeated relay in Neutral whether or not A
had won, and approximately no such relay in Game. Neutral's extra dependence
on A having won was modest (-0.257 logits discovery; -0.160 confirmation),
whereas Game's was near zero.

The result therefore establishes a strong policy difference in how original
option evidence is reinstated, but not a statistically conclusive winner-tag
on the identical source. Per the frozen rule, no suffix-localization stage was
launched. See `analysis/REPORT.md` and
`../../../../../figures/qwen36_w1_identical_source_selectedness_edge.png`.
