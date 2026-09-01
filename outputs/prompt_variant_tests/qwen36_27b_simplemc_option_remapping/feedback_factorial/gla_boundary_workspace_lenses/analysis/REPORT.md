# Complete residual states before and after every GLA write

## Bottom line

This is substantially more informative than applying a vocabulary lens to the
isolated GLA write.

The complete final-position residual already contains a coherent
Evaluation-versus-Neutral instruction state before the individual GLA writes:

- R-lens begins surfacing `different` around blocks 18--19 and
  `incorrect`/failure-related tokens around blocks 21--23.
- By blocks 25--29, both J-lens and R-lens strongly decode
  `incorrect`, `wrong`, `failure`, and related tokens.
- Around blocks 31--41, the contrast evolves toward
  `previous`, `another`, `again`, `try`, and `retry`.

The important qualification is that a single GLA usually changes this full
state only slightly. At block 33, the complete state is readable as
`trying`/`again`/`another`/`previous`, but the contextual change produced by
GLA 33 remains uninterpretable. Its immediate effect on the conflict-trial
W1-minus-W2 lens margin is essentially zero under J-lens and points in the
opposite direction under R-lens.

A clearer within-GLA sequence appears later:

- GLA 42: `incorrect`/`rejected`/`wrong`, opposed to
  `replacement`/`random`;
- GLA 43: `replace`/`instead`/`alternatives`;
- GLA 47: `again`/`override`/`second`, opposed to
  `previous`/`last`.

Those readable updates still do not immediately suppress W1. Their
condition-specific W1-minus-W2 margin effects are positive, not negative.
The first clearly behavior-aligned GLA margin movements in both lenses occur
at blocks 49 and 53. Thus the best supported staged account is:

1. a distributed instruction/evaluation state becomes readable early;
2. middle/late GLAs explicitly transform it through retry/replacement
   semantics;
3. answer-candidate redistribution is expressed later and non-monotonically.

This is a better cognitive description than “the GLAs add noise,” but it still
does not identify one localized `not W1` feature or single controlling module.

## Exact experiment

The collection used all 500 frozen SimpleMC remapping questions under the
action-matched contrast:

- Evaluation: `Your answer was incorrect. Choose the answer again.`
- Matched Neutral: `Your answer was lost. Choose the answer again.`

Prompts used exact raw ChatML, empty-thinking scaffolds, historical
batch-of-four cohorts, and SDPA. Natural A-D logits reproduced the trusted
factorial run bit-for-bit: maximum absolute error **0.0**.

At the final answer-decision position of every GLA block, the collector saved:

1. the complete residual immediately before the GLA;
2. the complete residual immediately after the GLA residual addition;
3. the complete residual after the following MLP.

Each question/state was separately transported through the matched J-lens and
R-lens, final-RMS-normalized, and unembedded. Only then were scores averaged.
All condition and boundary contrasts therefore subtract complete-state lens
scores; the isolated GLA output or difference vector is never lensed.

The post-MLP state is the workspace lens's native post-block boundary. The
before-GLA and after-GLA states are within-block diagnostic uses of the same
source-layer lens, so their agreement across J- and R-lens is more important
than any isolated token.

## What the complete state says

Representative positive tokens in the complete
Evaluation-minus-Neutral residual are:

| GLA block | Before GLA | After GLA | Interpretation |
|---:|---|---|---|
| 19 | R: `different`, `differently`, `alternative` | similar | Early inferred revision state; J-lens is less clean |
| 25 | J/R: `incorrect`, `failure`, `wrong`, `correct` | similar | Both lenses now decode the evaluation |
| 29 | J/R: `incorrect`, `failure`, `failed` | similar | Strong evaluation state already exists before GLA 29 |
| 31 | `previous`, `another`, `failure`, `again` | similar | Evaluation becomes historical-answer/retry semantics |
| 33 | `trying`, `again`, `another`, `previous` | similar | Readable full state, opaque GLA-33 increment |
| 37 | `again`, `another`, `retry` | similar | Retry state persists |
| 41 | `try`, `retry`, `failing`, `again` | similar | Explicit reattempt semantics |
| 43 | `try`, `failed`, `reject` | similar | Failure/rejection framing |
| 50--51 | `another`, `again`, `second` | similar | Alternative-answer framing |
| 57--59 | `again`, `correct`, `correction`, `retry` | similar | Late correction framing |

The near identity of the before and after columns is substantive: these
concepts reside in the accumulated residual state. They are not freshly
written in full by each GLA.

## What individual GLAs change in context

The contextual change is calculated as:

`(Evaluation after GLA - Evaluation before GLA) -
 (Neutral after GLA - Neutral before GLA)`

but only after the four complete residual states have been separately lensed.
This is not another lens applied to a raw difference vector.

| GLA block | J-lens contextual change | R-lens contextual change |
|---:|---|---|
| 33 | generic web/noise tokens | generic web/noise tokens |
| 39 | `answer`, `answers`, `answered` | `answer`, `answers`, `answered` |
| 42 | `incorrect`, `rejected`, `wrong`; opposite `replacement`, `random` | same qualitative result |
| 43 | `replace`, `instead`, `alternatives` | `replace`, `replaced`, `alternatives`, `instead` |
| 47 | `again`, `override`, `another`; opposite `previous`, `last` | `again`, `override`, `second`; opposite `previous`, `last` |

The agreement of the two independently trained workspace lenses at blocks
42--47 makes this sequence much more credible than the isolated block-33 token
lists.

## Relation to answer targeting

On the canonical 273 W1 != W2 conflict questions, the table below reports the
condition-specific effect of each GLA on the centered W1-minus-W2 lens margin:

`[(Evaluation after - before) - (Neutral after - before)]`.

Negative values move Evaluation relatively away from the previous semantic
answer W1 and toward the fresh remapped winner W2.

| GLA block | J-lens effect, 95% CI | R-lens effect, 95% CI |
|---:|---:|---:|
| 33 | -0.002 [-0.012, +0.008] | +0.057 [+0.049, +0.064] |
| 42 | +0.017 [+0.011, +0.023] | +0.025 [+0.018, +0.032] |
| 43 | +0.006 [+0.000, +0.011] | +0.017 [+0.010, +0.024] |
| 47 | +0.033 [+0.027, +0.040] | +0.091 [+0.079, +0.102] |
| 49 | -0.062 [-0.073, -0.051] | -0.059 [-0.071, -0.046] |
| 53 | -0.065 [-0.095, -0.037] | -0.053 [-0.086, -0.023] |
| 61 | -0.071 [-0.148, +0.008] | -0.085 [-0.165, -0.003] |
| 63 | -0.115 [-0.156, -0.073] | -0.115 [-0.156, -0.074] |

Confidence intervals are paired question bootstraps. The key point is that
the most semantically readable revision writes at blocks 42--47 are not the
same steps that immediately demote W1. Behavior-aligned candidate movement
appears later, especially at blocks 49 and 53, and continues to oscillate.

## Systematic semantic clustering of the vocabulary tails

The individual top-six token snapshots are easy to overread, especially when
one concept appears through many tokenizer variants. A local follow-up
therefore clustered the saved JLens vocabulary tails for the exact GLA
contextual contrast

`(Evaluation after - before) - (Neutral after - before)`.

No model inference was rerun. The candidate pool was the union of the saved
top and bottom 24 tokens at every GLA block. Clean alphabetic token renderings
were embedded with `sentence-transformers/all-MiniLM-L6-v2` and grouped with
average-linkage cosine clustering (distance threshold 0.25). Human-readable
families are explicitly anchor-labelled unions of those clusters; the full
unsupervised cluster catalog is retained for audit.

Two safeguards matter:

1. At a block, a family's score is the **single strongest member token**, not
   the sum of `fail`, `failed`, `failure`, and so forth. Morphological richness
   therefore cannot inflate the result.
2. The analysis is repeated using the top/bottom 6, 12, and 24 tokens. Absence
   from one of these truncated lists is recorded as missing, not as a zero
   semantic score.

The resulting [semantic-cluster heatmap](../../../../../../figures/qwen36_jlens_gla_semantic_clusters.png)
finds several isolated, interpretable updates:

| Block | Strongest stable semantic changes in Evaluation relative to Neutral |
|---:|---|
| 39 | adds answer/response (`answer`, +0.313) |
| 42 | adds incorrectness (`incorrect`, +0.438) while removing replacement (`replacement`, -0.570) |
| 43 | adds replacement/alternative (`replace`, +0.297; `instead`, +0.279) |
| 47 | adds retry/alternative (`again`, +0.516; `another`, +0.434) while removing prior-answer language (`last`, -0.469) |
| 49 | adds novelty and failure (`new`, +0.691; `failure`, +0.645) |
| 50 | adds another-answer language (`another`, +0.746) while removing failure and retry language (`unsuccessful`, -0.691; `retries`, -0.629) |
| 51 | adds response language (`Respond`, +0.668) |

All entries except the rank-10/11 `another`/`instead` terms at block 47 are
already visible within the top/bottom six. Every listed value is unchanged
when the clustering distance threshold is varied from 0.20 to 0.30.

The block-49 to block-50 sign reversal is robust at the level of this readout:
GLA 49 makes the Evaluation-specific incremental write more failure-like,
while the next GLA removes failure/retry-like content and adds `another`.
However, the heatmap does **not** show a continuous semantic trajectory across
the intervening GLAs. Most cells are missing or unrelated, and the listed
events were selected partly because they are interpretable in retrospect.

The defensible conclusion is therefore narrow: these particular GLAs have
stable, condition-specific vocabulary-lens readouts that resemble
incorrectness, replacement, retry, prior-answer, failure, or alternative
semantics. The cutoff and clustering checks establish the stability of those
individual readouts; they do not establish that the readouts form one circuit
or coherent staged computation. Nor do they show that the English-labelled
features are causal variables used by the model.

Reproducible outputs are in the
[semantic-clustering directory](../semantic_clustering/):
`cluster_catalog.csv` contains every unsupervised cluster,
`block_family_scores.csv` contains the signed blockwise scores at all three
cutoffs, `threshold_sensitivity.csv` contains the clustering-threshold check,
and `summary.json` records the complete family definitions.

## Interpretation

This result explains why decoding the isolated write was disappointing. The
instruction is already represented in the full residual, while most individual
GLAs make small context-dependent updates that do not need to resemble a
standalone English direction. Using workspace lenses on their intended object
exposes scattered readable states that are compatible with, but do not
establish, a broad progression:

> The complete residual distinguishes “incorrect” from “lost” by the 20s;
> several later readouts resemble retry or replacement language; and
> answer-candidate redistribution appears later still. The present evidence
> does not show that these observations are successive stages of one process.

What remains unresolved is how this accumulated instruction state is combined
with question-specific W1 identity. The present experiment makes some
condition-specific internal states more readable, but it does not establish an
instruction-processing mechanism or decode the matching operation that selects
which semantic answer to demote.

Use the [interactive boundary explorer](workspace_lens_boundary_explorer.html)
to inspect every GLA block, both lenses, all absolute states, and all contextual
changes. Complete displayed token lists are in
[readouts.json](../run/readouts.json), and question-level A-D lens scores are in
[results.npz](../run/results.npz). The reproducible numerical summary is
[summary.json](summary.json).
