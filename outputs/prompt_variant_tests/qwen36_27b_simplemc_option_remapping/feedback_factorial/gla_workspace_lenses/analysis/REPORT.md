# Workspace-lens decoding of action-matched GLA writes

## Bottom line

The RelP R-lens does **not** make the block-33 signal cleanly interpretable.
At block 33, both the matched J-lens and R-lens decode the mean
Evaluation-minus-Neutral GLA output at the final decision into a mixture of
generic web tokens (`COVID`, `https`, `GitHub`, `TikTok`) rather than a coherent
revision concept.

Both lenses become clearly task-relevant later, around blocks 42--47. They
independently recover a sequence resembling:

1. `incorrect` / `wrong` / `rejected` at block 42;
2. `replace` / `instead` / `alternatives` at block 43;
3. `exclude`-related tokens around block 45;
4. `again` / `override` / `second`, opposed to `previous` / `last`, at block 47.

This is useful corroboration that the condition-specific GLA computation
becomes vocabulary-aligned with evaluation and replacement semantics in the
40s. It does **not** explain the causal answer-specific source-trace effect that
begins around block 33, and R-lens is not materially better than J-lens for
that purpose.

## What was decoded

The frozen input is the completed 500-question action-matched SimpleMC run:

- Evaluation: `Your answer was incorrect. Choose the answer again.`
- Matched Neutral: `Your answer was lost. Choose the answer again.`
- exact raw ChatML, empty-thinking scaffolds, and batch-of-four SDPA.

For each of the 48 GLA blocks, the existing run already contained the mean
5,120-dimensional immediate GLA residual output at two positions:

- the period ending the evaluation clause;
- the final answer-decision position.

Each saved vector was transported through either the matched J-lens or matched
RelP R-lens, passed through Qwen's final RMSNorm, and unembedded into the full
vocabulary. The explorer contains Evaluation, Matched Neutral,
Evaluation-minus-Neutral, and W1-letter-stratified contrasts.

These scores are vocabulary-direction readouts. They are not probabilities,
not final logits from a natural forward pass, and not new causal effects. No
model behavior was regenerated and no new intervention forwards were run.

## Representative final-decision contrasts

The table lists the highest readable English tokens in the mean
Evaluation-minus-Neutral GLA output. Negative tokens point in the opposite
direction.

| Block | Lens | Positive direction | Negative direction |
|---:|---|---|---|
| 33 | J | COVID, https, GitHub, TikTok, likely | equipment, retarded, callback, modem |
| 33 | R | https, COVID, likely, GitHub, TikTok | equipment, retarded, reserve, norm |
| 42 | J | incorrect, wrong, correctness, rejected, failed | replacement, randomly, replace, random |
| 42 | R | incorrect, rejected, wrong, correctness, correct | randomly, random, replacement, replace |
| 43 | J | redesign, replace, instead, replacing, alternatives | no coherent task-related cluster |
| 43 | R | redesign, instead, replace, replacing, strategic | no coherent task-related cluster |
| 47 | J | again, override, instead, another, second | last, previous, directly, previously |
| 47 | R | again, override, second, fix | finally, directly, last, previous |

At block 29, R-lens does surface isolated relevant words such as `prior`,
`different`, `conditioned`, and `strategically`, but they appear among unrelated
tokens and do not form a stable readable state. Similar isolated prompt-related
words occur under J-lens. This is suggestive, not a successful early decode.

## Why this does not contradict the source trace

The earlier source-tracing experiment followed the downstream causal
consequence of removing the evaluation-period memory write from each GLA. Its
answer-aligned contrast begins around block 33. The present experiment decodes
a different object: the **whole mean immediate GLA output** at each block.

A causal effect can therefore begin at block 33 while the corresponding mean
component output remains poorly aligned with ordinary vocabulary directions.
The information may be encoded in a distributed or non-vocabulary-aligned
geometry and only become linearly readable as familiar revision words after
several more blocks of computation.

## Interpretation

R-lens was a sensible test because RelP changes the backward relevance geometry
while preserving the model's forward computation, and its authors report
cleaner early-layer readouts. On this signal, however, it does not supply the
missing mechanistic interpretation. The strongest defensible conclusion is:

> The causal evaluation-period GLA state begins affecting answer-specific
> computation around block 33, but neither matched workspace lens reveals a
> coherent vocabulary-level description of that early state. By blocks 42--47,
> both lenses read the resulting condition-specific GLA outputs as an
> incorrectness-to-replacement computation.

The full interactive comparison is
[workspace_lens_explorer.html](workspace_lens_explorer.html). The complete
displayed token lists and scores are in [readouts.json](readouts.json).

