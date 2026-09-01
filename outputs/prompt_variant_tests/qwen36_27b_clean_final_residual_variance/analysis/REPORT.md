# Clean final-residual answer-subspace variance

## Result

The historical layer-64 pattern substantially replicates under the cleaned
one-token `incorrect`/`lost` paradigm. In both datasets, the final residual
varies much less across questions in the three-dimensional A–D answer-contrast
subspace than it does in the remaining 5,117 dimensions. Game produces a
substantially stronger answer-specific contraction than Neutral.

On SimpleMC, the strongest historical claim replicates directly: Game A–D
contrast variance is **0.387 [0.348, 0.430]** times prompt-matched Baseline,
while variance in the orthogonal complement is **1.368 [1.283, 1.462]** times
Baseline. The complete normalized residual variance is **1.353 [1.269,
1.446]**. Thus the answer-choice geometry contracts while the rest of the
final representation expands.

On TriviaMC, Game A–D contrast variance is similarly **0.369 [0.354, 0.385]**
times Baseline. The orthogonal complement is **0.771 [0.748, 0.798]**, so this
dataset also has a broad Baseline-to-second-chance contraction. Crucially, the
answer contraction is still much stronger: its contraction relative to the
complement is **0.479 [0.457, 0.500]**. This repeats on the frozen discovery
half (**0.458 [0.426, 0.487]**) and confirmation half (**0.496 [0.465,
0.527]**).

## Exact estimates

All quantities are variances divided by the corresponding prompt-matched
Baseline variance. Confidence intervals are paired, Baseline-letter-stratified
bootstrap 95% intervals.

| Dataset | Condition | Full residual | A–D contrast | Orthogonal complement | A–D / complement |
|---|---:|---:|---:|---:|---:|
| SimpleMC | Game | 1.353 [1.269, 1.446] | 0.387 [0.348, 0.430] | 1.368 [1.283, 1.462] | 0.283 [0.253, 0.313] |
| SimpleMC | Neutral | 1.288 [1.216, 1.365] | 0.764 [0.702, 0.826] | 1.296 [1.224, 1.374] | 0.589 [0.533, 0.643] |
| TriviaMC | Game | 0.758 [0.734, 0.784] | 0.369 [0.354, 0.385] | 0.771 [0.748, 0.798] | 0.479 [0.457, 0.500] |
| TriviaMC | Neutral | 0.776 [0.752, 0.801] | 0.566 [0.544, 0.589] | 0.783 [0.759, 0.809] | 0.723 [0.697, 0.749] |

The most informative direct Game-versus-Neutral comparison reaches the same
conclusion without relying on Baseline. On SimpleMC, Game has only **0.507
[0.467, 0.552]** times Neutral's A–D contrast variance while its orthogonal
complement has **1.056 [1.033, 1.079]** times Neutral's variance. On TriviaMC,
the corresponding values are **0.652 [0.634, 0.670]** and **0.985 [0.970,
1.000]**. In both datasets, therefore, the distinctive Game–Neutral variance
difference is concentrated in answer-choice geometry rather than distributed
throughout the final state.

The supporting output-space check agrees. Aggregated A–D logit-contrast
variance in Game is 0.389 times Baseline on SimpleMC and 0.352 on TriviaMC;
realigning the remapped choices to their original semantic candidates gives
0.392 and 0.356. The effect is not an artifact of displayed A–D positions.

## What this says about “confusion”

This is strong evidence against a broad, undirected-confusion account of
Game's higher answer entropy. The model is deterministic here: “noise” would
have to mean that the `incorrect` prompt nonspecifically disrupts the final
representation. That is not the observed geometry.

- On SimpleMC, the non-answer residual varies *more* than Baseline while the
  answer-choice contrast collapses to 39% of Baseline.
- On TriviaMC, there is broad contraction relative to Baseline, so the strict
  “everything else is preserved” claim does not replicate. Nevertheless, the
  answer subspace contracts about twice as strongly as the rest, and Game and
  Neutral have nearly identical non-answer variance while differing sharply in
  answer variance.
- Neutral also contracts answer evidence, but Game does so substantially more.
  The generic second-presentation process therefore accounts for part of the
  flattening; the error-specific Game policy accounts for an additional,
  answer-focused contraction.

This activation result does not by itself prove which semantic answer is
suppressed or establish a causal strategy. Its role is narrower: it shows that
the higher Game entropy is implemented as a selective flattening of final
answer evidence, not as indiscriminate disorder throughout the residual. The
separate semantic-remapping, matching-history blockade, and policy×rank
interventions establish that the flattening is rank- and policy-structured.

## Relation to the historical result

The older SimpleMC prompt used `incorrect. Choose a different answer.` versus
`lost in transmission. Choose again.` At layer 64 it gave Game full/A–D/
complement ratios of 1.300/0.336/1.314. The cleaned prompt gives
1.353/0.387/1.368. The exact values move modestly, but the qualitative and
quantitative pattern is extremely close. Clean Neutral changes from
approximately 1.322/0.892/1.328 to 1.288/0.764/1.296, strengthening its generic
answer contraction while retaining the large Game–Neutral separation.

## Methods and controls

- Model: Qwen3.6-27B, frozen revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, BF16.
- Data: all 500 frozen remapped SimpleMC questions and all 500 frozen
  difficulty-filtered remapped TriviaMC questions.
- Conditions: prompt-matched Baseline; Game (`Your answer was incorrect.
  Choose the answer again.`); Neutral (`Your answer was lost. Choose the answer
  again.`).
- Representation: all 5,120 coordinates of the actual final RMSNorm output at
  the last prompt token, immediately before the output head. This is the real
  layer-64 endpoint, not a JLens reconstruction.
- Answer space: the rank-three span of centered mean unembedding rows for bare
  and leading-space A, B, C, and D tokens.
- Every one of the 3,000 prompts matched its trusted hash exactly. All residuals
  and logits were finite. Aggregated A–D logits reproduced the trusted runs
  with maximum absolute error 0.0 and 100% choice agreement.
- Evidence class: **activation/geometry**, with an output-logit consistency
  check. No activation was ablated or transplanted in this experiment.

## Artifacts

- [Frozen plan](../PLAN.md)
- [SimpleMC machine-readable summary](simplemc/summary.json)
- [TriviaMC machine-readable summary](triviamc/summary.json)
- [Canonical figure](../../../../figures/qwen36_clean_final_residual_variance.png)

