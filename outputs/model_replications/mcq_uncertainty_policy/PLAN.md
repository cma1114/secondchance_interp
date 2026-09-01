# Recomputed MCQ uncertainty at the second decision

## Status: complete

All four prespecified model-by-dataset cells completed on the frozen
confirmation splits, with every post-block residual L1-L64, both Game and
Neutral, all seven causal scenarios, and 10,000 paired bootstrap draws. The
final artifacts pass finiteness, duplicate-row, identity-layer, projection
ablation, signed-dose, and orthogonal-control gates. Superseded outputs that
failed a gate are preserved but excluded.

The frozen 1P direction generalizes strongly across datasets and remains an
entropy predictor after controlling the displayed old-winner letter. Its
natural 2P projection has a structured layerwise trajectory in all four cells.
Bidirectional steering, relative to the equal-dose orthogonal control,
causally changes the final centered rank vector: in the middle-to-late layers,
positive high-uncertainty steering generally lowers old R1 and raises lower
ranks. This effect is not specific to Game. It is approximately shared across
Game and Neutral in Seed and is stronger in Neutral in Qwen. Discrete choice
effects at the frozen dose are small. The result supports a general
uncertainty-like candidate-ranking control re-instantiated at 2P, but not a
condition-specific uncertainty readout that explains preferential Game
switching.

Canonical outputs: [report](analysis/REPORT.md),
[machine-readable summary](analysis/summary.json), and
[top-level figure](../../../figures/model_replications/mcq_uncertainty_policy.png).

## Question

Does a general multiple-choice uncertainty direction, learned independently at
the first-presentation decision position, appear again at the
second-presentation decision position and causally affect the candidate-ranking
behavior produced there?

This is a direct port of the `entropy_probes` identify -> project -> ablate /
steer workflow. It is not a test of whether the first-decision residual is
remembered. The complete first-decision source intervention already found that
the first-decision position is not a primary causal source for the canonical
second answer. Here the hypothesized uncertainty signal is recomputed at 2P.

## Frozen scope

- Models: pinned Qwen3.6-27B and Seed-OSS 36B revisions already used by the
  canonical mechanistic runs.
- Datasets: all 500 canonical SimpleMC questions and all 500 canonical
  difficulty-filtered TriviaMC questions.
- Conditions: canonical non-remapped Game (`incorrect`) and Neutral (`lost`),
  analyzed separately. No Game/Neutral state crossover is performed.
- Positions:
  - 1P: the final token of the first assistant answer-generation prefix, whose
    logits would generate the omitted first answer.
  - 2P: the final prompt token immediately before the second assistant answer.
- Layers: every post-block residual, L1-L64, for both descriptive projection
  and causal intervention. No prior layer cutoff is inherited.
- Splits: the frozen SimpleMC and TriviaMC discovery/confirmation splits.
  Directions are fit on discovery only; causal conclusions use confirmation.

## Stage 1: identify the 1P uncertainty direction

For every layer, collect the 1P decision residual and its exact A-D logits from
the standalone canonical Baseline forward. The Game and Neutral 2P prompts
must be token-identical to the standalone Baseline through the first assistant
turn opener. After that opener, Seed's official template differs by design:
the live Baseline includes its zero-budget thinking wrapper, whereas the 2P
prompt closes the deliberately empty first assistant turn. Using the
standalone forward measures the actual answer-generation position on which the
general uncertainty direction is defined and avoids sequence-shape-dependent
BF16 differences from extracting it inside a longer prompt.

Primary direction:

- Target: entropy of the normalized four-way A-D distribution.
- Estimator: unit mean difference between the highest and lowest discovery
  quartiles, matching the `entropy_probes` direction whose logit lens yields
  uncertainty-like tokens such as `none` and `neither`.
- Sign: positive means greater uncertainty.

Descriptive validation only:

- Discovery-fitted ridge prediction of entropy and top-two logit gap.
- Held-out within-dataset and cross-dataset prediction.
- Standard logit-lens top positive and negative tokens for each frozen
  mean-difference direction.
- Cosine and direct projection onto the centered A-D output-token subspace.

The primary causal direction remains the prespecified mean-difference entropy
direction regardless of the descriptive ridge result.

## Stage 2: measure the frozen direction at 2P

For each question, condition, and layer, record the scalar projection of the
natural 2P decision residual onto the frozen 1P entropy direction. Report the
raw projection and the projection standardized using the 1P discovery mean and
standard deviation.

This stage does not define or retrain the direction using 2P output entropy.
The projection itself is the measurement. Plot its complete layerwise
trajectory in Game and Neutral separately, with question-bootstrap confidence
intervals and the 1P reference scale.

Within-dataset directions are primary for the projection trajectory. Applying
the SimpleMC-fitted direction to TriviaMC and the TriviaMC-fitted direction to
SimpleMC is the cross-dataset generality check.

## Stage 3: ablate and steer at the 2P decision position

Use the discovery-fitted primary entropy direction at the corresponding layer.
For every layer, independently in Game and Neutral:

1. Natural forward.
2. Cached identity path used as the causal baseline.
3. Projection ablation: `h' = h - (h dot d) d` at the 2P decision position.
4. Negative steering: `h' = h - 3d`.
5. Positive steering: `h' = h + 3d`.
6. One frozen same-norm orthogonal random-direction ablation.
7. Frozen same-norm orthogonal random steering at `-3` and `+3`.

The signed steering magnitude matches the established `entropy_probes`
unit-direction convention. The three steering doses for a condition/layer may
be evaluated in one expanded batch, but each remains a complete model forward.

Primary causal readout, reported separately for Game and Neutral:

- intervention-minus-natural change in the final within-question-centered
  logits for old ranks R1-R4;
- especially the old-W1 centered logit and W1-minus-W2 margin;
- old-W1 choice and complete final choice distribution as secondary endpoints.

Two summaries are frozen:

- absolute question-average effect;
- initial-margin-weighted effect, using nonnegative weights equal to each
  question's 1P top-two logit margin divided by the confirmation-split mean
  margin. This is a weighted mean, not an unstable effect/margin ratio.

No new dose-response regression is added.

## Validity gates

- Exact configured model ID, revision, layer count, batch regime, prompt mode,
  serialization, and reasoning-off settings.
- Exact token audit for both decision positions.
- Exact reproduction of the standalone Baseline and trusted unsplit 2P natural A-D logits in state collection.
- Exact identity-hook invariance across every tested intervention layer. Qwen's
  recurrent cached final-token path is not bit-identical to its unsplit path;
  that drift is measured and reported, and every causal contrast uses the same
  cached identity path rather than mixing cached interventions with unsplit
  natural logits.
- All completed logits and projections finite.
- Every selected layer hook fires exactly once per forward.
- Ablation leaves less than 0.5% of the pre-edit uncertainty projection after
  BF16 rounding.
- Steering changes that projection by the requested signed dose within an
  explicit BF16 tolerance; orthogonal controls must leave the uncertainty
  projection unchanged within their stricter tolerance.
- Random directions are unit norm and numerically orthogonal to the uncertainty
  direction.

## Interpretation

- A natural 2P projection shows that the frozen general MCQ uncertainty axis is
  active at the second decision.
- Ablation or signed steering that changes the R1-R4 ranking beyond the matched
  random intervention shows causal use of that axis in producing the second
  decision.
- A change confined to generic answer-distribution sharpness, with no
  candidate-rank structure, supports a generic output-confidence effect rather
  than uncertainty-guided revision.
- A null single-direction ablation is bounded: the `entropy_probes` repository
  found direction degeneracy, so it cannot exclude a redundant uncertainty
  subspace.

## Operations

All long jobs are resumable and checkpoint after each canonical four-question
cohort. Before each GPU start: authenticated fleet audit, exact complete-forward
count, prestart guard, and exact-path benchmark. The combined Qwen + Seed batch
has the standing $15 cap. If the measured complete forecast exceeds it, all
GPUs remain stopped and the user is asked rather than silently narrowing the
experiment.
