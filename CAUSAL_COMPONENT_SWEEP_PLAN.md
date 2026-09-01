# Qwen3.6-27B exhaustive causal component sweep

## Objective

Localize the final-position components that causally produce two distinct parts
of Qwen3.6-27B's SimpleMC Second Chance behavior:

1. **Literal flattening:** higher A-D entropy and lower centered A-D logit
   spread in Game than Neutral.
2. **Winner-changing behavior:** lower advantage for the Baseline winner and a
   higher probability that another answer becomes the final A-D argmax.

The experiment deliberately does not call every Baseline-opposing movement
"compression." It separately measures total movement, movement parallel to the
Baseline evidence vector, movement orthogonal to that vector, literal
flattening, and switching.

## Model, data, and intervention

- Model: `Qwen/Qwen3.6-27B` at revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Dataset: the frozen 500-question SimpleMC manifest already used for the
  behavioral and natural-activation runs.
- Conditions: the exact corrected Baseline, Game, and Neutral prompts already
  hashed and validated.
- Position: the final prompt position that supplies the first answer token.
- Components: the mixer and MLP in each of 64 layers, for 128 components total.
- Patch: replace a component's complete final-position output with the output
  produced by the same component on the same question under the paired
  condition. The downstream network then runs normally.

## Fixed split

The existing stratified split is reused so that no new split is chosen after
looking at causal outcomes:

- 251 discovery questions: exhaustive causal sweep.
- 249 confirmation questions: only discovery-nominated components.

The 251/249 imbalance is an old per-letter rounding artifact. It has no role in
candidate ranking. Dataset-weighted estimates are primary because they explain
the behavior actually observed. Equal-letter macro averages and A/B/C/D splits
are mandatory heterogeneity analyses.

The eight components from the prior observational screen already have causal
outputs on the confirmation half. Those data remain valid evidence for the
originally selected components, but any newly noticed metric or letter-specific
interpretation is treated as post-hoc. The new pipeline writes a separate
confirmation run so that its provenance is unambiguous.

## Stage 1: exhaustive causal discovery

Before discovery, a four-question/four-component smoke run must complete and
produce analyzable shards with the expected model revision and prompt hashes.
Failure stops the script before the full sweep.

For every discovery question:

1. Run natural Game and collect all 128 component outputs.
2. Run natural Neutral and collect all 128 component outputs.
3. For each component separately, run Game with that one output replaced by
   its same-question Neutral output.
4. Save the final exact A, B, C, and D token logits and prompt hash.

Only Neutral-into-Game is used in discovery. This is the direct removal test
and halves the exhaustive-sweep cost. Reciprocal Game-into-Neutral patches are
reserved for held-out candidates.

## Causal measurements

Let `d = center(z_patched - z_natural)` be the patch-induced change in the
final A-D logits, and let `b = center(z_baseline)`.

Every component receives the following descriptive causal measurements:

- `causal_total_l1 = sum(abs(d))`: total absolute centered A-D logit movement.
- `causal_total_l2 = ||d||2`: Euclidean total movement.
- `causal_baseline_coefficient = (d . b) / ||b||2^2`: restoration or opposition
  along the same-question Baseline evidence pattern.
- `causal_parallel_l2`: magnitude of the Baseline-parallel part of `d`.
- `causal_orthogonal_l2`: magnitude of the part of `d` orthogonal to `b`.
- `causal_orthogonal_fraction`: fraction of total L2 movement that is
  orthogonal.

The final-outcome effects are:

- A-D entropy;
- centered A-D spread;
- Baseline-winner advantage over the other three letters;
- switch indicator relative to the self-hosted Baseline A-D argmax;
- accuracy;
- alignment of the final A-D vector with the Baseline evidence vector.

Bootstrap intervals are paired. The dataset aggregation resamples questions;
the equal-letter aggregation resamples within Baseline-winner letter and gives
A/B/C/D equal weight.

## Preregistered candidate selection

Candidate selection uses only dataset-weighted discovery effects.

### Literal-compression family

A component is eligible only if its Neutral-into-Game patch both:

- lowers A-D entropy; and
- raises A-D spread.

Eligible components are scored by the mean fraction of the natural
Game-minus-Neutral entropy and spread gaps removed.

### Switching family

A component is eligible only if its Neutral-into-Game patch both:

- lowers switch probability; and
- restores Baseline-winner advantage.

Eligible components are scored by the mean fraction of the natural switch and
winner-advantage gaps removed.

For every component, the overall selection score is the larger of its eligible
literal-flattening and switching scores. The eight highest-scoring components
overall are confirmed. There is no family quota: discovery may therefore
select 8+0, 7+1, 4+4, or any other composition. A selected component belongs to
every family for which it passes the sign-agreement rule, so the grouped family
patches may overlap. If fewer than eight components are eligible for either
family, fewer than eight are confirmed. Total movement and orthogonal movement
are reported but do not influence selection.

## Stage 2: held-out confirmation

On all 249 confirmation questions:

- patch each selected component Neutral-into-Game;
- patch each selected component Game-into-Neutral;
- patch the compression family jointly in both directions;
- patch the switching family jointly in both directions;
- patch the full selected union jointly in both directions.

A component or group is a confirmed literal-flattening mediator when both
directions move entropy and spread consistently toward the source condition.
It is a confirmed winner-changing mediator when both directions consistently
move winner advantage and switching toward the source condition. Accuracy is a
safety/outcome diagnostic, not a selection endpoint.

No 128-way confirmatory correction is required because the confirmation set is
fixed without using confirmation outcomes. Within each selected family, exact
effect sizes and 95% confidence intervals are reported; interpretation focuses
on bidirectionality and metric agreement rather than a binary p-value.

## Outputs

The pipeline writes:

- a frozen 128-component discovery plan;
- resumable per-question/per-scenario NPZ shards;
- complete dataset-weighted, letter-macro, and per-letter causal-effect CSVs;
- the frozen candidate rankings and confirmation plan;
- held-out individual and grouped bidirectional effects;
- a machine-readable summary and final Markdown report.

The primary paper figures will show, across all 64 layers:

1. causal effects on entropy and spread;
2. causal effects on switching and winner advantage;
3. total, Baseline-parallel, and orthogonal causal movement;
4. held-out bidirectional effects for selected components and groups.

## Compute estimate and stopping rule

Discovery requires `128 x 251 = 32,128` patched forwards plus 502 natural
reference forwards. Confirmation requires at most eight individual components
and three groups in two directions: at most `22 x 249 = 5,478` patched forwards
plus 498 references.

With the existing unoptimized A100 implementation, the expected total is about
6-8 GPU-hours and approximately **$7-$9**, with a conservative ceiling below
the standing **$15** batch limit. The run must stop before the cap. It is fully
resumable. The Vast instance is stopped—not destroyed—after outputs are
retrieved and verified.

No compute is launched until the user explicitly approves this documented run.
