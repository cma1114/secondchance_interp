# Qwen3.6-27B position-resolved causal component results

## Bottom line

An eight-component set selected on 251 SimpleMC discovery questions causally
accounts for most of Qwen3.6-27B's differential Second Chance switching and
about half of the accompanying continuous redistribution of A-D evidence on an
untouched 249-question confirmation set.

**Follow-up qualification:** cumulative and leave-one-out tests show that this
is a large effect of a coordinated eight-output transplant, not eight
individually explanatory mediators. The effect is built mainly by late
decision-position mixers; feedback MLP 44 contributes nothing detectable and
feedback MLP 32 is not conditionally necessary for switching. See
`POSITION_COMPONENT_INTERACTION_RESULTS.md`.

The result is reciprocal. Replacing these Game outputs with same-question
Neutral outputs almost eliminates the Game-Neutral switching gap. Inserting
the Game outputs into Neutral creates most of the gap. This is substantially
stronger evidence than the previous observational trajectories or single
direction steering experiments.

It is not yet a fully interpreted circuit. The intervention replaces complete
high-dimensional component outputs, and the components interact strongly. The
experiment localizes a compact causal mechanism but does not yet identify the
features inside those outputs or the connections among the components.

## Experiment

- Model: `Qwen/Qwen3.6-27B`, revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Prompts: corrected `baseline_matched_empty_history` format in
  `PROMPT_SPEC.md`.
- Discovery: 251 frozen questions; all 64 mixers and 64 MLPs at each of the
  feedback-end period, repeated-user final token, and final decision position
  (384 targets).
- Discovery intervention: same-question Neutral output into Game, one target
  at a time.
- Selection: top eight eligible effects across literal flattening, switching,
  and ordered rank redistribution, with no quota by family.
- Confirmation: 249 untouched questions; every selected component and the
  selected groups patched in both directions.
- Numerical control: every patched forward used 15 intervention rows plus an
  unpatched matched control row. Partial chunks were padded to the same physical
  batch size. Two independent calibration runs agreed exactly.

## Selected components

Layer indices in artifacts are zero-based; block numbers below are one-based.

| Position | Component |
|---|---|
| Final decision | Mixers 50, 52, 56, 60, 61, and 63 |
| Feedback-end period | MLPs 32 and 44 |
| Repeated-user final token | None |

Seven of the eight were independently eligible as flattening and ordered-rank
mediators. Mixer 63 was selected only through the switching criterion.

## Held-out joint causal effects

Natural confirmation-set values were:

- Game switch rate: 40.96%; Neutral: 26.51%; gap: 14.46 percentage points.
- Game A-D entropy: 1.113; Neutral: 0.976.
- Game A-D spread: 0.824; Neutral: 1.138.
- Game original-winner advantage: 1.043 logits; Neutral: 1.840 logits.

The table reports the fraction of each natural Game-Neutral gap mediated by
the eight-component joint patch, with paired 95% bootstrap intervals.

| Outcome | Remove Game outputs (Neutral into Game) | Insert Game outputs into Neutral |
|---|---:|---:|
| A-D entropy | 49.1% [35.5, 63.0] | 50.4% [35.9, 65.1] |
| A-D spread | 35.5% [27.8, 43.3] | 51.0% [41.1, 61.1] |
| Original-winner advantage | 48.2% [39.2, 57.3] | 56.2% [45.5, 67.0] |
| Game-Neutral switch-rate gap | 88.9% [52.8, 125.0] | 69.4% [36.1, 102.8] |
| Ordered rank redistribution | 45.3% [38.2, 52.5] | 54.4% [46.0, 63.0] |

In absolute terms, removal lowered Game switching by 12.85 points, from 40.96%
to 28.11%, leaving only 1.61 points above Neutral. Insertion raised Neutral
switching by 10.04 points, from 26.51% to 36.55%.

Equal-letter macro estimates were similar: 93.9% and 60.7% of the switch gap,
and 43.1% and 50.8% of ordered rank redistribution, for removal and insertion
respectively. The switch effect also had the predicted sign for every Baseline
winner letter in both directions.

## Individual effects and interaction

Five decision-position mixers replicated ordered rank redistribution in both
directions: blocks 50, 52, 56, 60, and 61. Feedback-end MLP 32 had a clear
Neutral-into-Game removal effect but essentially no reciprocal insertion
effect, so it looks more like a condition-dependent necessary carrier than a
standalone sufficient instruction.

Feedback-end MLP 44 did not replicate individually. Decision Mixer 63 was more
striking: by itself it moved continuous rank geometry in the wrong direction
or not at all on confirmation. Yet adding Mixer 63 to the seven-component joint
patch increased held-out switch mediation from 47.2% to 88.9% on removal and
from 30.6% to 69.4% on insertion, while also increasing the continuous effects.
Thus the strongest result is an interacting component set, not the sum of eight
independent mediators. A leave-one-out factorial test would be needed to assign
the interaction more precisely.

## Interpretation

The experiment supports the following mechanistic story for this model and
prompt format:

1. The incorrect-feedback condition changes an earlier representation at the
   feedback-end position; MLP 32 carries part of that difference forward.
2. A cluster of late decision-position mixers implements a question-specific,
   Baseline-rank-opposed redistribution: evidence is moved away from the
   Baseline winner and toward lower-ranked answers.
3. These components act non-additively. Their joint state is sufficient to
   induce much of the Game transformation in Neutral and necessary for nearly
   all of the differential switching in Game.

This is compatible with strategic compression/noise at the behavioral level,
but it is more specific: the causal effect is ordered by the question's own
Baseline answer ranks rather than being generic isotropic noise. It does not
show a simple threshold detector, nor does it yet reveal what semantic feature
inside the feedback-end MLP output activates the late mixer cluster.

## Figures and artifacts

- `outputs/causal/qwen36_27b_position_components_corrected/figures/position_component_rank_writes.png`
- `outputs/causal/qwen36_27b_position_components_corrected/figures/heldout_group_mediation.png`
- `outputs/causal/qwen36_27b_position_components_corrected/figures/heldout_individual_rank_mediation.png`
- Discovery and confirmation effect tables are in their respective `analysis/`
  directories.
- The frozen method is in `POSITION_COMPONENT_CAUSAL_PLAN.md`.

## Compute and instance state

The A100 instance ran for 288.0 minutes at $1.0389/hour, approximately $4.99.
It was stopped, not destroyed. Its stopped storage charge is $0.0389/hour
(about $0.93/day), preserving roughly 85 GB of model and research files for
follow-up work.
