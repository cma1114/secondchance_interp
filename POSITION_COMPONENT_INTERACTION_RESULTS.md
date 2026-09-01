# Qwen3.6-27B component-interaction results

## Bottom line

The large eight-output patch effect is real on the held-out 249 SimpleMC
questions, but it should not be interpreted as eight individually explanatory
components. Cumulative and leave-one-out patches show that the effect is built
principally by a sequence of late decision-position mixers. Feedback-end MLP
32 contributes modestly to continuous answer-rank redistribution but is not
conditionally necessary for switching; feedback-end MLP 44 contributes
nothing detectable.

Within the complete intervention, Decision Mixers 56 and 63 are the strongest
and most consistently necessary components for switching. The result therefore
localizes a coordinated late output transformation, not the earlier mechanism
that reads the feedback and initiates that transformation.

## Experiment

- Model, prompts, and frozen split are identical to the corrected empty-history
  position-component experiment.
- Questions: 249 untouched confirmation questions.
- Cumulative diagnostic: add paired source-condition outputs in causal order.
- Conditional diagnostic: patch all eight outputs, then omit each output in
  turn.
- Both Neutral-into-Game removal and Game-into-Neutral insertion were run.
- Intervals are paired question-bootstrap 95% confidence intervals.

The natural Game-minus-Neutral gaps were 14.46 percentage points of switching
and 0.302 units of ordered answer-rank redistribution.

## Cumulative effects

| Added through | Rank gap removed | Rank gap inserted | Switching removed | Switching inserted |
|---|---:|---:|---:|---:|
| Feedback MLP 32 | 6.1% | 0.2% | 1.61 pp | 0.80 pp |
| Feedback MLP 44 | 5.5% | -0.5% | 2.01 pp | -0.80 pp |
| Decision Mixer 50 | 15.1% | 12.8% | 2.81 pp | 1.61 pp |
| Decision Mixer 52 | 17.6% | 20.4% | 2.01 pp | 2.41 pp |
| Decision Mixer 56 | 23.0% | 27.1% | 4.42 pp | 3.61 pp |
| Decision Mixer 60 | 24.9% | 32.7% | 4.02 pp | 2.41 pp |
| Decision Mixer 61 | 32.2% | 38.5% | 6.83 pp | 4.42 pp |
| Decision Mixer 63 | 45.3% | 54.4% | 12.85 pp | 10.04 pp |

Continuous redistribution accumulates across the late mixers. Switching first
becomes reliable after Mixer 56 and rises most sharply when Mixer 63 is added.

## Conditional contribution within the complete patch

The table reports full-eight effect minus the effect after omitting the named
component.

| Component | Rank removal | Rank insertion | Switching removal | Switching insertion |
|---|---:|---:|---:|---:|
| Feedback MLP 32 | 2.4% | -0.5% | 0.00 pp | 1.20 pp |
| Feedback MLP 44 | -0.3% | -0.9% | 0.00 pp | 0.00 pp |
| Decision Mixer 50 | 6.0% | 8.8% | 2.41 pp | 0.40 pp |
| Decision Mixer 52 | 9.4% | 9.2% | 1.61 pp | 2.01 pp |
| Decision Mixer 56 | **16.1%** | **15.2%** | **4.02 pp** | **6.43 pp** |
| Decision Mixer 60 | 7.2% | 9.2% | 0.00 pp | 1.61 pp |
| Decision Mixer 61 | 9.5% | 8.2% | 2.01 pp | 2.41 pp |
| Decision Mixer 63 | **13.1%** | **15.9%** | **6.02 pp** | **5.62 pp** |

For continuous redistribution, all six decision mixers have reliable
conditional effects in both directions. For switching, only Mixers 56 and 63
have clearly nonzero conditional effects in both directions. Mixer 50 is also
nonzero for removal from Game. The remaining discrete-switch estimates are too
imprecise to distinguish from zero.

## Interpretation

The cumulative pattern explains why the joint patch greatly exceeds isolated
patches. A single source-condition output is inserted into an otherwise target-
condition trajectory and can be canceled downstream. The cumulative patch
inserts a coherent sequence of late source-condition writes. In addition,
switching is an argmax threshold: multiple modest continuous movements can
combine to change an answer even when none changes it alone.

This is still a coordinated output transplant. Later source outputs were cached
from the natural source-condition run rather than recomputed from the earlier
patched state. Consequently, these results justify saying that a sequence of
late decision outputs is jointly effective and that Mixers 56 and 63 are
conditionally important. They do not show that all eight outputs are the
natural circuit, and they do not identify the upstream feedback-reading
mechanism.

## Artifacts

- `outputs/causal/qwen36_27b_position_component_interactions/analysis/component_interactions.png`
- `outputs/causal/qwen36_27b_position_component_interactions/analysis/component_interactions.svg`
- `outputs/causal/qwen36_27b_position_component_interactions/analysis/interaction_effects.csv`
- `outputs/causal/qwen36_27b_position_component_interactions/analysis/interaction_summary.json`
- `POSITION_COMPONENT_INTERACTION_PLAN.md`

## Compute

The preserved A100 instance ran for 44.7 minutes at $1.0389/hour,
about $0.77. It was stopped after the compact results were retrieved and was
not destroyed.
