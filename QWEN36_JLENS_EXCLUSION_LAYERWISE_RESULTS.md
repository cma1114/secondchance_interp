# Qwen3.6-27B: layerwise causal test of the feedback-end exclusion direction

## Question

The unrestricted JLens explorer shows a clear exclusion-related vocabulary
cluster at the period ending the Game feedback sentence. The experiment asked
whether this specific, interpretable direction causes the model to abandon its
previous answer.

This extends the earlier combined L41--48 coordinate swap, which had essentially
no behavioral effect. The new experiment tests every causally actionable
post-block readout from L41 through L63 separately, so a narrow causal layer
cannot be averaged away. L64 is omitted because changing a source token after
the final block cannot affect the already-computed decision position.

## Intervention

At every layer, a unit exclusion direction was defined as the gradient of the
JLens exclusion-family score at the mean Game residual at the feedback-ending
period. The family contains inflections of *exclude*, *restrict*, *ban*,
*reject*, and *eliminate*.

For each of 128 held-out questions and each layer:

- **Remove from Game:** replace only Game's scalar coordinate along that
  direction with the same question's natural Neutral coordinate.
- **Insert into Neutral:** replace only Neutral's scalar coordinate with the
  same question's natural Game coordinate.

All residual dimensions orthogonal to the direction remain unchanged. In
addition to the 23 single-layer interventions in each direction, the experiment
tests joint L41--63 and L49--63 replacements. The primary outcomes are the
actual final A--D logits and generated A--D choice.

## Manipulation check

The intervention moved the JLens readout in the intended semantic direction at
every layer:

- Game ablation reduced the exclusion-family score by 1.94--5.71 JLens units.
- Neutral insertion increased it by 2.45--5.11 JLens units.
- Game's natural exclusion coordinate exceeded Neutral's at every layer, with
  the mean paired gap increasing from +1.93 at L41 to +17.43 at L63.

Thus the behavioral result below is not caused by a reversed sign, failed
steering operation, or condition-labeling error.

## Result

The direction does **not** behave like a causal instruction to exclude the
previous answer.

Natural macro switch rates in the 128-question sample are 43.7% for Game and
25.3% for Neutral.

### Removing the direction from Game

No single layer produced the predicted reduction in switching. Effects were
small and fluctuated around zero. The largest absolute single-layer change was
at L53, where switching increased by 1.4 percentage points rather than
decreased (95% bootstrap CI [0.0, 3.6]; three final choices changed).

Joint removal across L41--63 likewise increased switching by 2.0 points
(CI [0.0, 4.6]; three new switches), with essentially no change in the
prior-answer margin (+0.0006 logits, CI [-0.0127, 0.0132]). Joint removal across
L49--63 changed switching by only +0.7 points.

### Inserting the direction into Neutral

The dominant behavioral tendency was the opposite of the hypothesis: insertion
usually **reduced** Neutral switching. The largest change was at L41, -3.9
percentage points (CI [-7.6, -0.7]); all five affected trials were prevented
switches. Several other layers between L42 and L60 showed reductions of roughly
2--3.5 points.

Joint insertion across L41--63 reduced switching by 2.0 points, although its CI
included zero ([-5.4, +0.9]). Joint insertion across L49--63 reduced switching
by 2.4 points (CI [-6.1, +0.9]), lowered the prior-answer margin by 0.0149 logits
(CI [-0.0279, -0.0017]), and reduced A--D spread by 0.0059 logits SD
(CI [-0.0103, -0.0014]). Only five final choices changed in each joint
intervention.

## Interpretation

The feedback-end exclusion direction is a strong, nearly question-invariant
representation of the Game condition, but it is not the mechanism that makes
the model switch answers. Increasing its JLens exclusion score does not cause
more switching, and removing it does not preserve the previous answer.

The most conservative interpretation is that JLens makes a condition-related
state human-readable as exclusion vocabulary, but the corresponding local
readout direction is not the causal control variable used by the model. It may
be a downstream description, a redundant marker, or a superposed direction
whose isolated insertion perturbs computation differently from the natural
multi-dimensional state.

Together with the earlier null L41--48 test, this substantially weakens the
feedback-ending-period exclusion direction as the primary mechanism of Second
Chance behavior.

## Artifacts

- Canonical figure:
  `outputs/mechanistic/qwen36_27b_exclusion_layerwise/analysis/preserved_figures/exclusion_direction_layerwise.png`
- Layerwise estimates and bootstrap intervals:
  `outputs/mechanistic/qwen36_27b_exclusion_layerwise/analysis/layerwise_effects.csv`
- Complete summary:
  `outputs/mechanistic/qwen36_27b_exclusion_layerwise/analysis/layerwise_summary.json`
- JLens manipulation check:
  `outputs/mechanistic/qwen36_27b_exclusion_layerwise/analysis/exclusion_score_validation.json`
- Resumable per-question intervention outputs:
  `outputs/mechanistic/qwen36_27b_exclusion_layerwise/shards/`
- Intervention code: `mechanistic/run_jlens_exclusion_layerwise.py`
- Analysis code: `mechanistic/analyze_jlens_exclusion_layerwise.py`
- Manipulation-check code: `mechanistic/diagnose_jlens_exclusion_layerwise.py`

The preserved Vast instance was stopped after retrieval and validation. The
batch used approximately $0.9 of GPU compute, well below the $15 standing cap.
