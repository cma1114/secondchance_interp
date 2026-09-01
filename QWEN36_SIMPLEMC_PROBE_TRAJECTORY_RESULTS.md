# Qwen3.6-27B SimpleMC probe-trajectory results

## Scope

This report uses the corrected-Neutral activation collection in
`outputs/mechanistic/qwen36_27b_simplemc_clean`. Baseline and Second Chance are
identical to the original collection; Neutral does not contain the unintended
literal `None` prefix.

At every other residual readout, a five-fold cross-fitted linear centroid probe
was trained only on Baseline residuals to predict the final Baseline A-D winner.
Every question in Baseline, Second Chance, and Neutral was scored by a probe that
was not trained on that question. Candidate scores were centered within question
and standardized by held-out Baseline score dispersion at that layer.

## Essential reliability result

Probe-score trajectories are not interpretable as answer trajectories unless the
probe leader predicts the condition's eventual output. Reliability was therefore
measured against the native full-vocabulary greedy output letter, not merely the
final intermediate-lens argmax. The two agree on 98% of Baseline, 97% of Second
Chance, and 96% of corrected-Neutral trials.

Letter-balanced probe accuracy was:

| Readout | Baseline | Second Chance | Neutral |
|---:|---:|---:|---:|
| 40 | 33.3% | 24.6% | 26.4% |
| 44 | 43.5% | 23.6% | 27.1% |
| 48 | 67.4% | 45.1% | 52.4% |
| 52 | 78.0% | 58.6% | 66.1% |
| 56 | 79.0% | 61.4% | 74.7% |
| 60 | 79.0% | 61.2% | 74.7% |
| 64 | 76.2% | 42.7% | 50.1% |

Chance is 25%. Before readout 48, the Baseline-trained probe responds primarily
to condition-wide prompt shifts rather than question-specific answer identity.
For example, its Game prediction is A on 91.0% of questions at readout 30 and B
on 99.0% at readout 40. Baseline does not exhibit this collapse. Consequently,
the visually dramatic early probe-score trajectories must not be described as
answer emergence, flattening, or switching.

## Prior-answer and prior-runner analysis

The **prior answer** is the native Baseline output letter. The **prior runner-up**
is the strongest final Baseline A-D alternative after excluding that output.
For each redo condition, two complementary quantities were plotted:

1. candidate lead margin: the candidate's probe score minus its strongest
   competitor, where positive means the probe predicts that candidate;
2. the raw fraction of questions on which the probe leader is the prior answer,
   prior runner-up, or one of the other two letters.

The unreliable pre-48 region is shaded in the figure. Open diamonds give the
actual final-output fractions.

Actual final outcomes were:

| Condition | Prior answer | Prior runner-up | Other |
|---|---:|---:|---:|
| Second Chance | 39.8% | 40.8% | 19.4% |
| Neutral | 70.0% | 23.8% | 6.2% |

In Neutral, the prior-answer probe margin becomes positive at readout 48 and
rises to +0.66 Baseline-probe SD units at readout 56. The probe identifies the
prior answer on 52.6% of trials at readout 48 and 60.0% at readout 56. This is
consistent with Neutral regenerating the prior response as answer information
becomes decodable.

In Second Chance, the prior-answer margin is approximately zero at readout 48
and only modestly positive at readouts 52-60 (maximum +0.23). The prior runner-up
margin remains negative throughout the reliable window (approximately -0.4 to
-0.5). At readout 56, the probe leaders are almost evenly divided among prior
answer (32.8%), prior runner-up (33.4%), and other (33.8%). Thus this Baseline-
trained probe does not show a clean trajectory in which the prior answer is
regenerated and then replaced by the prior runner-up.

The absence of a positive prior-runner margin is not proof that the runner-up is
mechanistically absent. The probe transfers imperfectly to Second Chance and its
balanced accuracy falls again at the final readout, even though the actual final
answer is the prior runner-up on 40.8% of trials. The result establishes what is
and is not visible in the Baseline-trained linear coordinate system.

## Interpretation

- Answer-level use of these probe trajectories should begin at approximately
  readout 48, not at the visually striking earlier condition shifts.
- Corrected Neutral shows clear recovery of the prior answer.
- Second Chance shows reduced and unstable prior-answer evidence, but not clean
  positive emergence of the prior runner-up in this probe coordinate system.
- A decoder intended for stronger cross-condition claims should be trained
  jointly across conditions, hold out entire questions, and balance both answer
  letter and condition. This is distinct from the present Baseline-to-redo
  generalization test.

## Artifacts

- Canonical reliability figure:
  `outputs/mechanistic/qwen36_27b_simplemc_clean/analysis/preserved_figures/probe_final_answer_accuracy.png`
- Canonical prior-answer/runner figure:
  `outputs/mechanistic/qwen36_27b_simplemc_clean/analysis/preserved_figures/prior_answer_runner_trajectories.png`
- Numerical values:
  `outputs/mechanistic/qwen36_27b_simplemc_clean/analysis/probe_trajectories/simplemc_probe_final_answer_accuracy.csv` and
  `outputs/mechanistic/qwen36_27b_simplemc_clean/analysis/prior_answer_probe/prior_answer_runner_probe_trajectories.csv`
- Final fractions:
  `outputs/mechanistic/qwen36_27b_simplemc_clean/analysis/prior_answer_probe/prior_answer_runner_final_fractions.csv`
- Analysis code:
  `mechanistic/probe_mechanism_trajectories.py` and
  `mechanistic/prior_answer_probe_trajectories.py`
