# Qwen3.6-27B SimpleMC mechanistic-interpretability results

> **Neutral-condition contamination warning.** These self-hosted neutral runs
> contain the unintended literal system-prefix token `None`, inherited from the
> experimental runner's interpolation of `setup_text = None`. Baseline and Game
> trajectories are unaffected, but neutral contrasts are not clean controls and
> require rerunning after prompt correction.

## Scope

- Model: `Qwen/Qwen3.6-27B`, commit `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- Dataset: 500 SimpleMC questions
- Conditions: baseline capabilities, Second Chance (`incorrect`), and neutral/lost
- Readouts: embedding plus all 64 transformer blocks at the final prompt position
- Methods: logit lens over aggregated A-D token variants and held-out linear probes
- No activation patching or other causal intervention was performed.

The native prompts use the model-visible chat order: system message first, then the
user/assistant/user turns. This matches the Qwen chat template and the order the
successful OpenRouter behavioral run must have rendered for the model.

## Behavioral and final-readout checks

- Native greedy baseline tokens agreed with the recorded OpenRouter baseline
  choices on 88.6% of questions. Agreement was 82.0% in Second Chance and 84.0%
  in neutral, so this is an aggregate rather than an exact trial-level replication.
- Relative to the native greedy baseline token, the model changed answers on 60.2%
  of Second Chance trials and 32.2% of neutral trials. The corresponding rates from
  the aggregated A-D argmax used in the trajectory analysis are 58.2% and 32.8%.
- Mean final A-D entropy was 0.938 nats at baseline, 1.051 in Second Chance, and
  0.873 in neutral.

Mean final centered logits, aligned by the baseline aggregated-A-D ranking used in
the trajectory analysis, were:

| Condition | Baseline winner | Runner-up | Rank 3 | Rank 4 |
|---|---:|---:|---:|---:|
| Baseline | 1.681 | 0.445 | -0.503 | -1.622 |
| Second Chance | 0.544 | 0.306 | -0.142 | -0.708 |
| Neutral | 1.671 | 0.333 | -0.527 | -1.477 |

Thus the Second Chance prompt produces broad flattening, but the flattening is not
uniform: the baseline winner falls much more than the runner-up. Neutral remains
close to baseline and becomes slightly sharper in entropy, so this is not merely a
generic effect of taking another conversational turn.

A descriptive final-readout regression gives the same qualitative decomposition.
The Second Chance vector is approximately `0.53 * baseline`, with an additional
`-0.47` centered penalty on the baseline winner. Neutral is approximately
`0.95 * baseline`, with no winner penalty (`+0.10` descriptively). The sizeable
unexplained residual in the Second Chance fit is consistent with additional
question-specific perturbation, but does not by itself imply that the model injects
random noise.

## Layerwise hypothesis comparison

The primary contrast is Second Chance versus baseline; Second Chance versus neutral
tests whether the difference is feedback-specific.

The strongest early differential update is block 27 to 28. A generic compression
model explains 79.2% of the held-out Second-Chance-versus-baseline update variance
and 91.1% of the Second-Chance-versus-neutral update variance. Adding a threshold
barely improves these values (79.20% versus 79.17%, and 91.16% versus 91.12%), and
the extra gated coefficient is tiny. This favors generic compression over a
thresholded current-leader rule at the main early transition.

There is later evidence for extra suppression of the baseline-defined winner, but
it is distributed and not monotonic. For example, the targeted-winner coefficient
is negative at transitions 52->53, 53->54, 60->61, and 62->63 relative to baseline;
the Second-Chance-versus-neutral coefficient at 60->61 is -0.481. Other transitions
partly reverse these changes, so no single late block implements the whole effect.

The thresholded-current-leader model is not the overall winner. Its best
Second-Chance-versus-baseline fit has a positive, rather than suppressive, gated
leader coefficient. At block 27 to 28 it has suppressive coefficients, but explains
less held-out variance than ordinary compression (69.7% versus 79.2%). This does not
rule the hypothesis out, because logit-lens coordinates are imperfect intermediate
readouts, but it is not the leading observational account.

## Probe results

Five-fold held-out logistic probes were trained only on baseline residual streams
to predict the baseline final winner or runner-up letter. Because winner-letter
frequencies are imbalanced, emergence is summarized with balanced accuracy (the
mean of the four per-letter recalls), for which chance is 25%.

- Winner identity rises from 36.5% at layer 32 to 56.4% at layer 44 and 76.9% at
  layer 48. It peaks at 85.9% at layer 60 and is 82.2% at the final readout.
- Runner-up identity is much weaker: 30.8% at layer 32, 38.3% at layer 44, 41.4% at
  layer 48, and a peak of 56.1% at layer 59.

This independently confirms that the eventual answer identity becomes linearly
represented late in the network, and that the winner is represented much more
cleanly than the runner-up. It does not establish that the Second Chance prompt
causally acts on that representation.

## Capabilities-test answer-emergence figure

The publication version combines the letter-balanced native logit lens and held-out
probe trajectories. Shaded regions are 95% confidence intervals. There are no vertical
reference lines in the static figure: the dashed vertical line in the interactive view
is only the user-selected layer cursor and has no statistical meaning.

The logit-lens vertical axis is the letter-balanced mean centered A-D pseudo-logit, in
natural-logit units. Centering is within question; ranks are defined by the final
baseline A-D order. Consequently, curve differences have the familiar logit-margin
interpretation, while each individual curve is relative to that question's mean A-D
pseudo-logit. The exact formulas and interval estimands are documented in
`MECHINTERP_IMPLEMENTATION.md`.

### Game and neutral, separated by final switching

These figures are retained only as exploratory diagnostics. The switch groups are
defined separately by each condition's final answer, so comparing switch with switch
removes the major behavioral difference in switch frequency and compares different
question subsets. Letter balancing is also unstable inside the highly imbalanced switch
strata. Finally, rank-specific condition-minus-baseline changes do not constitute a new
ranking: generic compression mechanically gives the originally weakest option the
largest positive delta. No substantive conclusion should be based on those delta plots.

### Primary all-trial condition comparison

Combining all 500 trials restores the expected condition differentiation. Baseline and
neutral produce closely related late trajectories: the original winner becomes strongly
dominant and the four-option distribution spreads apart. Second Chance begins diverging
at approximately the same stage at which the original winner becomes output-aligned,
but its winner advantage grows much less and the total A-D spread remains lower.

This does not show a clean sequence in which a fully established original winner is
subsequently suppressed. It is better described as **attenuated amplification of the
original winner together with broad compression**. The original runner-up is not
absolutely boosted relative to baseline or neutral; it is comparatively preserved while
the winner is weakened and the lower-ranked options move toward zero. Neutral's close
tracking of baseline is consistent with the behavioral and entropy results and was
obscured by the condition-specific switch stratification.

## Current interpretation

Among the three proposed hypotheses, the observational evidence favors a hybrid:

1. **Generic compression plus question-specific perturbation is the dominant early
   signature.**
2. **Targeted suppression of the baseline winner is a plausible secondary, later
   component.** It is visible in final-readout decomposition and at several late
   transitions.
3. **Thresholded suppression of whichever option is currently leading is not the
   best account of these data.**

The evidence is mechanistic in the weak, representational sense: it localizes and
characterizes transformations in residual-stream readouts. It is not yet a causal
circuit claim.

## Output files

- `outputs/mechanistic/qwen36_27b_simplemc/run_metadata.json`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/trajectory_summary.json`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/rank_trajectories.csv`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/rank_trajectories.svg`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/strength_trajectories.csv`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/hypothesis_fits.csv`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/hypothesis_summary.json`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/centroid_probe_results.csv`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/logistic_probe_results.csv`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/paper_figures/answer_emergence_combined.{pdf,svg,png}`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/paper_figures/answer_emergence_logit_lens.{pdf,svg,png}`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/paper_figures/answer_emergence_probes.{pdf,svg,png}`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/paper_figures/answer_emergence_values.csv`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/paper_figures/{incorrect,neutral}_switch_stratified_{raw,delta}.{svg,png}`
- `output/pdf/{incorrect,neutral}_switch_stratified_{raw,delta}.pdf`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/paper_figures/condition_switch_trajectories.csv`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/paper_figures/cross_fitted_candidate_probe_scores.npz`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/paper_figures/all_trials_original_rank_trajectories.{svg,png}`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/paper_figures/all_trials_mechanism_summary.{svg,png}`
- `outputs/mechanistic/qwen36_27b_simplemc/analysis/paper_figures/all_trials_mechanism_values.csv`

All 1,500 activation shards were copied locally and matched the remote aggregate
checksum before the Vast instance was destroyed.
