# Causal feedback-direction steering: results

> **Invalidated as a clean Game-versus-neutral mechanism test.** The neutral
> prompt used in these runs contains the unintended literal system-prefix token
> `None`. The source runner sets `setup_text = None` and then interpolates it
> into an f-string, producing `None\nOutput ONLY ...` before the API call. The
> learned direction and all neutral comparisons therefore conflate feedback
> condition with this prompt-construction artifact. Raw results are retained for
> auditability, but they must not be presented as evidence about a clean neutral
> control.

## Bottom line

The experiment is complete, but it does **not** establish the learned
Game-versus-neutral direction as the mechanism controlling Second Chance
switching.

The direction is an almost perfect linear marker of whether the prompt is Game
or neutral. Adding it at the residual stream also has a real, small effect on
the final A-D logits in some settings—most clearly, it lowers the original
baseline winner relative to the other choices in an already-Game prompt.
However:

- it barely changes discrete answers;
- adding it to neutral does not reliably create Game-like switching;
- removing it from Game does not reliably eliminate Game-like switching;
- effects in neutral are non-monotonic and sometimes have the opposite sign;
- the entropy effects are also produced by the matched orthogonal control.

The defensible conclusion is therefore: **the contrastive direction carries
some causally usable information about the prompt condition and can weakly
modulate answer logits, but it is not a demonstrated feedback-control variable
and does not explain the behavioral Game effect.**

## Design and estimand

- Model: `Qwen/Qwen3.6-27B`, revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Dataset: SimpleMC.
- Direction: the unit-normalized training mean of the paired final-position
  residual difference, Game minus neutral, separately at each readout.
- Training/intervention split: direction fitted without the intervention fold;
  60 held-out intervention questions, balanced 15/15/15/15 by the original
  baseline winner letter.
- Intervention: add or subtract one natural training-set Game-neutral projection
  gap at readouts 24, 30, or 36. Readout 30 also used doses -0.5, +0.5, and +2
  and a matched orthogonal control direction.
- Estimand: paired change relative to the unsteered version of the same prompt
  condition, macro-averaged over original-winner letters.
- Intervals: percentile 95% confidence intervals from 10,000 bootstrap samples,
  resampling questions within original-winner-letter strata.

“Original winner” and ranks 1–4 below are fixed from the unsteered baseline
capabilities pass, not re-ranked after intervention.

## Run validation

| Check | Result |
|---|---:|
| Completed outputs | 1,440 / 1,440 |
| Scenarios | 24, each with 60 questions |
| Full-vocabulary top token was A–D | 1,440 / 1,440 |
| Unsteered reference checks | 120 / 120 passed |
| Mean / maximum unsteered reference absolute error | 0.0512 / 0.0688 logits |
| Mean / maximum relative error | 0.200% / 0.287% |
| Local tests after analysis changes | 17 / 17 passed |

The checkpoint environment matched the validated self-hosted run: PyTorch
2.13.0+cu130 and Transformers 5.14.1, with bfloat16 model computation.
Individual within-run logit changes occur predominantly in increments of 0.125
logits. The confidence intervals quantify question-sampling uncertainty;
they do not include uncertainty from fitting a new direction on a different
fold or running a new control direction.

## The held-out subset contains the behavioral target

Before considering steering, the unsteered Game prompt differs strongly from
the unsteered neutral prompt on these same 60 questions.

| Outcome: Game minus neutral | Estimate | 95% CI |
|---|---:|---:|
| Switch rate | +23.3 percentage points | [+13.3, +33.3] |
| Original-winner probability | -0.179 | [-0.220, -0.141] |
| Original-winner margin | -1.575 logits | [-1.958, -1.196] |
| A-D entropy | +0.225 nats | [+0.164, +0.290] |
| A-D logit spread | -0.629 logits | [-0.784, -0.477] |
| Overall accuracy | -6.7 percentage points | [-16.7, +3.3] |
| Accuracy among baseline-incorrect questions | +9.2 percentage points | [0.0, +20.0] |

Thus a null steering result cannot be attributed to selecting a held-out subset
on which the Game and neutral conditions behaved alike.

The rank decomposition of the unsteered Game-minus-neutral contrast is also
informative. After centering the four logits within each question, the original
winner falls by 1.315 logits, rank 2 changes by only -0.108, rank 3 rises by
0.449, and rank 4 rises by 0.974. This is broad flattening dominated by loss of
the winner and gains among the initially lower-ranked options—not simple
runner-up boosting.

## Primary causal results

Each row below is a paired change from the same condition without steering.
Positive direction means adding the learned Game-minus-neutral direction;
negative direction means subtracting it.

| Condition and intervention | Δ switch | Δ winner margin | Δ entropy | Δ A-D spread |
|---|---:|---:|---:|---:|
| Game, L24, -1 | 0.0 pp | +0.006 | +0.004 | -0.008 |
| Game, L24, +1 | 0.0 pp | -0.002 | -0.004 | +0.009 |
| Game, L30, -1 | -1.7 pp | +0.021 | +0.009 | -0.017 |
| Game, L30, +1 | -1.7 pp | **-0.050** | **+0.007** | **-0.016** |
| Game, L30, +2 | -1.7 pp | **-0.075** | **+0.026** | **-0.061** |
| Game, L36, -1 | -5.0 pp | **+0.102** | +0.006 | -0.007 |
| Game, L36, +1 | +3.3 pp | **-0.138** | **+0.010** | -0.019 |
| Neutral, L30, +1 | +1.7 pp | +0.025 | -0.001 | +0.007 |
| Neutral, L30, +2 | 0.0 pp | -0.042 | **+0.017** | **-0.036** |
| Neutral, L36, +1 | 0.0 pp | **+0.063** | **-0.015** | **+0.051** |
| Game orthogonal control, L30, +1 | -3.3 pp | +0.013 | **+0.011** | **-0.022** |

Bold continuous effects have bootstrap intervals excluding zero. None of the
switch-rate effects has a two-sided interval excluding zero. Selected exact
intervals are:

- Game L30 +1 winner margin: -0.050 [-0.077, -0.023].
- Game L30 +2 winner margin: -0.075 [-0.113, -0.038].
- Game L36 +1 winner margin: -0.138 [-0.181, -0.096].
- Game L36 -1 winner margin: +0.102 [+0.060, +0.142].
- Game L36 +1 switch rate: +3.3 pp [-3.3, +10.0].
- Game L36 -1 switch rate: -5.0 pp [-11.7, 0.0].
- Neutral L36 +1 winner margin: +0.063 [+0.021, +0.104], the opposite of
  the Game effect.

### What happens to the ranked choices?

At readout 30 in the Game condition, adding the direction produces a modest
dose-related fall in the original winner's centered logit:

| Dose | Δ rank 1 | Δ rank 2 | Δ rank 3 | Δ rank 4 |
|---:|---:|---:|---:|---:|
| +0.5 | -0.013 | -0.005 | +0.002 | +0.016 |
| +1 | **-0.045** | +0.003 | +0.014 | **+0.028** |
| +2 | **-0.081** | **+0.023** | **+0.036** | +0.021 |

This is leader suppression plus redistribution toward lower-ranked answers. It
is **not runner-up boosting**: at dose +1 the original runner is essentially
unchanged, and at readout 36 it changes by -0.006 while the winner falls by
0.096.

The readout-30 Game effect is direction-specific for the centered winner logit.
Compared directly with the +1 orthogonal control, the feedback direction lowers
rank 1 by an additional 0.047 logits [-0.068, -0.028] and raises rank 4 by an
additional 0.065 [+0.040, +0.091]. But entropy is not direction-specific there:
feedback minus control is -0.0035 nats [-0.0104, +0.0030].

There is also evidence of generic perturbation/nonlinearity. In Game at
readout 30, both +1 and -1 raise entropy and reduce spread. In neutral, -1
compresses the distribution, +0.5 and +1 slightly sharpen it, and +2 compresses
it again. That is not the ordered signed response expected from a single
portable “incorrect-feedback” control axis.

### How many answers actually changed?

- Game L30 +1 changed 1/60 final choices, and that one moved **back to** the
  original baseline winner. Net switching therefore fell by one question.
- Game L30 +2 changed 3/60: one new switch and two returns to the original
  winner. Net switching again fell by one.
- Game L36 +1 changed 4/60: three new switches and one return, for a net gain
  of two switches.
- Game L36 -1 changed 5/60 and reduced switching by three questions.
- Neutral L36 +1 changed 0/60 answers.

The strongest categorical pattern is directionally compatible with control in
Game at readout 36, but it is based on only a handful of boundary-crossing
trials and does not generalize to neutral.

## Assessment against the planned causal criteria

| Criterion | Assessment |
|---|---|
| Add direction to neutral makes it Game-like | **Fails.** No reliable switching; +1 at L36 moves margin and entropy in the opposite direction. |
| Remove direction from Game makes it neutral-like | **Weak/insufficient.** L36 changes continuous margin in the predicted direction and reduces switches by 3/60, but the switch interval includes zero and entropy does not reverse. |
| Add direction to Game amplifies Game behavior | **Partial for logits, not behavior.** Winner suppression is clear at L30/L36; switch amplification is small and uncertain only at L36. |
| Ordered dose and layer response | **Partial.** Game winner suppression is dose-related at L30; neutral is non-monotonic and categorical behavior is not dose-related. |
| Matched orthogonal control is null | **Partial.** It does not reproduce Game winner suppression, but it changes entropy/spread and at least as many categorical answers at L30. |
| Outputs remain valid | **Passes.** Every full-vocabulary top token is A–D. |

Because the pre-specified evidential standard required all six, the overall
causal claim fails.

## Mechanistic interpretation

The result rules out the simplest version of the proposed story: there is no
evidence here for one residual direction that can be added to make a neutral
prompt behave like “your answer was incorrect” and subtracted to undo the Game.

The most plausible interpretation of the small positive result is
context-dependent modulation. The learned direction contains a feature that,
when the rest of the network is already processing a Game prompt, weakly shifts
late computation away from the baseline winner and toward initially lower
options. The same injected vector enters a different local computational state
under the neutral prompt, so downstream nonlinearities do not read it the same
way. Perfect decodability of condition therefore reflects prompt-state identity
more than a standalone behavioral command.

This gives a narrower causal correlate—Game-context leader suppression—but not
an explanation of the much larger natural Game effect. For scale, the natural
Game-minus-neutral winner-margin difference is -1.575 logits; the largest +1
steering effect is -0.138, under 9% as large. The natural switch difference is
23.3 percentage points; the largest +1 steering increase is 3.3 points with an
interval spanning zero.

## Reproducible artifacts

- Full per-scenario estimates and confidence intervals:
  `outputs/causal/qwen36_27b_simplemc_feedback/analysis/steering_effects.csv`
- Machine-readable summary:
  `outputs/causal/qwen36_27b_simplemc_feedback/analysis/steering_summary.json`
- All 1,440 raw scenario/question shards:
  `outputs/causal/qwen36_27b_simplemc_feedback/shards/`
- Run and software metadata:
  `outputs/causal/qwen36_27b_simplemc_feedback/run_metadata.json`
- Prompt audit:
  `outputs/causal/qwen36_27b_simplemc_feedback/prompt_audit.json`
- Direction artifact and discovery diagnostics:
  `artifacts/feedback_direction_qwen36_simplemc.npz`, `.csv`, and `.json`
