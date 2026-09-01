# Qwen3.5-397B-A17B on SimpleMC: observational mechanistic analysis

## Bottom line

This run identifies late, answer-rank-dependent structure, but subsequent
outcome-symmetric, raw-contrast, and flexible nonlinear-compression checks do
not support a distinct runner-promotion mechanism on SimpleMC.

- Broad compression is present in both Second Chance and neutral reruns. The
  feedback-specific increment becomes sustained in the later half of the model.
- A winner-specific Second-Chance-minus-neutral difference becomes sustained at
  readout 44, just when final winner identity becomes strongly decodable.
- On the final block transition (59 to 60), a thresholded current-leader model
  predicts the Second-Chance-versus-baseline update better than proportional
  compression or fixed original-winner identity. The fitted threshold is about
  3.13 pseudo-logits and the above-threshold slope is negative (-0.673).
- A proportional compression model originally attributed the final residual to
  runner-up boosting. A later flexible nonlinear model eliminates that
  runner-specific residual: leave-runner-out Game minus neutral is -0.003,
  95% CI [-0.061, +0.057]. The apparent runner boost was therefore dependent on
  an overly rigid compression model.

All of this is observational. It identifies a trajectory signature and a layer,
not a causal circuit.

## Run and validation

- Model: `Qwen/Qwen3.5-397B-A17B-FP8`
- Dataset: all 500 frozen SimpleMC questions
- Conditions: capabilities baseline, redacted incorrect-feedback game, and
  redacted neutral rerun
- Readouts: embedding plus 60 post-block residuals at the answer-prediction
  position
- Saved records: 500 per condition (1,500 total)
- Native lens: actual final RMSNorm and canonical A-D unembedding rows
- Final-lens agreement audit: mean maximum A-D error 0.0500 logits; worst 0.0685;
  worst relative error 0.00315
- Probe evaluation: five-fold held-out, question-level, letter-balanced

The self-hosted final A-D choices agree with the recorded OpenRouter baseline on
91.8% of questions. Self-hosted change rates are 21.6% in the Game and 10.6% in
neutral, versus 20.8% and 14.8% in the OpenRouter behavioral run. Thus the Game
rate reproduces closely, while neutral is 4.2 points lower locally.

## Ordinary answer emergence

The eventual baseline answer is not meaningfully represented in the native A-D
unembedding basis until late. Its centered logit begins separating around
readouts 40-43 and rises sharply at 44-48.

The held-out logistic probe shows the same late emergence, slightly earlier and
more smoothly:

| Readout | Winner balanced accuracy | Runner-up balanced accuracy |
|---:|---:|---:|
| 40 | 32.9% | 25.4% |
| 42 | 43.7% | 25.7% |
| 43 | 54.8% | 25.2% |
| 44 | 69.7% | 28.8% |
| 48 | 81.4% | 34.8% |
| 52 | 83.1% | 49.7% |
| 60 | 93.4% | 43.5% |

Winner identity therefore becomes reliably linearly decodable at about readout
43-44. Runner-up identity emerges later and remains much less decodable.

## Condition trajectories

The direct all-trial curves show little meaningful separation among conditions
until the late answer-forming layers. At the end:

- Baseline final original-winner advantage is about 1.99 logits.
- Second Chance lowers it to about 1.44 logits.
- Neutral raises it to about 2.35 logits.

The complete A-D spread follows the same ordering: Second Chance is flattest,
neutral is sharpest. This is the activation-level counterpart of the behavioral
compression/entropy result.

The cumulative decomposition refines this:

- Second Chance and neutral both show strong broad compression relative to
  baseline through middle and late layers.
- Second Chance has a sustained additional compression component relative to
  neutral from roughly the mid-30s onward (with a one-layer interruption at 39).
- The Second-Chance-minus-neutral winner penalty becomes continuously positive
  from readout 44 through 60. This begins exactly when the winner probe becomes
  strongly accurate.

This timing is more consistent with answer-dependent control than with only a
fixed generic response to the unusual rerun prompt.

## Direct tests of the three hypotheses

The primary transition analysis predicts each Game-minus-reference block update
out of sample from the state entering that block.

### Final transition, Qwen3.5-397B (59 to 60)

Against baseline:

| Predictor | Held-out R2 versus zero |
|---|---:|
| Proportional compression | 0.017 |
| Fixed original-winner identity | 0.002 |
| Thresholded current leader | **0.102** |

For the thresholded model, the mean selected threshold is 3.13 pseudo-logits.
Its gate coefficient is +0.098, but its excess-margin coefficient is -0.673:
once the current leader is more than about 3.13 logits ahead, a larger lead
predicts a substantially more negative update to that leader.

Against neutral on the same transition, thresholded current leader also gives
the best of these three fits, though less decisively: R2 = 0.157 versus 0.107 for
compression and 0.118 for fixed original-winner identity.

### Contrast with Qwen3.6-27B

On the smaller model's final transition (63 to 64), compression remains the
strongest of these three signatures:

| Predictor | Qwen3.6-27B held-out R2 |
|---|---:|
| Proportional compression | **0.274** |
| Fixed original-winner identity | 0.194 |
| Thresholded current leader | 0.125 |

This final-transition regression is a difference in relative model fit, but it
is not by itself decisive evidence of a distinct mechanism. The outcome-split
trajectory analysis below shows that the larger model's most robust late
residual effect follows alternative-option evidence rather than a sustained
penalty to the baseline winner.

### Runner-up boost versus leader suppression

**Superseded qualification:** the estimates in this subsection use a
proportional compression model. The nonlinear reanalysis below shows that the
runner-specific component disappears when score-dependent compression is
modeled flexibly.

After cross-fitted removal of option-letter effects and proportional baseline
compression, the final readout has:

- Game runner-up boost: +0.293 logits, 95% CI [+0.227, +0.352]
- Game leader-suppression contrast: -0.099, 95% CI [-0.169, -0.031]

Here a positive leader-suppression contrast would mean the winner fell below
ranks 3-4. The negative value means the original winner is still relatively
enhanced at the final readout; it is simply much less enhanced than in neutral.
The runner-up receives the clear selective gain.

At readouts 56-59, however, the cumulative decomposition does show a significant
Game-versus-baseline original-winner penalty. The most defensible temporal story
is therefore:

1. generic rerun-related compression develops in both rerun conditions;
2. feedback adds extra compression;
3. once a strong leader is readable, late blocks apply a lead-sensitive negative
   update;
4. the final output also selectively boosts the runner-up.

The residual perturbation is not wholly structured: Game has more unexplained
residual RMS than neutral at many late readouts, and winner-runner energy is only
modestly above the one-third isotropic expectation. The mechanism is therefore
best described as **compression plus a late structured reranking component**, not
as purely surgical winner suppression.

## Switch-stratified descriptive result

Local Game switches occur on 108/500 questions. Both switch and non-switch
questions show late original-winner suppression relative to baseline, but switch
questions show a much stronger late runner-up gain. This is descriptive only;
conditioning on the outcome cannot establish the mechanism.

## Outcome-defined diagnostic split

To distinguish promotion of the fixed baseline runner-up from accumulation for
the option that ultimately wins, the compression-residual trajectories were
split into (a) Game no-switch trials and (b) Game switches to baseline rank 3
or 4. The 89 switches to the baseline runner-up are not diagnostic because both
hypotheses predict a runner-up gain there.

On the 392 no-switch trials, the baseline runner-up still rises relative to
ranks 3-4 in Game by +0.214 logits, 95% paired-bootstrap CI
[+0.149, +0.281]. The same contrast in Neutral is +0.010
[-0.050, +0.070], and the paired Game-minus-Neutral contrast is +0.205
[+0.138, +0.273]. Thus runner-up privilege is not confined to trials where the
runner-up becomes the final answer.

On the 19 switches to baseline rank 3 or 4, the Game-chosen lower option rises
above the baseline runner-up by +1.043 logits [+0.647, +1.509]. The same
identity contrast under Neutral on those questions is +0.290
[-0.059, +0.638]; the paired Game-minus-Neutral contrast is +0.753
[+0.293, +1.272]. Although this group is small and outcome-conditioning makes
the final endpoint partly tautological, the chosen lower option begins
separating in earlier late-layer readouts as well.

A symmetric outcome check changes the interpretation of the lower-switch
result. On the seven trials where Neutral selects baseline rank 3 or 4, the
Neutral-selected option exceeds the baseline runner-up by +1.056 logits under
Neutral versus +0.418 under Game. The paired Neutral-minus-Game contrast is
+0.638 [+0.311, +0.970]. This mirrors the Game-defined split: whichever
condition defines the selected lower option shows the larger late trajectory.
That symmetry means the selected-option result is substantially driven by
outcome conditioning and ordinary accumulation for the eventual choice; it is
not specific evidence for a Game mechanism.

The remaining Game-specific result is confined to the fitted compression
residual on trials where both conditions keep the baseline winner (n=384).
There the residual baseline runner-up-minus-lower contrast is +0.212
[+0.144, +0.278] in Game and +0.001 [-0.058, +0.060] in Neutral.
However, this is not a raw runner-up increase. Before subtracting the fitted
compression component, Game minus baseline changes the runner-up-minus-lower
margin by -0.032 [-0.100, +0.036], while Neutral minus baseline changes it by
+0.175 [+0.116, +0.234]; raw Game minus Neutral is -0.207
[-0.277, -0.137]. The positive residual therefore means that the runner-up is
preserved relative to what the proportional Game-compression fit predicts, not
that Game literally raises it relative to the lower options. This dependence
on the compression model makes the signature suggestive rather than a secure
mechanistic target.

## Subsequent nonlinear-compression reanalysis

A rank-agnostic cubic-spline model was fitted in held-out question folds using
baseline option score, interactions with winner margin and A-D spread, and
option-letter effects. It was then refitted while excluding every runner-up
observation from training.

- Leave-runner-out Game minus neutral: -0.003 [-0.061, +0.057]
- Score-and-geometry matched runner versus lower options: +0.056
  [+0.002, +0.112]
- Matching sensitivities range from -0.003 to +0.100; several wider-support
  intervals include zero

The leave-runner-out result remains essentially zero across ridge penalties and
four-to-eight-knot spline specifications. Therefore SimpleMC does not contain a
robust runner-specific exception after allowing nonlinear score-dependent
compression. The same analysis on PopMC does: leave-runner-out +0.180
[+0.095, +0.265], with score-and-geometry matching +0.289
[+0.200, +0.375]. This makes PopMC, rather than SimpleMC, the appropriate
dataset for searching for a runner-preserving mechanism.

However, the switch-predictive analysis reveals a narrower cross-dataset
pattern. On the 52 SimpleMC Game-only switches to the baseline runner, the
pre-output leave-runner-out signal adds +0.054 AUC [+0.017, +0.092] beyond
baseline and same-layer Neutral controls; nested selection chose readout 52 in
all five folds. It does not predict the 11 switches to ranks 3/4 (incremental
AUC 0.000 [-0.032, +0.029]). Therefore SimpleMC lacks a population-average
runner exception on no-switch trials but does show the same runner-specific
trajectory on trials where the Game actually selects the runner. This may be an
implementation of the emerging alternative answer rather than the upstream
control signal.

Artifacts: `analysis/compression_robustness.{png,svg,csv,json}` and
`analysis/runner_predictive_bridge.{png,svg,csv,json}`.

## Cost and instance state

The stopped B200 instance is preserved with the 397B checkpoint. Its stopped
storage price is $0.143/hour (about $3.44/day), so it should be retained only if
near-term follow-up is likely.

The Vast ledger reports $32.72 for the B200 instance. Adding the attributable
H200 ($3.90), RTX ($1.16), and estimated current-session share of the reused A100
attempt ($2.79) gives approximately **$40.57** for this batch. This exceeded the
$20 authorization. The cost estimate shown during collection was wrong because
the post-restart `duration` field did not represent cumulative billed GPU time;
the invoice ledger does. This should have been checked before continuing.

## Main artifacts

- `analysis/paper_figures/answer_emergence_combined.png`
- `analysis/paper_figures/all_trials_mechanism_summary.png`
- `analysis/cumulative_hypothesis_decomposition.png`
- `analysis/perturbation_decomposition.png`
- `analysis/leader_suppression_vs_runner_boost.png`
- `analysis/outcome_split_residual_trajectories.png`
- `analysis/outcome_split_residual_trajectories.csv`
- `../comparisons/qwen397_vs_qwen27_simplemc/symmetric_outcome_comparison.png`
- `../comparisons/qwen397_vs_qwen27_simplemc/symmetric_outcome_comparison.csv`
- `analysis/paper_figures/incorrect_switch_stratified_delta.png`
- `analysis/hypothesis_fits.csv`
- `analysis/cumulative_hypothesis_fits.csv`
- `analysis/cumulative_hypothesis_coefficients.csv`
- `analysis/perturbation_and_accumulation_values.csv`
- `analysis/logistic_probe_results.csv`
