# Cumulative condition-minus-baseline hypothesis analysis

This analysis compares each follow-up condition directly with the baseline
capabilities test at every residual-stream readout. It was added because a
Game-minus-neutral transition analysis makes both the outcome and the reference
a difference score and can obscure what each follow-up prompt is doing relative
to ordinary answer generation.

## Estimand

For question \(q\), option \(a\), and readout \(l\), the two outcomes are

\[
D^{G}_{qal}=z^{G}_{qal}-z^{B}_{qal},\qquad
D^{N}_{qal}=z^{N}_{qal}-z^{B}_{qal}.
\]

The \(z\)'s are A-D pseudo-logits after subtracting their within-question A-D
mean. Thus the analysis concerns relative answer strength; a common shift of all
four logits is not identifiable from these data and is irrelevant to the answer
distribution.

Both outcomes use exactly the same baseline geometry and the same definition of
the original winner: the highest-logit option at the final baseline readout.
Questions are inverse-weighted by the frequency of that winner letter, so A, B,
C, and D each contribute one quarter of the estimand.

## Models

Every model includes three option-letter nuisance terms. The substantive
predictors are:

- **Compression:** the four centered baseline pseudo-logits at the same readout.
  The reported compression strength is the negative regression coefficient. A
  value of zero means no proportional contraction and a value of one means
  complete removal of the baseline A-D geometry.
- **Original-winner penalty:** a centered one-hot indicator for the final
  baseline winner. Its reported value is also sign-reversed, so positive values
  mean that the original winner is selectively reduced relative to each other
  option, after accounting for compression and option-letter effects.
- **Thresholded current leader:** a centered one-hot indicator for the option
  leading in the baseline at that readout, gated by whether its lead exceeds a
  threshold, plus a hinge term for the amount by which the margin exceeds the
  threshold. The threshold is selected inside each training fold.

The held-out comparisons are letter-only, compression, winner, compression plus
winner, thresholded current leader, and a full model containing all terms.
Five-fold cross-validation is stratified by original-winner letter. Coefficient
intervals use 300 question-clustered, winner-letter-stratified bootstrap
samples.

## Current SimpleMC result

The direct baseline comparisons change the emphasis of the story:

1. **Broad compression is the main stable Game-minus-baseline signature.** At
   the final readout its strength is 0.440 (95% CI 0.401 to 0.472). Compression
   alone explains 0.679 held-out \(R^2\) beyond option-letter effects.
2. **There is a transient, very late original-winner penalty in the Game.** It
   is reliably positive at readouts 61-63 and peaks at 0.438 (0.289 to 0.577) at
   readout 63. It largely disappears in the exact final logits: 0.052 (-0.021 to
   0.135) at readout 64. Adding winner identity to compression adds only 0.011
   held-out \(R^2\) at its late maximum and approximately zero at the final
   readout.
3. **Neutral is not simply an attenuated version of Game.** Its final
   compression is approximately zero, 0.038 (-0.001 to 0.070), while its final
   winner term is negative, -0.333 (-0.441 to -0.212): relative to baseline,
   neutral selectively *amplifies* the original winner at that readout.
4. **The thresholded-leader hypothesis is not favored.** Once compression and
   original-winner identity are included, the threshold terms add little
   held-out fit in the Game (at most 0.011 \(R^2\) over readouts 61-64, and 0.005
   at the final readout). Their larger late additions occur in neutral and do
   not have a stable suppression sign.

This is observational evidence about trajectory shape, not a causal circuit
claim. In particular, the last few readouts are strongly transformed, and
baseline geometry and winner identity become correlated there. The held-out
incremental fits are therefore more trustworthy for choosing among hypotheses
than any single late-layer coefficient.

## Compression residual: is the model also adding noise?

The follow-up analysis cross-fits the compression-plus-option-letter model and
defines the remaining within-question A-D vector as the perturbation. This is a
deterministic residual at temperature zero; “noise” describes its behavior
across questions, not an internal random-number generator.

The user's proposed timing is supported, but the effect is modest:

- At readout 30, perturbation RMS is 0.105 in the Game and 0.094 in neutral. The
  paired Game-minus-neutral difference is 0.0114 natural-logit units (95% CI
  0.0072 to 0.0154). The paired interval is positive continuously at readouts
  24-31 and again at 34-36.
- At readout 30, the perturbation has no reliable runner-up direction: the
  runner-up-minus-winner projection is -0.004 (-0.016 to 0.010).
- Roughly one third of perturbation energy lies in the winner-runner direction,
  close to the 1/3 expectation for an isotropic vector in the three-dimensional
  centered A-D subspace. This remains approximately true across most readouts.
- Directional structure appears only late in pointwise intervals. After a
  studentized max-|t| correction over all 65 readouts, however, no positive Game
  leader-suppression or runner-boost effect survives. Neutral is more robust:
  relative leader amplification survives at readouts 56, 57, and 64, and
  relative runner suppression survives at 60-63.

Thus the mid-layer signature is best described as **extra generic perturbation
plus stronger compression**, not early targeted suppression or runner-up
boosting. A plot of individual block updates projected onto the current baseline
answer geometry shows no sustained Game-specific negative write. That does not
identify active cancellation: the cumulative effect may be implemented as a
distributed reduction of answer-evidence gain, or the raw logit-lens coordinate
system may rotate too much between adjacent layers for the transition projection
to be decisive.

The paired Game-minus-neutral perturbation RMS result is statistically stronger
than the rank-direction result. A simultaneous 95% band over all 65 readouts
remains positive at 24-31 and 34-36 (at readout 30: 0.0114, simultaneous CI
0.0047 to 0.0182). This establishes a small condition-specific difference in
the chosen residual metric, not a large effect or a uniquely identified neural
mechanism.

The perturbation analysis is implemented in
`mechanistic/perturbation_analysis.py`; its CSV, JSON, PNG, and SVG outputs are
under `outputs/mechanistic/qwen36_27b_simplemc/analysis/paper_figures/perturbation`.

## Reproduction

Run:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/secondchance-mpl \
python3 -m mechanistic.cumulative_hypothesis_analysis \
  --input outputs/mechanistic/qwen36_27b_simplemc \
  --output outputs/mechanistic/qwen36_27b_simplemc/analysis/paper_figures/cumulative \
  --bootstrap 300
```

The output directory contains the complete coefficient and held-out-fit tables,
a JSON summary, and PNG/SVG versions of both figures.
