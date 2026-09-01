# Qwen3.6-27B pooled answer-probe results

## Design

- Datasets: 500 SimpleMC Baseline questions and 500 TriviaMC Baseline questions.
- Labels: the model's actually generated Baseline letter, not the reconstructed
  maximum among four canonical token logits.
- Probe: a dataset-centered four-class centroid probe at every residual readout.
  Letter centroids were estimated separately within each dataset and then
  averaged with equal dataset weight, preventing SimpleMC's A-heavy output
  distribution from dominating the common directions.
- Evaluation: five-fold cross-fitting. Each fold trained on approximately 800
  Baseline questions and held out approximately 200. A held-out question's
  Baseline, Second Chance, and Neutral activations were all scored by a probe
  that had not seen that question in either dataset.
- Candidate scores were centered within question and expressed in units of the
  pooled held-out Baseline score dispersion.

## Results

At the final residual readout, letter-balanced Baseline accuracy was:

| Dataset | Separate within-dataset probe | Pooled cross-fitted probe |
|---|---:|---:|
| SimpleMC | 80.5% | 81.0% |
| TriviaMC | 89.8% | 89.7% |

Pooling therefore stabilizes a shared answer-letter coordinate system without
creating an artificial accuracy increase. On SimpleMC, the pooled probe's
final-layer leader matched the eventual generated output on 50.7% of Second
Chance trials and 61.6% of Neutral trials (letter-balanced accuracy), confirming
that the Baseline representation transfers imperfectly to both redo conditions
and especially poorly to Second Chance.

The pooled trajectories reproduce the qualitative late-layer result from the
SimpleMC-only probe. At readout 56, the prior-answer advantage was +1.00 in
Baseline, +0.63 in Neutral, and only +0.22 in Second Chance. At the final
readout it was +0.84, +0.42, and -0.06, respectively. Thus the Game-specific
loss of the prior answer is not an artifact of estimating the probe from only
500 SimpleMC questions.

Before approximately readout 48, held-out answer accuracy is too low to treat
the apparent candidate trajectories as answer identity. The early curves are
included so that this reliability boundary is visible, not as evidence of
early winner or runner-up representations.

## Artifacts

- Canonical figure:
  `outputs/mechanistic/qwen36_27b_pooled_probe/preserved_figures/pooled_probe_trajectories.png`
- Cross-fitted scores:
  `outputs/mechanistic/qwen36_27b_pooled_probe/pooled_cross_fitted_probe_scores.npz`
- Accuracy table:
  `outputs/mechanistic/qwen36_27b_pooled_probe/pooled_probe_accuracy.csv`
- Trajectory table:
  `outputs/mechanistic/qwen36_27b_pooled_probe/pooled_probe_trajectories.csv`
- Summary:
  `outputs/mechanistic/qwen36_27b_pooled_probe/pooled_probe_summary.json`
- Analysis code: `mechanistic/pooled_probe_analysis.py`
