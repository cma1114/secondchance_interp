# Qwen3.6-27B SimpleMC–TriviaMC probe transfer

## Design

- Model: `Qwen/Qwen3.6-27B`, revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Datasets: 500 frozen SimpleMC Baseline questions and 500 frozen TriviaMC
  Baseline questions.
- Readout: the 5,120-dimensional residual stream at the final prompt position,
  from the embedding through all 64 blocks.
- Target: the model's native full-vocabulary greedy A-D output letter.
- Probe: standardized four-class centroid probe, matching the candidate probe
  used in the SimpleMC trajectory analysis.
- Within-dataset evaluation: five-fold held-out, stratified by output letter.
- Cross-dataset evaluation: train on all 500 questions from one dataset and
  apply unchanged to all 500 questions from the other dataset.

The TriviaMC residual collection used the exact prior Baseline prompts: prompt
hashes agree on 500/500 trials with the prior logit-lens collection, and native
output tokens agree on 499/500 trials.

Baseline output-letter counts were:

| Dataset | A | B | C | D |
|---|---:|---:|---:|---:|
| SimpleMC | 256 | 72 | 98 | 74 |
| TriviaMC | 164 | 105 | 111 | 120 |

## Predictive transfer

Letter-balanced accuracy was:

| Readout | Simple within | Trivia within | Simple → Trivia | Trivia → Simple |
|---:|---:|---:|---:|---:|
| 40 | 31.4% | 39.7% | 26.0% | 36.2% |
| 44 | 41.5% | 58.9% | 34.6% | 45.0% |
| 48 | 66.3% | 78.8% | 51.0% | 65.7% |
| 52 | 79.0% | 87.8% | 61.1% | 78.5% |
| 56 | 80.5% | 89.8% | 68.3% | 80.9% |
| 60 | 81.4% | 89.9% | 70.3% | 78.1% |
| 64 | 76.7% | 90.9% | 64.1% | 80.7% |

Chance is 25%. Transfer is weak before the late answer-emergence phase, rises
sharply at readouts 44-52, and remains strong thereafter. The TriviaMC probe
transfers particularly well: at readout 56 it predicts SimpleMC slightly better
than the SimpleMC five-fold estimate itself (80.9% versus 80.5%).

SimpleMC-to-TriviaMC transfer is weaker than the reverse direction but still far
above chance. The asymmetry is consistent with estimation quality: SimpleMC has
256 A outputs but only 72-98 examples for each other letter, whereas TriviaMC is
substantially more balanced. It is not evidence that the two datasets use
different late answer representations.

## Direction similarity

For each fitted probe, a letter direction was defined as that letter's raw-space
weight minus the mean of all four letter weights. This removes the arbitrary
common offset of a multiclass linear decoder. Corresponding SimpleMC and TriviaMC
directions were compared with cosine similarity. Split-half fits within each
dataset provide a sampling-noise ceiling.

At readout 56:

| Direction | Cross-dataset cosine | Noise ceiling | Fraction of ceiling |
|---|---:|---:|---:|
| A versus rest | 0.867 | 0.967 | 89.7% |
| B versus rest | 0.889 | 0.962 | 92.4% |
| C versus rest | 0.860 | 0.948 | 90.7% |
| D versus rest | 0.908 | 0.964 | 94.2% |

The mean matched-letter cosine is 0.881, while the mean mismatched-letter cosine
is -0.290. Thus the similarity is letter-specific rather than generic alignment
between the two datasets' residual distributions.

At readout 48 the mean matched cosine is already 0.789; at readout 64 it is
0.816. The strongest shared geometry occurs around readouts 52-60, coinciding
with the period in which answer identity becomes reliably decodable.

## Conclusion

The late centroid probes are not primarily exploiting dataset-specific semantic
correlates. They recover a substantially shared, letter-specific answer geometry
across SimpleMC and TriviaMC. In particular, the SimpleMC and TriviaMC A probes
are highly similar in the late layers.

The result does not rescue early probe trajectories: before approximately
readout 44, predictive transfer is near chance and the directions are much less
stable. Nor does it show that a Baseline-trained probe must transfer unchanged to
Game or Neutral, whose prompts induce a larger representational distribution
shift than changing the question dataset.

## Artifacts and compute

- Canonical figure:
  `outputs/mechanistic/qwen36_27b_cross_dataset_probe_transfer/preserved_figures/cross_dataset_probe_transfer.png`
- Layerwise values:
  `outputs/mechanistic/qwen36_27b_cross_dataset_probe_transfer/cross_dataset_probe_transfer.csv`
- Selected-layer summary:
  `outputs/mechanistic/qwen36_27b_cross_dataset_probe_transfer/cross_dataset_probe_transfer_summary.json`
- TriviaMC residuals:
  `outputs/mechanistic/qwen36_27b_triviamc_baseline_residuals/`
- Analysis:
  `mechanistic/cross_dataset_probe_transfer.py`

The baseline-only collection used approximately 0.15 A100-hours. Including the
brief transfer restart, compute cost was approximately $0.20 at $1.0389/hour.
Vast instance 46566562 was stopped and preserved after retrieval.
