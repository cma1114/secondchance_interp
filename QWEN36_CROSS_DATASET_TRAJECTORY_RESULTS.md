# Qwen3.6-27B matched SimpleMC and TriviaMC trajectories

## Design

- Model revision: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- 500 frozen questions per dataset.
- Conditions: Baseline, Second Chance, and corrected Neutral.
- Readout: aggregated canonical A-D pseudo-logits at the final prompt position,
  from the embedding through all 64 blocks.
- Original winner and ranks are defined by each question's final Baseline A-D
  ordering. Means and 95% intervals give original-winner A/B/C/D equal weight.
- TriviaMC used batch size 1, matching the original SimpleMC collection. All
  1,500 TriviaMC shards passed the existing combined absolute/relative
  final-lens validation.

## Result

The two datasets have the same qualitative timing. A-D answer identity is
barely output-readable until approximately readouts 48-50. The three
conditions diverge primarily after that point, with the largest Second Chance
loss of original-winner advantage and A-D spread occurring in readouts 60-64.

Selected late values are:

| Dataset | Metric | Readout | Baseline | Second Chance | Neutral |
|---|---|---:|---:|---:|---:|
| SimpleMC | Winner advantage | 63 | 2.114 | 0.179 | 1.901 |
| SimpleMC | Winner advantage | 64 | 1.368 | 0.374 | 1.590 |
| SimpleMC | A-D spread | 64 | 1.385 | 1.062 | 1.544 |
| TriviaMC | Winner advantage | 63 | 5.257 | 2.891 | 5.670 |
| TriviaMC | Winner advantage | 64 | 5.526 | 2.059 | 5.157 |
| TriviaMC | A-D spread | 64 | 3.236 | 1.878 | 3.041 |

Thus Mixer 62 and MLP 63 lie inside a major late answer-finalization phase,
not after an already stable final answer. The earlier statistically detectable
compression around readouts 27-30 occurred while absolute A-D separation was
very small. It should not be described as the main visible answer-choice
transformation.

The timing does not show that the Game instruction is first interpreted in
these late components. It shows that the identified causal mediators execute or
finalize their answer-distribution effects at the same stage when answer
identity becomes strongly output-readable.

## Artifacts

- Matched summary figure and values:
  `outputs/mechanistic/qwen36_27b_cross_dataset_trajectories/`
- TriviaMC all-rank figures and values:
  `outputs/mechanistic/qwen36_27b_triviamc_trajectories/analysis/paper_figures/`
- SimpleMC all-rank figures and values:
  `outputs/mechanistic/qwen36_27b_simplemc_clean/analysis/paper_figures/`
- Figure generator: `mechanistic/cross_dataset_trajectories.py`.

The TriviaMC collection used approximately 0.46 A100-hours, about $0.48 at the
instance's $1.0389/hour running rate. Vast instance 46566562 was stopped and
preserved after artifact retrieval.
