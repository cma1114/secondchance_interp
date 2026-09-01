# TriviaMC strategic-switching replication — Step 1

## Design

The same pinned Qwen3.6-27B model was run on all 500 questions in the existing difficulty-filtered TriviaMC manifest. The first presentation is unchanged. In the second presentation every answer content moves to a different A-D letter using a frozen balanced derangement. Game and Neutral differ only at `incorrect` versus `lost` in `Your answer was ... Choose the answer again.` Unrestricted top-token choices are primary; aggregated A-D logits provide the continuous rank analysis.

## Validation

- Questions: 500; discovery: 250; confirmation: 250.
- Baseline unrestricted A-D fraction: 100.0%.
- Baseline aggregated/unrestricted agreement: 99.2%.
- All 500 paired prompts differ only at `incorrect`/`lost`: True.
- All logits finite: True.

## Behavioral result

| Split | Game semantic switch | Neutral semantic switch | Game − Neutral | Game old-letter avoidance | Neutral old-letter avoidance | Game − Neutral |
|---|---:|---:|---:|---:|---:|---:|
| Discovery | 31.6 [26.0, 37.2]% | 22.8 [17.6, 28.4]% | +8.8 [+4.8, +12.8] pp | 84.8 [80.4, 89.2]% | 88.0 [84.0, 91.6]% | -3.2 [-6.4, +0.0] pp |
| Confirmation | 32.0 [26.4, 37.6]% | 26.0 [20.8, 31.2]% | +6.0 [+2.0, +10.4] pp | 86.0 [81.6, 90.0]% | 90.0 [86.0, 93.6]% | -4.0 [-7.2, -0.8] pp |
| All | 31.8 [27.8, 35.8]% | 24.4 [20.8, 28.2]% | +7.4 [+4.6, +10.4] pp | 85.4 [82.4, 88.4]% | 89.0 [86.2, 91.8]% | -3.6 [-6.0, -1.2] pp |

## First-to-final old-rank transformation

The tables below use the untouched confirmation half. Candidates are ordered by their first-presentation aggregated A-D logits. Raw changes preserve the common four-logit shift; centered changes remove it within question.

| Old rank | First raw logit | Game final raw logit | Game raw change | Game centered change | Neutral final raw logit | Neutral raw change | Neutral centered change |
|---|---:|---:|---:|---:|---:|---:|---:|
| W1 | +28.354 [+27.915, +28.786] | +23.721 [+23.480, +23.961] | -4.633 [-4.968, -4.293] | -2.856 [-3.141, -2.577] | +24.590 [+24.334, +24.851] | -3.764 [-4.089, -3.446] | -2.148 [-2.415, -1.885] |
| W2 | +22.738 [+22.455, +23.025] | +20.748 [+20.439, +21.043] | -1.990 [-2.246, -1.724] | -0.212 [-0.423, +0.008] | +20.698 [+20.335, +21.050] | -2.040 [-2.336, -1.750] | -0.424 [-0.668, -0.179] |
| W3 | +21.471 [+21.205, +21.744] | +20.705 [+20.427, +20.977] | -0.766 [-1.055, -0.479] | +1.012 [+0.786, +1.243] | +20.635 [+20.298, +20.962] | -0.836 [-1.135, -0.533] | +0.780 [+0.512, +1.053] |
| W4 | +20.317 [+20.045, +20.591] | +20.596 [+20.322, +20.862] | +0.279 [-0.012, +0.575] | +2.056 [+1.821, +2.299] | +20.493 [+20.175, +20.803] | +0.176 [-0.133, +0.488] | +1.792 [+1.539, +2.064] |

### Mean per-question A-D probabilities

| Old rank | First presentation | Game final | Neutral final |
|---|---:|---:|---:|
| W1 | +82.0 [+79.3, +84.7]% | +62.2 [+58.0, +66.4]% | +67.3 [+63.0, +71.5]% |
| W2 | +11.4 [+9.7, +13.1]% | +16.6 [+13.7, +19.6]% | +15.5 [+12.6, +18.7]% |
| W3 | +4.6 [+3.8, +5.5]% | +11.9 [+9.7, +14.2]% | +10.1 [+7.9, +12.3]% |
| W4 | +2.0 [+1.5, +2.5]% | +9.3 [+7.7, +11.0]% | +7.1 [+5.7, +8.7]% |

### Aggregated A-D choice robustness

| Split | Game semantic switch | Neutral semantic switch | Game − Neutral | Game old-letter avoidance | Neutral old-letter avoidance | Game − Neutral |
|---|---:|---:|---:|---:|---:|---:|
| Discovery | 32.0 [26.4, 37.6]% | 21.6 [16.8, 26.8]% | +10.4 [+6.4, +14.8] pp | 84.4 [80.0, 88.8]% | 88.8 [84.8, 92.4]% | -4.4 [-8.0, -1.2] pp |
| Confirmation | 31.6 [26.0, 37.6]% | 26.8 [21.6, 32.4]% | +4.8 [+0.8, +8.8] pp | 85.6 [81.2, 90.0]% | 89.6 [85.6, 93.2]% | -4.0 [-7.6, -0.8] pp |
| All | 31.8 [27.8, 35.8]% | 24.2 [20.4, 28.0]% | +7.6 [+4.6, +10.6] pp | 85.0 [82.0, 88.0]% | 89.2 [86.4, 91.8]% | -4.2 [-6.6, -1.8] pp |

## Conclusion

On the untouched confirmation half, Game leaves the semantic first-presentation winner on 32.0 [26.4, 37.6]% of questions versus 26.0 [20.8, 31.2]% in Neutral, a paired Game-minus-Neutral difference of +6.0 [+2.0, +10.4] percentage points. The same direction appears on discovery and under the secondary aggregated A-D choice rule.

The literal old-letter result goes the other way: Game-minus-Neutral old-letter avoidance is -4.0 [-7.2, -0.8] points on confirmation. The extra Game switching therefore follows the earlier winner's semantic content after it moves, not its old A-D character.

The continuous final evidence is also rank-shaped rather than an equal perturbation to all four candidates. Relative to Neutral, Game's centered final evidence for W1/W2/W3/W4 is **-0.708 [-0.803, -0.613] / +0.212 [+0.117, +0.308] / +0.231 [+0.150, +0.313] / +0.265 [+0.184, +0.346] logits** on confirmation. Game also has +0.131 [+0.099, +0.163] more bits of A-D entropy. Thus TriviaMC reproduces the qualitative behavioral target: Game is more uncertain overall, but the task difference is specifically concentrated in suppressing the semantic old winner and redistributing relative evidence toward all three alternatives. The effect is smaller behaviorally than the canonical SimpleMC remapping gap, so later causal steps should be treated as a cross-dataset replication rather than assumed to have the original magnitude.

## Interpretation boundary

Step 1 is behavioral and descriptive. A positive semantic-switch difference together with a rank-shaped Game transformation argues against equal undirected candidate noise, but it does not establish causal recollection. That question belongs to the gated Step 2 matching-versus-wrong history blockade.

Canonical figure: `figures/qwen36_triviamc_strategic_replication_step1.png`.
