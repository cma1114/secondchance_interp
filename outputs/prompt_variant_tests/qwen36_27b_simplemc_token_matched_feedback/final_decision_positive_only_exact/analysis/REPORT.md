# Positive-only W1 semantic ablation at the final decision position

At every post-block readout, the intervention subtracts `max(h · v_W1, 0) v_W1`. Negative projections are left untouched. This distinguishes removing positive W1-aligned activation from the earlier signed projection-zeroing intervention, which moved negative projections toward W1.

## Frozen confirmation

| Subset | Condition | Natural W1 | Positive-only W1 | W1 change (95% CI) | Natural W2 | Positive-only W2 | Centered W1-logit change (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|
| Conflict (n=136) | Game | 19.1% | 25.0% | +5.88 [+1.47, +11.03] pp | 45.6% | 39.7% | +0.07 [+0.02, +0.12] |
| Conflict (n=136) | Neutral | 39.7% | 43.4% | +3.68 [-0.74, +8.82] pp | 38.2% | 33.8% | -0.04 [-0.10, +0.01] |
| Agreement (n=113) | Game | 53.1% | 51.3% | -1.77 [-7.96, +4.42] pp | 53.1% | 51.3% | -0.06 [-0.13, +0.01] |
| Agreement (n=113) | Neutral | 73.5% | 68.1% | -5.31 [-10.62, -0.88] pp | 73.5% | 68.1% | -0.16 [-0.23, -0.09] |
| All (n=249) | Game | 34.5% | 36.9% | +2.41 [-1.61, +6.43] pp | 49.0% | 45.0% | +0.01 [-0.03, +0.05] |
| All (n=249) | Neutral | 55.0% | 54.6% | -0.40 [-3.61, +2.81] pp | 54.2% | 49.4% | -0.09 [-0.14, -0.05] |

## Game-specific effects

Here a positive W1-selection value means that the intervention reduces the natural Neutral-minus-Game W1-selection gap: Game moves toward W1 more than Neutral does. This is the behavioral quantity relevant to the semantic-suppression hypothesis.

| Subset | Reduction in Neutral-minus-Game W1-selection gap (95% CI) | Game-minus-Neutral centered-W1 logit change (95% CI) |
|---|---:|---:|
| Conflict (n=136) | +2.21 [-3.68, +8.09] pp | +0.11 [+0.08, +0.15] |
| Agreement (n=113) | +3.54 [-3.54, +10.62] pp | +0.10 [+0.05, +0.16] |
| All (n=249) | +2.81 [-2.01, +7.63] pp | +0.11 [+0.07, +0.14] |

On the frozen discovery conflict subset, the corresponding reduction in the W1-selection gap was +1.46 [-4.38, +7.30] percentage points, and the centered-W1 logit contrast was +0.04 [-0.00, +0.07]. Thus the behavioral effect is small and uncertain in both splits, while the larger confirmation logit effect does not reproduce at comparable magnitude in discovery.

## Positive-projection dose at readout 64

| Subset | Condition | Natural positive trials | Natural mean positive projection | Intervention positive trials | Mean positive projection removed |
|---|---|---:|---:|---:|---:|
| Conflict | Game | 58.8% | 14.07 | 32.4% | 6.07 |
| Conflict | Neutral | 58.1% | 15.21 | 35.3% | 6.39 |
| Agreement | Game | 59.3% | 14.80 | 27.4% | 5.25 |
| Agreement | Neutral | 67.3% | 16.75 | 28.3% | 5.45 |
| All | Game | 59.0% | 14.40 | 30.1% | 5.70 |
| All | Neutral | 62.2% | 15.91 | 32.1% | 5.96 |

## Validation

The natural companion was compared question-by-question with the previous exact historical run across all 500 questions. Natural A-D logits, all 64 W1 projection readouts, and all 64 residual norms were bit-for-bit identical (maximum absolute difference 0.0 for every array).

## Data files

- `../all/results.npz` is the canonical raw result. Its condition axis is `[Game, Neutral]`. It contains 500 question IDs; natural and positive-only A-D logits with shape `[2, 500, 4]`; and natural projections, natural residual norms, intervention projections before removal, intervention residual norms, and projections after removal with shape `[2, 500, 64]`.
- `../all/run_metadata.json` records the exact model revision, prompt and serialization configuration, answer-token IDs, vector definition, intervention, package versions, and host platform.
- `../data/per_question_condition.csv` is the human-readable trial table: one row for each of 500 questions times two conditions. It gives the frozen split, W1, W2, natural and intervened answers, correctness, entropy, W1-centered logits, W1-vs-W2 margins, and all four logits both in displayed-letter and original-semantic-content coordinates.
- `../data/per_question_condition_layer.csv` is the human-readable layer table: one row for each question times condition times 64 readouts. It gives the natural W1 projection and residual norm, the live intervention projection before and after removal, the intervention residual norm, and the positive projection removed at that readout.
- `summary.json` contains the discovery and confirmation aggregates and letter-stratified bootstrap confidence intervals used in this report.

Emitted answers are determined by taking the A-D argmax in displayed second-presentation letter coordinates and only then mapping that letter back to original semantic content. This preserves the model's actual A-before-B-before-C-before-D tie-break on exact aggregated-logit ties. Taking argmax after semantic reordering would silently change the answer on 15 of the 2,000 natural or intervened condition rows; the exported CSV and this report use the displayed-letter tie-break.
