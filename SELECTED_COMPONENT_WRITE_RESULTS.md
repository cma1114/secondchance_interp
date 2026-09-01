# Qwen3.6-27B selected-component natural writes

## Estimand

For each SimpleMC question and condition, the immediate A-D write of Mixer 62
is the centered pseudo-logit vector immediately before MLP 62 minus the vector
immediately before Mixer 62. The immediate write of MLP 63 is the centered
pseudo-logit vector after MLP 63 minus the vector immediately before MLP 63.

The four entries are reordered by that question's final Baseline A-D ranking
before aggregation. Reported means and question-bootstrap 95% intervals give
Baseline-winner A/B/C/D strata equal weight. Thus the result is four directly
interpretable answer-strength changes, not an average raw residual vector.

## Results

### Mixer 62

| Condition | Winner | Runner-up | Rank 3 | Rank 4 |
|---|---:|---:|---:|---:|
| Baseline | -0.141 [-0.170, -0.113] | -0.007 [-0.051, +0.033] | +0.043 [+0.000, +0.086] | +0.105 [+0.065, +0.143] |
| Second Chance | -0.006 [-0.030, +0.018] | -0.186 [-0.242, -0.127] | -0.023 [-0.080, +0.034] | +0.215 [+0.175, +0.257] |
| Neutral | +0.136 [+0.109, +0.164] | -0.030 [-0.063, +0.002] | -0.086 [-0.117, -0.052] | -0.020 [-0.054, +0.015] |

### MLP 63

| Condition | Winner | Runner-up | Rank 3 | Rank 4 |
|---|---:|---:|---:|---:|
| Baseline | -0.279 [-0.340, -0.218] | +0.003 [-0.033, +0.035] | +0.120 [+0.090, +0.150] | +0.157 [+0.111, +0.203] |
| Second Chance | -0.203 [-0.230, -0.175] | +0.104 [+0.048, +0.163] | +0.078 [+0.027, +0.130] | +0.020 [-0.015, +0.059] |
| Neutral | +0.047 [-0.013, +0.109] | +0.063 [+0.030, +0.094] | +0.013 [-0.020, +0.042] | -0.123 [-0.161, -0.085] |

## Interpretation

Both components are active answer-rebalancing modules in ordinary Baseline
generation; neither is a Game-only circuit. Mixer 62's natural write is highly
condition dependent: it boosts the original winner in Neutral, mildly
suppresses it in Baseline, and in Second Chance primarily suppresses the
runner-up while boosting rank 4. MLP 63 flattens the Baseline distribution and
also suppresses the original winner in Second Chance, where it additionally
raises the runner-up. Neutral instead raises the top two and suppresses rank 4.

These immediate writes and the previously measured downstream patch effects
answer different questions. The write table describes the module's locally
output-readable A-D update. Patching measures how replacing that output changes
the final computation after all remaining nonlinear processing.

## Artifacts

- `outputs/mechanistic/qwen36_27b_simplemc_sublayers/analysis/selected_writes/selected_component_writes.csv`
- `outputs/mechanistic/qwen36_27b_simplemc_sublayers/analysis/selected_writes/selected_component_writes.{png,svg}`
- Analysis: `mechanistic/analyze_selected_component_writes.py`
