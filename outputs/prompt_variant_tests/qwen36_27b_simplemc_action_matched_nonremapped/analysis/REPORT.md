# Non-remapped action-matched SimpleMC behavior

## Design

The option mapping is identical on both presentations. Evaluation and Neutral differ by exactly one model-visible word:

- Evaluation: `Your answer was incorrect. Choose the answer again.`
- Neutral: `Your answer was lost. Choose the answer again.`

Baseline and Neutral are reused from the exact existing run; only Evaluation was newly collected.

### Baseline comparator

The Baseline artifact contains two answer fields from two different runs:

- `answer`: the model's answer in the current, exactly matched prompt format;
- `baseline_answer`: an answer imported from an earlier compiled run.

This analysis uses `answer`, because it is the Baseline generated under the same current formatting as Evaluation and Neutral. The two fields disagree on 59/500 questions and must not be pooled or interchanged. An older report in `qwen36_27b_simplemc_token_matched_feedback/analysis/` used the imported field; that report is now explicitly marked as historical.

Here **switch** is defined by the aggregated A-D answer argmax, not the unrestricted top token. The feedback factorial's primary unrestricted-token endpoint is a different readout and should not be mixed with these rates.

## Results

| Condition | Accuracy | Switch rate | Mean A-D entropy |
|---|---:|---:|---:|
| Baseline | 42.0% | -- | 1.359 |
| Evaluation | 39.4% | 35.2% | 1.576 |
| Neutral | 42.8% | 26.4% | 1.344 |

Evaluation-minus-Neutral switching: **+8.8 [+4.6, +12.8] percentage points**.
Evaluation-minus-Neutral entropy: **+0.232 [+0.205, +0.260] bits**.
Among Evaluation switches, 125/176 (71.0%) select the Baseline runner-up.
The one-sided binomial p-value against 1/3 is 2.03e-24.
Among changed Baseline-wrong Evaluation trials, 54/109 (49.5%) move to the correct answer.
The one-sided binomial p-value against 1/3 is 0.000339.

Thus Evaluation passes lift, runner-up, and changed-wrong accuracy checks, but fails entropy preservation: its A-D entropy rises substantially relative to both Baseline and Neutral.

## Switching by Baseline answer letter

| Baseline letter | n | Evaluation | Neutral | Difference |
|---|---:|---:|---:|---:|
| A | 240 | 46.2% | 40.4% | +5.8 [-0.8, +12.1] pp |
| B | 73 | 16.4% | 5.5% | +11.0 [+2.7, +19.2] pp |
| C | 106 | 26.4% | 12.3% | +14.2 [+5.7, +22.6] pp |
| D | 81 | 30.9% | 22.2% | +8.6 [-1.2, +18.5] pp |

Complete continuous rank-aligned logit redistributions and confidence intervals are in `summary.json`.

## Context: explicit different-answer instruction

Under the same current Baseline comparator, the existing `incorrect + different` condition switches on 40.6% of questions, versus 26.4% for Neutral: a 14.2% lift. The single-word action-matched Evaluation condition therefore reproduces 62.0% of that raw switching lift.
