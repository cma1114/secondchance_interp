# Qwen Second Chance results across datasets and model sizes

All runs used 500 questions and temperature zero. AccIncor is accuracy conditional on the baseline answer being wrong and the Game answer changing; because the original answer has then been excluded, its null is 1/3. Unconditional correction rates are shown separately. The Qwen3.6-27B and Qwen3-235B rows within each dataset use identical question objects and A-D assignments.

## Behavioral and entropy tests

| Model | Dataset | Normalized lift | Canonical runner-up | AccIncor | Exact p vs 1/3 | Unconditional correction | A-D entropy change | Tests passed |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-235B | SimpleMC | .144 | 64.3% | 43/76 = 56.6% | 2.66e-5 | 43/200 = 21.5% | +.070* | Lift, AccIncor, SecChoice |
| Qwen3-235B | TriviaMC | .066 | 66.7% | 28/56 = 50.0% | .00724 | 28/142 = 19.7% | +.083 | Lift, AccIncor, SecChoice |
| Qwen3-235B | PopMC | .108 | 64.2% | 44/73 = 60.3% | 2.20e-6 | 44/162 = 27.2% | +.054 | Lift, AccIncor, SecChoice |
| Qwen3.6-27B | TriviaMC | .147 | 68.6% | 35/88 = 39.8% | .122 | 35/183 = 19.1% | +.342 | Lift, SecChoice |
| Qwen3.6-27B | SimpleMC | .335 | 67.3% | 90/193 = 46.6% | 8.69e-5 | 90/302 = 29.8% | +.170 | Lift, AccIncor, SecChoice |

Every run fails NoEntInc because incorrect feedback significantly increases answer entropy. Qwen3.6-27B TriviaMC additionally fails AccIncor against 1/3; its SimpleMC result passes comfortably. Conditional runner-up selection is higher in neutral than Game in all five runs, so runner-up preference is not itself specific to the incorrect-feedback instruction.

The Qwen3.6-27B dataset comparison is especially informative. The feedback-triggered suppression/flattening response appears on both TriviaMC and SimpleMC, but above-one-third correction among changed baseline-wrong answers appears only on SimpleMC. This separates a robust control response from dataset-dependent success at selecting the correct remaining alternative.

## Suppression and flattening signature

Values are incorrect-feedback minus baseline changes in relative effective answer logits, in nats.

| Model | Dataset | First minus runner-up | First relative to ranks 3-4 | Runner-up relative to ranks 3-4 |
|---|---|---:|---:|---:|
| Qwen3-235B | SimpleMC | -1.360 | -1.665 | -.834 |
| Qwen3-235B | TriviaMC | -2.391 | -2.907 | -.515 |
| Qwen3-235B | PopMC | -1.563 | -2.232 | -.669 |
| Qwen3.6-27B | TriviaMC | -3.504 | -4.723 | -1.219 |
| Qwen3.6-27B | SimpleMC | -1.124 | -2.037 | -.913 |

The direct incorrect-feedback-minus-neutral first-versus-runner changes are -1.319, -2.050, -1.621, -3.134, and -1.111 nats respectively. In every run, the original answer loses relative to the runner-up, while the runner-up also loses relative to ranks 3-4. That is broad redistribution with disproportionate original-answer suppression, not a specific runner-up-boost mechanism.

On matched SimpleMC questions, Qwen3.6-27B is much less accurate at baseline than Qwen3-235B (39.6% versus 60.0%) but shows much greater strategic lift (.335 versus .144). Its first-versus-runner collapse is slightly smaller, while its suppression of the first choice relative to ranks 3-4 is larger. The extra switching therefore reflects more aggressive broad flattening, not a more selective runner-up transfer.

On matched TriviaMC questions, Qwen3.6-27B likewise has lower baseline accuracy and greater lift, but its flattening is much more extreme and its AccIncor result fails against 1/3.

All Qwen3.6-27B calls were served by Io Net; the fresh Qwen3-235B calls were served by Parasail. Provider was fixed within each run, but model-size comparisons also change serving stack.

\* The Qwen3-235B SimpleMC A-D entropy estimate uses the all-trial censored-letter sensitivity analysis because that run requested only four raw tokens. All other rows have complete captured A-D distributions.
