# Qwen3.6-27B SimpleMC: empty historical assistant response

## Design

This 500-question run reuses the existing raw-ChatML Baseline. Only Game and
Neutral were rerun. The first historical assistant turn contains the explicit
Qwen no-thinking scaffold and nothing else:

```text
<|im_start|>assistant
<think>

</think>

<|im_end|>
```

There is no `[redacted]` string. The final Game and Neutral decision boundary
uses the same assistant header and empty-thinking scaffold as Baseline. The
preflight verified that the first decision boundary is exactly the existing
Baseline prompt before the assistant turn is closed.

## Primary behavioral results

Every unrestricted top token was a valid bare or leading-space A-D token.
Behavior below therefore uses the actually generated full-vocabulary top token,
matching the paper, rather than selecting a letter after A-D renormalization.

| Condition | Accuracy | Switch rate | Mean A-D entropy |
|---|---:|---:|---:|
| Baseline | 40.8% | -- | 1.359 bits |
| Game | 40.6% | 49.0% | 1.638 bits |
| Neutral | 42.6% | 29.4% | 1.385 bits |

- Game-minus-Neutral switching: **+19.6 points**, 95% paired bootstrap CI
  **[+15.4, +23.8]**. There were 115 Game-only switches, 17 Neutral-only
  switches, 130 switches in both, and 238 in neither (exact paired one-sided
  binomial p = 2.31e-19).
- Of 245 Game switches, **190 (77.6%)** went to the Baseline runner-up
  (one-sided binomial p against 1/3 = 1.50e-45).
- On changed, Baseline-wrong Game trials, **88/156 (56.4%)** moved to the
  correct answer (one-sided binomial p against 1/3 = 2.85e-9).
- Game-minus-Baseline A-D entropy: **+0.279 bits**, 95% CI
  **[+0.245, +0.314]**.
- Neutral-minus-Baseline A-D entropy: **+0.026 bits**, 95% CI
  **[+0.005, +0.047]**.
- Game-minus-Neutral A-D entropy: **+0.252 bits**, 95% CI
  **[+0.221, +0.285]**.

Thus this version passes Lift, second-choice, and changed-trial AccIncor, but
still fails the entropy-preservation test.

## Unrestricted output audit

| Condition | Unrestricted A-D top token | `[` top token | `[` in top 4 | `[` in top 10 | Mean A-D probability mass |
|---|---:|---:|---:|---:|---:|
| Game | 500/500 | 0/500 | 0/500 | 0/500 | 99.850% |
| Neutral | 500/500 | 0/500 | 0/500 | 0/500 | 99.867% |

Mean `[` probability fell to 4.43e-8 in Game and 1.33e-8 in Neutral. The
historical `[redacted]` string was therefore responsible for the bracket sink.

## Comparison with the `[redacted]` run

For an apples-to-apples comparison with the earlier explicitly A-D-constrained
analysis, selecting the letter with the greatest aggregate bare-plus-space
probability gives:

| Historical assistant content | Game switch | Neutral switch | Lift | Game entropy | Neutral entropy |
|---|---:|---:|---:|---:|---:|
| `[redacted]` | 34.8% | 19.8% | +15.0 points | 1.672 | 1.421 |
| Empty scaffold | 46.2% | 25.8% | +20.4 points | 1.638 | 1.385 |

Removing `[redacted]` therefore does not remove the Game-specific behavioral
and A-D entropy effects. It eliminates the invalid bracket continuation while
increasing switching in both conditions and increasing the Game-Neutral switch
gap by 5.4 points. The Game-Neutral entropy contrast is essentially unchanged
(+0.251 versus +0.252 bits). This strongly suggests that the differential A-D
flattening is driven by the feedback instruction rather than by the bracket
artifact.

## Artifacts

- Exact prompt/token preflight:
  `outputs/mechanistic/qwen36_27b_simplemc_empty_history/preflight.json`
- Complete Game and Neutral records:
  `outputs/mechanistic/qwen36_27b_simplemc_empty_history/incorrect_results.json`
  and `neutral_results.json`
- Summary and aligned per-question table:
  `outputs/mechanistic/qwen36_27b_simplemc_empty_history/analysis/`
- Canonical mechanistic results and figure index:
  `QWEN36_EMPTY_HISTORY_INTERP_RESULTS.md`
