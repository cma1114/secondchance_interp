# Large-model SimpleMC behavioral screen

## Outcome

**DeepSeek V3.2 passes all four Second Chance tests.** Screening stopped after
this result, as planned.

All models saw the same frozen 500 SimpleMC questions and option assignments,
the historical redacted-answer message sequence, the clean neutral prompt, and
temperature-zero generation. Each accepted run was pinned to one serving
provider, returned 20 top-token log probabilities on every call, and had no
format exclusions.

| Model | Provider | Baseline accuracy | Game switch | Neutral switch | AccIncor | Second choice | Game − baseline entropy | Tests passed |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| MiMo-V2.5-Pro | DigitalOcean | 50.4% | 34.8% | 24.0% | 41.3% | 66.7% | +0.423 bits | ✓ / X / ✓ / X |
| GLM-5.2 | StreamLake | 58.8% | 33.4% | 28.6% | 59.3% | 65.9% | +0.366 bits | ✓ / ✓ / ✓ / X |
| Kimi K2.6 | DigitalOcean | 61.6% | 19.0% | 16.4% | 54.0% | 83.2% | +0.421 bits | X / ✓ / ✓ / X |
| **DeepSeek V3.2** | **DigitalOcean** | **48.0%** | **36.2%** | **30.0%** | **57.0%** | **65.7%** | **−0.073 bits** | **✓ / ✓ / ✓ / ✓** |

The final column is Lift / changed-trial AccIncor / SecChoice / NoEntInc.

## DeepSeek V3.2 paper tests

- **Lift:** Game switched on 181/500 trials (36.2%) and neutral on 150/500
  (30.0%). The paired exact p-value was 0.000752; normalized lift was 0.089.
- **AccIncor:** Among baseline-incorrect Game trials that changed, 69/121
  (57.0%) became correct. The two-sided z-test p-value against 1/3 was
  1.41e-7; the exact one-sided p-value was 7.71e-8.
- **SecChoice:** 119/181 changed Game answers (65.7%) selected the paper-defined
  baseline runner-up; p=3.97e-20. The canonical A-D sensitivity result was
  115/181 (63.5%).
- **NoEntInc:** Paper-exact Game-minus-baseline entropy was −0.073 bits, with
  95% CI [−0.110, −0.037] and Wilcoxon p=1.22e-5. The coverage-robust A-D
  result was −0.085 bits [−0.122, −0.048].

## DeepSeek probability signature

The entropy pass does not mean that incorrect feedback leaves the distribution
unchanged. Relative to baseline, the mean A-D-normalized movements were:

| Contrast | Original choice | Runner-up | Each rank-3/4 option |
|---|---:|---:|---:|
| Game − baseline | −0.111 | +0.029 | +0.041 |
| Neutral − baseline | −0.044 | +0.019 | +0.013 |
| Game − neutral | −0.067 | +0.010 | +0.028 |

Thus the clearest feedback-specific effect is additional suppression of the
baseline winner accompanied by increased mass on the lower-ranked options.
The mean Game-minus-neutral runner-up change is small and its normal-theory 95%
CI includes zero. This is not clean evidence for selective runner-up boosting.

Both redo conditions sharpen the raw top-four distribution relative to
baseline, but neutral sharpens it much more: neutral-minus-baseline entropy is
−0.203 bits, while Game-minus-neutral is +0.130 bits. In other words, Game is
flatter than neutral while remaining slightly sharper than baseline. This is
qualitatively different from the Qwen, MiMo, GLM, and Kimi entropy-increase
profile.

## API audit

- DeepSeek used DigitalOcean for all 1,500 accepted calls, with 20/20 requested
  logprobs returned and zero missing metadata.
- Every accepted DeepSeek call reported zero reasoning tokens and no reasoning
  content.
- StreamLake's DeepSeek route was rejected during preflight because it caps
  `top_logprobs` at five; no StreamLake trials entered the experiment.
- Kimi's StreamLake route counted one reasoning token on its first preflight
  call. The full Kimi experiment therefore used DigitalOcean, which passed the
  zero-reasoning audit.
- One MiMo Game call ignored the reasoning-off request and emitted a long
  reasoning response. That record was preserved separately and the one trial
  was rerun with both generic and model-native thinking controls before the
  dataset was accepted.

## Stored artifacts

Each model directory contains the raw baseline, Game, and neutral responses;
preflight prompt audit; run state; paper-test summary; probability/entropy
analysis; and Markdown replication report. The DeepSeek directory is
`outputs/reproduction/simplemc_deepseek_v3_2/`.
