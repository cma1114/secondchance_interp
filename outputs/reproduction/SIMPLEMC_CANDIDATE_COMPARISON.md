# SimpleMC candidate behavioral screen

## Main result

**Qwen3.5-122B-A10B reproduces the established Qwen three-of-four profile.** It shows a large, statistically reliable Game-versus-neutral switching lift, above-chance correctness after changing from a baseline-incorrect answer, and strong second-choice selection. It fails entropy preservation decisively: Game entropy rises by +0.301 bits.

**Gemma 4 26B-A4B IT is not behaviorally successful in the Game.** It switches almost equally often in Game and neutral (32.3% versus 30.9%; paired p=0.392). Its AccIncor and SecChoice passes therefore do not identify a feedback-specific ability: the neutral redo produces essentially the same redistribution.

## Paper tests

The final column is Lift / changed-trial AccIncor / SecChoice / NoEntInc.

| Model | Paired valid n | Baseline accuracy | Game switch | Neutral switch | Absolute lift | AccIncor | Second choice | Game − baseline entropy | Passes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Qwen3.5-122B-A10B | 500 | 54.4% | 34.8% | 15.6% | +0.192 (p=4.55e-16) | 59.1% | 76.4% | +0.301 bits | ✓ / ✓ / ✓ / X |
| Gemma 4 26B-A4B IT | 495 | 37.2% | 32.3% | 30.9% | +0.014 (p=0.392) | 53.1% | 72.5% | -0.118 bits | X / ✓ / ✓ / ✓ |

## Where the A–D probability goes

Probabilities are aggregated over letter-token variants and renormalized over A–D. “Lower option” is the mean of the baseline-defined third- and fourth-ranked choices.

| Model | Condition | Covered n | Original choice | Runner-up | Mean lower option |
|---|---|---:|---:|---:|---:|
| Qwen3.5-122B-A10B | Baseline | 498 | 66.0% | 20.0% | 7.0% |
| Qwen3.5-122B-A10B | Game | 498 | 47.9% | 26.7% | 12.7% |
| Qwen3.5-122B-A10B | Neutral | 498 | 65.4% | 21.3% | 6.7% |
| Gemma 4 26B-A4B IT | Baseline | 489 | 86.3% | 11.2% | 1.3% |
| Gemma 4 26B-A4B IT | Game | 480 | 65.6% | 24.3% | 5.0% |
| Gemma 4 26B-A4B IT | Neutral | 480 | 65.8% | 24.6% | 4.8% |

| Model | Contrast | Original choice Δ | Runner-up Δ | Each lower option Δ |
|---|---|---:|---:|---:|
| Qwen3.5-122B-A10B | Game − baseline | -0.181 | +0.068 | +0.057 |
| Qwen3.5-122B-A10B | Neutral − baseline | -0.006 | +0.013 | -0.003 |
| Qwen3.5-122B-A10B | Game − neutral | -0.175 | +0.055 | +0.060 |
| Gemma 4 26B-A4B IT | Game − baseline | -0.205 | +0.129 | +0.038 |
| Gemma 4 26B-A4B IT | Neutral − baseline | -0.202 | +0.132 | +0.035 |
| Gemma 4 26B-A4B IT | Game − neutral | -0.002 | -0.002 | +0.002 |

For Qwen, incorrect feedback selectively changes the distribution relative to both baseline and neutral: the original choice loses about 17.5 percentage points relative to neutral, while the runner-up gains 5.5 and each lower option gains 6.0. This is targeted original-choice suppression plus broad flattening—not runner-up boosting alone. Neutral is close to baseline.

For Gemma, the large baseline-to-redo redistribution is generic. Game and neutral each lower the original choice by about 20 percentage points, raise the runner-up by about 13, and raise each lower option by about 3.5. Their direct differences are approximately zero. Thus Gemma supplies a useful negative control for the distinction between responding to an unusual redo context and responding strategically to incorrect feedback.

## Relation to Qwen3.6-27B

The 122B Qwen has higher baseline SimpleMC accuracy than Qwen3.6-27B (54.4% versus 39.6%) and a similar qualitative signature. Its absolute switching lift is 19.2%, compared with 21.8% for Qwen3.6-27B. Its entropy increase is more extreme: +0.301 versus +0.173 bits.

## Capture and exclusions

- Both models used the exact frozen 500 SimpleMC questions and option assignments, temperature zero, redacted first response, clean neutral prompt, and explicit reasoning-off routing.
- Qwen was pinned to Novita BF16 and returned five top alternatives, with complete A–D coverage on 498/500 baseline trials and all 500 redo trials.
- Gemma was pinned to NextBit BF16 and returned twenty alternatives. Five trials had a non-A–D generated response in at least one condition and were excluded as a paired union from behavioral tests (495 usable). All 500 raw probability captures are retained and used in the separate probability sensitivity file.
- Preflight logs record zero reasoning tokens and no reasoning content for both models.
