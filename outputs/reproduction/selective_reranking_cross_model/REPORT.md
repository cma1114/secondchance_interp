# Selective re-ranking across low-entropy-lift Second Chance runs

Ranks are defined separately on every question from the capabilities-test A–D probabilities. A **Game-only switch** changes under incorrect feedback but not under the neutral redo prompt; a **neutral-only switch** does the reverse. These discordant trials generate the paired behavioral lift.

## Main result

| Model | Dataset | Game-only / neutral-only | Game-only switches to runner-up | Game-only ΔH mean / median | Runner-switch median ΔH | Runner switches with ΔH≤0 | Runner switches with ΔH≤.05 |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-4.1 | TriviaMC | 87/7 | 37/54 = 68.5% | 0.192 / 0.015 | 0.011 | 16.2% | 59.5% |
| Qwen3.6-27B | SimpleMC | 140/13 | 95/140 = 67.9% | 0.405 / 0.289 | 0.333 | 16.8% | 21.1% |
| Qwen3-235B | SimpleMC | 82/24 | 45/82 = 54.9% | 0.227 / 0.235 | 0.341 | 26.7% | 35.6% |
| Qwen3-235B | PopMC | 56/10 | 36/56 = 64.3% | 0.291 / 0.286 | 0.440 | 27.8% | 27.8% |
| Qwen3.5-397B | SimpleMC | 48/18 | 37/48 = 77.1% | 0.171 / 0.154 | 0.084 | 37.8% | 43.2% |
| Qwen3.5-397B | TriviaMC | 25/9 | 22/25 = 88.0% | 0.182 / 0.101 | 0.116 | 22.7% | 31.8% |
| Qwen3.5-397B | PopMC | 41/11 | 32/41 = 78.0% | 0.194 / 0.172 | 0.172 | 28.1% | 28.1% |

ΔH in this table is Game minus baseline A–D entropy. The final two columns use the all-trial A–D sensitivity measure: captured letter-token variants are aggregated, censored letters receive zero, and observed A–D mass is renormalized. A .05-bit cutoff is descriptive rather than a formal equivalence bound; ΔH≤0 requires no cutoff.

## Destination and top-two margin

| Model | Dataset | Net excess switches | Net runner-up excess | Net lower-rank excess | Net unknown rank | Low-entropy runner switches, Game−neutral | Margin coverage | Runner leads in Game | Mean margin change |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-4.1 | TriviaMC | 80 | 31 | 16 | 33 | 22−4 = 18 | 36 | 100.0% | -39.029 |
| Qwen3.6-27B | SimpleMC | 127 | 85 | 42 | 0 | 20−5 = 15 | 95 | 98.9% | -2.183 |
| Qwen3-235B | SimpleMC | 58 | 29 | 29 | 0 | 16−14 = 2 | 44 | 97.7% | -3.244 |
| Qwen3-235B | PopMC | 46 | 27 | 19 | 0 | 10−5 = 5 | 36 | 97.2% | -3.229 |
| Qwen3.5-397B | SimpleMC | 30 | 23 | 7 | 0 | 16−11 = 5 | 37 | 91.9% | -1.487 |
| Qwen3.5-397B | TriviaMC | 16 | 15 | 1 | 0 | 7−4 = 3 | 22 | 90.9% | -1.779 |
| Qwen3.5-397B | PopMC | 30 | 24 | 6 | 0 | 9−7 = 2 | 32 | 84.4% | -1.809 |

The low-entropy count uses ΔH≤.05 relative to each condition's baseline (Game−baseline for Game-only switches and Neutral−baseline for neutral-only switches). The margin is `log p(original winner) − log p(baseline runner-up)`, so a negative Game−neutral change means selective movement toward the runner-up. “Runner actually leads” checks whether the stored Game probabilities themselves put the runner-up above the original winner on trials where the generated Game answer is the runner-up.

## Coverage and strict complete-A–D sensitivity

| Model | Dataset | Runner identifiable, all trials | Complete baseline A–D | Complete Game/baseline entropy pairs | All-trial ΔH | Complete-pair ΔH |
|---|---|---:|---:|---:|---:|---:|
| GPT-4.1 | TriviaMC | 74.5% | 40.3% | 195/499 | 0.033 | 0.002 |
| Qwen3.6-27B | SimpleMC | 100.0% | 100.0% | 500/500 | 0.170 | 0.170 |
| Qwen3-235B | SimpleMC | 100.0% | 98.2% | 157/500 | 0.070 | 0.069 |
| Qwen3-235B | PopMC | 100.0% | 100.0% | 500/500 | 0.054 | 0.054 |
| Qwen3.5-397B | SimpleMC | 100.0% | 100.0% | 498/500 | 0.051 | 0.051 |
| Qwen3.5-397B | TriviaMC | 100.0% | 100.0% | 494/500 | 0.054 | 0.054 |
| Qwen3.5-397B | PopMC | 100.0% | 100.0% | 498/500 | 0.066 | 0.066 |

The strict column requires all four A–D probabilities in both baseline and Game. It is highly selective for GPT-4.1 because even 20 returned tokens often contain multiple spellings of the same high-confidence answer. The all-trial sensitivity estimate is therefore retained as the primary descriptive entropy measure, with coverage stated explicitly.

## What the decomposition says

GPT-4.1 is the clearest case in this set of a low-entropy top-two reversal. Its all-trial entropy increase is only 0.033 bits. More importantly, the Game-only mean of 0.192 bits is highly skewed: the median is 0.015 bits, and the median among identifiable runner-up switches is 0.011 bits. 22 of 37 identifiable Game-only runner switches have ΔH≤.05, compared with 4 neutral-only cases. Requiring complete A–D probability coverage gives the same qualitative result (56.5%, 23 trials).

Qwen3.5-397B is the closest partial analogue on SimpleMC: 77.1% of covered Game-only switches go to the runner-up and 43.2% of those have ΔH≤.05. On TriviaMC, the corresponding values are 88.0% and 31.8%; on PopMC they are 78.0% and 28.1%. The other Qwen runs more often combine switching with appreciable entropy growth or distribute more of the net excess over lower-ranked options.

Entropy in bits and normalized switch lift are not commensurate quantities, so their numerical ratio is not an explanatory test. The informative result is the trial-level mixture: GPT-4.1 combines many nearly entropy-preserving top-two reversals with a minority of large-entropy changes that raise the mean.

## Interpretation rule

Evidence for selective re-ranking is strongest when the net excess switches predominantly go to the baseline runner-up, the original-winner-versus-runner margin falls specifically in Game, and many such reversals occur with non-increasing or negligible entropy. Conversely, frequent movement to ranks 3–4 together with substantial entropy growth supports broad flattening or noise.

![Cross-model selective re-ranking summary](/Users/christopherackerman/repos/secondchance_interp/outputs/reproduction/selective_reranking_cross_model/selective_reranking_summary.png)
