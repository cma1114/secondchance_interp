# Policy-adjusted prospective-answer information

This analysis uses every held-out confirmation question and asks when the final decision position contains the question-specific Game-minus-Neutral change in the eventual four-answer score pattern. Game and Neutral prompts are paired within question and differ only at `incorrect`/`lost`. The shared prospective decoder is used for both conditions, so the contrast is expressed in one learned basis. Discovery-only displayed-letter means are removed before candidates are aligned by first-presentation rank.

The first panel compares the decoded Game-minus-Neutral vector with the exact final Game-minus-Neutral vector for the same held-out question. The condition-sign null randomly reverses which condition is called Game within each question. The second panel decomposes the decoded difference into R1--R4 components; diamonds show the exact final confirmation means.

## Main findings

The question-specific policy-adjusted answer pattern becomes stably decodable at L33 on SimpleMC and L32 on TriviaMC. At L40, learned-decoder similarity is 0.311 on SimpleMC and 0.520 on TriviaMC, while fixed JLens is 0.006 and -0.035. Thus the final decision position contains a non-output-aligned, question-specific Game/Neutral adjustment by the low-to-mid 30s, well before that adjustment is expressed as answer-token logits.

The full-population rank profile matches the strategic-switching account. On SimpleMC, the exact held-out Game-minus-Neutral effect is R1 -0.532, R2 +0.001, R3 +0.181, R4 +0.350: Game selectively lowers the old winner and raises the bottom two old-ranked candidates, with essentially no average R2 change. On TriviaMC the corresponding effects are R1 -1.600, R2 +0.392, R3 +0.604, R4 +0.604; all alternatives rise, but R3 and R4 rise most. The decoded candidate components acquire these signs around L32--L36, except TriviaMC R2, which becomes persistently positive at L43.

Unlike the earlier Game-switch/Neutral-stay panel, this result uses every held-out question and therefore does not obtain its policy difference by selecting questions on the eventual outcome.

## SimpleMC

Discovery n=251; confirmation n=249. [Figure](../../../../../figures/prospective_decoding/qwen36_simplemc_policy_adjusted_prospective_decoding.png)

The learned policy-pattern similarity first has a three-layer positive 95% CI run at L33 and remains continuously positive from L33. The latter is the stable onset used in the interpretation.

- L24: learned cosine 0.006 [-0.063, 0.078]; JLens -0.006; sign-null -0.001 [-0.073, 0.071]
- L32: learned cosine 0.046 [-0.026, 0.120]; JLens 0.015; sign-null -0.001 [-0.073, 0.075]
- L35: learned cosine 0.112 [0.038, 0.191]; JLens 0.012; sign-null 0.000 [-0.076, 0.078]
- L40: learned cosine 0.311 [0.236, 0.386]; JLens 0.006; sign-null 0.000 [-0.086, 0.087]
- L44: learned cosine 0.370 [0.300, 0.436]; JLens 0.071; sign-null 0.001 [-0.085, 0.082]
- L48: learned cosine 0.559 [0.499, 0.616]; JLens 0.224; sign-null -0.000 [-0.090, 0.092]
- L52: learned cosine 0.686 [0.635, 0.735]; JLens 0.541; sign-null 0.001 [-0.100, 0.099]
- L56: learned cosine 0.765 [0.723, 0.802]; JLens 0.698; sign-null -0.000 [-0.104, 0.103]
- L64: learned cosine 0.880 [0.854, 0.905]; JLens 1.000; sign-null -0.000 [-0.112, 0.113]

Exact final Game-minus-Neutral centered rank effects: R1 -0.532, R2 +0.001, R3 +0.181, R4 +0.350.

First sustained layer in each candidate's exact-final direction: R1 L34, R2 none, R3 L36, R4 L34.

## TriviaMC difficulty-filtered

Discovery n=250; confirmation n=250. [Figure](../../../../../figures/prospective_decoding/qwen36_triviamc_policy_adjusted_prospective_decoding.png)

The learned policy-pattern similarity first has a three-layer positive 95% CI run at L18 and remains continuously positive from L32. The latter is the stable onset used in the interpretation.

- L24: learned cosine 0.122 [0.048, 0.197]; JLens 0.017; sign-null -0.000 [-0.077, 0.079]
- L32: learned cosine 0.154 [0.080, 0.225]; JLens -0.024; sign-null 0.001 [-0.074, 0.075]
- L35: learned cosine 0.292 [0.228, 0.357]; JLens -0.051; sign-null 0.000 [-0.076, 0.078]
- L40: learned cosine 0.520 [0.457, 0.577]; JLens -0.035; sign-null 0.001 [-0.091, 0.092]
- L44: learned cosine 0.543 [0.484, 0.598]; JLens -0.086; sign-null 0.000 [-0.090, 0.091]
- L48: learned cosine 0.719 [0.673, 0.760]; JLens 0.137; sign-null 0.001 [-0.102, 0.100]
- L52: learned cosine 0.812 [0.776, 0.845]; JLens 0.442; sign-null 0.000 [-0.109, 0.109]
- L56: learned cosine 0.863 [0.832, 0.892]; JLens 0.588; sign-null 0.001 [-0.115, 0.115]
- L64: learned cosine 0.953 [0.940, 0.964]; JLens 1.000; sign-null 0.000 [-0.122, 0.118]

Exact final Game-minus-Neutral centered rank effects: R1 -1.600, R2 +0.392, R3 +0.604, R4 +0.604.

First sustained layer in each candidate's exact-final direction: R1 L32, R2 L43, R3 L33, R4 L34.

## Scope

This is held-out activation/decoding evidence on the full confirmation populations. It is not conditioned on whether either task switches, avoiding the outcome-selection problem in the earlier paired Game-switch/Neutral-stay panel. It still does not identify a causal source or prove that the model uses the decoded linear direction.
