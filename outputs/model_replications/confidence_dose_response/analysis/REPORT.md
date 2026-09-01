# Confidence dose-response of the Game revision policy

**Evidence class:** descriptive/observational analysis of canonical natural
runs. No representation, margin, or confidence state is intervened on, so
none of the associations below establishes a causal confidence readout.

## Prespecified question and interpretation

The primary regression is the continuous Game-minus-Neutral suppression of
the model's own first-presentation winner on that model's own first-pass
top-1-versus-top-2 logit margin. A coefficient near zero, together with flat
confidence terciles, supports a fixed revision reflex whose observed choices
are gated by the decision margin. A positive coefficient means more confident
first answers receive a larger final suppression; a negative coefficient means
the adjustment concentrates on weakly held answers. Mixed signs mean graded
dose response is not shared across models.

The scale-reference decomposition distinguishes a larger complete policy vector
from more selective targeting. `Amplitude` is the norm of the complete centered
Game-minus-Neutral rank vector. `Targeting` is its cosine with
`(-3,+1,+1,+1)/sqrt(12)`. Rising suppression and amplitude with flat targeting
means confidence scales the established policy as a whole; rising targeting
means the policy becomes increasingly specific to old W1.

All logit endpoints use plain within-question centering, not the project's
`×4/3` advantage convention. Predictors are z-scored within cell and within
each of 10,000 question-bootstrap resamples (seed
`20260901`). Reported primary slopes are logits per one-SD increase
in first-pass confidence. The forest plot also reports fully standardized betas.

## Result

Old-winner suppression grows linearly with first-pass confidence in 5 of six cells; the remaining cell is not reliably linear.
The scale-free old-W1-targeting direction increases with confidence in 6 of six cells. Thus the main result is not explained by a generic increase in the size of every Game-minus-Neutral adjustment. The exception to the linear W1-push result is Gemma TriviaMC: its middle confidence tercile is highest, its quadratic term is negative, and the positive discovery slope does not replicate on confirmation.

| Model | Dataset | Mean W1 suppression | C1 slope on W1 suppression | Standardized beta | Scale-reference reading |
|---|---|---:|---:|---:|---|
| Qwen3.6-27B | SimpleMC | 0.571 | 0.727 [0.594, 0.850] | 0.821 [0.767, 0.860] | larger and increasingly W1-targeted revision |
| Qwen3.6-27B | TriviaMC | 1.633 | 1.260 [1.192, 1.324] | 0.867 [0.846, 0.887] | larger and increasingly W1-targeted revision |
| Seed-OSS-36B | SimpleMC | 1.517 | 0.947 [0.800, 1.091] | 0.629 [0.569, 0.684] | larger and increasingly W1-targeted revision |
| Seed-OSS-36B | TriviaMC | 2.258 | 0.318 [0.179, 0.458] | 0.179 [0.103, 0.255] | larger and increasingly W1-targeted revision |
| Gemma-4-31B | SimpleMC | 0.907 | 0.557 [0.455, 0.660] | 0.416 [0.346, 0.484] | larger and increasingly W1-targeted revision |
| Gemma-4-31B | TriviaMC | 1.008 | 0.099 [-0.015, 0.214] | 0.064 [-0.009, 0.139] | no reliable linear W1-suppression dose response |

### Generic scaling versus targeted suppression

| Model | Dataset | W1 push slope | Full-vector amplitude slope | Targeting-cosine slope | R4 Game−Neutral slope |
|---|---|---:|---:|---:|---:|
| Qwen3.6-27B | SimpleMC | 0.727 [0.594, 0.850] | 0.780 [0.620, 0.929] | 0.173 [0.151, 0.197] | 0.247 [0.170, 0.322] |
| Qwen3.6-27B | TriviaMC | 1.260 [1.192, 1.324] | 1.410 [1.331, 1.484] | 0.204 [0.176, 0.233] | 0.374 [0.311, 0.435] |
| Seed-OSS-36B | SimpleMC | 0.947 [0.800, 1.091] | 0.808 [0.637, 0.979] | 0.204 [0.176, 0.232] | 0.218 [0.110, 0.332] |
| Seed-OSS-36B | TriviaMC | 0.318 [0.179, 0.458] | 0.183 [0.037, 0.333] | 0.096 [0.062, 0.131] | -0.353 [-0.451, -0.251] |
| Gemma-4-31B | SimpleMC | 0.557 [0.455, 0.660] | 0.403 [0.278, 0.528] | 0.182 [0.152, 0.213] | 0.122 [0.035, 0.211] |
| Gemma-4-31B | TriviaMC | 0.099 [-0.015, 0.214] | -0.369 [-0.508, -0.235] | 0.154 [0.110, 0.197] | 0.134 [0.061, 0.208] |

The R4 column is reported as a readable rankwise reference, but it is not a
literal placebo: the established Game policy can itself raise R4. The norm and
cosine decomposition is the cleaner test of generic magnitude versus changing
direction. The complete rankwise slopes are preserved in `summary.json`.

### Choice expression after the signed Neutral margin

The companion linear-probability model uses `D = Game switch − Neutral switch`
and the signed Neutral margin of old W1 over its strongest competitor. This
margin is an outcome of Neutral processing and shares final logits with the
dependent quantities, so the model is descriptive—not a causal adjustment.

| Model | Dataset | corr(C1, signed margin) | C1 coefficient | Margin coefficient | VIF |
|---|---|---:|---:|---:|---:|
| Qwen3.6-27B | SimpleMC | +0.885 | -0.109 [-0.180, -0.044] | 0.115 [0.049, 0.189] | 4.63 |
| Qwen3.6-27B | TriviaMC | +0.928 | -0.014 [-0.065, 0.037] | -0.043 [-0.103, 0.016] | 7.17 |
| Seed-OSS-36B | SimpleMC | +0.898 | -0.092 [-0.171, -0.020] | 0.069 [-0.002, 0.144] | 5.16 |
| Seed-OSS-36B | TriviaMC | +0.917 | -0.079 [-0.150, -0.009] | 0.021 [-0.052, 0.092] | 6.28 |
| Gemma-4-31B | SimpleMC | +0.808 | -0.008 [-0.044, 0.028] | 0.017 [-0.017, 0.055] | 2.88 |
| Gemma-4-31B | TriviaMC | +0.775 | -0.016 [-0.041, 0.009] | 0.006 [-0.025, 0.037] | 2.51 |

### Frozen-split robustness of the primary slope

| Model | Dataset | Discovery | Confirmation | Quadratic term (full) |
|---|---|---:|---:|---:|
| Qwen3.6-27B | SimpleMC | 0.794 [0.592, 0.983] | 0.653 [0.488, 0.807] | -0.022 [-0.053, 0.018] |
| Qwen3.6-27B | TriviaMC | 1.250 [1.151, 1.343] | 1.272 [1.175, 1.358] | -0.264 [-0.358, -0.168] |
| Seed-OSS-36B | SimpleMC | 0.992 [0.781, 1.198] | 0.895 [0.687, 1.104] | -0.069 [-0.158, 0.043] |
| Seed-OSS-36B | TriviaMC | 0.236 [0.045, 0.437] | 0.398 [0.206, 0.595] | -0.689 [-0.837, -0.539] |
| Gemma-4-31B | SimpleMC | 0.541 [0.411, 0.677] | 0.581 [0.422, 0.738] | -0.137 [-0.231, -0.041] |
| Gemma-4-31B | TriviaMC | 0.203 [0.006, 0.394] | -0.005 [-0.137, 0.121] | -0.456 [-0.626, -0.298] |

### Confidence terciles

The terciles are stable rank-based groups within each cell, so each group has
nearly equal size even if confidence values tie.

| Model | Dataset | Low / middle / high mean W1 suppression | Low / middle / high mean differential switching |
|---|---|---:|---:|
| Qwen3.6-27B | SimpleMC | 0.167 / 0.317 / 1.233 | +9.6pp / +6.6pp / +9.0pp |
| Qwen3.6-27B | TriviaMC | 0.372 / 1.244 / 3.292 | +12.6pp / +9.6pp / +0.0pp |
| Seed-OSS-36B | SimpleMC | 0.633 / 1.336 / 2.590 | +10.2pp / +24.0pp / +8.4pp |
| Seed-OSS-36B | TriviaMC | 1.482 / 3.327 / 1.964 | +14.4pp / +8.4pp / +0.0pp |
| Gemma-4-31B | SimpleMC | 0.316 / 0.757 / 1.652 | -1.2pp / +1.2pp / +1.8pp |
| Gemma-4-31B | TriviaMC | 0.609 / 1.624 / 0.789 | +1.2pp / +4.8pp / +0.0pp |

## Validity and provenance

Every cell contains 500 questions, finite baseline and final A-D logits, and
exact Game/Neutral condition order. The displayed-order-stable argsort of each
model's own canonical baseline logits reproduced the trajectory array's stored
first-presentation rank order on every question; any mismatch would have aborted
the analysis. Model IDs and pinned revisions were checked in both trajectory
metadata and baseline artifacts. File hashes and exact source paths are in
`summary.json`.

## Scope

Even a consistent positive slope would not by itself demonstrate metacognitive
access to confidence. Because suppression is measured only after all downstream
processing, graded output can arise through nonlinear saturation, interaction
with other question features, or a policy whose amplitude—not selectivity—is
larger on high-confidence questions. The magnitude/direction decomposition narrows
that ambiguity but does not turn this natural-run analysis into a causal test.
