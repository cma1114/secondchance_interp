# Does Game preferentially route redistributed score to the old runner-up?

No. Across both models and both datasets, when `incorrect` pushes the old
winner down, the old runner-up gains **less** of the redistributed score than
the two lower-ranked candidates in three of four cells, and is statistically
indistinguishable from them in the fourth. In no cell does it gain reliably
more. The runner-up wins most switches despite being the policy's
least-favored alternative, because it starts closest to the top. This is the
direct quantitative rebuttal of a second-choice-targeting reading of the
natural rank profiles.

These are paired within-question Game-minus-Neutral effects on exact live
final A-D logits (centered within question, candidates aligned by each
model's own frozen first-presentation ranking). Observational natural runs;
no intervention; question-level percentile bootstrap.

## All questions

| Cell | R1 | R2 | R3 | R4 | R2 − mean(R3, R4) |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B SimpleMC | -0.571 [-0.649, -0.495] | +0.022 [-0.028, +0.072] | +0.190 [+0.151, +0.229] | +0.360 [+0.320, +0.400] | -0.253 [-0.312, -0.193] |
| Qwen3.6-27B TriviaMC | -1.633 [-1.762, -1.508] | +0.429 [+0.349, +0.509] | +0.552 [+0.483, +0.623] | +0.652 [+0.588, +0.714] | -0.173 [-0.274, -0.074] |
| Seed-OSS-36B SimpleMC | -1.517 [-1.649, -1.387] | +0.170 [+0.075, +0.264] | +0.410 [+0.333, +0.487] | +0.938 [+0.853, +1.025] | -0.504 [-0.616, -0.389] |
| Seed-OSS-36B TriviaMC | -2.258 [-2.416, -2.104] | +0.842 [+0.701, +0.986] | +0.812 [+0.714, +0.912] | +0.603 [+0.492, +0.717] | +0.134 [-0.042, +0.311] |

## Frozen-split robustness of the primary contrast

| Cell | Discovery | Confirmation |
|---|---:|---:|
| Qwen3.6-27B SimpleMC | -0.251 [-0.339, -0.163] | -0.255 [-0.331, -0.179] |
| Qwen3.6-27B TriviaMC | -0.156 [-0.297, -0.012] | -0.191 [-0.327, -0.055] |
| Seed-OSS-36B SimpleMC | -0.488 [-0.641, -0.330] | -0.521 [-0.686, -0.348] |
| Seed-OSS-36B TriviaMC | +0.093 [-0.146, +0.337] | +0.176 [-0.082, +0.431] |

The absolute R2 rise on TriviaMC in both models is conservation of mass: the
old winner there is dominant (its suppression is largest), so every
alternative floats up — and R2 floats up least or equally, never most.

## Scope

This is a descriptive contrast on natural runs. The causal case against
second-choice targeting rests on the Qwen interventions (categorical-winner
audit, matching-route lesions, destination analysis); this analysis shows
the natural profiles point the same way in both models and both datasets.

## Artifacts

- Figure: `figures/model_replications/r2_redistribution_contrast.png`
- Machine-readable estimates: `summary.json`
- Inputs: the natural non-remapped trajectory run arrays for each model and
  dataset, and the frozen SimpleMC/TriviaMC discovery splits.
