# Relative margin expression across models

Descriptive comparison of each model's natural Game-minus-Neutral R1 score
adjustment against its own final-position decision margins, on the canonical
non-remapped trajectory endpoint. Absolute adjustments are similar across
models; the adjustment-to-margin ratio is what orders the models the way
their behavioral choice-rate effects do. No margin is intervened on.

| Model | Dataset | Median margin (Neutral) | R1 adjustment | Adjustment / margin | Margin < adjustment |
|---|---|---:|---:|---:|---:|
| Qwen3.6-27B | SimpleMC | 0.73 [0.62, 0.75] | 0.57 [0.50, 0.65] | 0.78 [0.70, 0.98] | 43% [38, 49] |
| Qwen3.6-27B | TriviaMC | 3.25 [2.56, 3.94] | 1.63 [1.51, 1.76] | 0.50 [0.44, 0.61] | 35% [30, 37] |
| Seed-OSS-36B | SimpleMC | 2.25 [2.00, 2.75] | 1.52 [1.39, 1.65] | 0.67 [0.56, 0.73] | 38% [32, 41] |
| Seed-OSS-36B | TriviaMC | 12.00 [10.50, 13.25] | 2.26 [2.11, 2.42] | 0.19 [0.17, 0.22] | 16% [12, 19] |
| Gemma-4-31B | SimpleMC | 5.50 [4.75, 6.12] | 0.91 [0.79, 1.02] | 0.16 [0.14, 0.20] | 10% [7, 13] |
| Gemma-4-31B | TriviaMC | 18.37 [17.12, 19.36] | 1.01 [0.87, 1.14] | 0.05 [0.05, 0.06] | 2% [1, 3] |

The flippable fraction is a proxy: it compares each question's Neutral
margin with the mean adjustment, while the true adjustment varies by
question. Behavioral choice-rate gaps for context: Qwen large, Seed
moderate, Gemma null; see the indexed behavioral reports.
