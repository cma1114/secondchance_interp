# Identical-source selectedness attention-edge test

## Bottom line

The identical original semantic-A option line did not show the prespecified replicated policy-dependent selectedness signature. The post-A comparison-suffix localization stage is therefore not launched automatically.

A directional interaction nevertheless replicated: the Game-minus-Neutral selectedness contrast was +0.300 logits in discovery and +0.193 logits in confirmation, whose 95% interval narrowly crossed zero. The decomposition shows why this is not evidence for a Game-specific suppressive read: blocking the edge removed about 1.1--1.6 logits of A support in Neutral regardless of whether A had won, but had approximately zero effect in Game. Neutral showed a modest additional 0.16--0.26-logit dependence on whether A had won; Game did not.

The source itself was held unusually tightly: every token through the original A line was identical, and its ordinary-attention K/V vectors were bit-exact across chosen and unchosen histories. Only the later B-D ordering changed whether A won.

## Primary centered-target endpoint

Values below are **chosen-minus-unchosen lesion effects**. Positive means blocking the original-A→repeated-A edge raises A more (or lowers it less) when A had won.

| Split | Game | Neutral | Game minus Neutral |
|---|---:|---:|---:|
| Discovery | +0.044 [-0.159, +0.261] | -0.257 [-0.546, +0.033] | +0.300 [+0.040, +0.552] |
| Confirmation | +0.032 [-0.179, +0.258] | -0.160 [-0.342, +0.031] | +0.193 [-0.020, +0.414] |

## Absolute original-A edge effects

These are intervention-minus-natural effects on centered A evidence. Negative means the original-A→repeated-A edge normally supports A.

| Split | Game, A won | Game, A lost | Neutral, A won | Neutral, A lost |
|---|---:|---:|---:|---:|
| Discovery | -0.118 [-0.450, +0.228] | -0.162 [-0.472, +0.159] | -1.585 [-1.984, -1.189] | -1.328 [-1.755, -0.937] |
| Confirmation | +0.161 [-0.161, +0.517] | +0.128 [-0.161, +0.426] | -1.281 [-1.576, -0.989] | -1.121 [-1.440, -0.816] |

## A-edge specificity relative to other first-option sources

| Split | Game | Neutral | Game minus Neutral |
|---|---:|---:|---:|
| Discovery | +0.037 [-0.155, +0.241] | -0.286 [-0.557, -0.024] | +0.322 [+0.075, +0.560] |
| Confirmation | +0.020 [-0.188, +0.244] | -0.183 [-0.382, +0.028] | +0.203 [-0.013, +0.425] |

## Validation

- Discovery exact eligibility: 36/41.
- Confirmation exact eligibility: 33/36.
- Maximum chosen-versus-unchosen original-A K/V error: 0.0.
- Prespecified suffix-localization gate: failed.

Canonical figure: `figures/qwen36_w1_identical_source_selectedness_edge.png`.
