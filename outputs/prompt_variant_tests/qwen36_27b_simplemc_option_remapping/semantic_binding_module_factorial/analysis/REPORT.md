# Semantic-W1 binding: whole-attention versus whole-MLP factorial

## What was isolated

The second presentation was identical in all cells. Two first presentations had matched Baseline decisions of literal `A`, but `A` named different answer content (X versus Y). As in the canonical empty-history paradigm, no answer token was inserted into the historical assistant turn. The natural interaction therefore measures whether `incorrect` selectively changes the content Qwen would have selected on the first presentation, without changing that Baseline answer letter.

At each tested token position, the causal intervention removed that evaluation-by-semantic-history interaction from all ordinary-attention outputs, all MLP outputs, or both classes together. Individual heads were not selected or patched.

## Bottom line

The behavioral premise replicated decisively: with the matched Baseline first decision held fixed at `A` and the second presentation held fixed, `incorrect` selectively penalized whichever *semantic answer* `A` had named on the first presentation. On confirmation, this interaction was 1.018 centered W1 logits, 2.036 W1-versus-alternative margin logits, and 40.4 percentage points of answer selection.

The tested module-output interactions do **not** explain most of that effect. The only clean behaviorally aligned result that replicated was at the newline closing the repeated W1 option. Removing the interaction from all ordinary-attention and MLP outputs there restored 0.082 logits of the W1-versus-alternative margin, about 4% of the directly comparable natural 2.036-logit targeting effect; its 2.1-point selection effect included zero. Attention-only and MLP-only removal each produced smaller effects, so both contribute, but neither is sufficient.

At the final decision, removing the MLP interaction moved the model strongly *away* from W1 in both splits. That is the opposite of the proposed simple MLP-conjunction mechanism in which an `incorrect`-by-W1 feature suppresses W1 there. The evaluation-period interventions were near zero or did not replicate cleanly.

Therefore, this experiment strengthens the evidence for semantic binding at the behavioral level but does not localize that binding to one of the two proposed module classes at these three positions. It rules out a simple final-position MLP conjunction as the dominant mechanism and identifies only a small distributed contribution while the repeated W1 option is being processed. Because every matched first decision was `A`, the design cleanly controls literal-A bias within this cohort but does not establish letter-general binding for first decisions B-D. A post-run local audit of the two existing Baseline mappings found 138 eligible same-letter/different-content pairs for A, but only 6 for B, 8 for C, and 9 for D. No additional model run is needed to establish that insufficiency; obtaining adequately powered B-D cohorts will require additional remappings and model evaluations.

## Natural semantic targeting

Positive values mean that incorrectness suppresses a candidate more when it was the first-pass answer than when the paired alternative was the first-pass answer.

| Split | Centered-W1 targeting | W1-vs-alternative margin targeting | Selection targeting |
|---|---:|---:|---:|
| Discovery | +0.876 [+0.711, +1.061] | +1.751 [+1.424, +2.121] | +31.2 [+21.1, +42.2] pp |
| Confirmation | +1.018 [+0.851, +1.200] | +2.036 [+1.717, +2.394] | +40.4 [+29.5, +51.4] pp |

For transparency, the held-out raw choice rates were as follows. When X was the first semantic answer, X was chosen on 15.1% of Evaluation/X trials versus 54.8% of Neutral/X trials; when Y rather than X was the first semantic answer, X was chosen on 16.4% of Evaluation/Y and 11.0% of Neutral/Y trials. Symmetrically, Y was chosen on 21.9% of Evaluation/Y versus 43.8% of Neutral/Y trials, but on 16.4% of Evaluation/X and 2.7% of Neutral/X trials. The interaction, rather than any one raw condition difference, isolates dependence on first-pass semantic identity.

## Causal recovery of the prior-answer versus alternative margin

Positive values mean that removing the component-output interaction restores the semantic answer selected on the first presentation.

| Position | Component outputs | Discovery | Confirmation |
|---|---|---:|---:|
| `evaluation_period` | `attention` | +0.008 [-0.014, +0.030] | +0.027 [+0.011, +0.044] |
| `evaluation_period` | `mlp` | +0.014 [-0.004, +0.032] | +0.018 [-0.002, +0.038] |
| `evaluation_period` | `both` | -0.002 [-0.021, +0.018] | +0.009 [-0.010, +0.028] |
| `repeated_candidate` | `attention` | +0.041 [+0.017, +0.065] | +0.044 [+0.023, +0.063] |
| `repeated_candidate` | `mlp` | +0.065 [+0.038, +0.093] | +0.059 [+0.035, +0.083] |
| `repeated_candidate` | `both` | +0.095 [+0.062, +0.129] | +0.082 [+0.058, +0.107] |
| `decision` | `attention` | +0.018 [-0.077, +0.113] | +0.092 [-0.035, +0.217] |
| `decision` | `mlp` | -0.445 [-0.590, -0.301] | -0.474 [-0.593, -0.355] |
| `decision` | `both` | -0.217 [-0.354, -0.073] | -0.149 [-0.310, +0.017] |

## Where the natural interaction was largest

The natural output interaction was not uniformly distributed across layers. On confirmation, ordinary attention peaked at block 48 at the evaluation period (relative norm 0.127), at blocks 48 and 56 at the repeated X/Y option ends (0.359 and 0.392), and at block 56 at the final decision (0.368). MLP interactions peaked at block 46 for repeated X (0.430), block 57 for repeated Y (0.428), and block 48 at the final decision (0.310). The same late bands were prominent in discovery. These are descriptive magnitudes, not causal effect sizes: the whole-class interventions show that a large interaction norm does not imply that the corresponding output drives semantic suppression.

The numerical summary also reports centered-logit recovery, prior-answer selection changes, and entropy changes. Complete per-layer interaction norms remain in the compact `results.npz` files for both frozen splits.

Canonical figure: `figures/qwen36_semantic_binding_module_factorial.png`.
