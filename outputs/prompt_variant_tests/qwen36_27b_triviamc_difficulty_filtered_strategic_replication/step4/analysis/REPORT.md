# Qwen3.6-27B TriviaMC policy × retrieved-rank factorial

## What was tested

The experiment crossed two causal manipulations on all 500 frozen questions. First, it reciprocally transplanted the evaluation-closing period's GLA update between Game and Neutral at all 48 GLA layers. Second, it blocked all four matching complete 1P-option-line to 2P-option-line reads at every ordinary-attention layer, with an equal-structure cyclic wrong-line blockade as the route control. The old MLP-49 restoration and individual-layer screens were not included.

The period transplant uses `preserve_source_output=False`, exactly matching the historical SimpleMC factorial. It therefore changes the complete GLA update at that token, including the donor-conditioned local output; it is not an output-preserved isolation of recurrent memory alone.

## Confirmation results

Matching blockade minus cyclic control, in candidate-centered A-D logits:

| Recipient task and policy state | W1 | W2 | W3 | W4 |
|---|---:|---:|---:|---:|
| Game, natural policy | +0.698 [+0.517, +0.874] | -0.090 [-0.236, +0.062] | -0.166 [-0.288, -0.041] | -0.442 [-0.564, -0.320] |
| Game, opposite policy installed | +0.091 [-0.096, +0.277] | +0.077 [-0.081, +0.235] | +0.046 [-0.082, +0.177] | -0.214 [-0.326, -0.099] |
| Game, policy × route interaction | -0.607 [-0.704, -0.510] | +0.167 [+0.065, +0.264] | +0.212 [+0.138, +0.287] | +0.228 [+0.150, +0.310] |
| Neutral, natural policy | -0.068 [-0.266, +0.123] | +0.044 [-0.124, +0.212] | +0.124 [-0.011, +0.262] | -0.100 [-0.221, +0.026] |
| Neutral, opposite policy installed | +0.822 [+0.632, +1.010] | -0.097 [-0.255, +0.070] | -0.206 [-0.341, -0.071] | -0.519 [-0.663, -0.374] |
| Neutral, policy × route interaction | +0.890 [+0.747, +1.031] | -0.141 [-0.275, -0.010] | -0.330 [-0.436, -0.221] | -0.419 [-0.545, -0.299] |

Conflict-trial old-W1 choice (W1 differs from the standalone remapped second-presentation winner):

| Recipient task | Natural policy | Opposite policy installed | Change |
|---|---:|---:|---:|
| Game | +33.8 [+23.0, +44.6] | +47.3 [+36.5, +58.1] | +13.5 [+5.4, +21.6] pp |
| Neutral | +44.6 [+33.8, +55.4] | +32.4 [+21.6, +43.2] | -12.2 [-21.6, -2.7] pp |

## Interpretation

The answer is yes. Under natural Game, blocking the true matching route rather than the cyclic wrong route raises old-W1 evidence by 0.698 logits and lowers W3/W4 by 0.166/0.442. Therefore the intact matching route does the opposite: it selectively suppresses the recollected old winner and supports weaker old candidates. Installing the Neutral period update into the same Game prompts nearly removes that profile: the W1 lesion effect falls by 0.607 logits to 0.091.

Natural Neutral again has no stable matching-specific rank profile on the confirmation split. But installing the Game period update into Neutral creates a strong Game-like route: matching blockade raises W1 by 0.822 and lowers W3/W4 by 0.206/0.519. Relative to natural Neutral, the policy × route interaction is +0.890 at W1 and -0.330/-0.419 at W3/W4. Thus the Game policy update is causally sufficient to make Neutral use recalled candidate rank in the Game pattern; the Neutral update is sufficient to turn off most of that pattern in Game.

The behavioral readout agrees. On confirmation conflict trials, installing Neutral policy in Game raises old-W1 choice by 13.5 points, while installing Game policy in Neutral lowers it by 12.2 points; both paired intervals exclude zero. Discovery independently gives +16.4 and -19.7 points. The interaction is therefore present in rank-shaped logits and actual choices on both frozen halves.

This shows that policy is not merely added after recollection. The state written at the feedback period changes how the later matching-history route uses retrieved W1-W4 information. The asymmetry also matters: TriviaMC replicates the Game-conditioned route strongly, but not the SimpleMC claim that natural Neutral has a stable supportive rank profile through this exact route.

The numerical direction and replication status are stated from the table above and both frozen halves in `summary.json`; no correctness endpoint is computed.

## Validation

- Questions: 500; discovery/confirmation: 250/250; confirmation conflict trials: 74.
- Same-batch natural maximum absolute error to trusted Step 1: 0.49995422.
- Corrected natural maximum absolute error: 0.00000000.
- Raw paired-natural displayed-choice agreement with trusted Step 1: 100.00%.
- Fresh unrestricted/aggregated A-D winner agreement: 98.40%.
- Policy liveness maximum absolute change: 4.933323; route liveness: 7.334286.
- All logits are finite and every source/receiver span is nonempty.
- Choices use displayed A-D first-maximum tie resolution before semantic remapping; 11 scenario cells had an exact maximum tie.

See `figures/qwen36_triviamc_policy_rank_step4.png` and `summary.json`.
