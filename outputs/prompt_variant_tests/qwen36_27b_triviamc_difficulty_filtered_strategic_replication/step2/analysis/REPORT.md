# Qwen3.6-27B TriviaMC matching-history blockade

## What was tested

On all 500 frozen difficulty-filtered TriviaMC questions, in both Game and Neutral, we blocked ordinary-attention reads from every complete 2P option line to its semantically matching complete 1P option line at all 16 ordinary-attention layers (4, 8, ..., 64). The control blocked the same four receiver lines and layers but used the next old-rank source line cyclically. This is a whole-line causal test, not a token-localization claim.

## Confirmation results

Matching blockade minus cyclic control, candidate-centered A-D logits:

| Task | W1 | W2 | W3 | W4 |
|---|---:|---:|---:|---:|
| Game | +0.698 [+0.516, +0.874] | -0.091 [-0.236, +0.064] | -0.164 [-0.282, -0.039] | -0.442 [-0.566, -0.320] |
| Neutral | -0.069 [-0.267, +0.125] | +0.046 [-0.124, +0.214] | +0.124 [-0.012, +0.259] | -0.101 [-0.228, +0.025] |
| Game minus Neutral | +0.767 [+0.637, +0.892] | -0.137 [-0.260, -0.013] | -0.288 [-0.386, -0.188] | -0.341 [-0.439, -0.243] |

Aggregated A-D old-W1 choice rates:

| Scenario | Game | Neutral | Game minus Neutral |
|---|---:|---:|---:|
| Natural | +68.4 [+62.8, +74.0] | +73.2 [+67.6, +78.4] | -4.8 [-9.2, -0.8] |
| Matching blockade | +74.4 [+68.8, +79.6] | +74.0 [+68.4, +79.2] | +0.4 [-2.4, +3.2] |
| Cyclic wrong-line blockade | +69.2 [+63.6, +74.8] | +78.4 [+73.2, +83.2] | -9.2 [-14.0, -4.4] |

The primary change in the Game-minus-Neutral W1-choice gap, matching blockade minus cyclic control, is +9.6 [+4.4, +14.8] percentage points.

## Interpretation

The causal Game result is clear and rank-specific. Relative to the cyclic control, blocking the true semantic matches raises old-W1 evidence by 0.698 logits and lowers W3/W4 evidence by 0.164/0.442. Therefore, when the matching-history route is intact in Game, it is doing the opposite: selectively lowering the old winner and supporting weaker old candidates. This is not equal noise added to all candidates.

At the discrete readout, the natural confirmation Game-minus-Neutral old-W1 choice gap is -4.8 [-9.2, -0.8] percentage points. Under the matching blockade it is +0.4 [-2.4, +3.2], a matching-minus-natural change of +5.2 [+0.4, +9.6] points. In plain terms: the causal cut removes the observed extra Game avoidance of the old winner. The cyclic control retains and slightly enlarges that difference, giving the primary matching-minus-cyclic change of +9.6 [+4.4, +14.8] points.

The result is not a complete replication of the earlier SimpleMC task-shared recollection profile. Neutral's confirmation matching-minus-cyclic rank effects all have intervals spanning zero, and its discovery W1 effect has the expected supportive sign but does not repeat on confirmation. Thus this dataset strongly replicates policy-dependent Game use of matching semantic history and the causal removal of preferential Game W1 avoidance; it does not independently establish a stable rankwise Neutral support profile.

The Game pattern and the Game-minus-Neutral rank interaction reproduce on both frozen halves. The discovery matching-minus-cyclic W1 effects are +0.795 in Game and -0.198 in Neutral; the discovery change in the task W1-choice gap is +8.0 [+2.4, +13.6] points.

## Validation and scope

- Natural reproduction maximum absolute aggregated-logit error: 0.00000000.
- Natural displayed-choice agreement with trusted Step 1: 100.00%.
- All intervention logits are finite and every executed source and receiver span is nonempty.
- The cyclic control matches the number of edited semantic relations and layers, not exact source-token count; the observed token-count distributions are stored in `summary.json`.
- These outcomes are causal for direct ordinary-attention reads from complete 1P lines into complete matching 2P lines. They do not isolate semantic wordpieces, edit GLA memory, or test correctness.

See the canonical figure at `figures/qwen36_triviamc_matching_history_step2.png` and machine-readable estimates in `summary.json`.
