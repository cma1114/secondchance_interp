# Original/repeated question-stem access factorial

## Conclusion

- Direct ordinary-attention rereading of the original question wording after the first answer is **not necessary** for preferential Game switching. Blocking it changes held-out Game old-W1 avoidance by -0.7% [-6.6, 5.2], while increasing Neutral avoidance by 14.0% [6.6, 22.1]. The same asymmetry appears in discovery (Game 2.2% [-4.4, 8.0]; Neutral 15.3% [7.3, 23.4]).
- Direct rereading of the repeated question wording contributes to the decoded fresh-evidence state at the 2P semantic tokens, but it is not required for choosing the fresh W2. Its blockade reduces held-out semantic-token fresh alignment at L40/L48 in both tasks, yet fresh-W2 choice increases in Game by 11.0% [4.4, 18.4] and in Neutral by 8.8% [1.5, 16.2].
- Blocking both question copies shrinks the held-out Game-minus-Neutral avoidance gap from 22.1% [14.7, 30.1] to 5.9% [0.7, 11.8]. This happens because Neutral becomes much more switch-prone (+21.3 points), not because Game loses its old-winner avoidance.
- The best-supported interpretation is therefore that direct question-stem access is a stabilizing/reconsideration input—especially under Neutral—not the source of Game's strategic old-winner suppression. The intervention does not erase question information already embedded in option states or GLA memory, so it does not rule out every distributed form of recomputation.

## What was tested

Every query from the feedback token onward is prevented from reading 1P question-stem/separator K/V at every ordinary-attention layer.

Every 2P option-line, post-list cue/query, and final-prefix query is prevented from reading causally prior 2P question-stem/separator K/V at every ordinary-attention layer.

This is a causal test of direct ordinary-attention rereading of question words. It does not erase question information already stored elsewhere and it leaves GLA recurrent memory untouched.

## Confirmation conflict behavior

| Scenario | Game avoids old W1 | Neutral avoids old W1 | Game - Neutral |
|---|---:|---:|---:|
| identity_monitor | 82.4% [75.7, 88.2] | 60.3% [52.2, 68.4] | 22.1% [14.7, 30.1] |
| block_first_stem | 81.6% [75.0, 88.2] | 74.3% [66.9, 81.6] | 7.4% [0.7, 14.0] |
| block_second_stem | 87.5% [81.6, 92.6] | 72.1% [64.7, 79.4] | 15.4% [8.8, 22.8] |
| block_both_stems | 87.5% [81.6, 92.6] | 81.6% [75.0, 87.5] | 5.9% [0.7, 11.8] |

## Causal changes from identity

- **block_first_stem:** avoidance change Game -0.7% [-6.6, 5.2]; Neutral 14.0% [6.6, 22.1]; task interaction -14.7% [-23.5, -5.9]. Game W1-minus-fresh-W2 logit change -0.005 [-0.138, +0.130].
- **block_second_stem:** avoidance change Game 5.1% [0.7, 10.3]; Neutral 11.8% [4.4, 19.1]; task interaction -6.6% [-14.7, 0.7]. Game W1-minus-fresh-W2 logit change -0.062 [-0.172, +0.050].
- **block_both_stems:** avoidance change Game 5.1% [-2.2, 12.5]; Neutral 21.3% [12.5, 30.1]; task interaction -16.2% [-25.7, -6.6]. Game W1-minus-fresh-W2 logit change -0.048 [-0.206, +0.111].

## Fresh-W2 choice on confirmation conflicts

W2 is the semantic candidate selected by the one-pass remapped baseline. These changes distinguish a directed move toward the freshly favored candidate from arbitrary switching to some other option.

| Blockade | Game change | Neutral change |
|---|---:|---:|
| block_first_stem | 5.1% [-2.2, 13.2] | 11.8% [3.7, 19.9] |
| block_second_stem | 11.0% [4.4, 18.4] | 8.8% [1.5, 16.2] |
| block_both_stems | 8.1% [-0.7, 16.9] | 18.4% [10.3, 27.2] |

## Fresh-evidence coordinate inside repeated options

The table reports the absolute change in held-out fresh-evidence alignment at the 2P semantic wordpieces. Unlike a ratio, this remains defined when natural alignment is near zero.

| Task | Blockade | L31 | L40 | L48 | L64 |
|---|---|---:|---:|---:|---:|
| Game | block_first_stem | +0.236 | +0.127 | +0.230 | -0.057 |
| Game | block_second_stem | -0.052 | -0.184 | -0.211 | -0.027 |
| Game | block_both_stems | -0.055 | -0.301 | -0.143 | +0.106 |
| Neutral | block_first_stem | +0.236 | +0.147 | +0.270 | -0.003 |
| Neutral | block_second_stem | -0.067 | -0.168 | -0.244 | -0.193 |
| Neutral | block_both_stems | -0.054 | -0.259 | -0.105 | +0.023 |

## Validity

- Natural reproduction maximum error: 0.
- Identity-monitor maximum error: 0.
- Canonical conflicts: 273.
- The complete layerwise fresh/old coordinate manipulation audit is in `summary.json` and the canonical figure.
