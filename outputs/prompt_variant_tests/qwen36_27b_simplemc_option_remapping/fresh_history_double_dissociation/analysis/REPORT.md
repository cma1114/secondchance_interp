# Fresh 2P × recollected-history double dissociation

## Corrected conclusion

- The intervention removed 98.84% of the discovery-fitted unique-fresh coordinate from every second-option semantic wordpiece and newline while preserving the decoded old-score coordinate.
- On the complete SimpleMC confirmation split, the natural Game-minus-Neutral old-W1-avoidance gap is 17.7% [11.6, 23.7] and the centered old-W1 logit gap is -0.521 [-0.609, -0.437]. On the prespecified conflict subset, the corresponding choice gap is 22.1% [14.7, 30.1] and is clearly detected.
- Relative to the same-dose random edit, fresh removal changes the centered old-W1 task gap by +0.055 [+0.025, +0.084] on the complete confirmation split and +0.041 [+0.002, +0.080] on confirmation conflicts. This is the score-level test of whether the decoded fresh coordinate contributes to policy-conditioned old-winner suppression.
- Matching-history blockade changes the choice gap by -22.1% [-29.7, -14.5] over the complete confirmation split, -24.3% [-34.6, -14.7] on conflicts, and -19.5% [-31.0, -8.0] on non-conflicts. Reporting all three prevents an aggregate recollection result from being silently treated as a conflict-specific decomposition.
- After matching blockade and fresh removal are combined, the conflict-set centered old-W1 task gap remains -0.153 [-0.233, -0.073]. These two interventions do not exhaust the conflict pathway.
- **Corrected Qwen verdict:** matching recollection is the dominant causal route and is necessary for the choice-level preferential-switching effect, but the decoded fresh coordinate is not purely task-shared. Its removal produces a small, statistically reliable reduction of the Game-minus-Neutral old-W1 logit gap on both the complete confirmation split and confirmation conflicts. The intervention does not reliably remove the choice-level gap, so the supported claim is that this fresh coordinate contributes to policy-conditioned scoring but is not necessary for the existence of preferential switching. This remains bounded to the decoded linear coordinate; distributed nonlinear fresh computation was not removed.

## Validity

- Canonical conflict questions: 273 total.
- Native natural max logit error: 0.
- Complete-sequence identity max logit error: 0.

## Confirmation natural task gaps

| Subset | n | Old-W1 avoidance: Game − Neutral | Centered old-W1 logits: Game − Neutral |
|---|---:|---:|---:|
| All confirmation | 249 | 17.7% [11.6, 23.7] | -0.521 [-0.609, -0.437] |
| Conflict | 136 | 22.1% [14.7, 30.1] | -0.522 [-0.628, -0.418] |
| Non-conflict | 113 | 12.4% [3.5, 21.2] | -0.521 [-0.669, -0.382] |

## Confirmation causal decomposition

Each entry is the intervention-minus-control change in the Game-minus-Neutral task gap. Fresh removal is compared with its exact same-L2 random edit; matching blockade is compared with natural.

| Subset | Contrast | Old-W1 avoidance interaction | Centered old-W1 logit interaction |
|---|---|---:|---:|
| All confirmation | Fresh scrub − random | -3.2% [-9.2, 2.8] | +0.055 [+0.025, +0.084] |
| All confirmation | Matching block − natural | -22.1% [-29.7, -14.5] | +0.459 [+0.364, +0.560] |
| All confirmation | Joint − matching+random | 3.2% [-2.0, 8.4] | +0.012 [-0.018, +0.043] |
| Conflict | Fresh scrub − random | -5.1% [-13.2, 2.9] | +0.041 [+0.002, +0.080] |
| Conflict | Matching block − natural | -24.3% [-34.6, -14.7] | +0.378 [+0.260, +0.500] |
| Conflict | Joint − matching+random | 7.4% [0.7, 14.0] | -0.021 [-0.061, +0.018] |
| Non-conflict | Fresh scrub − random | -0.9% [-10.6, 8.8] | +0.071 [+0.026, +0.116] |
| Non-conflict | Matching block − natural | -19.5% [-31.0, -8.0] | +0.557 [+0.400, +0.716] |
| Non-conflict | Joint − matching+random | -1.8% [-9.7, 6.2] | +0.052 [+0.008, +0.100] |

## Confirmation conflict scenarios

| Scenario | Game avoidance | Neutral avoidance | Avoidance gap | Centered old-W1 logit gap |
|---|---:|---:|---:|---:|
| complete_sequence_natural | 82.4% [75.7, 88.2] | 60.3% [52.2, 68.4] | 22.1% [14.7, 30.1] | -0.522 [-0.628, -0.418] |
| fresh_scrub | 80.1% [73.5, 86.8] | 62.5% [54.4, 70.6] | 17.6% [10.3, 25.0] | -0.468 [-0.563, -0.374] |
| dose_matched_random | 81.6% [75.0, 87.5] | 58.8% [50.7, 66.9] | 22.8% [14.7, 30.9] | -0.509 [-0.610, -0.408] |
| matching_history_blockade | 71.3% [63.2, 78.7] | 73.5% [66.2, 80.9] | -2.2% [-8.1, 3.7] | -0.144 [-0.234, -0.053] |
| matching_plus_fresh | 77.2% [69.9, 83.8] | 71.3% [64.0, 78.7] | 5.9% [-0.7, 13.2] | -0.153 [-0.233, -0.073] |
| matching_plus_random | 72.1% [64.0, 79.4] | 73.5% [66.2, 80.9] | -1.5% [-7.4, 4.4] | -0.132 [-0.221, -0.045] |

## Original prespecified conflict-choice contrasts

- **fresh_minus_random:** Game -1.5% [-7.4, 4.4]; Neutral 3.7% [-3.7, 11.0]; interaction -5.1% [-13.2, 2.9].
- **joint_minus_matching_random:** Game 5.1% [-1.5, 11.8]; Neutral -2.2% [-8.8, 4.4]; interaction 7.4% [0.7, 14.0].
- **matching_minus_natural:** Game -11.0% [-19.1, -2.9]; Neutral 13.2% [4.4, 22.1]; interaction -24.3% [-34.6, -14.0].
- **fresh_minus_natural:** Game -2.2% [-7.4, 2.9]; Neutral 2.2% [-5.1, 9.6]; interaction -4.4% [-11.8, 2.9].
- **joint_minus_matching:** Game 5.9% [-0.7, 12.5]; Neutral -2.2% [-8.8, 4.4]; interaction 8.1% [0.7, 15.4].

## Confirmation destination choice

The fixed destination subset contains questions where fresh W2 differs from both old W1 and old R2.

| Scenario | Task | Fresh-W2 choice | Old-R2 choice | Fresh W2 − old R2 |
|---|---|---:|---:|---:|
| complete_sequence_natural | Game | 54.3% [42.9, 65.7] | 21.4% [12.9, 31.4] | 32.9% [12.9, 51.4] |
| complete_sequence_natural | Neutral | 34.3% [22.9, 45.7] | 21.4% [12.9, 31.4] | 12.9% [-4.3, 30.0] |
| fresh_scrub | Game | 50.0% [38.6, 61.4] | 24.3% [14.3, 34.3] | 25.7% [5.7, 44.3] |
| fresh_scrub | Neutral | 40.0% [28.6, 51.4] | 21.4% [11.4, 31.4] | 18.6% [0.0, 35.7] |
| dose_matched_random | Game | 57.1% [45.7, 68.6] | 20.0% [11.4, 30.0] | 37.1% [18.6, 55.7] |
| dose_matched_random | Neutral | 34.3% [24.3, 45.7] | 18.6% [10.0, 28.6] | 15.7% [-1.4, 32.9] |
| matching_history_blockade | Game | 34.3% [22.9, 45.7] | 27.1% [17.1, 38.6] | 7.1% [-11.4, 25.7] |
| matching_history_blockade | Neutral | 30.0% [20.0, 41.4] | 34.3% [22.9, 45.7] | -4.3% [-22.9, 14.3] |
| matching_plus_fresh | Game | 47.1% [35.7, 58.6] | 25.7% [15.7, 35.7] | 21.4% [1.4, 40.0] |
| matching_plus_fresh | Neutral | 25.7% [15.7, 35.7] | 34.3% [22.9, 45.7] | -8.6% [-27.1, 10.0] |
| matching_plus_random | Game | 37.1% [25.7, 48.6] | 27.1% [17.1, 37.1] | 10.0% [-8.6, 28.6] |
| matching_plus_random | Neutral | 30.0% [20.0, 41.4] | 34.3% [24.3, 45.7] | -4.3% [-22.9, 14.3] |

## Destination causal contrasts

- **fresh_minus_random, destination_fresh_winner_choice:** Game -7.1% [-17.1, 2.9]; Neutral 5.7% [-4.3, 15.7]; interaction -12.9% [-24.3, -1.4].
- **fresh_minus_random, destination_old_runner_choice:** Game 4.3% [-4.3, 12.9]; Neutral 2.9% [-7.1, 12.9]; interaction 1.4% [-7.1, 10.0].
- **fresh_minus_random, destination_fresh_minus_old_runner:** Game -11.4% [-28.6, 4.3]; Neutral 2.9% [-14.3, 20.0]; interaction -14.3% [-32.9, 2.9].
- **joint_minus_matching_random, destination_fresh_winner_choice:** Game 10.0% [2.9, 18.6]; Neutral -4.3% [-11.4, 2.9]; interaction 14.3% [5.7, 24.3].
- **joint_minus_matching_random, destination_old_runner_choice:** Game -1.4% [-10.0, 7.1]; Neutral 0.0% [-10.0, 10.0]; interaction -1.4% [-10.0, 7.1].
- **joint_minus_matching_random, destination_fresh_minus_old_runner:** Game 11.4% [-1.4, 25.7]; Neutral -4.3% [-20.0, 8.6]; interaction 15.7% [1.4, 31.4].

## Manipulation checks

- **identity_hook:** fresh fraction remaining 1.0000; mean |old-coordinate change| 0; mean L2 dose 0.
- **fresh_scrub:** fresh fraction remaining 0.0116; mean |old-coordinate change| 0.00200773; mean L2 dose 0.694089.
- **matching_plus_fresh:** fresh fraction remaining 0.0166; mean |old-coordinate change| 0.00151687; mean L2 dose 0.531468.
- **dose_matched_random:** fresh fraction remaining 0.9997; mean |old-coordinate change| 0.00125221; mean L2 dose 0.694089.
- **matching_plus_random:** fresh fraction remaining 0.9998; mean |old-coordinate change| 0.000883609; mean L2 dose 0.531468.
