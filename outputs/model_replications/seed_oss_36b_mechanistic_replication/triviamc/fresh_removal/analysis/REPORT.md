# Fresh 2P × recollected-history double dissociation

## Corrected conclusion

- The intervention removed 98.31% of the discovery-fitted unique-fresh coordinate from every second-option semantic wordpiece and newline while preserving the decoded old-score coordinate.
- On the complete TriviaMC confirmation split, the natural Game-minus-Neutral old-W1-avoidance gap is 7.2% [2.8, 11.6] and the centered old-W1 logit gap is -1.065 [-1.308, -0.831]. On the prespecified conflict subset, the choice gap is only 5.5% [-5.5, 16.4], so a null choice interaction there cannot establish that an intervention preserves a clearly detected natural effect.
- Relative to the same-dose random edit, fresh removal changes the centered old-W1 task gap by +0.111 [-0.034, +0.253] on the complete confirmation split and +0.249 [+0.010, +0.491] on confirmation conflicts. This is the score-level test of whether the decoded fresh coordinate contributes to policy-conditioned old-winner suppression.
- Matching-history blockade changes the choice gap by -7.2% [-12.4, -2.0] over the complete confirmation split, 0.0% [-14.5, 14.5] on conflicts, and -9.2% [-14.4, -4.1] on non-conflicts. Reporting all three prevents an aggregate recollection result from being silently treated as a conflict-specific decomposition.
- After matching blockade and fresh removal are combined, the conflict-set centered old-W1 task gap remains -0.362 [-0.683, -0.029]. These two interventions do not exhaust the conflict pathway.
- **Corrected Seed verdict:** aggregate dependence on matching recollection replicates, but the Qwen recollection-versus-fresh-evidence dissociation does not. The measured fresh 2P score coordinate contributes a material minority of Seed's score-level policy effect in at least one informative slice of each dataset, while conflict questions retain an unlocalized policy-conditioned component after both lesions.
- The matching lesion blocks only first-option-line to matching second-option-line attention. Direct final-position reads of first-presentation history are one plausible Seed-specific bypass, but this run does not localize the surviving component.

## Validity

- Canonical conflict questions: 107 total.
- Native natural max logit error: 0.
- Complete-sequence identity max logit error: 0.

## Confirmation natural task gaps

| Subset | n | Old-W1 avoidance: Game − Neutral | Centered old-W1 logits: Game − Neutral |
|---|---:|---:|---:|
| All confirmation | 250 | 7.2% [2.8, 11.6] | -1.065 [-1.308, -0.831] |
| Conflict | 55 | 5.5% [-5.5, 16.4] | -0.553 [-1.120, -0.018] |
| Non-conflict | 195 | 7.7% [3.1, 12.3] | -1.209 [-1.478, -0.957] |

## Confirmation causal decomposition

Each entry is the intervention-minus-control change in the Game-minus-Neutral task gap. Fresh removal is compared with its exact same-L2 random edit; matching blockade is compared with natural.

| Subset | Contrast | Old-W1 avoidance interaction | Centered old-W1 logit interaction |
|---|---|---:|---:|
| All confirmation | Fresh scrub − random | 0.4% [-4.4, 5.6] | +0.111 [-0.034, +0.253] |
| All confirmation | Matching block − natural | -7.2% [-12.4, -2.0] | +0.728 [+0.484, +0.980] |
| All confirmation | Joint − matching+random | -0.8% [-4.4, 2.8] | -0.029 [-0.096, +0.039] |
| Conflict | Fresh scrub − random | -3.6% [-16.4, 9.1] | +0.249 [+0.010, +0.491] |
| Conflict | Matching block − natural | 0.0% [-14.5, 14.5] | +0.138 [-0.434, +0.729] |
| Conflict | Joint − matching+random | -10.9% [-21.8, 0.0] | +0.043 [-0.067, +0.156] |
| Non-conflict | Fresh scrub − random | 1.5% [-3.6, 6.7] | +0.072 [-0.099, +0.237] |
| Non-conflict | Matching block − natural | -9.2% [-14.4, -4.1] | +0.894 [+0.626, +1.164] |
| Non-conflict | Joint − matching+random | 2.1% [-1.0, 5.1] | -0.049 [-0.131, +0.031] |

## Confirmation conflict scenarios

| Scenario | Game avoidance | Neutral avoidance | Avoidance gap | Centered old-W1 logit gap |
|---|---:|---:|---:|---:|
| complete_path_natural | 72.7% [60.0, 83.6] | 67.3% [54.5, 80.0] | 5.5% [-5.5, 16.4] | -0.553 [-1.120, -0.018] |
| fresh_scrub | 74.5% [61.8, 85.5] | 72.7% [60.0, 83.6] | 1.8% [-5.5, 9.1] | -0.314 [-0.740, +0.108] |
| dose_matched_random | 72.7% [61.8, 83.6] | 67.3% [54.5, 80.0] | 5.5% [-5.5, 16.4] | -0.563 [-1.134, -0.015] |
| matching_history_blockade | 72.7% [60.0, 83.6] | 67.3% [54.5, 80.0] | 5.5% [-5.5, 16.4] | -0.415 [-0.764, -0.051] |
| matching_plus_fresh | 69.1% [56.4, 80.0] | 70.9% [58.2, 81.8] | -1.8% [-12.7, 9.1] | -0.362 [-0.683, -0.029] |
| matching_plus_random | 74.5% [61.8, 85.5] | 65.5% [52.7, 78.2] | 9.1% [-1.8, 20.0] | -0.405 [-0.757, -0.045] |

## Original prespecified conflict-choice contrasts

- **fresh_minus_random:** Game 1.8% [-5.5, 9.1]; Neutral 5.5% [-5.5, 16.4]; interaction -3.6% [-16.4, 9.1].
- **joint_minus_matching_random:** Game -5.5% [-14.5, 3.6]; Neutral 5.5% [0.0, 12.7]; interaction -10.9% [-21.8, 0.0].
- **matching_minus_natural:** Game 0.0% [-12.7, 12.7]; Neutral 0.0% [-16.4, 16.4]; interaction 0.0% [-14.5, 14.5].
- **fresh_minus_natural:** Game 1.8% [-5.5, 9.1]; Neutral 5.5% [-5.5, 16.4]; interaction -3.6% [-14.5, 9.1].
- **joint_minus_matching:** Game -3.6% [-12.7, 5.5]; Neutral 3.6% [-5.5, 12.7]; interaction -7.3% [-20.0, 5.5].

## Confirmation destination choice

The fixed destination subset contains questions where fresh W2 differs from both old W1 and old R2.

| Scenario | Task | Fresh-W2 choice | Old-R2 choice | Fresh W2 − old R2 |
|---|---|---:|---:|---:|
| complete_path_natural | Game | 41.7% [20.8, 62.5] | 12.5% [0.0, 25.0] | 29.2% [4.2, 54.2] |
| complete_path_natural | Neutral | 29.2% [12.5, 50.0] | 12.5% [0.0, 25.0] | 16.7% [-8.3, 41.7] |
| fresh_scrub | Game | 37.5% [16.7, 58.3] | 20.8% [8.3, 37.5] | 16.7% [-12.5, 45.8] |
| fresh_scrub | Neutral | 37.5% [20.8, 58.3] | 20.8% [4.2, 37.5] | 16.7% [-12.5, 45.8] |
| dose_matched_random | Game | 41.7% [20.8, 62.5] | 12.5% [0.0, 25.1] | 29.2% [4.2, 54.2] |
| dose_matched_random | Neutral | 29.2% [12.5, 45.8] | 12.5% [0.0, 25.0] | 16.7% [-8.3, 41.7] |
| matching_history_blockade | Game | 54.2% [33.3, 75.0] | 8.3% [0.0, 20.8] | 45.8% [20.8, 70.8] |
| matching_history_blockade | Neutral | 37.5% [16.7, 58.3] | 20.8% [4.2, 37.5] | 16.7% [-12.5, 45.8] |
| matching_plus_fresh | Game | 58.3% [37.5, 79.2] | 4.2% [0.0, 12.5] | 54.2% [29.2, 75.0] |
| matching_plus_fresh | Neutral | 54.2% [33.3, 75.0] | 12.5% [0.0, 25.0] | 41.7% [12.5, 70.8] |
| matching_plus_random | Game | 54.2% [33.3, 75.0] | 12.5% [0.0, 25.0] | 41.7% [12.5, 66.7] |
| matching_plus_random | Neutral | 41.7% [20.8, 62.5] | 16.7% [4.2, 33.3] | 25.0% [-4.2, 54.2] |

## Destination causal contrasts

- **fresh_minus_random, destination_fresh_winner_choice:** Game -4.2% [-12.5, 0.0]; Neutral 8.3% [-8.3, 25.0]; interaction -12.5% [-25.0, 0.0].
- **fresh_minus_random, destination_old_runner_choice:** Game 8.3% [0.0, 20.8]; Neutral 8.3% [-8.3, 25.0]; interaction 0.0% [-12.5, 12.5].
- **fresh_minus_random, destination_fresh_minus_old_runner:** Game -12.5% [-33.3, 0.0]; Neutral 0.0% [-25.0, 25.0]; interaction -12.5% [-29.2, 4.2].
- **joint_minus_matching_random, destination_fresh_winner_choice:** Game 4.2% [-8.3, 16.7]; Neutral 12.5% [0.0, 25.0]; interaction -8.3% [-25.0, 8.3].
- **joint_minus_matching_random, destination_old_runner_choice:** Game -8.3% [-20.8, 0.0]; Neutral -4.2% [-12.5, 0.0]; interaction -4.2% [-16.7, 8.3].
- **joint_minus_matching_random, destination_fresh_minus_old_runner:** Game 12.5% [-4.2, 37.5]; Neutral 16.7% [0.0, 37.5]; interaction -4.2% [-29.2, 20.9].

## Manipulation checks

- **identity_hook:** fresh fraction remaining 1.0000; mean |old-coordinate change| 0; mean L2 dose 0.
- **fresh_scrub:** fresh fraction remaining 0.0169; mean |old-coordinate change| 0.00190314; mean L2 dose 3.1496.
- **matching_plus_fresh:** fresh fraction remaining 0.0176; mean |old-coordinate change| 0.00192344; mean L2 dose 3.16034.
- **dose_matched_random:** fresh fraction remaining 0.9999; mean |old-coordinate change| 0.00123638; mean L2 dose 3.1496.
- **matching_plus_random:** fresh fraction remaining 0.9999; mean |old-coordinate change| 0.00124392; mean L2 dose 3.16034.
