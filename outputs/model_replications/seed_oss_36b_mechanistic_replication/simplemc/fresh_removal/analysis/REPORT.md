# Fresh 2P × recollected-history double dissociation

## Corrected conclusion

- The intervention removed 97.26% of the discovery-fitted unique-fresh coordinate from every second-option semantic wordpiece and newline while preserving the decoded old-score coordinate.
- On the complete SimpleMC confirmation split, the natural Game-minus-Neutral old-W1-avoidance gap is 9.2% [4.8, 13.7] and the centered old-W1 logit gap is -0.604 [-0.795, -0.418]. On the prespecified conflict subset, the choice gap is only 4.8% [0.0, 10.3], so a null choice interaction there cannot establish that an intervention preserves a clearly detected natural effect.
- Relative to the same-dose random edit, fresh removal changes the centered old-W1 task gap by +0.215 [+0.143, +0.292] on the complete confirmation split and +0.066 [-0.018, +0.150] on confirmation conflicts. This is the score-level test of whether the decoded fresh coordinate contributes to policy-conditioned old-winner suppression.
- Matching-history blockade changes the choice gap by -10.4% [-16.9, -4.0] over the complete confirmation split, -0.8% [-9.5, 7.9] on conflicts, and -20.3% [-29.3, -11.4] on non-conflicts. Reporting all three prevents an aggregate recollection result from being silently treated as a conflict-specific decomposition.
- After matching blockade and fresh removal are combined, the conflict-set centered old-W1 task gap remains -0.259 [-0.483, -0.029]. These two interventions do not exhaust the conflict pathway.
- **Corrected Seed verdict:** aggregate dependence on matching recollection replicates, but the Qwen recollection-versus-fresh-evidence dissociation does not. The measured fresh 2P score coordinate contributes a material minority of Seed's score-level policy effect in at least one informative slice of each dataset, while conflict questions retain an unlocalized policy-conditioned component after both lesions.
- The matching lesion blocks only first-option-line to matching second-option-line attention. Direct final-position reads of first-presentation history are one plausible Seed-specific bypass, but this run does not localize the surviving component.

## Validity

- Canonical conflict questions: 259 total.
- Native natural max logit error: 0.
- Complete-sequence identity max logit error: 0.

## Confirmation natural task gaps

| Subset | n | Old-W1 avoidance: Game − Neutral | Centered old-W1 logits: Game − Neutral |
|---|---:|---:|---:|
| All confirmation | 249 | 9.2% [4.8, 13.7] | -0.604 [-0.795, -0.418] |
| Conflict | 126 | 4.8% [0.0, 10.3] | -0.413 [-0.669, -0.166] |
| Non-conflict | 123 | 13.8% [7.3, 21.1] | -0.800 [-1.087, -0.518] |

## Confirmation causal decomposition

Each entry is the intervention-minus-control change in the Game-minus-Neutral task gap. Fresh removal is compared with its exact same-L2 random edit; matching blockade is compared with natural.

| Subset | Contrast | Old-W1 avoidance interaction | Centered old-W1 logit interaction |
|---|---|---:|---:|
| All confirmation | Fresh scrub − random | -2.4% [-6.8, 2.0] | +0.215 [+0.143, +0.292] |
| All confirmation | Matching block − natural | -10.4% [-16.9, -4.0] | +0.680 [+0.450, +0.918] |
| All confirmation | Joint − matching+random | 3.2% [-1.2, 8.0] | -0.057 [-0.097, -0.018] |
| Conflict | Fresh scrub − random | 4.0% [-2.4, 10.3] | +0.066 [-0.018, +0.150] |
| Conflict | Matching block − natural | -0.8% [-9.5, 7.9] | +0.158 [-0.162, +0.487] |
| Conflict | Joint − matching+random | 0.8% [-5.6, 7.1] | -0.022 [-0.068, +0.027] |
| Non-conflict | Fresh scrub − random | -8.9% [-15.4, -2.4] | +0.369 [+0.253, +0.492] |
| Non-conflict | Matching block − natural | -20.3% [-29.3, -11.4] | +1.215 [+0.910, +1.524] |
| Non-conflict | Joint − matching+random | 5.7% [-0.8, 12.2] | -0.093 [-0.157, -0.030] |

## Confirmation conflict scenarios

| Scenario | Game avoidance | Neutral avoidance | Avoidance gap | Centered old-W1 logit gap |
|---|---:|---:|---:|---:|
| complete_path_natural | 77.0% [69.8, 84.1] | 72.2% [64.3, 80.2] | 4.8% [0.0, 10.3] | -0.413 [-0.669, -0.166] |
| fresh_scrub | 80.2% [73.0, 86.5] | 71.4% [63.5, 78.6] | 8.7% [2.4, 15.1] | -0.338 [-0.569, -0.114] |
| dose_matched_random | 77.0% [69.0, 84.1] | 72.2% [64.3, 80.2] | 4.8% [0.0, 9.5] | -0.404 [-0.661, -0.160] |
| matching_history_blockade | 75.4% [67.5, 82.5] | 71.4% [63.5, 79.4] | 4.0% [-3.2, 11.1] | -0.255 [-0.491, -0.012] |
| matching_plus_fresh | 78.6% [71.4, 85.7] | 73.0% [65.1, 81.0] | 5.6% [-0.8, 12.7] | -0.259 [-0.483, -0.029] |
| matching_plus_random | 76.2% [68.3, 83.3] | 71.4% [63.5, 79.4] | 4.8% [-2.4, 11.9] | -0.237 [-0.478, +0.007] |

## Original prespecified conflict-choice contrasts

- **fresh_minus_random:** Game 3.2% [-0.8, 7.9]; Neutral -0.8% [-4.8, 3.2]; interaction 4.0% [-2.4, 10.3].
- **joint_minus_matching_random:** Game 2.4% [-3.2, 7.9]; Neutral 1.6% [-4.0, 7.1]; interaction 0.8% [-5.6, 7.1].
- **matching_minus_natural:** Game -1.6% [-11.1, 7.9]; Neutral -0.8% [-11.9, 10.3]; interaction -0.8% [-9.5, 7.9].
- **fresh_minus_natural:** Game 3.2% [-0.8, 7.9]; Neutral -0.8% [-4.8, 3.2]; interaction 4.0% [-2.4, 10.3].
- **joint_minus_matching:** Game 3.2% [-2.4, 8.7]; Neutral 1.6% [-4.0, 7.1]; interaction 1.6% [-4.8, 7.9].

## Confirmation destination choice

The fixed destination subset contains questions where fresh W2 differs from both old W1 and old R2.

| Scenario | Task | Fresh-W2 choice | Old-R2 choice | Fresh W2 − old R2 |
|---|---|---:|---:|---:|
| complete_path_natural | Game | 35.9% [25.0, 48.4] | 17.2% [7.8, 26.6] | 18.8% [1.6, 35.9] |
| complete_path_natural | Neutral | 34.4% [23.4, 45.3] | 20.3% [10.9, 31.2] | 14.1% [-4.7, 31.2] |
| fresh_scrub | Game | 35.9% [23.4, 48.4] | 17.2% [7.8, 26.6] | 18.8% [1.6, 35.9] |
| fresh_scrub | Neutral | 34.4% [23.4, 46.9] | 18.8% [9.4, 28.1] | 15.6% [-1.6, 32.8] |
| dose_matched_random | Game | 35.9% [25.0, 48.4] | 17.2% [7.8, 26.6] | 18.8% [1.6, 35.9] |
| dose_matched_random | Neutral | 34.4% [23.4, 46.9] | 20.3% [10.9, 29.7] | 14.1% [-3.1, 31.2] |
| matching_history_blockade | Game | 39.1% [28.1, 51.6] | 23.4% [14.1, 34.4] | 15.6% [-3.1, 34.4] |
| matching_history_blockade | Neutral | 28.1% [17.2, 39.1] | 34.4% [23.4, 46.9] | -6.2% [-25.0, 12.5] |
| matching_plus_fresh | Game | 45.3% [32.8, 57.8] | 21.9% [12.5, 32.8] | 23.4% [4.7, 42.2] |
| matching_plus_fresh | Neutral | 32.8% [21.9, 45.3] | 26.6% [15.6, 37.5] | 6.2% [-12.5, 25.0] |
| matching_plus_random | Game | 39.1% [26.6, 51.6] | 25.0% [15.6, 35.9] | 14.1% [-4.7, 32.8] |
| matching_plus_random | Neutral | 29.7% [18.8, 40.6] | 34.4% [23.4, 46.9] | -4.7% [-25.0, 15.6] |

## Destination causal contrasts

- **fresh_minus_random, destination_fresh_winner_choice:** Game 0.0% [-6.2, 6.2]; Neutral 0.0% [-6.2, 6.2]; interaction 0.0% [-9.4, 7.8].
- **fresh_minus_random, destination_old_runner_choice:** Game 0.0% [-4.7, 4.7]; Neutral -1.6% [-6.2, 3.1]; interaction 1.6% [-4.7, 7.8].
- **fresh_minus_random, destination_fresh_minus_old_runner:** Game 0.0% [-9.4, 7.8]; Neutral 1.6% [-7.8, 12.5]; interaction -1.6% [-15.6, 10.9].
- **joint_minus_matching_random, destination_fresh_winner_choice:** Game 6.2% [-1.6, 14.1]; Neutral 3.1% [-4.7, 10.9]; interaction 3.1% [-6.2, 12.5].
- **joint_minus_matching_random, destination_old_runner_choice:** Game -3.1% [-9.4, 3.1]; Neutral -7.8% [-15.6, -1.6]; interaction 4.7% [-4.7, 14.1].
- **joint_minus_matching_random, destination_fresh_minus_old_runner:** Game 9.4% [-1.6, 21.9]; Neutral 10.9% [0.0, 23.4]; interaction -1.6% [-17.2, 14.1].

## Manipulation checks

- **identity_hook:** fresh fraction remaining 1.0000; mean |old-coordinate change| 0; mean L2 dose 0.
- **fresh_scrub:** fresh fraction remaining 0.0274; mean |old-coordinate change| 0.00266525; mean L2 dose 2.33615.
- **matching_plus_fresh:** fresh fraction remaining 0.0317; mean |old-coordinate change| 0.00235577; mean L2 dose 2.15896.
- **dose_matched_random:** fresh fraction remaining 0.9998; mean |old-coordinate change| 0.00127073; mean L2 dose 2.33615.
- **matching_plus_random:** fresh fraction remaining 0.9998; mean |old-coordinate change| 0.0011103; mean L2 dose 2.15896.
