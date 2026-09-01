# Remapped A-D argmax tie audit

The affected analysis pattern reordered displayed A-D logits into original semantic-content order and then took `argmax`. On exact ties this changed the model's actual displayed-letter tie-break. The corrected analysis resolves the displayed A-D winner first and only then maps it to semantic content.

| Experiment | Split | Result array | Exact max ties | Changed semantic choices | Total cells |
|---|---|---|---:|---:|---:|
| continuous final-decision signed ablation | discovery | `natural_logits` | 8 | 8 (1.59%) | 502 |
| continuous final-decision signed ablation | discovery | `ablated_logits` | 3 | 2 (0.40%) | 502 |
| continuous final-decision signed ablation | confirmation | `natural_logits` | 4 | 4 (0.80%) | 498 |
| continuous final-decision signed ablation | confirmation | `ablated_logits` | 2 | 2 (0.40%) | 498 |
| first-span GLA lesions | discovery | `natural_logits` | 8 | 8 (1.59%) | 502 |
| first-span GLA lesions | discovery | `ablated_logits` | 22 | 15 (0.75%) | 2008 |
| first-span GLA lesions | confirmation | `natural_logits` | 4 | 4 (0.80%) | 498 |
| first-span GLA lesions | confirmation | `ablated_logits` | 23 | 11 (0.55%) | 1992 |
| first-decision cross-order patching | discovery | `natural_logits` | 4 | 3 (0.60%) | 502 |
| first-decision cross-order patching | discovery | `donor_patched_logits` | 40 | 26 (0.47%) | 5522 |
| first-decision cross-order patching | discovery | `identity_patched_logits` | 44 | 33 (0.60%) | 5522 |
| first-decision cross-order patching | confirmation | `natural_logits` | 4 | 4 (0.80%) | 498 |
| first-decision cross-order patching | confirmation | `donor_patched_logits` | 3 | 2 (0.40%) | 498 |
| first-decision cross-order patching | confirmation | `identity_patched_logits` | 4 | 4 (0.80%) | 498 |
| first-boundary GLA memory rewrite | discovery | `natural_logits` | 8 | 8 (1.59%) | 502 |
| first-boundary GLA memory rewrite | discovery | `donor_patched_logits` | 3 | 2 (0.40%) | 502 |
| first-boundary GLA memory rewrite | discovery | `identity_patched_logits` | 8 | 8 (1.59%) | 502 |
| first-boundary GLA memory rewrite | confirmation | `natural_logits` | 4 | 4 (0.80%) | 498 |
| first-boundary GLA memory rewrite | confirmation | `donor_patched_logits` | 6 | 3 (0.60%) | 498 |
| first-boundary GLA memory rewrite | confirmation | `identity_patched_logits` | 4 | 4 (0.80%) | 498 |
| first-boundary accumulated-state transplant | discovery | `natural_logits` | 8 | 8 (1.59%) | 502 |
| first-boundary accumulated-state transplant | discovery | `identity_state_logits` | 5 | 3 (0.60%) | 502 |
| first-boundary accumulated-state transplant | discovery | `different_winner_state_logits` | 5 | 1 (0.20%) | 502 |
| first-boundary accumulated-state transplant | discovery | `same_winner_state_logits` | 5 | 3 (0.60%) | 502 |
| first-boundary accumulated-state transplant | confirmation | `natural_logits` | 4 | 4 (0.80%) | 498 |
| first-boundary accumulated-state transplant | confirmation | `identity_state_logits` | 2 | 2 (0.40%) | 498 |
| first-boundary accumulated-state transplant | confirmation | `different_winner_state_logits` | 2 | 1 (0.20%) | 498 |
| first-boundary accumulated-state transplant | confirmation | `same_winner_state_logits` | 5 | 4 (0.80%) | 498 |
| all_candidate_matched_relay/run | stored | `baseline_logits` | 6 | 4 (0.80%) | 500 |
| all_candidate_matched_relay/run | stored | `natural_logits` | 4 | 4 (0.40%) | 1000 |
| all_candidate_matched_relay/run | stored | `trusted_natural_logits` | 4 | 4 (0.40%) | 1000 |
| all_candidate_matched_relay/run | stored | `matched_logits` | 22 | 14 (0.35%) | 4000 |
| all_candidate_matched_relay/run | stored | `control_logits` | 24 | 16 (0.40%) | 4000 |
| all_candidate_matched_relay/run | stored | `joint_logits` | 3 | 1 (0.10%) | 1000 |
| all_candidate_matched_relay_full_range/run | stored | `baseline_logits` | 6 | 4 (0.80%) | 500 |
| all_candidate_matched_relay_full_range/run | stored | `natural_logits` | 4 | 4 (0.40%) | 1000 |
| all_candidate_matched_relay_full_range/run | stored | `trusted_natural_logits` | 4 | 4 (0.40%) | 1000 |
| all_candidate_matched_relay_full_range/run | stored | `matched_logits` | 40 | 31 (0.39%) | 8000 |
| all_candidate_matched_relay_full_range/run | stored | `control_logits` | 33 | 22 (0.27%) | 8000 |
| all_candidate_matched_relay_full_range/run | stored | `joint_matched_logits` | 14 | 9 (0.45%) | 2000 |
| all_candidate_matched_relay_full_range/run | stored | `joint_control_logits` | 8 | 6 (0.30%) | 2000 |
| feedback_factorial/action_period_mediation/run | stored | `trusted_natural_logits` | 4 | 4 (0.40%) | 1000 |
| feedback_factorial/action_period_mediation/run | stored | `same_batch_natural_logits` | 4 | 4 (0.40%) | 1000 |
| feedback_factorial/action_period_mediation/run | stored | `identity_state_logits` | 6 | 4 (0.40%) | 1000 |
| feedback_factorial/action_period_mediation/run | stored | `patched_logits` | 7 | 5 (0.17%) | 3000 |
| feedback_factorial/action_period_source_lesion/run | stored | `trusted_natural_logits` | 4 | 4 (0.40%) | 1000 |
| feedback_factorial/action_period_source_lesion/run | stored | `same_batch_natural_logits` | 4 | 4 (0.40%) | 1000 |
| feedback_factorial/action_period_source_lesion/run | stored | `lesioned_logits` | 16 | 14 (0.47%) | 3000 |
| feedback_factorial/evaluation_update_transplant/confirmation_bands | stored | `trusted_natural_logits` | 2 | 2 (0.40%) | 498 |
| feedback_factorial/evaluation_update_transplant/confirmation_bands | stored | `patched_logits` | 0 | 0 (0.00%) | 996 |
| feedback_factorial/evaluation_update_transplant/confirmation_blocks | stored | `trusted_natural_logits` | 2 | 2 (0.40%) | 498 |
| feedback_factorial/evaluation_update_transplant/confirmation_blocks | stored | `patched_logits` | 7 | 6 (0.10%) | 5976 |
| feedback_factorial/evaluation_update_transplant/confirmation_gate | stored | `trusted_natural_logits` | 2 | 2 (0.40%) | 498 |
| feedback_factorial/evaluation_update_transplant/confirmation_gate | stored | `patched_logits` | 0 | 0 (0.00%) | 498 |
| feedback_factorial/evaluation_update_transplant/discovery_bands | stored | `trusted_natural_logits` | 2 | 2 (0.40%) | 502 |
| feedback_factorial/evaluation_update_transplant/discovery_bands | stored | `patched_logits` | 1 | 1 (0.02%) | 4016 |
| feedback_factorial/evaluation_update_transplant/discovery_blocks | stored | `trusted_natural_logits` | 2 | 2 (0.40%) | 502 |
| feedback_factorial/evaluation_update_transplant/discovery_blocks | stored | `patched_logits` | 2 | 2 (0.03%) | 6024 |
| feedback_factorial/evaluation_update_transplant/discovery_gate | stored | `trusted_natural_logits` | 2 | 2 (0.40%) | 502 |
| feedback_factorial/evaluation_update_transplant/discovery_gate | stored | `patched_logits` | 0 | 0 (0.00%) | 502 |
| final_query_edge_ablation/corrected_run | stored | `trusted_natural_logits` | 4 | 4 (0.40%) | 1000 |
| final_query_edge_ablation/corrected_run | stored | `same_batch_natural_logits` | 4 | 4 (0.40%) | 1000 |
| final_query_edge_ablation/corrected_run | stored | `intervention_logits` | 30 | 21 (0.35%) | 6000 |
| final_query_repeated_option_ablation/run | stored | `trusted_natural_logits` | 4 | 4 (0.40%) | 1000 |
| final_query_repeated_option_ablation/run | stored | `same_batch_natural_logits` | 7 | 6 (0.60%) | 1000 |
| final_query_repeated_option_ablation/run | stored | `intervention_logits` | 27 | 20 (0.50%) | 4000 |
| joint_option_score_decision_letter/run | stored | `logits` | 25 | 18 (0.36%) | 5000 |
| joint_option_score_decision_letter/run | stored | `first_decision_logits` | 9 | 6 (1.20%) | 500 |
| nonmatching_history_factorial/run | stored | `baseline_logits` | 6 | 4 (0.80%) | 500 |
| nonmatching_history_factorial/run | stored | `logits` | 31 | 21 (0.53%) | 4000 |
| nonmatching_history_factorial/run | stored | `trusted_natural_logits` | 4 | 4 (0.40%) | 1000 |
| option_newline_all_four_centered_projection | stored | `logits` | 18 | 14 (0.47%) | 3000 |
| option_newline_value_causal/confirmation | stored | `logits` | 1 | 1 (0.18%) | 568 |
| option_newline_value_causal/discovery | stored | `logits` | 4 | 4 (0.68%) | 592 |
| receiver_path_search/validation | stored | `trusted_natural_logits` | 4 | 4 (0.40%) | 1000 |
| receiver_path_search/validation | stored | `same_batch_natural_logits` | 4 | 4 (0.40%) | 1000 |
| receiver_path_search/validation | stored | `intervention_logits` | 115 | 94 (0.39%) | 24000 |
| repeated_w1_relay/run | stored | `trusted_natural_logits` | 4 | 4 (0.40%) | 1000 |
| repeated_w1_relay/run | stored | `same_batch_natural_logits` | 7 | 6 (0.60%) | 1000 |
| repeated_w1_relay/run | stored | `intervention_logits` | 20 | 15 (0.38%) | 4000 |
| second_presentation_residual_workspace/policy_rank_factorial/run | stored | `baseline_logits` | 6 | 4 (0.80%) | 500 |
| second_presentation_residual_workspace/policy_rank_factorial/run | stored | `trusted_natural_logits` | 4 | 4 (0.40%) | 1000 |
| second_presentation_residual_workspace/policy_rank_factorial/run | stored | `same_batch_natural_logits` | 4 | 4 (0.40%) | 1000 |
| second_presentation_residual_workspace/policy_rank_factorial/run | stored | `scenario_logits` | 18 | 15 (0.21%) | 7000 |

Only discrete answer identities and quantities derived from them (selection, change, accuracy, and transition counts) can move. Raw A-D logits, margins, entropy, projections, activation norms, and all other continuous causal effects are invariant to this correction.
