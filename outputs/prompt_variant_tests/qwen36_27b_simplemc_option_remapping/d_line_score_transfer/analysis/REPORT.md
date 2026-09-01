# D-line old-score transfer and D-closing-state crossover

## Bottom line

This report keeps the local candidate-line and global D-closing-state hypotheses separate. The table reports direct within-task effects; Game-minus-Neutral contrasts follow only after those effects.

## Discovery

| Task | Current evidence | Complete D-line K/V | D-closing state | Complete history |
|---|---|---:|---:|---:|
| Game | low | +0.331 [+0.244, +0.419] | +0.161 [+0.095, +0.238] | +0.311 [+0.141, +0.486] |
| Game | high | +0.299 [+0.217, +0.386] | +0.169 [+0.103, +0.243] | +0.978 [+0.766, +1.169] |
| Neutral | low | +0.417 [+0.302, +0.554] | +0.239 [+0.132, +0.374] | +0.383 [+0.207, +0.559] |
| Neutral | high | +0.396 [+0.289, +0.519] | +0.249 [+0.156, +0.364] | +1.399 [+1.161, +1.653] |

Exact eligible: 35/70.
The frozen screen chose high/low histories using the earlier 24-permutation forward path. Exact eligibility was then rechecked inside the cached causal cohort and required the target's measured centered first-decision score to remain higher in the nominated high history than in the nominated low history. Rows lost here are numerical/path-regime reversals of that ordering, not intervention-outcome exclusions.

## Confirmation

| Task | Current evidence | Complete D-line K/V | D-closing state | Complete history |
|---|---|---:|---:|---:|
| Game | low | +0.387 [+0.299, +0.488] | +0.168 [+0.119, +0.220] | +0.485 [+0.211, +0.790] |
| Game | high | +0.359 [+0.279, +0.447] | +0.190 [+0.146, +0.235] | +0.996 [+0.766, +1.253] |
| Neutral | low | +0.440 [+0.331, +0.585] | +0.186 [+0.138, +0.238] | +0.392 [+0.064, +0.770] |
| Neutral | high | +0.446 [+0.333, +0.588] | +0.224 [+0.171, +0.284] | +1.546 [+1.299, +1.822] |

Exact eligible: 36/70.
The frozen screen chose high/low histories using the earlier 24-permutation forward path. Exact eligibility was then rechecked inside the cached causal cohort and required the target's measured centered first-decision score to remain higher in the nominated high history than in the nominated low history. Rows lost here are numerical/path-regime reversals of that ordering, not intervention-outcome exclusions.

## Interpretation rules

- A replicating D-line K/V effect means the text-identical final candidate line carries causally usable information about its old value.
- A replicating D-closing-state effect means that one closing token carries portable old-history information. Because D is also the target's own line, the target-logit effect alone does not distinguish a local target state from a global comparison summary; that distinction requires the four-candidate transfer-vector analysis.
- Current-high versus current-low differences test whether either old state is combined non-additively with fresh evidence.
- A complete-history effect alone confirms history dependence but does not localize the information.

## Validation

- Discovery: {"all_complete": true, "all_exact_rows_used": 35, "max_full_history_decision_error": 0.0, "max_full_history_final_error": 0.0, "max_trajectory_identity_decision_error": 0.6251697540283203, "max_trajectory_identity_final_error": 0.6248779296875, "model_calls": [20], "ordinary_layer_counts": [16]}
- Confirmation: {"all_complete": true, "all_exact_rows_used": 36, "max_full_history_decision_error": 0.0, "max_full_history_final_error": 0.0, "max_trajectory_identity_decision_error": 0.4999427795410156, "max_trajectory_identity_final_error": 0.6238765716552734, "model_calls": [20], "ordinary_layer_counts": [16]}

Canonical figure: `figures/qwen36_d_line_score_transfer.png`.
