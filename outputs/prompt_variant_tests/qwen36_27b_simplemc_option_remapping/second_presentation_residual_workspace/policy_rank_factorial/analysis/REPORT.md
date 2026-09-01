# Policy × retrieved-rank causal factorial

## Method

The action-matched Game and Neutral prompts differ only at `incorrect` versus `lost`. For each recipient condition, the experiment reciprocally transplanted the evaluation-closing period's GLA write across all 48 GLA layers, blocked all four complete matching 1P-to-2P option-line routes across all 16 ordinary-attention layers, and crossed those interventions. A cyclic nonmatching-line blockade is the route control. A final condition restored the natural recipient MLP-49 output at all four final 2P semantic tokens while leaving the policy swap intact.

## Scope of the policy transplant

This historical factorial used `preserve_source_output=False`. It swaps the recurrent GLA write and also allows the donor-conditioned GLA output at the evaluation-closing period to flow onward. It therefore causally tests the complete evaluation-period GLA update, not an output-preserved isolation of persistent recurrent memory alone.

Natural A--D logits reproduce exactly. Discrete answers resolve exact ties in displayed A--D order before mapping the selected letter to semantic rank: 18 scenario cells were tied and 15 choices differ from the invalid reorder-before-argmax rule. Discovery contains 251 questions and confirmation 249.

## Interpretation

The evaluation-period GLA update causally changes how the matching route uses retrieved rank. The reciprocal interaction replicates in both frozen splits and reaches conflict-trial W1 choices. Restoring the natural MLP-49 output removes the local nominated write difference but leaves nearly all of the final and behavioral policy effect, so MLP 49 is a readout rather than a necessary local mediator.

Rankwise lesion levels jointly block all four matching routes versus all four cyclic controls; they are not the earlier four separate single-route estimates.

## Replication across frozen splits

| Task | Split | Policy × route bivalent interaction | Policy effect at MLP 49 | Policy effect on final evidence | Remaining after MLP-49 restore |
|---|---|---:|---:|---:|---:|
| Game | Discovery | +0.453 [+0.356, +0.551] | -0.278 [-0.346, -0.211] | -0.410 [-0.496, -0.325] | -0.401 [-0.490, -0.311] |
| Game | Confirmation | +0.426 [+0.335, +0.519] | -0.262 [-0.323, -0.205] | -0.440 [-0.530, -0.351] | -0.412 [-0.501, -0.320] |
| Neutral | Discovery | -0.773 [-0.895, -0.644] | +0.422 [+0.343, +0.506] | +0.805 [+0.676, +0.942] | +0.807 [+0.676, +0.941] |
| Neutral | Confirmation | -0.727 [-0.848, -0.606] | +0.396 [+0.324, +0.473] | +0.759 [+0.624, +0.900] | +0.740 [+0.602, +0.873] |

Rankwise tables below report the untouched confirmation split.

## Game

### Natural-policy matching-specific effects by old rank

| Rank | Lesion effect (logits) |
|---|---:|
| R1 | +0.440 [+0.260, +0.620] |
| R2 | +0.113 [-0.043, +0.281] |
| R3 | -0.104 [-0.230, +0.028] |
| R4 | -0.449 [-0.543, -0.352] |

### After reciprocal policy swap

| Rank | Lesion effect (logits) | Policy × route interaction |
|---|---:|---:|
| R1 | -0.042 [-0.214, +0.133] | -0.482 [-0.577, -0.387] |
| R2 | +0.182 [+0.041, +0.334] | +0.068 [-0.016, +0.159] |
| R3 | +0.091 [-0.042, +0.229] | +0.195 [+0.115, +0.277] |
| R4 | -0.230 [-0.333, -0.121] | +0.219 [+0.157, +0.278] |

Bivalent policy × route interaction: +0.426 [+0.335, +0.519].

Policy-swap effect on MLP-49 bivalent rank write: -0.262 [-0.323, -0.205].
Policy-swap effect on final bivalent candidate evidence: -0.440 [-0.530, -0.351].
Remaining final bivalent effect after restoring natural MLP 49: -0.412 [-0.501, -0.320].

### Conflict-trial W1 choice

- Natural: 17.6%.
- Policy swapped: 33.1%.
- Policy swapped with natural MLP 49 restored: 30.9%.
- Policy-swap effect: +15.441 [+8.824, +22.794] percentage points.
- Remaining effect after MLP-49 restoration: +13.235 [+6.618, +19.853] percentage points.

## Neutral

### Natural-policy matching-specific effects by old rank

| Rank | Lesion effect (logits) |
|---|---:|
| R1 | -0.248 [-0.443, -0.056] |
| R2 | +0.235 [+0.072, +0.395] |
| R3 | +0.146 [+0.002, +0.287] |
| R4 | -0.132 [-0.244, -0.015] |

### After reciprocal policy swap

| Rank | Lesion effect (logits) | Policy × route interaction |
|---|---:|---:|
| R1 | +0.572 [+0.381, +0.763] | +0.820 [+0.689, +0.954] |
| R2 | +0.086 [-0.084, +0.250] | -0.149 [-0.274, -0.029] |
| R3 | -0.134 [-0.270, +0.005] | -0.280 [-0.372, -0.182] |
| R4 | -0.524 [-0.630, -0.417] | -0.391 [-0.473, -0.312] |

Bivalent policy × route interaction: -0.727 [-0.848, -0.606].

Policy-swap effect on MLP-49 bivalent rank write: +0.396 [+0.324, +0.473].
Policy-swap effect on final bivalent candidate evidence: +0.759 [+0.624, +0.900].
Remaining final bivalent effect after restoring natural MLP 49: +0.740 [+0.602, +0.873].

### Conflict-trial W1 choice

- Natural: 39.7%.
- Policy swapped: 20.6%.
- Policy swapped with natural MLP 49 restored: 22.8%.
- Policy-swap effect: -19.118 [-26.471, -11.765] percentage points.
- Remaining effect after MLP-49 restoration: -16.912 [-24.265, -9.559] percentage points.

## Evidence status

The policy and route manipulations are causal. MLP-49 restoration is a direct mediation test at the four nominated semantic-token positions. Rank summaries and confidence intervals were computed separately on the frozen discovery and confirmation questions; the report above gives confirmation results.

