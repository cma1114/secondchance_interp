# Fixed-A delta-rule recurrent-matrix transplant

Both histories produce the literal first answer `A`, but `A` names different semantic answers X and Y. At all 48 GLA layers jointly, the accumulated **delta-rule recurrent matrix** immediately after that answer boundary is exchanged X↔Y. Everything visible after the boundary is held fixed.

## Post-run scope correction

This was originally described as a transplant of the "complete accumulated GLA state." That description was incorrect. The runner intercepts `chunk_gated_delta_rule` and exchanges only its `last_recurrent_state` matrix. Qwen's Gated DeltaNet computes Q/K/V through a causal convolution before that function is called, so the suffix retains recipient-history convolutional features. The recipient residual trajectory and the prefix memories of conventional-attention blocks are also unchanged. The intervention therefore creates a hybrid of donor delta-rule matrices and recipient states on the other causal routes.

Consequently, this experiment tests whether the delta-rule matrices are, by themselves, a portable and sufficient semantic-answer memory. It does **not** test whether the complete first-boundary model state carries the selected answer, and it cannot show that the GLA stack contains no semantic-answer information.

Because splitting the recurrent kernel has a numerical effect, every causal estimate compares cross-semantic transplantation with reinserting the recipient's own state through the identical segmented computation.

## Results

| Split | Natural semantic-history effect | Game target transfer | Neutral target transfer | Game − Neutral | Game selection transfer | Neutral selection transfer |
|---|---:|---:|---:|---:|---:|---:|
| Discovery | +0.871 [+0.713, +1.050] logits | +0.050 [-0.008, +0.108] logits | +0.065 [+0.004, +0.129] logits | -0.015 [-0.059, +0.028] logits | +6.250 [+1.562, +11.719] pp | -1.562 [-7.031, +4.688] pp |
| Confirmation | +1.018 [+0.856, +1.201] logits | +0.030 [-0.038, +0.095] logits | +0.023 [-0.072, +0.111] logits | +0.008 [-0.043, +0.059] logits | +5.479 [-1.370, +12.329] pp | +0.685 [-5.479, +6.849] pp |

## Interpretation rule

If the transplanted delta-rule matrices were sufficient to carry semantic identity independently of the model's other recipient-history states, the held-out result should be positive in Game (the donor semantic answer becomes the one suppressed) and negative in Neutral (the donor answer becomes the one repeated). A null result rules out only that strong sufficiency claim.

## Validation

- Discovery maximum identity-reinsertion versus unsplit A–D logit difference: 0.625.
- Discovery mean cross-semantic GLA-state difference relative to recipient-state norm: 10.49%.
- Confirmation maximum identity-reinsertion versus unsplit A–D logit difference: 0.623.
- Confirmation mean cross-semantic GLA-state difference relative to recipient-state norm: 9.98%.

The replacement hosts did not reproduce the trusted natural logits bit-for-bit, despite using the same batch-of-four prompts and Transformers version:

| Split | Mean absolute A-D logit difference | Maximum difference | Answer differences |
|---|---:|---:|---:|
| Discovery | 0.134 | 0.874 | 9/256 |
| Confirmation | 0.150 | 0.867 | 10/292 |

The natural semantic-history interaction nevertheless reproduced closely. Every transplant effect is a within-host comparison against identity-state reinsertion under the identical segmented kernel, so the causal contrast does not compare raw logits across hosts.

## Host/shard sensitivity

| Shard | N | Game target transfer | Neutral target transfer | Game − Neutral |
|---|---:|---:|---:|---:|
| discovery_shard0 | 32 | +0.051 [-0.037, +0.136] logits | +0.029 [-0.073, +0.128] logits | +0.022 [-0.030, +0.070] logits |
| discovery_shard1 | 32 | +0.049 [-0.023, +0.125] logits | +0.102 [+0.029, +0.174] logits | -0.052 [-0.125, +0.015] logits |
| confirmation_shard0 | 37 | +0.088 [+0.004, +0.178] logits | +0.066 [-0.030, +0.175] logits | +0.023 [-0.053, +0.096] logits |
| confirmation_shard1 | 36 | -0.030 [-0.136, +0.063] logits | -0.022 [-0.183, +0.110] logits | -0.008 [-0.071, +0.064] logits |

## Bottom line

The natural fixed-A semantic-history effect is robust, but transplanting the delta-rule recurrent matrices alone does **not** produce a replicated semantic-target transfer. The held-out Game effect is small and uncertain, Neutral moves in the same rather than the predicted opposite direction, the Game-minus-Neutral contrast is essentially zero, and the shard estimates are heterogeneous. The transplant was genuinely applied: untouched rows were bit-exact, while targeted rows changed answer on 22/256 discovery cells and 32/292 confirmation cells. The perturbation was simply not aligned reliably with the donor semantic answer.

The valid conclusion is narrow: the delta-rule matrices are not independently sufficient as a clean portable representation of `which semantic answer I chose`. Because convolutional GLA history, conventional-attention memory, and the recipient residual trajectory were not jointly transplanted, this result does not locate or exclude the actual semantic-answer memory.
