# Llama 3.1 405B Second Chance observational mechinterp

## Bottom line

The native logit lens does add something beyond the behavioral/logprob analysis, but it does **not** support a single, simple “winner emerges and is then suppressed” account.

The best cross-dataset account is a multi-stage reconsideration process:

1. Around block 36, Game feedback prevents a strong leader-amplifying update that occurs in neutral re-answering. This is a feedback-specific gate, but not yet literal negative suppression: the mean Game leader update is approximately zero.
2. Around blocks 39–45, Game applies a large, genuinely negative update to the current leader together with broad A–D compression. Both effects replicate on SimpleMC and TriviaMC.
3. After ordinary winner emergence (median readout 48 on SimpleMC and 45 on TriviaMC), the Game trajectory becomes selectively less favorable to the ordinary winner than neutral. This event-locked effect is much stronger on TriviaMC.
4. On baseline-wrong trials, late layers selectively increase the objectively correct option beyond what broad compression and original-winner suppression predict. This correctness-directed component is present on both datasets and grows toward the final layers.
5. TriviaMC additionally shows a margin-dependent leader penalty: high-margin leaders receive a much more negative Game update, and a thresholded-leader term adds substantial held-out explanatory value. The corresponding effect is weak on SimpleMC.

Thus the most plausible current story is **broad compression plus late correctness-directed recomputation, with an additional thresholded leader penalty on high-margin/TriviaMC trajectories**.

This is observational localization, not a causal circuit result.

## Collection and validation

The experiment used the behaviorally validated `RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic` checkpoint through vLLM's native compressed-tensors path on 8×A100 80 GB GPUs.

- 495 usable SimpleMC questions × baseline, Game, and neutral.
- 495 usable TriviaMC questions × baseline, Game, and neutral.
- 127 readouts per prompt: embedding plus 126 post-block residual streams.
- At every readout, the actual final RMSNorm and the canonical and leading-space A–D unembedding rows were applied.
- Exact behavioral prompt lengths agree on 100% of shards in every dataset and condition.
- All 2,970 shards have the expected shapes and finite values.
- Canonical and leading-space variants choose the same letter on 100% of shards.
- Where same-run generation validation metadata is available, selected-row lens and generated-letter agreement is 98.6–100%, depending on dataset and condition. The rare flips are retained and flagged; they arise in close races between vLLM's distributed full-vocabulary argmax and the separate selected-row matmul.
- Against the independently compiled behavioral run, final canonical-lens agreement is 93.9–94.7% on SimpleMC and 97.4–98.6% on TriviaMC. This is consistent with the already observed trial-level FP8/rerun variation; aggregate switch rates remain close to the behavioral replication.

The generic Hugging Face loader was rejected after it attempted to dequantize the checkpoint and exceeded GPU memory. No activations from that incorrect path were analyzed.

## Final-readout effects

All values below are paired mean pseudo-logit differences. “Winner advantage” is the final baseline winner versus the other three options. “Runner advantage” is the baseline runner-up versus ranks 3 and 4 only, so winner suppression cannot masquerade as runner boosting.

| Dataset | Contrast | A–D spread | Winner advantage | Runner vs. ranks 3–4 |
|---|---:|---:|---:|---:|
| SimpleMC | Game − baseline | +0.214 | −0.097 | −0.107 |
| SimpleMC | Neutral − baseline | +0.470 | +0.240 | +0.118 |
| SimpleMC | Game − neutral | **−0.255** | **−0.337** | **−0.225** |
| TriviaMC | Game − baseline | **−0.819** | **−1.222** | +0.034 |
| TriviaMC | Neutral − baseline | +0.696 | +0.455 | +0.109 |
| TriviaMC | Game − neutral | **−1.514** | **−1.677** | −0.076 |

The strategic Game-minus-neutral effect is compression plus reduced winner advantage on both datasets. There is no independent final runner-up boost; runner strength relative to ranks 3 and 4 is slightly lower in Game than neutral.

TriviaMC shows a much stronger final effect than SimpleMC, even though both datasets behaviorally pass the Second Chance tests.

## Block-localized changes

The current leader is defined dynamically from the Game trajectory immediately before each block. Positive compression means that the block update projects onto the negative centered A–D evidence vector.

### Block 36: feedback gates off leader amplification

| Dataset | Game leader update | Game − baseline | Game − neutral | Game compression | Game − neutral compression |
|---|---:|---:|---:|---:|---:|
| SimpleMC | +0.012 [0.009, 0.015] | −0.112 [−0.116, −0.108] | −0.549 [−0.552, −0.546] | +0.169 | +0.244 |
| TriviaMC | +0.005 [0.001, 0.008] | −0.114 [−0.118, −0.110] | −0.565 [−0.568, −0.562] | +0.229 | +0.309 |

Neutral strongly amplifies the current leader at this block (about +0.56 on both datasets), while Game leaves it nearly unchanged. It is accurate to call this a feedback-specific inhibition of leader amplification, but not literal negative suppression.

### Block 39: broad compression and actual leader suppression

| Dataset | Game leader update | Game − baseline | Game − neutral | Game compression | Game − baseline compression | Game − neutral compression |
|---|---:|---:|---:|---:|---:|---:|
| SimpleMC | **−0.609** [−0.614, −0.605] | −0.350 | −0.361 | **+0.739** | +0.087 | +0.361 |
| TriviaMC | **−0.616** [−0.621, −0.612] | −0.371 | −0.376 | **+0.808** | +0.197 | +0.396 |

This is the cleanest replicated block-level result. The Game current leader is actually pushed downward, not merely less boosted, and the complete A–D update is strongly compression-like.

### Block 45: a second negative leader update near ordinary emergence

| Dataset | Game leader update | Game − baseline | Game − neutral | Game compression | Game − neutral compression |
|---|---:|---:|---:|---:|---:|
| SimpleMC | **−0.452** [−0.501, −0.401] | −0.399 | −0.281 | +0.265 | +0.151 |
| TriviaMC | **−0.673** [−0.737, −0.606] | −0.228 | −0.321 | +0.275 | +0.143 |

This occurs close to the median baseline-winner emergence readout (45–48), so it is the best absolute-layer candidate for the proposed “leader has emerged, then gets suppressed” operation. Broad compression still explains more of the four-option differential update than leader identity alone.

## Event alignment to winner emergence

Winner emergence was defined with a two-fold cross-fitted margin threshold and a short stability requirement. The threshold was calibrated on the opposite fold, then applied to held-out trials.

- SimpleMC median emergence: readout 48.
- TriviaMC median emergence: readout 45.

For the paired Game-minus-neutral original-winner advantage:

| Dataset | 5 readouts before | At emergence | 5 after | 10 after | 20 after |
|---|---:|---:|---:|---:|---:|
| SimpleMC | +0.080 [0.031, 0.125] | −0.051 [−0.105, −0.001] | −0.016 [−0.049, 0.017] | +0.044 [0.007, 0.080] | −0.081 [−0.121, −0.042] |
| TriviaMC | +0.169 [0.107, 0.230] | +0.048 [−0.013, 0.116] | **−0.387** [−0.443, −0.329] | **−0.319** [−0.377, −0.264] | **−0.374** [−0.427, −0.312] |

TriviaMC shows the predicted timing particularly clearly: Game is not suppressing the ordinary winner before emergence, but becomes substantially less favorable to it several readouts after emergence. SimpleMC does not show the same clean sustained event-locked transition.

## Compression versus fixed-winner and thresholded-leader terms

The cumulative Game-minus-neutral A–D difference was decomposed using held-out questions into:

- baseline A–D geometry (broad compression),
- final baseline-winner identity,
- current baseline-layer leader with a trained-only margin threshold,
- fixed option-letter nuisance effects.

At the final readout:

| Dataset | Compression strength | Separate fixed-winner penalty | Threshold term added held-out R² |
|---|---:|---:|---:|
| SimpleMC | **0.083** [0.058, 0.107] | −0.074 [−0.200, 0.035] | 0.012 |
| TriviaMC | **0.172** [0.152, 0.191] | −0.136 [−0.380, 0.117] | 0.051 |

After accounting for compression, there is no positive fixed-winner penalty at the final readout. If anything, its point estimate offsets some of the compression-induced winner loss.

TriviaMC nevertheless contains a clear intermediate thresholded signal. At readout 63:

- Game-minus-neutral compression: 0.040 [0.020, 0.064].
- Separate original-winner penalty: 0.240 [0.188, 0.288].
- Adding the thresholded current-layer leader improves held-out R² by 0.193 beyond compression plus fixed-winner identity.
- The fitted hinge is negative (−0.550) above a mean threshold of 0.62 pseudo-logits.

SimpleMC at readout 63 has comparable weak compression (0.036 [0.019, 0.051]) but no reliable fixed-winner penalty and no added value from the threshold rule.

Across all blocks, the highest Game leader-margin quintile has a strongly negative Game-minus-baseline relative update on TriviaMC (−0.185, trial-clustered 95% CI [−0.224, −0.148]); the SimpleMC estimate is −0.019 with a CI crossing zero. This is evidence for thresholded leader suppression on TriviaMC, not a universal cross-dataset rule.

## Correctness-directed recomputation

This analysis uses only trials where the ordinary baseline answer was wrong: 199 SimpleMC and 94 TriviaMC questions. It predicts the paired Game-minus-neutral A–D change with baseline geometry, original-winner identity, fixed option-letter effects, and an objectively correct-option indicator. Evaluation is five-fold by question; coefficient uncertainty is clustered and stratified by correct letter.

| Dataset | Final correct-option advantage, Game − neutral | Controlled correct-option coefficient | 95% CI | Incremental held-out R² |
|---|---:|---:|---:|---:|
| SimpleMC | +0.433 [0.305, 0.573] | **+0.424** | [0.296, 0.561] | 0.073 |
| TriviaMC | +0.844 [0.554, 1.142] | **+0.848** | [0.566, 1.132] | 0.087 |

The controlled coefficient is already positive at readout 63 and grows late:

| Dataset | Readout 63 | 94 | 110 | 120 | 126 |
|---|---:|---:|---:|---:|---:|
| SimpleMC | +0.063 | +0.074 | +0.109 | +0.193 | +0.424 |
| TriviaMC | +0.099 | +0.326 | +0.370 | +0.553 | +0.848 |

This is not merely the centered runner-up rising when the winner is suppressed: the effect survives direct control for baseline geometry and winner identity and is evaluated out of sample. It is consistent with the model recomputing question evidence and preferentially recovering the correct alternative in late layers.

## Hypothesis assessment

### 1. Thresholded current-leader suppression

**Qualified support, strongest on TriviaMC.** Blocks 39 and 45 make genuinely negative, feedback-specific updates to the current Game leader. TriviaMC also shows a strong high-margin effect, event-locked post-emergence suppression, and held-out value from a thresholded leader term. SimpleMC has the replicated block-localized negative updates but not a stable margin-threshold signature.

### 2. Targeted original-winner suppression

**Not supported as the primary general mechanism.** Fixed original-winner identity adds little after compression at the final readout. TriviaMC has an intermediate fixed-winner penalty, but dynamic threshold information explains substantially more there. This is more consistent with targeting a currently strong candidate than with remembering and penalizing one fixed letter throughout the computation.

### 3. Generic compression plus perturbation

**Strongest replicated backbone.** Broad compression dominates the four-option update around block 39 and remains the clearest final Game-minus-neutral signature on both datasets. It is not mere isotropic noise: it is structured movement toward reduced A–D evidence separation.

### 4. Independent runner-up boosting

**Not supported.** Using the correct runner-versus-ranks-3-and-4 contrast, Game-minus-neutral is −0.225 on SimpleMC and −0.076 on TriviaMC at the final readout. Earlier apparent runner gains were artifacts of comparing the runner to a mean that included the suppressed winner.

### 5. Correctness-directed recomputation

**Supported as a late complementary mechanism.** On baseline-wrong trials, the correct option gains beyond compression and winner identity, with a growing late-layer trajectory and held-out replication across datasets.

## Current mechanistic story

The most economical account is:

> Negative feedback first prevents ordinary re-answering from amplifying the current leader. A few blocks later, the model broadly compresses A–D evidence and makes a negative update to the current leader. Once an ordinary winner has emerged, high-margin TriviaMC trajectories receive an additional thresholded leader penalty. In later layers, the model recomputes task evidence and selectively increases the correct alternative when its baseline answer was wrong.

This is more structured than generic noise, but less simple than “wait for any leader, suppress it, and let the runner-up win.” It combines control over confidence/competition with renewed content-level inference.

## Limitations and next step

- Intermediate values are native logit-lens pseudo-logits, not behavioral logits.
- The decomposition is observational. It identifies when and in what A–D geometry the computation changes, not which internal component causes it.
- Baseline, Game, and neutral prompts differ in length and content; baseline remains the conceptual primary comparison, while Game-minus-neutral isolates strategic specificity.
- Event thresholds and regression thresholds were selected only on training folds, but model families were chosen in advance rather than preregistered formally.
- No probes, component attribution, patching, or steering were run in this pass.

The best causal follow-up is now narrower than before: focus on blocks 36–45 for the feedback gate/compression operation, then blocks 94–126 for correctness-directed evidence. A causal experiment should separately test whether perturbing the early compression/leader update changes switching and whether perturbing the late correct-option signal changes the probability of switching specifically to the correct alternative.

## Main artifacts

- `LLAMA405_OBSERVATIONAL_PLAN.md`
- `outputs/mechanistic/llama31_405b_simplemc/analysis/leader_dynamics.png`
- `outputs/mechanistic/llama31_405b_triviamc/analysis/leader_dynamics.png`
- `outputs/mechanistic/llama31_405b_simplemc/analysis/cumulative_hypothesis_decomposition.png`
- `outputs/mechanistic/llama31_405b_triviamc/analysis/cumulative_hypothesis_decomposition.png`
- `outputs/mechanistic/llama31_405b_simplemc/analysis/correctness_recomputation.png`
- `outputs/mechanistic/llama31_405b_triviamc/analysis/correctness_recomputation.png`
- Per-readout CSVs, bootstrap intervals, held-out fit tables, collection audits, and JSON summaries are stored beside those figures.

## Compute and instance state

- GPU instance: Vast `46647487`, 8×A100 SXM4 80 GB.
- Estimated GPU charge for this batch: approximately **$9.68**.
- The instance was stopped after retrieval; it was not destroyed. The downloaded 382 GB checkpoint remains on its negligible-cost stopped volume for follow-up work.
