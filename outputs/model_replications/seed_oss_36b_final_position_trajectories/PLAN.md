# Seed-OSS 36B final-decision trajectories

## Question

Does Seed reproduce Qwen's organized final-decision trajectory: an eventual
switch is represented as R2 already leading R1 when semantic rank first becomes
output-readable, rather than as an initially prepared R1 answer that is later
suppressed? When does the final Game-versus-Neutral answer adjustment become
linearly decodable at the final decision position?

## Frozen scope

- **Model:** `ByteDance-Seed/Seed-OSS-36B-Instruct`, revision
  `497f1dca95ebdec98e41d517b9f060ee753c902f`, BF16, native chat template,
  `thinking_budget=0`.
- **Datasets:** all 500 frozen SimpleMC and all 500 frozen difficulty-filtered
  TriviaMC questions. Existing result-independent discovery/confirmation
  splits are reused for decoder fitting and evaluation.
- **Prompt:** current clean empty-first-assistant, non-remapped second
  presentation. Game and Neutral differ at exactly the single tokenizer token
  `incorrect`/`lost`.
- **Position:** the final prompt token immediately before the second assistant
  answer.
- **Layers:** every post-block residual L1--L64. No layer is omitted.
- **Candidate rank:** R1--R4 are frozen from Seed's same-format first-presentation
  aggregated A--D logits, using stable displayed-order tie handling.
- **Outcome slices:** all questions are primary. Switch and no-switch panels are
  descriptive postselection, defined separately within Game and Neutral by the
  final aggregated-A--D argmax.

## Readouts

No published or local Jacobian lens exists for this Seed revision. A Qwen
Jacobian matrix is model-specific and must not be reused. Two complementary
Seed-native measurements therefore preserve the scientific content of the
Qwen package:

1. **Standard logit lens:** apply Seed's exact final RMS norm and A--D
   unembedding rows to every post-block residual. This measures when answer
   evidence is directly output-readable. Produce raw, candidate-centered, and
   displayed-letter-controlled trajectories, with per-question similarity to
   the exact final A--D vector as background tint.
2. **Held-out prospective decoding:** at every layer, fit shared Game+Neutral,
   Game-only, and Neutral-only ridge decoders on discovery questions only to
   predict the exact final centered A--D vector. Evaluate only on confirmation
   questions, including cross-condition decoder transfer, W1-matched shuffle
   controls, switch-trial R2-minus-R1 timing, and the all-question paired
   Game-minus-Neutral policy-vector analysis.

The logit lens is an activation readout and the decoders are held-out linear
decoding evidence. Neither is a causal intervention. The previously completed
matching-history blockade supplies the separate causal evidence that matching
semantic recollection affects preferential Game W1 avoidance.

## Complete model work and validity

Two conditions × 500 questions × two datasets = 2,000 prompt sequences, or
500 complete batched forwards after one model load at batch size four. Each
forward captures the exact final-position residual after all 64 blocks. Require
all residuals, readout scores, and logits finite; all 64 hooks firing; exact
prompt hashes against the trusted Seed behavioral runs; exactly one
Game/Neutral token difference; and L64 logit-lens reconstruction error below
 0.10 logit when the stored L64 residual is passed through the model's native
 full final norm and output head. Store residuals as FP32 representations of the model's BF16
 values: the exact benchmark showed that an FP16 storage conversion alone
 produced a 0.113-logit L64 reconstruction error and therefore failed the
 frozen control when selected FP32 unembedding rows were incorrectly compared
 to the model's full BF16 head. L1--L63 remain the clearly labeled standard
 selected-row logit lens; L64 is exact native-head reconstruction/live logits.

The run is checkpointed after each complete condition/cohort and is resumable.
The complete exact path must be benchmarked before launch and the combined
batch remains under the standing $15 cap.

## Completed result

The complete 500-question SimpleMC and 500-question difficulty-filtered
TriviaMC runs passed every control and covered all post-block residuals
L1--L64. The standard logit lens makes the candidate-rank pattern directly
readable abruptly around L40--L42 in both datasets. On all questions, the
exact final Game-minus-Neutral effect selectively lowers first-presentation R1
and raises lower-ranked candidates: `-1.530/+0.121/+0.450/+0.958` centered
logits on SimpleMC and `-2.142/+0.766/+0.720/+0.656` on TriviaMC.

Held-out decoders show that the paired question-specific policy adjustment is
linearly accessible somewhat earlier than the fixed output readout: its cosine
with the exact final Game-minus-Neutral pattern becomes persistently positive
at L36 on SimpleMC and L39 on TriviaMC. At L40 the learned/readout cosines are
0.575/0.223 and 0.446/0.154, respectively. Shared-condition decoders perform
about as well as condition-specific decoders once the answer program emerges,
so Game and Neutral mostly use a common prospective-answer basis while
encoding different rank adjustments in it.

The narrow Qwen timing claim does not exactly replicate. On Seed's eventual
switch trials, R1 is often already visible when the late output-readable
pattern first separates, and R2 overtakes it later. These panels are
postselected activation descriptions, not evidence that an R1 answer was
causally prepared and then suppressed. The all-question paired policy analysis
is the primary timing result; the completed matching-history blockade remains
the causal evidence linking Seed's semantic recollection to preferential Game
avoidance of R1.

A completed matched-readout check rules out the main measurement confound:
Qwen's cached residuals were reread with Qwen's own standard logit lens, using
the same displayed-letter control used here. Qwen still remains nearly
unseparated until about L50 and then shows R2 already ahead on switch trials.
The Seed/Qwen difference therefore survives matching the readout method. It is
best treated as a descriptive difference in when each model exposes its
intermediate answer computation to its output vocabulary, not evidence for a
literal serial algorithm. [Matched Qwen comparison](../../../outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/standard_logit_lens/comparison/REPORT.md).

- [Trajectory report](analysis/REPORT.md) · [summary](analysis/summary.json)
- [Held-out decoder report](prospective_decoding/analysis/REPORT.md) ·
  [summary](prospective_decoding/analysis/summary.json)
- Paper-facing displayed-letter-controlled figures:
  [SimpleMC](../../../figures/model_replications/seed_oss_36b_simplemc_nonremapped_rank_trajectories_letter_controlled.png) ·
  [TriviaMC](../../../figures/model_replications/seed_oss_36b_triviamc_nonremapped_rank_trajectories_letter_controlled.png)
- Policy-adjusted decoder figures:
  [SimpleMC](../../../figures/prospective_decoding/seed_oss_36b_simplemc_policy_adjusted_prospective_decoding.png) ·
  [TriviaMC](../../../figures/prospective_decoding/seed_oss_36b_triviamc_policy_adjusted_prospective_decoding.png)

The authenticated provider rows increased by $0.492 for this trajectory batch,
including benchmarking, collection, retrieval transfer, and stop latency,
against the $15 authorization. Both retained Vast reservations were finalized
in the stopped state.
