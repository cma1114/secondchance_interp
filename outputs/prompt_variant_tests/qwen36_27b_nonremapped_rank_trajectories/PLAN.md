# Qwen3.6-27B non-remapped 1P-rank trajectories

## Execution status

Complete on all 500 questions in both datasets. All 2,000 condition/question rows and all JLens scores are finite; every audited Game/Neutral pair differs at exactly one tokenizer token. Six trajectory PNGs, compact arrays, and the [trajectory report](analysis/REPORT.md) are finalized and indexed from the root README. The prospective-answer follow-up is also complete at every L1–L64 layer, with [report](prospective_decoding/analysis/REPORT.md), [machine-readable summary](prospective_decoding/analysis/summary.json), and two top-level figures. Both retained Vast instances are stopped.

## Question

At the final decision token in the current clean paradigm, how do candidates ranked R1–R4 by the first presentation evolve across all 64 model blocks in Game and Neutral, on SimpleMC and difficulty-filtered TriviaMC? Show the full population and, descriptively, trials on which the task's final aggregated A-D choice does or does not leave first-presentation R1.

## Frozen scope

- Model and revision: Qwen3.6-27B at the repository-pinned revision, BF16, reasoning disabled, raw Qwen ChatML.
- Prompts: explicit empty first assistant response. The question and options are repeated without remapping. Game and Neutral are identical except for the single `incorrect`/`lost` tokenizer token in `Your answer was ... Choose the answer again.`
- Datasets: all 500 frozen SimpleMC questions and all 500 frozen difficulty-filtered TriviaMC questions. No discovery-only truncation.
- Position: only the final prompt token immediately before the second assistant answer. Earlier option-line activations are outside this descriptive question.
- Layers: every post-block residual, L1–L64. L1–L63 are transported to final-output space with the fixed Qwen3.6-27B Jacobian lens; L64 uses exact live natural logits.
- Candidate score: log-sum-exp over the bare and leading-space token variants for each displayed A-D answer, centered over A-D within question.
- Rank: R1–R4 are frozen from the same-format first-presentation aggregated A-D logits with displayed-order tie handling.
- Outcome split: for each task separately, `switch` means the final aggregated-A-D argmax differs from 1P R1; `stay` means it equals R1. The aggregate panels are primary. Outcome-conditioned panels are descriptive postselection, not causal evidence.
- Statistics: raw question mean with first-presentation-winner-letter-stratified question bootstrap 95% confidence bands.

## Deliverables

Exactly six top-level PNGs: dataset × {all, switch, stay}. Each PNG has separate Game and Neutral panels and four R1–R4 curves in each panel. Compact arrays, prompt audit, metadata, a machine-readable summary, a canonical report, and a README index entry accompany the figures.

## Complete model work

Two conditions × 500 questions × two datasets = 2,000 prompt sequences. With physical batch size four, collection makes 500 complete model forward calls after one model load. No baseline forward is repeated: same-format baseline logits already on disk define the ranks. JLens transport is post-processing, not another model forward.

## Prospective-answer decoder follow-up

The JLens trajectories measure when the candidate ordering is already aligned with the model's fixed output readout. They do not test whether the same eventual four-answer ordering is linearly present earlier in another residual-stream basis. The prospective-answer follow-up therefore repeats only the validated natural Game and Neutral collection while retaining the final-decision-position residual at every block.

At each layer, ridge decoders predict the exact final centered A–D logit vector. All fitting and hyperparameter selection use the frozen discovery questions; confirmation questions are untouched until evaluation, and Game/Neutral rows from a question always remain in the same fold. Three coefficient bases are fit: one shared Game+Neutral decoder, one Game-only decoder, and one Neutral-only decoder. Every coefficient basis is evaluated on both conditions. Game-trained→Neutral and Neutral-trained→Game transfer is the primary test of whether the prospective ranking occupies the same linear subspace; matched per-condition versus shared performance alone is secondary because the shared fit has twice as many rows. Inputs and targets are centered separately within each condition using discovery-only means, so a condition-level mean shift cannot masquerade as a basis difference.

The new decoder is descriptive activation/decoding evidence, not a causal intervention. Earlier held-out decoding than JLens would show a linearly available but not yet output-aligned answer ordering. A similarly late onset would strengthen the claim that the prospective ordering is constructed linearly only in late blocks. Failure to decode does not rule out a nonlinear representation.

## Prospective-answer result

The final four-answer pattern is linearly decodable before it is aligned with the fixed output readout. The separation is clearest on TriviaMC, where the held-out shared-decoder cosine is 0.369 at L32 and 0.676 at L40 while fixed-JLens cosine is approximately zero. Bidirectional Game↔Neutral decoder transfer remains close to matched-condition decoding, especially in later layers, so the prospective code is predominantly condition-general rather than encoded in wholly different task-specific bases. The specific R2-over-R1 ordering on held-out eventual-switch trials is later, becoming reliably positive around L44–L48. On the paired Game-switch/Neutral-stay subset, the across-condition mean remains R1-favoring while the Game-minus-Neutral R2−R1 difference becomes linearly decodable at L34–L35. This is a descriptive, outcome-selected timing result; it does not establish causal use.

Chance is evaluated both by a fully shuffled paired-question target assignment, whose mean is approximately zero, and by a stronger W1-matched shuffle that preserves the displayed first-presentation winner letter while breaking question-specific final geometry. The first figure panel shows the stronger null. At L40, the shared decoder versus W1-matched-null cosine is 0.403 versus 0.170 on SimpleMC and 0.676 versus 0.462 on TriviaMC.

## All-question policy-adjusted follow-up

The policy follow-up uses every held-out confirmation question, pairs its Game and Neutral runs, and subtracts the Neutral four-answer vector from Game after discovery-only displayed-letter control. The shared decoder is used for both conditions, so the difference is expressed in one learned basis. A within-question condition-sign null randomly reverses Game and Neutral target orientation. The analysis reports cosine to the exact final question-specific Game-minus-Neutral vector and decomposes the decoded difference by first-presentation rank.

The policy pattern becomes continuously decodable at L33 on SimpleMC and L32 on TriviaMC. At L40, learned cosine is 0.311 and 0.520 while fixed JLens is 0.006 and -0.035. Exact held-out Game-minus-Neutral rank effects are SimpleMC R1 -0.532, R2 +0.001, R3 +0.181, R4 +0.350 and TriviaMC R1 -1.600, R2 +0.392, R3 +0.604, R4 +0.604. The decoded candidate components acquire the final signs around L32--L36, except TriviaMC R2 at L43. This full-population result is not selected on eventual switching, but remains activation/decoding rather than causal evidence. See the [policy report](prospective_decoding/policy_analysis/REPORT.md).
