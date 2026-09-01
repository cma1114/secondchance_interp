# OLMo 2 32B SimpleMC behavioral gate

## Scientific question

Before spending on mechanistic replication, test whether OLMo 2 32B Instruct reproduces the behavioral phenomenon in the current clean paradigm: preferential switching after `incorrect` rather than `lost`, where the two prompts otherwise say exactly `Choose the answer again.`

## Frozen scope

- Model: `allenai/OLMo-2-0325-32B-Instruct`, revision `b96024342a77a69aa0dda815c3454a671f477463`.
- Dataset: all 500 questions in the canonical SimpleMC stimulus manifest, in its existing order.
- Prompt serialization: OLMo's native Hugging Face chat template.
- First-answer history: the canonical empty first assistant turn. This removes any visible old answer letter while leaving the complete first question and option lines in context.
- Conditions: first-presentation Baseline; non-remapped Game and Neutral; remapped Game and Neutral.
- Game/Neutral contrast: exactly `incorrect` versus `lost` in `Your answer was ... Choose the answer again.`
- Remapping: the existing frozen balanced four-option derangement; every semantic option moves to a new letter.
- Layers and activations: none. This is a behavioral gate, not a mechanistic run.
- Questions: no accuracy filtering and no outcome-based selection.

The exact runner executes five complete forwards per four-question cohort: Baseline, non-remapped Game, non-remapped Neutral, remapped Game, and remapped Neutral. The representative benchmark adds one duplicate-Baseline forward as a deterministic identity control. The full 500-question run therefore comprises 625 physical batched forwards and 2,500 condition-question outputs.

## Endpoints

1. Unrestricted top-token A-D switching, with aggregated A-D logits as a prespecified continuous and coverage-robust readout.
2. Game-minus-Neutral semantic switching in the non-remapped and remapped presentations.
3. In the remapped presentation, semantic old-winner suppression versus suppression of the old literal letter.
4. Baseline-rank-resolved centered A-D logit redistribution (W1-W4).
5. A-D entropy in Baseline, Game, and Neutral.
6. Full-sample and frozen 251/249 discovery/confirmation summaries with paired bootstrap confidence intervals.

## Decision rule

Mechanistic follow-up is worthwhile if OLMo shows a positive Game-minus-Neutral semantic switch gap that survives option remapping, together with content-targeted W1 suppression rather than a comparable effect on the old literal letter. A weak or letter-bound remapped effect argues against paying for the mechanistic suite on this model.

## Controls and stopping rules

- The rendered Game and Neutral prompts must differ only at `incorrect`/`lost` in both mappings.
- All A-D and space-prefixed A-D variants must be single tokens.
- Every remapping must be a derangement.
- Benchmark duplicate-Baseline A-D logits must reproduce exactly; all logits must be finite; unrestricted answer-only compliance is inspected before the full launch.
- The full exact path is benchmarked before launch and remains within the standing $15 batch cap.
