# Seed-OSS 36B clean behavioral replication

## Scientific question

Test whether `ByteDance-Seed/Seed-OSS-36B-Instruct` reproduces the clean Second Chance behavior before any mechanistic work: preferential semantic switching after `incorrect` rather than the token-matched control `lost`, with content-targeted suppression of the model's first-presentation winner that survives a complete option-letter remapping.

## Stage 1: SimpleMC gate

- Model: revision `497f1dca95ebdec98e41d517b9f060ee753c902f`, BF16, native Hugging Face template, `thinking_budget=0`.
- Dataset: all 500 canonical SimpleMC questions, without accuracy or outcome filtering.
- First-answer history: an empty first assistant turn, so no old answer letter is visible.
- Conditions: Baseline, same-order Game and Neutral, fully remapped Game and Neutral.
- Game/Neutral difference: exactly `incorrect` versus `lost`; both say `Choose the answer again.`
- Remapping: the existing frozen balanced derangement in which all four semantic options move to new letters.
- Layers and activations: none; this is a behavioral gate.

The exact production path runs five complete forwards per four-question cohort. The benchmark adds a duplicate Baseline forward. Thus the full SimpleMC stage is 625 physical batched forwards and 2,500 question-condition outputs.

Seed advances only if the frozen confirmation split shows (1) a positive remapped Game-minus-Neutral semantic switching gap, (2) positive Game suppression of the semantic old winner, and (3) stronger targeting of that semantic content than of its former literal letter. This is the same prespecified gate used for OLMo.

## Stage 2: conditional TriviaMC replication

If Stage 1 passes, run the same five-condition design on all 500 questions in the frozen difficulty-filtered TriviaMC manifest, using its existing balanced complete remapping and frozen 250/250 split. This stage is also 625 physical batched forwards and 2,500 question-condition outputs. It tests dataset generalization; it does not change the SimpleMC gate after observing the data.

## Validity and stopping rules

- Pin the model revision and Transformers 4.57.6; do not use the earlier historical Seed prompt protocol.
- Render with Seed's native template and verify that `thinking_budget=0` is present.
- Verify that Game and Neutral rendered prompts differ only at `incorrect`/`lost` in both mappings.
- Require every remapping to be a derangement, all A-D variants to be single tokens, all logits finite, exact duplicate-Baseline A-D logits, and inspect unrestricted answer-only compliance before each full launch.
- Benchmark the complete exact path before launch. Track the combined SimpleMC and conditional TriviaMC batch against the standing $15 cap.
