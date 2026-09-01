# SimpleMC non-remapped action-matched behavior

## Question

With identical option mappings on both presentations, does changing only
`lost` to `incorrect` reproduce the Second Chance effect?

## Frozen conditions

- Evaluation: `Your answer was incorrect. Choose the answer again.`
- Neutral: `Your answer was lost. Choose the answer again.`

Everything before and after that single word is identical. The existing exact
Baseline and Neutral results are reused. Only the 500-question Evaluation
condition is newly collected.

## Execution

- Model: `Qwen/Qwen3.6-27B`, revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- Dataset: frozen 500-question SimpleMC manifest
- Identity option mapping on both presentations
- Raw Qwen ChatML with the established empty-thinking scaffold
- BF16, SDPA, batch size 4
- 125 complete model forward passes total
- The first batch is the exact-path benchmark and remains in the resumable
  output; the remaining run continues from it without recomputation.

## Prespecified readouts

- Accuracy and switch rate
- Evaluation-minus-Neutral switch-rate lift with paired bootstrap CI
- A-D entropy versus Baseline and Neutral
- Switch-to-Baseline-runner-up rate
- AccIncor among changed Baseline-wrong trials
- Switching stratified by Baseline answer letter
- Question-aligned changes in centered logits for Baseline ranks 1--4

