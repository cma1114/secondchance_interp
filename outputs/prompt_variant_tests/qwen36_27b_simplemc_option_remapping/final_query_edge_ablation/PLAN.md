# Canonical remapped final-query W1-line edge ablation

## Question

Does Qwen's preferential semantic switching require the final pre-answer query to read the complete first-presentation option line containing its original answer, W1, through ordinary attention?

## Frozen design

- Use the existing 500-question canonical remapping plan and trusted Game, Neutral, original-Baseline, and remapped-Baseline outputs.
- Preserve the exact batch-of-four SDPA execution and explicit raw ChatML prompts.
- Define conflict as W1 differing from the fresh remapped-Baseline winner W2; analyze the 268 conflict and 232 no-conflict questions separately.
- At only the final pre-answer query, block every attention head from reading the complete first-presentation W1 option line.
- Test the previously fixed block sets: block 44; blocks 36, 40, 44, and 48; and all ordinary-attention blocks 4 through 48.
- For every selected-line intervention, run a nearest-token-count unselected-option-line control.

## Primary outcomes

- Game and Neutral W1 selection and switching away from W1.
- On conflict trials, W1-minus-W2 logit margin.
- W1 centered A-D advantage and A-D entropy.
- Selected-line effect minus matched-control effect.

Natural logits must reproduce the trusted canonical remapped outputs exactly before results are accepted.
