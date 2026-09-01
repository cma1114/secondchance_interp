# Remapped versus non-remapped 2P→1P attention

## Design

The comparison uses the same 500 questions, prompts differing only in whether the second-presentation option order is permuted or left identical, both Game (`incorrect`) and Neutral (`lost`), and every ordinary-attention layer L4--64. Each query is a complete 2P option line. Sources exhaustively partition every prompt token. Candidates are aligned by semantic identity and first-pass rank, not displayed letter.

## Held-out confirmation summary

### Game

- Attention to all four 1P option lines across all 16 layers: remapped 31.4% [31.0%, 31.8%]; non-remapped 31.0% [30.5%, 31.5%]; paired difference -0.4% [-0.5%, -0.3%].
- Attention to matching 1P option line across all 16 layers: remapped 12.3% [11.9%, 12.8%]; non-remapped 15.6% [15.1%, 16.0%]; paired difference 3.3% [3.1%, 3.4%].
- Match minus mean nonmatch across all 16 layers: remapped 5.9% [5.5%, 6.4%]; non-remapped 10.4% [10.0%, 10.8%]; paired difference 4.5% [4.3%, 4.6%].

### Neutral

- Attention to all four 1P option lines across all 16 layers: remapped 31.9% [31.5%, 32.3%]; non-remapped 31.5% [31.0%, 31.9%]; paired difference -0.4% [-0.5%, -0.3%].
- Attention to matching 1P option line across all 16 layers: remapped 12.4% [12.0%, 12.9%]; non-remapped 15.7% [15.2%, 16.1%]; paired difference 3.3% [3.1%, 3.4%].
- Match minus mean nonmatch across all 16 layers: remapped 5.9% [5.4%, 6.4%]; non-remapped 10.4% [10.0%, 10.8%]; paired difference 4.5% [4.3%, 4.6%].

## Semantic identity versus displayed-letter position

In the non-remapped prompt, a candidate's semantic match is also the line with the same displayed letter and list position. The remapped prompt separates those two possible targets. Within the remapped confirmation run:

- Game: semantic match 12.3% [11.8%, 12.8%]; same displayed letter 8.5% [8.3%, 8.6%]; semantic advantage 3.8% [3.3%, 4.4%].
- Neutral: semantic match 12.4% [12.0%, 12.9%]; same displayed letter 8.6% [8.5%, 8.8%]; semantic advantage 3.8% [3.2%, 4.3%].

## Interpretation rule

Total 1P-option attention answers whether 2P reads the old candidate set as much. Matching-line attention answers whether it reads the same semantic candidate as much. Matching selectivity subtracts the average wrong 1P line, separating semantic matching from a generic increase in attention to the entire first option list. In the non-remapped prompt, semantic identity and displayed letter/position coincide; therefore matching selectivity must be interpreted alongside the remapped condition, where those factors are separated.

## Mechanistic interpretation

The complete question and all four answer texts are already present again in 2P. Therefore, the 2P-to-1P read is not needed merely to recover problem text missing from the second presentation. The information distinctive to a 1P option-line state is the model's earlier, context-dependent processing of that candidate: its first-pass evidence and its relation to the other candidates. The attention comparison is observational by itself, so it cannot prove which feature is read. Combined with the separate balanced matching-line lesions—which causally change final candidate scores according to first-pass rank and remove the discrete Game-minus-Neutral switching difference—the best-supported interpretation is that both remapped and non-remapped prompts reuse prior candidate evaluation while 2P also constructs fresh candidate evidence. Identity order makes that retrieval sharper because semantic identity, displayed letter, and list position all point to the same old line; remapping separates those cues but leaves a clear semantic preference.

## Validation

- Questions: 500; discovery: 251; confirmation: 249.
- Remapped/non-remapped maximum partition errors: 0.004067 / 0.004403.
- Remapped/non-remapped natural choice agreement with trusted outputs: 100.0% / 97.7%.

Canonical figure: `figures/qwen36_remapped_nonremapped_2p_1p_attention.png`.
