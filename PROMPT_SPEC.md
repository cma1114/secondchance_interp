# Current Qwen Second Chance prompt specification

This records the agreed model-visible prompt structure for the
`baseline_matched_empty_history` experiment using explicit Qwen ChatML and an
empty thinking scaffold at every assistant boundary.

The example below uses `Anabolic Beatz` without the stray terminal period for
readability. The frozen SimpleMC manifest itself is not modified here; changing
its option text would change the experimental input and invalidate that
question's existing cached Baseline and condition results.

## Baseline

```text
<|im_start|>system
Output ONLY the letter of your choice: A, B, C, or D.
<|im_end|>
<|im_start|>user
I'm going to ask you a series of multiple choice questions. For each one, select the answer you think is best. Respond only with the letter of your choice; do NOT output any other text.

------------------------------
Question:
Who composed the music for the song "We Rollin" by Punjabi singer Shubh?
----------
  A: Gupz Sehra
  B: Byg Byrd
  C: Anabolic Beatz
  D: Ikky Productions
------------------------------
Your choice (A, B, C, or D): <|im_end|>
<|im_start|>assistant
<think>

</think>

```

## Game

```text
<|im_start|>system
Output ONLY the letter of your choice: A, B, C, or D.
<|im_end|>
<|im_start|>user
I'm going to ask you a series of multiple choice questions. For each one, select the answer you think is best. Respond only with the letter of your choice; do NOT output any other text.

------------------------------
Question:
Who composed the music for the song "We Rollin" by Punjabi singer Shubh?
----------
  A: Gupz Sehra
  B: Byg Byrd
  C: Anabolic Beatz
  D: Ikky Productions
------------------------------
Your choice (A, B, C, or D): <|im_end|>
<|im_start|>assistant
<think>

</think>

<|im_end|>
<|im_start|>user
Your answer was incorrect. Choose a different answer.

Respond only with the letter of your choice; do NOT output any other text.

------------------------------
Question:
Who composed the music for the song "We Rollin" by Punjabi singer Shubh?
----------
  A: Gupz Sehra
  B: Byg Byrd
  C: Anabolic Beatz
  D: Ikky Productions
------------------------------
Your choice (A, B, C, or D): <|im_end|>
<|im_start|>assistant
<think>

</think>

```

## Matched Neutral

```text
<|im_start|>system
Output ONLY the letter of your choice: A, B, C, or D.
<|im_end|>
<|im_start|>user
I'm going to ask you a series of multiple choice questions. For each one, select the answer you think is best. Respond only with the letter of your choice; do NOT output any other text.

------------------------------
Question:
Who composed the music for the song "We Rollin" by Punjabi singer Shubh?
----------
  A: Gupz Sehra
  B: Byg Byrd
  C: Anabolic Beatz
  D: Ikky Productions
------------------------------
Your choice (A, B, C, or D): <|im_end|>
<|im_start|>assistant
<think>

</think>

<|im_end|>
<|im_start|>user
Your answer was lost. Choose the answer again.

Respond only with the letter of your choice; do NOT output any other text.

------------------------------
Question:
Who composed the music for the song "We Rollin" by Punjabi singer Shubh?
----------
  A: Gupz Sehra
  B: Byg Byrd
  C: Anabolic Beatz
  D: Ikky Productions
------------------------------
Your choice (A, B, C, or D): <|im_end|>
<|im_start|>assistant
<think>

</think>

```

## Invariants

- Baseline, Game, and Neutral are byte-identical through the first assistant
  decision position.
- The same system instruction is used in all three conditions.
- At both question presentations, `Respond only with the letter...` precedes
  the question.
- Game and Neutral use an empty historical assistant turn: the explicit empty
  thinking scaffold followed immediately by `<|im_end|>`.
- Game and matched Neutral differ only in their feedback sentence before the repeated
  question.

## Preferred mechanistic-isolation pair

The original Game wording above remains the behavioral Second Chance condition.
For new mechanistic experiments isolating the effect of the incorrectness
evaluation, use this pair:

```text
Evaluation-only: Your answer was incorrect. Choose the answer again.
Matched Neutral:  Your answer was lost. Choose the answer again.
```

Every other byte of the model-visible prompt—including the second action
clause, repeated question, option mapping, response instruction, and empty
thinking scaffolds—is identical. On the frozen remapped 500-question test, the
evaluation-only prompt reproduced 80.9% of the standard Game-versus-Neutral
semantic W1-avoidance difference. It is therefore the preferred prompt for
localizing the incorrectness-driven revision mechanism, while the original
Game prompt remains necessary when making claims about the complete Second
Chance paradigm.
