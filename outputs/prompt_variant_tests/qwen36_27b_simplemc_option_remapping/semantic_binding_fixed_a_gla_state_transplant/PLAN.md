# Fixed-A accumulated GLA-state transplant

## Question

Does the accumulated GLA memory at the first-answer boundary carry the semantic
identity of the answer that Game later suppresses?

## Frozen design

- Reuse the 137-question fixed-A cohort: 64 discovery and 73 held-out
  confirmation questions.
- Within each question, both first presentations produce literal answer `A`,
  but `A` denotes different semantic answers X and Y.
- Feedback and the second presentation are identical across X/Y histories.
- At all 48 GLA layers jointly, capture the accumulated recurrent state
  immediately after the first-answer boundary and exchange it X↔Y within Game
  or within Neutral.
- Keep the recipient's visible prompt and every post-boundary token fixed.
- Compare the cross-semantic transplant with recipient-state reinsertion under
  the same segmented recurrent-kernel computation. The unsplit natural pass is
  only a validation reference.

## Prespecified prediction

- In Game, a Y-state transplanted into the X history should make Y rather than X
  the semantic answer that is suppressed: the X-minus-Y margin should rise.
- In Neutral, the same transplant should make Y rather than X the answer that is
  repeated: the X-minus-Y margin should fall.
- The symmetric Y-target/X-donor effect is signed and averaged with this one.

No layer or endpoint is selected from discovery. Results are reported
separately for discovery and held-out confirmation.
