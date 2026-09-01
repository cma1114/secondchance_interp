# Mixer 56 historical-answer read diagnostic

The prior causal test changed the JLens-decoded A-D representation at the final
token of the empty historical assistant scaffold, at `resid_pre[56]`, but did
not specifically change behavior. This diagnostic distinguishes three reasons:

1. Mixer 56 attends to the historical turn but not to the answer-decodable
   endpoint within that turn.
2. Its value or key projections suppress the JLens A-D directions even when a
   head attends to the endpoint.
3. The directions survive and are read, but are transformed into outputs that
   do not write A-D answer evidence.

On the frozen 249-question confirmation set, record every Mixer 56 head's exact
attention to the complete historical assistant span and to each token within
it, including the manipulated `\n\n` endpoint. Record the final-query output
gate and the historical span's immediate A-D-aligned route write.

Separately pass the exact four JLens answer-contrast directions through every
head's key and value projections. Compare their retained norms with 512
isotropic unit directions. Compose value and output projections to measure the
immediate A-D write available from each head. Finally, combine those fixed
matrices with the exact held-out winner-erasure perturbations and observed
attention/gates to estimate the answer-relevant value-path write in Game and
Neutral.
