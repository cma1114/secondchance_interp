# Candidate-history convolution-safe joint relay control

## Bottom line

The kernel-minus-one control leaves the final three assistant-prefix tokens free, matching the preceding-token support of a single four-token causal GLA convolution read. The conservative control leaves four tokens free, adding one boundary token so it can recompute rather than inject a pinned lesioned output.

On confirmation conflict trials, Game recovery is 36.8% [33.9%, 39.5%] with all five regions pinned, 73.3% [71.0%, 75.7%] with the exact window free, 97.7% [96.3%, 99.1%] with the conservative window free, and 94.1% [91.2%, 96.5%] with the entire prefix free.

On confirmation conflict trials, Neutral recovery is 48.4% [45.4%, 51.3%] with all five regions pinned, 76.6% [74.5%, 78.4%] with the exact window free, 96.5% [95.7%, 97.4%] with the conservative window free, and 94.1% [92.6%, 95.6%] with the entire prefix free.

## Validation

- 500/500 questions; 273 canonical W1!=W2 conflicts.
- Natural maximum A-D error: 0.00000000.
- Real convolution-safe restoration-only error: 0.00000000.
- Maximum error across five shared Stage-B controls: 0.00000000.

## Conclusion

The prespecified gate passed in both tasks and both frozen splits. Freeing the final four prefix tokens raises recovery to 97.7% in Game and 96.5% in Neutral on confirmation, with 97.9% and 96.3% on discovery. The prior 36.8%/48.4% nominal all-five result was therefore caused by the restoration convention pinning lesioned local prefix outputs beside the unintercepted multilayer GLA convolution. It is not evidence for an antagonistic assistant-prefix relay.

Freeing only three tokens gives partial recovery (73.3% Game, 76.6% Neutral on confirmation). That is a boundary result: in the multilayer computation, the pinned fourth token can contaminate the three freely recomputed tokens before the final readout. One additional free boundary token removes that contamination.

Scientifically, the five-region causal-tail inventory now accounts for essentially the whole measured candidate-history pathway. The result does not make the ordinary-attention/GLA carrier percentages additive; convolution is part of the implemented computation and was not separately intercepted.

Machine-readable estimates and paired bootstrap intervals are in `summary.json`.
