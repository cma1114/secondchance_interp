# Candidate-history downstream relay mediation

## Bottom line

The causal path is distributed but not anonymous. On held-out conflict trials, restoring only the 2P semantic wordpieces' outgoing state recovers 59.0% [54.1%, 64.4%] of the source-lesion R1–R4 rank vector in Game and 61.9% [55.4%, 68.8%] in Neutral. It is the strongest single relay in both tasks.

Newlines are a real secondary relay, while structural option tokens and the post-list cue/query also carry substantial recoverable history. The final assistant prefix is the weakest single relay. The discovery split independently preserves this ordering and the broad magnitudes.

Restoring all regions except the final assistant prefix recovers 94.1% [91.3%, 96.5%] in Game and 94.1% [92.6%, 95.6%] in Neutral. The nominal all-five downstream-only restoration recovers only 36.8% [34.0%, 39.5%] and 48.4% [45.4%, 51.3%], respectively. The completed convolution-safe control confirms that this contrast is an intervention artifact rather than prefix physiology: leaving the final four assistant-prefix tokens free raises recovery to 97.7% [96.3%, 99.1%] and 96.5% [95.7%, 97.4%] on confirmation, with 97.9% and 96.3% on discovery. See the [control report](../../convolution_control/analysis/REPORT.md).

## Validation and scope

- 500/500 questions completed; 251 discovery and 249 confirmation.
- Canonical W1!=W2 conflict trials: 273 total, 137 discovery, 136 confirmation.
- Canonical remapped-baseline SHA-256: `e9d522e08430641ddb5651719497503cb0229a4dc54ae473ed2b3a7e67151228`.
- Trusted-natural maximum A–D error: 0.00000000.
- Real no-source restoration-only maximum raw error across 28 frozen sentinels and ordinary-only, GLA-only, and both modes: 0.00000000.
- Every main and identity output is finite.
- The restoration covers outgoing ordinary-attention K/V and recurrent GLA writes. It does not intercept the short causal GLA q/k/v convolution, and it deliberately preserves each relay token's source-perturbed local output. Joint cells that restore tokens immediately before the readout are therefore convolution-confounded.

## The source effect being traced

Natural minus matching-edge-lesioned candidate-centered logits on confirmation conflict trials:

| Rank | Game | Neutral |
|---|---:|---:|
| R1 | -0.127 [-0.335, +0.088] | +0.377 [+0.136, +0.627] |
| R2 | -0.258 [-0.455, -0.077] | -0.279 [-0.489, -0.076] |
| R3 | +0.030 [-0.154, +0.213] | -0.156 [-0.359, +0.040] |
| R4 | +0.355 [+0.223, +0.485] | +0.058 [-0.100, +0.210] |

The lesion removes a graded, candidate-specific history vector rather than one scalar. That is why the normalized R1–R4 recovery projection is the most stable summary; all prespecified scalar endpoints remain available below and in `summary.json`.

## Matching-source specificity

The balanced cyclic wrong-line lesion is much smaller than the matching lesion on the held-out conflict set, so the traced path is not a generic consequence of deleting the same number of attention edges.

| Task | Matching source-vector norm | Wrong source-vector norm | Matching-specific joint recovery |
|---|---:|---:|---:|
| Game | +1.957 [+1.798, +2.123] | +0.527 [+0.472, +0.588] | 36.8% [34.4%, 39.2%] |
| Neutral | +2.180 [+1.986, +2.374] | +0.627 [+0.562, +0.695] | 48.9% [45.6%, 51.8%] |

The same specificity pattern replicates in discovery. Joint rescue of the matching-minus-wrong vector is therefore attributable to semantic history, not merely to the intervention's size.

## Relay inventory: confirmation conflict trials

| Restoration | Game rank-vector recovery | Neutral rank-vector recovery |
|---|---:|---:|
| 2P semantic wordpieces | 59.0% [54.1%, 64.4%] | 61.9% [55.4%, 68.8%] |
| 2P option newlines | 37.6% [32.3%, 42.8%] | 39.5% [34.4%, 44.5%] |
| 2P option structure | 31.7% [24.4%, 39.3%] | 40.1% [32.8%, 47.7%] |
| Post-list cue/query | 25.0% [21.3%, 28.4%] | 31.9% [27.3%, 36.0%] |
| Final assistant prefix | 16.5% [14.3%, 18.7%] | 19.0% [16.5%, 21.4%] |
| Newlines + cue/query | 56.8% [51.9%, 61.6%] | 63.2% [58.7%, 67.5%] |
| Newlines + prefix | 35.1% [31.9%, 38.5%] | 40.6% [36.8%, 44.4%] |
| Cue/query + prefix | 22.6% [19.9%, 25.0%] | 30.7% [27.5%, 33.6%] |
| All except semantic | 48.0% [42.3%, 53.7%] | 60.4% [55.5%, 65.0%] |
| All except newline | 47.2% [43.2%, 51.2%] | 60.2% [56.5%, 63.5%] |
| All except structure | 40.7% [37.1%, 44.3%] | 53.3% [50.3%, 56.3%] |
| All except cue/query | 52.5% [49.5%, 55.7%] | 62.8% [58.9%, 66.6%] |
| All except prefix | 94.1% [91.3%, 96.5%] | 94.1% [92.6%, 95.6%] |
| All five | 36.8% [34.0%, 39.5%] | 48.4% [45.4%, 51.3%] |

## Carrier mechanism

| Joint restoration mode | Game | Neutral |
|---|---:|---:|
| Ordinary attention only | 21.5% [18.4%, 24.5%] | 30.1% [27.3%, 32.7%] |
| GLA recurrent writes only | 13.7% [12.0%, 15.2%] | 17.3% [15.5%, 19.3%] |
| Both | 36.8% [34.0%, 39.5%] | 48.4% [45.4%, 51.3%] |

Ordinary-attention K/V and recurrent GLA writes each recover a nonzero part of the history effect. The nominal joint carrier cells are not an exhaustive mechanism decomposition: they preserve lesioned local prefix outputs and omit the short GLA convolution, so their remaining deficit cannot be interpreted as a physiological bypass fraction.

## Standard endpoints on confirmation conflict trials

Mediated amount is restored minus lesioned. Fractions are printed only when the paired bootstrap denominator excludes zero.

| Task | Scenario | Endpoint | Mediated amount | Mediated fraction |
|---|---|---|---:|---:|
| Game | Semantic | W1-W2 | +0.090 [-0.147, +0.333] | unstable denominator |
| Game | Semantic | W1_choice | -5.882 [-13.971, +2.206] | 53.3% [-33.3%, 119.2%] |
| Game | Newlines+cue | W1-W2 | +0.056 [-0.168, +0.286] | unstable denominator |
| Game | Newlines+cue | W1_choice | -0.735 [-8.088, +6.618] | 6.7% [-140.0%, 57.9%] |
| Game | All except prefix | W1-W2 | +0.117 [-0.205, +0.441] | unstable denominator |
| Game | All except prefix | W1_choice | -9.559 [-18.382, -0.735] | 86.7% [33.3%, 133.3%] |
| Game | Joint | W1-W2 | +0.063 [-0.080, +0.205] | unstable denominator |
| Game | Joint | W1_choice | -1.471 [-5.147, +1.471] | 13.3% [-33.3%, 50.0%] |
| Neutral | Semantic | W1-W2 | +0.147 [-0.124, +0.423] | 22.4% [-35.5%, 46.6%] |
| Neutral | Semantic | W1_choice | +2.941 [-5.147, +11.029] | 22.2% [-83.3%, 66.7%] |
| Neutral | Newlines+cue | W1-W2 | +0.345 [+0.074, +0.622] | 52.6% [26.5%, 66.5%] |
| Neutral | Newlines+cue | W1_choice | +8.824 [+1.471, +16.176] | 66.7% [20.0%, 140.0%] |
| Neutral | All except prefix | W1-W2 | +0.572 [+0.208, +0.922] | 87.2% [71.3%, 94.2%] |
| Neutral | All except prefix | W1_choice | +14.706 [+5.882, +23.529] | 111.1% [100.0%, 150.0%] |
| Neutral | Joint | W1-W2 | +0.432 [+0.230, +0.636] | 65.8% [54.9%, 94.4%] |
| Neutral | Joint | W1_choice | +11.029 [+5.147, +17.647] | 83.3% [43.5%, 200.0%] |

## Named-pair screen and artifact gate

No named pair shows a same-sign nonzero interaction on the same standard endpoint in both frozen conflict splits. The prespecified triple escalation is therefore not earned.

## Interpretation

The clean serial story is now: matching 1P history is first read mainly into the 2P semantic wordpieces; those same semantic positions are also the strongest downstream relay, but they are not a unique bottleneck. Newlines, structural option tokens, and the post-list cue/query redundantly re-express parts of the history vector through both ordinary attention and GLA. Restoring all four pre-prefix groups while allowing the assistant scaffold to recompute recovers about 94% of the history vector. A separately executed convolution-safe joint restoration recovers 96.5--97.7% on confirmation and 96.3--97.9% on discovery, establishing that the five-region causal-tail inventory carries essentially the whole measured candidate-history effect. The lower nominal all-five value is an artifact of pinning perturbed local prefix outputs beside the unintercepted GLA convolution.

This stage localizes transport of old candidate history. It does not yet decide whether the transported state already contains the Game-versus-Neutral policy product. That is the Stage-C donor-policy crossover question, now to be targeted at the replicated semantic relay and the broader pre-prefix relay set rather than at an undifferentiated suffix.

Machine-readable results include every task, split, conflict stratum, scalar endpoint, single, complement, named pair, carrier mode, task contrast, and pair interaction in `summary.json`; `scenario_metrics.csv` provides the flat scalar table.
