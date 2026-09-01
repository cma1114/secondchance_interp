from __future__ import annotations

import inspect
from typing import Any


def _replace_only_causally_later_gla_outputs(
    natural_result: Any,
    intervened_result: Any,
    cutoffs_by_row: dict[int, int],
) -> Any:
    """Keep each source token's own GLA output natural and replace only its future.

    Qwen's chunked GLA rule returns ``(sequence_output, final_state)``.  A source
    memory edit should affect queries strictly after the source token without
    altering the residual being computed at the source itself.  The edited
    recurrent final state is retained for selected rows so cached execution,
    if requested by a caller, remains consistent with the edited history.
    """
    if not isinstance(natural_result, (tuple, list)) or not isinstance(
        intervened_result, (tuple, list)
    ):
        raise RuntimeError("Expected the Qwen GLA rule to return output and state")
    if len(natural_result) != 2 or len(intervened_result) != 2:
        raise RuntimeError("Unexpected Qwen GLA result structure")
    natural_output, natural_state = natural_result
    edited_output, edited_state = intervened_result
    output = natural_output.clone()
    for row, cutoff in cutoffs_by_row.items():
        if cutoff + 1 < output.shape[1]:
            output[row, cutoff + 1 :] = edited_output[row, cutoff + 1 :]
    state = natural_state
    if natural_state is not None or edited_state is not None:
        if natural_state is None or edited_state is None:
            raise RuntimeError("Natural and edited GLA calls returned inconsistent state")
        state = natural_state.clone()
        for row in cutoffs_by_row:
            state[row] = edited_state[row]
    result = (output, state)
    return result if isinstance(natural_result, tuple) else list(result)


class BatchedSDPAFinalQueryAttentionAblator:
    """Block exact final-query-to-source edges in selected SDPA layers.

    ``specs_by_layer`` maps zero-based model-layer indices to per-batch-row
    source-token positions.  Only the final physical query is modified; every
    earlier query and every unlisted ordinary-attention layer is untouched.
    This preserves Qwen's established batched SDPA numerical regime while
    testing the clean read edge from the final answer decision to a historical
    option line.
    """

    def __init__(
        self,
        parts: Any,
        specs_by_layer: dict[int, dict[int, list[int]]],
    ) -> None:
        import torch

        if not specs_by_layer:
            raise ValueError("No final-query attention-edge ablations were specified")
        self.specs = {
            int(layer): {
                int(row): tuple(sorted(set(int(position) for position in positions)))
                for row, positions in rows.items()
            }
            for layer, rows in specs_by_layer.items()
        }
        if any(not rows for rows in self.specs.values()):
            raise ValueError("Every selected layer needs at least one batch row")
        if any(
            not positions
            for rows in self.specs.values()
            for positions in rows.values()
        ):
            raise ValueError("Every selected row needs at least one source position")

        self.active_layer: int | None = None
        self.handles: list[Any] = []
        for layer_index in sorted(self.specs):
            attention = getattr(parts.layers[layer_index], "self_attn", None)
            if attention is None:
                self.close()
                raise ValueError(
                    f"Layer {layer_index} is not a conventional-attention block"
                )
            self.handles.append(
                attention.register_forward_pre_hook(self._enter(layer_index))
            )
            self.handles.append(
                attention.register_forward_hook(self._leave(layer_index))
            )

        self.original_sdpa = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = self._wrapped_sdpa

    def _enter(self, layer_index: int):
        def enter(_module: Any, _inputs: Any) -> None:
            if self.active_layer is not None:
                raise RuntimeError("Nested ordinary-attention calls are unsupported")
            self.active_layer = layer_index
        return enter

    def _leave(self, layer_index: int):
        def leave(_module: Any, _inputs: Any, _output: Any) -> None:
            if self.active_layer != layer_index:
                raise RuntimeError("Ordinary-attention layer stack became inconsistent")
            self.active_layer = None
        return leave

    def _wrapped_sdpa(self, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
        import torch

        layer = self.active_layer
        if layer is None:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        rows = self.specs[layer]
        if args:
            raise RuntimeError(
                "The validated Qwen SDPA path passes attention options by keyword"
            )
        mask = kwargs.get("attn_mask")
        batch, _heads, query_length, _dim = query.shape
        key_length = key.shape[-2]
        if max(rows) >= batch:
            raise RuntimeError("Attention-ablation row exceeds the SDPA batch")

        if mask is None:
            # Keep Qwen's complete causal SDPA call byte-for-byte intact.  The
            # final query is already causally allowed to read every key.  Compute
            # the masked-minus-unmasked change for just that attention row with
            # one shared float32 implementation, then add only that delta to the
            # untouched natural result.  This avoids the real numerical
            # difference between the full-query and one-query SDPA kernels.
            natural = self.original_sdpa(query, key, value, **kwargs)
            final_query_tensor = query[:, :, -1:, :]
            edge_mask = torch.ones(
                (batch, 1, 1, key_length),
                dtype=torch.bool,
                device=query.device,
            )
            for row, sources in rows.items():
                for source in sources:
                    if source < 0 or source >= key_length:
                        raise RuntimeError(
                            f"Source position {source} outside key length {key_length}"
                        )
                    if source >= query_length - 1:
                        raise RuntimeError(
                            f"Source position {source} is not before final query "
                            f"{query_length - 1}"
                        )
                    edge_mask[row, :, :, source] = False

            manual_key = key
            manual_value = value
            if final_query_tensor.shape[1] != key.shape[1]:
                if final_query_tensor.shape[1] % key.shape[1] != 0:
                    raise RuntimeError("Query heads are not an integer multiple of KV heads")
                repeats = final_query_tensor.shape[1] // key.shape[1]
                manual_key = key.repeat_interleave(repeats, dim=1)
                manual_value = value.repeat_interleave(repeats, dim=1)
            scale = kwargs.get("scale")
            if scale is None:
                scale = final_query_tensor.shape[-1] ** -0.5
            scores = torch.matmul(
                final_query_tensor.float(), manual_key.float().transpose(-2, -1)
            ) * float(scale)
            probabilities = torch.softmax(scores, dim=-1)
            masked_probabilities = torch.softmax(
                scores.masked_fill(~edge_mask, -torch.inf), dim=-1
            )
            manual_value_float = manual_value.float()
            natural_final = torch.matmul(probabilities, manual_value_float)
            masked_final = torch.matmul(masked_probabilities, manual_value_float)
            patched = natural.clone()
            patched[:, :, -1:, :] += (masked_final - natural_final).to(natural.dtype)
            return patched
        if mask.ndim != 4:
            raise RuntimeError(f"Unexpected SDPA mask shape {tuple(mask.shape)}")
        if mask.shape[0] == 1 and batch > 1:
            patched = mask.expand(batch, *mask.shape[1:]).clone()
        elif mask.shape[0] == batch:
            patched = mask.clone()
        else:
            raise RuntimeError(
                f"SDPA mask batch {mask.shape[0]} does not match query batch {batch}"
            )
        final_query = query_length - 1
        for row, sources in rows.items():
            for source in sources:
                if source < 0 or source >= key_length:
                    raise RuntimeError(
                        f"Source position {source} outside key length {key_length}"
                    )
                if source >= final_query:
                    raise RuntimeError(
                        f"Source position {source} is not before final query {final_query}"
                    )
                # PyTorch SDPA uses opposite conventions for Boolean and
                # additive masks: True means "allowed" for a Boolean mask,
                # whereas -inf means "blocked" for an additive mask.  Writing
                # -inf into a Boolean tensor casts to True and is therefore a
                # silent no-op.  Variable-length historical cohorts exercise
                # this Boolean-padding-mask path.
                if patched.dtype == torch.bool:
                    patched[row, :, final_query, source] = False
                else:
                    patched[row, :, final_query, source] = -torch.inf

        patched_kwargs = dict(kwargs)
        patched_kwargs["attn_mask"] = patched
        return self.original_sdpa(query, key, value, *args, **patched_kwargs)

    def close(self) -> None:
        import torch

        if hasattr(self, "original_sdpa"):
            torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
        for handle in reversed(getattr(self, "handles", [])):
            handle.remove()
        self.handles = []
        self.active_layer = None

    def __enter__(self) -> "BatchedSDPAFinalQueryAttentionAblator":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class BatchedSDPAFinalQuerySourceKVPatcher:
    """Patch selected source K/V only for the exact final physical query.

    ``specs_by_layer`` maps zero-based ordinary-attention layer indices to
    target rows, then ``(donor row, source positions)``.  Donor and recipient
    positions must be physically aligned.  The natural full SDPA call is kept
    for every query except the final query in the selected recipient rows.
    """

    def __init__(
        self,
        parts: Any,
        specs_by_layer: dict[int, dict[int, tuple[int, list[int]]]],
    ) -> None:
        import torch

        if not specs_by_layer:
            raise ValueError("No final-query K/V patches were specified")
        self.specs = {
            int(layer): {
                int(target): (
                    int(donor),
                    tuple(sorted(set(int(position) for position in positions))),
                )
                for target, (donor, positions) in rows.items()
            }
            for layer, rows in specs_by_layer.items()
        }
        if any(not rows for rows in self.specs.values()):
            raise ValueError("Every selected layer needs a recipient row")
        if any(
            target == donor or not positions
            for rows in self.specs.values()
            for target, (donor, positions) in rows.items()
        ):
            raise ValueError("Every patch needs a distinct donor and source positions")

        self.active_layer: int | None = None
        self.handles: list[Any] = []
        for layer_index in sorted(self.specs):
            attention = getattr(parts.layers[layer_index], "self_attn", None)
            if attention is None:
                self.close()
                raise ValueError(f"Layer {layer_index} is not ordinary attention")
            self.handles.append(
                attention.register_forward_pre_hook(self._enter(layer_index))
            )
            self.handles.append(
                attention.register_forward_hook(self._leave(layer_index))
            )
        self.original_sdpa = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = self._wrapped_sdpa

    def _enter(self, layer_index: int):
        def enter(_module: Any, _inputs: Any) -> None:
            if self.active_layer is not None:
                raise RuntimeError("Nested ordinary-attention calls are unsupported")
            self.active_layer = layer_index

        return enter

    def _leave(self, layer_index: int):
        def leave(_module: Any, _inputs: Any, _output: Any) -> None:
            if self.active_layer != layer_index:
                raise RuntimeError("Ordinary-attention layer stack became inconsistent")
            self.active_layer = None

        return leave

    def _wrapped_sdpa(self, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
        layer = self.active_layer
        if layer is None or layer not in self.specs:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        natural = self.original_sdpa(query, key, value, *args, **kwargs)
        patched_key = key.clone()
        patched_value = value.clone()
        for target, (donor, positions) in self.specs[layer].items():
            if max(target, donor) >= key.shape[0]:
                raise RuntimeError("K/V patch row exceeds the SDPA batch")
            selected = list(positions)
            if min(selected) < 0 or max(selected) >= key.shape[-2] - 1:
                raise RuntimeError("K/V source must precede the final query")
            patched_key[target, :, selected, :] = key[donor, :, selected, :]
            patched_value[target, :, selected, :] = value[donor, :, selected, :]
        edited = self.original_sdpa(
            query, patched_key, patched_value, *args, **kwargs
        )
        output = natural.clone()
        for target in self.specs[layer]:
            output[target, :, -1:, :] = edited[target, :, -1:, :]
        return output

    def close(self) -> None:
        import torch

        if hasattr(self, "original_sdpa"):
            torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
        for handle in reversed(getattr(self, "handles", [])):
            handle.remove()
        self.handles = []
        self.active_layer = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class BatchedSDPAQuerySourceAttentionAblator:
    """Block exact query-to-source edges in selected SDPA layers.

    ``specs_by_layer`` maps zero-based model-layer indices to batch rows, then
    to query positions and the source-token positions that query may no longer
    read.  All unlisted queries, sources, rows, and layers remain untouched.
    This is the arbitrary-query counterpart of
    :class:`BatchedSDPAFinalQueryAttentionAblator`.
    """

    def __init__(
        self,
        parts: Any,
        specs_by_layer: dict[int, dict[int, dict[int, list[int]]]],
    ) -> None:
        import torch

        if not specs_by_layer:
            raise ValueError("No query-to-source attention-edge ablations were specified")
        self.specs = {
            int(layer): {
                int(row): {
                    int(query): tuple(sorted(set(int(source) for source in sources)))
                    for query, sources in queries.items()
                }
                for row, queries in rows.items()
            }
            for layer, rows in specs_by_layer.items()
        }
        if any(not rows for rows in self.specs.values()):
            raise ValueError("Every selected layer needs at least one batch row")
        if any(
            not queries
            for rows in self.specs.values()
            for queries in rows.values()
        ):
            raise ValueError("Every selected row needs at least one query position")
        if any(
            not sources
            for rows in self.specs.values()
            for queries in rows.values()
            for sources in queries.values()
        ):
            raise ValueError("Every selected query needs at least one source position")

        self.active_layer: int | None = None
        self.handles: list[Any] = []
        self.layers_seen: list[int] = []
        self.sdpa_calls = 0
        self.edited_edge_count = 0
        for layer_index in sorted(self.specs):
            attention = getattr(parts.layers[layer_index], "self_attn", None)
            if attention is None:
                self.close()
                raise ValueError(
                    f"Layer {layer_index} is not a conventional-attention block"
                )
            self.handles.append(
                attention.register_forward_pre_hook(self._enter(layer_index))
            )
            self.handles.append(
                attention.register_forward_hook(self._leave(layer_index))
            )

        self.original_sdpa = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = self._wrapped_sdpa

    def _enter(self, layer_index: int):
        def enter(_module: Any, _inputs: Any) -> None:
            if self.active_layer is not None:
                raise RuntimeError("Nested ordinary-attention calls are unsupported")
            self.active_layer = layer_index
        return enter

    def _leave(self, layer_index: int):
        def leave(_module: Any, _inputs: Any, _output: Any) -> None:
            if self.active_layer != layer_index:
                raise RuntimeError("Ordinary-attention layer stack became inconsistent")
            self.active_layer = None
        return leave

    def _wrapped_sdpa(self, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
        import torch

        layer = self.active_layer
        if layer is None:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        self.layers_seen.append(layer)
        self.sdpa_calls += 1
        if args:
            raise RuntimeError(
                "The validated Qwen SDPA path passes attention options by keyword"
            )

        rows = self.specs[layer]
        self.edited_edge_count += sum(
            len(sources)
            for queries in rows.values()
            for sources in queries.values()
        )
        batch, _heads, query_length, _dim = query.shape
        key_length = key.shape[-2]
        if max(rows) >= batch:
            raise RuntimeError("Attention-ablation row exceeds the SDPA batch")
        for row, queries in rows.items():
            for query_position, sources in queries.items():
                if query_position < 0 or query_position >= query_length:
                    raise RuntimeError(
                        f"Query position {query_position} outside query length {query_length}"
                    )
                for source in sources:
                    if source < 0 or source >= key_length:
                        raise RuntimeError(
                            f"Source position {source} outside key length {key_length}"
                        )
                    if source > query_position:
                        raise RuntimeError(
                            f"Source position {source} follows causal query {query_position}"
                        )

        mask = kwargs.get("attn_mask")
        if mask is not None:
            if mask.ndim != 4:
                raise RuntimeError(f"Unexpected SDPA mask shape {tuple(mask.shape)}")
            if mask.shape[0] == 1 and batch > 1:
                patched = mask.expand(batch, *mask.shape[1:]).clone()
            elif mask.shape[0] == batch:
                patched = mask.clone()
            else:
                raise RuntimeError(
                    f"SDPA mask batch {mask.shape[0]} does not match query batch {batch}"
                )
            for row, queries in rows.items():
                for query_position, sources in queries.items():
                    for source in sources:
                        if patched.dtype == torch.bool:
                            patched[row, :, query_position, source] = False
                        else:
                            patched[row, :, query_position, source] = -torch.inf
            patched_kwargs = dict(kwargs)
            patched_kwargs["attn_mask"] = patched
            return self.original_sdpa(query, key, value, **patched_kwargs)

        # With no explicit padding mask, preserve the complete natural SDPA
        # result and add a float32 masked-minus-unmasked delta only at selected
        # query rows.  This avoids changing the numerical kernel for all other
        # queries while respecting the implicit causal mask.
        natural = self.original_sdpa(query, key, value, **kwargs)
        patched = natural.clone()
        causal = bool(kwargs.get("is_causal", False))
        key_indices = torch.arange(key_length, device=query.device)
        # Vectorize every selected query for a batch row.  The earlier exact
        # implementation evaluated one tiny matmul per query token, which is
        # numerically equivalent but prohibitively slow for a complete prompt
        # region.  This retains the untouched natural SDPA result and adds the
        # same float32 masked-minus-unmasked delta only at selected queries.
        for row, queries in rows.items():
            query_positions = sorted(queries)
            position_tensor = torch.as_tensor(
                query_positions, dtype=torch.long, device=query.device
            )
            selected_query = query[
                row : row + 1, :, position_tensor, :
            ]
            allowed = torch.ones(
                (1, 1, len(query_positions), key_length),
                dtype=torch.bool,
                device=query.device,
            )
            if causal:
                allowed &= (
                    key_indices.view(1, 1, 1, -1)
                    <= position_tensor.view(1, 1, -1, 1)
                )
            masked_allowed = allowed.clone()
            for local_index, query_position in enumerate(query_positions):
                masked_allowed[
                    :, :, local_index, list(queries[query_position])
                ] = False
            row_key = key[row : row + 1]
            row_value = value[row : row + 1]
            if selected_query.shape[1] != row_key.shape[1]:
                if selected_query.shape[1] % row_key.shape[1] != 0:
                    raise RuntimeError(
                        "Query heads are not an integer multiple of KV heads"
                    )
                repeats = selected_query.shape[1] // row_key.shape[1]
                row_key = row_key.repeat_interleave(repeats, dim=1)
                row_value = row_value.repeat_interleave(repeats, dim=1)
            scale = kwargs.get("scale")
            if scale is None:
                scale = selected_query.shape[-1] ** -0.5
            scores = torch.matmul(
                selected_query.float(), row_key.float().transpose(-2, -1)
            ) * float(scale)
            natural_probabilities = torch.softmax(
                scores.masked_fill(~allowed, -torch.inf), dim=-1
            )
            masked_probabilities = torch.softmax(
                scores.masked_fill(~masked_allowed, -torch.inf), dim=-1
            )
            row_value_float = row_value.float()
            natural_queries = torch.matmul(
                natural_probabilities, row_value_float
            )
            masked_queries = torch.matmul(
                masked_probabilities, row_value_float
            )
            patched[row, :, position_tensor, :] = (
                patched[row, :, position_tensor, :]
                + (masked_queries - natural_queries)[0].to(natural.dtype)
            )
        return patched

    def close(self) -> None:
        import torch

        if hasattr(self, "original_sdpa"):
            torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
        for handle in reversed(getattr(self, "handles", [])):
            handle.remove()
        self.handles = []
        self.active_layer = None

    def __enter__(self):
        return self

    def assert_fired(self) -> None:
        expected_layers = sorted(self.specs)
        if sorted(set(self.layers_seen)) != expected_layers:
            raise RuntimeError(
                "Query/source attention ablator did not fire at every selected layer: "
                f"seen={sorted(set(self.layers_seen))}, expected={expected_layers}"
            )
        if self.sdpa_calls != len(expected_layers) or self.edited_edge_count <= 0:
            raise RuntimeError(
                "Query/source attention ablator call or edge count was invalid: "
                f"calls={self.sdpa_calls}, edges={self.edited_edge_count}"
            )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class BatchedSDPADownstreamAttentionAblator:
    """Block later ordinary-attention queries from one or more source tokens.

    This is the SDPA-preserving counterpart of
    :class:`BatchedDownstreamAttentionAblator`.  It changes only the causal
    mask entries ``query > source, key == source`` in conventional-attention
    blocks.  The model's SDPA kernel, batching, and every other attention edge
    remain unchanged.
    """

    def __init__(self, parts: Any, source_positions_by_row: dict[int, list[int]]) -> None:
        import torch

        if not source_positions_by_row:
            raise ValueError("No downstream attention source ablations were specified")
        self.specs = {
            int(row): tuple(sorted(set(int(position) for position in positions)))
            for row, positions in source_positions_by_row.items()
        }
        if any(not positions for positions in self.specs.values()):
            raise ValueError("Every selected row needs at least one source position")

        self.active_layer: int | None = None
        self.handles: list[Any] = []
        self.layers: list[int] = []
        self.sdpa_calls = 0
        self.edited_edge_count = 0
        self.layers_seen: set[int] = set()
        for layer_index, layer in enumerate(parts.layers):
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                continue
            self.layers.append(layer_index)
            self.handles.append(attention.register_forward_pre_hook(self._enter(layer_index)))
            self.handles.append(attention.register_forward_hook(self._leave(layer_index)))
        if not self.layers:
            self.close()
            raise ValueError("The model has no conventional-attention blocks")

        self.original_sdpa = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = self._wrapped_sdpa

    def _enter(self, layer_index: int):
        def enter(_module: Any, _inputs: Any) -> None:
            if self.active_layer is not None:
                raise RuntimeError("Nested ordinary-attention calls are unsupported")
            self.active_layer = layer_index
        return enter

    def _leave(self, layer_index: int):
        def leave(_module: Any, _inputs: Any, _output: Any) -> None:
            if self.active_layer != layer_index:
                raise RuntimeError("Ordinary-attention layer stack became inconsistent")
            self.active_layer = None
        return leave

    def _wrapped_sdpa(self, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
        import torch

        if self.active_layer is None:
            return self.original_sdpa(query, key, value, *args, **kwargs)

        self.sdpa_calls += 1
        self.layers_seen.add(self.active_layer)

        positional_mask = bool(args)
        mask = args[0] if positional_mask else kwargs.get("attn_mask")
        batch, _heads, query_length, _dim = query.shape
        key_length = key.shape[-2]
        if max(self.specs) >= batch:
            raise RuntimeError("Attention-ablation row exceeds the SDPA batch")

        if mask is None:
            patched = torch.zeros(
                (batch, 1, query_length, key_length),
                device=query.device,
                dtype=query.dtype,
            )
        else:
            if mask.ndim != 4:
                raise RuntimeError(f"Unexpected SDPA mask shape {tuple(mask.shape)}")
            if mask.shape[0] == 1 and batch > 1:
                patched = mask.expand(batch, *mask.shape[1:]).clone()
            elif mask.shape[0] == batch:
                patched = mask.clone()
            else:
                raise RuntimeError(
                    f"SDPA mask batch {mask.shape[0]} does not match query batch {batch}"
                )
        for row, sources in self.specs.items():
            for source in sources:
                if source < 0 or source >= key_length:
                    raise RuntimeError(f"Source position {source} outside key length {key_length}")
                if source + 1 < query_length:
                    # PyTorch SDPA uses opposite conventions for Boolean and
                    # additive masks: True means "allowed" for a Boolean mask,
                    # whereas -inf means "blocked" for an additive mask.
                    # Assigning -inf to a Boolean tensor casts to True and is a
                    # silent no-op, which is precisely the left-padded cohort
                    # path exercised by the cue and action-period experiments.
                    if patched.dtype == torch.bool:
                        patched[row, :, source + 1 :, source] = False
                    else:
                        patched[row, :, source + 1 :, source] = -torch.inf
                    self.edited_edge_count += query_length - (source + 1)

        if positional_mask:
            patched_args = (patched,) + args[1:]
            return self.original_sdpa(query, key, value, *patched_args, **kwargs)
        patched_kwargs = dict(kwargs)
        patched_kwargs["attn_mask"] = patched
        return self.original_sdpa(query, key, value, *args, **patched_kwargs)

    def assert_fired(self) -> None:
        """Fail closed when a requested intervention never edited an edge."""

        if self.sdpa_calls <= 0 or self.edited_edge_count <= 0:
            raise RuntimeError(
                "Downstream attention ablator did not edit any SDPA edge "
                f"(calls={self.sdpa_calls}, edges={self.edited_edge_count})"
            )
        missing = set(self.layers) - self.layers_seen
        if missing:
            raise RuntimeError(
                "Downstream attention ablator did not observe selected layers: "
                f"{sorted(missing)}"
            )

    def close(self) -> None:
        import torch

        if hasattr(self, "original_sdpa"):
            torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
        for handle in reversed(getattr(self, "handles", [])):
            handle.remove()
        self.handles = []
        self.active_layer = None


    def __enter__(self) -> "BatchedSDPADownstreamAttentionAblator":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class BatchedSDPADownstreamSourceKVPatcher:
    """Make only later queries see paired donor K/V at selected source tokens.

    ``specs_by_target_row`` maps a target row to ``(source row, positions,
    model-layer indices)``.  The source positions must be physically aligned
    across the paired rows.  The target's own source-token output is kept
    natural; only queries strictly after the source receive the donor K/V.
    """

    def __init__(
        self,
        parts: Any,
        specs_by_target_row: dict[int, tuple[int, list[int], list[int]]],
    ) -> None:
        import torch

        if not specs_by_target_row:
            raise ValueError("No downstream K/V patches were specified")
        self.by_layer: dict[int, dict[int, tuple[int, tuple[int, ...]]]] = {}
        for target, (source, positions, layers) in specs_by_target_row.items():
            selected = tuple(sorted(set(int(value) for value in positions)))
            if not selected or not layers:
                raise ValueError("Every K/V patch needs positions and layers")
            if int(target) == int(source):
                raise ValueError("K/V donor and recipient rows must differ")
            for layer in sorted(set(int(value) for value in layers)):
                self.by_layer.setdefault(layer, {})[int(target)] = (
                    int(source), selected
                )

        self.active_layer: int | None = None
        self.handles: list[Any] = []
        self.selected_layers = set(self.by_layer)
        self.sdpa_calls = 0
        self.patched_position_count = 0
        self.layers_seen: set[int] = set()
        for layer_index in sorted(self.by_layer):
            attention = getattr(parts.layers[layer_index], "self_attn", None)
            if attention is None:
                self.close()
                raise ValueError(f"Layer {layer_index} is not ordinary attention")
            self.handles.append(attention.register_forward_pre_hook(self._enter(layer_index)))
            self.handles.append(attention.register_forward_hook(self._leave(layer_index)))
        self.original_sdpa = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = self._wrapped_sdpa

    def _enter(self, layer_index: int):
        def enter(_module: Any, _inputs: Any) -> None:
            if self.active_layer is not None:
                raise RuntimeError("Nested ordinary-attention calls are unsupported")
            self.active_layer = layer_index
        return enter

    def _leave(self, layer_index: int):
        def leave(_module: Any, _inputs: Any, _output: Any) -> None:
            if self.active_layer != layer_index:
                raise RuntimeError("Ordinary-attention layer stack became inconsistent")
            self.active_layer = None
        return leave

    def _wrapped_sdpa(self, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
        layer = self.active_layer
        if layer is None or layer not in self.by_layer:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        self.sdpa_calls += 1
        self.layers_seen.add(layer)
        natural = self.original_sdpa(query, key, value, *args, **kwargs)
        patched_key = key.clone()
        patched_value = value.clone()
        cutoffs: dict[int, int] = {}
        for target, (source, positions) in self.by_layer[layer].items():
            if max(target, source) >= key.shape[0]:
                raise RuntimeError("K/V patch row exceeds the SDPA batch")
            selected = list(positions)
            if min(selected) < 0 or max(selected) >= key.shape[-2]:
                raise RuntimeError("K/V source position exceeds the sequence")
            patched_key[target, :, selected, :] = key[source, :, selected, :]
            patched_value[target, :, selected, :] = value[source, :, selected, :]
            cutoffs[target] = max(selected)
            self.patched_position_count += len(selected)
        edited = self.original_sdpa(
            query, patched_key, patched_value, *args, **kwargs
        )
        output = natural.clone()
        for row, cutoff in cutoffs.items():
            if cutoff + 1 < output.shape[-2]:
                output[row, :, cutoff + 1 :, :] = edited[row, :, cutoff + 1 :, :]
        return output

    def assert_fired(self) -> None:
        """Fail closed when no selected downstream K/V state was patched."""

        if self.sdpa_calls <= 0 or self.patched_position_count <= 0:
            raise RuntimeError(
                "Downstream source K/V patcher did not patch any position "
                f"(calls={self.sdpa_calls}, positions={self.patched_position_count})"
            )
        missing = self.selected_layers - self.layers_seen
        if missing:
            raise RuntimeError(
                "Downstream source K/V patcher did not observe selected layers: "
                f"{sorted(missing)}"
            )

    def close(self) -> None:
        import torch

        if hasattr(self, "original_sdpa"):
            torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
        for handle in reversed(getattr(self, "handles", [])):
            handle.remove()
        self.handles = []
        self.active_layer = None


class BatchedSDPAFeedbackHistoryFactorial:
    """Combine downstream suffix K/V transfer with exact history-edge lesions.

    The suffix specification maps each target row to a distinct donor row,
    physically aligned suffix positions, and selected model layers.  Only
    queries after the suffix see donor K/V.  The optional history specification
    maps layers to rows, query positions, and blocked source positions.  Both
    edits are applied in the same SDPA evaluation so the resulting factorial
    cell is not inferred from separate forwards.
    """

    def __init__(
        self,
        parts: Any,
        suffix_by_target_row: dict[int, tuple[int, list[int], list[int]]],
        history_by_layer: dict[int, dict[int, dict[int, list[int]]]] | None = None,
    ) -> None:
        import torch

        if not suffix_by_target_row:
            raise ValueError("No feedback-suffix K/V patches were specified")
        self.suffix_by_layer: dict[int, dict[int, tuple[int, tuple[int, ...]]]] = {}
        for target, (donor, positions, layers) in suffix_by_target_row.items():
            selected = tuple(sorted(set(int(value) for value in positions)))
            if int(target) == int(donor) or not selected or not layers:
                raise ValueError("Every suffix patch needs a distinct donor, positions, and layers")
            for layer in sorted(set(int(value) for value in layers)):
                self.suffix_by_layer.setdefault(layer, {})[int(target)] = (
                    int(donor), selected
                )
        self.history = {
            int(layer): {
                int(row): {
                    int(query): tuple(sorted(set(int(source) for source in sources)))
                    for query, sources in queries.items()
                }
                for row, queries in rows.items()
            }
            for layer, rows in (history_by_layer or {}).items()
        }
        if set(self.history) - set(self.suffix_by_layer):
            raise ValueError("History lesions select layers without suffix patches")
        if any(
            not sources
            for rows in self.history.values()
            for queries in rows.values()
            for sources in queries.values()
        ):
            raise ValueError("Every selected history query needs source positions")

        self.selected_layers = set(self.suffix_by_layer)
        self.active_layer: int | None = None
        self.handles: list[Any] = []
        self.layers_seen: set[int] = set()
        self.sdpa_calls = 0
        self.patched_position_count = 0
        self.edited_edge_count = 0
        for layer_index in sorted(self.selected_layers):
            attention = getattr(parts.layers[layer_index], "self_attn", None)
            if attention is None:
                self.close()
                raise ValueError(f"Layer {layer_index} is not ordinary attention")
            self.handles.append(attention.register_forward_pre_hook(self._enter(layer_index)))
            self.handles.append(attention.register_forward_hook(self._leave(layer_index)))
        self.original_sdpa = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = self._wrapped_sdpa

    def _enter(self, layer_index: int):
        def enter(_module: Any, _inputs: Any) -> None:
            if self.active_layer is not None:
                raise RuntimeError("Nested ordinary-attention calls are unsupported")
            self.active_layer = layer_index
        return enter

    def _leave(self, layer_index: int):
        def leave(_module: Any, _inputs: Any, _output: Any) -> None:
            if self.active_layer != layer_index:
                raise RuntimeError("Ordinary-attention layer stack became inconsistent")
            self.active_layer = None
        return leave

    def _wrapped_sdpa(self, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
        import torch

        layer = self.active_layer
        if layer is None or layer not in self.suffix_by_layer:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        if args:
            raise RuntimeError("The validated Seed SDPA path passes options by keyword")
        self.sdpa_calls += 1
        self.layers_seen.add(layer)
        natural = self.original_sdpa(query, key, value, **kwargs)
        patched_key = key.clone()
        patched_value = value.clone()
        cutoffs: dict[int, int] = {}
        for target, (donor, positions) in self.suffix_by_layer[layer].items():
            if max(target, donor) >= key.shape[0]:
                raise RuntimeError("Suffix K/V patch row exceeds the SDPA batch")
            selected = list(positions)
            if min(selected) < 0 or max(selected) >= key.shape[-2]:
                raise RuntimeError("Suffix source position exceeds the sequence")
            patched_key[target, :, selected, :] = key[donor, :, selected, :]
            patched_value[target, :, selected, :] = value[donor, :, selected, :]
            cutoffs[target] = max(selected)
            self.patched_position_count += len(selected)

        policy_output = self.original_sdpa(query, patched_key, patched_value, **kwargs)
        output = natural.clone()
        for row, cutoff in cutoffs.items():
            if cutoff + 1 < output.shape[-2]:
                output[row, :, cutoff + 1 :, :] = policy_output[row, :, cutoff + 1 :, :]

        rows = self.history.get(layer)
        if not rows:
            return output
        self.edited_edge_count += sum(
            len(sources)
            for queries in rows.values()
            for sources in queries.values()
        )
        batch, _heads, query_length, _dim = query.shape
        key_length = key.shape[-2]
        if max(rows) >= batch:
            raise RuntimeError("History-lesion row exceeds the SDPA batch")
        for row, queries in rows.items():
            if row not in cutoffs:
                raise RuntimeError("History lesion targets a row without suffix installation")
            for query_position, sources in queries.items():
                if query_position <= cutoffs[row] or query_position >= query_length:
                    raise RuntimeError("History query must be after the installed suffix")
                if any(source < 0 or source > query_position or source >= key_length for source in sources):
                    raise RuntimeError("Invalid causal history source position")

        mask = kwargs.get("attn_mask")
        if mask is not None:
            if mask.ndim != 4:
                raise RuntimeError(f"Unexpected SDPA mask shape {tuple(mask.shape)}")
            if mask.shape[0] == 1 and batch > 1:
                edited_mask = mask.expand(batch, *mask.shape[1:]).clone()
            elif mask.shape[0] == batch:
                edited_mask = mask.clone()
            else:
                raise RuntimeError("SDPA mask batch does not match query batch")
            for row, queries in rows.items():
                for query_position, sources in queries.items():
                    for source in sources:
                        if edited_mask.dtype == torch.bool:
                            edited_mask[row, :, query_position, source] = False
                        else:
                            edited_mask[row, :, query_position, source] = -torch.inf
            edited_kwargs = dict(kwargs)
            edited_kwargs["attn_mask"] = edited_mask
            combined = self.original_sdpa(
                query, patched_key, patched_value, **edited_kwargs
            )
            for row, queries in rows.items():
                selected = list(queries)
                output[row, :, selected, :] = combined[row, :, selected, :]
            return output

        causal = bool(kwargs.get("is_causal", False))
        key_indices = torch.arange(key_length, device=query.device)
        for row, queries in rows.items():
            query_positions = sorted(queries)
            position_tensor = torch.as_tensor(
                query_positions, dtype=torch.long, device=query.device
            )
            selected_query = query[row : row + 1, :, position_tensor, :]
            allowed = torch.ones(
                (1, 1, len(query_positions), key_length),
                dtype=torch.bool,
                device=query.device,
            )
            if causal:
                allowed &= (
                    key_indices.view(1, 1, 1, -1)
                    <= position_tensor.view(1, 1, -1, 1)
                )
            blocked_allowed = allowed.clone()
            for local, query_position in enumerate(query_positions):
                blocked_allowed[:, :, local, list(queries[query_position])] = False
            row_key = patched_key[row : row + 1]
            row_value = patched_value[row : row + 1]
            if selected_query.shape[1] != row_key.shape[1]:
                if selected_query.shape[1] % row_key.shape[1] != 0:
                    raise RuntimeError("Query heads are not an integer multiple of KV heads")
                repeats = selected_query.shape[1] // row_key.shape[1]
                row_key = row_key.repeat_interleave(repeats, dim=1)
                row_value = row_value.repeat_interleave(repeats, dim=1)
            scale = kwargs.get("scale")
            if scale is None:
                scale = selected_query.shape[-1] ** -0.5
            scores = torch.matmul(
                selected_query.float(), row_key.float().transpose(-2, -1)
            ) * float(scale)
            natural_probabilities = torch.softmax(
                scores.masked_fill(~allowed, -torch.inf), dim=-1
            )
            blocked_probabilities = torch.softmax(
                scores.masked_fill(~blocked_allowed, -torch.inf), dim=-1
            )
            row_value_float = row_value.float()
            natural_queries = torch.matmul(natural_probabilities, row_value_float)
            blocked_queries = torch.matmul(blocked_probabilities, row_value_float)
            output[row, :, position_tensor, :] += (
                blocked_queries - natural_queries
            )[0].to(output.dtype)
        return output

    def assert_fired(self) -> None:
        if self.layers_seen != self.selected_layers:
            raise RuntimeError(
                "Combined factorial did not fire at every layer: "
                f"seen={sorted(self.layers_seen)}, expected={sorted(self.selected_layers)}"
            )
        if self.sdpa_calls != len(self.selected_layers) or self.patched_position_count <= 0:
            raise RuntimeError("Combined suffix patch did not execute completely")
        if self.history and self.edited_edge_count <= 0:
            raise RuntimeError("Combined history lesion did not edit any edge")

    def close(self) -> None:
        import torch

        if hasattr(self, "original_sdpa"):
            torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
        for handle in reversed(getattr(self, "handles", [])):
            handle.remove()
        self.handles = []
        self.active_layer = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class BatchedDownstreamAttentionAblator:
    """Remove source-token edges for every causally later query and every head.

    ``source_positions_by_row`` maps selected batch rows to the prompt positions
    whose ordinary-attention broadcasts should be blocked.  For each source at
    position ``s``, all attention logits ``query > s, key == s`` are set to
    negative infinity before softmax in every conventional-attention block.
    """

    _attribute = "_secondchance_downstream_source_ablation"

    def __init__(self, parts: Any, source_positions_by_row: dict[int, list[int]]) -> None:
        if not source_positions_by_row:
            raise ValueError("No downstream attention source ablations were specified")
        self.specs = {
            int(row): tuple(sorted(set(int(position) for position in positions)))
            for row, positions in source_positions_by_row.items()
        }
        if any(not positions for positions in self.specs.values()):
            raise ValueError("Every selected row needs at least one source position")

        self.modules = []
        self.modeling_module = None
        self.original = None
        self.attention_calls = 0
        self.edited_edge_count = 0
        for layer in parts.layers:
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                continue
            module = inspect.getmodule(type(attention))
            if module is None or not hasattr(module, "eager_attention_forward"):
                self.close()
                raise RuntimeError("Could not locate Qwen eager attention implementation")
            if self.modeling_module is not None and module is not self.modeling_module:
                self.close()
                raise RuntimeError("Ordinary-attention blocks use different implementations")
            self.modeling_module = module
            setattr(attention, self._attribute, self.specs)
            self.modules.append(attention)
        if not self.modules:
            raise ValueError("The model has no conventional-attention blocks")

        self.original = self.modeling_module.eager_attention_forward
        repeat_kv = self.original.__globals__.get("repeat_kv")
        if repeat_kv is None:
            self.close()
            raise RuntimeError("Installed eager attention function does not expose repeat_kv")
        original = self.original

        def intervened_eager_attention_forward(
            module: Any,
            query: Any,
            key: Any,
            value: Any,
            attention_mask: Any,
            scaling: float,
            dropout: float = 0.0,
            **kwargs: Any,
        ):
            import torch

            specs = getattr(module, self._attribute, None)
            if specs is None:
                return original(
                    module, query, key, value, attention_mask, scaling,
                    dropout=dropout, **kwargs,
                )
            key_states = repeat_kv(key, module.num_key_value_groups)
            self.attention_calls += 1
            value_states = repeat_kv(value, module.num_key_value_groups)
            weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                weights = weights + attention_mask
            weights = weights.clone()
            sequence_length = weights.shape[-2]
            for row, sources in specs.items():
                for source in sources:
                    if source + 1 < sequence_length:
                        weights[row, :, source + 1 :, source] = -torch.inf
                        self.edited_edge_count += sequence_length - source - 1
            weights = torch.nn.functional.softmax(
                weights, dim=-1, dtype=torch.float32
            ).to(query.dtype)
            weights = torch.nn.functional.dropout(
                weights, p=dropout, training=module.training
            )
            output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
            return output, weights

        self.modeling_module.eager_attention_forward = intervened_eager_attention_forward

    def assert_fired(self) -> None:
        if self.attention_calls <= 0 or self.edited_edge_count <= 0:
            raise RuntimeError(
                "Downstream eager-attention ablator did not edit any edge "
                f"(calls={self.attention_calls}, edges={self.edited_edge_count})"
            )

    def close(self) -> None:
        for module in getattr(self, "modules", []):
            if hasattr(module, self._attribute):
                delattr(module, self._attribute)
        if self.modeling_module is not None and self.original is not None:
            self.modeling_module.eager_attention_forward = self.original


class BatchedGDNSourceWriteAblator:
    """Remove selected source-token writes from every Gated DeltaNet block."""

    def __init__(self, parts: Any, source_positions_by_row: dict[int, list[int]]) -> None:
        if not source_positions_by_row:
            raise ValueError("No GDN source-write ablations were specified")
        self.specs = {
            int(row): tuple(sorted(set(int(position) for position in positions)))
            for row, positions in source_positions_by_row.items()
        }
        if any(not positions for positions in self.specs.values()):
            raise ValueError("Every selected row needs at least one source position")
        self.originals: list[tuple[Any, Any]] = []
        self.rule_calls = 0
        self.edited_position_count = 0
        for layer in parts.layers:
            module = getattr(layer, "linear_attn", None)
            if module is None:
                continue
            original = module.chunk_gated_delta_rule

            def wrapped(*args: Any, _original=original, **kwargs: Any):
                if "beta" not in kwargs:
                    raise RuntimeError("Qwen Gated DeltaNet did not pass beta by keyword")
                self.rule_calls += 1
                beta = kwargs["beta"].clone()
                for row, positions in self.specs.items():
                    beta[row, list(positions), :] = 0
                    self.edited_position_count += len(positions)
                kwargs["beta"] = beta
                return _original(*args, **kwargs)

            self.originals.append((module, original))
            module.chunk_gated_delta_rule = wrapped
        if not self.originals:
            raise ValueError("The model has no Gated DeltaNet blocks")

    def assert_fired(self) -> None:
        """Fail closed when the recurrent-memory intervention was inert."""

        if self.rule_calls <= 0 or self.edited_position_count <= 0:
            raise RuntimeError(
                "GLA source-write ablator did not edit any write "
                f"(calls={self.rule_calls}, positions={self.edited_position_count})"
            )

    def close(self) -> None:
        for module, original in reversed(self.originals):
            module.chunk_gated_delta_rule = original
        self.originals.clear()


class BatchedSelectiveGDNSourceWriteAblator:
    """Apply row-specific token-write ablations to selected GLA layers.

    ``specs_by_row`` maps a batch row to ``(source positions, model-layer
    indices)``.  At a selected Gated DeltaNet layer, beta is set to zero for
    those source positions in every value head.  Unselected rows and layers
    remain natural.
    """

    def __init__(
        self,
        parts: Any,
        specs_by_row: dict[int, tuple[list[int], list[int]]],
        preserve_source_output: bool = False,
    ) -> None:
        if not specs_by_row:
            raise ValueError("No selective GDN source-write ablations were specified")
        by_layer: dict[int, dict[int, tuple[int, ...]]] = {}
        for row, (positions, layers) in specs_by_row.items():
            source = tuple(sorted(set(int(position) for position in positions)))
            if not source:
                raise ValueError(f"Batch row {row} has no source positions")
            if not layers:
                raise ValueError(f"Batch row {row} has no selected layers")
            for layer in sorted(set(int(value) for value in layers)):
                by_layer.setdefault(layer, {})[int(row)] = source

        self.originals: list[tuple[Any, Any]] = []
        self.hook_handles: list[Any] = []
        self.modeling_module: Any | None = None
        self.original_global_rule: Any | None = None
        self.active_layer: int | None = None
        self.preserve_source_output = bool(preserve_source_output)
        self.selected_layers = set(by_layer)
        self.rule_calls = 0
        self.edited_position_count = 0
        self.layers_seen: set[int] = set()

        selected_modules = {
            layer_index: getattr(parts.layers[layer_index], "linear_attn", None)
            for layer_index in by_layer
        }
        use_global_rule = bool(selected_modules) and all(
            module is not None and not hasattr(module, "chunk_gated_delta_rule")
            for module in selected_modules.values()
        )
        if use_global_rule:
            modeling_modules = {
                inspect.getmodule(type(module)) for module in selected_modules.values()
            }
            if len(modeling_modules) != 1 or None in modeling_modules:
                raise RuntimeError("Could not identify one shared Qwen GLA modeling module")
            self.modeling_module = modeling_modules.pop()
            if not hasattr(self.modeling_module, "torch_chunk_gated_delta_rule"):
                raise RuntimeError("Qwen modeling module has no chunk GLA rule")
            self.original_global_rule = self.modeling_module.torch_chunk_gated_delta_rule

            for layer_index, module in selected_modules.items():
                def mark_active(_module: Any, _args: Any, _layer=layer_index) -> None:
                    if self.active_layer is not None:
                        raise RuntimeError("Nested Qwen GLA forward calls are unsupported")
                    self.active_layer = _layer

                def clear_active(_module: Any, _args: Any, _output: Any) -> None:
                    self.active_layer = None

                self.hook_handles.append(module.register_forward_pre_hook(mark_active))
                self.hook_handles.append(module.register_forward_hook(clear_active))

            original_global_rule = self.original_global_rule

            def wrapped_global_rule(
                query: Any,
                key: Any,
                value: Any,
                *args: Any,
                **kwargs: Any,
            ):
                layer_index = self.active_layer
                if layer_index is None or layer_index not in by_layer:
                    return original_global_rule(query, key, value, *args, **kwargs)
                if "beta" not in kwargs:
                    raise RuntimeError("Qwen GLA rule did not pass beta by keyword")
                self.rule_calls += 1
                self.layers_seen.add(layer_index)
                beta = kwargs["beta"].clone()
                for row, positions in by_layer[layer_index].items():
                    beta[row, list(positions), :] = 0
                    self.edited_position_count += len(positions)
                patched_kwargs = dict(kwargs)
                patched_kwargs["beta"] = beta
                edited = original_global_rule(
                    query, key, value, *args, **patched_kwargs
                )
                if not self.preserve_source_output:
                    return edited
                natural = original_global_rule(query, key, value, *args, **kwargs)
                cutoffs = {
                    row: max(positions)
                    for row, positions in by_layer[layer_index].items()
                }
                return _replace_only_causally_later_gla_outputs(
                    natural, edited, cutoffs
                )

            self.modeling_module.torch_chunk_gated_delta_rule = wrapped_global_rule
            return

        for layer_index, row_specs in by_layer.items():
            if layer_index < 0 or layer_index >= len(parts.layers):
                self.close()
                raise ValueError(f"Invalid model layer: {layer_index}")
            module = getattr(parts.layers[layer_index], "linear_attn", None)
            if module is None:
                self.close()
                raise ValueError(f"Layer {layer_index} is not Gated DeltaNet")
            original = module.chunk_gated_delta_rule

            def wrapped(
                *args: Any,
                _original=original,
                _row_specs=row_specs,
                _layer_index=layer_index,
                **kwargs: Any,
            ):
                if "beta" not in kwargs:
                    raise RuntimeError("Qwen Gated DeltaNet did not pass beta by keyword")
                self.rule_calls += 1
                self.layers_seen.add(_layer_index)
                beta = kwargs["beta"].clone()
                for row, positions in _row_specs.items():
                    beta[row, list(positions), :] = 0
                    self.edited_position_count += len(positions)
                patched_kwargs = dict(kwargs)
                patched_kwargs["beta"] = beta
                edited = _original(*args, **patched_kwargs)
                if not self.preserve_source_output:
                    return edited
                natural = _original(*args, **kwargs)
                cutoffs = {row: max(positions) for row, positions in _row_specs.items()}
                return _replace_only_causally_later_gla_outputs(
                    natural, edited, cutoffs
                )

            self.originals.append((module, original))
            module.chunk_gated_delta_rule = wrapped

    def assert_fired(self) -> None:
        """Fail closed when no selected recurrent-memory write was edited."""

        if self.rule_calls <= 0 or self.edited_position_count <= 0:
            raise RuntimeError(
                "Selective GLA source-write ablator did not edit any write "
                f"(calls={self.rule_calls}, positions={self.edited_position_count})"
            )
        missing = self.selected_layers - self.layers_seen
        if missing:
            raise RuntimeError(
                "Selective GLA source-write ablator did not observe selected layers: "
                f"{sorted(missing)}"
            )

    def close(self) -> None:
        for handle in reversed(getattr(self, "hook_handles", [])):
            handle.remove()
        self.hook_handles.clear()
        if self.modeling_module is not None and self.original_global_rule is not None:
            self.modeling_module.torch_chunk_gated_delta_rule = self.original_global_rule
            self.original_global_rule = None
            self.modeling_module = None
            self.active_layer = None
        for module, original in reversed(self.originals):
            module.chunk_gated_delta_rule = original
        self.originals.clear()


class BatchedSelectiveGDNSourceWritePatcher:
    """Transplant paired source-token GLA memory writes between batch rows.

    ``specs_by_target_row`` maps a target batch row to
    ``(source_row, token positions, model-layer indices)``.  At each selected
    Gated DeltaNet layer, the target row receives the source row's key, value,
    decay gate ``g``, and write strength ``beta`` at exactly those positions.
    Those four tensors fully determine the recurrent memory update; the query
    is deliberately left untouched because it reads from, rather than writes
    to, the recurrent state.

    This is the directional counterpart of setting ``beta=0`` in
    :class:`BatchedSelectiveGDNSourceWriteAblator`: it asks whether the Game
    write is sufficient to make a token-aligned Neutral prompt behave more
    like Game while preserving the rest of Neutral's computation.
    """

    def __init__(
        self,
        parts: Any,
        specs_by_target_row: dict[
            int,
            tuple[int, list[int], list[int]]
            | list[tuple[int, list[int], list[int]]],
        ],
        preserve_source_output: bool = False,
    ) -> None:
        if not specs_by_target_row:
            raise ValueError("No selective GDN source-write patches were specified")
        by_layer: dict[
            int, dict[int, list[tuple[int, tuple[int, ...]]]]
        ] = {}
        for target_row, raw_groups in specs_by_target_row.items():
            groups = raw_groups if isinstance(raw_groups, list) else [raw_groups]
            if not groups:
                raise ValueError(f"Batch row {target_row} has no patch groups")
            for source_row, positions, layers in groups:
                selected_positions = tuple(
                    sorted(set(int(position) for position in positions))
                )
                if not selected_positions:
                    raise ValueError(f"Batch row {target_row} has no source positions")
                if not layers:
                    raise ValueError(f"Batch row {target_row} has no selected layers")
                if int(target_row) == int(source_row):
                    raise ValueError("Source and target rows must differ")
                for layer in sorted(set(int(value) for value in layers)):
                    by_layer.setdefault(layer, {}).setdefault(
                        int(target_row), []
                    ).append((int(source_row), selected_positions))

        self.originals: list[tuple[Any, Any]] = []
        self.hook_handles: list[Any] = []
        self.modeling_module: Any | None = None
        self.original_global_rule: Any | None = None
        self.active_layer: int | None = None
        self.preserve_source_output = bool(preserve_source_output)
        self.selected_layers = set(by_layer)
        self.rule_calls = 0
        self.edited_position_count = 0
        self.layers_seen: set[int] = set()

        # Transformers releases through the original Qwen3.6 experiments kept
        # ``chunk_gated_delta_rule`` as an instance attribute.  Newer releases
        # call a module-global ``torch_chunk_gated_delta_rule`` instead.  The
        # tensors and intervention are identical, but the hook point moved.
        # Handle the newer layout with per-layer pre/post hooks that identify
        # which GLA block is currently invoking the shared global rule.
        selected_modules = {
            layer_index: getattr(parts.layers[layer_index], "linear_attn", None)
            for layer_index in by_layer
        }
        use_global_rule = bool(selected_modules) and all(
            module is not None and not hasattr(module, "chunk_gated_delta_rule")
            for module in selected_modules.values()
        )
        if use_global_rule:
            modeling_modules = {
                inspect.getmodule(type(module)) for module in selected_modules.values()
            }
            if len(modeling_modules) != 1 or None in modeling_modules:
                raise RuntimeError("Could not identify one shared Qwen GLA modeling module")
            self.modeling_module = modeling_modules.pop()
            if not hasattr(self.modeling_module, "torch_chunk_gated_delta_rule"):
                raise RuntimeError("Qwen modeling module has no chunk GLA rule")
            self.original_global_rule = self.modeling_module.torch_chunk_gated_delta_rule

            for layer_index, module in selected_modules.items():
                def mark_active(_module: Any, _args: Any, _layer=layer_index) -> None:
                    if self.active_layer is not None:
                        raise RuntimeError("Nested Qwen GLA forward calls are unsupported")
                    self.active_layer = _layer

                def clear_active(_module: Any, _args: Any, _output: Any) -> None:
                    self.active_layer = None

                self.hook_handles.append(module.register_forward_pre_hook(mark_active))
                self.hook_handles.append(module.register_forward_hook(clear_active))

            original_global_rule = self.original_global_rule

            def wrapped_global_rule(
                query: Any,
                key: Any,
                value: Any,
                *args: Any,
                **kwargs: Any,
            ):
                layer_index = self.active_layer
                if layer_index is None or layer_index not in by_layer:
                    return original_global_rule(query, key, value, *args, **kwargs)
                if "g" not in kwargs or "beta" not in kwargs:
                    raise RuntimeError("Qwen GLA rule did not pass g and beta by keyword")
                self.rule_calls += 1
                self.layers_seen.add(layer_index)
                patched_key = key.clone()
                patched_value = value.clone()
                patched_g = kwargs["g"].clone()
                patched_beta = kwargs["beta"].clone()
                for target_row, groups in by_layer[layer_index].items():
                    for source_row, positions in groups:
                        selected = list(positions)
                        self.edited_position_count += len(positions)
                        patched_key[target_row, selected] = key[source_row, selected]
                        patched_value[target_row, selected] = value[source_row, selected]
                        patched_g[target_row, selected] = kwargs["g"][source_row, selected]
                        patched_beta[target_row, selected] = kwargs["beta"][source_row, selected]
                patched_kwargs = dict(kwargs)
                patched_kwargs["g"] = patched_g
                patched_kwargs["beta"] = patched_beta
                edited = original_global_rule(
                    query, patched_key, patched_value, *args, **patched_kwargs
                )
                if not self.preserve_source_output:
                    return edited
                natural = original_global_rule(query, key, value, *args, **kwargs)
                cutoffs = {
                    target: max(position for _source, positions in groups for position in positions)
                    for target, groups in by_layer[layer_index].items()
                }
                return _replace_only_causally_later_gla_outputs(
                    natural, edited, cutoffs
                )

            self.modeling_module.torch_chunk_gated_delta_rule = wrapped_global_rule
            return

        for layer_index, row_specs in by_layer.items():
            if layer_index < 0 or layer_index >= len(parts.layers):
                self.close()
                raise ValueError(f"Invalid model layer: {layer_index}")
            module = getattr(parts.layers[layer_index], "linear_attn", None)
            if module is None:
                self.close()
                raise ValueError(f"Layer {layer_index} is not Gated DeltaNet")
            original = module.chunk_gated_delta_rule

            def wrapped(
                query: Any,
                key: Any,
                value: Any,
                *args: Any,
                _original=original,
                _row_specs=row_specs,
                _layer_index=layer_index,
                **kwargs: Any,
            ):
                if "g" not in kwargs or "beta" not in kwargs:
                    raise RuntimeError("Qwen Gated DeltaNet did not pass g and beta by keyword")
                self.rule_calls += 1
                self.layers_seen.add(_layer_index)
                patched_key = key.clone()
                patched_value = value.clone()
                patched_g = kwargs["g"].clone()
                patched_beta = kwargs["beta"].clone()
                for target_row, groups in _row_specs.items():
                    for source_row, positions in groups:
                        selected = list(positions)
                        self.edited_position_count += len(positions)
                        patched_key[target_row, selected] = key[source_row, selected]
                        patched_value[target_row, selected] = value[source_row, selected]
                        patched_g[target_row, selected] = kwargs["g"][source_row, selected]
                        patched_beta[target_row, selected] = kwargs["beta"][source_row, selected]
                patched_kwargs = dict(kwargs)
                patched_kwargs["g"] = patched_g
                patched_kwargs["beta"] = patched_beta
                edited = _original(
                    query, patched_key, patched_value, *args, **patched_kwargs
                )
                if not self.preserve_source_output:
                    return edited
                natural = _original(query, key, value, *args, **kwargs)
                cutoffs = {
                    target: max(position for _source, positions in groups for position in positions)
                    for target, groups in _row_specs.items()
                }
                return _replace_only_causally_later_gla_outputs(
                    natural, edited, cutoffs
                )

            self.originals.append((module, original))
            module.chunk_gated_delta_rule = wrapped

    def assert_fired(self) -> None:
        """Fail closed when no selected recurrent-memory write was patched."""

        if self.rule_calls <= 0 or self.edited_position_count <= 0:
            raise RuntimeError(
                "Selective GLA source-write patcher did not edit any write "
                f"(calls={self.rule_calls}, positions={self.edited_position_count})"
            )
        missing = self.selected_layers - self.layers_seen
        if missing:
            raise RuntimeError(
                "Selective GLA source-write patcher did not observe selected layers: "
                f"{sorted(missing)}"
            )

    def close(self) -> None:
        for handle in reversed(getattr(self, "hook_handles", [])):
            handle.remove()
        self.hook_handles.clear()
        if self.modeling_module is not None and self.original_global_rule is not None:
            self.modeling_module.torch_chunk_gated_delta_rule = self.original_global_rule
            self.original_global_rule = None
            self.modeling_module = None
            self.active_layer = None
        for module, original in reversed(self.originals):
            module.chunk_gated_delta_rule = original
        self.originals.clear()
