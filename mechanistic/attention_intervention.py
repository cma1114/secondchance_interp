from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EdgeTarget:
    layer: int
    heads: tuple[int, ...]


def _ablate_scores(scores: Any, heads: tuple[int, ...], query_positions: tuple[int, ...], key_positions: tuple[int, ...]) -> Any:
    """Set selected attention logits to -inf before softmax.

    `scores` has shape (batch, head, query, key). The operation is deliberately
    narrow: all batches share the same final-query and source-token positions,
    which is sufficient for the batch-size-one causal experiment.
    """
    import torch

    if not heads or not query_positions or not key_positions:
        raise ValueError("An edge ablation needs at least one head, query, and key position")
    result = scores.clone()
    head_index = torch.as_tensor(heads, device=result.device)
    query_index = torch.as_tensor(query_positions, device=result.device)
    key_index = torch.as_tensor(key_positions, device=result.device)
    result[:, head_index[:, None, None], query_index[None, :, None], key_index[None, None, :]] = -torch.inf
    return result


def _ablate_scores_batched(
    scores: Any,
    specs: dict[int, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
) -> Any:
    """Apply row-specific head/query/key masks to batched attention logits."""
    import torch

    result = scores.clone()
    for row, (heads, queries, keys) in specs.items():
        key_index = torch.as_tensor(keys, device=result.device)
        for head in heads:
            for query_position in queries:
                result[row, head, query_position, key_index] = -torch.inf
    return result


class AttentionEdgeAblator:
    """Temporarily remove particular Qwen eager-attention edges.

    The installed Qwen implementation dispatches eager attention through the
    module-global `eager_attention_forward`. We temporarily replace that
    function and mark only the requested self-attention modules. The wrapper
    duplicates the installed eager calculation but inserts the edge mask after
    the causal mask and before softmax, so the remaining attention is exactly
    renormalized.
    """

    _attribute = "_secondchance_edge_ablation"

    def __init__(
        self,
        parts: Any,
        targets: list[EdgeTarget],
        query_positions: list[int],
        key_positions: list[int],
    ) -> None:
        if not targets:
            raise ValueError("No attention heads were selected")
        self.modules = []
        self.modeling_module = None
        self.original = None
        query = tuple(int(value) for value in query_positions)
        keys = tuple(int(value) for value in key_positions)
        for target in targets:
            attention = getattr(parts.layers[target.layer], "self_attn", None)
            if attention is None:
                raise ValueError(f"Layer {target.layer} is not a conventional attention block")
            n_heads = getattr(attention, "num_heads", None)
            if n_heads is None:
                # Qwen3.5/3.6 projects both query and per-head gate values and
                # therefore has 2 * n_heads * head_dim q_proj outputs.
                q_projection = getattr(attention, "q_proj", None)
                head_dim = getattr(attention, "head_dim", None)
                if q_projection is None or head_dim is None:
                    raise RuntimeError(f"Cannot infer the head count at layer {target.layer}")
                n_heads = int(q_projection.out_features // (2 * head_dim))
            n_heads = int(n_heads)
            if any(head < 0 or head >= n_heads for head in target.heads):
                raise ValueError(f"Invalid head for layer {target.layer}; model has {n_heads} heads")
            module = inspect.getmodule(type(attention))
            if module is None or not hasattr(module, "eager_attention_forward"):
                raise RuntimeError("Could not locate the model's eager attention function")
            if self.modeling_module is not None and module is not self.modeling_module:
                raise RuntimeError("Selected attention modules use different modeling implementations")
            self.modeling_module = module
            setattr(attention, self._attribute, (tuple(target.heads), query, keys))
            self.modules.append(attention)

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

            spec = getattr(module, self._attribute, None)
            if spec is None:
                return original(
                    module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
                )
            key_states = repeat_kv(key, module.num_key_value_groups)
            value_states = repeat_kv(value, module.num_key_value_groups)
            weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                weights = weights + attention_mask
            weights = _ablate_scores(weights, *spec)
            weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
            weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
            output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
            return output, weights

        self.modeling_module.eager_attention_forward = intervened_eager_attention_forward

    def close(self) -> None:
        for module in self.modules:
            if hasattr(module, self._attribute):
                delattr(module, self._attribute)
        if self.modeling_module is not None and self.original is not None:
            self.modeling_module.eager_attention_forward = self.original

    def __enter__(self) -> "AttentionEdgeAblator":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class BatchedAttentionEdgeAblator:
    """Apply a different final-query edge ablation to every batch row."""

    _attribute = "_secondchance_batched_edge_ablation"

    def __init__(
        self,
        parts: Any,
        targets_by_row: list[list[EdgeTarget]],
        query_positions: list[int],
        key_positions_by_row: list[list[int]],
    ) -> None:
        if not (len(targets_by_row) == len(query_positions) == len(key_positions_by_row)):
            raise ValueError("Targets, queries, and key spans must have one entry per batch row")
        by_layer: dict[int, dict[int, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]] = {}
        for row, targets in enumerate(targets_by_row):
            if not key_positions_by_row[row]:
                raise ValueError(f"Batch row {row} has no source positions")
            for target in targets:
                by_layer.setdefault(target.layer, {})[row] = (
                    target.heads,
                    (int(query_positions[row]),),
                    tuple(int(value) for value in key_positions_by_row[row]),
                )
        if not by_layer:
            raise ValueError("No attention heads were selected")

        self.modules = []
        self.modeling_module = None
        self.original = None
        for layer, specs in by_layer.items():
            attention = getattr(parts.layers[layer], "self_attn", None)
            if attention is None:
                raise ValueError(f"Layer {layer} is not a conventional attention block")
            n_heads = getattr(attention, "num_heads", None)
            if n_heads is None:
                n_heads = int(attention.q_proj.out_features // (2 * attention.head_dim))
            for heads, _queries, _keys in specs.values():
                if any(head < 0 or head >= int(n_heads) for head in heads):
                    raise ValueError(f"Invalid head for layer {layer}; model has {n_heads} heads")
            module = inspect.getmodule(type(attention))
            if module is None or not hasattr(module, "eager_attention_forward"):
                raise RuntimeError("Could not locate the model's eager attention function")
            if self.modeling_module is not None and module is not self.modeling_module:
                raise RuntimeError("Selected attention modules use different modeling implementations")
            self.modeling_module = module
            setattr(attention, self._attribute, specs)
            self.modules.append(attention)

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
                    module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
                )
            key_states = repeat_kv(key, module.num_key_value_groups)
            value_states = repeat_kv(value, module.num_key_value_groups)
            weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                weights = weights + attention_mask
            weights = _ablate_scores_batched(weights, specs)
            weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
            weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
            output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
            return output, weights

        self.modeling_module.eager_attention_forward = intervened_eager_attention_forward

    def close(self) -> None:
        for module in self.modules:
            if hasattr(module, self._attribute):
                delattr(module, self._attribute)
        if self.modeling_module is not None and self.original is not None:
            self.modeling_module.eager_attention_forward = self.original
