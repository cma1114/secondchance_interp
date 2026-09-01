from __future__ import annotations

from types import SimpleNamespace

import torch

from mechanistic.downstream_source_intervention import (
    BatchedSDPADownstreamAttentionAblator,
    BatchedSDPAFinalQueryAttentionAblator,
    BatchedSDPAQuerySourceAttentionAblator,
)


class _Attention(torch.nn.Module):
    def forward(self, query, key, value, mask):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0
        )


class _ImplicitCausalAttention(torch.nn.Module):
    def forward(self, query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
        )


def test_final_query_ablation_changes_only_selected_final_edge():
    attention = _Attention()
    parts = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=attention)]
    )
    query = torch.zeros(2, 1, 3, 1)
    key = torch.zeros_like(query)
    value = torch.tensor(
        [[[[1.0], [3.0], [8.0]]], [[[2.0], [4.0], [9.0]]]]
    )
    mask = torch.full((2, 1, 3, 3), -torch.inf)
    for position in range(3):
        mask[:, :, position, : position + 1] = 0.0

    natural = attention(query, key, value, mask)
    with BatchedSDPAFinalQueryAttentionAblator(
        parts, {0: {0: [0], 1: [1]}}
    ):
        ablated = attention(query, key, value, mask)

    # Earlier queries are bit-identical.
    torch.testing.assert_close(ablated[:, :, :2], natural[:, :, :2], rtol=0, atol=0)
    # Row 0 loses value 1 from its final-query average: (3+8)/2.
    torch.testing.assert_close(ablated[0, 0, 2, 0], torch.tensor(5.5))
    # Row 1 loses value 4: (2+9)/2.
    torch.testing.assert_close(ablated[1, 0, 2, 0], torch.tensor(5.5))


def test_implicit_causal_path_preserves_earlier_queries_and_changes_only_edge():
    attention = _ImplicitCausalAttention()
    parts = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    query = torch.zeros(2, 1, 3, 1)
    key = torch.zeros_like(query)
    value = torch.tensor(
        [[[[1.0], [3.0], [8.0]]], [[[2.0], [4.0], [9.0]]]]
    )
    natural = attention(query, key, value)
    with BatchedSDPAFinalQueryAttentionAblator(
        parts, {0: {0: [0], 1: [1]}}
    ):
        ablated = attention(query, key, value)

    torch.testing.assert_close(ablated[:, :, :2], natural[:, :, :2], rtol=0, atol=0)
    torch.testing.assert_close(ablated[0, 0, 2, 0], torch.tensor(5.5))
    torch.testing.assert_close(ablated[1, 0, 2, 0], torch.tensor(5.5))


def test_boolean_padding_mask_blocks_selected_final_edge():
    attention = _Attention()
    parts = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    query = torch.zeros(2, 1, 3, 1)
    key = torch.zeros_like(query)
    value = torch.tensor(
        [[[[1.0], [3.0], [8.0]]], [[[2.0], [4.0], [9.0]]]]
    )
    mask = torch.zeros((2, 1, 3, 3), dtype=torch.bool)
    for position in range(3):
        mask[:, :, position, : position + 1] = True

    natural = attention(query, key, value, mask)
    with BatchedSDPAFinalQueryAttentionAblator(
        parts, {0: {0: [0], 1: [1]}}
    ):
        ablated = attention(query, key, value, mask)

    torch.testing.assert_close(ablated[:, :, :2], natural[:, :, :2], rtol=0, atol=0)
    torch.testing.assert_close(ablated[0, 0, 2, 0], torch.tensor(5.5))
    torch.testing.assert_close(ablated[1, 0, 2, 0], torch.tensor(5.5))


def _query_source_specs():
    return {0: {0: {1: [0]}, 1: {2: [1]}}}


def _assert_arbitrary_query_result(natural, ablated):
    # Row 0 changes only query 1 by removing source value 1: it becomes 3.
    torch.testing.assert_close(ablated[0, 0, 0], natural[0, 0, 0], rtol=0, atol=0)
    torch.testing.assert_close(ablated[0, 0, 1, 0], torch.tensor(3.0))
    torch.testing.assert_close(ablated[0, 0, 2], natural[0, 0, 2], rtol=0, atol=0)
    # Row 1 changes only query 2 by removing source value 4: (2+9)/2.
    torch.testing.assert_close(ablated[1, 0, :2], natural[1, 0, :2], rtol=0, atol=0)
    torch.testing.assert_close(ablated[1, 0, 2, 0], torch.tensor(5.5))


def test_arbitrary_query_source_ablation_with_additive_mask():
    attention = _Attention()
    parts = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    query = torch.zeros(2, 1, 3, 1)
    key = torch.zeros_like(query)
    value = torch.tensor(
        [[[[1.0], [3.0], [8.0]]], [[[2.0], [4.0], [9.0]]]]
    )
    mask = torch.full((2, 1, 3, 3), -torch.inf)
    for position in range(3):
        mask[:, :, position, : position + 1] = 0.0
    natural = attention(query, key, value, mask)
    with BatchedSDPAQuerySourceAttentionAblator(parts, _query_source_specs()):
        ablated = attention(query, key, value, mask)
    _assert_arbitrary_query_result(natural, ablated)


def test_arbitrary_query_source_ablation_with_boolean_mask():
    attention = _Attention()
    parts = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    query = torch.zeros(2, 1, 3, 1)
    key = torch.zeros_like(query)
    value = torch.tensor(
        [[[[1.0], [3.0], [8.0]]], [[[2.0], [4.0], [9.0]]]]
    )
    mask = torch.zeros((2, 1, 3, 3), dtype=torch.bool)
    for position in range(3):
        mask[:, :, position, : position + 1] = True
    natural = attention(query, key, value, mask)
    with BatchedSDPAQuerySourceAttentionAblator(parts, _query_source_specs()):
        ablated = attention(query, key, value, mask)
    _assert_arbitrary_query_result(natural, ablated)


def test_arbitrary_query_source_ablation_with_implicit_causal_mask():
    attention = _ImplicitCausalAttention()
    parts = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    query = torch.zeros(2, 1, 3, 1)
    key = torch.zeros_like(query)
    value = torch.tensor(
        [[[[1.0], [3.0], [8.0]]], [[[2.0], [4.0], [9.0]]]]
    )
    natural = attention(query, key, value)
    with BatchedSDPAQuerySourceAttentionAblator(parts, _query_source_specs()):
        ablated = attention(query, key, value)
    _assert_arbitrary_query_result(natural, ablated)


def _assert_downstream_source_result(natural, ablated):
    # Row 0 source 0 remains available to its own query, then is removed from
    # all later queries: q1 becomes 3 and q2 becomes (3+8)/2.
    torch.testing.assert_close(ablated[0, 0, 0], natural[0, 0, 0], rtol=0, atol=0)
    torch.testing.assert_close(ablated[0, 0, 1, 0], torch.tensor(3.0))
    torch.testing.assert_close(ablated[0, 0, 2, 0], torch.tensor(5.5))
    # Row 1 source 1 affects q0/q1 naturally and is removed only from q2.
    torch.testing.assert_close(ablated[1, 0, :2], natural[1, 0, :2], rtol=0, atol=0)
    torch.testing.assert_close(ablated[1, 0, 2, 0], torch.tensor(5.5))


def test_downstream_source_ablation_with_additive_mask():
    attention = _Attention()
    parts = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    query = torch.zeros(2, 1, 3, 1)
    key = torch.zeros_like(query)
    value = torch.tensor(
        [[[[1.0], [3.0], [8.0]]], [[[2.0], [4.0], [9.0]]]]
    )
    mask = torch.full((2, 1, 3, 3), -torch.inf)
    for position in range(3):
        mask[:, :, position, : position + 1] = 0.0
    natural = attention(query, key, value, mask)
    with BatchedSDPADownstreamAttentionAblator(parts, {0: [0], 1: [1]}) as ablator:
        ablated = attention(query, key, value, mask)
        ablator.assert_fired()
    _assert_downstream_source_result(natural, ablated)


def test_downstream_source_ablation_with_boolean_mask():
    attention = _Attention()
    parts = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    query = torch.zeros(2, 1, 3, 1)
    key = torch.zeros_like(query)
    value = torch.tensor(
        [[[[1.0], [3.0], [8.0]]], [[[2.0], [4.0], [9.0]]]]
    )
    mask = torch.zeros((2, 1, 3, 3), dtype=torch.bool)
    for position in range(3):
        mask[:, :, position, : position + 1] = True
    natural = attention(query, key, value, mask)
    with BatchedSDPADownstreamAttentionAblator(parts, {0: [0], 1: [1]}) as ablator:
        ablated = attention(query, key, value, mask)
        ablator.assert_fired()
    _assert_downstream_source_result(natural, ablated)
