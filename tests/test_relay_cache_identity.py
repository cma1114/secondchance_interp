from __future__ import annotations

from types import SimpleNamespace

import torch

from mechanistic.relay_interception import (
    BatchedGLACachedRelayDownstreamRestorer,
    BatchedGLACachedRelayRestorer,
    BatchedGLARelayWriteCache,
    BatchedSDPACachedRelayDownstreamRestorer,
    BatchedSDPACachedRelayRestorer,
    BatchedSDPARelayWriteCache,
)


class _Attention(torch.nn.Module):
    def forward(self, query, key, value, mask):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0
        )


def test_ordinary_clean_cache_restoration_is_identity_and_fires():
    attention = _Attention()
    parts = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    query = torch.zeros(2, 1, 3, 1)
    key = torch.zeros_like(query)
    value = torch.tensor([[[[1.0], [3.0], [8.0]]], [[[2.0], [4.0], [9.0]]]])
    mask = torch.zeros((2, 1, 3, 3), dtype=torch.bool)
    for position in range(3):
        mask[:, :, position, : position + 1] = True
    positions = {0: [1], 1: [1]}
    cache = BatchedSDPARelayWriteCache(parts, positions, [0])
    try:
        natural = attention(query, key, value, mask)
    finally:
        cache.close()
    restorer = BatchedSDPACachedRelayRestorer(parts, positions, [0], cache.cache)
    try:
        restored = attention(query, key, value, mask)
        restorer.assert_fired()
    finally:
        restorer.close()
    torch.testing.assert_close(restored, natural, rtol=0, atol=0)


def test_ordinary_downstream_restoration_preserves_local_relay_output():
    attention = _Attention()
    parts = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    query = torch.zeros(1, 1, 3, 1)
    key = torch.zeros_like(query)
    clean_value = torch.tensor([[[[1.0], [3.0], [8.0]]]])
    mask = torch.zeros((1, 1, 3, 3), dtype=torch.bool)
    for position in range(3):
        mask[:, :, position, : position + 1] = True
    positions = {0: [1]}
    cache = BatchedSDPARelayWriteCache(parts, positions, [0])
    try:
        clean = attention(query, key, clean_value, mask)
    finally:
        cache.close()

    perturbed_value = clean_value.clone()
    perturbed_value[0, 0, 1, 0] = 30.0
    perturbed = attention(query, key, perturbed_value, mask)
    restorer = BatchedSDPACachedRelayDownstreamRestorer(
        parts, positions, [0], cache.cache
    )
    try:
        restored = attention(query, key, perturbed_value, mask)
        restorer.assert_fired()
    finally:
        restorer.close()
    torch.testing.assert_close(restored[:, :, 1], perturbed[:, :, 1], rtol=0, atol=0)
    torch.testing.assert_close(restored[:, :, 2], clean[:, :, 2], rtol=0, atol=0)


class _LinearAttention:
    def __init__(self):
        self.chunk_gated_delta_rule = self._rule

    @staticmethod
    def _rule(query, key, value, *, g, beta):
        output = query + key + value + g + beta
        state = output[:, -1].clone()
        return output, state


def test_gla_clean_cache_restoration_is_identity_and_fires():
    module = _LinearAttention()
    parts = SimpleNamespace(layers=[SimpleNamespace(linear_attn=module)])
    query = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    key = query + 10
    value = query + 20
    g = query + 30
    beta = query + 40
    positions = {0: [1], 1: [1]}
    cache = BatchedGLARelayWriteCache(parts, positions, [0])
    try:
        natural = module.chunk_gated_delta_rule(query, key, value, g=g, beta=beta)
    finally:
        cache.close()
    restorer = BatchedGLACachedRelayRestorer(parts, positions, [0], cache.cache)
    try:
        restored = module.chunk_gated_delta_rule(query, key, value, g=g, beta=beta)
        restorer.assert_fired()
    finally:
        restorer.close()
    torch.testing.assert_close(restored[0], natural[0], rtol=0, atol=0)
    torch.testing.assert_close(restored[1], natural[1], rtol=0, atol=0)


class _RecurrentLinearAttention:
    def __init__(self):
        self.chunk_gated_delta_rule = self._rule

    @staticmethod
    def _rule(query, key, value, *, g, beta):
        writes = key + value + g + beta
        output = query + torch.cumsum(writes, dim=1)
        return output, output[:, -1].clone()


def test_gla_downstream_restoration_preserves_local_relay_output():
    module = _RecurrentLinearAttention()
    parts = SimpleNamespace(layers=[SimpleNamespace(linear_attn=module)])
    query = torch.zeros(1, 3, 1)
    clean_key = torch.tensor([[[1.0], [2.0], [3.0]]])
    value = torch.zeros_like(clean_key)
    g = torch.zeros_like(clean_key)
    beta = torch.zeros_like(clean_key)
    positions = {0: [1]}
    cache = BatchedGLARelayWriteCache(parts, positions, [0])
    try:
        clean = module.chunk_gated_delta_rule(query, clean_key, value, g=g, beta=beta)
    finally:
        cache.close()

    perturbed_key = clean_key.clone()
    perturbed_key[0, 1, 0] = 20.0
    perturbed = module.chunk_gated_delta_rule(
        query, perturbed_key, value, g=g, beta=beta
    )
    restorer = BatchedGLACachedRelayDownstreamRestorer(
        parts, positions, [0], cache.cache
    )
    try:
        restored = module.chunk_gated_delta_rule(
            query, perturbed_key, value, g=g, beta=beta
        )
        restorer.assert_fired()
    finally:
        restorer.close()
    torch.testing.assert_close(restored[0][:, 1], perturbed[0][:, 1], rtol=0, atol=0)
    torch.testing.assert_close(restored[0][:, 2], clean[0][:, 2], rtol=0, atol=0)
    torch.testing.assert_close(restored[1], clean[1], rtol=0, atol=0)
