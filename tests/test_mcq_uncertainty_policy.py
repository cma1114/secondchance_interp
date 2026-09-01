from __future__ import annotations

import numpy as np
import pytest

from mechanistic.analyze_mcq_uncertainty_policy import _bootstrap_means, _effects
from mechanistic.fit_mcq_uncertainty_directions import _mean_difference, _metrics
from mechanistic.run_mcq_uncertainty_intervention import (
    SCENARIOS,
    STEERING_MAGNITUDE,
    BatchedDirectionHook,
    _direction_lens,
    _quantized_projection_ablation,
    _quantized_two_coordinate_edit,
    _repeat_cache,
    _random_direction,
)


def test_entropy_metrics_and_mean_difference_sign() -> None:
    logits = np.asarray(
        [
            [10.0, 0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    entropy, gap = _metrics(logits)
    assert entropy[0] < entropy[-1]
    assert gap[0] > gap[-1]

    residuals = np.zeros((4, 4), dtype=np.float32)
    residuals[:, 0] = entropy
    direction = _mean_difference(residuals, entropy)
    assert direction[0] == pytest.approx(1.0)
    assert np.linalg.norm(direction) == pytest.approx(1.0)


def test_random_control_is_unit_and_orthogonal() -> None:
    direction = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    random = _random_direction(direction, 17)
    assert np.linalg.norm(random) == pytest.approx(1.0, abs=1e-6)
    assert float(random @ direction) == pytest.approx(0.0, abs=1e-6)


def test_batched_hook_applies_exact_seven_scenarios() -> None:
    torch = pytest.importorskip("torch")
    layer = torch.nn.Identity()
    direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    random = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    hook = BatchedDirectionHook(layer, direction, random)
    try:
        state = torch.tensor([[[2.0, 4.0, 8.0]]]).repeat_interleave(
            len(SCENARIOS), dim=0
        )
        result = layer(state)[:, 0]
    finally:
        hook.close()

    assert torch.equal(result[0], torch.tensor([2.0, 4.0, 8.0]))
    assert torch.equal(result[1], torch.tensor([0.0, 4.0, 8.0]))
    assert torch.equal(result[2], torch.tensor([2.0 - STEERING_MAGNITUDE, 4.0, 8.0]))
    assert torch.equal(result[3], torch.tensor([2.0 + STEERING_MAGNITUDE, 4.0, 8.0]))
    assert torch.equal(result[4], torch.tensor([2.0, 0.0, 8.0]))
    assert torch.equal(result[5], torch.tensor([2.0, 4.0 - STEERING_MAGNITUDE, 8.0]))
    assert torch.equal(result[6], torch.tensor([2.0, 4.0 + STEERING_MAGNITUDE, 8.0]))
    assert hook.calls == 1
    assert hook.post is not None
    assert float(hook.post[0, 1]) == pytest.approx(0.0, abs=1e-6)
    assert float(hook.post[0, 2] - hook.pre[0, 2]) == pytest.approx(
        -STEERING_MAGNITUDE, abs=1e-6
    )
    assert float(hook.post[0, 3] - hook.pre[0, 3]) == pytest.approx(
        STEERING_MAGNITUDE, abs=1e-6
    )


def test_hybrid_cache_repeat_uses_reorder_path() -> None:
    torch = pytest.importorskip("torch")

    class Layer:
        def __init__(self) -> None:
            self.keys = torch.arange(2, dtype=torch.float32)[:, None]
            self.conv_states = {0: torch.arange(2, dtype=torch.float32)[:, None]}
            self.recurrent_states = {}

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer()]

        def reorder_cache(self, indices):
            for layer in self.layers:
                layer.keys = layer.keys.index_select(0, indices)
                layer.conv_states[0] = layer.conv_states[0].index_select(0, indices)

    source = Cache()
    repeated = _repeat_cache(source, 3)
    assert repeated.layers[0].keys[:, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert repeated.layers[0].conv_states[0][:, 0].tolist() == [
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0
    ]
    assert source.layers[0].keys[:, 0].tolist() == [0.0, 1.0]


def test_analyzer_uses_old_rank_and_cached_identity() -> None:
    ranks = np.asarray([[2, 0, 3, 1]], dtype=np.int64)
    logits = np.zeros((1, len(SCENARIOS), 1, 4), dtype=np.float64)
    logits[0, :, 0] = np.asarray([2.0, 1.0, 4.0, 0.0])
    # Lower the old winner (candidate 2) only in the uncertainty-ablation cell.
    ablation = SCENARIOS.index("uncertainty_ablation")
    logits[0, ablation, 0, 2] = 1.5
    effects = _effects(logits, ranks)
    assert effects["identity:w1_choice"][0, 0] == 0.0
    assert effects["uncertainty_ablation:w1_choice"][0, 0] == -1.0
    assert effects["uncertainty_ablation:w1_minus_w2"][0, 0] == pytest.approx(-2.5)
    # Centering preserves a candidate-specific intervention while removing its
    # common four-logit offset.
    ranked = effects["uncertainty_ablation:rank_centered"][0, 0]
    assert ranked[0] == pytest.approx(-1.875)
    assert ranked[1] == pytest.approx(0.625)


def test_paired_condition_difference_keeps_intervention_interaction() -> None:
    ranks = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    logits = np.zeros((2, 1, len(SCENARIOS), 1, 4), dtype=np.float64)
    logits[:, 0, :, 0] = np.asarray([4.0, 3.0, 2.0, 1.0])
    positive = SCENARIOS.index("uncertainty_steer_positive")
    random_positive = SCENARIOS.index("random_steer_positive")
    logits[0, 0, positive, 0, 0] += 2.0
    logits[0, 0, random_positive, 0, 0] += 0.5
    logits[1, 0, positive, 0, 0] += 1.0
    logits[1, 0, random_positive, 0, 0] += 0.5
    game = _effects(logits[0], ranks)
    neutral = _effects(logits[1], ranks)
    interaction = (
        game["uncertainty_steer_positive:w1_minus_w2"]
        - game["random_steer_positive:w1_minus_w2"]
        - neutral["uncertainty_steer_positive:w1_minus_w2"]
        + neutral["random_steer_positive:w1_minus_w2"]
    )
    assert interaction[0, 0] == pytest.approx(1.0)


def test_chunked_bootstrap_means_match_direct_indexing() -> None:
    values = np.arange(24, dtype=np.float64).reshape(3, 8)
    indices = np.random.default_rng(9).integers(0, 8, size=(17, 8))
    weights = np.linspace(0.5, 1.5, 8)
    direct = values[:, indices].mean(axis=-1)
    assert np.array_equal(_bootstrap_means(values, indices, chunk_size=3), direct)
    direct_weighted = (
        (values[:, indices] * weights[indices]).sum(axis=-1)
        / weights[indices].sum(axis=-1)
    )
    assert np.allclose(
        _bootstrap_means(values, indices, weights, chunk_size=4), direct_weighted
    )


def test_direction_lens_reports_answer_contrast_geometry() -> None:
    torch = pytest.importorskip("torch")

    class Norm(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(width))

        def forward(self, value):
            return value

    class Tokenizer:
        @staticmethod
        def decode(ids):
            return f"t{ids[0]}"

    class Parts:
        final_norm = Norm(4)
        output_head = torch.nn.Linear(4, 8, bias=False)

    with torch.no_grad():
        Parts.output_head.weight.copy_(torch.arange(32).reshape(8, 4) / 10)
    directions = np.zeros((64, 4), dtype=np.float32)
    directions[:, 0] = 1.0
    result = _direction_lens(
        Parts, Tokenizer(), directions, [[0, 4], [1, 5], [2, 6], [3, 7]]
    )
    assert result["_metadata"]["format_version"] == 2
    assert result["_metadata"]["answer_subspace_rank"] <= 3
    assert len(result["64"]["answer_variant_scores"]) == 4
    assert 0.0 <= result["64"]["centered_answer_subspace_fraction"] <= 1.000001


def test_iterated_bfloat16_ablation_improves_live_projection() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(19)
    direction = torch.randn(512, generator=generator)
    direction /= torch.linalg.vector_norm(direction)
    rows = (50 * torch.randn(16, 512, generator=generator)).to(torch.bfloat16)
    quantized_direction = direction.to(torch.bfloat16)
    pre = rows.float() @ direction
    naive = (
        rows.float() - pre[:, None] * quantized_direction.float()[None]
    ).to(torch.bfloat16)
    corrected = _quantized_projection_ablation(
        rows, quantized_direction, direction
    )
    naive_post = torch.abs(naive.float() @ direction)
    corrected_post = torch.abs(corrected.float() @ direction)
    assert torch.all(corrected_post <= naive_post)
    assert float(corrected_post.max() / torch.abs(pre).max()) < 0.005


def test_bfloat16_random_edit_preserves_orthogonal_coordinate() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(23)
    uncertainty = torch.randn(512, generator=generator)
    uncertainty /= torch.linalg.vector_norm(uncertainty)
    random = torch.randn(512, generator=generator)
    random -= (random @ uncertainty) * uncertainty
    random /= torch.linalg.vector_norm(random)
    rows = (50 * torch.randn(16, 512, generator=generator)).to(torch.bfloat16)
    uncertainty_q = uncertainty.to(torch.bfloat16)
    random_q = random.to(torch.bfloat16)
    naive = rows - STEERING_MAGNITUDE * random_q[None]
    corrected = _quantized_two_coordinate_edit(
        rows, random_q, random, uncertainty_q, uncertainty,
        -STEERING_MAGNITUDE,
    )
    naive_leak = torch.abs((naive.float() - rows.float()) @ uncertainty)
    corrected_leak = torch.abs((corrected.float() - rows.float()) @ uncertainty)
    random_dose = (corrected.float() - rows.float()) @ random
    assert torch.all(corrected_leak <= naive_leak)
    assert float(corrected_leak.max()) < (
        STEERING_MAGNITUDE * 0.05
    )
    assert float(torch.mean(random_dose)) == pytest.approx(
        -STEERING_MAGNITUDE, abs=STEERING_MAGNITUDE * 0.03
    )
