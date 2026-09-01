from __future__ import annotations

import types
import unittest

import torch

from mechanistic.sublayer import (
    BatchedPositionComponentOutputPatcher,
    BatchedRowSourcePositionComponentOutputPatcher,
    PositionComponentOutputCollector,
    PositionComponentTarget,
)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.Identity()
        self.mlp = torch.nn.Identity()


class PositionComponentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layer = _Layer()
        self.parts = types.SimpleNamespace(layers=[self.layer])

    def test_collects_multiple_semantic_positions(self) -> None:
        targets = [
            PositionComponentTarget(0, "mixer", "feedback_end"),
            PositionComponentTarget(0, "mlp", "second_user_end"),
        ]
        collector = PositionComponentOutputCollector(
            self.parts,
            targets,
            {"feedback_end": 1, "second_user_end": 2},
        )
        values = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        try:
            self.layer.self_attn(values)
            self.layer.mlp(values)
        finally:
            collector.close()
        self.assertTrue(
            torch.equal(
                collector.values["feedback_end__mixer_l0"].float(),
                values[:, 1],
            )
        )
        self.assertTrue(
            torch.equal(
                collector.values["second_user_end__mlp_l0"].float(),
                values[:, 2],
            )
        )

    def test_patches_different_components_and_positions_by_batch_row(self) -> None:
        feedback = PositionComponentTarget(0, "mixer", "feedback_end")
        second = PositionComponentTarget(0, "mlp", "second_user_end")
        source = {
            feedback.source_key: torch.full((1, 4), 101.0),
            second.source_key: torch.full((1, 4), 202.0),
        }
        patcher = BatchedPositionComponentOutputPatcher(
            self.parts,
            [[feedback], [second]],
            source,
            {"feedback_end": 1, "second_user_end": 2},
        )
        values = torch.zeros((2, 3, 4))
        try:
            after_mixer = self.layer.self_attn(values)
            after_mlp = self.layer.mlp(after_mixer)
        finally:
            patcher.close()
        self.assertTrue(torch.equal(after_mlp[0, 1], torch.full((4,), 101.0)))
        self.assertTrue(torch.equal(after_mlp[1, 2], torch.full((4,), 202.0)))
        self.assertEqual(float(after_mlp[0, 2].sum()), 0.0)
        self.assertEqual(float(after_mlp[1, 1].sum()), 0.0)

    def test_patches_same_targets_from_row_specific_sources(self) -> None:
        feedback = PositionComponentTarget(0, "mixer", "feedback_end")
        sources = [
            {feedback.source_key: torch.full((1, 4), 101.0)},
            {feedback.source_key: torch.full((1, 4), 202.0)},
            {},
        ]
        patcher = BatchedRowSourcePositionComponentOutputPatcher(
            self.parts,
            [[feedback], [feedback], []],
            sources,
            {"feedback_end": 1},
        )
        values = torch.zeros((3, 3, 4))
        try:
            result = self.layer.self_attn(values)
        finally:
            patcher.close()
        self.assertTrue(torch.equal(result[0, 1], torch.full((4,), 101.0)))
        self.assertTrue(torch.equal(result[1, 1], torch.full((4,), 202.0)))
        self.assertEqual(float(result[2].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
