import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from mechanistic.head_patching import (
    BatchedScenarioHeadContextPatcher,
    BatchedSingleHeadContextPatcher,
    FinalHeadContextPatcher,
    HeadTarget,
)
from mechanistic.prepare_onset_circuit import (
    prepare_component_plan,
    prepare_head_discovery_plan,
)
from mechanistic.run_attention_source_ablation import source_positions


class OnsetCircuitPlanTests(unittest.TestCase):
    def test_component_plan_has_joint_leaveout_and_reciprocal_scenarios(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "confirmation.json"
            source.write_text(json.dumps({"question_ids": ["q1", "q2"]}))
            output = root / "component.json"
            plan = prepare_component_plan(source, output)
            self.assertEqual(len(plan["targets"]), 4)
            self.assertEqual(len(plan["groups"]), 7)
            self.assertEqual(len(plan["scenarios"]), 14)
            all4 = plan["groups"]["onset_all4"]
            self.assertEqual([row["component"] for row in all4], [
                "mixer_l47", "mlp_l49", "mixer_l50", "mixer_l51"
            ])
            self.assertTrue(any(row["id"] == "game_into_neutral__onset_all4" for row in plan["scenarios"]))

    def test_head_discovery_sweeps_all_heads_at_both_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "discovery.json"
            source.write_text(json.dumps({"question_ids": ["q1"]}))
            output = root / "heads.json"
            plan = prepare_head_discovery_plan(source, output)
            self.assertEqual(len(plan["targets"]), 48)
            self.assertEqual(len(plan["scenarios"]), 48)
            self.assertEqual({row["layer"] for row in plan["targets"]}, {47, 51})
            self.assertEqual(
                {row["heads"][0] for row in plan["targets"] if row["layer"] == 47},
                set(range(24)),
            )

    def test_source_selectors_use_exact_semantic_spans(self):
        spans = {
            "condition_keyword": [3],
            "action_keyword": [5, 6],
            "feedback_sentence": [2, 3, 4, 5, 6],
            "system_condition": [0, 1],
            "first_question": [7, 8],
            "repeated_question": [9, 10],
            "redacted_answer": [11],
            "previous_8": [12, 13],
        }
        self.assertEqual(source_positions(spans, "feedback_sentence"), [2, 3, 4, 5, 6])
        self.assertEqual(source_positions(spans, "local_answer_cue"), [12, 13])
        with self.assertRaises(ValueError):
            source_positions(spans, "unknown")

    def test_head_patcher_replaces_only_selected_head_at_final_position(self):
        projection = torch.nn.Linear(4, 4, bias=False)
        projection.weight.data.copy_(torch.eye(4))
        attention = SimpleNamespace(o_proj=projection, num_heads=2)
        parts = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
        hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        source = {0: torch.tensor([[100.0, 101.0, 102.0, 103.0]])}
        patcher = FinalHeadContextPatcher(
            parts, [HeadTarget(0, (1,))], source, last_indices=[2]
        )
        try:
            result = projection(hidden)
        finally:
            patcher.close()
        torch.testing.assert_close(result[0, :2], hidden[0, :2])
        torch.testing.assert_close(result[0, 2, :2], hidden[0, 2, :2])
        torch.testing.assert_close(result[0, 2, 2:], source[0][0, 2:])

    def test_batched_head_patcher_uses_one_head_per_row(self):
        projection = torch.nn.Linear(4, 4, bias=False)
        projection.weight.data.copy_(torch.eye(4))
        parts = SimpleNamespace(layers=[SimpleNamespace(
            self_attn=SimpleNamespace(o_proj=projection, num_heads=2)
        )])
        hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        source = torch.tensor([[100.0, 101.0, 102.0, 103.0]])
        patcher = BatchedSingleHeadContextPatcher(
            parts, 0, heads_by_row=[0, 1], source=source, last_indices=[2, 2]
        )
        try:
            result = projection(hidden)
        finally:
            patcher.close()
        torch.testing.assert_close(result[0, 2, :2], source[0, :2])
        torch.testing.assert_close(result[0, 2, 2:], hidden[0, 2, 2:])
        torch.testing.assert_close(result[1, 2, :2], hidden[1, 2, :2])
        torch.testing.assert_close(result[1, 2, 2:], source[0, 2:])

    def test_batched_scenario_patcher_supports_multiple_layers(self):
        projections = [torch.nn.Linear(4, 4, bias=False) for _ in range(2)]
        for projection in projections:
            projection.weight.data.copy_(torch.eye(4))
        parts = SimpleNamespace(layers=[SimpleNamespace(
            self_attn=SimpleNamespace(o_proj=projection, num_heads=2)
        ) for projection in projections])
        source = {
            0: torch.tensor([[100.0, 101.0, 102.0, 103.0]]),
            1: torch.tensor([[200.0, 201.0, 202.0, 203.0]]),
        }
        scenarios = [
            [HeadTarget(0, (1,)), HeadTarget(1, (0,))],
            [HeadTarget(1, (1,))],
        ]
        patcher = BatchedScenarioHeadContextPatcher(parts, scenarios, source, [2, 1])
        hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        try:
            first = projections[0](hidden)
            second = projections[1](hidden)
        finally:
            patcher.close()
        torch.testing.assert_close(first[0, 2, 2:], source[0][0, 2:])
        torch.testing.assert_close(first[1], hidden[1])
        torch.testing.assert_close(second[0, 2, :2], source[1][0, :2])
        torch.testing.assert_close(second[1, 1, 2:], source[1][0, 2:])


if __name__ == "__main__":
    unittest.main()
