import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mechanistic.analyze_component_causal_sweep import analyze
from mechanistic.io import atomic_save_npz, json_array, shard_path
from mechanistic.select_component_causal_candidates import select
from mechanistic.prepare_component_transfer import prepare as prepare_transfer


class CandidateSelectionTests(unittest.TestCase):
    def test_separates_compression_and_switching_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = [
                {"component": "mixer_l0", "kind": "mixer", "layer": 0},
                {"component": "mlp_l0", "kind": "mlp", "layer": 0},
            ]
            discovery = root / "discovery.json"
            discovery.write_text(json.dumps({"targets": targets}))
            split = root / "split.json"
            split.write_text(json.dumps({"confirmation_question_ids": ["q1", "q2"]}))
            effects = root / "effects.csv"
            rows = []
            values = {
                "mixer_l0": {
                    "ad_entropy": (-0.2, 0.8), "ad_spread": (0.3, 0.7),
                    "switch": (0.01, -0.1), "winner_advantage": (-0.01, -0.1),
                },
                "mlp_l0": {
                    "ad_entropy": (0.01, -0.1), "ad_spread": (-0.01, -0.1),
                    "switch": (-0.2, 0.9), "winner_advantage": (0.4, 0.8),
                },
            }
            for component, metrics in values.items():
                for metric, (effect, fraction) in metrics.items():
                    rows.append({
                        "component": component, "metric": metric,
                        "aggregation": "dataset", "direction": "neutral_into_game",
                        "effect_mean": effect, "fraction_gap_mediated": fraction,
                    })
            with effects.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            output = root / "confirmation.json"
            plan = select(effects, discovery, split, output, max_candidates=2)
            self.assertEqual(plan["selected_compression"], ["mixer_l0"])
            self.assertEqual(plan["selected_switching"], ["mlp_l0"])
            self.assertEqual(len(plan["targets"]), 2)
            self.assertEqual(len(plan["scenarios"]), 10)


class TransferPlanTests(unittest.TestCase):
    def test_freezes_source_targets_and_uses_manifest_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = {"component": "mixer_l2", "kind": "mixer", "layer": 2}
            source = root / "source.json"
            source.write_text(json.dumps({
                "targets": [target],
                "scenarios": [{
                    "id": "neutral_into_game__mixer_l2",
                    "source_condition": "neutral",
                    "target_condition": "incorrect",
                    "targets": [target],
                }],
                "selected_compression": ["mixer_l2"],
                "selected_switching": [],
            }))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "dataset": "TriviaMC",
                "questions": [{"id": "q2"}, {"id": "q1"}],
            }))
            output = root / "transfer.json"
            plan = prepare_transfer(source, manifest, output)
            self.assertEqual(plan["question_ids"], ["q2", "q1"])
            self.assertEqual(plan["targets"], [target])
            self.assertEqual(plan["scenarios"], json.loads(source.read_text())["scenarios"])
            self.assertTrue(plan["collect_baseline"])
            self.assertIn("No TriviaMC outcome", plan["selection_data_policy"])


class SweepAnalysisTests(unittest.TestCase):
    def test_end_to_end_analysis_writes_geometry_and_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            natural_root = root / "natural"
            patch_root = root / "patch"
            qids = [f"q{letter}" for letter in "ABCD"]
            scenario = "neutral_into_game__mixer_l0"
            for index, qid in enumerate(qids):
                baseline = np.full(4, -1.0)
                baseline[index] = 3.0
                neutral = baseline.copy()
                game = 0.5 * baseline
                patched = neutral.copy()
                atomic_save_npz(
                    shard_path(natural_root, "baseline", qid),
                    final_canonical_logits=baseline,
                    metadata=json_array({"question_id": qid, "correct_answer": "ABCD"[index]}),
                )
                for group, values in (("natural_game", game), ("natural_neutral", neutral), (scenario, patched)):
                    atomic_save_npz(
                        shard_path(patch_root, group, qid),
                        final_canonical_logits=values,
                        metadata=json_array({"question_id": qid}),
                    )
            plan_path = root / "plan.json"
            target = {"component": "mixer_l0", "kind": "mixer", "layer": 0}
            plan_path.write_text(json.dumps({
                "question_ids": qids,
                "targets": [target],
                "scenarios": [{
                    "id": scenario, "source_condition": "neutral",
                    "target_condition": "incorrect", "targets": [target],
                }],
            }))
            output = root / "analysis"
            summary = analyze(natural_root, patch_root, plan_path, output, samples=100, seed=7)
            self.assertTrue(summary["complete"])
            self.assertTrue((output / "component_causal_effects.csv").exists())
            self.assertTrue((output / "causal_outcome_sweep.png").exists())
            self.assertTrue((output / "COMPONENT_CAUSAL_REPORT.md").exists())
