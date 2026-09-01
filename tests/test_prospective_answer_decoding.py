import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mechanistic.analyze_prospective_answer_decoding import analyze_dataset


class ProspectiveAnswerDecoderTests(unittest.TestCase):
    def test_shared_basis_and_condition_offsets_transfer(self):
        rng = np.random.default_rng(7)
        n = 48
        width = 12
        qids = [f"q{index:03d}" for index in range(n)]
        states = rng.normal(size=(2, n, 64, width)).astype(np.float16)
        normalized = states[:, :, 0].astype(np.float32)
        normalized /= np.sqrt(np.mean(normalized * normalized, axis=-1, keepdims=True))
        basis = rng.normal(size=(width, 4)).astype(np.float32)
        condition_offsets = np.asarray(
            [[0.3, -0.1, -0.1, -0.1], [-0.2, 0.2, 0.0, 0.0]], dtype=np.float32
        )
        targets = np.stack(
            [normalized[ci] @ basis + condition_offsets[ci] for ci in range(2)], axis=0
        )
        targets -= targets.mean(axis=-1, keepdims=True)
        rank_order = np.tile(np.arange(4, dtype=np.int64), (n, 1))
        discovery_qids = qids[:32]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "decision_residuals.npy", states)
            np.savez_compressed(
                root / "results.npz",
                question_ids=np.asarray(qids),
                conditions=np.asarray(["game", "neutral"]),
                jlens_scores=np.zeros((2, n, 64, 4), dtype=np.float32),
                direct_logits=targets,
                rank_order=rank_order,
            )
            (root / "split.json").write_text(
                json.dumps({"discovery_question_ids": discovery_qids})
            )
            spec = {
                "name": "synthetic",
                "results": str(root / "results.npz"),
                "residuals": str(root / "decision_residuals.npy"),
                "discovery_plan": str(root / "split.json"),
                "output": str(root),
                "analysis_output": str(root / "analysis"),
                "seed": 3,
            }
            analyze_dataset(spec, max_layers=1)
            payload = np.load(root / "analysis" / "predictions.npz")
            prediction = payload["predictions"].astype(np.float32)
            confirmation = ~payload["discovery"]
            exact = payload["exact_final_scores"]

            # Shared and both cross-condition coefficient bases recover the same mapping.
            for decoder_index in range(3):
                for condition_index in range(2):
                    pred = prediction[decoder_index, condition_index, confirmation, 0]
                    target = exact[condition_index, confirmation]
                    correlation = np.corrcoef(pred.reshape(-1), target.reshape(-1))[0, 1]
                    self.assertGreater(correlation, 0.98)


if __name__ == "__main__":
    unittest.main()
