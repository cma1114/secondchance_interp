import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mechanistic.answer_emergence_figures import macro_mean_and_se
from mechanistic.attention_spans import SPAN_NAMES, attention_span_indices
from mechanistic.attention_intervention import _ablate_scores, _ablate_scores_batched
from mechanistic.run_attention_ablation import _source_positions
from mechanistic.gdn_intervention import zero_beta_writes
from mechanistic.gdn_tokens import structural_control_positions, user_incorrect_positions
from mechanistic.analyze_steering import _macro_bootstrap
from mechanistic.condition_switch_figures import centroid_candidate_scores
from mechanistic.component_causal_metrics import causal_geometry, outcome_metrics
from mechanistic.direction_logit_lens import rms_norm_directions
from mechanistic.cumulative_hypothesis_analysis import _fit_decomposition, _question_weights
from mechanistic.hypothesis_analysis import _design, _ridge_fit_predict
from mechanistic.feedback_direction import _unit_rows
from mechanistic.io import atomic_save_npz, read_metadata
from mechanistic.modeling import cpu_lens, locate_model_parts, render_chat
from mechanistic.perturbation_analysis import _residual_metrics
from mechanistic.prompts import (
    GAME_FEEDBACK,
    NEUTRAL_FEEDBACK,
    baseline_question_turn,
    build_messages,
    present_question,
    repeated_question_turn,
)
from mechanistic.steering import ResidualSteerer, build_schedule
from mechanistic.switch_direction import _compression, _fit_directions, _nuisance_design


QUESTION = {
    "id": "q1", "question": "Example?",
    "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
    "correct_answer": "A",
}


class PromptTests(unittest.TestCase):
    def test_hf_template_forwards_model_specific_reasoning_budget(self):
        class RecordingProcessor:
            def __init__(self):
                self.kwargs = None

            def apply_chat_template(self, _messages, **kwargs):
                self.kwargs = kwargs
                return "rendered"

        processor = RecordingProcessor()
        result = render_chat(
            processor,
            [{"role": "user", "content": "Question"}],
            True,
            "hf_template",
            {"thinking_budget": 0},
        )
        self.assertEqual(result, "rendered")
        self.assertEqual(processor.kwargs["thinking_budget"], 0)
        self.assertFalse(processor.kwargs["enable_thinking"])

    def test_question_format_matches_runner(self):
        q = present_question(QUESTION)
        self.assertTrue(q.startswith("------------------------------\nQuestion:\nExample?"))
        self.assertIn("  D: four\n------------------------------", q)

    def test_baseline_roles(self):
        messages = build_messages(QUESTION, "baseline")
        self.assertEqual([x["role"] for x in messages], ["system", "user"])
        self.assertIn("Your choice (A, B, C, or D): ", messages[-1]["content"])

    def test_game_and_neutral_use_model_visible_system_prefix(self):
        game = build_messages(QUESTION, "incorrect")
        neutral = build_messages(QUESTION, "neutral")
        self.assertEqual([x["role"] for x in game], ["system", "user", "assistant", "user"])
        self.assertEqual(game[2]["content"], "[redacted]")
        self.assertFalse(neutral[0]["content"].startswith("None"))
        self.assertEqual(
            neutral[0]["content"],
            "Output ONLY the letter of your choice: A, B, C, or D.\n",
        )

    def test_attention_span_indices_find_condition_terms(self):
        class CharacterTokenizer:
            def __call__(self, text, **_kwargs):
                return {
                    "input_ids": list(range(len(text))),
                    "offset_mapping": [(i, i + 1) for i in range(len(text))],
                }

        tokenizer = CharacterTokenizer()
        game_messages = build_messages(QUESTION, "incorrect")
        game_prompt = "\n".join(message["content"] for message in game_messages)
        ids, spans = attention_span_indices(tokenizer, game_prompt, "incorrect", QUESTION)
        self.assertEqual(list(spans), list(SPAN_NAMES))
        self.assertEqual(len(spans["condition_keyword"]), 18)  # two copies of "incorrect"
        self.assertEqual(len(spans["action_keyword"]), len("different answer"))
        self.assertEqual(len(spans["repeated_question"]), len(present_question(QUESTION)))
        self.assertEqual(spans["query_self"], [len(ids) - 1])

    def test_source_positions_distinguish_user_and_system_incorrect(self):
        spans = {
            "condition_keyword": [3, 17],
            "feedback_sentence": [15, 16, 17, 18],
            "system_condition": [1, 2, 3, 4],
        }
        self.assertEqual(_source_positions(spans, "user_incorrect"), [17])
        self.assertEqual(_source_positions(spans, "system_incorrect"), [3])

    def test_incorrect_system_setup_ablation_preserves_user_feedback(self):
        messages = build_messages(QUESTION, "incorrect_no_system_setup")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(
            messages[0]["content"],
            "Output ONLY the letter of your choice: A, B, C, or D.\n",
        )
        self.assertTrue(messages[-1]["content"].startswith(GAME_FEEDBACK))
        self.assertNotIn("previous answer was incorrect", messages[0]["content"])

    def test_prompt_mode_removes_only_incorrect_system_setup(self):
        game = build_messages(QUESTION, "incorrect", "no_system_incorrect")
        neutral = build_messages(QUESTION, "neutral", "no_system_incorrect")
        self.assertEqual(game[0], neutral[0])
        self.assertEqual(game[:3], neutral[:3])
        self.assertTrue(game[-1]["content"].startswith(GAME_FEEDBACK))

    def test_baseline_matched_mode_matches_first_question_and_preserves_original_repeat(self):
        baseline = build_messages(QUESTION, "baseline", "baseline_matched")
        game = build_messages(QUESTION, "incorrect", "baseline_matched")
        neutral = build_messages(QUESTION, "neutral", "baseline_matched")
        question_turn = baseline_question_turn(QUESTION)
        repeated_turn = repeated_question_turn(QUESTION)

        self.assertEqual(game[0], baseline[0])
        self.assertEqual(neutral[0], baseline[0])
        self.assertEqual(game[1], baseline[1])
        self.assertEqual(neutral[1], baseline[1])
        self.assertEqual(game[1]["content"], question_turn)
        self.assertEqual(game[-1]["content"], GAME_FEEDBACK + "\n\n" + repeated_turn)
        self.assertEqual(neutral[-1]["content"], NEUTRAL_FEEDBACK + "\n\n" + repeated_turn)
        self.assertNotIn("I'm going to ask you", game[-1]["content"])
        self.assertLess(
            game[-1]["content"].index("Respond only with"),
            game[-1]["content"].index("Question:"),
        )
        self.assertEqual(game[:3], neutral[:3])

    def test_raw_qwen_chatml_has_exact_baseline_prefix_before_redacted(self):
        class NoTemplateProcessor:
            def apply_chat_template(self, *_args, **_kwargs):
                raise AssertionError("Hugging Face chat template must not be called")

        processor = NoTemplateProcessor()
        baseline = render_chat(
            processor,
            build_messages(QUESTION, "baseline", "baseline_matched"),
            True,
            "raw_qwen_chatml",
        )
        game = render_chat(
            processor,
            build_messages(QUESTION, "incorrect", "baseline_matched"),
            True,
            "raw_qwen_chatml",
        )
        neutral = render_chat(
            processor,
            build_messages(QUESTION, "neutral", "baseline_matched"),
            True,
            "raw_qwen_chatml",
        )
        self.assertEqual(game.split("[redacted]", 1)[0], baseline)
        self.assertEqual(neutral.split("[redacted]", 1)[0], baseline)
        self.assertTrue(baseline.endswith("<think>\n\n</think>\n\n"))

    def test_empty_history_uses_only_the_same_thinking_scaffold(self):
        class NoTemplateProcessor:
            def apply_chat_template(self, *_args, **_kwargs):
                raise AssertionError("Hugging Face chat template must not be called")

        processor = NoTemplateProcessor()
        baseline = render_chat(
            processor,
            build_messages(QUESTION, "baseline", "baseline_matched_empty_history"),
            True,
            "raw_qwen_chatml",
        )
        game_messages = build_messages(
            QUESTION, "incorrect", "baseline_matched_empty_history"
        )
        neutral_messages = build_messages(
            QUESTION, "neutral", "baseline_matched_empty_history"
        )
        self.assertEqual(game_messages[2]["content"], "")
        self.assertEqual(neutral_messages[2]["content"], "")
        game = render_chat(processor, game_messages, True, "raw_qwen_chatml")
        neutral = render_chat(processor, neutral_messages, True, "raw_qwen_chatml")
        scaffold = "<think>\n\n</think>\n\n"
        historical_turn = (
            "<|im_start|>assistant\n" + scaffold + "<|im_end|>\n"
        )
        self.assertEqual(game.count(historical_turn), 1)
        self.assertEqual(neutral.count(historical_turn), 1)
        self.assertTrue(game.endswith("<|im_start|>assistant\n" + scaffold))
        self.assertTrue(neutral.endswith("<|im_start|>assistant\n" + scaffold))
        self.assertTrue(baseline.endswith("<|im_start|>assistant\n" + scaffold))
        self.assertNotIn("[redacted]", game + neutral)

    def test_bare_raw_qwen_chatml_matches_prefix_without_thinking_tokens(self):
        class NoTemplateProcessor:
            def apply_chat_template(self, *_args, **_kwargs):
                raise AssertionError("Hugging Face chat template must not be called")

        processor = NoTemplateProcessor()
        baseline = render_chat(
            processor,
            build_messages(QUESTION, "baseline", "baseline_matched"),
            True,
            "raw_qwen_chatml_bare",
        )
        game = render_chat(
            processor,
            build_messages(QUESTION, "incorrect", "baseline_matched"),
            True,
            "raw_qwen_chatml_bare",
        )
        neutral = render_chat(
            processor,
            build_messages(QUESTION, "neutral", "baseline_matched"),
            True,
            "raw_qwen_chatml_bare",
        )
        self.assertEqual(game.split("[redacted]", 1)[0], baseline)
        self.assertEqual(neutral.split("[redacted]", 1)[0], baseline)
        self.assertTrue(baseline.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>", baseline + game + neutral)
        self.assertNotIn("</think>", baseline + game + neutral)

    def test_gdn_user_incorrect_is_user_feedback_intersection(self):
        spans = {
            "condition_keyword": [3, 17],
            "feedback_sentence": [15, 16, 17, 18],
        }
        self.assertEqual(user_incorrect_positions(spans), [17])

    def test_structural_controls_are_distinct_and_deterministic(self):
        class Tokenizer:
            all_special_ids = []
            def decode(self, ids, **_kwargs):
                return ["-", ":", ".", "\n", "word", "!", "?", ";"][ids[0] % 8]

        ids = list(range(80))
        spans = {
            "condition_keyword": [20], "feedback_sentence": [19, 20, 21],
            "system_condition": [2], "redacted_answer": [10],
            "previous_8": list(range(72, 80)), "query_self": [79],
        }
        first = structural_control_positions(Tokenizer(), ids, spans, "q", 4, 42)
        second = structural_control_positions(Tokenizer(), ids, spans, "q", 4, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 4)


class IOTests(unittest.TestCase):
    def test_atomic_npz(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.npz"
            atomic_save_npz(path, values=np.arange(3), metadata=np.asarray(json.dumps({"x": 1})))
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z["values"], np.arange(3))
                self.assertEqual(read_metadata(z), {"x": 1})


class HypothesisTests(unittest.TestCase):
    def test_causal_geometry_does_not_call_a_rank_swap_compression(self):
        baseline = np.asarray([[3.0, 1.0, -1.0, -3.0]])
        natural = baseline.copy()
        swapped = np.asarray([[-3.0, 1.0, -1.0, 3.0]])
        geometry = causal_geometry(swapped, natural, baseline)
        natural_metrics = outcome_metrics(natural, baseline, np.asarray([0]), np.asarray([0]))
        swapped_metrics = outcome_metrics(swapped, baseline, np.asarray([0]), np.asarray([0]))
        self.assertGreater(float(geometry["causal_total_l1"][0]), 0)
        self.assertGreater(float(geometry["causal_orthogonal_l2"][0]), 0)
        self.assertAlmostEqual(
            float(swapped_metrics["ad_spread"][0]),
            float(natural_metrics["ad_spread"][0]),
        )
        self.assertAlmostEqual(
            float(swapped_metrics["ad_entropy"][0]),
            float(natural_metrics["ad_entropy"][0]),
        )

    def test_switch_direction_recovers_common_signal(self):
        rng = np.random.default_rng(19)
        n, hidden = 160, 24
        original = np.repeat(np.arange(4), n // 4)
        groups = np.tile(np.repeat(np.asarray([0, 1]), n // 8), 4)
        baseline_logits = rng.normal(size=(n, 4))
        baseline_logits[np.arange(n), original] += 3
        baseline = rng.normal(size=(n, hidden))
        signal = rng.normal(size=hidden); signal /= np.linalg.norm(signal)
        delta = rng.normal(scale=.1, size=(n, hidden)) + (groups == 1)[:, None] * signal
        design = _nuisance_design(baseline_logits, original, np.zeros(n, dtype=bool))
        directions, _ = _fit_directions(
            delta, baseline, groups, original, design, np.ones(n, dtype=bool)
        )
        self.assertGreater(float(directions["answer_orthogonal"] @ signal), .8)

    def test_compression_score_is_positive_for_scaling_down(self):
        neutral = np.asarray([[[2.0, 1.0, -1.0, -2.0]]])
        game = .6 * neutral
        np.testing.assert_allclose(_compression(game, neutral), .4)

    def test_targeted_prior_design_recovers_suppression(self):
        rng = np.random.default_rng(1); n = 200
        x = rng.normal(size=(n, 4)); prior = rng.integers(0, 4, n)
        leader = np.argmax(x, axis=1); margin = np.sort(x, axis=1)[:, -1] - np.sort(x, axis=1)[:, -2]
        design = _design("targeted_prior_winner", x, prior, leader, margin, 0)
        y = -2 * design[:, 0] + rng.normal(scale=.05, size=n * 4)
        pred, coef = _ridge_fit_predict(design[:600], y[:600], design[600:])
        self.assertLess(np.mean((pred - y[600:]) ** 2), .01)
        self.assertLess(coef[1], -1.9)

    def test_cumulative_decomposition_recovers_compression_and_winner_penalty(self):
        rng = np.random.default_rng(7); n = 400
        winner = np.tile(np.arange(4), n // 4)
        baseline = rng.normal(size=(n, 4))
        baseline -= baseline.mean(axis=1, keepdims=True)
        winner_column = np.eye(4)[winner] - .25
        letter_effect = np.asarray([.3, -.1, .2, -.4])
        target = -.4 * baseline - .7 * winner_column + letter_effect
        compression, winner_penalty = _fit_decomposition(
            baseline, winner, target, _question_weights(winner)
        )
        self.assertAlmostEqual(compression, .4, places=4)
        self.assertAlmostEqual(winner_penalty, .7, places=4)

    def test_cumulative_weights_balance_winner_letters(self):
        labels = np.asarray([0, 0, 0, 1, 2, 2, 3])
        weights = _question_weights(labels)
        totals = np.asarray([weights[labels == label].sum() for label in range(4)])
        np.testing.assert_allclose(totals, totals[0])

    def test_runner_winner_residual_direction(self):
        # Original winner A, runner-up B. The residual lies entirely along B-A.
        residuals = np.asarray([[[-1.0, 1.0, 0.0, 0.0]]])
        order = np.asarray([[0, 1, 2, 3]])
        metrics = _residual_metrics(residuals, order)
        self.assertAlmostEqual(float(metrics["runner_vs_winner_projection"][0, 0]), np.sqrt(2))
        self.assertAlmostEqual(float(metrics["winner_runner_energy_fraction"][0, 0]), 1.0)

    def test_feedback_direction_rows_are_unit_normalized(self):
        direction, norms = _unit_rows(np.asarray([[3.0, 4.0], [0.0, 2.0]]))
        np.testing.assert_allclose(norms, [5.0, 2.0])
        np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1.0)


class SteeringTests(unittest.TestCase):
    def test_macro_bootstrap_is_letter_balanced(self):
        values = np.asarray([0.0, 2.0, 10.0, 20.0, 30.0])
        labels = np.asarray([0, 0, 1, 2, 3])
        mean, low, high = _macro_bootstrap(
            values, labels, 2000, np.random.default_rng(3)
        )
        self.assertAlmostEqual(mean, 15.25)
        self.assertLess(low, mean)
        self.assertGreater(high, mean)

    def test_macro_bootstrap_requires_every_letter(self):
        with self.assertRaises(ValueError):
            _macro_bootstrap(
                np.asarray([1.0, 2.0, 3.0]),
                np.asarray([0, 1, 2]),
                10,
                np.random.default_rng(3),
            )

    def test_schedule_contains_bidirectional_amplification_and_controls(self):
        schedule = build_schedule(
            ["incorrect", "neutral"], [24, 30, 36], [-1, 1], 30, [-.5, .5, 2], 30, [-1, 1]
        )
        self.assertEqual(len(schedule), 24)
        ids = {spec.scenario_id for spec in schedule}
        self.assertIn("incorrect__feedback__l30__p2", ids)
        self.assertIn("incorrect__feedback__l30__m1", ids)
        self.assertIn("neutral__feedback__l30__p1", ids)
        self.assertIn("incorrect__control__l30__p1", ids)

    def test_residual_steerer_modifies_only_selected_position(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        class Block(torch.nn.Module):
            def forward(self, values):
                return (values, "cache")

        block = Block()
        parts = SimpleNamespace(layers=[block], embedding=SimpleNamespace())
        steerer = ResidualSteerer(parts, 1, [1], torch.tensor([1.0, -1.0]), 2.0, .5)
        try:
            output, cache = block(torch.zeros(1, 3, 2))
        finally:
            steerer.close()
        self.assertEqual(cache, "cache")
        np.testing.assert_allclose(output.numpy()[0, 0], [0.0, 0.0])
        np.testing.assert_allclose(output.numpy()[0, 1], [1.0, -1.0])
        np.testing.assert_allclose(output.numpy()[0, 2], [0.0, 0.0])

    def test_attention_edge_mask_changes_only_selected_entries(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        scores = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
        got = _ablate_scores(scores, (1,), (3,), (2,))
        self.assertTrue(torch.isneginf(got[:, 1, 3, 2]).all())
        unchanged = torch.ones_like(scores, dtype=torch.bool)
        unchanged[:, 1, 3, 2] = False
        torch.testing.assert_close(got[unchanged], scores[unchanged])

    def test_batched_attention_edge_mask_is_row_specific(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        scores = torch.zeros(2, 2, 4, 5)
        got = _ablate_scores_batched(scores, {
            0: ((1,), (3,), (2,)),
            1: ((0,), (2,), (1, 4)),
        })
        self.assertTrue(torch.isneginf(got[0, 1, 3, 2]))
        self.assertTrue(torch.isneginf(got[1, 0, 2, 1]))
        self.assertTrue(torch.isneginf(got[1, 0, 2, 4]))
        self.assertEqual(torch.isneginf(got).sum().item(), 3)

    def test_zero_beta_write_is_narrow(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        beta = torch.ones(1, 6, 4)
        got = zero_beta_writes(beta, (2,), (1, 3))
        expected = beta.clone(); expected[:, 2, [1, 3]] = 0
        torch.testing.assert_close(got, expected)
        torch.testing.assert_close(beta, torch.ones_like(beta))


class LensTests(unittest.TestCase):
    def test_direction_lens_uses_checkpoint_rms_norm(self):
        directions = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        weight = np.asarray([2.0, 0.5], dtype=np.float32)
        got = rms_norm_directions(directions, weight, 0.0)
        scale = np.sqrt((3.0 ** 2 + 4.0 ** 2) / 2.0)
        np.testing.assert_allclose(got[0], [6.0 / scale, 2.0 / scale])
        np.testing.assert_allclose(got[1], [0.0, 0.0])

    def test_letter_balanced_mean_weights_letters_equally(self):
        values = np.asarray([[-1.0], [1.0], [3.0], [5.0], [7.0], [9.0], [11.0], [13.0]])
        strata = np.repeat(np.arange(4), 2)
        mean, se = macro_mean_and_se(values, strata)
        np.testing.assert_allclose(mean, [6.0])
        self.assertGreater(float(se[0]), 0.0)

    def test_letter_balanced_mean_requires_every_letter(self):
        values = np.asarray([[0.0], [0.0], [4.0], [4.0]])
        strata = np.asarray([0, 0, 1, 1])
        with self.assertRaises(ValueError):
            macro_mean_and_se(values, strata)

    def test_cross_fitted_candidate_scores_use_one_common_decoder(self):
        labels = np.tile(np.arange(4), 2)
        baseline = np.eye(4, dtype=np.float32)[labels]
        residuals = np.repeat(baseline[None, :, None, :], 3, axis=0)
        scores = centroid_candidate_scores(residuals, labels, folds=2, seed=1)
        self.assertEqual(scores.shape, (3, 8, 1, 4))
        np.testing.assert_allclose(scores[0], scores[1])
        np.testing.assert_allclose(scores[0], scores[2])
        np.testing.assert_allclose(scores.mean(axis=-1), 0.0, atol=1e-6)
        np.testing.assert_allclose(scores[0].std(ddof=1), 1.0, atol=1e-6)

    def test_selected_row_lens_matches_direct_head(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        class Trunk(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.embed_tokens = torch.nn.Embedding(11, 6)
                self.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
                self.norm = torch.nn.RMSNorm(6)

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.model = torch.nn.Module()
                self.model.language_model = Trunk(); self.lm_head = torch.nn.Linear(6, 11, bias=False)
            def get_output_embeddings(self): return self.lm_head

        model = Wrapper(); parts = locate_model_parts(model)
        residuals = torch.randn(3, 3, 6); ids = [1, 4, 7, 9]
        got = cpu_lens(parts, residuals, ids)
        expected = parts.final_norm(residuals.float()) @ parts.output_head.weight[ids].T
        torch.testing.assert_close(got, expected)
        self.assertFalse(got.requires_grad)


if __name__ == "__main__": unittest.main()
