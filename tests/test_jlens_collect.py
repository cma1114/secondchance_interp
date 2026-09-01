import unittest

from mechanistic.jlens_collect import ANCHORS, _anchor_positions
from mechanistic.prompts import GAME_FEEDBACK, build_messages


QUESTION = {
    "id": "q1",
    "question": "Example?",
    "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
    "correct_answer": "A",
}


class CharacterTokenizer:
    def __call__(self, text, **_kwargs):
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class JLensAnchorTests(unittest.TestCase):
    def test_requested_game_source_tokens(self):
        tokenizer = CharacterTokenizer()
        game_messages = build_messages(
            QUESTION, "incorrect", "baseline_matched_empty_history"
        )
        game_system = game_messages[0]["content"]
        second_user = game_messages[-1]["content"]
        game_prompt = game_system + "\n" + second_user
        feedback_start = game_prompt.index(GAME_FEEDBACK)
        spans = {
            "first_question": [len(game_prompt) - 1],
            "redacted_answer": [feedback_start - 1],
            "condition_keyword": [feedback_start + GAME_FEEDBACK.index("incorrect")],
            "action_keyword": [feedback_start + GAME_FEEDBACK.rindex("answer")],
            "feedback_sentence": list(range(feedback_start, feedback_start + len(GAME_FEEDBACK))),
            "repeated_question": [len(game_prompt) - 1],
            "query_self": [len(game_prompt) - 1],
        }
        positions = dict(zip(
            ANCHORS,
            _anchor_positions(
                tokenizer, game_prompt, "incorrect", spans, game_system, second_user
            ),
        ))
        self.assertEqual(game_prompt[positions["feedback_subject_end"]], "r")
        self.assertEqual(game_prompt[positions["condition_keyword_end"]], "t")
        self.assertEqual(game_prompt[positions["user_different"]], "t")
        self.assertEqual(game_prompt[positions["action_keyword_end"]], "r")
        self.assertEqual(game_prompt[positions["feedback_end"]], ".")
        self.assertEqual(game_prompt[positions["instruction_letter"]], "r")
        self.assertEqual(game_prompt[positions["instruction_choice"]], "e")
        self.assertEqual(game_prompt[positions["instruction_end"]], ".")
        self.assertEqual(game_prompt[positions["repeated_choice"]], "e")
        self.assertEqual(game_prompt[positions["second_user_end"]], ":")

    def test_game_only_source_tokens_are_unavailable_in_neutral(self):
        tokenizer = CharacterTokenizer()
        neutral_messages = build_messages(
            QUESTION, "neutral", "baseline_matched_empty_history"
        )
        system = neutral_messages[0]["content"]
        feedback = neutral_messages[-1]["content"].split("\n\n", 1)[0]
        second_user = neutral_messages[-1]["content"]
        prompt = system + "\n" + second_user
        start = prompt.index(feedback)
        spans = {
            "first_question": [len(prompt) - 1],
            "redacted_answer": [start - 1],
            "condition_keyword": [start + feedback.index("transmission")],
            "action_keyword": [start + feedback.index("again")],
            "feedback_sentence": list(range(start, start + len(feedback))),
            "repeated_question": [len(prompt) - 1],
            "query_self": [len(prompt) - 1],
        }
        positions = dict(zip(
            ANCHORS,
            _anchor_positions(
                tokenizer, prompt, "neutral", spans, system, second_user
            ),
        ))
        self.assertIsNone(positions["user_different"])
        self.assertEqual(prompt[positions["feedback_subject_end"]], "e")
        self.assertEqual(prompt[positions["condition_keyword_end"]], "t")
        self.assertEqual(prompt[positions["action_keyword_end"]], "n")
        self.assertEqual(prompt[positions["feedback_end"]], ".")
        self.assertEqual(prompt[positions["instruction_letter"]], "r")
        self.assertEqual(prompt[positions["instruction_choice"]], "e")
        self.assertEqual(prompt[positions["instruction_end"]], ".")
        self.assertEqual(prompt[positions["repeated_choice"]], "e")
        self.assertEqual(prompt[positions["second_user_end"]], ":")


if __name__ == "__main__":
    unittest.main()
