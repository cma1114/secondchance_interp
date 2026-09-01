import unittest

import numpy as np

from mechanistic.jlens_answer_content import option_token_lists, pad_option_tokens


class WordTokenizer:
    def __init__(self):
        self.ids = {"cat": 1, "dog": 2, "shared": 3, "!": 4, "blue": 5}
        self.text = {value: key for key, value in self.ids.items()}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        words = text.strip().replace("!", " !").split()
        return [self.ids[word.lower()] for word in words]

    def decode(self, ids):
        return " " + self.text[int(ids[0])]


class OptionContentTests(unittest.TestCase):
    def test_selects_alphanumeric_option_distinctive_tokens(self):
        tokenizer = WordTokenizer()
        options = {"A": "cat shared", "B": "dog shared", "C": "blue!", "D": "shared"}
        selected, audit = option_token_lists(tokenizer, options)
        self.assertEqual(selected[0], [1])
        self.assertEqual(selected[1], [2])
        self.assertEqual(selected[2], [5])
        self.assertEqual(selected[3], [3])
        self.assertEqual(audit["fallback_letters"], ["D"])

    def test_padding_preserves_lengths(self):
        ids, mask = pad_option_tokens([[[1], [2, 3], [4], [5, 6, 7]]])
        self.assertEqual(ids.shape, (1, 4, 3))
        np.testing.assert_array_equal(mask.sum(axis=-1), [[1, 2, 1, 3]])


if __name__ == "__main__":
    unittest.main()
