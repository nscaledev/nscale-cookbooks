"""Acceptance checks for the coding exercise; no API access required."""

from copy import deepcopy
import unittest

from usage import summarise_usage


class UsageTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(summarise_usage([]), {})

    def test_groups_models_and_preserves_input(self):
        records = [
            {"model": "model-a", "prompt_tokens": 12, "completion_tokens": 3},
            {"model": "model-b", "prompt_tokens": 0, "completion_tokens": 9},
            {"model": "model-a", "prompt_tokens": 8, "completion_tokens": 7},
        ]
        original = deepcopy(records)
        self.assertEqual(summarise_usage(records), {
            "model-a": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            "model-b": {"prompt_tokens": 0, "completion_tokens": 9, "total_tokens": 9},
        })
        self.assertEqual(records, original)

    def test_zero_tokens(self):
        self.assertEqual(summarise_usage([
            {"model": "model-a", "prompt_tokens": 0, "completion_tokens": 0}
        ]), {"model-a": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    def test_rejects_negative_counts(self):
        for prompt, completion in [(-1, 0), (0, -1)]:
            with self.subTest(prompt=prompt, completion=completion):
                with self.assertRaises(ValueError):
                    summarise_usage([
                        {"model": "model-a", "prompt_tokens": prompt, "completion_tokens": completion}
                    ])


if __name__ == "__main__":
    unittest.main()
