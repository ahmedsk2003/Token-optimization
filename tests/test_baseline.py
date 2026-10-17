"""Tests for baseline prompt measurement system."""

import os
import tempfile
import unittest
from pathlib import Path

from src.evaluation.baseline import (
    PromptMeasurement,
    count_characters,
    count_words,
    measure_baseline,
    measure_prompt,
    save_baseline_to_csv,
)
from src.utils.tokenizer import BaseTokenizer, TiktokenTokenizer, get_tokenizer


class DummyCustomTokenizer(BaseTokenizer):
    """Custom tokenizer test double to verify tokenizer isolation & replaceability."""

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split("-"))

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return [len(part) for part in text.split("-")]

    def decode(self, tokens: list[int]) -> str:
        return "-".join(str(t) for t in tokens)


class TestBaselineMeasurement(unittest.TestCase):

    def setUp(self):
        self.tokenizer = get_tokenizer("cl100k_base")

    def test_token_count_produced_by_tokenizer(self):
        prompt = "Hello, how are you doing today?"
        expected_tokens = len(self.tokenizer.encode(prompt))
        measurement = measure_prompt(prompt, prompt_id="P001", tokenizer=self.tokenizer)

        self.assertIsInstance(measurement.token_count, int)
        self.assertEqual(measurement.token_count, expected_tokens)
        self.assertGreater(measurement.token_count, 0)

    def test_word_and_character_counts_produced(self):
        prompt = "Explain quantum computing in simple terms.\nInclude 3 examples."
        measurement = measure_prompt(prompt, prompt_id="P001", tokenizer=self.tokenizer)

        expected_chars = len(prompt)
        expected_words = len(prompt.split())

        self.assertEqual(measurement.character_count, expected_chars)
        self.assertEqual(measurement.word_count, expected_words)

    def test_original_prompt_text_unchanged(self):
        prompts = [
            "  Leading and trailing whitespace  ",
            "Special characters: !@#$%^&*()_+{}|:\"<>?~`\n\t",
            "Multi-line\nstring\nwith\nexact\nformatting.",
        ]
        measurements = measure_baseline(prompts, tokenizer=self.tokenizer)

        for original, measured in zip(prompts, measurements):
            self.assertEqual(measured.prompt, original)

    def test_multiple_prompts_receive_unique_sequential_ids(self):
        prompts = [
            "First prompt text",
            "Second prompt text",
            "Third prompt text",
        ]
        measurements = measure_baseline(prompts, tokenizer=self.tokenizer)

        ids = [m.prompt_id for m in measurements]
        self.assertEqual(ids, ["P001", "P002", "P003"])
        self.assertEqual(len(ids), len(set(ids)))

    def test_repeating_same_measurement_gives_same_result(self):
        prompt = "Determine whether determinism holds across multiple runs."
        run1 = measure_prompt(prompt, prompt_id="P001", tokenizer=self.tokenizer)
        run2 = measure_prompt(prompt, prompt_id="P001", tokenizer=self.tokenizer)

        self.assertEqual(run1, run2)
        self.assertEqual(run1.token_count, run2.token_count)
        self.assertEqual(run1.word_count, run2.word_count)
        self.assertEqual(run1.character_count, run2.character_count)

    def test_tokenizer_isolation_and_replaceability(self):
        custom_tok = DummyCustomTokenizer()
        prompt = "a-b-c-d"
        measurement = measure_prompt(prompt, prompt_id="P001", tokenizer=custom_tok)

        self.assertEqual(measurement.token_count, 4)

    def test_measurement_works_on_compressed_prompts_later(self):
        orig_prompt = "Please provide a detailed summary of the history of computing."
        comp_prompt = "Summarize history of computing."

        orig_measurement = measure_prompt(orig_prompt, prompt_id="P001", tokenizer=self.tokenizer)
        comp_measurement = measure_prompt(comp_prompt, prompt_id="P001_comp", tokenizer=self.tokenizer)

        self.assertLess(comp_measurement.token_count, orig_measurement.token_count)
        self.assertLess(comp_measurement.word_count, orig_measurement.word_count)
        self.assertLess(comp_measurement.character_count, orig_measurement.character_count)

    def test_csv_saving_and_no_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompts = ["Test prompt 1", "Test prompt 2"]
            measurements = measure_baseline(prompts, tokenizer=self.tokenizer)

            out_csv = save_baseline_to_csv(
                measurements,
                output_dir=tmp_dir,
                filename="baseline_test.csv",
                overwrite=False,
            )

            self.assertTrue(out_csv.exists())

            # Attempting to save again with overwrite=False must raise FileExistsError
            with self.assertRaises(FileExistsError):
                save_baseline_to_csv(
                    measurements,
                    output_dir=tmp_dir,
                    filename="baseline_test.csv",
                    overwrite=False,
                )

            # Saving with overwrite=True should succeed
            save_baseline_to_csv(
                measurements,
                output_dir=tmp_dir,
                filename="baseline_test.csv",
                overwrite=True,
            )


if __name__ == "__main__":
    unittest.main()
