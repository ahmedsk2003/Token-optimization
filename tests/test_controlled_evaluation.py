"""Unit tests for controlled evaluation calculations, edge cases, and summary generation."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.evaluation.controlled_runner import (
    ControlledEvaluationRunner,
    PromptEvaluationResult,
    build_summary_dataframe,
    check_must_preserve,
    compute_token_metrics,
    parse_must_preserve,
)
from src.utils.tokenizer import BaseTokenizer, get_tokenizer


class DummyTestTokenizer(BaseTokenizer):
    """Deterministic word-based tokenizer double for unit testing."""

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return [len(w) for w in text.split()]

    def decode(self, tokens: list[int]) -> str:
        return " ".join(str(t) for t in tokens)


class DummyFailingCompressor:
    """Mock compressor that raises an exception to verify error handling."""

    def compress(self, prompt: str) -> str:
        raise RuntimeError("Simulated compressor failure")


class TestControlledEvaluationMetrics(unittest.TestCase):
    """Test suite for metric calculations and edge cases."""

    # 1. Token Metrics & Denominators
    def test_compute_token_metrics_standard(self):
        saved, pct = compute_token_metrics(100, 75)
        self.assertEqual(saved, 25)
        self.assertEqual(pct, 25.0)

    def test_compute_token_metrics_zero_denominator(self):
        saved, pct = compute_token_metrics(0, 0)
        self.assertEqual(saved, 0)
        self.assertEqual(pct, 0.0)

    def test_compute_token_metrics_negative_denominator(self):
        saved, pct = compute_token_metrics(-5, 0)
        self.assertEqual(saved, 0)
        self.assertEqual(pct, 0.0)

    def test_compute_token_metrics_expansion_or_unchanged(self):
        saved, pct = compute_token_metrics(50, 60)
        self.assertEqual(saved, 0)
        self.assertEqual(pct, 0.0)

        saved_same, pct_same = compute_token_metrics(50, 50)
        self.assertEqual(saved_same, 0)
        self.assertEqual(pct_same, 0.0)

    # 2. Must-Preserve Parsing
    def test_parse_must_preserve_standard(self):
        raw = "three bullet points;;40-kilometer;;March;;two years"
        parsed = parse_must_preserve(raw)
        self.assertEqual(parsed, ["three bullet points", "40-kilometer", "March", "two years"])

    def test_parse_must_preserve_whitespace_and_empty(self):
        self.assertEqual(parse_must_preserve(""), [])
        self.assertEqual(parse_must_preserve("   "), [])
        self.assertEqual(parse_must_preserve(None), [])
        self.assertEqual(parse_must_preserve(float("nan")), [])
        self.assertEqual(parse_must_preserve("item1 ;;  ;; item2;;  "), ["item1", "item2"])

    # 3. Must-Preserve Preservation Check
    def test_check_must_preserve_all_present(self):
        items = ["Python", "Fibonacci", "memoization"]
        text = "Write a python function that returns the nth fibonacci number with memoization."
        total, pres, all_pres, missing = check_must_preserve(items, text)

        self.assertEqual(total, 3)
        self.assertEqual(pres, 3)
        self.assertTrue(all_pres)
        self.assertEqual(missing, [])

    def test_check_must_preserve_missing_items(self):
        items = ["Python", "Docker", "Kubernetes"]
        text = "A Python script deployed in Docker."
        total, pres, all_pres, missing = check_must_preserve(items, text)

        self.assertEqual(total, 3)
        self.assertEqual(pres, 2)
        self.assertFalse(all_pres)
        self.assertEqual(missing, ["Kubernetes"])

    def test_check_must_preserve_empty_items(self):
        total, pres, all_pres, missing = check_must_preserve([], "Any prompt text")
        self.assertEqual(total, 0)
        self.assertEqual(pres, 0)
        self.assertTrue(all_pres)
        self.assertEqual(missing, [])


class TestControlledEvaluationRunner(unittest.TestCase):
    """Test suite for ControlledEvaluationRunner execution and safety."""

    def setUp(self):
        self.tokenizer = DummyTestTokenizer()
        self.runner = ControlledEvaluationRunner(tokenizer=self.tokenizer)

    # 1. Baseline Evaluation
    def test_baseline_prompt_unchanged(self):
        prompt = "Explain photosynthesis in simple terms."
        res = self.runner.evaluate_prompt(
            prompt_id="TEST-01",
            category="Test Category",
            expected_action="no_compression",
            duplicate_type="none",
            prompt=prompt,
            must_preserve_raw="photosynthesis;;simple terms",
            experiment_name="baseline",
        )

        self.assertEqual(res.prompt_id, "TEST-01")
        self.assertEqual(res.experiment_name, "baseline")
        self.assertEqual(res.compressed_prompt, prompt)
        self.assertFalse(res.prompt_changed)
        self.assertEqual(res.tokens_saved, 0)
        self.assertEqual(res.reduction_percentage, 0.0)
        self.assertTrue(res.must_preserve_all_preserved)
        self.assertTrue(res.validator_is_valid)
        self.assertEqual(res.error_message, "")

    # 2. Exact Duplicate Removal Evaluation
    def test_exact_duplicate_removal_execution(self):
        prompt = (
            "Summarize the quarterly earnings in three bullets.\n"
            "Summarize the quarterly earnings in three bullets."
        )
        res = self.runner.evaluate_prompt(
            prompt_id="DUP-01",
            category="Exact and near-duplicate content",
            expected_action="exact_duplicate_removal",
            duplicate_type="exact",
            prompt=prompt,
            must_preserve_raw="quarterly earnings;;three bullets",
            experiment_name="exact_duplicate_removal",
        )

        self.assertTrue(res.prompt_changed)
        self.assertGreater(res.tokens_saved, 0)
        self.assertGreater(res.reduction_percentage, 0.0)
        self.assertTrue(res.must_preserve_all_preserved)
        self.assertEqual(res.compressed_prompt, "Summarize the quarterly earnings in three bullets.")

    # 3. Empty and None Prompt Handling
    def test_empty_and_none_prompt_safety(self):
        res_empty = self.runner.evaluate_prompt(
            prompt_id="EMPTY-01",
            category="Edge Cases",
            expected_action="no_compression",
            duplicate_type="none",
            prompt="",
            must_preserve_raw="",
            experiment_name="exact_duplicate_removal",
        )
        self.assertEqual(res_empty.tokens_before, 0)
        self.assertEqual(res_empty.tokens_after, 0)
        self.assertEqual(res_empty.tokens_saved, 0)
        self.assertEqual(res_empty.reduction_percentage, 0.0)
        self.assertFalse(res_empty.prompt_changed)
        self.assertTrue(res_empty.validator_is_valid)

        res_none = self.runner.evaluate_prompt(
            prompt_id="NONE-01",
            category="Edge Cases",
            expected_action="no_compression",
            duplicate_type="none",
            prompt=None,
            must_preserve_raw=None,
            experiment_name="baseline",
        )
        self.assertEqual(res_none.tokens_before, 0)
        self.assertEqual(res_none.reduction_percentage, 0.0)

    # 4. Exception Handling without Dropping Records
    def test_exception_handling_safeguard(self):
        failing_runner = ControlledEvaluationRunner(
            tokenizer=self.tokenizer,
            exact_compressor=DummyFailingCompressor(),
        )
        prompt = "Test prompt that triggers an exception."
        res = failing_runner.evaluate_prompt(
            prompt_id="FAIL-01",
            category="Error Handling",
            expected_action="exact_duplicate_removal",
            duplicate_type="none",
            prompt=prompt,
            must_preserve_raw="Test prompt",
            experiment_name="exact_duplicate_removal",
        )

        # Fallback to original prompt
        self.assertEqual(res.compressed_prompt, prompt)
        self.assertFalse(res.prompt_changed)
        self.assertEqual(res.tokens_saved, 0)
        self.assertIn("Simulated compressor failure", res.error_message)

    # 5. Unsupported Experiment Rejection
    def test_unsupported_experiment_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.runner.evaluate_prompt(
                prompt_id="BAD-01",
                category="Invalid",
                expected_action="none",
                duplicate_type="none",
                prompt="Prompt text",
                must_preserve_raw="",
                experiment_name="non_existent_technique",
            )


class TestSummaryAggregation(unittest.TestCase):
    """Test suite for summary metrics and CSV generation."""

    def test_build_summary_dataframe(self):
        results = [
            PromptEvaluationResult(
                prompt_id="P1",
                category="CatA",
                expected_action="act1",
                duplicate_type="dup",
                experiment_name="exp1",
                original_prompt="hello world test",
                compressed_prompt="hello world",
                prompt_changed=True,
                tokens_before=10,
                tokens_after=8,
                tokens_saved=2,
                reduction_percentage=20.0,
                processing_time_ms=1.5,
                must_preserve_count=1,
                must_preserve_preserved_count=1,
                must_preserve_all_preserved=True,
                missing_must_preserve="",
                validator_is_valid=True,
                validator_confidence="high",
                validator_error_count=0,
                validator_warning_count=0,
                validator_failed_categories="",
                validator_uncertain_categories="",
                validator_warnings="",
            ),
            PromptEvaluationResult(
                prompt_id="P2",
                category="CatA",
                expected_action="act1",
                duplicate_type="dup",
                experiment_name="exp1",
                original_prompt="another prompt test",
                compressed_prompt="another prompt test",
                prompt_changed=False,
                tokens_before=10,
                tokens_after=10,
                tokens_saved=0,
                reduction_percentage=0.0,
                processing_time_ms=0.5,
                must_preserve_count=1,
                must_preserve_preserved_count=1,
                must_preserve_all_preserved=True,
                missing_must_preserve="",
                validator_is_valid=False,
                validator_confidence="medium",
                validator_error_count=1,
                validator_warning_count=1,
                validator_failed_categories="negation",
                validator_uncertain_categories="uncertainty",
                validator_warnings="warning 1",
            ),
        ]

        summary_df = build_summary_dataframe(results)
        self.assertFalse(summary_df.empty)

        # Check CatA row
        cat_row = summary_df[(summary_df["experiment_name"] == "exp1") & (summary_df["category"] == "CatA")].iloc[0]
        self.assertEqual(cat_row["prompt_count"], 2)
        self.assertEqual(cat_row["prompts_changed_count"], 1)
        self.assertEqual(cat_row["prompts_changed_percentage"], 50.0)
        self.assertEqual(cat_row["mean_reduction_percentage"], 10.0)
        self.assertEqual(cat_row["validation_failure_count"], 1)

        # Check Overall row
        overall_row = summary_df[(summary_df["experiment_name"] == "exp1") & (summary_df["category"] == "[OVERALL]")].iloc[0]
        self.assertEqual(overall_row["prompt_count"], 2)
        self.assertEqual(overall_row["mean_reduction_percentage"], 10.0)

    def test_run_and_save_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sample_csv = tmp_path / "sample_dataset.csv"
            out_dir = tmp_path / "results"

            df = pd.DataFrame([
                {
                    "id": "S01",
                    "category": "Sample",
                    "expected_action": "no_compression",
                    "duplicate_type": "none",
                    "prompt": "Hello world.",
                    "must_preserve": "Hello",
                }
            ])
            df.to_csv(sample_csv, index=False)

            runner = ControlledEvaluationRunner(tokenizer=DummyTestTokenizer())
            per_prompt_path, summary_path = runner.run_and_save(
                dataset_path=sample_csv,
                output_dir=out_dir,
                techniques=["baseline"],
            )

            self.assertTrue(per_prompt_path.exists())
            self.assertTrue(summary_path.exists())

            read_per_prompt = pd.read_csv(per_prompt_path)
            self.assertEqual(len(read_per_prompt), 1)
            self.assertEqual(read_per_prompt.iloc[0]["prompt_id"], "S01")


if __name__ == "__main__":
    unittest.main()
