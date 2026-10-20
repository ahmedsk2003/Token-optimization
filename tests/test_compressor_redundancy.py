"""Unit tests for Phase 4.1A: Exact Duplicate Removal Compressor."""

import unittest

import numpy as np

from src.compressors import (
    BaseCompressor,
    DuplicateRemovalResult,
    ExactDuplicateCompressor,
    SimilarityRedundancyCompressor,
    SimilarityRemovalResult,
    get_exact_duplicate_compressor,
    get_similarity_redundancy_compressor,
    remove_exact_duplicates,
    remove_similar_redundancy,
)
from src.utils.tokenizer import BaseTokenizer, get_tokenizer


class DummyTestTokenizer(BaseTokenizer):
    """Custom tokenizer test double for deterministic token calculation."""

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return [len(word) for word in text.split()]

    def decode(self, tokens: list[int]) -> str:
        return " ".join(str(t) for t in tokens)


class TestExactDuplicateCompressor(unittest.TestCase):
    """Test suite for ExactDuplicateCompressor."""

    def setUp(self):
        self.compressor = ExactDuplicateCompressor()
        self.tokenizer = get_tokenizer("cl100k_base")

    # 1. Empty & Whitespace Input
    def test_empty_input(self):
        self.assertEqual(self.compressor.compress(""), "")
        self.assertEqual(self.compressor.compress(None), "")
        self.assertEqual(remove_exact_duplicates(""), "")
        self.assertEqual(remove_exact_duplicates(None), "")

    def test_whitespace_only_input(self):
        self.assertEqual(self.compressor.compress("   "), "")
        self.assertEqual(self.compressor.compress("\n\t  \n"), "")
        self.assertEqual(remove_exact_duplicates("   \n   "), "")

    # 2. Single Sentence Input
    def test_single_sentence(self):
        prompt = "Explain quantum computing in simple terms."
        compressed = self.compressor.compress(prompt)
        self.assertEqual(compressed, prompt)

    def test_single_sentence_with_surrounding_whitespace(self):
        prompt = "  Explain quantum computing in simple terms.  "
        compressed = self.compressor.compress(prompt)
        self.assertEqual(compressed, "Explain quantum computing in simple terms.")

    # 3. Exact Duplicate Sentences
    def test_consecutive_exact_duplicates(self):
        prompt = "Explain machine learning. Explain machine learning."
        expected = "Explain machine learning."
        self.assertEqual(self.compressor.compress(prompt), expected)

    def test_three_consecutive_exact_duplicates(self):
        prompt = "Write clean code. Write clean code. Write clean code."
        expected = "Write clean code."
        self.assertEqual(self.compressor.compress(prompt), expected)

    def test_non_consecutive_exact_duplicates(self):
        prompt = "Task A is important. Task B is secondary. Task A is important."
        expected = "Task A is important. Task B is secondary."
        self.assertEqual(self.compressor.compress(prompt), expected)

    def test_multiple_distinct_duplicates(self):
        prompt = (
            "First directive. Second directive. First directive. "
            "Third directive. Second directive."
        )
        expected = "First directive. Second directive. Third directive."
        self.assertEqual(self.compressor.compress(prompt), expected)

    # 4. Whitespace and Case Normalization
    def test_case_insensitive_matching(self):
        prompt = "Machine learning is useful. machine learning is useful. MACHINE LEARNING IS USEFUL."
        expected = "Machine learning is useful."
        self.assertEqual(self.compressor.compress(prompt), expected)

    def test_whitespace_variations_between_sentences(self):
        prompt = "Explain machine learning.    explain machine learning. "
        expected = "Explain machine learning."
        self.assertEqual(self.compressor.compress(prompt), expected)

    def test_leading_trailing_whitespace_on_duplicate_sentences(self):
        prompt = "Data analysis is required.   data analysis is required.  "
        expected = "Data analysis is required."
        self.assertEqual(self.compressor.compress(prompt), expected)

    # 5. Preservation of Original Text and Order
    def test_original_text_and_casing_not_rewritten(self):
        # The first occurrence has specific capitalization and should not be modified
        prompt = "Summarize the findings in Q3. summarize the findings in q3."
        compressed = self.compressor.compress(prompt)
        # Original casing "Summarize the findings in Q3." must be preserved verbatim
        self.assertEqual(compressed, "Summarize the findings in Q3.")
        self.assertTrue(compressed.startswith("Summarize"))
        self.assertIn("Q3", compressed)

    def test_preserves_sentence_order(self):
        prompt = "Alpha. Beta. Gamma. Alpha. Delta. Beta."
        expected = "Alpha. Beta. Gamma. Delta."
        self.assertEqual(self.compressor.compress(prompt), expected)

    # 6. Prompts Without Duplicates
    def test_prompts_without_duplicates_remain_intact(self):
        prompt = "Step 1: Gather data. Step 2: Clean data. Step 3: Train model."
        compressed = self.compressor.compress(prompt)
        self.assertEqual(compressed, prompt)

    # 7. Multiline Prompts With Newlines
    def test_multiline_prompt_with_newlines(self):
        prompt = "Explain backpropagation.\nExplain backpropagation.\nInclude 2 examples."
        expected = "Explain backpropagation.\nInclude 2 examples."
        self.assertEqual(self.compressor.compress(prompt), expected)

    def test_multiline_prompt_without_trailing_punctuation(self):
        prompt = "Line one\nLine two\nLine one"
        expected = "Line one\nLine two"
        self.assertEqual(self.compressor.compress(prompt), expected)

    def test_paragraph_breaks_preserved(self):
        prompt = "Paragraph one.\n\nParagraph two.\n\nParagraph one."
        expected = "Paragraph one.\n\nParagraph two."
        self.assertEqual(self.compressor.compress(prompt), expected)

    # 8. Non-Removal of Merely Similar Sentences (Phase 4.1B boundary check)
    def test_does_not_remove_merely_similar_sentences(self):
        prompt = (
            "Machine learning allows computers to learn from data. "
            "Machine learning enables computers to learn using data."
        )
        compressed = self.compressor.compress(prompt)
        # Both sentences must be retained because they are not exact duplicates
        self.assertIn("allows computers to learn from data", compressed)
        self.assertIn("enables computers to learn using data", compressed)
        self.assertEqual(compressed, prompt)

    def test_does_not_remove_paraphrased_instructions(self):
        prompt = (
            "Summarize the main points of the report. "
            "Provide a brief summary of the key findings."
        )
        compressed = self.compressor.compress(prompt)
        # Paraphrased sentences differ textually and must both be preserved
        self.assertEqual(compressed, prompt)

    # 9. Modular Interface and Factory Function
    def test_modular_compressor_hierarchy(self):
        self.assertIsInstance(self.compressor, BaseCompressor)
        factory_compressor = get_exact_duplicate_compressor()
        self.assertIsInstance(factory_compressor, ExactDuplicateCompressor)

    def test_convenience_function(self):
        prompt = "Run the tests. Run the tests."
        self.assertEqual(remove_exact_duplicates(prompt), "Run the tests.")

    # 10. Compression With Statistics
    def test_compress_with_stats_metrics(self):
        prompt = "Explain quantum computing. Explain quantum computing."
        result = self.compressor.compress_with_stats(prompt, tokenizer=self.tokenizer)

        self.assertIsInstance(result, DuplicateRemovalResult)
        self.assertEqual(result.original_prompt, prompt)
        self.assertEqual(result.compressed_prompt, "Explain quantum computing.")
        self.assertEqual(result.original_sentence_count, 2)
        self.assertEqual(result.retained_sentence_count, 1)
        self.assertEqual(result.removed_sentence_count, 1)

        self.assertIsNotNone(result.original_token_count)
        self.assertIsNotNone(result.compressed_token_count)
        self.assertIsNotNone(result.tokens_saved)
        self.assertIsNotNone(result.reduction_percentage)

        self.assertGreater(result.original_token_count, result.compressed_token_count)
        self.assertEqual(
            result.tokens_saved,
            result.original_token_count - result.compressed_token_count,
        )
        self.assertGreater(result.reduction_percentage, 0.0)

    def test_compress_with_stats_zero_token_handling(self):
        # Empty input has 0 tokens and should not raise ZeroDivisionError
        result = self.compressor.compress_with_stats("", tokenizer=self.tokenizer)
        self.assertEqual(result.original_token_count, 0)
        self.assertEqual(result.compressed_token_count, 0)
        self.assertEqual(result.tokens_saved, 0)
        self.assertEqual(result.reduction_percentage, 0.0)

        # None input
        result_none = self.compressor.compress_with_stats(None, tokenizer=self.tokenizer)
        self.assertEqual(result_none.original_token_count, 0)
        self.assertEqual(result_none.compressed_token_count, 0)
        self.assertEqual(result_none.tokens_saved, 0)
        self.assertEqual(result_none.reduction_percentage, 0.0)

    def test_custom_tokenizer_compatibility(self):
        dummy_tokenizer = DummyTestTokenizer()
        prompt = "Hello world. Hello world."
        result = self.compressor.compress_with_stats(prompt, tokenizer=dummy_tokenizer)
        self.assertEqual(result.original_token_count, 4)
        self.assertEqual(result.compressed_token_count, 2)
        self.assertEqual(result.tokens_saved, 2)
        self.assertEqual(result.reduction_percentage, 50.0)


class MockEmbeddingModel:
    """Mock SentenceTransformer model returning deterministic normalized vectors for offline unit tests."""

    def __init__(self):
        self.call_count = 0

    def encode(self, sentences: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        self.call_count += 1
        vectors = []
        for s in sentences:
            s_clean = s.strip().lower()
            if "machine learning allows" in s_clean or "explain quantum computing" in s_clean:
                v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif "machine learning enables" in s_clean or "describe quantum computing" in s_clean:
                # Cosine similarity with [1, 0, 0, 0] is ~0.95
                v = np.array([0.95, 0.0, 0.31225, 0.0], dtype=np.float32)
            elif "moderate similarity" in s_clean:
                # Cosine similarity with [1, 0, 0, 0] is ~0.85
                v = np.array([0.85, 0.0, 0.52678, 0.0], dtype=np.float32)
            elif "unrelated" in s_clean or "ocean" in s_clean or "weather" in s_clean:
                # Orthogonal (cosine similarity 0.0)
                v = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            elif "limit the response to 100 words" in s_clean or "limit to 100 words" in s_clean:
                v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif "limit the response to 50 words" in s_clean or "limit to 50 words" in s_clean:
                v = np.array([0.95, 0.0, 0.31225, 0.0], dtype=np.float32)
            elif "include unit tests in python" in s_clean or "include docstrings" in s_clean:
                v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif "do not include unit tests in python" in s_clean or "do not include docstrings" in s_clean:
                v = np.array([0.95, 0.0, 0.31225, 0.0], dtype=np.float32)
            elif "format the output as json" in s_clean:
                v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif "format the output as yaml" in s_clean:
                v = np.array([0.95, 0.0, 0.31225, 0.0], dtype=np.float32)
            elif "deploy the service to aws" in s_clean:
                v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif "deploy the service to azure" in s_clean:
                v = np.array([0.95, 0.0, 0.31225, 0.0], dtype=np.float32)
            elif "explain machine learning" in s_clean:
                v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                h = abs(hash(s_clean)) % 1000
                v = np.array([float(h % 7 + 1), float(h % 5 + 1), float(h % 3 + 1), 1.0], dtype=np.float32)

            if normalize_embeddings:
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
            vectors.append(v)
        return np.array(vectors)


class TestSimilarityRedundancyCompressor(unittest.TestCase):
    """Test suite for Phase 4.1B: SimilarityRedundancyCompressor."""

    def setUp(self):
        self.mock_model = MockEmbeddingModel()
        self.compressor = SimilarityRedundancyCompressor(
            similarity_threshold=0.80,
            model=self.mock_model,
        )
        self.tokenizer = get_tokenizer("cl100k_base")

    # 1. Empty & Whitespace Input
    def test_empty_input(self):
        self.assertEqual(self.compressor.compress(""), "")
        self.assertEqual(self.compressor.compress(None), "")
        self.assertEqual(remove_similar_redundancy("", model=self.mock_model), "")
        self.assertEqual(remove_similar_redundancy(None, model=self.mock_model), "")
        # Model should not be called for empty inputs
        self.assertEqual(self.mock_model.call_count, 0)

    def test_whitespace_only_input(self):
        self.assertEqual(self.compressor.compress("   "), "")
        self.assertEqual(self.compressor.compress("\n\t  \n"), "")
        self.assertEqual(self.mock_model.call_count, 0)

    # 2. Single Sentence Input
    def test_single_sentence(self):
        prompt = "Explain quantum computing in simple terms."
        compressed = self.compressor.compress(prompt)
        self.assertEqual(compressed, prompt)
        # Single sentence should bypass model inference
        self.assertEqual(self.mock_model.call_count, 0)

    # 3. Unrelated Sentences (Below Threshold)
    def test_unrelated_sentences_retained(self):
        prompt = (
            "Machine learning allows computers to learn from data. "
            "The Pacific Ocean is the largest ocean on Earth."
        )
        compressed = self.compressor.compress(prompt)
        # Orthogonal sentences must both be retained
        self.assertEqual(compressed, prompt)
        self.assertGreater(self.mock_model.call_count, 0)

    # 4. Highly Similar Redundant Sentences (Removed)
    def test_highly_similar_redundant_sentences_removed(self):
        prompt = (
            "Machine learning allows computers to learn from data. "
            "Machine learning enables computers to learn using data."
        )
        compressed = self.compressor.compress(prompt)
        # Second sentence is redundant (sim ~0.95 >= 0.80) and has no critical differences
        self.assertEqual(compressed, "Machine learning allows computers to learn from data.")

    def test_exact_duplicates_also_removed(self):
        prompt = "Explain quantum computing. Explain quantum computing."
        compressed = self.compressor.compress(prompt)
        self.assertEqual(compressed, "Explain quantum computing.")

    # 5. Order and Text Preservation
    def test_order_and_original_text_preserved(self):
        prompt = (
            "Explain quantum computing in simple terms. "
            "The Pacific Ocean is the largest ocean on Earth. "
            "Describe quantum computing in simple terms."
        )
        compressed = self.compressor.compress(prompt)
        # First sentence and ocean sentence retained in original order; 3rd removed
        expected = (
            "Explain quantum computing in simple terms. "
            "The Pacific Ocean is the largest ocean on Earth."
        )
        self.assertEqual(compressed, expected)
        # Original casing not rewritten
        self.assertTrue(compressed.startswith("Explain quantum computing in simple terms."))

    # 6. Configurable Threshold
    def test_configurable_threshold(self):
        prompt = "Explain quantum computing. Moderate similarity text."
        # At default threshold=0.80, moderate similarity (~0.85) is redundant
        compressed_default = self.compressor.compress(prompt, threshold=0.80)
        self.assertEqual(compressed_default, "Explain quantum computing.")

        # At strict threshold=0.90, moderate similarity (~0.85) is NOT redundant
        compressed_strict = self.compressor.compress(prompt, threshold=0.90)
        self.assertEqual(compressed_strict, prompt)

    # 7. Conservative Safeguards: Numbers / Quantities
    def test_conservative_safeguard_numbers(self):
        prompt = "Limit the response to 100 words. Limit the response to 50 words."
        result = self.compressor.compress_with_stats(prompt)
        # High similarity (~0.95), but distinct numbers (100 vs 50) prevent removal
        self.assertEqual(result.compressed_prompt, prompt)
        self.assertEqual(result.retained_sentence_count, 2)
        self.assertEqual(len(result.safeguarded_sentences), 1)
        self.assertEqual(result.safeguarded_sentences[0][3], "distinct_numbers_or_quantities")

    # 8. Conservative Safeguards: Negations
    def test_conservative_safeguard_negation(self):
        prompt = "Include unit tests in Python. Do not include unit tests in Python."
        result = self.compressor.compress_with_stats(prompt)
        # High similarity (~0.95), but negation mismatch prevents removal
        self.assertEqual(result.compressed_prompt, prompt)
        self.assertEqual(result.retained_sentence_count, 2)
        self.assertEqual(len(result.safeguarded_sentences), 1)
        self.assertEqual(result.safeguarded_sentences[0][3], "negation_mismatch")

    # 9. Conservative Safeguards: Output Format Constraints
    def test_conservative_safeguard_format(self):
        prompt = "Format the output as JSON. Format the output as YAML."
        result = self.compressor.compress_with_stats(prompt)
        # High similarity (~0.95), but distinct format constraints prevent removal
        self.assertEqual(result.compressed_prompt, prompt)
        self.assertEqual(result.retained_sentence_count, 2)
        self.assertEqual(len(result.safeguarded_sentences), 1)
        self.assertEqual(result.safeguarded_sentences[0][3], "distinct_format_constraint")

    # 10. Conservative Safeguards: Named Entities
    def test_conservative_safeguard_entities(self):
        prompt = "Deploy the service to AWS. Deploy the service to Azure."
        result = self.compressor.compress_with_stats(prompt)
        # High similarity (~0.95), but distinct entities (AWS vs Azure) prevent removal
        self.assertEqual(result.compressed_prompt, prompt)
        self.assertEqual(result.retained_sentence_count, 2)
        self.assertEqual(len(result.safeguarded_sentences), 1)
        self.assertEqual(result.safeguarded_sentences[0][3], "distinct_named_entities")

    # 11. Conservative Safeguards: Additional Substantive Detail
    def test_conservative_safeguard_additional_detail(self):
        prompt = (
            "Explain machine learning. "
            "Explain machine learning with attention mechanisms, transformer architectures, and deep neural networks."
        )
        result = self.compressor.compress_with_stats(prompt)
        # Second sentence adds significant substantive technical detail
        self.assertEqual(result.compressed_prompt, prompt)
        self.assertEqual(result.retained_sentence_count, 2)

    # 12. Mixed Prompts
    def test_mixed_prompt_reduction(self):
        prompt = (
            "Machine learning allows computers to learn from data. "
            "Machine learning enables computers to learn using data. "
            "Limit the response to 100 words. "
            "Limit the response to 50 words. "
            "The Pacific Ocean is the largest ocean on Earth."
        )
        result = self.compressor.compress_with_stats(prompt)
        # 1. Base sentence (retained)
        # 2. Similar paraphrase (removed)
        # 3. 100 words constraint (retained)
        # 4. 50 words constraint (retained due to number safeguard)
        # 5. Ocean sentence (retained due to low similarity)
        expected = (
            "Machine learning allows computers to learn from data. "
            "Limit the response to 100 words. "
            "Limit the response to 50 words. "
            "The Pacific Ocean is the largest ocean on Earth."
        )
        self.assertEqual(result.compressed_prompt, expected)
        self.assertEqual(result.original_sentence_count, 5)
        self.assertEqual(result.retained_sentence_count, 4)
        self.assertEqual(result.removed_sentence_count, 1)

    # 13. Token Metrics & Zero-Token Safe Handling
    def test_compress_with_stats_metrics(self):
        prompt = (
            "Machine learning allows computers to learn from data. "
            "Machine learning enables computers to learn using data."
        )
        result = self.compressor.compress_with_stats(prompt, tokenizer=self.tokenizer)
        self.assertIsInstance(result, SimilarityRemovalResult)
        self.assertEqual(result.original_sentence_count, 2)
        self.assertEqual(result.retained_sentence_count, 1)
        self.assertEqual(result.removed_sentence_count, 1)
        self.assertGreater(result.tokens_saved, 0)
        self.assertGreater(result.reduction_percentage, 0.0)
        self.assertEqual(len(result.removed_sentences), 1)

    def test_compress_with_stats_zero_token_handling(self):
        result = self.compressor.compress_with_stats("", tokenizer=self.tokenizer)
        self.assertEqual(result.original_token_count, 0)
        self.assertEqual(result.compressed_token_count, 0)
        self.assertEqual(result.tokens_saved, 0)
        self.assertEqual(result.reduction_percentage, 0.0)

        result_none = self.compressor.compress_with_stats(None, tokenizer=self.tokenizer)
        self.assertEqual(result_none.original_token_count, 0)
        self.assertEqual(result_none.tokens_saved, 0)
        self.assertEqual(result_none.reduction_percentage, 0.0)

    # 14. Modular Interface and Factory Function
    def test_modular_compressor_hierarchy(self):
        self.assertIsInstance(self.compressor, BaseCompressor)
        factory_comp = get_similarity_redundancy_compressor(model=self.mock_model)
        self.assertIsInstance(factory_comp, SimilarityRedundancyCompressor)

    def test_convenience_function(self):
        prompt = (
            "Machine learning allows computers to learn from data. "
            "Machine learning enables computers to learn using data."
        )
        res = remove_similar_redundancy(prompt, model=self.mock_model)
        self.assertEqual(res, "Machine learning allows computers to learn from data.")


if __name__ == "__main__":
    unittest.main()

