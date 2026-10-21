"""Unit tests for Phase 4.2A: Rule-Based Instruction Compressor."""

import unittest

from src.compressors import (
    BaseCompressor,
    InstructionCompressionResult,
    InstructionRule,
    RuleBasedInstructionCompressor,
    compress_instruction,
    get_instruction_compressor,
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


class TestRuleBasedInstructionCompressor(unittest.TestCase):
    """Test suite for RuleBasedInstructionCompressor."""

    def setUp(self):
        self.compressor = RuleBasedInstructionCompressor()
        self.tokenizer = get_tokenizer("cl100k_base")

    # 1. Empty and Whitespace Inputs
    def test_empty_input(self):
        self.assertEqual(self.compressor.compress(""), "")
        self.assertEqual(self.compressor.compress(None), "")
        self.assertEqual(compress_instruction(""), "")
        self.assertEqual(compress_instruction(None), "")

    def test_whitespace_only_input(self):
        self.assertEqual(self.compressor.compress("   "), "")
        self.assertEqual(self.compressor.compress("\n\t  \n"), "")
        self.assertEqual(compress_instruction("   \n   "), "")

    # 2. Unchanged Prompts (Already concise, questions, or non-instruction text)
    def test_already_concise_action_verbs(self):
        prompt = "Explain photosynthesis in simple terms."
        self.assertEqual(self.compressor.compress(prompt), prompt)

        prompt2 = "Summarize the meeting notes into 3 bullets."
        self.assertEqual(self.compressor.compress(prompt2), prompt2)

    def test_direct_factual_questions(self):
        prompt = "What is the capital of Australia?"
        self.assertEqual(self.compressor.compress(prompt), prompt)

        prompt2 = "How do neural networks learn representations?"
        self.assertEqual(self.compressor.compress(prompt2), prompt2)

    def test_code_generation_prompt_without_verbose_request(self):
        prompt = "Write a Python script that calculates Fibonacci numbers."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    # 3. Individual Replacement Rules
    def test_rule_explain_provide_explanation(self):
        p1 = "Please provide an explanation of quantum entanglement."
        self.assertEqual(self.compressor.compress(p1), "Explain quantum entanglement.")

        p2 = "Provide an explanation of the greenhouse effect."
        self.assertEqual(self.compressor.compress(p2), "Explain the greenhouse effect.")

        p3 = "Kindly give an explanation of Bayes' theorem."
        self.assertEqual(self.compressor.compress(p3), "Explain Bayes' theorem.")

    def test_rule_summarize_provide_summary(self):
        p1 = "Please provide a summary of the quarterly financial earnings."
        self.assertEqual(self.compressor.compress(p1), "Summarize the quarterly financial earnings.")

        p2 = "Give a summary of chapter five."
        self.assertEqual(self.compressor.compress(p2), "Summarize chapter five.")

    def test_rule_describe_provide_description(self):
        p1 = "Please provide a description of the microservice architecture."
        self.assertEqual(self.compressor.compress(p1), "Describe the microservice architecture.")

        p2 = "Give a description of the observed bug."
        self.assertEqual(self.compressor.compress(p2), "Describe the observed bug.")

    def test_rule_analyze_conduct_analysis(self):
        p1 = "Please conduct an analysis of customer retention data."
        self.assertEqual(self.compressor.compress(p1), "Analyze customer retention data.")

        p2 = "Perform an analysis of the benchmark figures."
        self.assertEqual(self.compressor.compress(p2), "Analyze the benchmark figures.")

        p3 = "Provide an analysis of the memory leak."
        self.assertEqual(self.compressor.compress(p3), "Analyze the memory leak.")

    def test_rule_compare_make_comparison(self):
        p1 = "Please make a comparison between PostgreSQL and MongoDB."
        self.assertEqual(self.compressor.compress(p1), "Compare PostgreSQL and MongoDB.")

        p2 = "Provide a comparison of REST and GraphQL."
        self.assertEqual(self.compressor.compress(p2), "Compare REST and GraphQL.")

    def test_rule_define_provide_definition(self):
        p1 = "Please provide a definition of idempotency in distributed systems."
        self.assertEqual(self.compressor.compress(p1), "Define idempotency in distributed systems.")

        p2 = "Give a definition of semantic entropy."
        self.assertEqual(self.compressor.compress(p2), "Define semantic entropy.")

    def test_rule_clarify_provide_clarification(self):
        p1 = "Please provide clarification on the refund policy."
        self.assertEqual(self.compressor.compress(p1), "Clarify the refund policy.")

        p2 = "Give clarification regarding the breaking changes."
        self.assertEqual(self.compressor.compress(p2), "Clarify the breaking changes.")

    def test_rule_list_provide_list(self):
        p1 = "Please provide a list of all European capitals."
        self.assertEqual(self.compressor.compress(p1), "List all European capitals.")

        p2 = "Give a list of available command line flags."
        self.assertEqual(self.compressor.compress(p2), "List available command line flags.")

    def test_rule_modal_provide_explanation(self):
        p1 = "Could you please provide an explanation of general relativity?"
        self.assertEqual(self.compressor.compress(p1), "Explain general relativity?")

        p2 = "Can you provide an explanation of how transformers work?"
        self.assertEqual(self.compressor.compress(p2), "Explain how transformers work?")

    def test_rule_modal_provide_summary(self):
        p1 = "Could you please provide a summary of this document?"
        self.assertEqual(self.compressor.compress(p1), "Summarize this document?")

        p2 = "Can you provide a summary of the article?"
        self.assertEqual(self.compressor.compress(p2), "Summarize the article?")

    def test_rule_modal_provide_description(self):
        p = "Could you please provide a description of the system workflow?"
        self.assertEqual(self.compressor.compress(p), "Describe the system workflow?")

    def test_rule_modal_provide_list(self):
        p = "Could you please provide a list of prime numbers under 50?"
        self.assertEqual(self.compressor.compress(p), "List prime numbers under 50?")

    def test_rule_modal_direct_verb(self):
        p1 = "Could you please explain how public key cryptography works?"
        self.assertEqual(self.compressor.compress(p1), "Explain how public key cryptography works?")

        p2 = "Can you please summarize the key findings?"
        self.assertEqual(self.compressor.compress(p2), "Summarize the key findings?")

    def test_rule_desire_direct_verb(self):
        p1 = "I would like you to explain the trade-offs of sharding."
        self.assertEqual(self.compressor.compress(p1), "Explain the trade-offs of sharding.")

        p2 = "I want you to summarize this quarterly report."
        self.assertEqual(self.compressor.compress(p2), "Summarize this quarterly report.")

        p3 = "I need you to list the prerequisites."
        self.assertEqual(self.compressor.compress(p3), "List the prerequisites.")

    def test_rule_desire_provide_explanation(self):
        p = "I would like you to provide an explanation of backpropagation."
        self.assertEqual(self.compressor.compress(p), "Explain backpropagation.")

    def test_rule_desire_provide_summary(self):
        p = "I want you to provide a summary of the security audit."
        self.assertEqual(self.compressor.compress(p), "Summarize the security audit.")

    def test_rule_desire_provide_description(self):
        p = "I need you to provide a description of the API endpoints."
        self.assertEqual(self.compressor.compress(p), "Describe the API endpoints.")

    def test_rule_desire_provide_list(self):
        p = "I would like you to provide a list of required libraries."
        self.assertEqual(self.compressor.compress(p), "List required libraries.")

    # 4. Preservation of Meaningful Modifiers (Safeguard)
    def test_modifiers_preserved_detailed(self):
        prompt = "Please provide a detailed explanation of the consensus protocol."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_modifiers_preserved_brief(self):
        prompt = "Provide a brief explanation of cellular respiration."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_modifiers_preserved_step_by_step(self):
        prompt = "Please provide a step-by-step explanation of the quicksort algorithm."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_modifiers_preserved_comprehensive(self):
        prompt = "Please provide a comprehensive explanation of antitrust law."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_modifiers_preserved_in_depth(self):
        prompt = "Provide an in-depth explanation of compiler optimization passes."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_modifiers_preserved_list_format(self):
        p1 = "Please provide a numbered list of action items."
        self.assertEqual(self.compressor.compress(p1), p1)

        p2 = "Provide a bulleted list of advantages and disadvantages."
        self.assertEqual(self.compressor.compress(p2), p2)

    # 5. Negation Safeguards (Must leave text unchanged)
    def test_negation_do_not_provide(self):
        prompt = "Please do not provide an explanation of the plot twist."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_negation_never_provide(self):
        prompt = "Never provide an explanation of internal company secrets."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_negation_avoid_providing(self):
        prompt = "Avoid providing an explanation of personal data."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_negation_without_providing(self):
        prompt = "Write code without providing an explanation of the syntax."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    # 6. Subordinating Conjunctions and Prepositional Contexts (Must leave unchanged)
    def test_conditional_if_clause(self):
        prompt = "If you provide an explanation of the issue, keep it brief."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_instead_of_prepositional_clause(self):
        prompt = "Instead of providing an explanation of the problem, write a patch."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_before_dependent_clause(self):
        prompt = "Before providing an explanation of the findings, review the source data."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_by_prepositional_clause(self):
        prompt = "By providing an explanation of the workflow, you assisted the team."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_thank_you_for_prepositional_clause(self):
        prompt = "Thank you for providing an explanation of the error."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    # 7. Third-Person Subjects / Descriptive Statements (Must leave unchanged)
    def test_third_person_subject_documentation(self):
        prompt = "This documentation provides an explanation of authentication mechanisms."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_third_person_subject_author(self):
        prompt = "The author provides a summary of the historical evidence."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_third_person_subject_guide(self):
        prompt = "Each guide provides a description of a specific software design pattern."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    # 8. Protection of Quoted Text and Code Blocks
    def test_phrase_inside_double_quotes_unchanged(self):
        prompt = 'The user input was "please provide an explanation of" and needs review.'
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_phrase_inside_code_backticks_unchanged(self):
        prompt = "Run the script with `please provide an explanation of` as an argument."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_phrase_inside_fenced_code_block_unchanged(self):
        prompt = "Here is the test:\n```\nplease provide an explanation of\n```"
        self.assertEqual(self.compressor.compress(prompt), prompt)

    # 9. Explicit Constraints, Numbers, and Output Format Preservation
    def test_number_and_format_preserved(self):
        prompt = "Please provide an explanation of the 3 primary laws in JSON format."
        compressed = self.compressor.compress(prompt)
        self.assertEqual(compressed, "Explain the 3 primary laws in JSON format.")
        self.assertIn("3", compressed)
        self.assertIn("JSON", compressed)

    def test_entity_and_length_constraint_preserved(self):
        prompt = "Please provide a summary of the Python 3.12 release notes in under 100 words."
        compressed = self.compressor.compress(prompt)
        self.assertEqual(compressed, "Summarize the Python 3.12 release notes in under 100 words.")
        self.assertIn("Python 3.12", compressed)
        self.assertIn("under 100 words", compressed)

    # 10. Multiple Verbose Instructions in a Single Prompt
    def test_multiple_instructions_in_one_prompt(self):
        prompt = (
            "Please provide an explanation of quicksort. "
            "Also, please provide a summary of its worst-case complexity."
        )
        expected = (
            "Explain quicksort. "
            "Also, summarize its worst-case complexity."
        )
        self.assertEqual(self.compressor.compress(prompt), expected)

    # 11. Mid-Sentence Casing
    def test_mid_sentence_casing(self):
        prompt = "First, please provide an explanation of recursion."
        self.assertEqual(self.compressor.compress(prompt), "First, explain recursion.")

    # 12. compress_with_stats and Tokenizer Integration
    def test_compress_with_stats_tiktoken(self):
        prompt = "Please provide an explanation of quantum entanglement."
        result = self.compressor.compress_with_stats(prompt, tokenizer=self.tokenizer)

        self.assertIsInstance(result, InstructionCompressionResult)
        self.assertEqual(result.compressed_prompt, "Explain quantum entanglement.")
        self.assertIn("EXPLAIN_PROVIDE_EXPLANATION", result.rules_applied)
        self.assertEqual(len(result.replacements), 1)
        self.assertGreater(result.original_token_count, result.compressed_token_count)
        self.assertEqual(
            result.tokens_saved,
            result.original_token_count - result.compressed_token_count,
        )
        self.assertGreater(result.reduction_percentage, 0.0)

    def test_compress_with_stats_dummy_tokenizer(self):
        dummy = DummyTestTokenizer()
        prompt = "Please provide an explanation of quantum entanglement."
        result = self.compressor.compress_with_stats(prompt, tokenizer=dummy)

        self.assertEqual(result.original_token_count, 7)
        self.assertEqual(result.compressed_token_count, 3)
        self.assertEqual(result.tokens_saved, 4)
        self.assertAlmostEqual(result.reduction_percentage, (4 / 7) * 100.0, places=2)

    def test_compress_with_stats_zero_tokens_safe(self):
        dummy = DummyTestTokenizer()
        result = self.compressor.compress_with_stats("", tokenizer=dummy)

        self.assertEqual(result.compressed_prompt, "")
        self.assertEqual(result.original_token_count, 0)
        self.assertEqual(result.compressed_token_count, 0)
        self.assertEqual(result.tokens_saved, 0)
        self.assertEqual(result.reduction_percentage, 0.0)

    # 13. Interface and Factory Functions
    def test_factory_and_convenience_functions(self):
        compressor = get_instruction_compressor()
        self.assertIsInstance(compressor, RuleBasedInstructionCompressor)
        self.assertIsInstance(compressor, BaseCompressor)

        res = compress_instruction("Please provide an explanation of gravity.")
        self.assertEqual(res, "Explain gravity.")

    # 14. Targeted Regression Tests: Negation Safeguards
    def test_regression_negation_cannot_and_should_not(self):
        p1 = "You cannot provide an explanation of proprietary algorithms."
        self.assertEqual(self.compressor.compress(p1), p1)

        p2 = "You should not provide a summary of confidential data."
        self.assertEqual(self.compressor.compress(p2), p2)

    def test_regression_negation_must_not_and_wont(self):
        p1 = "You must not provide a list of internal server passwords."
        self.assertEqual(self.compressor.compress(p1), p1)

        p2 = "We won't provide an explanation of the private repository."
        self.assertEqual(self.compressor.compress(p2), p2)

    def test_regression_negation_contractions(self):
        p1 = "Please don't provide a description of the unreleased product."
        self.assertEqual(self.compressor.compress(p1), p1)

        p2 = "You can't provide a summary without explicit authorization."
        self.assertEqual(self.compressor.compress(p2), p2)

    def test_regression_negation_safeguard_prevents_loss(self):
        prompt = "Make sure to not give an explanation of user credentials."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    # 15. Targeted Regression Tests: Conditional and Subordinate Instructions
    def test_regression_conditional_unless(self):
        prompt = "Unless instructed otherwise, please provide an explanation of default settings."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_regression_conditional_in_order_to(self):
        prompt = "In order to resolve the bug, please provide a description of the symptoms."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_regression_conditional_so_that(self):
        prompt = "Refactor the module so that you can provide an explanation of the logic."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_regression_conditional_whether(self):
        prompt = "Whether or not it succeeds, please provide a summary of the metrics."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_regression_conditional_while_and_since(self):
        p1 = "While reviewing the code, please provide a list of security risks."
        self.assertEqual(self.compressor.compress(p1), p1)

        p2 = "Since the cluster crashed, please provide an explanation of the root cause."
        self.assertEqual(self.compressor.compress(p2), p2)

    # 16. Targeted Regression Tests: Numerical Limits and Constraints
    def test_regression_numerical_limits_word_and_digit_preservation(self):
        prompt = "Please provide an explanation of the 3 primary architectural tenets in under 500 words."
        compressed = self.compressor.compress(prompt)
        self.assertEqual(compressed, "Explain the 3 primary architectural tenets in under 500 words.")
        self.assertIn("3", compressed)
        self.assertIn("500", compressed)

    def test_regression_numerical_range_preservation(self):
        prompt = "Please provide a list of between 5 and 10 relevant research publications."
        compressed = self.compressor.compress(prompt)
        self.assertEqual(compressed, "List between 5 and 10 relevant research publications.")
        self.assertIn("5", compressed)
        self.assertIn("10", compressed)
        self.assertIn("between", compressed)

    def test_regression_numerical_spelled_numbers_preservation(self):
        prompt = "Please provide a summary of the two competing proposals in four distinct paragraphs."
        compressed = self.compressor.compress(prompt)
        self.assertEqual(compressed, "Summarize the two competing proposals in four distinct paragraphs.")
        self.assertIn("two", compressed)
        self.assertIn("four", compressed)

    def test_regression_numerical_thresholds_with_constraints(self):
        prompt = "Please provide a list of top 10 items with at least 5 positive reviews and at most 2 negative reviews."
        compressed = self.compressor.compress(prompt)
        self.assertEqual(compressed, "List top 10 items with at least 5 positive reviews and at most 2 negative reviews.")
        self.assertIn("10", compressed)
        self.assertIn("5", compressed)
        self.assertIn("2", compressed)
        self.assertIn("at least", compressed)
        self.assertIn("at most", compressed)

    # 17. Targeted Regression Tests: Required Examples
    def test_regression_required_examples_attached_clause(self):
        prompt = "Please provide an explanation of dynamic programming, including 2 working examples."
        compressed = self.compressor.compress(prompt)
        self.assertEqual(compressed, "Explain dynamic programming, including 2 working examples.")
        self.assertIn("including 2 working examples", compressed)

    def test_regression_required_examples_intervening_phrase_unchanged(self):
        prompt = "Please provide an explanation with examples of polymorphic dispatch."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_regression_required_examples_subsequent_sentence(self):
        prompt = "Please provide an explanation of gradient descent. Include at least two concrete examples in Python."
        compressed = self.compressor.compress(prompt)
        self.assertEqual(
            compressed,
            "Explain gradient descent. Include at least two concrete examples in Python.",
        )
        self.assertIn("Include at least two concrete examples in Python.", compressed)

    def test_regression_required_examples_runnable_code_sample(self):
        prompt = "Please provide a description of the singleton pattern with a runnable code example."
        compressed = self.compressor.compress(prompt)
        self.assertEqual(compressed, "Describe the singleton pattern with a runnable code example.")
        self.assertIn("with a runnable code example", compressed)

    # 18. Targeted Regression Tests: Meaningful Modifiers
    def test_regression_modifiers_detailed_preserved_safely(self):
        prompt = "Please provide a detailed explanation of the consensus protocol."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_regression_modifiers_high_level_preserved(self):
        prompt = "Please provide a high-level summary of the distributed architecture."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_regression_modifiers_exhaustive_preserved(self):
        prompt = "Could you please provide an exhaustive list of all deprecated methods?"
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_regression_modifiers_beginner_friendly_preserved(self):
        prompt = "Please provide a beginner-friendly explanation of asynchronous programming."
        self.assertEqual(self.compressor.compress(prompt), prompt)

    def test_regression_modifiers_concise_and_thorough_preserved(self):
        p1 = "Please provide a concise description of the reproduction steps."
        self.assertEqual(self.compressor.compress(p1), p1)

        p2 = "Provide a thorough analysis of the failure logs."
        self.assertEqual(self.compressor.compress(p2), p2)


if __name__ == "__main__":
    unittest.main()
