"""Unit tests for Phase 4.2B: Constraint Preservation and Validation."""

import re
import unittest

from src.compressors.instruction import (
    InstructionRule,
    RuleBasedInstructionCompressor,
    compress_instruction,
    get_instruction_compressor,
)
from src.validators.constraint_validator import (
    BasePromptValidator,
    InstructionConstraintValidator,
    ValidationIssue,
    ValidationResult,
    get_constraint_validator,
    validate_instruction_compression,
)


class TestInstructionConstraintValidator(unittest.TestCase):
    """Test suite for InstructionConstraintValidator and compressor validation integration."""

    def setUp(self):
        self.validator = InstructionConstraintValidator()

    # 1. Empty and Whitespace Inputs
    def test_both_empty_inputs(self):
        result = self.validator.validate("", "")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.confidence, "high")
        self.assertEqual(len(result.issues), 0)

        result_none = self.validator.validate(None, None)
        self.assertTrue(result_none.is_valid)

        result_ws = self.validator.validate("   \n\t", "  ")
        self.assertTrue(result_ws.is_valid)

    def test_empty_original_with_nonempty_compressed(self):
        result = self.validator.validate("", "Explain quantum computing.")
        self.assertFalse(result.is_valid)
        self.assertIn("uncertainty", result.failed_categories)
        self.assertTrue(any(i.category == "uncertainty" and i.severity == "error" for i in result.issues))

    # 2. Unchanged Prompts
    def test_identical_prompts_pass_cleanly(self):
        prompt = "Explain photosynthesis in simple terms."
        result = self.validator.validate(prompt, prompt)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.confidence, "high")
        self.assertEqual(len(result.issues), 0)
        self.assertIn("negation", result.preserved_categories)
        self.assertIn("technical_term", result.preserved_categories)

    def test_identical_complex_prompt(self):
        prompt = "If the file exists, do not overwrite it. Provide 2 JSON examples for beginners."
        result = self.validator.validate(prompt, prompt)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.issues), 0)

    # 3. Preserved Constraints (Rule-Based Shortening Scenarios)
    def test_preserved_verbose_explanation(self):
        original = "Please provide an explanation of how transformer attention mechanisms work."
        compressed = "Explain how transformer attention mechanisms work."
        result = self.validator.validate(original, compressed)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.confidence, "high")
        self.assertEqual(len(result.issues), 0)

    def test_preserved_verbose_summary(self):
        original = "Please provide a summary of the quarterly financial earnings report."
        compressed = "Summarize the quarterly financial earnings report."
        result = self.validator.validate(original, compressed)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.confidence, "high")

    def test_preserved_desire_preamble_with_acronyms(self):
        original = "I would like you to explain the key differences between SQL and NoSQL databases."
        compressed = "Explain the key differences between SQL and NoSQL databases."
        result = self.validator.validate(original, compressed)
        self.assertTrue(result.is_valid)
        self.assertIn("SQL", result.details["entity"]["original_entities"])
        self.assertIn("NoSQL", result.details["entity"]["original_entities"])

    def test_preserved_number_and_format_constraints(self):
        original = "Please provide an explanation of the 3 primary laws of robotics in JSON format."
        compressed = "Explain the 3 primary laws of robotics in JSON format."
        result = self.validator.validate(original, compressed)
        self.assertTrue(result.is_valid)
        self.assertIn("format", result.preserved_categories)
        self.assertIn("numerical", result.preserved_categories)

    def test_preserved_audience_and_tone(self):
        original = "Could you please explain quantum annealing for beginners in simple terms?"
        compressed = "Explain quantum annealing for beginners in simple terms?"
        result = self.validator.validate(original, compressed)
        self.assertTrue(result.is_valid)
        self.assertIn("audience_tone", result.preserved_categories)

    def test_preserved_examples_requirement(self):
        original = "Please provide a description of bubble sort with 2 code examples."
        compressed = "Describe bubble sort with 2 code examples."
        result = self.validator.validate(original, compressed)
        self.assertTrue(result.is_valid)
        self.assertIn("examples", result.preserved_categories)
        self.assertIn("numerical", result.preserved_categories)

    def test_preserved_entities_and_technical_terms(self):
        original = "Please provide an explanation of Bayes' theorem and Euler's formula in Australia."
        compressed = "Explain Bayes' theorem and Euler's formula in Australia."
        result = self.validator.validate(original, compressed)
        self.assertTrue(result.is_valid)
        self.assertIn("entity", result.preserved_categories)
        self.assertIn("technical_term", result.preserved_categories)

    # 4. Deliberately Altered Constraints (Must Fail Validation)
    def test_deliberately_dropped_negation_fails(self):
        original = "Please do not provide an explanation of the plot twist."
        compressed = "Explain the plot twist."
        result = self.validator.validate(original, compressed)
        self.assertFalse(result.is_valid)
        self.assertIn("negation", result.failed_categories)
        self.assertTrue(any(i.category == "negation" for i in result.issues))

    def test_deliberately_introduced_negation_fails(self):
        original = "Please provide an explanation of the theorem."
        compressed = "Do not explain the theorem."
        result = self.validator.validate(original, compressed)
        self.assertFalse(result.is_valid)
        self.assertIn("negation", result.failed_categories)

    def test_deliberately_dropped_numerical_constraint_fails(self):
        original = "Please provide an explanation of the 3 primary laws of robotics."
        compressed = "Explain the primary laws of robotics."
        result = self.validator.validate(original, compressed)
        self.assertFalse(result.is_valid)
        self.assertIn("numerical", result.failed_categories)
        self.assertTrue(any("3" in str(i.original_evidence) for i in result.issues))

    def test_deliberately_altered_word_number_fails(self):
        original = "Summarize the report in three bullet points."
        compressed = "Summarize the report in two bullet points."
        result = self.validator.validate(original, compressed)
        self.assertFalse(result.is_valid)
        self.assertIn("numerical", result.failed_categories)

    def test_deliberately_dropped_conditional_fails(self):
        original = "If the server is down, please provide a summary of the incident."
        compressed = "Summarize the incident."
        result = self.validator.validate(original, compressed)
        self.assertFalse(result.is_valid)
        self.assertIn("conditional", result.failed_categories)
        self.assertTrue(any(i.category == "conditional" for i in result.issues))

    def test_deliberately_dropped_format_fails(self):
        original = "Please provide an explanation of the protocol in JSON format."
        compressed = "Explain the protocol."
        result = self.validator.validate(original, compressed)
        self.assertFalse(result.is_valid)
        self.assertIn("format", result.failed_categories)

    def test_deliberately_dropped_table_format_fails(self):
        original = "Please provide a comparison of SQL and NoSQL as a table."
        compressed = "Compare SQL and NoSQL."
        result = self.validator.validate(original, compressed)
        self.assertFalse(result.is_valid)
        self.assertIn("format", result.failed_categories)

    def test_deliberately_dropped_audience_tone_fails(self):
        original = "Please explain quantum computing for beginners in simple terms."
        compressed = "Explain quantum computing."
        result = self.validator.validate(original, compressed)
        self.assertFalse(result.is_valid)
        self.assertIn("audience_tone", result.failed_categories)

    def test_deliberately_dropped_examples_requirement_fails(self):
        original = "Please provide an explanation of recursion with 2 code examples."
        compressed = "Explain recursion."
        result = self.validator.validate(original, compressed)
        self.assertFalse(result.is_valid)
        self.assertIn("examples", result.failed_categories)

    def test_deliberately_dropped_entity_fails(self):
        original = "Please provide an explanation of AWS architecture in Australia."
        compressed = "Explain architecture."
        result = self.validator.validate(original, compressed)
        self.assertFalse(result.is_valid)
        self.assertIn("entity", result.failed_categories)

    def test_deliberately_dropped_technical_terms_fails(self):
        original = "Please provide an analysis of the quicksort worst-case complexity."
        compressed = "Analyze sorting behavior."
        result = self.validator.validate(original, compressed)
        self.assertFalse(result.is_valid)
        self.assertIn("technical_term", result.failed_categories)

    # 5. False-Positive Risks
    def test_polite_boilerplate_removal_is_not_flagged_as_entity_or_technical_drop(self):
        original = "Could you please kindly provide an explanation of photosynthesis?"
        compressed = "Explain photosynthesis?"
        result = self.validator.validate(original, compressed)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.issues), 0)

    def test_capitalized_sentence_starter_is_not_flagged_as_entity(self):
        original = "Please provide a definition of latency."
        compressed = "Define latency."
        result = self.validator.validate(original, compressed)
        self.assertTrue(result.is_valid)
        # "Please" should not be treated as a missing named entity
        self.assertNotIn("Please", result.details["entity"]["missing_entities"])

    def test_word_number_to_digit_equivalence(self):
        original = "List three advantages of unit testing."
        compressed = "List 3 advantages of unit testing."
        result = self.validator.validate(original, compressed)
        self.assertTrue(result.is_valid)
        self.assertNotIn("numerical", result.failed_categories)

    # 6. Uncertainty Flagging
    def test_extreme_reduction_flags_uncertainty_warning(self):
        original = "Could you please please kindly provide an explanation of what this is and how it is done?"
        compressed = "Explain."
        result = self.validator.validate(original, compressed)
        # Extreme reduction produces uncertainty warning and uncertain confidence
        self.assertTrue(any(i.category == "uncertainty" and i.severity == "warning" for i in result.issues))
        self.assertIn("uncertainty", result.uncertain_categories)
        self.assertEqual(result.confidence, "uncertain")

    def test_dropped_qualifying_modifier_flags_uncertainty(self):
        original = "Please explain the consensus protocol step by step."
        compressed = "Explain the consensus protocol."
        result = self.validator.validate(original, compressed)
        self.assertTrue(any("step by step" in str(i.original_evidence) for i in result.issues))

    def test_dropped_qualifying_modifier_thoroughly_flags_uncertainty(self):
        original = "Please explain the consensus protocol thoroughly."
        compressed = "Explain the consensus protocol."
        result = self.validator.validate(original, compressed)
        self.assertTrue(any("thoroughly" in str(i.original_evidence) for i in result.issues))
        self.assertTrue(any(i.category == "uncertainty" and i.severity == "warning" for i in result.issues))

    def test_reduction_at_exactly_70_percent_does_not_flag_uncertainty(self):
        # 100 chars original, 30 chars compressed -> exactly 70.0% reduction (strict > 0.70 threshold)
        orig = "a" * 100
        comp = "a" * 30
        issues, info = self.validator._check_uncertainty(orig, comp)
        self.assertEqual(len(issues), 0)
        self.assertEqual(info["warning_count"], 0)

    def test_reduction_below_70_percent_does_not_flag_uncertainty(self):
        # 100 chars original, 31 chars compressed -> 69.0% reduction (< 70%)
        orig = "a" * 100
        comp = "a" * 31
        issues, info = self.validator._check_uncertainty(orig, comp)
        self.assertEqual(len(issues), 0)
        self.assertEqual(info["warning_count"], 0)

        # Realistic prompt pair with below 70% reduction
        original = "Please provide an explanation of how transformer attention mechanisms work."
        compressed = "Explain how transformer attention mechanisms work."
        result = self.validator.validate(original, compressed)
        self.assertNotIn("uncertainty", result.uncertain_categories)
        self.assertFalse(any(i.category == "uncertainty" for i in result.issues))

    def test_reduction_with_short_prompt_below_minimum_length_does_not_flag_uncertainty(self):
        # Original length <= 30 chars: even > 70% reduction does not flag uncertainty warning
        orig = "a" * 20
        comp = "a" * 4  # 80% reduction, but orig_len <= 30
        issues, info = self.validator._check_uncertainty(orig, comp)
        self.assertEqual(len(issues), 0)
        self.assertEqual(info["warning_count"], 0)

    # 7. Compressor Integration
    def test_compressor_without_validator_backward_compatible(self):
        compressor = RuleBasedInstructionCompressor()
        prompt = "Please provide an explanation of how transformer attention mechanisms work."
        res = compressor.compress_with_stats(prompt)
        self.assertEqual(res.compressed_prompt, "Explain how transformer attention mechanisms work.")
        self.assertIsNone(res.validation_result)

    def test_compressor_with_validator_attaches_result(self):
        compressor = RuleBasedInstructionCompressor(validator=self.validator)
        prompt = "Please provide an explanation of how transformer attention mechanisms work."
        res = compressor.compress_with_stats(prompt)
        self.assertEqual(res.compressed_prompt, "Explain how transformer attention mechanisms work.")
        self.assertIsNotNone(res.validation_result)
        self.assertTrue(res.validation_result.is_valid)
        self.assertEqual(res.validation_result.confidence, "high")

    def test_compressor_rejects_candidate_on_validation_failure(self):
        # Create a deliberately faulty rule that strips constraints
        faulty_rule = InstructionRule(
            rule_id="FAULTY_STRIP_CONSTRAINTS",
            description="Deliberately strips output constraints for testing rejection.",
            pattern=re.compile(r"Please provide an explanation of.*?in JSON format\.", re.IGNORECASE),
            replacement="Explain.",
        )
        compressor_reject = RuleBasedInstructionCompressor(
            rules=[faulty_rule],
            validator=self.validator,
            reject_on_validation_failure=True,
        )
        prompt = "Please provide an explanation of the 3 primary laws of robotics in JSON format."
        res = compressor_reject.compress_with_stats(prompt)

        # Candidate should be rejected because "3" and "JSON format" were stripped!
        self.assertEqual(res.compressed_prompt, prompt)  # Reverted to original prompt!
        self.assertEqual(res.rules_applied, [])
        self.assertEqual(res.replacements, [])
        self.assertIsNotNone(res.validation_result)
        self.assertFalse(res.validation_result.is_valid)

    def test_compress_convenience_functions(self):
        prompt = "Please provide an explanation of quantum entanglement."
        res = compress_instruction(prompt, validator=self.validator, reject_on_validation_failure=True)
        self.assertEqual(res, "Explain quantum entanglement.")

        comp_inst = get_instruction_compressor(validator=self.validator, reject_on_validation_failure=True)
        self.assertTrue(comp_inst.reject_on_validation_failure)
        self.assertIsNotNone(comp_inst.validator)

    def test_to_dict_serialization(self):
        result = validate_instruction_compression(
            "Please provide an explanation in JSON format.",
            "Explain in JSON format.",
        )
        as_dict = result.to_dict()
        self.assertIsInstance(as_dict, dict)
        self.assertTrue(as_dict["is_valid"])
        self.assertIn("confidence", as_dict)
        self.assertIn("preserved_categories", as_dict)


if __name__ == "__main__":
    unittest.main()
