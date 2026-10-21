"""Phase 4.2B Experiment: Constraint Preservation and Validation in Instruction Compression.

Evaluates RuleBasedInstructionCompressor integrated with InstructionConstraintValidator
across representative instruction compression scenarios using the cl100k_base tokenizer.
Compares original vs. compressed prompts, validation outcomes, preserved/failed categories,
and token reduction metrics. Handles zero-token and empty prompts safely.
"""

import re
from typing import List, Tuple

from src.compressors.instruction import (
    InstructionRule,
    RuleBasedInstructionCompressor,
)
from src.utils.tokenizer import get_tokenizer
from src.validators.constraint_validator import InstructionConstraintValidator


def run_experiment() -> None:
    tokenizer = get_tokenizer("cl100k_base")
    validator = InstructionConstraintValidator()

    # Standard compressor with validation integration and rejection enabled
    compressor = RuleBasedInstructionCompressor(
        validator=validator,
        reject_on_validation_failure=True,
    )

    test_scenarios: List[Tuple[str, str, str]] = [
        # 1. Standard Verbose Explanation
        (
            "Verbose Explanation Request",
            "Please provide an explanation of how transformer attention mechanisms work.",
            "Standard verbose phrasing; preserves technical terms and intent.",
        ),
        # 2. Verbose Summary Request
        (
            "Verbose Summary Request",
            "Please provide a summary of the quarterly financial earnings report.",
            "Standard deverbal phrase; preserves financial domain intent.",
        ),
        # 3. Desire Preamble with Technical Entities
        (
            "Desire Preamble with Acronym Entities",
            "I would like you to explain the key differences between SQL and NoSQL databases.",
            "Shortens desire preamble while preserving SQL/NoSQL entities.",
        ),
        # 4. Numerical and Output Format Constraints
        (
            "Numerical and Format Constraints Preserved",
            "Please provide an explanation of the 3 primary laws of robotics in JSON format.",
            "Shortens request while preserving numerical '3' and 'JSON' format constraint.",
        ),
        # 5. Audience and Tone Modifiers
        (
            "Audience and Tone Modifiers Preserved",
            "Could you please explain quantum annealing for beginners in simple terms?",
            "Converts polite modal to direct verb while preserving audience and tone.",
        ),
        # 6. Example Requirement
        (
            "Required Examples Preserved",
            "Please provide a description of bubble sort with 2 code examples.",
            "Preserves example requirement and quantity count.",
        ),
        # 7. Named Entities and Technical Vocabulary
        (
            "Named Entities and Technical Terms Preserved",
            "Please provide an explanation of Bayes' theorem and Euler's formula in Australia.",
            "Preserves proper nouns and mathematical entities.",
        ),
        # 8. Negated Instruction (Safeguard)
        (
            "Negated Instruction Clause (Safeguard Unchanged)",
            "Please do not provide an explanation of the plot twist.",
            "Negative constraint safeguard prevents shortening; prompt retained unchanged.",
        ),
        # 9. Conditional Clause (Safeguard)
        (
            "Conditional Clause (Safeguard Unchanged)",
            "If the server is down, please provide a summary of the incident.",
            "Conditional clause safeguard prevents shortening; prompt retained unchanged.",
        ),
        # 10. Already Concise Prompt
        (
            "Already Concise Action Verb",
            "Explain quantum annealing in simple terms.",
            "Already concise instruction verb; no redundant preamble to compress.",
        ),
        # 11. Empty Input (Zero-Token Safe Handling)
        (
            "Empty Input (Zero-Token Safe)",
            "",
            "Safe handling of empty string input without crashing.",
        ),
    ]

    print("=" * 80)
    print("PHASE 4.2B: CONSTRAINT PRESERVATION AND VALIDATION EXPERIMENTAL EVALUATION")
    print("=" * 80)

    for idx, (label, prompt, note) in enumerate(test_scenarios, 1):
        result = compressor.compress_with_stats(
            prompt,
            tokenizer=tokenizer,
            validator=validator,
            reject_on_validation_failure=True,
        )

        val = result.validation_result
        is_valid_str = str(val.is_valid) if val is not None else "N/A"
        confidence_str = val.confidence if val is not None else "N/A"
        issues_count = len(val.issues) if val is not None else 0

        print(f"\n--- Scenario {idx}: {label} ---")
        print(f"Description:        {note}")
        print(f"Original Prompt:    {repr(result.original_prompt)}")
        print(f"Final Prompt:       {repr(result.compressed_prompt)}")
        print(f"Rules Applied:      {result.rules_applied}")
        print(f"Validation Pass:    {is_valid_str} (Confidence: {confidence_str}, Issues: {issues_count})")

        if val and val.preserved_categories:
            print(f"Preserved Checks:   {val.preserved_categories}")
        if val and val.failed_categories:
            print(f"Failed Checks:      {val.failed_categories}")
        if val and val.uncertain_categories:
            print(f"Uncertain Checks:   {val.uncertain_categories}")

        print(f"Original Tokens:    {result.original_token_count}")
        print(f"Compressed Tokens:  {result.compressed_token_count}")
        print(f"Tokens Saved:       {result.tokens_saved}")
        print(f"Reduction Rate:     {result.reduction_percentage:.2f}%" if result.reduction_percentage is not None else "Reduction Rate:     N/A")

    # 12. Deliberate Alteration / Failure Rejection Demonstration
    print("\n" + "-" * 80)
    print("--- Scenario 12: Deliberate Constraint Alteration & Rejection Demonstration ---")
    print("Simulates a faulty rule that strips 'in JSON format' and '3' from the prompt.")
    faulty_rule = InstructionRule(
        rule_id="FAULTY_STRIP_RULE",
        description="Deliberately strips constraints to test rejection mechanism.",
        pattern=re.compile(r"Please provide an explanation of.*?in JSON format\.", re.IGNORECASE),
        replacement="Explain.",
    )
    test_faulty_compressor = RuleBasedInstructionCompressor(
        rules=[faulty_rule],
        validator=validator,
        reject_on_validation_failure=True,
    )
    faulty_test_prompt = "Please provide an explanation of the 3 primary laws of robotics in JSON format."
    faulty_result = test_faulty_compressor.compress_with_stats(
        faulty_test_prompt,
        tokenizer=tokenizer,
        validator=validator,
        reject_on_validation_failure=True,
    )
    val_f = faulty_result.validation_result
    print(f"Original Prompt:    {repr(faulty_result.original_prompt)}")
    print(f"Retained Prompt:    {repr(faulty_result.compressed_prompt)} (Reverted on failure: {faulty_result.compressed_prompt == faulty_test_prompt})")
    print(f"Validation Pass:    {val_f.is_valid if val_f else 'N/A'}")
    print(f"Failed Checks:      {val_f.failed_categories if val_f else []}")
    print(f"Tokens Saved:       {faulty_result.tokens_saved} (Safeguard Rejection Kept 0 Tokens Saved)")

    print("\n" + "=" * 80)
    print("NOTE ON EVALUATION:")
    print("Instruction compression validation checks explicit prompt constraints, numerical bounds,")
    print("output formats, conditional guards, required examples, entities, and technical terms.")
    print("Conservative local validation verifies that these structural requirements are preserved.")
    print("It does not claim full semantic equivalence across arbitrary generative prompts.")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
