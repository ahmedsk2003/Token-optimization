"""Phase 4.2A Experiment: Rule-Based Instruction Compression Token Reduction.

Evaluates token reduction achieved by RuleBasedInstructionCompressor across a range
of representative instruction scenarios using the project's configured tokenizer.
Handles zero-token and empty prompts safely.
"""

from src.compressors.instruction import RuleBasedInstructionCompressor
from src.utils.tokenizer import get_tokenizer


def run_experiment() -> None:
    tokenizer = get_tokenizer("cl100k_base")
    compressor = RuleBasedInstructionCompressor()

    test_prompts = [
        # 1. Common verbose explanation request
        (
            "Verbose Explanation Request",
            "Please provide an explanation of how transformer attention mechanisms work.",
        ),
        # 2. Verbose summary request
        (
            "Verbose Summary Request",
            "Please provide a summary of the quarterly financial earnings report.",
        ),
        # 3. Desire preamble request
        (
            "Desire Preamble Instruction",
            "I would like you to explain the key differences between SQL and NoSQL databases.",
        ),
        # 4. Polite modal instruction request
        (
            "Modal Polite Request",
            "Could you please explain how public key cryptography ensures secure communication?",
        ),
        # 5. Multiple verbose instructions in one prompt
        (
            "Multiple Verbose Instructions",
            (
                "Please provide an explanation of quicksort. "
                "Also, please provide a summary of its worst-case complexity."
            ),
        ),
        # 6. Preserved modifier (Safeguard: should NOT be shortened)
        (
            "Preserved Meaningful Modifier (Safeguard)",
            "Please provide a detailed explanation of the consensus protocol.",
        ),
        # 7. Negated instruction (Safeguard: should NOT be shortened)
        (
            "Negated Instruction Clause (Safeguard)",
            "Please do not provide an explanation of the plot twist.",
        ),
        # 8. Conditional instruction clause (Safeguard: should NOT be shortened)
        (
            "Conditional Subordinate Clause (Safeguard)",
            "If you provide an explanation of the system architecture, keep it concise.",
        ),
        # 9. Format and entity constraints preserved
        (
            "Constraints and Entities Preserved",
            "Please provide an explanation of the 3 primary laws of robotics in JSON format.",
        ),
        # 10. Already concise prompt (no replacement needed)
        (
            "Already Concise Prompt",
            "Explain quantum annealing in simple terms.",
        ),
        # 11. Direct factual question (no replacement needed)
        (
            "Direct Factual Question",
            "What is the capital of Australia?",
        ),
        # 12. Empty input (Zero-Token Safe Handling)
        (
            "Empty Input (Zero-Token)",
            "",
        ),
    ]

    print("=" * 80)
    print("PHASE 4.2A: RULE-BASED INSTRUCTION COMPRESSION EXPERIMENTAL EVALUATION")
    print("=" * 80)

    for idx, (label, prompt) in enumerate(test_prompts, 1):
        result = compressor.compress_with_stats(prompt, tokenizer=tokenizer)

        print(f"\n--- Scenario {idx}: {label} ---")
        print(f"Original Prompt:   {repr(result.original_prompt)}")
        print(f"Compressed Prompt: {repr(result.compressed_prompt)}")
        print(f"Rules Applied:     {result.rules_applied}")
        print(f"Original Tokens:   {result.original_token_count}")
        print(f"Compressed Tokens: {result.compressed_token_count}")
        print(f"Tokens Saved:      {result.tokens_saved}")
        print(f"Reduction Rate:    {result.reduction_percentage:.2f}%")

    print("\n" + "=" * 80)
    print("NOTE ON EVALUATION:")
    print("Rule-based instruction compression reduces tokens by shortening verbose instruction")
    print("phrases into concise action verbs while applying conservative safeguards.")
    print("Semantic preservation must be evaluated through dedicated validation; passing unit")
    print("tests does not automatically prove semantic preservation across general prompts.")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
