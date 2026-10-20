"""Phase 4.1A Experiment: Exact Duplicate Removal Token Reduction.

Evaluates token reduction achieved by ExactDuplicateCompressor across a range
of representative prompt scenarios using the project's configured tokenizer.
Handles zero-token and empty prompts safely.
"""

from src.compressors.redundancy import ExactDuplicateCompressor
from src.utils.tokenizer import get_tokenizer


def run_experiment() -> None:
    tokenizer = get_tokenizer("cl100k_base")
    compressor = ExactDuplicateCompressor()

    test_prompts = [
        # 1. Exact consecutive duplicate sentences
        (
            "Exact Consecutive Duplicates",
            "Explain quantum computing in simple terms. Explain quantum computing in simple terms.",
        ),
        # 2. Non-consecutive duplicates across prompt sections
        (
            "Non-Consecutive Duplicates",
            (
                "Write a Python script to sort customer records.\n"
                "Ensure all functions have type annotations.\n"
                "Include unit tests for edge cases.\n"
                "Ensure all functions have type annotations."
            ),
        ),
        # 3. Case and whitespace variations
        (
            "Case & Whitespace Variations",
            (
                "Analyze the quarterly financial dataset.   "
                "analyze the quarterly financial dataset.    "
                "ANALYZED THE QUARTERLY FINANCIAL DATASET."  # different wording, should stay
            ),
        ),
        # 4. Multi-sentence prompt with multiple duplicates
        (
            "Multiple Repeated Instructions",
            (
                "Summarize the attached contract in 3 bullet points. "
                "Highlight key liabilities and indemnities. "
                "Summarize the attached contract in 3 bullet points. "
                "Highlight key liabilities and indemnities. "
                "Format output as Markdown."
            ),
        ),
        # 5. Prompt without duplicates (zero removal expected)
        (
            "Prompt Without Duplicates",
            (
                "Compare PostgreSQL and MongoDB for time-series data. "
                "List performance advantages of each database. "
                "Provide a recommended indexing strategy."
            ),
        ),
        # 6. Merely similar sentences (Phase 4.1B check: should NOT be removed)
        (
            "Merely Similar Sentences (Not Exact Duplicates)",
            (
                "Machine learning allows computers to learn from data. "
                "Machine learning enables computers to learn using data."
            ),
        ),
        # 7. Single sentence prompt
        (
            "Single Sentence",
            "What is the capital of Australia?",
        ),
        # 8. Empty input (zero tokens, safe handling)
        (
            "Empty Input (Zero-Token)",
            "",
        ),
    ]

    print("=" * 80)
    print("PHASE 4.1A: EXACT DUPLICATE REMOVAL EXPERIMENTAL EVALUATION")
    print("=" * 80)

    for idx, (label, prompt) in enumerate(test_prompts, 1):
        result = compressor.compress_with_stats(prompt, tokenizer=tokenizer)

        print(f"\n--- Scenario {idx}: {label} ---")
        print(f"Original Prompt:   {repr(result.original_prompt)}")
        print(f"Compressed Prompt: {repr(result.compressed_prompt)}")
        print(f"Sentence Count:    {result.original_sentence_count} -> {result.retained_sentence_count} (Removed: {result.removed_sentence_count})")
        print(f"Original Tokens:   {result.original_token_count}")
        print(f"Compressed Tokens: {result.compressed_token_count}")
        print(f"Tokens Saved:      {result.tokens_saved}")
        print(f"Reduction Rate:    {result.reduction_percentage:.2f}%")

    print("\n" + "=" * 80)
    print("NOTE ON EVALUATION:")
    print("Exact duplicate removal reduces tokens by eliminating repeated identical sentences.")
    print("Semantic preservation must be evaluated through dedicated validation; passing duplicate")
    print("removal tests does not automatically prove semantic preservation across general prompts.")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
