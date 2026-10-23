"""Controlled evaluation runner script for Adaptive User-Prompt Token Compression.

Loads prompt compression dataset (150 prompts), evaluates original prompts against:
1. Baseline (original prompt)
2. Exact-duplicate removal
3. Similarity-based redundancy removal
4. Rule-based instruction compression
5. Combined technique (safe composition)

Produces:
- per_prompt_results.csv: Record per prompt and experimental condition
- summary_results.csv: Metrics grouped by experiment and prompt category

Usage:
    python experiments/run_controlled_evaluation.py
    python experiments/run_controlled_evaluation.py --dataset data/raw/prompt_compression_dataset1.csv --output-dir data/result/controlled_evaluation

NOTE: Run this script manually to generate and inspect benchmark experimental results.
"""

import argparse
import os
from pathlib import Path
import sys
import time

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.controlled_runner import (
    SUPPORTED_EXPERIMENTS,
    ControlledEvaluationRunner,
)
from src.utils.tokenizer import get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run controlled evaluation of prompt token compressors across dataset conditions."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/raw/prompt_compression_dataset1.csv",
        help="Path to input dataset CSV (default: data/raw/prompt_compression_dataset1.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/result/controlled_evaluation",
        help="Directory where evaluation CSV results will be saved (default: data/result/controlled_evaluation)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="cl100k_base",
        help="Tokenizer encoding identifier (default: cl100k_base)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = (PROJECT_ROOT / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    print("=" * 80)
    print("CONTROLLED EVALUATION RUNNER: ADAPTIVE PROMPT TOKEN COMPRESSION")
    print("=" * 80)
    print(f"Dataset path:   {dataset_path}")
    print(f"Output directory: {output_dir}")
    print(f"Tokenizer:      {args.encoding}")
    print(f"Techniques:     {', '.join(SUPPORTED_EXPERIMENTS)}")
    print("-" * 80)

    if not dataset_path.exists():
        print(f"ERROR: Dataset file not found at {dataset_path}")
        sys.exit(1)

    tokenizer = get_tokenizer(args.encoding)
    runner = ControlledEvaluationRunner(tokenizer=tokenizer)

    print("Running controlled evaluation across all conditions...")
    start_time = time.time()
    per_prompt_csv, summary_csv = runner.run_and_save(
        dataset_path=dataset_path,
        output_dir=output_dir,
        techniques=SUPPORTED_EXPERIMENTS,
    )
    elapsed = time.time() - start_time

    print("\nEvaluation successfully completed.")
    print(f"Elapsed total time: {elapsed:.2f} seconds")
    print(f"Per-prompt results saved to: {per_prompt_csv}")
    print(f"Summary results saved to:    {summary_csv}")
    print("-" * 80)
    print("Preservation Note:")
    print("  Structural constraint validation verifies local structural rules (quantities,")
    print("  negations, output formats, audience/tone, entities, code identifiers).")
    print("  This local check does NOT constitute a formal proof of semantic equivalence")
    print("  or target LLM task performance, which requires downstream model benchmarking.")
    print("=" * 80)


if __name__ == "__main__":
    main()
