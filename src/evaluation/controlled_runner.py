"""Controlled evaluation runner for Adaptive User-Prompt Token Compression.

Evaluates user prompts across experimental conditions:
1. Baseline (original prompt)
2. Exact duplicate removal
3. Similarity-based redundancy removal
4. Rule-based instruction compression
5. Combined technique (safe sequential composition)

Maintains comparability by evaluating each technique independently on the original prompt
(except for the explicit combined condition). Accurately calculates token metrics,
latency, must-preserve integrity, and structural constraint preservation.
"""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.compressors.instruction import (
    RuleBasedInstructionCompressor,
    get_instruction_compressor,
)
from src.compressors.redundancy import (
    ExactDuplicateCompressor,
    SimilarityRedundancyCompressor,
    get_exact_duplicate_compressor,
    get_similarity_redundancy_compressor,
)
from src.utils.tokenizer import BaseTokenizer, get_tokenizer
from src.validators.constraint_validator import (
    InstructionConstraintValidator,
    ValidationResult,
    get_constraint_validator,
)

SUPPORTED_EXPERIMENTS: Tuple[str, ...] = (
    "baseline",
    "exact_duplicate_removal",
    "similarity_based_redundancy_removal",
    "rule_based_instruction_compression",
    "combined_technique",
)


@dataclass(frozen=True)
class PromptEvaluationResult:
    """Detailed evaluation result for a single prompt under a specific experimental condition.

    Attributes:
        prompt_id: Identifier from the dataset (e.g. 'DUP-01').
        category: Dataset evaluation category.
        expected_action: Intended compression behavior from dataset annotation.
        duplicate_type: Annotated redundancy type if applicable.
        experiment_name: Name of the evaluated compression condition.
        original_prompt: Original prompt text.
        compressed_prompt: Candidate prompt text after compression.
        prompt_changed: Boolean flag indicating if text changed.
        tokens_before: Token count of original prompt.
        tokens_after: Token count of compressed prompt.
        tokens_saved: Number of tokens saved (max(0, tokens_before - tokens_after)).
        reduction_percentage: Percentage of tokens reduced.
        processing_time_ms: Wall-clock compression time in milliseconds.
        must_preserve_count: Total number of parsed must-preserve items.
        must_preserve_preserved_count: Number of must-preserve items present in compressed text.
        must_preserve_all_preserved: Boolean indicating if all must-preserve items were retained.
        missing_must_preserve: Delimited list of missing must-preserve items (if any).
        validator_is_valid: Structural constraint validator pass/fail result.
        validator_confidence: Validator assessment confidence level.
        validator_error_count: Number of error-severity constraint violations.
        validator_warning_count: Number of warning-severity preservation issues.
        validator_failed_categories: Delimited list of failed constraint categories.
        validator_uncertain_categories: Delimited list of uncertain constraint categories.
        validator_warnings: Explanatory warning notes from the validator.
        error_message: Exception message if compressor failed (fallback to original prompt).
    """

    prompt_id: str
    category: str
    expected_action: str
    duplicate_type: str
    experiment_name: str
    original_prompt: str
    compressed_prompt: str
    prompt_changed: bool
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    reduction_percentage: float
    processing_time_ms: float
    must_preserve_count: int
    must_preserve_preserved_count: int
    must_preserve_all_preserved: bool
    missing_must_preserve: str
    validator_is_valid: bool
    validator_confidence: str
    validator_error_count: int
    validator_warning_count: int
    validator_failed_categories: str
    validator_uncertain_categories: str
    validator_warnings: str
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary format."""
        return asdict(self)


def parse_must_preserve(raw: Any) -> List[str]:
    """Parse must-preserve items delimited by ';;'.

    Safely handles None, NaN, numeric types, and whitespace-only entries.
    """
    if raw is None:
        return []
    if isinstance(raw, float) and np.isnan(raw):
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(";;") if item.strip()]


def check_must_preserve(items: Sequence[str], text: str) -> Tuple[int, int, bool, List[str]]:
    """Check case-insensitive substring presence of must-preserve items in prompt text.

    Args:
        items: List of key phrases or entities required to be preserved.
        text: Prompt text to check.

    Returns:
        Tuple of (total_count, preserved_count, all_preserved_flag, missing_items_list).
    """
    if not items:
        return 0, 0, True, []

    text_lower = (text or "").lower()
    missing: List[str] = []
    for item in items:
        if item.lower() not in text_lower:
            missing.append(item)

    preserved_count = len(items) - len(missing)
    all_preserved = len(missing) == 0
    return len(items), preserved_count, all_preserved, missing


def compute_token_metrics(tokens_before: int, tokens_after: int) -> Tuple[int, float]:
    """Compute tokens saved and percentage reduction safely without zero division.

    Args:
        tokens_before: Number of tokens before compression.
        tokens_after: Number of tokens after compression.

    Returns:
        Tuple of (tokens_saved, reduction_percentage).
    """
    tokens_saved = max(0, tokens_before - tokens_after)
    if tokens_before <= 0:
        reduction_percentage = 0.0
    else:
        reduction_percentage = (tokens_saved / tokens_before) * 100.0
    return tokens_saved, reduction_percentage


class ControlledEvaluationRunner:
    """Controlled evaluation runner for benchmark datasets.

    Runs compressors independently on the original prompt, measures token savings,
    validates structural constraints, checks must-preserve elements, and records
    detailed diagnostics.
    """

    def __init__(
        self,
        tokenizer: Optional[BaseTokenizer] = None,
        validator: Optional[InstructionConstraintValidator] = None,
        exact_compressor: Optional[ExactDuplicateCompressor] = None,
        similarity_compressor: Optional[SimilarityRedundancyCompressor] = None,
        instruction_compressor: Optional[RuleBasedInstructionCompressor] = None,
    ):
        self.tokenizer = tokenizer or get_tokenizer("cl100k_base")
        self.validator = validator or get_constraint_validator()
        self.exact_compressor = exact_compressor or get_exact_duplicate_compressor()
        self.similarity_compressor = similarity_compressor or get_similarity_redundancy_compressor()
        self.instruction_compressor = instruction_compressor or get_instruction_compressor()

    def evaluate_prompt(
        self,
        prompt_id: str,
        category: str,
        expected_action: str,
        duplicate_type: str,
        prompt: Any,
        must_preserve_raw: Any,
        experiment_name: str,
    ) -> PromptEvaluationResult:
        """Evaluate a single prompt under a given experimental condition.

        Args:
            prompt_id: Identifier of the prompt.
            category: Prompt category.
            expected_action: Target action annotation.
            duplicate_type: Redundancy classification.
            prompt: Original prompt text.
            must_preserve_raw: Raw must-preserve value from dataset.
            experiment_name: Technique name to evaluate.

        Returns:
            PromptEvaluationResult record.
        """
        if experiment_name not in SUPPORTED_EXPERIMENTS:
            raise ValueError(
                f"Unsupported experiment '{experiment_name}'. Must be one of {SUPPORTED_EXPERIMENTS}"
            )

        orig_text = "" if prompt is None or (isinstance(prompt, float) and np.isnan(prompt)) else str(prompt)
        must_preserve_items = parse_must_preserve(must_preserve_raw)
        tokens_before = self.tokenizer.count_tokens(orig_text)

        error_msg = ""
        compressed_text = orig_text

        t0 = time.perf_counter()
        try:
            if experiment_name == "baseline":
                compressed_text = orig_text
            elif experiment_name == "exact_duplicate_removal":
                compressed_text = self.exact_compressor.compress(orig_text)
            elif experiment_name == "similarity_based_redundancy_removal":
                compressed_text = self.similarity_compressor.compress(orig_text)
            elif experiment_name == "rule_based_instruction_compression":
                compressed_text = self.instruction_compressor.compress(orig_text)
            elif experiment_name == "combined_technique":
                # Safe sequential composition of compressors
                p1 = self.exact_compressor.compress(orig_text)
                p2 = self.similarity_compressor.compress(p1)
                p3 = self.instruction_compressor.compress(p2)
                compressed_text = p3
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            compressed_text = orig_text
        t1 = time.perf_counter()

        processing_time_ms = (t1 - t0) * 1000.0
        prompt_changed = bool(orig_text != compressed_text)
        tokens_after = self.tokenizer.count_tokens(compressed_text)
        tokens_saved, reduction_pct = compute_token_metrics(tokens_before, tokens_after)

        # Must-preserve entity/phrase checks
        mp_total, mp_pres, mp_all, missing_items = check_must_preserve(must_preserve_items, compressed_text)
        missing_mp_str = ";; ".join(missing_items)

        # Structural constraint validator assessment
        val_result = self.validator.validate(orig_text, compressed_text)
        error_issues = [issue for issue in val_result.issues if issue.severity == "error"]
        warning_issues = [issue for issue in val_result.issues if issue.severity == "warning"]
        warnings_str = ";; ".join(issue.description for issue in warning_issues)

        return PromptEvaluationResult(
            prompt_id=str(prompt_id),
            category=str(category),
            expected_action=str(expected_action),
            duplicate_type=str(duplicate_type),
            experiment_name=experiment_name,
            original_prompt=orig_text,
            compressed_prompt=compressed_text,
            prompt_changed=prompt_changed,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_saved,
            reduction_percentage=reduction_pct,
            processing_time_ms=processing_time_ms,
            must_preserve_count=mp_total,
            must_preserve_preserved_count=mp_pres,
            must_preserve_all_preserved=mp_all,
            missing_must_preserve=missing_mp_str,
            validator_is_valid=val_result.is_valid,
            validator_confidence=val_result.confidence,
            validator_error_count=len(error_issues),
            validator_warning_count=len(warning_issues),
            validator_failed_categories=";; ".join(sorted(val_result.failed_categories)),
            validator_uncertain_categories=";; ".join(sorted(val_result.uncertain_categories)),
            validator_warnings=warnings_str,
            error_message=error_msg,
        )

    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        techniques: Optional[Sequence[str]] = None,
    ) -> List[PromptEvaluationResult]:
        """Evaluate an entire dataset DataFrame across chosen techniques.

        Args:
            df: DataFrame containing at minimum 'id', 'category', 'prompt', 'expected_action'.
            techniques: List of experiment techniques to evaluate. Defaults to all 5 techniques.

        Returns:
            List of PromptEvaluationResult objects.
        """
        active_techniques = list(techniques) if techniques is not None else list(SUPPORTED_EXPERIMENTS)
        results: List[PromptEvaluationResult] = []

        for _, row in df.iterrows():
            prompt_id = row.get("id", "")
            category = row.get("category", "")
            expected_action = row.get("expected_action", "")
            duplicate_type = row.get("duplicate_type", "")
            prompt = row.get("prompt", "")
            must_preserve = row.get("must_preserve", "")

            for technique in active_techniques:
                res = self.evaluate_prompt(
                    prompt_id=prompt_id,
                    category=category,
                    expected_action=expected_action,
                    duplicate_type=duplicate_type,
                    prompt=prompt,
                    must_preserve_raw=must_preserve,
                    experiment_name=technique,
                )
                results.append(res)

        return results

    def run_and_save(
        self,
        dataset_path: Union[str, Path],
        output_dir: Union[str, Path],
        techniques: Optional[Sequence[str]] = None,
    ) -> Tuple[Path, Path]:
        """Load dataset CSV, run evaluation, and save per-prompt and summary CSVs.

        Args:
            dataset_path: Path to dataset CSV file.
            output_dir: Directory where results will be written.
            techniques: Techniques to run (default: all supported).

        Returns:
            Tuple of (per_prompt_csv_path, summary_csv_path).
        """
        dataset_p = Path(dataset_path)
        if not dataset_p.exists():
            raise FileNotFoundError(f"Dataset CSV not found at '{dataset_p.resolve()}'")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(dataset_p)
        results = self.evaluate_dataset(df, techniques=techniques)

        # Per-prompt dataframe
        per_prompt_df = pd.DataFrame([r.to_dict() for r in results])
        per_prompt_path = out_dir / "per_prompt_results.csv"
        per_prompt_df.to_csv(per_prompt_path, index=False, encoding="utf-8")

        # Summary dataframe
        summary_df = build_summary_dataframe(results)
        summary_path = out_dir / "summary_results.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8")

        return per_prompt_path, summary_path


def build_summary_dataframe(results: List[PromptEvaluationResult]) -> pd.DataFrame:
    """Group evaluation results by experiment and category to calculate summary metrics.

    Calculates mean/median token reduction, change rates, validation failures,
    and preservation counts.
    """
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([r.to_dict() for r in results])
    summary_rows: List[Dict[str, Any]] = []

    experiments = list(dict.fromkeys(df["experiment_name"]))

    for exp in experiments:
        exp_df = df[df["experiment_name"] == exp]
        categories = list(dict.fromkeys(exp_df["category"]))

        # 1. Per-category metrics
        for cat in categories:
            cat_df = exp_df[exp_df["category"] == cat]
            summary_rows.append(_aggregate_subset(exp, cat, cat_df))

        # 2. Overall aggregate across all categories for this experiment
        summary_rows.append(_aggregate_subset(exp, "[OVERALL]", exp_df))

    return pd.DataFrame(summary_rows)


def _aggregate_subset(experiment: str, category: str, sub_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute aggregate summary dictionary for a subset of evaluation records."""
    total_prompts = len(sub_df)
    if total_prompts == 0:
        return {
            "experiment_name": experiment,
            "category": category,
            "prompt_count": 0,
            "prompts_changed_count": 0,
            "prompts_changed_percentage": 0.0,
            "mean_tokens_before": 0.0,
            "mean_tokens_after": 0.0,
            "mean_tokens_saved": 0.0,
            "mean_reduction_percentage": 0.0,
            "median_reduction_percentage": 0.0,
            "mean_processing_time_ms": 0.0,
            "validation_failure_count": 0,
            "validation_warning_count": 0,
            "total_validation_warnings": 0,
            "must_preserve_all_preserved_count": 0,
            "must_preserve_all_preserved_percentage": 0.0,
        }

    changed_count = int(sub_df["prompt_changed"].sum())
    val_failures = int((~sub_df["validator_is_valid"]).sum())
    val_warnings_count = int((sub_df["validator_warning_count"] > 0).sum())
    total_warnings = int(sub_df["validator_warning_count"].sum())
    mp_all_count = int(sub_df["must_preserve_all_preserved"].sum())

    return {
        "experiment_name": experiment,
        "category": category,
        "prompt_count": total_prompts,
        "prompts_changed_count": changed_count,
        "prompts_changed_percentage": round((changed_count / total_prompts) * 100.0, 2),
        "mean_tokens_before": round(float(sub_df["tokens_before"].mean()), 2),
        "mean_tokens_after": round(float(sub_df["tokens_after"].mean()), 2),
        "mean_tokens_saved": round(float(sub_df["tokens_saved"].mean()), 2),
        "mean_reduction_percentage": round(float(sub_df["reduction_percentage"].mean()), 2),
        "median_reduction_percentage": round(float(sub_df["reduction_percentage"].median()), 2),
        "mean_processing_time_ms": round(float(sub_df["processing_time_ms"].mean()), 2),
        "validation_failure_count": val_failures,
        "validation_warning_count": val_warnings_count,
        "total_validation_warnings": total_warnings,
        "must_preserve_all_preserved_count": mp_all_count,
        "must_preserve_all_preserved_percentage": round((mp_all_count / total_prompts) * 100.0, 2),
    }
