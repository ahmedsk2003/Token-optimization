"""Baseline measurement system for prompt token, word, and character metrics.

Provides reusable measurement logic that applies consistently across original
and compressed prompts without modifying prompt contents.
"""

import csv
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from src.utils.tokenizer import BaseTokenizer, get_tokenizer


@dataclass(frozen=True)
class PromptMeasurement:
    """Immutable data record for a prompt measurement."""
    prompt_id: str
    prompt: str
    token_count: int
    word_count: int
    character_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)


def count_words(text: str) -> int:
    """Consistent word count using whitespace splitting."""
    if not text:
        return 0
    return len(text.split())


def count_characters(text: str) -> int:
    """Consistent character count using exact string length."""
    return len(text)


def measure_prompt(
    prompt: str,
    prompt_id: str = "",
    tokenizer: Optional[BaseTokenizer] = None,
) -> PromptMeasurement:
    """Measure token, word, and character counts for any given prompt.

    This function is reusable for both original user prompts and compressed prompts.

    Args:
        prompt: Exact prompt text to measure.
        prompt_id: Optional prompt identifier (e.g., 'P001').
        tokenizer: Tokenizer instance. If None, default tokenizer is used.

    Returns:
        PromptMeasurement containing prompt_id, prompt, token_count,
        word_count, and character_count.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    token_count = tokenizer.count_tokens(prompt)
    word_cnt = count_words(prompt)
    char_cnt = count_characters(prompt)

    return PromptMeasurement(
        prompt_id=prompt_id,
        prompt=prompt,
        token_count=token_count,
        word_count=word_cnt,
        character_count=char_cnt,
    )


def measure_baseline(
    prompts: Iterable[str],
    tokenizer: Optional[BaseTokenizer] = None,
    id_prefix: str = "P",
    start_index: int = 1,
    id_padding: int = 3,
) -> List[PromptMeasurement]:
    """Measure a collection of baseline prompts with sequential IDs.

    Args:
        prompts: Iterable of prompt strings.
        tokenizer: Tokenizer instance to use. If None, default is used.
        id_prefix: Prefix for sequential prompt IDs (default: 'P').
        start_index: Starting integer for ID generation (default: 1).
        id_padding: Zero-padding width for ID (default: 3, e.g., 'P001').

    Returns:
        List of PromptMeasurement objects.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    measurements: List[PromptMeasurement] = []
    for idx, prompt in enumerate(prompts, start=start_index):
        # Determine appropriate padding dynamically if index exceeds id_padding
        pad = max(id_padding, len(str(idx)))
        prompt_id = f"{id_prefix}{idx:0{pad}d}"
        measurement = measure_prompt(
            prompt=prompt,
            prompt_id=prompt_id,
            tokenizer=tokenizer,
        )
        measurements.append(measurement)

    return measurements


def save_baseline_to_csv(
    records: List[Union[PromptMeasurement, Dict[str, Any]]],
    output_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "data/results/baseline",
    filename: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Save baseline measurement records to a CSV file.

    Will not overwrite an existing file unless overwrite=True is explicitly set.

    Args:
        records: List of PromptMeasurement instances or dicts.
        output_path: Direct file path to save CSV. If given, output_dir/filename are ignored.
        output_dir: Directory to save CSV in if output_path is not specified.
        filename: File name to use under output_dir (default: 'baseline_measurements.csv').
        overwrite: Whether to overwrite existing file. Defaults to False.

    Returns:
        Path object pointing to the written CSV file.

    Raises:
        FileExistsError: If target file already exists and overwrite is False.
    """
    if output_path is not None:
        target_path = Path(output_path)
    else:
        out_dir = Path(output_dir)
        fname = filename if filename is not None else "baseline_measurements.csv"
        target_path = out_dir / fname

    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"Target baseline CSV already exists at '{target_path}'. "
            f"Set overwrite=True or specify a different filename to proceed."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["prompt_id", "prompt", "token_count", "word_count", "character_count"]

    with open(target_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = rec.to_dict() if isinstance(rec, PromptMeasurement) else rec
            writer.writerow(row)

    return target_path
