"""Redundancy-based prompt token compressors.

This module provides modular, local, deterministic compressors that reduce user-prompt
tokens by removing duplicate and redundant information without making external LLM or network calls.

Phase 4.1A: Exact Duplicate Removal
- Removes exact duplicate sentences while preserving order and original text of retained sentences.
- Normalizes only for comparison: trims surrounding whitespace and compares case-insensitively.
- Preserves the exact original casing and text of retained sentences (does not rewrite them).
- Handles empty input, single sentences, duplicate sentences, whitespace/case variations, and prompts without duplicates.

Phase 4.1B: Similarity-Based Redundancy Detection and Removal
- Reuses SentenceTransformer embeddings (default: all-MiniLM-L6-v2) for semantic similarity comparison.
- Evaluates pairwise sentence similarity against a configurable threshold (default: 0.80).
- Applies conservative safeguards: prevents removing sentences with distinct constraints,
  numbers/quantities, negations, output formats, named entities, or substantive additional details.
- Preserves original text, casing, and relative sentence order for retained sentences.
- Separately callable from exact duplicate removal without introducing hybrid/adaptive selection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Any, List, Optional, Set, Tuple

import numpy as np

from src.analyzer.semantic_scorer import (
    DEFAULT_MODEL_NAME,
    DEFAULT_REDUNDANCY_THRESHOLD,
    BaseSemanticScorer,
    get_shared_model,
)


class BaseCompressor(ABC):
    """Abstract base class for all prompt compressors."""

    @abstractmethod
    def compress(self, prompt: str) -> str:
        """Compress the prompt string and return the compressed prompt."""
        pass


@dataclass(frozen=True)
class DuplicateRemovalResult:
    """Detailed result container for exact duplicate removal compression.

    Attributes:
        original_prompt: The raw input prompt string.
        compressed_prompt: The compressed prompt string after duplicate removal.
        original_sentence_count: Number of candidate sentences identified in the original prompt.
        retained_sentence_count: Number of unique sentences retained.
        removed_sentence_count: Number of duplicate sentences eliminated.
        original_token_count: Token count of original prompt (if tokenizer provided).
        compressed_token_count: Token count of compressed prompt (if tokenizer provided).
        tokens_saved: Number of tokens saved (if tokenizer provided).
        reduction_percentage: Percentage of tokens reduced (if tokenizer provided).
    """

    original_prompt: str
    compressed_prompt: str
    original_sentence_count: int
    retained_sentence_count: int
    removed_sentence_count: int
    original_token_count: Optional[int] = None
    compressed_token_count: Optional[int] = None
    tokens_saved: Optional[int] = None
    reduction_percentage: Optional[float] = None


@dataclass(frozen=True)
class SimilarityRemovalResult:
    """Detailed result container for similarity-based redundancy removal compression.

    Attributes:
        original_prompt: The raw input prompt string.
        compressed_prompt: The compressed prompt string after similarity-based reduction.
        original_sentence_count: Number of candidate sentences identified in the original prompt.
        retained_sentence_count: Number of sentences retained.
        removed_sentence_count: Number of redundant sentences eliminated.
        similarity_threshold: Cosine similarity threshold used for redundancy detection.
        removed_sentences: List of tuples (removed_sentence, reference_sentence, cosine_similarity).
        safeguarded_sentences: List of tuples (retained_sentence, reference_sentence, cosine_similarity, safeguard_reason).
        original_token_count: Token count of original prompt (if tokenizer provided).
        compressed_token_count: Token count of compressed prompt (if tokenizer provided).
        tokens_saved: Number of tokens saved (if tokenizer provided).
        reduction_percentage: Percentage of tokens reduced (if tokenizer provided).
    """

    original_prompt: str
    compressed_prompt: str
    original_sentence_count: int
    retained_sentence_count: int
    removed_sentence_count: int
    similarity_threshold: float
    removed_sentences: List[Tuple[str, str, float]]
    safeguarded_sentences: List[Tuple[str, str, float, str]]
    original_token_count: Optional[int] = None
    compressed_token_count: Optional[int] = None
    tokens_saved: Optional[int] = None
    reduction_percentage: Optional[float] = None


class ExactDuplicateCompressor(BaseCompressor):
    """Modular compressor that removes exact duplicate sentences from user prompts.

    Normalization is strictly applied only during comparison:
    - Trims surrounding whitespace
    - Compares case-insensitively

    Retained sentences preserve their exact original text, casing, and punctuation.
    """

    # Split pattern: sentence-ending punctuation (.!?) followed by whitespace, or newline sequences
    _SPLIT_PATTERN = re.compile(r"((?<=[.!?])\s+|\n+)")

    def _split_with_delimiters(self, text: str) -> Tuple[List[str], List[str]]:
        """Split text into sentence segments and their intervening delimiters."""
        stripped = text.strip()
        if not stripped:
            return [], []

        parts = self._SPLIT_PATTERN.split(stripped)
        sentences = [parts[i] for i in range(0, len(parts), 2)]
        delimiters = [parts[i] for i in range(1, len(parts), 2)]
        return sentences, delimiters

    def compress(self, prompt: str) -> str:
        """Remove exact duplicate sentences from the prompt.

        Args:
            prompt: Raw user input prompt string.

        Returns:
            Compressed prompt string with duplicate sentences removed.
        """
        if prompt is None:
            return ""

        stripped = prompt.strip()
        if not stripped:
            return ""

        sentences, delims = self._split_with_delimiters(prompt)
        if not sentences:
            return ""

        seen_normalized: Set[str] = set()
        retained: List[Tuple[str, str]] = []  # (delimiter_before, original_sentence)

        for i, sentence in enumerate(sentences):
            norm = sentence.strip().lower()
            if not norm:
                continue

            if norm not in seen_normalized:
                seen_normalized.add(norm)
                delim = delims[i - 1] if i > 0 and (i - 1) < len(delims) else ""
                retained.append((delim, sentence))

        if not retained:
            return ""

        output_parts: List[str] = []
        for i, (delim, s) in enumerate(retained):
            if i == 0:
                output_parts.append(s)
            else:
                output_parts.append(delim if delim else " ")
                output_parts.append(s)

        return "".join(output_parts)

    def compress_with_stats(
        self, prompt: str, tokenizer: Optional[Any] = None
    ) -> DuplicateRemovalResult:
        """Compress the prompt and compute token savings statistics.

        Args:
            prompt: Raw user input prompt string.
            tokenizer: Optional tokenizer instance implementing count_tokens(str) -> int.

        Returns:
            DuplicateRemovalResult with sentence counts and token reduction metrics.
        """
        safe_prompt = prompt if prompt is not None else ""
        compressed = self.compress(safe_prompt)

        sentences, _ = self._split_with_delimiters(safe_prompt)
        orig_count = len([s for s in sentences if s.strip()])

        comp_sentences, _ = self._split_with_delimiters(compressed)
        retained_count = len([s for s in comp_sentences if s.strip()])
        removed_count = max(0, orig_count - retained_count)

        orig_tokens: Optional[int] = None
        comp_tokens: Optional[int] = None
        tokens_saved: Optional[int] = None
        reduction_pct: Optional[float] = None

        if tokenizer is not None:
            orig_tokens = tokenizer.count_tokens(safe_prompt)
            comp_tokens = tokenizer.count_tokens(compressed)
            tokens_saved = max(0, orig_tokens - comp_tokens)
            if orig_tokens > 0:
                reduction_pct = (tokens_saved / orig_tokens) * 100.0
            else:
                reduction_pct = 0.0

        return DuplicateRemovalResult(
            original_prompt=safe_prompt,
            compressed_prompt=compressed,
            original_sentence_count=orig_count,
            retained_sentence_count=retained_count,
            removed_sentence_count=removed_count,
            original_token_count=orig_tokens,
            compressed_token_count=comp_tokens,
            tokens_saved=tokens_saved,
            reduction_percentage=reduction_pct,
        )


class SimilarityRedundancyCompressor(BaseCompressor):
    """Modular compressor that removes semantically redundant sentences based on cosine similarity.

    Phase 4.1B:
    - Reuses SentenceTransformer sentence embeddings (default: all-MiniLM-L6-v2).
    - Configurable similarity threshold (default: 0.80).
    - Conservative safeguards prevent deleting sentences with distinct constraints, numbers,
      negations, output formats, named entities, or substantive additional detail.
    - Preserves original sentence text, casing, punctuation, and relative order.
    """

    _SPLIT_PATTERN = re.compile(r"((?<=[.!?])\s+|\n+)")

    # Conservative safeguard patterns
    _NUMBER_PATTERN = re.compile(
        r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"hundred|thousand|million|billion)\b",
        re.IGNORECASE,
    )

    _NEGATION_PATTERN = re.compile(
        r"\b(?:not|no|never|don't|do\s+not|without|must\s+not|cannot|can't|shouldn't|should\s+not|"
        r"won't|will\s+not|exclude|omit)\b",
        re.IGNORECASE,
    )

    _FORMAT_KEYWORDS = frozenset({
        "json", "yaml", "xml", "csv", "table", "markdown", "bullet",
        "bullets", "list", "html", "latex", "sql", "code",
    })

    _CONSTRAINT_DIRECTIVE_PATTERN = re.compile(
        r"\b(?:must|only|required|mandatory|limit|maximum|max|minimum|min|at\s+least|at\s+most|except)\b",
        re.IGNORECASE,
    )

    _STOPWORDS = frozenset({
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
        "is", "are", "was", "were", "it", "this", "that", "with", "as", "by",
        "from", "be", "so", "if", "then", "into", "onto", "your", "my", "our",
    })

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
        model_name: str = DEFAULT_MODEL_NAME,
        model: Optional[Any] = None,
        scorer: Optional[BaseSemanticScorer] = None,
    ):
        """Initialize SimilarityRedundancyCompressor.

        Args:
            similarity_threshold: Cosine similarity threshold above which a sentence
                is considered candidate redundant (default: 0.80).
            model_name: SentenceTransformer model name to load lazily.
            model: Optional pre-loaded or mock embedding model.
            scorer: Optional pre-configured BaseSemanticScorer.
        """
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self._model = model
        self._scorer = scorer

    def _get_model(self) -> Any:
        """Lazily retrieve the embedding model."""
        if self._model is not None:
            return self._model
        if self._scorer is not None and hasattr(self._scorer, "_get_model"):
            self._model = self._scorer._get_model()
            return self._model
        self._model = get_shared_model(self.model_name)
        return self._model

    def _split_with_delimiters(self, text: str) -> Tuple[List[str], List[str]]:
        """Split text into sentence segments and their intervening delimiters."""
        stripped = text.strip()
        if not stripped:
            return [], []

        parts = self._SPLIT_PATTERN.split(stripped)
        sentences = [parts[i] for i in range(0, len(parts), 2)]
        delimiters = [parts[i] for i in range(1, len(parts), 2)]
        return sentences, delimiters

    def _check_conservative_safeguards(
        self, candidate: str, reference: str
    ) -> Tuple[bool, str]:
        """Check whether candidate sentence can be safely deleted as redundant with reference.

        Args:
            candidate: The candidate sentence considered for removal.
            reference: The already-retained sentence it is similar to.

        Returns:
            Tuple of (can_remove: bool, reason: str).
            If can_remove is False, reason describes which safeguard protected the sentence.
        """
        # Exact match trivially has no critical differences
        if candidate.strip().lower() == reference.strip().lower():
            return True, ""

        # 1. Number / Quantity differences
        cand_nums = set(m.lower() for m in self._NUMBER_PATTERN.findall(candidate))
        ref_nums = set(m.lower() for m in self._NUMBER_PATTERN.findall(reference))
        if cand_nums != ref_nums:
            return False, "distinct_numbers_or_quantities"

        # 2. Negation mismatch
        cand_negs = set(m.lower() for m in self._NEGATION_PATTERN.findall(candidate))
        ref_negs = set(m.lower() for m in self._NEGATION_PATTERN.findall(reference))
        if bool(cand_negs) != bool(ref_negs) or cand_negs != ref_negs:
            return False, "negation_mismatch"

        # 3. Distinct output format constraints
        cand_words = set(re.findall(r"\b[a-zA-Z0-9_\-+#]+\b", candidate.lower()))
        ref_words = set(re.findall(r"\b[a-zA-Z0-9_\-+#]+\b", reference.lower()))
        cand_formats = cand_words & self._FORMAT_KEYWORDS
        ref_formats = ref_words & self._FORMAT_KEYWORDS
        if cand_formats != ref_formats:
            return False, "distinct_format_constraint"

        # 4. Distinct constraint directives
        cand_directives = set(m.lower() for m in self._CONSTRAINT_DIRECTIVE_PATTERN.findall(candidate))
        ref_directives = set(m.lower() for m in self._CONSTRAINT_DIRECTIVE_PATTERN.findall(reference))
        if cand_directives != ref_directives:
            return False, "distinct_constraint_directive"

        # 5. Distinct named entities (capitalized tokens not at sentence beginning)
        cand_entities = set(re.findall(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-zA-Z0-9_+#.-]*\b", candidate))
        ref_entities = set(re.findall(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-zA-Z0-9_+#.-]*\b", reference))
        if cand_entities != ref_entities:
            return False, "distinct_named_entities"

        # 6. Asymmetric additional substantive content
        cand_content = cand_words - self._STOPWORDS
        ref_content = ref_words - self._STOPWORDS
        extra_content = cand_content - ref_content
        if len(extra_content) >= 3 and len(cand_content) > len(ref_content) * 1.5:
            return False, "additional_substantive_detail"

        return True, ""

    def compress(self, prompt: str, threshold: Optional[float] = None) -> str:
        """Remove semantically redundant sentences exceeding the similarity threshold.

        Args:
            prompt: Raw user input prompt string.
            threshold: Optional threshold override. Defaults to self.similarity_threshold.

        Returns:
            Compressed prompt string with redundant sentences eliminated.
        """
        result = self.compress_with_stats(prompt, tokenizer=None, threshold=threshold)
        return result.compressed_prompt

    def compress_with_stats(
        self,
        prompt: str,
        tokenizer: Optional[Any] = None,
        threshold: Optional[float] = None,
    ) -> SimilarityRemovalResult:
        """Compress the prompt and return comprehensive redundancy reduction metrics.

        Args:
            prompt: Raw user input prompt string.
            tokenizer: Optional tokenizer instance implementing count_tokens(str) -> int.
            threshold: Optional threshold override.

        Returns:
            SimilarityRemovalResult instance with full details.
        """
        eff_threshold = (
            self.similarity_threshold if threshold is None else threshold
        )

        safe_prompt = prompt if prompt is not None else ""
        stripped = safe_prompt.strip()

        if not stripped:
            return SimilarityRemovalResult(
                original_prompt=safe_prompt,
                compressed_prompt="",
                original_sentence_count=0,
                retained_sentence_count=0,
                removed_sentence_count=0,
                similarity_threshold=eff_threshold,
                removed_sentences=[],
                safeguarded_sentences=[],
                original_token_count=0 if tokenizer is not None else None,
                compressed_token_count=0 if tokenizer is not None else None,
                tokens_saved=0 if tokenizer is not None else None,
                reduction_percentage=0.0 if tokenizer is not None else None,
            )

        sentences, delims = self._split_with_delimiters(safe_prompt)
        clean_sentences = [s for s in sentences if s.strip()]
        orig_count = len(clean_sentences)

        if orig_count <= 1:
            orig_tokens = tokenizer.count_tokens(safe_prompt) if tokenizer is not None else None
            comp_tokens = orig_tokens
            tokens_saved = 0 if tokenizer is not None else None
            reduction_pct = 0.0 if tokenizer is not None else None

            return SimilarityRemovalResult(
                original_prompt=safe_prompt,
                compressed_prompt=stripped,
                original_sentence_count=orig_count,
                retained_sentence_count=orig_count,
                removed_sentence_count=0,
                similarity_threshold=eff_threshold,
                removed_sentences=[],
                safeguarded_sentences=[],
                original_token_count=orig_tokens,
                compressed_token_count=comp_tokens,
                tokens_saved=tokens_saved,
                reduction_percentage=reduction_pct,
            )

        # Compute embeddings for all non-empty sentences
        model = self._get_model()
        embeddings = model.encode(clean_sentences, normalize_embeddings=True)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        retained: List[Tuple[str, str]] = []  # (delimiter, original_sentence)
        retained_indices: List[int] = []      # index in clean_sentences
        removed_sentences: List[Tuple[str, str, float]] = []
        safeguarded_sentences: List[Tuple[str, str, float, str]] = []

        clean_to_orig_idx = [i for i, s in enumerate(sentences) if s.strip()]

        for j, s_cand in enumerate(clean_sentences):
            orig_idx = clean_to_orig_idx[j]
            delim = delims[orig_idx - 1] if orig_idx > 0 and (orig_idx - 1) < len(delims) else ""

            if j == 0:
                retained.append((delim, s_cand))
                retained_indices.append(j)
                continue

            is_redundant = False

            for r in retained_indices:
                s_ref = clean_sentences[r]
                sim = float(np.dot(embeddings[j], embeddings[r]))
                sim = max(-1.0, min(1.0, sim))

                if sim >= eff_threshold:
                    can_remove, reason = self._check_conservative_safeguards(
                        candidate=s_cand, reference=s_ref
                    )
                    if can_remove:
                        is_redundant = True
                        removed_sentences.append((s_cand, s_ref, sim))
                        break
                    else:
                        safeguarded_sentences.append((s_cand, s_ref, sim, reason))

            if not is_redundant:
                retained.append((delim, s_cand))
                retained_indices.append(j)

        output_parts: List[str] = []
        for i, (delim, s) in enumerate(retained):
            if i == 0:
                output_parts.append(s)
            else:
                output_parts.append(delim if delim else " ")
                output_parts.append(s)

        compressed_prompt = "".join(output_parts)
        retained_count = len(retained)
        removed_count = orig_count - retained_count

        orig_tokens = None
        comp_tokens = None
        tokens_saved = None
        reduction_pct = None

        if tokenizer is not None:
            orig_tokens = tokenizer.count_tokens(safe_prompt)
            comp_tokens = tokenizer.count_tokens(compressed_prompt)
            tokens_saved = max(0, orig_tokens - comp_tokens)
            if orig_tokens > 0:
                reduction_pct = (tokens_saved / orig_tokens) * 100.0
            else:
                reduction_pct = 0.0

        return SimilarityRemovalResult(
            original_prompt=safe_prompt,
            compressed_prompt=compressed_prompt,
            original_sentence_count=orig_count,
            retained_sentence_count=retained_count,
            removed_sentence_count=removed_count,
            similarity_threshold=eff_threshold,
            removed_sentences=removed_sentences,
            safeguarded_sentences=safeguarded_sentences,
            original_token_count=orig_tokens,
            compressed_token_count=comp_tokens,
            tokens_saved=tokens_saved,
            reduction_percentage=reduction_pct,
        )


def remove_exact_duplicates(prompt: str) -> str:
    """Convenience function to remove exact duplicate sentences from a prompt."""
    return ExactDuplicateCompressor().compress(prompt)


def get_exact_duplicate_compressor() -> ExactDuplicateCompressor:
    """Factory function returning the default exact duplicate compressor instance."""
    return ExactDuplicateCompressor()


def remove_similar_redundancy(
    prompt: str,
    threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
    model: Optional[Any] = None,
) -> str:
    """Convenience function to remove semantically redundant sentences from a prompt."""
    return SimilarityRedundancyCompressor(similarity_threshold=threshold, model=model).compress(prompt)


def get_similarity_redundancy_compressor(
    threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
    model: Optional[Any] = None,
) -> SimilarityRedundancyCompressor:
    """Factory function returning a SimilarityRedundancyCompressor instance."""
    return SimilarityRedundancyCompressor(similarity_threshold=threshold, model=model)
