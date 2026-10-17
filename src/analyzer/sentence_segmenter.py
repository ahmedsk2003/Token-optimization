"""Sentence segmentation module for prompt analysis.

Provides deterministic, local sentence segmentation abstractions and implementations
that can be easily replaced or refined in future phases without breaking analyzer logic.
"""

import re
from abc import ABC, abstractmethod
from typing import List


class BaseSentenceSegmenter(ABC):
    """Abstract base class for sentence segmentation."""

    @abstractmethod
    def split_sentences(self, text: str) -> List[str]:
        """Split input text into a list of sentence strings."""
        pass

    def count_sentences(self, text: str) -> int:
        """Count the number of sentences in the input text."""
        return len(self.split_sentences(text))


class RegexSentenceSegmenter(BaseSentenceSegmenter):
    """Deterministic local sentence segmenter using regular expressions.

    Splits text along standard sentence-ending punctuation (.!?), newlines,
    and trailing sentence boundaries while preserving determinism and avoiding
    external network dependencies.
    """

    # Split pattern: matches sequence of sentence delimiters (.!?) followed by whitespace, or newline sequences
    _SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences deterministically.

        Args:
            text: Input prompt string.

        Returns:
            List of non-empty stripped sentence segments.
        """
        if not text or not text.strip():
            return []

        segments = self._SPLIT_PATTERN.split(text.strip())
        # Filter out empty or whitespace-only chunks
        return [seg.strip() for seg in segments if seg.strip()]


def get_sentence_segmenter() -> BaseSentenceSegmenter:
    """Factory function returning the default sentence segmenter instance."""
    return RegexSentenceSegmenter()
