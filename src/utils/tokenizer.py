"""Tokenizer abstraction and implementations for token counting.

This module isolates tokenizer logic so that tokenizers can be easily swapped
without modifying measurement or evaluation logic.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import tiktoken


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text into a list of token IDs."""
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode a list of token IDs back into text."""
        pass


class TiktokenTokenizer(BaseTokenizer):
    """Tokenizer implementation backed by tiktoken for fast local BPE tokenization."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken BPE encoding."""
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using tiktoken."""
        if not text:
            return []
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text using tiktoken."""
        if not tokens:
            return ""
        return self._encoding.decode(tokens)


def get_tokenizer(name_or_type: str = "cl100k_base") -> BaseTokenizer:
    """Factory function to get a tokenizer instance.

    Args:
        name_or_type: Tokenizer identifier (default: 'cl100k_base').

    Returns:
        An instance of BaseTokenizer.
    """
    return TiktokenTokenizer(encoding_name=name_or_type)
