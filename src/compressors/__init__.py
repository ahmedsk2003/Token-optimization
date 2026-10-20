"""Compressors module for prompt token optimization.

Exposes base compressor abstraction, exact duplicate compressor, similarity-based
redundancy compressor, and utility functions.
"""

from src.compressors.redundancy import (
    BaseCompressor,
    DuplicateRemovalResult,
    ExactDuplicateCompressor,
    SimilarityRedundancyCompressor,
    SimilarityRemovalResult,
    get_exact_duplicate_compressor,
    get_similarity_redundancy_compressor,
    remove_exact_duplicates,
    remove_similar_redundancy,
)

__all__ = [
    "BaseCompressor",
    "ExactDuplicateCompressor",
    "DuplicateRemovalResult",
    "get_exact_duplicate_compressor",
    "remove_exact_duplicates",
    "SimilarityRedundancyCompressor",
    "SimilarityRemovalResult",
    "get_similarity_redundancy_compressor",
    "remove_similar_redundancy",
]
