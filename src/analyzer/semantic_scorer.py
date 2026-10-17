"""Semantic similarity and redundancy scoring module for prompt analysis (Phase 3C).

Computes pairwise semantic similarity and redundancy metrics across sentences in a prompt
using sentence embeddings from a pretrained SentenceTransformer model.

Metrics defined:
- semantic_similarity: Mean pairwise cosine similarity across all unique sentence pairs.
- redundancy_score: Proportion of unique sentence pairs whose cosine similarity exceeds
  a configurable threshold (default: 0.80).

Note on default redundancy threshold:
The 0.80 threshold is an empirical heuristic representing near-paraphrase or high semantic
overlap between sentences in general domain text. It can be tuned or configured as needed.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REDUNDANCY_THRESHOLD = 0.80

# Module-level model cache to reuse loaded SentenceTransformer models across analyzer instances
_GLOBAL_MODEL_CACHE: Dict[str, Any] = {}


def get_shared_model(model_name: str) -> Any:
    """Retrieve or lazily instantiate and cache a SentenceTransformer model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        SentenceTransformer model instance.
    """
    if model_name not in _GLOBAL_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _GLOBAL_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _GLOBAL_MODEL_CACHE[model_name]


class BaseSemanticScorer(ABC):
    """Abstract base class for semantic similarity and redundancy scoring."""

    @abstractmethod
    def compute_scores(
        self,
        sentences: List[str],
        redundancy_threshold: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Compute semantic similarity and redundancy scores for the given sentences.

        Args:
            sentences: List of segmented sentence strings.
            redundancy_threshold: Optional threshold override for redundancy evaluation.

        Returns:
            Tuple of (semantic_similarity, redundancy_score).
        """
        pass


class SentenceTransformerSemanticScorer(BaseSemanticScorer):
    """Semantic scorer using SentenceTransformer embeddings.

    Computes cosine similarity for all unique sentence pairs and calculates:
    1. semantic_similarity: Average pairwise cosine similarity.
    2. redundancy_score: Proportion of sentence pairs exceeding redundancy_threshold.

    The model is loaded lazily only when two or more sentences are processed,
    and is reused across subsequent analyses.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model: Optional[Any] = None,
        redundancy_threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
    ):
        """Initialize the semantic scorer.

        Args:
            model_name: Name of the SentenceTransformer model to load lazily.
            model: Optional pre-loaded or mock model instance. If provided, model_name
                is not loaded.
            redundancy_threshold: Cosine similarity threshold above which a sentence pair
                is classified as redundant. Default is 0.80 (heuristic).
        """
        self.model_name = model_name
        self._model = model
        self.redundancy_threshold = redundancy_threshold

    def _get_model(self) -> Any:
        """Lazily load and return the SentenceTransformer model instance."""
        if self._model is None:
            self._model = get_shared_model(self.model_name)
        return self._model

    def compute_scores(
        self,
        sentences: List[str],
        redundancy_threshold: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Compute semantic similarity and redundancy scores.

        For fewer than 2 sentences, returns (0.0, 0.0) without loading the model.

        Args:
            sentences: List of segmented sentence strings.
            redundancy_threshold: Optional threshold override. Defaults to self.redundancy_threshold.

        Returns:
            Tuple of (semantic_similarity, redundancy_score).
        """
        if not sentences or len(sentences) < 2:
            return 0.0, 0.0

        threshold = (
            self.redundancy_threshold
            if redundancy_threshold is None
            else redundancy_threshold
        )

        model = self._get_model()
        embeddings = model.encode(sentences, normalize_embeddings=True)

        # Convert to numpy array if not already
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        n = len(sentences)
        pairwise_similarities: List[float] = []

        # Compute cosine similarity for each unique pair (i, j) with 0 <= i < j < n
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                # Clamp to [-1.0, 1.0] to guard against floating-point precision issues
                sim = max(-1.0, min(1.0, sim))
                pairwise_similarities.append(sim)

        num_pairs = len(pairwise_similarities)
        if num_pairs == 0:
            return 0.0, 0.0

        mean_sim = float(np.mean(pairwise_similarities))
        redundant_count = sum(1 for s in pairwise_similarities if s > threshold)
        redundancy_prop = float(redundant_count / num_pairs)

        return mean_sim, redundancy_prop


def get_semantic_scorer(
    model_name: str = DEFAULT_MODEL_NAME,
    redundancy_threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
) -> BaseSemanticScorer:
    """Factory function returning the default semantic scorer instance."""
    return SentenceTransformerSemanticScorer(
        model_name=model_name,
        redundancy_threshold=redundancy_threshold,
    )
