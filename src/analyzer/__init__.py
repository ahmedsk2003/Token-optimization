"""Prompt Analyzer module.

Exposes PromptAnalyzer, PromptAnalysisResult, analyze_prompt, and segmenter/detector/scorer interfaces.
"""

from src.analyzer.constraint_detector import (
    BaseConstraintDetector,
    DetectedConstraint,
    RuleBasedConstraintDetector,
    get_constraint_detector,
)
from src.analyzer.example_detector import (
    BaseExampleDetector,
    RuleBasedExampleDetector,
    get_example_detector,
)
from src.analyzer.format_detector import (
    BaseOutputFormatDetector,
    RuleBasedOutputFormatDetector,
    get_output_format_detector,
)
from src.analyzer.instruction_detector import (
    BaseInstructionDetector,
    RuleBasedInstructionDetector,
    get_instruction_detector,
)
from src.analyzer.prompt_analyzer import (
    PromptAnalysisResult,
    PromptAnalyzer,
    analyze_prompt,
)
from src.analyzer.semantic_scorer import (
    DEFAULT_REDUNDANCY_THRESHOLD,
    BaseSemanticScorer,
    SentenceTransformerSemanticScorer,
    get_semantic_scorer,
)
from src.analyzer.sentence_segmenter import (
    BaseSentenceSegmenter,
    RegexSentenceSegmenter,
    get_sentence_segmenter,
)

__all__ = [
    "PromptAnalyzer",
    "PromptAnalysisResult",
    "analyze_prompt",
    "BaseSentenceSegmenter",
    "RegexSentenceSegmenter",
    "get_sentence_segmenter",
    "BaseInstructionDetector",
    "RuleBasedInstructionDetector",
    "get_instruction_detector",
    "BaseConstraintDetector",
    "DetectedConstraint",
    "RuleBasedConstraintDetector",
    "get_constraint_detector",
    "BaseExampleDetector",
    "RuleBasedExampleDetector",
    "get_example_detector",
    "BaseOutputFormatDetector",
    "RuleBasedOutputFormatDetector",
    "get_output_format_detector",
    "BaseSemanticScorer",
    "SentenceTransformerSemanticScorer",
    "get_semantic_scorer",
    "DEFAULT_REDUNDANCY_THRESHOLD",
]
