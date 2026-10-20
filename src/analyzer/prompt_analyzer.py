"""Prompt analyzer foundation module.

Extracts measurable structural, instruction, constraint, example requirement,
output format, semantic similarity/redundancy, and task characteristics features from user prompts:
- token_count: Number of tokens computed by the tokenizer.
- word_count: Number of whitespace-delimited words.
- sentence_count: Number of sentences identified by the sentence segmenter.
- prompt_length: Total character count of the exact prompt string (len(prompt)).
- instruction_count: Number of primary task directives requested by the user.
- constraint_count: Number of explicit constraints restricting execution/output.
- has_examples_requirement: Boolean indicating whether examples are explicitly requested.
- has_output_format: Boolean indicating whether a specific output format or structure is requested.
- semantic_similarity: Mean pairwise cosine similarity across all unique sentence pairs.
- redundancy_score: Proportion of unique sentence pairs exceeding the redundancy threshold.
- task_type: Categorical primary task characteristic (e.g. explanation, summarization, etc.).

Designed modularly to allow future feature extractors to be integrated seamlessly.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from src.analyzer.constraint_detector import (
    BaseConstraintDetector,
    get_constraint_detector,
)
from src.analyzer.example_detector import (
    BaseExampleDetector,
    get_example_detector,
)
from src.analyzer.format_detector import (
    BaseOutputFormatDetector,
    get_output_format_detector,
)
from src.analyzer.instruction_detector import (
    BaseInstructionDetector,
    get_instruction_detector,
)
from src.analyzer.semantic_scorer import (
    DEFAULT_REDUNDANCY_THRESHOLD,
    BaseSemanticScorer,
    get_semantic_scorer,
)
from src.analyzer.sentence_segmenter import (
    BaseSentenceSegmenter,
    get_sentence_segmenter,
)
from src.analyzer.task_classifier import (
    BaseTaskClassifier,
    get_task_classifier,
)
from src.evaluation.baseline import count_characters, count_words
from src.utils.tokenizer import BaseTokenizer, get_tokenizer


@dataclass(frozen=True)
class PromptAnalysisResult:
    """Structured data container for prompt analysis features.

    Attributes:
        token_count: Number of tokens calculated by the configured tokenizer.
        word_count: Number of whitespace-separated words in the prompt.
        sentence_count: Number of sentences identified by the segmenter.
        prompt_length: Total character length (len(prompt)) including spaces and punctuation.
        instruction_count: Number of primary task directives detected.
        constraint_count: Number of explicit constraints detected.
        has_examples_requirement: Boolean indicating if examples are explicitly requested.
        has_output_format: Boolean indicating if an output format is explicitly requested.
        semantic_similarity: Mean pairwise cosine similarity across unique sentence pairs.
        redundancy_score: Proportion of unique sentence pairs exceeding redundancy threshold.
        task_type: Primary task type classification of the user prompt.
    """

    token_count: int
    word_count: int
    sentence_count: int
    prompt_length: int
    instruction_count: int
    constraint_count: int
    has_examples_requirement: bool
    has_output_format: bool
    semantic_similarity: float
    redundancy_score: float
    task_type: str = "other"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a standard dictionary format."""
        return asdict(self)


class PromptAnalyzer:
    """Reusable and modular prompt feature analyzer.

    Extracts basic measurable metrics, instructions, constraints, example requirements,
    requested output formats, semantic redundancy metrics, and task characteristics from
    raw prompts without performing any compression or external API calls.
    """

    def __init__(
        self,
        tokenizer: Optional[BaseTokenizer] = None,
        sentence_segmenter: Optional[BaseSentenceSegmenter] = None,
        instruction_detector: Optional[BaseInstructionDetector] = None,
        constraint_detector: Optional[BaseConstraintDetector] = None,
        example_detector: Optional[BaseExampleDetector] = None,
        output_format_detector: Optional[BaseOutputFormatDetector] = None,
        semantic_scorer: Optional[BaseSemanticScorer] = None,
        task_classifier: Optional[BaseTaskClassifier] = None,
        redundancy_threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
    ):
        """Initialize the analyzer with pluggable components.

        Args:
            tokenizer: Tokenizer instance for token counting. Defaults to default TiktokenTokenizer.
            sentence_segmenter: Segmenter instance for sentence counting. Defaults to RegexSentenceSegmenter.
            instruction_detector: Detector instance for counting primary tasks. Defaults to RuleBasedInstructionDetector.
            constraint_detector: Detector instance for counting explicit constraints. Defaults to RuleBasedConstraintDetector.
            example_detector: Detector instance for identifying example requirements. Defaults to RuleBasedExampleDetector.
            output_format_detector: Detector instance for identifying requested output formats. Defaults to RuleBasedOutputFormatDetector.
            semantic_scorer: Scorer instance for pairwise sentence similarity and redundancy. Defaults to SentenceTransformerSemanticScorer.
            task_classifier: Classifier instance for identifying the primary task type. Defaults to RuleBasedTaskClassifier.
            redundancy_threshold: Cosine similarity threshold above which a sentence pair is considered redundant. Defaults to 0.80.
        """
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()
        self.sentence_segmenter = (
            sentence_segmenter
            if sentence_segmenter is not None
            else get_sentence_segmenter()
        )
        self.instruction_detector = (
            instruction_detector
            if instruction_detector is not None
            else get_instruction_detector()
        )
        self.constraint_detector = (
            constraint_detector
            if constraint_detector is not None
            else get_constraint_detector()
        )
        self.example_detector = (
            example_detector
            if example_detector is not None
            else get_example_detector()
        )
        self.output_format_detector = (
            output_format_detector
            if output_format_detector is not None
            else get_output_format_detector()
        )
        self.redundancy_threshold = redundancy_threshold
        self.semantic_scorer = (
            semantic_scorer
            if semantic_scorer is not None
            else get_semantic_scorer(redundancy_threshold=redundancy_threshold)
        )
        self.task_classifier = (
            task_classifier
            if task_classifier is not None
            else get_task_classifier()
        )

    def analyze(self, prompt: str) -> Dict[str, Any]:
        """Analyze a prompt and extract measurable features.

        Args:
            prompt: Raw user input prompt string.

        Returns:
            Dictionary containing token_count, word_count, sentence_count, prompt_length,
            instruction_count, constraint_count, has_examples_requirement, has_output_format,
            semantic_similarity, redundancy_score, and task_type.
        """
        result = self.analyze_structured(prompt)
        return result.to_dict()

    def analyze_structured(self, prompt: str) -> PromptAnalysisResult:
        """Analyze a prompt and return a structured PromptAnalysisResult dataclass.

        Args:
            prompt: Raw user input prompt string.

        Returns:
            PromptAnalysisResult instance.
        """
        if prompt is None:
            prompt = ""

        # Token count reuses Phase 2 tokenizer logic
        token_cnt = self.tokenizer.count_tokens(prompt)

        # Word count reuses Phase 2 word counting logic
        word_cnt = count_words(prompt)

        # Sentence segmentation and count using pluggable deterministic local segmenter
        sentences = self.sentence_segmenter.split_sentences(prompt)
        sentence_cnt = len(sentences)

        # Prompt length is explicitly defined as exact character length (len(prompt))
        prompt_len = count_characters(prompt)

        # Instruction count uses pluggable deterministic rule-based detector
        instruction_cnt = self.instruction_detector.count_instructions(prompt)

        # Constraint count uses pluggable deterministic rule-based detector
        constraint_cnt = self.constraint_detector.count_constraints(prompt)

        # Example requirement detection uses deterministic rule-based detector
        has_examples = self.example_detector.has_examples_requirement(prompt)

        # Output format detection uses deterministic rule-based detector
        has_format = self.output_format_detector.has_output_format(prompt)

        # Semantic similarity and redundancy scoring across unique sentence pairs
        sem_sim, red_score = self.semantic_scorer.compute_scores(
            sentences, redundancy_threshold=self.redundancy_threshold
        )

        # Primary task type classification uses pluggable deterministic rule-based classifier
        task_type = self.task_classifier.classify(prompt)

        return PromptAnalysisResult(
            token_count=token_cnt,
            word_count=word_cnt,
            sentence_count=sentence_cnt,
            prompt_length=prompt_len,
            instruction_count=instruction_cnt,
            constraint_count=constraint_cnt,
            has_examples_requirement=has_examples,
            has_output_format=has_format,
            semantic_similarity=sem_sim,
            redundancy_score=red_score,
            task_type=task_type,
        )


def analyze_prompt(
    prompt: str,
    tokenizer: Optional[BaseTokenizer] = None,
    sentence_segmenter: Optional[BaseSentenceSegmenter] = None,
    instruction_detector: Optional[BaseInstructionDetector] = None,
    constraint_detector: Optional[BaseConstraintDetector] = None,
    example_detector: Optional[BaseExampleDetector] = None,
    output_format_detector: Optional[BaseOutputFormatDetector] = None,
    semantic_scorer: Optional[BaseSemanticScorer] = None,
    task_classifier: Optional[BaseTaskClassifier] = None,
    redundancy_threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
) -> Dict[str, Any]:
    """Convenience function to analyze a prompt with default or custom components.

    Args:
        prompt: Raw user input prompt string.
        tokenizer: Optional custom tokenizer.
        sentence_segmenter: Optional custom sentence segmenter.
        instruction_detector: Optional custom instruction detector.
        constraint_detector: Optional custom constraint detector.
        example_detector: Optional custom example detector.
        output_format_detector: Optional custom output format detector.
        semantic_scorer: Optional custom semantic scorer.
        task_classifier: Optional custom task classifier.
        redundancy_threshold: Optional redundancy threshold.

    Returns:
        Dictionary with all extracted features including task_type.
    """
    analyzer = PromptAnalyzer(
        tokenizer=tokenizer,
        sentence_segmenter=sentence_segmenter,
        instruction_detector=instruction_detector,
        constraint_detector=constraint_detector,
        example_detector=example_detector,
        output_format_detector=output_format_detector,
        semantic_scorer=semantic_scorer,
        task_classifier=task_classifier,
        redundancy_threshold=redundancy_threshold,
    )
    return analyzer.analyze(prompt)
