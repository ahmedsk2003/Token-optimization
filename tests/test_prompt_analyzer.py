"""Tests for Prompt Analyzer Foundation, Instruction/Constraint Analysis, Example/Format Detection, and Semantic Scoring."""

import unittest
from typing import List, Optional, Tuple

import numpy as np

from src.analyzer import (
    DEFAULT_REDUNDANCY_THRESHOLD,
    SUPPORTED_TASK_TYPES,
    BaseConstraintDetector,
    BaseExampleDetector,
    BaseInstructionDetector,
    BaseOutputFormatDetector,
    BaseSemanticScorer,
    BaseSentenceSegmenter,
    BaseTaskClassifier,
    DetectedConstraint,
    PromptAnalysisResult,
    PromptAnalyzer,
    RuleBasedTaskClassifier,
    SentenceTransformerSemanticScorer,
    analyze_prompt,
    get_semantic_scorer,
    get_sentence_segmenter,
    get_task_classifier,
)
from src.evaluation.baseline import measure_prompt
from src.utils.tokenizer import BaseTokenizer, get_tokenizer


class DummySentenceSegmenter(BaseSentenceSegmenter):
    """Custom segmenter test double to verify segmenter replaceability."""

    def split_sentences(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        return [s.strip() for s in text.split("||") if s.strip()]


class DummyCustomTokenizer(BaseTokenizer):
    """Custom tokenizer test double to verify tokenizer replaceability."""

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split("-"))

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        return [len(part) for part in text.split("-")]

    def decode(self, tokens: List[int]) -> str:
        return "-".join(str(t) for t in tokens)


class DummyInstructionDetector(BaseInstructionDetector):
    """Custom instruction detector test double."""

    def detect_instructions(self, prompt: str) -> List[str]:
        return ["dummy_instruction_1", "dummy_instruction_2"]


class DummyConstraintDetector(BaseConstraintDetector):
    """Custom constraint detector test double."""

    def detect_constraints(self, prompt: str) -> List[DetectedConstraint]:
        return [
            DetectedConstraint(category="dummy", matched_text="dummy constraint", start=0, end=16)
        ]


class DummyExampleDetector(BaseExampleDetector):
    """Custom example detector test double."""

    def has_examples_requirement(self, prompt: str) -> bool:
        return True


class DummyOutputFormatDetector(BaseOutputFormatDetector):
    """Custom output format detector test double."""

    def has_output_format(self, prompt: str) -> bool:
        return True


class DummySemanticScorer(BaseSemanticScorer):
    """Custom semantic scorer test double returning deterministic scores."""

    def __init__(self, sim: float = 0.5, red: float = 0.25):
        self.sim = sim
        self.red = red

    def compute_scores(
        self, sentences: List[str], redundancy_threshold: Optional[float] = None
    ) -> Tuple[float, float]:
        if not sentences or len(sentences) < 2:
            return 0.0, 0.0
        return self.sim, self.red


class MockEmbeddingModel:
    """Mock SentenceTransformer model returning deterministic normalized vectors for offline unit tests."""

    def __init__(self):
        self.call_count = 0

    def encode(self, sentences: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        self.call_count += 1
        vectors = []
        for s in sentences:
            s_clean = s.strip().lower()
            if "repeat" in s_clean or "identical" in s_clean:
                v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif "similar" in s_clean or "describe ml" in s_clean:
                # Cosine similarity with [1, 0, 0, 0] is 0.95, orthogonal to [0, 1, 0, 0]
                v = np.array([0.95, 0.0, 0.31225, 0.0], dtype=np.float32)
            elif "unrelated" in s_clean or "weather" in s_clean:
                # Orthogonal vector (cosine similarity 0.0)
                v = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            elif "explain ml" in s_clean or "explain machine learning" in s_clean:
                v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif "third" in s_clean:
                # Distinct vector
                v = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            else:
                # Deterministic normalized vector based on length
                val = float(len(s_clean) % 5 + 1)
                v = np.array([val, 1.0, 0.5, 0.2], dtype=np.float32)

            if normalize_embeddings:
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
            vectors.append(v)
        return np.array(vectors)


class TestPromptAnalyzer(unittest.TestCase):
    """Test suite for PromptAnalyzer (Phase 3A, Phase 3B, and Phase 3C features)."""

    def setUp(self):
        self.tokenizer = get_tokenizer("cl100k_base")
        # Use DummySemanticScorer in standard analyzer unit tests for rapid offline test runs
        self.dummy_scorer = DummySemanticScorer(sim=0.0, red=0.0)
        self.analyzer = PromptAnalyzer(
            tokenizer=self.tokenizer,
            semantic_scorer=self.dummy_scorer,
        )

    # ------------------ Phase 3A Foundation Tests ------------------

    def test_empty_prompt(self):
        result = self.analyzer.analyze("")
        self.assertEqual(
            result,
            {
                "token_count": 0,
                "word_count": 0,
                "sentence_count": 0,
                "prompt_length": 0,
                "instruction_count": 0,
                "constraint_count": 0,
                "has_examples_requirement": False,
                "has_output_format": False,
                "semantic_similarity": 0.0,
                "redundancy_score": 0.0,
                "task_type": "other",
            },
        )

    def test_whitespace_only_prompt(self):
        prompt = "   \n\t  "
        result = self.analyzer.analyze(prompt)
        self.assertEqual(result["word_count"], 0)
        self.assertEqual(result["sentence_count"], 0)
        self.assertEqual(result["prompt_length"], len(prompt))
        self.assertEqual(result["instruction_count"], 0)
        self.assertEqual(result["constraint_count"], 0)
        self.assertFalse(result["has_examples_requirement"])
        self.assertFalse(result["has_output_format"])
        self.assertEqual(result["semantic_similarity"], 0.0)
        self.assertEqual(result["redundancy_score"], 0.0)

    def test_simple_single_sentence_prompt(self):
        prompt = "Summarize the key points of the report."
        result = self.analyzer.analyze(prompt)

        self.assertEqual(result["token_count"], self.tokenizer.count_tokens(prompt))
        self.assertEqual(result["word_count"], len(prompt.split()))
        self.assertEqual(result["sentence_count"], 1)
        self.assertEqual(result["prompt_length"], len(prompt))
        self.assertEqual(result["instruction_count"], 1)
        self.assertEqual(result["constraint_count"], 0)
        self.assertFalse(result["has_examples_requirement"])
        self.assertFalse(result["has_output_format"])
        self.assertEqual(result["semantic_similarity"], 0.0)
        self.assertEqual(result["redundancy_score"], 0.0)

    def test_multiple_sentences_with_various_punctuation(self):
        prompt = "What is machine learning? It is a subfield of AI! Please explain in detail."
        result = self.analyzer.analyze(prompt)

        self.assertEqual(result["sentence_count"], 3)
        self.assertEqual(result["word_count"], len(prompt.split()))
        self.assertEqual(result["prompt_length"], len(prompt))
        self.assertEqual(result["token_count"], self.tokenizer.count_tokens(prompt))

    def test_multiple_consecutive_punctuation_marks(self):
        prompt = "Is this real?! Yes, absolutely... Let's proceed."
        result = self.analyzer.analyze(prompt)

        self.assertEqual(result["sentence_count"], 3)
        self.assertEqual(result["word_count"], len(prompt.split()))

    def test_multiline_prompt_with_newlines(self):
        prompt = "Task: Write a function.\nStep 1: Parse the input.\nStep 2: Return output."
        result = self.analyzer.analyze(prompt)

        self.assertEqual(result["sentence_count"], 3)
        self.assertEqual(result["prompt_length"], len(prompt))
        self.assertEqual(result["word_count"], len(prompt.split()))

    def test_deterministic_repeated_analysis(self):
        prompt = "Analyze the reproducibility of feature extraction over multiple runs."
        run1 = self.analyzer.analyze(prompt)
        run2 = self.analyzer.analyze(prompt)
        run3 = analyze_prompt(
            prompt,
            tokenizer=self.tokenizer,
            semantic_scorer=self.dummy_scorer,
        )

        self.assertEqual(run1, run2)
        self.assertEqual(run1, run3)
        self.assertEqual(
            set(run1.keys()),
            {
                "token_count",
                "word_count",
                "sentence_count",
                "prompt_length",
                "instruction_count",
                "constraint_count",
                "has_examples_requirement",
                "has_output_format",
                "semantic_similarity",
                "redundancy_score",
                "task_type",
            },
        )

    def test_consistency_with_phase2_baseline_metrics(self):
        test_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a python function to sort a list of dictionaries by key.\nInclude docstring.",
            "Special characters: !@#$%^&*()_+{}|:\"<>?~`\n\t",
            "   Leading and trailing spaces   ",
        ]

        for prompt in test_prompts:
            analyzer_result = self.analyzer.analyze(prompt)
            baseline_result = measure_prompt(prompt, tokenizer=self.tokenizer)

            self.assertEqual(
                analyzer_result["token_count"], baseline_result.token_count
            )
            self.assertEqual(
                analyzer_result["word_count"], baseline_result.word_count
            )
            self.assertEqual(
                analyzer_result["prompt_length"], baseline_result.character_count
            )

    # ------------------ Phase 3B Instruction & Constraint Tests ------------------

    def test_single_task_prompt_no_constraints(self):
        prompt = "Explain quantum computing."
        result = self.analyzer.analyze(prompt)

        self.assertEqual(result["instruction_count"], 1)
        self.assertEqual(result["constraint_count"], 0)
        self.assertFalse(result["has_examples_requirement"])
        self.assertFalse(result["has_output_format"])

    def test_user_canonical_prompt_with_multiple_constraints(self):
        prompt = "Explain machine learning for a beginner. Include two examples. Use bullet points. Do not use mathematical notation."
        result = self.analyzer.analyze(prompt)

        self.assertEqual(result["instruction_count"], 1)
        self.assertEqual(result["constraint_count"], 4)
        self.assertTrue(result["has_examples_requirement"])
        self.assertTrue(result["has_output_format"])

    def test_length_and_word_constraints(self):
        prompt1 = "Summarize the history of computing in under 100 words."
        result1 = self.analyzer.analyze(prompt1)
        self.assertEqual(result1["instruction_count"], 1)
        self.assertEqual(result1["constraint_count"], 1)

        prompt2 = "Describe solar energy. Limit to 3 sentences."
        result2 = self.analyzer.analyze(prompt2)
        self.assertEqual(result2["instruction_count"], 1)
        self.assertEqual(result2["constraint_count"], 1)

        # Hyphenated word count constraint
        prompt3 = "Write a 500-word article about cybersecurity."
        result3 = self.analyzer.analyze(prompt3)
        self.assertEqual(result3["instruction_count"], 1)
        self.assertEqual(result3["constraint_count"], 1)
        self.assertFalse(result3["has_examples_requirement"])
        self.assertFalse(result3["has_output_format"])

        # Explicit words count constraint
        prompt4 = "Draft an introduction in 500 words."
        result4 = self.analyzer.analyze(prompt4)
        self.assertEqual(result4["instruction_count"], 1)
        self.assertEqual(result4["constraint_count"], 1)

        # Maximum words constraint
        prompt5 = "Write an essay about AI, maximum 300 words."
        result5 = self.analyzer.analyze(prompt5)
        self.assertEqual(result5["instruction_count"], 1)
        self.assertEqual(result5["constraint_count"], 1)
        self.assertTrue(result5["has_output_format"])

    def test_format_constraints(self):
        prompt1 = "List top 5 programming languages in markdown format."
        result1 = self.analyzer.analyze(prompt1)
        self.assertEqual(result1["instruction_count"], 1)
        self.assertEqual(result1["constraint_count"], 1)
        self.assertTrue(result1["has_output_format"])

        prompt2 = "Generate sample user data as a json object."
        result2 = self.analyzer.analyze(prompt2)
        self.assertEqual(result2["instruction_count"], 1)
        self.assertEqual(result2["constraint_count"], 1)
        self.assertTrue(result2["has_output_format"])

        prompt3 = "Compare SQL and NoSQL in a table."
        result3 = self.analyzer.analyze(prompt3)
        self.assertEqual(result3["instruction_count"], 1)
        self.assertEqual(result3["constraint_count"], 1)
        self.assertTrue(result3["has_output_format"])

    def test_inclusion_and_exclusion_constraints(self):
        prompt = "Write a Python script for web scraping. Include error handling. Do not use external libraries."
        result = self.analyzer.analyze(prompt)

        self.assertEqual(result["instruction_count"], 1)
        self.assertEqual(result["constraint_count"], 3)

    def test_multiple_requirements_in_one_prompt(self):
        prompt = (
            "Please explain how neural networks work for high school students. "
            "Provide two examples. Use bullet points. "
            "Keep it under 200 words. Avoid complex mathematical formulas."
        )
        result = self.analyzer.analyze(prompt)

        self.assertEqual(result["instruction_count"], 1)
        self.assertEqual(result["constraint_count"], 5)
        self.assertTrue(result["has_examples_requirement"])
        self.assertTrue(result["has_output_format"])

    def test_interrogative_prompt_no_constraints(self):
        prompt = "What is the capital of France?"
        result = self.analyzer.analyze(prompt)

        self.assertEqual(result["instruction_count"], 1)
        self.assertEqual(result["constraint_count"], 0)
        self.assertFalse(result["has_examples_requirement"])
        self.assertFalse(result["has_output_format"])

    def test_modular_pluggable_instruction_and_constraint_detectors(self):
        custom_inst = DummyInstructionDetector()
        custom_const = DummyConstraintDetector()
        custom_ex = DummyExampleDetector()
        custom_fmt = DummyOutputFormatDetector()
        custom_semantic = DummySemanticScorer(sim=0.77, red=0.5)
        custom_analyzer = PromptAnalyzer(
            tokenizer=self.tokenizer,
            instruction_detector=custom_inst,
            constraint_detector=custom_const,
            example_detector=custom_ex,
            output_format_detector=custom_fmt,
            semantic_scorer=custom_semantic,
        )

        result = custom_analyzer.analyze("Sentence one. Sentence two.")
        self.assertEqual(result["instruction_count"], 2)
        self.assertEqual(result["constraint_count"], 1)
        self.assertTrue(result["has_examples_requirement"])
        self.assertTrue(result["has_output_format"])
        self.assertEqual(result["semantic_similarity"], 0.77)
        self.assertEqual(result["redundancy_score"], 0.5)

    def test_structured_result_dataclass_fields(self):
        prompt = "Explain relativity for kids. Include an example."
        structured = self.analyzer.analyze_structured(prompt)

        self.assertIsInstance(structured, PromptAnalysisResult)
        self.assertEqual(structured.token_count, self.tokenizer.count_tokens(prompt))
        self.assertEqual(structured.word_count, len(prompt.split()))
        self.assertEqual(structured.sentence_count, 2)
        self.assertEqual(structured.prompt_length, len(prompt))
        self.assertEqual(structured.instruction_count, 1)
        self.assertEqual(structured.constraint_count, 2)
        self.assertTrue(structured.has_examples_requirement)
        self.assertFalse(structured.has_output_format)
        self.assertEqual(structured.semantic_similarity, 0.0)
        self.assertEqual(structured.redundancy_score, 0.0)
        self.assertEqual(structured.task_type, "explanation")
        self.assertEqual(structured.to_dict(), self.analyzer.analyze(prompt))

    # ------------------ Example Requirement Detection Tests ------------------

    def test_examples_requirement_detection_positive(self):
        positive_prompts = [
            "Explain machine learning for a beginner. Include two examples.",
            "Provide an example of polymorphism in Java.",
            "Give 3 examples of renewable energy sources.",
            "Explain Python decorators with code examples.",
            "Illustrate how DNS resolution works with an example.",
            "Can you give an example of an NP-hard problem?",
            "Explain recursion using practical examples.",
            "Supply several examples of machine learning applications.",
            "Explain microservices accompanied by two examples.",
            "Describe photosynthesis. Examples are required.",
            "Provide at least one example of design patterns in C++.",
        ]

        for prompt in positive_prompts:
            with self.subTest(prompt=prompt):
                result = self.analyzer.analyze(prompt)
                self.assertTrue(
                    result["has_examples_requirement"],
                    f"Expected has_examples_requirement=True for: {prompt}",
                )

    def test_examples_requirement_detection_negative(self):
        negative_prompts = [
            "For example, Python is interpreted. What about Java?",
            "Here is an example: def foo(): pass. Explain what it does.",
            "In the following example, identify the security vulnerability.",
            "Do not include examples in the summary.",
            "Explain photosynthesis without any examples.",
            "No examples are needed for this explanation.",
            "Explain machine learning.",
            "Write a 500-word article about cybersecurity.",
            "Take the example of Apple Inc. How do they handle supply chains?",
        ]

        for prompt in negative_prompts:
            with self.subTest(prompt=prompt):
                result = self.analyzer.analyze(prompt)
                self.assertFalse(
                    result["has_examples_requirement"],
                    f"Expected has_examples_requirement=False for: {prompt}",
                )

    # ------------------ Output Format Detection Tests ------------------

    def test_output_format_detection_positive(self):
        format_prompts = [
            # Bullet / numbered lists
            ("Explain ML. Use bullet points.", True),
            ("Summarize this text in 5 bullet points.", True),
            ("List the installation steps in a numbered list.", True),
            ("Provide the advantages as a bulleted list.", True),
            ("Format your answer as a list.", True),
            # Tables
            ("Compare Python and JavaScript in a table.", True),
            ("Present the benchmark results in tabular format.", True),
            ("Create a table comparing SQL and NoSQL databases.", True),
            ("Format the findings as a table.", True),
            # JSON / XML
            ("Generate sample user profile data as valid JSON.", True),
            ("Output the response in XML format.", True),
            ("Convert this CSV string to JSON.", True),
            ("Format the output as a json object.", True),
            # Markdown
            ("Format your answer in Markdown.", True),
            ("Use markdown formatting for your response.", True),
            ("Render the output as markdown.", True),
            # Headings / Sections
            ("Explain cloud architecture with headings and subheadings.", True),
            ("Structure your response into clear sections.", True),
            ("Organize the guide with headings.", True),
            # Report / Essay structures
            ("Write an essay about renewable energy solutions.", True),
            ("Write a report on cybersecurity trends in 2026.", True),
            ("Format the analysis as a formal report.", True),
            ("Structure the response as an essay.", True),
            # Mixed with exclusions of other formats
            ("Use bullet points. Do not use tables.", True),
        ]

        for prompt, expected in format_prompts:
            with self.subTest(prompt=prompt):
                result = self.analyzer.analyze(prompt)
                self.assertEqual(
                    result["has_output_format"],
                    expected,
                    f"Expected has_output_format={expected} for: {prompt}",
                )

    def test_output_format_detection_negative(self):
        negative_format_prompts = [
            # Formats merely mentioned as topics
            "What is JSON?",
            "Explain how XML parsing works.",
            "Compare JSON and XML specifications.",
            "What are the advantages of Markdown?",
            "Why do people use bullet points in slides?",
            "Discuss the history of the personal essay.",
            "Explain what an annual report is in corporate finance.",
            # How-to tutorial questions
            "How do I create a table in HTML?",
            "How to write an essay?",
            "Explain how to structure a report.",
            # Input references
            "Parse this JSON string and return the id.",
            "Given this table of numbers, find the average.",
            "Proofread my essay for grammatical errors.",
            "Summarize this report for the executive team.",
            # Prohibitions only
            "Do not use bullet points or tables.",
            "Explain quantum computing without using Markdown.",
            # Plain prompts
            "Explain machine learning.",
            "Write a 500-word article about cybersecurity.",
        ]

        for prompt in negative_format_prompts:
            with self.subTest(prompt=prompt):
                result = self.analyzer.analyze(prompt)
                self.assertFalse(
                    result["has_output_format"],
                    f"Expected has_output_format=False for: {prompt}",
                )

    # ------------------ Ambiguous & Edge Case Tests ------------------

    def test_example_and_format_ambiguous_and_edge_cases(self):
        test_cases = [
            (
                "For example, consider climate change. Write a report on renewable solutions.",
                False,  # "For example" is discourse marker, not a request
                True,   # "Write a report" requests report structure
            ),
            (
                "Explain machine learning. Give an example in a table.",
                True,   # "Give an example"
                True,   # "in a table"
            ),
            (
                "Summarize this report. Do not format as a table or use bullet points.",
                False,  # No example request
                False,  # "this report" is input; table & bullets are negated
            ),
            (
                "Can you provide an example? Please be concise.",
                True,   # "provide an example"
                False,  # "concise" is style constraint, not a requested structure
            ),
            (
                "Discuss how XML is used in modern web development.",
                False,  # XML is topic
                False,  # No format directive
            ),
            (
                "Please respond in JSON.",
                False,  # No example request
                True,   # "respond in JSON"
            ),
            (
                "What is an essay and how is it different from a report?",
                False,  # No example request
                False,  # Topic comparison of writing genres
            ),
            (
                "Structure your analysis with headings and bullet points.",
                False,  # No example request
                True,   # Both headings and bullet points requested
            ),
            (
                "Explain this code example: print('hello')",
                False,  # Input code reference
                False,  # No format request
            ),
            (
                "Make a table comparing SQL and NoSQL databases.",
                False,  # No example request
                True,   # Table requested
            ),
            (
                "Explain how to write an essay.",
                False,  # No example request
                False,  # "how to write an essay" is tutorial question
            ),
            (
                "Write an essay about AI ethics.",
                False,  # No example request
                True,   # Essay structure requested
            ),
            (
                "Provide at least two examples.",
                True,   # Explicit count of examples requested
                False,  # No format requested
            ),
            (
                "Do not include examples. Write in plain text.",
                False,  # Examples explicitly excluded
                False,  # Plain text is default/unformatted
            ),
        ]

        for prompt, exp_examples, exp_format in test_cases:
            with self.subTest(prompt=prompt):
                result = self.analyzer.analyze(prompt)
                self.assertEqual(
                    result["has_examples_requirement"],
                    exp_examples,
                    f"Example detection mismatch for: {prompt}",
                )
                self.assertEqual(
                    result["has_output_format"],
                    exp_format,
                    f"Format detection mismatch for: {prompt}",
                )


class TestSemanticScorer(unittest.TestCase):
    """Unit test suite for Phase 3C Semantic Similarity and Redundancy Scoring."""

    def setUp(self):
        self.mock_model = MockEmbeddingModel()
        self.scorer = SentenceTransformerSemanticScorer(
            model=self.mock_model,
            redundancy_threshold=0.80,
        )

    def test_empty_prompt_and_empty_sentences(self):
        # Empty sentences list returns 0.0 for both
        sim, red = self.scorer.compute_scores([])
        self.assertEqual(sim, 0.0)
        self.assertEqual(red, 0.0)
        # Model should not have been called
        self.assertEqual(self.mock_model.call_count, 0)

    def test_single_sentence_prompt(self):
        # Fewer than 2 sentences returns 0.0 for both without calling model
        sim, red = self.scorer.compute_scores(["Explain machine learning."])
        self.assertEqual(sim, 0.0)
        self.assertEqual(red, 0.0)
        self.assertEqual(self.mock_model.call_count, 0)

    def test_repeated_identical_sentences(self):
        # Repeated identical sentences yield cosine similarity = 1.0, redundancy = 1.0
        sentences = [
            "Repeat this exact sentence.",
            "Repeat this exact sentence.",
        ]
        sim, red = self.scorer.compute_scores(sentences)
        self.assertAlmostEqual(sim, 1.0, places=4)
        self.assertEqual(red, 1.0)
        self.assertEqual(self.mock_model.call_count, 1)

    def test_semantically_similar_sentences(self):
        # Similar sentences yield similarity > 0.80 (0.95 in mock), redundancy = 1.0
        sentences = [
            "Explain machine learning.",
            "Describe ML similarly.",
        ]
        sim, red = self.scorer.compute_scores(sentences)
        self.assertGreater(sim, 0.80)
        self.assertEqual(red, 1.0)

    def test_unrelated_sentences(self):
        # Unrelated orthogonal sentences yield similarity ~ 0.0, redundancy = 0.0
        sentences = [
            "Explain machine learning.",
            "The weather is unrelated today.",
        ]
        sim, red = self.scorer.compute_scores(sentences)
        self.assertAlmostEqual(sim, 0.0, places=4)
        self.assertEqual(red, 0.0)

    def test_mixed_sentence_pairs(self):
        # 3 sentences: s0 & s1 similar (0.95), s0 & s2 unrelated (0.0), s1 & s2 unrelated (0.0)
        # Pairs: 3. Redundant pairs: 1 (s0, s1).
        # Mean similarity: (0.95 + 0.0 + 0.0) / 3 = 0.3167
        # Redundancy proportion: 1 / 3 = 0.3333...
        sentences = [
            "Explain machine learning.",
            "Describe ML similarly.",
            "The weather is unrelated today.",
        ]
        sim, red = self.scorer.compute_scores(sentences)
        self.assertAlmostEqual(sim, 0.95 / 3.0, places=3)
        self.assertAlmostEqual(red, 1.0 / 3.0, places=3)

    def test_configurable_redundancy_threshold(self):
        # Pair similarity is 0.95.
        # Under threshold=0.90, redundancy = 1.0
        # Under threshold=0.98, redundancy = 0.0
        sentences = [
            "Explain machine learning.",
            "Describe ML similarly.",
        ]
        sim, red_default = self.scorer.compute_scores(sentences, redundancy_threshold=0.90)
        self.assertEqual(red_default, 1.0)

        sim, red_strict = self.scorer.compute_scores(sentences, redundancy_threshold=0.98)
        self.assertEqual(red_strict, 0.0)

    def test_lazy_loading_and_instance_reuse(self):
        # Scorer initialized without model should have _model as None
        fresh_scorer = SentenceTransformerSemanticScorer(
            model=None,
            redundancy_threshold=0.80,
        )
        self.assertIsNone(fresh_scorer._model)

        # Single sentence or empty sentence does not trigger load
        sim, red = fresh_scorer.compute_scores(["Single sentence."])
        self.assertEqual(sim, 0.0)
        self.assertEqual(red, 0.0)
        self.assertIsNone(fresh_scorer._model)

    def test_prompt_analyzer_integration_with_mock_scorer(self):
        tokenizer = get_tokenizer("cl100k_base")
        analyzer = PromptAnalyzer(
            tokenizer=tokenizer,
            semantic_scorer=self.scorer,
        )

        prompt = "Explain machine learning. Describe ML similarly."
        result = analyzer.analyze(prompt)

        self.assertIn("semantic_similarity", result)
        self.assertIn("redundancy_score", result)
        self.assertGreater(result["semantic_similarity"], 0.80)
        self.assertEqual(result["redundancy_score"], 1.0)


class TestSemanticScorerRealModel(unittest.TestCase):
    """Integration test verifying real SentenceTransformer embedding model execution."""

    def test_real_model_pairwise_scoring(self):
        # Uses real cached SentenceTransformer model
        scorer = get_semantic_scorer()
        sentences = [
            "Explain machine learning.",
            "The weather is sunny today.",
        ]
        sim, red = scorer.compute_scores(sentences)

        self.assertIsInstance(sim, float)
        self.assertIsInstance(red, float)
        # ML and weather should have very low similarity and 0 redundancy
        self.assertLess(sim, 0.30)
        self.assertEqual(red, 0.0)


class DummyTaskClassifier(BaseTaskClassifier):
    """Custom classifier test double to verify task classifier replaceability."""

    def __init__(self, fixed_task: str = "custom_task"):
        self.fixed_task = fixed_task

    def classify(self, prompt: str) -> str:
        return self.fixed_task


class TestTaskClassifier(unittest.TestCase):
    """Comprehensive unit tests for RuleBasedTaskClassifier across all categories."""

    def setUp(self):
        self.classifier = RuleBasedTaskClassifier()
        self.analyzer = PromptAnalyzer()

    # 1. Explanation
    def test_task_explanation(self):
        prompts = [
            "Explain quantum computing in simple terms.",
            "How does the immune system protect the human body against viral infections?",
            "Please elaborate on the principles of object-oriented programming.",
            "Describe how photosynthesis works in green plants.",
            "Walk me through the process of DNA replication.",
            "Help me understand why the sky is blue during the day.",
        ]
        for p in prompts:
            with self.subTest(prompt=p):
                self.assertEqual(self.classifier.classify(p), "explanation")

    # 2. Summarization
    def test_task_summarization(self):
        prompts = [
            "Summarize the key points of the quarterly earnings report.",
            "Provide a brief summary of the attached research paper.",
            "TL;DR of this article please: The economic forecast shows continued growth...",
            "Condense the following text into two concise sentences.",
            "Give me the key takeaways of the meeting notes.",
            "Briefly summarize what happened in chapter 4.",
        ]
        for p in prompts:
            with self.subTest(prompt=p):
                self.assertEqual(self.classifier.classify(p), "summarization")

    # 3. Comparison
    def test_task_comparison(self):
        prompts = [
            "Compare React and Vue for building single-page applications.",
            "What is the difference between TCP and UDP?",
            "Compare and contrast supervised and unsupervised machine learning.",
            "PostgreSQL vs MongoDB: which database is better for analytics?",
            "Distinguish between synchronous and asynchronous programming.",
            "What are the pros and cons of microservices versus monoliths?",
            "Which is faster: C++ or Python?",
        ]
        for p in prompts:
            with self.subTest(prompt=p):
                self.assertEqual(self.classifier.classify(p), "comparison")

    # 4. Translation
    def test_task_translation(self):
        prompts = [
            "Translate the following user guide into Spanish: 'Click save to continue.'",
            "Provide the French translation of this welcome message.",
            "How do you say 'Where is the train station?' in German?",
            "Translate this paragraph from English to Japanese.",
            "Translate into Italian: Hello, nice to meet you.",
        ]
        for p in prompts:
            with self.subTest(prompt=p):
                self.assertEqual(self.classifier.classify(p), "translation")

    # 5. Code Generation
    def test_task_code_generation(self):
        prompts = [
            "Write a Python script to download files from an S3 bucket.",
            "Implement a binary search tree in C++.",
            "Create a REST API endpoint using FastAPI.",
            "Translate this Python function into Rust code.",
            "Generate a regex to validate international phone numbers.",
            "Write an SQL query to find the top 5 customers by sales volume.",
            "Code a quicksort algorithm in Java.",
        ]
        for p in prompts:
            with self.subTest(prompt=p):
                self.assertEqual(self.classifier.classify(p), "code_generation")

    # 6. Debugging
    def test_task_debugging(self):
        prompts = [
            "Debug this Python snippet and fix the IndexError: list index out of range.",
            "Why is my JavaScript function returning undefined instead of the object?",
            "Fix the bug in this SQL query that causes duplicate rows.",
            "Here is a traceback (most recent call last): RuntimeError: CUDA out of memory. How can I resolve it?",
            "What is wrong with my code? def add(a, b): return a - b",
            "Find the syntax error in this function.",
            "Fix this function so it handles null values properly.",
        ]
        for p in prompts:
            with self.subTest(prompt=p):
                self.assertEqual(self.classifier.classify(p), "debugging")

    # 7. Data Analysis
    def test_task_data_analysis(self):
        prompts = [
            "Analyze this customer churn dataset and calculate the mean and standard deviation of monthly charges.",
            "Perform exploratory data analysis on the following sales figures CSV.",
            "What trends and patterns can you extract from this survey data table?",
            "Compute the median and 95th percentile latency from this metrics log.",
            "Analyze the data in the table below and find correlations between price and sales.",
        ]
        for p in prompts:
            with self.subTest(prompt=p):
                self.assertEqual(self.classifier.classify(p), "data_analysis")

    # 8. Question Answering
    def test_task_question_answering(self):
        prompts = [
            "What is the capital of Australia?",
            "Who was the first person to walk on the Moon?",
            "When was Python first released?",
            "Where is Mount Kilimanjaro located?",
            "How many planets are in the Solar System?",
            "Based on the text above, who discovered penicillin?",
            "Can penguins fly?",
            "Which element has the chemical symbol Au?",
        ]
        for p in prompts:
            with self.subTest(prompt=p):
                self.assertEqual(self.classifier.classify(p), "question_answering")

    # 9. Creative Writing
    def test_task_creative_writing(self):
        prompts = [
            "Write a short sci-fi story about a sentient rover exploring Europa.",
            "Compose a haiku about cherry blossoms in spring.",
            "Write a dialogue between Sherlock Holmes and Albert Einstein.",
            "Tell me a bedtime story about a curious little dragon.",
            "Write a poem about the ocean and the lonely lighthouse.",
            "Draft a screenplay scene where two spies meet at a cafe in Vienna.",
            "Tell me a joke about computer programmers.",
        ]
        for p in prompts:
            with self.subTest(prompt=p):
                self.assertEqual(self.classifier.classify(p), "creative_writing")

    # 10. Other: Empty, Whitespace, Non-matching, Chit-Chat
    def test_task_other_empty_whitespace_and_non_matching(self):
        prompts = [
            "",
            "   \n\t  ",
            "Hello there, how are you today?",
            "OK, sounds good to me.",
            "12345 67890",
            "Proofread this email for typos and grammatical errors.",
            "Format this text in all capital letters.",
            "Please",
            "Yes",
            "No",
        ]
        for p in prompts:
            with self.subTest(prompt=p):
                self.assertEqual(self.classifier.classify(p), "other")

    # Disambiguation & False Positive Avoidance Tests
    def test_disambiguation_and_false_positive_avoidance(self):
        # 1. "Explain difference" -> comparison, not general explanation
        self.assertEqual(
            self.classifier.classify("Explain the difference between Docker and virtual machines."),
            "comparison",
        )
        # 2. "What is the difference" -> comparison, not question_answering
        self.assertEqual(
            self.classifier.classify("What is the difference between SQL and NoSQL databases?"),
            "comparison",
        )
        # 3. "Explain what a bug is" -> explanation, not debugging
        self.assertEqual(
            self.classifier.classify("Explain what a software bug is to a beginner."),
            "explanation",
        )
        # 4. "Write code without bugs" -> code_generation, not debugging
        self.assertEqual(
            self.classifier.classify("Write a Python script to sort names, make sure it has no bugs."),
            "code_generation",
        )
        # 5. "Translate code from Python to C++" -> code_generation, not natural language translation
        self.assertEqual(
            self.classifier.classify("Translate this Python function into C++."),
            "code_generation",
        )
        # 6. "Write a summary" -> summarization, not creative_writing
        self.assertEqual(
            self.classifier.classify("Write a summary of the quarterly financial results."),
            "summarization",
        )
        # 7. "Write a Python script" -> code_generation, not creative_writing
        self.assertEqual(
            self.classifier.classify("Write a Python script that calculates primes."),
            "code_generation",
        )
        # 8. "Analyze this poem" -> not data_analysis
        self.assertNotEqual(
            self.classifier.classify("Analyze the rhythm and metaphors in this poem."),
            "data_analysis",
        )

    # Ambiguous and Conflicting Prompts
    def test_ambiguous_and_conflicting_prompts(self):
        # Two conflicting category requests in one prompt -> other
        ambiguous_1 = "Summarize this article and also write a fantasy story about dragons."
        self.assertEqual(self.classifier.classify(ambiguous_1), "other")

        ambiguous_2 = "Explain quantum physics or write a poem about it, whichever you prefer."
        self.assertEqual(self.classifier.classify(ambiguous_2), "other")

    # Pluggability & Replacement
    def test_modular_task_classifier_replaceability(self):
        dummy_classifier = DummyTaskClassifier(fixed_task="custom_task")
        custom_analyzer = PromptAnalyzer(task_classifier=dummy_classifier)

        result = custom_analyzer.analyze("Explain machine learning.")
        self.assertEqual(result["task_type"], "custom_task")

    # Integration through PromptAnalyzer and analyze_prompt
    def test_prompt_analyzer_task_type_integration(self):
        prompt = "Write a Python script to sort a list of numbers."
        result = self.analyzer.analyze(prompt)
        self.assertIn("task_type", result)
        self.assertEqual(result["task_type"], "code_generation")

        # analyze_prompt convenience function
        conv_result = analyze_prompt("Summarize the meeting notes.")
        self.assertEqual(conv_result["task_type"], "summarization")

        # Supported types set check
        self.assertIn("task_type", result)
        self.assertIn(result["task_type"], SUPPORTED_TASK_TYPES)


if __name__ == "__main__":
    unittest.main()

