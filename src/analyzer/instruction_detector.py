"""Instruction detection module for prompt analysis.

Identifies and counts primary task instructions (the main action/task the user asks
the model to perform) using deterministic, rule-based heuristics.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional

from src.analyzer.sentence_segmenter import (
    BaseSentenceSegmenter,
    get_sentence_segmenter,
)


class BaseInstructionDetector(ABC):
    """Abstract base class for instruction detection."""

    @abstractmethod
    def detect_instructions(self, prompt: str) -> List[str]:
        """Detect and return identified instruction strings."""
        pass

    def count_instructions(self, prompt: str) -> int:
        """Count the number of primary instructions in the prompt."""
        return len(self.detect_instructions(prompt))


class RuleBasedInstructionDetector(BaseInstructionDetector):
    """Deterministic rule-based instruction detector.

    Identifies primary task directives based on action verbs, interrogative phrasing,
    and task coordination while filtering out constraint-only clauses.
    """

    # Primary action verbs indicating a direct user task
    ACTION_VERBS = {
        "explain", "summarize", "write", "compare", "analyze", "generate",
        "translate", "debug", "create", "describe", "list", "classify",
        "extract", "convert", "implement", "rewrite", "draft", "solve",
        "evaluate", "optimize", "discuss", "identify", "calculate", "build",
        "code", "clarify", "define", "find", "compose", "suggest",
        "recommend", "paraphrase", "outline", "illustrate", "develop", "construct",
        "review", "examine", "assess", "refactor", "critique", "demonstrate", "parse"
    }

    # Polite preambles to strip before detecting primary verbs
    _PREAMBLE_PATTERN = re.compile(
        r"^(?:please\s+|kindly\s+|can\s+you\s+|could\s+you\s+|i\s+want\s+(?:you\s+)?to\s+|"
        r"help\s+me\s+|i\s+need\s+(?:you\s+)?to\s+|i\s+would\s+like\s+(?:you\s+)?to\s+|"
        r"your\s+task\s+is\s+to\s+|make\s+sure\s+to\s+)+",
        re.IGNORECASE,
    )

    # Patterns indicating pure constraint clauses that should not count as primary tasks
    _CONSTRAINT_CLAUSE_PATTERN = re.compile(
        r"^(?:do\s+not|don't|never|avoid|without|must\s+not|should\s+not|cannot|can't|no\s+|"
        r"use\s+bullet|in\s+markdown|as\s+a\s+table|in\s+a\s+table|in\s+json|as\s+(?:a\s+)?json|"
        r"(?:include|provide|give|with)\s+(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+examples?|an?\s+example|examples?\b)|"
        r"(?:include|provide|give)\s+(?:error\s+handling|code|comments|test\s+cases|details)|"
        r"limit\s+(?:it\s+)?to|maximum|max|keep\s+it\s+under|be\s+concise|for\s+a\s+beginner|suitable\s+for)",
        re.IGNORECASE,
    )

    # Interrogative starts
    _QUESTION_PATTERN = re.compile(
        r"^(?:what|how|why|who|where|when|which|is|are|can|could|will|would)\b",
        re.IGNORECASE,
    )

    def __init__(self, segmenter: Optional[BaseSentenceSegmenter] = None):
        self.segmenter = segmenter if segmenter is not None else get_sentence_segmenter()

    def _strip_preamble(self, text: str) -> str:
        """Remove polite openers to isolate core action."""
        stripped = self._PREAMBLE_PATTERN.sub("", text.strip())
        return stripped.strip()

    def _is_action_clause(self, clause: str) -> bool:
        """Check if a clause represents a primary task instruction."""
        cleaned = self._strip_preamble(clause)
        if not cleaned:
            return False

        # If it's a pure constraint clause, it's not a primary instruction
        if self._CONSTRAINT_CLAUSE_PATTERN.search(cleaned):
            return False

        # Check for first word as action verb
        tokens = re.findall(r"\b[a-zA-Z]+\b", cleaned)
        if tokens:
            first_word = tokens[0].lower()
            if first_word in self.ACTION_VERBS:
                return True

        # Check for interrogative question
        if self._QUESTION_PATTERN.match(cleaned) or cleaned.endswith("?"):
            return True

        return False

    def detect_instructions(self, prompt: str) -> List[str]:
        """Detect primary instructions in a user prompt.

        Args:
            prompt: Raw user input prompt string.

        Returns:
            List of detected instruction clauses.
        """
        if not prompt or not prompt.strip():
            return []

        sentences = self.segmenter.split_sentences(prompt)
        instructions: List[str] = []

        for sentence in sentences:
            cleaned_sentence = self._strip_preamble(sentence)
            if not cleaned_sentence:
                continue

            # Skip pure constraint sentences
            if self._CONSTRAINT_CLAUSE_PATTERN.search(cleaned_sentence):
                continue

            # Check for coordinated actions within the sentence (e.g. "Write X and analyze Y")
            sub_clauses = re.split(
                r"\b(?:and\s+also|and\s+then|\band\b|,\s*then\b)\s+",
                cleaned_sentence,
                flags=re.IGNORECASE,
            )
            found_sub = False
            for clause in sub_clauses:
                clause_stripped = self._strip_preamble(clause)
                tokens = re.findall(r"\b[a-zA-Z]+\b", clause_stripped)
                if (
                    tokens
                    and tokens[0].lower() in self.ACTION_VERBS
                    and not self._CONSTRAINT_CLAUSE_PATTERN.search(clause_stripped)
                ):
                    instructions.append(clause.strip())
                    found_sub = True

            if not found_sub:
                if self._is_action_clause(sentence):
                    instructions.append(sentence.strip())

        # If prompt is non-empty and no explicit action verbs were detected,
        # treat the overall prompt as 1 implicit task
        if not instructions and prompt.strip():
            # Check if it was entirely constraints
            if not all(
                self._CONSTRAINT_CLAUSE_PATTERN.search(self._strip_preamble(s))
                for s in sentences
            ):
                instructions.append(prompt.strip())

        return instructions


def get_instruction_detector() -> BaseInstructionDetector:
    """Factory function returning the default instruction detector instance."""
    return RuleBasedInstructionDetector()
