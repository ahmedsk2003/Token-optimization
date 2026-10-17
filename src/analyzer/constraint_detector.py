"""Constraint detection module for prompt analysis.

Identifies and counts explicit user constraints that restrict how a task should
be performed or what the output should contain, avoid, or format.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class DetectedConstraint:
    """Represents an individual detected constraint."""

    category: str
    matched_text: str
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "matched_text": self.matched_text,
            "start": self.start,
            "end": self.end,
        }


class BaseConstraintDetector(ABC):
    """Abstract base class for constraint detection."""

    @abstractmethod
    def detect_constraints(self, prompt: str) -> List[DetectedConstraint]:
        """Detect and return a list of identified constraint details."""
        pass

    def count_constraints(self, prompt: str) -> int:
        """Count the number of detected constraints in the prompt."""
        return len(self.detect_constraints(prompt))


class RuleBasedConstraintDetector(BaseConstraintDetector):
    """Deterministic rule-based constraint detector.

    Detects explicit restrictions across multiple categories:
    - Exclusion / Prohibition (negative constraints)
    - Required format / Structure
    - Length / Volume limits (words, sentences, tokens)
    - Target audience / Tone / Level
    - Example count requirements
    - Content / Topic inclusion requirements
    - Language / Programming language requirements
    """

    # Categorized regex patterns for explicit constraint detection
    _PATTERNS: List[Tuple[str, re.Pattern]] = [
        # 1. Exclusion / Negative constraints
        (
            "exclusion",
            re.compile(
                r"\b(?:do\s+not|don't|never|avoid|without|must\s+not|should\s+not|cannot|can't|omit|exclude)\b[^.\n!?]*|"
                r"\bno\s+(?:mathematical\s+notation|math|jargon|external\s+libraries?|code|fluff|intro(?:duction)?|conclusion|preamble)\b",
                re.IGNORECASE,
            ),
        ),
        # 2. Number of Examples requirement
        (
            "example_count",
            re.compile(
                r"\b(?:(?:please\s+)?(?:include|provide|give|with)\s+(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+examples?|an?\s+example|at\s+least\s+\d+\s+examples?|examples?\b))",
                re.IGNORECASE,
            ),
        ),
        # 3. Format / Output Structure constraints
        (
            "format",
            re.compile(
                r"\b(?:use\s+bullet\s+points?|in\s+bullet\s+points?|bullet\s+points?|in\s+markdown(?:\s+format)?|"
                r"in\s+json(?:\s+format)?|as\s+(?:a\s+)?(?:valid\s+)?json(?:\s+object|\s+format)?|json\s+format|as\s+a\s+table|in\s+a\s+table|table\s+format|"
                r"as\s+a\s+list|numbered\s+list|csv\s+format|in\s+csv|in\s+a\s+single\s+line|single\s+line|"
                r"step\s*by\s*step|stepwise|yaml\s+format|in\s+yaml|using\s+headings?)\b",
                re.IGNORECASE,
            ),
        ),
        # 4. Length / Word / Token constraints
        (
            "length",
            re.compile(
                r"\bbetween\s+(?:\d+|(?:one|two|three|four|five|six|seven|eight|nine|ten))\s+and\s+(?:\d+|(?:one|two|three|four|five|six|seven|eight|nine|ten))\s+(?:words?|tokens?|sentences?|paragraphs?|lines?|characters?|chars?|pages?)\b|"
                r"\b(?:\d+|(?:one|two|three|four|five|six|seven|eight|nine|ten))\s*(?:to|-)\s*(?:\d+|(?:one|two|three|four|five|six|seven|eight|nine|ten))\s+(?:words?|tokens?|sentences?|paragraphs?|lines?|characters?|chars?|pages?)\b|"
                r"\b(?:(?:under|maximum|max|at\s+most|at\s+least|min|minimum|within|no\s+more\s+than|less\s+than|fewer\s+than|limit\s+(?:it\s+)?to|up\s+to|around|approximately|approx\.?|about|exactly|in)\s+)?(?:\d+|(?:one|two|three|four|five|six|seven|eight|nine|ten))\s*-\s*(?:words?|tokens?|sentences?|paragraphs?|lines?|characters?|chars?|pages?)\b|"
                r"\b(?:under|maximum|max|at\s+most|at\s+least|min|minimum|within|no\s+more\s+than|less\s+than|fewer\s+than|limit\s+(?:it\s+)?to|up\s+to|around|approximately|approx\.?|about|exactly|in)\s+(?:\d+|(?:one|two|three|four|five|six|seven|eight|nine|ten))\s+(?:words?|tokens?|sentences?|paragraphs?|lines?|characters?|chars?|pages?)(?:\s+or\s+(?:less|fewer|more))?\b|"
                r"\b(?:\d+|(?:one|two|three|four|five|six|seven|eight|nine|ten))\s+(?:words?|tokens?|sentences?|paragraphs?|lines?|characters?|chars?|pages?)(?:\s+or\s+(?:less|fewer|more))?\b|"
                r"\b(?:briefly|in\s+short|concise|concisely|in\s+brief|short\s+summary)\b",
                re.IGNORECASE,
            ),
        ),
        # 5. Target Audience / Tone / Level constraints
        (
            "audience",
            re.compile(
                r"\b(?:for\s+(?:a\s+|an\s+)?(?:beginner|beginners|child|children|kids?|expert|experts|high\s+school\s+students?|undergraduate|layman|non-technical\s+(?:person|user|audience)|5\s+year\s+old|novice))\b|"
                r"\b(?:suitable\s+for\s+(?:a\s+|an\s+)?(?:beginner|beginners|child|children|kids?|expert|experts|high\s+school\s+students?|undergraduate|layman|novice))\b|"
                r"\b(?:in\s+(?:simple|plain|layman's|easy)\s+terms|explain\s+like\s+i'm\s+5|eli5)\b|"
                r"\b(?:formal|informal|professional|casual|academic)\s+tone\b",
                re.IGNORECASE,
            ),
        ),
        # 6. Programming / Target Language constraints
        (
            "language",
            re.compile(
                r"\b(?:(?:in|using|with)\s+(?:python|javascript|typescript|c\+\+|java|rust|go|c#|sql|bash|html|css)|"
                r"(?:python|javascript|typescript|c\+\+|java|rust|go|c#|sql|bash|html|css)\s+(?:script|program|code|function|app))\b|"
                r"\b(?:in|translate\s+to)\s+(?:french|spanish|german|chinese|japanese|russian|italian|portuguese|arabic|hindi)\b",
                re.IGNORECASE,
            ),
        ),
        # 7. Explicit Inclusion / Content requirement
        (
            "inclusion",
            re.compile(
                r"\b(?:(?:please\s+)?include|must\s+include|must\s+contain|make\s+sure\s+to\s+(?:include|cover|mention)|ensure\s+you\s+(?:cover|include|mention))\b[^.\n!?]*",
                re.IGNORECASE,
            ),
        ),
    ]

    def detect_constraints(self, prompt: str) -> List[DetectedConstraint]:
        """Detect all non-overlapping explicit constraints in the prompt.

        Args:
            prompt: Raw user input prompt string.

        Returns:
            List of DetectedConstraint objects.
        """
        if not prompt or not prompt.strip():
            return []

        detected: List[DetectedConstraint] = []
        occupied_spans: List[Tuple[int, int]] = []

        def spans_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
            return max(s1, s2) < min(e1, e2)

        for category, pattern in self._PATTERNS:
            for match in pattern.finditer(prompt):
                start, end = match.start(), match.end()
                matched_str = match.group(0).strip()
                if not matched_str:
                    continue

                # Verify this match does not substantially overlap with an already detected constraint
                if any(spans_overlap(start, end, occ_s, occ_e) for occ_s, occ_e in occupied_spans):
                    continue

                detected.append(
                    DetectedConstraint(
                        category=category,
                        matched_text=matched_str,
                        start=start,
                        end=end,
                    )
                )
                occupied_spans.append((start, end))

        # Sort detected constraints by their position in the prompt text
        detected.sort(key=lambda c: c.start)
        return detected


def get_constraint_detector() -> BaseConstraintDetector:
    """Factory function returning the default constraint detector instance."""
    return RuleBasedConstraintDetector()
