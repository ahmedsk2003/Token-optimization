"""Example requirement detection module for prompt analysis.

Identifies whether a prompt explicitly requests examples (e.g., code examples,
practical scenarios, counterexamples, concrete cases) using deterministic,
rule-based heuristics.
"""

import re
from abc import ABC, abstractmethod
from typing import Optional


class BaseExampleDetector(ABC):
    """Abstract base class for example requirement detection."""

    @abstractmethod
    def has_examples_requirement(self, prompt: str) -> bool:
        """Return True if the prompt contains an explicit request for examples."""
        pass


class RuleBasedExampleDetector(BaseExampleDetector):
    """Deterministic rule-based detector for example requirements.

    Detects user instructions asking the model to include, provide, illustrate with,
    or generate examples, while filtering out:
    - Conversational discourse markers ("for example, ...", "as an example, ...")
    - Mentions of existing input examples ("in this example:", "here is an example")
    - Negative constraints / prohibitions ("do not include examples", "without examples")
    """

    # Directives requesting examples
    _DIRECTIVE_PATTERNS = [
        re.compile(
            r"\b(?:(?:please\s+|kindly\s+|can\s+you\s+|could\s+you\s+|make\s+sure\s+to\s+|ensure\s+you\s+|i\s+(?:want|need|would\s+like)\s+(?:you\s+)?to\s+)?"
            r"(?:include|provide|give|show|add|supply|share|list|contain)\s+(?:me\s+)?"
            r"(?:(?:at\s+least\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:[a-zA-Z-]+\s+){0,2}examples?|"
            r"(?:(?:an?|some|several|many|concrete|practical|real[- ]world|real[- ]life|code|detailed|relevant|[a-zA-Z-]+)\s+){0,3}examples?))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:illustrate|demonstrate|support|back\s+up|explain)\s+(?:this\s+|it\s+)?"
            r"(?:with|using|through)\s+(?:an?\s+|[a-zA-Z-]+\s+){0,3}examples?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:with|using|accompanied\s+by|supported\s+by)\s+(?:at\s+least\s+)?"
            r"(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:[a-zA-Z-]+\s+){0,2}examples?|"
            r"(?:(?:an?|some|several|many|concrete|practical|real[- ]world|real[- ]life|code|detailed|[a-zA-Z-]+)\s+){0,3}examples?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bexamples?\s+(?:are\s+)?(?:required|needed|appreciated|mandatory|welcome)\b",
            re.IGNORECASE,
        ),
    ]

    # Conversational or input references to examples that are not instructions to provide them
    _INPUT_OR_TOPIC_PATTERN = re.compile(
        r"\b(?:for\s+example|as\s+an\s+example|take\s+(?:for\s+example|the\s+example\s+of)|"
        r"(?:this|the|that|my|our|following|above|below)\s+(?:code\s+)?example|here\s+is\s+an?\s+example|"
        r"example\s+(?:code|input|data|output|scenario|case|below|above))\b",
        re.IGNORECASE,
    )

    # Local negation / prohibition pattern
    _NEGATION_LOOKBACK = re.compile(
        r"\b(?:do\s+not|don't|never|avoid|without|must\s+not|should\s+not|cannot|can't|no)\b",
        re.IGNORECASE,
    )

    def has_examples_requirement(self, prompt: str) -> bool:
        """Evaluate if the prompt requests examples.

        Args:
            prompt: Raw user input prompt string.

        Returns:
            Boolean indicating presence of an explicit example requirement.
        """
        if not prompt or not prompt.strip():
            return False

        cleaned = prompt.strip()

        for pattern in self._DIRECTIVE_PATTERNS:
            for match in pattern.finditer(cleaned):
                start = match.start()
                # Find start of current clause (after last terminal punctuation / newline)
                clause_start = max(
                    cleaned.rfind(".", 0, start),
                    cleaned.rfind("!", 0, start),
                    cleaned.rfind("?", 0, start),
                    cleaned.rfind(";", 0, start),
                    cleaned.rfind("\n", 0, start),
                )
                preceding_clause_text = cleaned[clause_start + 1 : start]

                # If preceded by a negation trigger within the clause, it is a prohibition
                if self._NEGATION_LOOKBACK.search(preceding_clause_text):
                    continue

                # If the matched phrase itself is merely a discourse marker or input reference
                matched_text = match.group(0)
                if self._INPUT_OR_TOPIC_PATTERN.search(matched_text):
                    continue

                return True

        return False


def get_example_detector() -> BaseExampleDetector:
    """Factory function returning the default example detector instance."""
    return RuleBasedExampleDetector()
