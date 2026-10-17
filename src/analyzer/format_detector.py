"""Output format detection module for prompt analysis.

Identifies whether a prompt explicitly requests a specific output format or
document structure (lists, tables, JSON/XML, Markdown, headings/sections, reports/essays),
while distinguishing format requests from merely mentioning a format or topic.
"""

import re
from abc import ABC, abstractmethod
from typing import Optional


class BaseOutputFormatDetector(ABC):
    """Abstract base class for output format detection."""

    @abstractmethod
    def has_output_format(self, prompt: str) -> bool:
        """Return True if the prompt explicitly requests a specific output format."""
        pass


class RuleBasedOutputFormatDetector(BaseOutputFormatDetector):
    """Deterministic rule-based detector for requested output formats.

    Detects user instructions asking the model to structure the output as:
    - Bullet / numbered lists
    - Tables / tabular layouts
    - JSON / XML schemas
    - Markdown formatting
    - Headings / sections
    - Report / essay structures

    Distinguishes requested output formats from:
    - General knowledge inquiries about a format ("What is JSON?", "How does XML work?")
    - Educational how-to inquiries ("How do I create a table in HTML?", "How to write an essay?")
    - Existing input documents being referenced ("Summarize this report", "Proofread my essay")
    - Prohibitions or exclusions ("Do not use tables", "Without Markdown")
    """

    # Patterns indicating explicit directives for desired output formats
    _FORMAT_DIRECTIVE_PATTERNS = [
        # 1. Bullet / Numbered Lists
        re.compile(
            r"\b(?:in|using|with|into)\s+(?:\d+\s+)?(?:bullet\s+points?|bulleted\s+points?|bullets?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:as|in)\s+(?:a\s+)?(?:bulleted\s+list|bullet\s+list)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\buse\s+(?:bullet\s+points?|bullets?|a\s+bulleted\s+list|a\s+bullet\s+list)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:in|using|with|into)\s+(?:\d+\s+)?(?:numbered\s+list|numbered\s+points?|numbered\s+items?|numbered\s+steps?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:as|in)\s+(?:a\s+)?numbered\s+list\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\buse\s+(?:a\s+)?numbered\s+(?:list|points?|items?|steps?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:as|in)\s+a\s+list\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bformat\s+(?:as|in)\s+(?:a\s+)?(?:list|bullet\s+points?|numbered\s+list)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:bullet\s+points?|bulleted\s+list|numbered\s+list|list)\s+format\b",
            re.IGNORECASE,
        ),
        # 2. Tables
        re.compile(
            r"\b(?:in|as)\s+(?:a\s+)?table\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:in|as)\s+tabular\s+(?:format|form)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\btable\s+format\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\buse\s+(?:a\s+)?(?:table|tabular\s+format)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bformat\s+(?:as|in)\s+(?:a\s+)?table\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:present|display|show|summarize|compare|put)\s+(?:the\s+)?(?:data|results?|findings?|information|comparison|this)?\s*(?:in|as)\s+(?:a\s+)?table\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:create|make|generate|provide)\s+a\s+table\s+(?:comparing|summarizing|showing|listing|of)\b",
            re.IGNORECASE,
        ),
        # 3. JSON / XML
        re.compile(
            r"\b(?:in|as|into)\s+(?:valid\s+)?(?:json|xml)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:in|as)\s+(?:a\s+)?(?:valid\s+)?(?:json\s+(?:object|format|array)|xml\s+(?:format|document))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:json|xml)\s+format\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bformat\s+(?:the\s+(?:response|output)\s+)?(?:as|in)\s+(?:valid\s+)?(?:json|xml)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\boutput\s+(?:the\s+(?:response|output)\s+)?(?:as|in)\s+(?:valid\s+)?(?:json|xml)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\breturn\s+(?:the\s+(?:response|output)\s+)?(?:as|in)\s+(?:valid\s+)?(?:json|xml)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\brespond\s+(?:in|as|with)\s+(?:valid\s+)?(?:json|xml)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bconvert\s+(?:.*?)\s+to\s+(?:valid\s+)?(?:json|xml)\b",
            re.IGNORECASE,
        ),
        # 4. Markdown
        re.compile(
            r"\b(?:in|as)\s+markdown(?:\s+format)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bmarkdown\s+format\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\buse\s+markdown(?:\s+formatting)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bformat\s+(?:the\s+(?:response|output)\s+)?(?:as|in)\s+markdown\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\boutput\s+(?:the\s+(?:response|output)\s+)?(?:as|in)\s+markdown\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\brender\s+(?:as|in)\s+markdown\b",
            re.IGNORECASE,
        ),
        # 5. Headings / Sections
        re.compile(
            r"\b(?:use|using|with|include)\s+(?:headings?|subheadings?|sections?|section\s+headers?)(?:\s+(?:and|&)\s+(?:headings?|subheadings?|sections?))?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:organize|structure|divide)\s+(?:the\s+|your\s+)?(?:response|output|essay|report|answer|it)?\s*(?:into|with|by)\s*(?:[a-zA-Z-]+\s+)?(?:headings?|sections?|subheadings?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:organized?|structured?|divided?)\s+(?:into|with|by)\s*(?:[a-zA-Z-]+\s+)?(?:headings?|sections?|subheadings?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bbreak\s+(?:it\s+)?down\s+into\s+(?:clear\s+)?sections?\b",
            re.IGNORECASE,
        ),
        # 6. Report / Essay Structures
        re.compile(
            r"\b(?:write|compose|draft|prepare|structure)\s+(?:an?\s+)?(?:report|essay)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:in|as)\s+(?:an?\s+)?(?:formal\s+)?(?:report|essay)(?:\s+format|\s+structure)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:report|essay)\s+(?:format|structure)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bformat\s+(?:(?:the|your)\s+(?:response|output|analysis|findings?|answer|results?)\s+)?(?:as|in)\s+(?:an?\s+)?(?:formal\s+)?(?:report|essay)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bpresent\s+(?:as|in)\s+(?:an?\s+)?(?:formal\s+)?report\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bstructure\s+(?:as|in)\s+(?:an?\s+)?(?:report|essay)\b",
            re.IGNORECASE,
        ),
    ]

    # Lookback patterns indicating negation or prohibition
    _NEGATION_LOOKBACK = re.compile(
        r"\b(?:do\s+not|don't|never|avoid|without|must\s+not|should\s+not|cannot|can't|no)\b",
        re.IGNORECASE,
    )

    # Lookback patterns indicating educational how-to queries rather than formatting directives
    _HOW_TO_LOOKBACK = re.compile(
        r"\bhow\s+(?:to|one\s+can|you\s+can|do\s+i|can\s+i|should\s+i|would\s+one)\b",
        re.IGNORECASE,
    )

    # Meta inquiries asking about general concepts / third-party usage
    _META_INQUIRY_PATTERN = re.compile(
        r"^(?:why\s+(?:do|did|does|would|should|are|is)\s+(?:people|users|we|developers|authors|someone|they)\b|"
        r"how\s+(?:do\s+i|can\s+i|to|should\s+i|would\s+one|does|do)\b|"
        r"what\s+(?:is|are|was|were)\s+(?:the\s+)?(?:purpose|benefit|reason|advantage|difference)\b|"
        r"can\s+(?:markdown|json|xml|html)\s+(?:support|handle|do)\b)",
        re.IGNORECASE,
    )

    # Input reference patterns where formats/documents are objects of analysis rather than outputs
    _INPUT_DOCUMENT_PATTERN = re.compile(
        r"^(?:(?:analyze|analyse|summarize|summarise|proofread|review|read|parse|critique|check|edit|grade|evaluate|extract\s+from)\s+(?:this|the|my|our|following|above|below|attached)\s+(?:report|essay|table|json|xml|markdown)\b|"
        r"(?:given|here\s+is|based\s+on)\s+(?:this|the|my|our|following|above|below|attached)?\s*(?:report|essay|table|json|xml|markdown|data)\b)",
        re.IGNORECASE,
    )

    def has_output_format(self, prompt: str) -> bool:
        """Evaluate if the prompt explicitly requests a specific output format.

        Args:
            prompt: Raw user input prompt string.

        Returns:
            Boolean indicating presence of an explicit output format requirement.
        """
        if not prompt or not prompt.strip():
            return False

        cleaned = prompt.strip()
        if self._META_INQUIRY_PATTERN.match(cleaned):
            return False

        for pattern in self._FORMAT_DIRECTIVE_PATTERNS:
            for match in pattern.finditer(cleaned):
                start = match.start()
                clause_start = max(
                    cleaned.rfind(".", 0, start),
                    cleaned.rfind("!", 0, start),
                    cleaned.rfind("?", 0, start),
                    cleaned.rfind(";", 0, start),
                    cleaned.rfind("\n", 0, start),
                )
                preceding_clause_text = cleaned[clause_start + 1 : start]

                # Check if preceding clause indicates input document analysis
                if self._INPUT_DOCUMENT_PATTERN.search(cleaned[clause_start + 1 : match.end()]):
                    continue

                # Check if negated/prohibited in current clause
                if self._NEGATION_LOOKBACK.search(preceding_clause_text):
                    continue

                # Check if educational how-to query
                if self._HOW_TO_LOOKBACK.search(preceding_clause_text):
                    continue

                return True

        return False


def get_output_format_detector() -> BaseOutputFormatDetector:
    """Factory function returning the default output format detector instance."""
    return RuleBasedOutputFormatDetector()
