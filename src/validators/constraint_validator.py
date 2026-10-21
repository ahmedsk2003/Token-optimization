"""Constraint preservation validator for prompt token compression.

Phase 4.2B: Constraint Preservation and Validation
- Modular, deterministic, local validator to check whether prompt compression
  preserves user constraints, directives, formats, and meaning-critical content.
- Checks:
  1. Negation (prohibitions, exclusions, polarity shifts)
  2. Numerical constraints (quantities, counts, bounds, word limits)
  3. Conditional clauses (hypotheticals, guard conditions, subordinate clauses)
  4. Output formats (JSON, Markdown, tables, lists, schemas)
  5. Audience/tone modifiers (beginners, experts, tone directives, ELI5)
  6. Required examples (example counts, illustrative requests)
  7. Entities (acronyms, proper nouns, quoted terms)
  8. Important technical terms (code identifiers, domain terminology, specialized vocabulary)
- Reuses existing analyzer detectors (RuleBasedConstraintDetector, RuleBasedOutputFormatDetector,
  RuleBasedExampleDetector) without introducing new dependencies, LLM calls, or new models.
- Uses conservative local rules and flags uncertainty when preservation cannot be confidently verified.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple

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


@dataclass(frozen=True)
class ValidationIssue:
    """Represents a specific preservation issue detected during validation.

    Attributes:
        category: Check category ('negation', 'numerical', 'conditional', 'format',
                  'audience_tone', 'examples', 'entity', 'technical_term', 'uncertainty').
        description: Explanatory text describing the detected discrepancy.
        severity: Issue severity ('error' for constraint violation, 'warning' for uncertainty).
        original_evidence: String snippet or term from original prompt.
        compressed_evidence: String snippet or term from compressed prompt (if any).
    """

    category: str
    description: str
    severity: str = "error"
    original_evidence: Optional[str] = None
    compressed_evidence: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to standard dictionary format."""
        return {
            "category": self.category,
            "description": self.description,
            "severity": self.severity,
            "original_evidence": self.original_evidence,
            "compressed_evidence": self.compressed_evidence,
        }


@dataclass(frozen=True)
class ValidationResult:
    """Structured result of prompt constraint preservation validation.

    Attributes:
        is_valid: True if all critical constraint checks passed (no error-severity issues).
        issues: List of all detected ValidationIssue objects (errors and warnings).
        confidence: Assessment confidence ('high', 'medium', 'low', 'uncertain').
        preserved_categories: List of checked categories confirmed preserved.
        failed_categories: List of categories with error-severity violations.
        uncertain_categories: List of categories with warning-severity issues.
        details: Category-by-category details and extracted items.
    """

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    confidence: str = "high"
    preserved_categories: List[str] = field(default_factory=list)
    failed_categories: List[str] = field(default_factory=list)
    uncertain_categories: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to standard dictionary format."""
        return {
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "confidence": self.confidence,
            "preserved_categories": list(self.preserved_categories),
            "failed_categories": list(self.failed_categories),
            "uncertain_categories": list(self.uncertain_categories),
            "details": self.details,
        }


class BasePromptValidator(ABC):
    """Abstract base class for prompt preservation validators."""

    @abstractmethod
    def validate(self, original_prompt: str, compressed_prompt: str) -> ValidationResult:
        """Validate whether compressed prompt preserves original prompt requirements.

        Args:
            original_prompt: Original user input prompt.
            compressed_prompt: Compressed prompt candidate.

        Returns:
            Structured ValidationResult indicating pass/fail, issues, and confidence.
        """
        pass


# ---------------------------------------------------------------------------
# Constants and Regex Patterns for Conservative Local Validation
# ---------------------------------------------------------------------------

_DISCOURSE_AND_BOILERPLATE = frozenset({
    "a", "an", "the", "this", "that", "these", "those",
    "please", "kindly", "could", "can", "would", "should", "will", "shall",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our", "its",
    "provide", "provides", "providing", "give", "gives", "giving",
    "explanation", "explanations", "explain", "explains", "explained",
    "summary", "summaries", "summarize", "summarizes", "summarized",
    "description", "descriptions", "describe", "describes", "described",
    "analysis", "analyses", "analyze", "analyzes", "analyzed",
    "comparison", "comparisons", "compare", "compares", "compared",
    "definition", "definitions", "define", "defines", "defined",
    "clarification", "clarifications", "clarify", "clarifies", "clarified",
    "list", "lists", "listing", "listed",
    "write", "writes", "writing", "written",
    "what", "how", "why", "when", "where", "which", "who", "whom", "whose",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "done",
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "and", "but", "or", "so", "as", "than", "also", "here", "there",
    "just", "well", "like", "want", "need", "make", "perform", "conduct",
})

_WORD_NUMBERS: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "million": "1000000", "billion": "1000000000",
}

_ORDINAL_MAP: Dict[str, str] = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "1st": "1", "2nd": "2", "3rd": "3", "4th": "4", "5th": "5",
}

_DIGIT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_WORD_NUMBER_PATTERN = re.compile(
    r"\b(?:" + "|".join(_WORD_NUMBERS.keys()) + r")\b",
    re.IGNORECASE,
)
_ORDINAL_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in _ORDINAL_MAP.keys()) + r"|\d+th)\b",
    re.IGNORECASE,
)

_NEGATION_WORDS = frozenset({
    "not", "no", "never", "none", "neither", "nor", "without",
    "cannot", "can't", "cant", "don't", "dont", "avoid", "avoids",
    "avoiding", "omit", "omits", "omitting", "exclude", "excludes",
    "excluding",
})

_NEGATION_PHRASES = [
    re.compile(r"\bdo\s+not\b", re.IGNORECASE),
    re.compile(r"\bmust\s+not\b", re.IGNORECASE),
    re.compile(r"\bshould\s+not\b", re.IGNORECASE),
    re.compile(r"\bwill\s+not\b", re.IGNORECASE),
    re.compile(r"\bwon't\b", re.IGNORECASE),
    re.compile(r"\bwould\s+not\b", re.IGNORECASE),
]

_CONDITIONAL_PATTERNS = [
    re.compile(r"\bif\b", re.IGNORECASE),
    re.compile(r"\bunless\b", re.IGNORECASE),
    re.compile(r"\bwhether\b", re.IGNORECASE),
    re.compile(r"\bin\s+case\b", re.IGNORECASE),
    re.compile(r"\bprovided\s+(?:that\s+)?\b", re.IGNORECASE),
    re.compile(r"\bassuming\s+(?:that\s+)?\b", re.IGNORECASE),
    re.compile(r"\bonly\s+if\b", re.IGNORECASE),
    re.compile(r"\beven\s+if\b", re.IGNORECASE),
    re.compile(r"\bas\s+long\s+as\b", re.IGNORECASE),
    re.compile(r"\bso\s+long\s+as\b", re.IGNORECASE),
    re.compile(r"\bsupposing\b", re.IGNORECASE),
    re.compile(r"\bwhereas\b", re.IGNORECASE),
    re.compile(r"\balthough\b", re.IGNORECASE),
    re.compile(r"\bthough\b", re.IGNORECASE),
]

_FORMAT_KEYWORDS = frozenset({
    "json", "yaml", "xml", "csv", "table", "tabular", "bullet",
    "bullets", "numbered", "markdown", "latex", "sql", "html",
    "single line", "headings",
})

_AUDIENCE_KEYWORDS = frozenset({
    "beginner", "beginners", "child", "children", "kids", "kid",
    "expert", "experts", "student", "students", "undergraduate",
    "layman", "laymans", "layman's", "novice", "5 year old",
    "five year old", "non-technical",
})

_TONE_KEYWORDS = frozenset({
    "formal", "informal", "casual", "academic", "professional",
    "conversational", "friendly", "humorous", "serious", "eli5",
    "simple terms", "plain terms", "easy terms",
})

_QUOTED_OR_CODE_PATTERN = re.compile(r"```[\s\S]*?```|`([^`]+)`|\"([^\"]+)\"|'([^']+)'")
_ACRONYM_PATTERN = re.compile(r"\b[A-Z0-9]{2,}\b")
_CAPITALIZED_WORD_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:'[a-z]+)?\b")
_TECHNICAL_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:[a-zA-Z]+_[a-zA-Z0-9_]+|[a-z]+[A-Z][a-zA-Z0-9]*|[a-zA-Z_]\w*\.[a-zA-Z_]\w*|[a-z]+-[a-z]+)\b|"
    r"\b(?:c\+\+|c#|\.net)\b",
    re.IGNORECASE,
)

_COMMON_TECHNICAL_TERMS = frozenset({
    "attention mechanism", "attention mechanisms", "transformer", "transformers",
    "quicksort", "cryptography", "consensus protocol", "microservice", "microservices",
    "blockchain", "eigenvector", "eigenvectors", "eigenvalue", "eigenvalues",
    "backpropagation", "neural network", "neural networks", "binary search",
    "hash table", "time complexity", "space complexity", "worst-case",
    "recursion", "polymorphism", "concurrency", "deadlock", "mutex",
    "reinforcement learning", "supervised learning", "unsupervised learning",
    "quantum computing", "quantum entanglement", "quantum annealing",
    "public key", "private key", "asymmetric encryption", "fibonacci",
})


class InstructionConstraintValidator(BasePromptValidator):
    """Modular, deterministic local validator for constraint preservation in compressed prompts.

    Evaluates whether compressed candidate prompts preserve critical constraints from
    the original prompt across 8 core dimensions:
    1. Negation
    2. Numerical constraints
    3. Conditional clauses
    4. Output formats
    5. Audience and tone modifiers
    6. Required examples
    7. Named entities and acronyms
    8. Technical terms and code identifiers

    Reuses existing analyzer detectors without adding new dependencies or models.
    Flags uncertainty conservatively when semantic preservation cannot be verified locally.
    """

    def __init__(
        self,
        constraint_detector: Optional[BaseConstraintDetector] = None,
        format_detector: Optional[BaseOutputFormatDetector] = None,
        example_detector: Optional[BaseExampleDetector] = None,
    ):
        """Initialize validator with pluggable or default detectors.

        Args:
            constraint_detector: Detector for explicit constraints. Defaults to RuleBasedConstraintDetector.
            format_detector: Detector for output format requests. Defaults to RuleBasedOutputFormatDetector.
            example_detector: Detector for example requests. Defaults to RuleBasedExampleDetector.
        """
        self._constraint_detector = constraint_detector or RuleBasedConstraintDetector()
        self._format_detector = format_detector or RuleBasedOutputFormatDetector()
        self._example_detector = example_detector or RuleBasedExampleDetector()

    def validate(self, original_prompt: str, compressed_prompt: str) -> ValidationResult:
        """Validate whether compressed prompt preserves original prompt requirements.

        Args:
            original_prompt: Original user input prompt.
            compressed_prompt: Compressed prompt candidate.

        Returns:
            Structured ValidationResult indicating pass/fail, issues, and confidence.
        """
        orig_safe = original_prompt if original_prompt is not None else ""
        comp_safe = compressed_prompt if compressed_prompt is not None else ""

        orig_stripped = orig_safe.strip()
        comp_stripped = comp_safe.strip()

        # 1. Handle empty inputs
        if not orig_stripped:
            if not comp_stripped:
                return ValidationResult(
                    is_valid=True,
                    issues=[],
                    confidence="high",
                    preserved_categories=[
                        "negation", "numerical", "conditional", "format",
                        "audience_tone", "examples", "entity", "technical_term",
                    ],
                    failed_categories=[],
                    uncertain_categories=[],
                    details={"status": "both_empty"},
                )
            else:
                issue = ValidationIssue(
                    category="uncertainty",
                    description="Compressed text generated from an empty original prompt.",
                    severity="error",
                    compressed_evidence=comp_stripped,
                )
                return ValidationResult(
                    is_valid=False,
                    issues=[issue],
                    confidence="high",
                    preserved_categories=[],
                    failed_categories=["uncertainty"],
                    uncertain_categories=[],
                    details={"status": "empty_original_with_nonempty_compressed"},
                )

        # 2. Handle unchanged prompts
        if orig_stripped == comp_stripped:
            return ValidationResult(
                is_valid=True,
                issues=[],
                confidence="high",
                preserved_categories=[
                    "negation", "numerical", "conditional", "format",
                    "audience_tone", "examples", "entity", "technical_term",
                ],
                failed_categories=[],
                uncertain_categories=[],
                details={"status": "identical_prompts"},
            )

        # 3. Perform detailed category checks
        issues: List[ValidationIssue] = []
        preserved_categories: List[str] = []
        failed_categories: List[str] = []
        uncertain_categories: List[str] = []
        details: Dict[str, Any] = {}

        # 3.1 Negation check
        neg_issues, neg_info = self._check_negation(orig_safe, comp_safe)
        details["negation"] = neg_info
        if neg_issues:
            issues.extend(neg_issues)
            failed_categories.append("negation")
        else:
            preserved_categories.append("negation")

        # 3.2 Numerical constraints check
        num_issues, num_info = self._check_numerical(orig_safe, comp_safe)
        details["numerical"] = num_info
        if num_issues:
            issues.extend(num_issues)
            failed_categories.append("numerical")
        else:
            preserved_categories.append("numerical")

        # 3.3 Conditional clauses check
        cond_issues, cond_info = self._check_conditionals(orig_safe, comp_safe)
        details["conditional"] = cond_info
        if cond_issues:
            issues.extend(cond_issues)
            failed_categories.append("conditional")
        else:
            preserved_categories.append("conditional")

        # 3.4 Output formats check
        fmt_issues, fmt_info = self._check_format(orig_safe, comp_safe)
        details["format"] = fmt_info
        if fmt_issues:
            issues.extend(fmt_issues)
            failed_categories.append("format")
        else:
            preserved_categories.append("format")

        # 3.5 Audience and tone modifiers check
        aud_issues, aud_info = self._check_audience_tone(orig_safe, comp_safe)
        details["audience_tone"] = aud_info
        if aud_issues:
            issues.extend(aud_issues)
            failed_categories.append("audience_tone")
        else:
            preserved_categories.append("audience_tone")

        # 3.6 Required examples check
        ex_issues, ex_info = self._check_examples(orig_safe, comp_safe)
        details["examples"] = ex_info
        if ex_issues:
            issues.extend(ex_issues)
            failed_categories.append("examples")
        else:
            preserved_categories.append("examples")

        # 3.7 Entities check
        ent_issues, ent_info = self._check_entities(orig_safe, comp_safe)
        details["entity"] = ent_info
        if ent_issues:
            issues.extend(ent_issues)
            failed_categories.append("entity")
        else:
            preserved_categories.append("entity")

        # 3.8 Important technical terms check
        tech_issues, tech_info = self._check_technical_terms(orig_safe, comp_safe)
        details["technical_term"] = tech_info
        if tech_issues:
            issues.extend(tech_issues)
            failed_categories.append("technical_term")
        else:
            preserved_categories.append("technical_term")

        # 3.9 Conservative uncertainty check
        uncert_issues, uncert_info = self._check_uncertainty(orig_safe, comp_safe)
        details["uncertainty"] = uncert_info
        if uncert_issues:
            issues.extend(uncert_issues)
            uncertain_categories.append("uncertainty")

        # 4. Synthesize final validation result
        has_errors = any(issue.severity == "error" for issue in issues)
        has_warnings = any(issue.severity == "warning" for issue in issues)

        is_valid = not has_errors
        if has_errors:
            confidence = "low"
        elif has_warnings:
            confidence = "uncertain"
        else:
            confidence = "high"

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            confidence=confidence,
            preserved_categories=preserved_categories,
            failed_categories=failed_categories,
            uncertain_categories=uncertain_categories,
            details=details,
        )

    # -----------------------------------------------------------------------
    # Individual Category Check Implementations
    # -----------------------------------------------------------------------

    def _extract_negation_tokens(self, text: str) -> Set[str]:
        """Extract individual negation words and normalized phrases from text."""
        tokens = set()
        words = set(re.findall(r"\b[a-zA-Z']+\b", text.lower()))
        for w in words:
            if w in _NEGATION_WORDS:
                tokens.add(w)
        for pattern in _NEGATION_PHRASES:
            for match in pattern.finditer(text):
                tokens.add(match.group(0).lower())
        return tokens

    def _check_negation(
        self, original: str, compressed: str
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Validate preservation of negation constraints."""
        issues: List[ValidationIssue] = []
        orig_negs = self._extract_negation_tokens(original)
        comp_negs = self._extract_negation_tokens(compressed)

        # Check exclusion constraints from analyzer
        orig_constraints = self._constraint_detector.detect_constraints(original)
        orig_exclusions = [c for c in orig_constraints if c.category == "exclusion"]

        # 1. Missing negation present in original
        missing = orig_negs - comp_negs
        for term in sorted(missing):
            issues.append(
                ValidationIssue(
                    category="negation",
                    description=f"Negation marker '{term}' from original prompt is missing in compressed prompt.",
                    severity="error",
                    original_evidence=term,
                )
            )

        # If exclusion constraint was present in original, compressed must still have exclusion
        if orig_exclusions and not comp_negs:
            issues.append(
                ValidationIssue(
                    category="negation",
                    description=f"Exclusion constraint '{orig_exclusions[0].matched_text}' missing in compressed prompt.",
                    severity="error",
                    original_evidence=orig_exclusions[0].matched_text,
                )
            )

        # 2. Polarity flip: compressed introduced negation where original had none
        introduced = comp_negs - orig_negs
        if introduced and not orig_negs:
            for term in sorted(introduced):
                issues.append(
                    ValidationIssue(
                        category="negation",
                        description=f"New negation marker '{term}' introduced in compressed prompt that was not in original prompt.",
                        severity="error",
                        compressed_evidence=term,
                    )
                )

        info = {
            "original_negations": sorted(orig_negs),
            "compressed_negations": sorted(comp_negs),
            "missing": sorted(missing),
            "introduced": sorted(introduced),
        }
        return issues, info

    def _extract_numbers(self, text: str) -> Set[str]:
        """Extract canonical numeric representations (digits and normalized words)."""
        numbers = set()
        # Direct digits
        for m in _DIGIT_PATTERN.finditer(text):
            numbers.add(m.group(0).lower().rstrip("%"))

        # Word numbers
        for m in _WORD_NUMBER_PATTERN.finditer(text):
            w = m.group(0).lower()
            if w in _WORD_NUMBERS:
                numbers.add(_WORD_NUMBERS[w])

        # Ordinals
        for m in _ORDINAL_PATTERN.finditer(text):
            w = m.group(0).lower()
            if w in _ORDINAL_MAP:
                numbers.add(_ORDINAL_MAP[w])

        return numbers

    def _check_numerical(
        self, original: str, compressed: str
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Validate preservation of numerical and quantity constraints."""
        issues: List[ValidationIssue] = []
        orig_nums = self._extract_numbers(original)
        comp_nums = self._extract_numbers(compressed)

        missing_nums = orig_nums - comp_nums
        for num in sorted(missing_nums):
            issues.append(
                ValidationIssue(
                    category="numerical",
                    description=f"Numerical constraint '{num}' from original prompt is missing in compressed prompt.",
                    severity="error",
                    original_evidence=num,
                )
            )

        # Length constraints from analyzer
        orig_constraints = self._constraint_detector.detect_constraints(original)
        orig_lengths = [c for c in orig_constraints if c.category == "length"]
        comp_constraints = self._constraint_detector.detect_constraints(compressed)
        comp_lengths = [c for c in comp_constraints if c.category == "length"]

        if orig_lengths and not comp_lengths and missing_nums:
            issues.append(
                ValidationIssue(
                    category="numerical",
                    description=f"Length constraint '{orig_lengths[0].matched_text}' missing in compressed prompt.",
                    severity="error",
                    original_evidence=orig_lengths[0].matched_text,
                )
            )

        info = {
            "original_numbers": sorted(orig_nums),
            "compressed_numbers": sorted(comp_nums),
            "missing_numbers": sorted(missing_nums),
        }
        return issues, info

    def _extract_conditional_markers(self, text: str) -> Set[str]:
        """Extract conditional clause markers from text."""
        markers = set()
        for pat in _CONDITIONAL_PATTERNS:
            for match in pat.finditer(text):
                markers.add(match.group(0).lower())
        return markers

    def _check_conditionals(
        self, original: str, compressed: str
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Validate preservation of conditional and hypothetical clauses."""
        issues: List[ValidationIssue] = []
        orig_conds = self._extract_conditional_markers(original)
        comp_conds = self._extract_conditional_markers(compressed)

        missing_conds = orig_conds - comp_conds
        for marker in sorted(missing_conds):
            issues.append(
                ValidationIssue(
                    category="conditional",
                    description=f"Conditional clause marker '{marker}' from original prompt is missing in compressed prompt.",
                    severity="error",
                    original_evidence=marker,
                )
            )

        info = {
            "original_conditionals": sorted(orig_conds),
            "compressed_conditionals": sorted(comp_conds),
            "missing_conditionals": sorted(missing_conds),
        }
        return issues, info

    def _check_format(
        self, original: str, compressed: str
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Validate preservation of requested output formats and structures."""
        issues: List[ValidationIssue] = []
        orig_has_format = self._format_detector.has_output_format(original)
        comp_has_format = self._format_detector.has_output_format(compressed)

        orig_words = set(re.findall(r"\b[a-zA-Z0-9_\-+#]+\b", original.lower()))
        comp_words = set(re.findall(r"\b[a-zA-Z0-9_\-+#]+\b", compressed.lower()))

        orig_fmts = orig_words & _FORMAT_KEYWORDS
        comp_fmts = comp_words & _FORMAT_KEYWORDS
        missing_fmts = orig_fmts - comp_fmts

        for fmt in sorted(missing_fmts):
            issues.append(
                ValidationIssue(
                    category="format",
                    description=f"Output format keyword '{fmt}' from original prompt is missing in compressed prompt.",
                    severity="error",
                    original_evidence=fmt,
                )
            )

        if orig_has_format and not comp_has_format and not (comp_fmts & orig_fmts):
            issues.append(
                ValidationIssue(
                    category="format",
                    description="Output format requirement detected in original prompt was lost in compressed prompt.",
                    severity="error",
                    original_evidence="has_output_format=True",
                )
            )

        info = {
            "original_has_format": orig_has_format,
            "compressed_has_format": comp_has_format,
            "original_format_keywords": sorted(orig_fmts),
            "compressed_format_keywords": sorted(comp_fmts),
            "missing_format_keywords": sorted(missing_fmts),
        }
        return issues, info

    def _check_audience_tone(
        self, original: str, compressed: str
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Validate preservation of audience and tone modifiers."""
        issues: List[ValidationIssue] = []
        orig_constraints = self._constraint_detector.detect_constraints(original)
        orig_aud_constraints = [c for c in orig_constraints if c.category == "audience"]

        orig_words = set(re.findall(r"\b[a-zA-Z0-9_\-']+\b", original.lower()))
        comp_words = set(re.findall(r"\b[a-zA-Z0-9_\-']+\b", compressed.lower()))

        orig_aud_words = orig_words & _AUDIENCE_KEYWORDS
        comp_aud_words = comp_words & _AUDIENCE_KEYWORDS
        missing_aud = orig_aud_words - comp_aud_words

        orig_tone_words = orig_words & _TONE_KEYWORDS
        comp_tone_words = comp_words & _TONE_KEYWORDS
        missing_tone = orig_tone_words - comp_tone_words

        # Check multi-word audience/tone phrases
        orig_lower = original.lower()
        comp_lower = compressed.lower()

        for phrase in ["simple terms", "plain terms", "easy terms", "5 year old", "five year old", "high school"]:
            if phrase in orig_lower and phrase not in comp_lower:
                missing_tone.add(phrase)

        for item in sorted(missing_aud):
            issues.append(
                ValidationIssue(
                    category="audience_tone",
                    description=f"Audience modifier '{item}' from original prompt is missing in compressed prompt.",
                    severity="error",
                    original_evidence=item,
                )
            )

        for item in sorted(missing_tone):
            issues.append(
                ValidationIssue(
                    category="audience_tone",
                    description=f"Tone modifier '{item}' from original prompt is missing in compressed prompt.",
                    severity="error",
                    original_evidence=item,
                )
            )

        if orig_aud_constraints and (missing_aud or missing_tone):
            # Check if any original constraint was completely eliminated
            comp_constraints = self._constraint_detector.detect_constraints(compressed)
            comp_aud_constraints = [c for c in comp_constraints if c.category == "audience"]
            if not comp_aud_constraints:
                issues.append(
                    ValidationIssue(
                        category="audience_tone",
                        description=f"Audience constraint '{orig_aud_constraints[0].matched_text}' missing in compressed prompt.",
                        severity="error",
                        original_evidence=orig_aud_constraints[0].matched_text,
                    )
                )

        info = {
            "original_audience_words": sorted(orig_aud_words),
            "compressed_audience_words": sorted(comp_aud_words),
            "original_tone_words": sorted(orig_tone_words),
            "compressed_tone_words": sorted(comp_tone_words),
            "missing_audience": sorted(missing_aud),
            "missing_tone": sorted(missing_tone),
        }
        return issues, info

    def _check_examples(
        self, original: str, compressed: str
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Validate preservation of required example directives."""
        issues: List[ValidationIssue] = []
        orig_has_ex = self._example_detector.has_examples_requirement(original)
        comp_has_ex = self._example_detector.has_examples_requirement(compressed)

        orig_constraints = self._constraint_detector.detect_constraints(original)
        orig_ex_counts = [c for c in orig_constraints if c.category == "example_count"]

        if orig_has_ex and not comp_has_ex:
            evidence = orig_ex_counts[0].matched_text if orig_ex_counts else "examples_requirement"
            issues.append(
                ValidationIssue(
                    category="examples",
                    description="Example requirement present in original prompt was lost in compressed prompt.",
                    severity="error",
                    original_evidence=evidence,
                )
            )

        info = {
            "original_has_examples_requirement": orig_has_ex,
            "compressed_has_examples_requirement": comp_has_ex,
            "original_example_constraints": [c.matched_text for c in orig_ex_counts],
        }
        return issues, info

    def _extract_entities(self, text: str) -> Set[str]:
        """Extract quoted expressions, acronyms, mixed-case entities, and proper nouns."""
        entities = set()

        # 1. Quoted terms or code spans
        for m in _QUOTED_OR_CODE_PATTERN.finditer(text):
            val = m.group(1) or m.group(2) or m.group(3) or m.group(0)
            if val and len(val.strip()) > 1:
                entities.add(val.strip())

        # 2. Acronyms (length >= 2, uppercase) and mixed-case entities (e.g. NoSQL, PostgreSQL, GraphQL)
        for m in re.finditer(
            r"\b(?:[A-Z0-9]{2,}|[A-Z][a-z]+[A-Z][a-zA-Z0-9]*|[A-Z]{2,}[a-z]+[a-zA-Z0-9]*)\b",
            text,
        ):
            acr = m.group(0)
            if acr.lower() not in _DISCOURSE_AND_BOILERPLATE and acr.lower() not in _WORD_NUMBERS:
                entities.add(acr)

        # 3. Capitalized words not at sentence start
        sentences = re.split(r"(?<=[.!?\n])\s+", text)
        for sent in sentences:
            tokens = re.findall(r"\b[A-Z][a-z]+(?:'[a-z]+)?\b", sent)
            # First token of sentence might just be capitalized for grammatical reasons
            start_tokens = tokens[:1]
            non_start_tokens = tokens[1:]

            for t in non_start_tokens:
                if t.lower() not in _DISCOURSE_AND_BOILERPLATE and t.lower() not in _WORD_NUMBERS:
                    entities.add(t)

            # Check start tokens if they are distinct entities not in boilerplate
            for t in start_tokens:
                if (
                    t.lower() not in _DISCOURSE_AND_BOILERPLATE
                    and t.lower() not in _WORD_NUMBERS
                    and len(t) > 3
                ):
                    # If it's a known proper entity (e.g. Australia, Fibonacci, Bayes), include it
                    if t.lower() in {
                        "australia", "fibonacci", "bayes", "newton", "euler", "python",
                        "docker", "kubernetes", "sql", "nosql",
                    }:
                        entities.add(t)

        return entities

    def _check_entities(
        self, original: str, compressed: str
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Validate preservation of named entities, acronyms, and proper nouns."""
        issues: List[ValidationIssue] = []
        orig_entities = self._extract_entities(original)
        comp_lower = compressed.lower()

        missing_entities: Set[str] = set()
        for ent in orig_entities:
            # Case-insensitive containment check in compressed
            if ent.lower() not in comp_lower:
                missing_entities.add(ent)
                issues.append(
                    ValidationIssue(
                        category="entity",
                        description=f"Entity or proper noun '{ent}' from original prompt is missing in compressed prompt.",
                        severity="error",
                        original_evidence=ent,
                    )
                )

        info = {
            "original_entities": sorted(orig_entities),
            "missing_entities": sorted(missing_entities),
        }
        return issues, info

    def _extract_technical_terms(self, text: str) -> Set[str]:
        """Extract technical terms, code identifiers, and domain jargon."""
        terms = set()
        text_lower = text.lower()

        # 1. Multi-word technical terms
        for term in _COMMON_TECHNICAL_TERMS:
            if term in text_lower:
                terms.add(term)

        # 2. Code identifiers (snake_case, camelCase, dot notations, symbols)
        for m in _TECHNICAL_IDENTIFIER_PATTERN.finditer(text):
            ident = m.group(0)
            if ident.lower() not in _DISCOURSE_AND_BOILERPLATE and ident.lower() not in _WORD_NUMBERS:
                terms.add(ident.lower())

        # 3. Salient technical content words (length >= 5, not in boilerplate or numbers)
        words = re.findall(r"\b[a-zA-Z]{5,}\b", text.lower())
        for w in words:
            if (
                w not in _DISCOURSE_AND_BOILERPLATE
                and w not in _WORD_NUMBERS
                and w not in _ORDINAL_MAP
                and w not in _FORMAT_KEYWORDS
                and w not in _AUDIENCE_KEYWORDS
                and w not in _TONE_KEYWORDS
            ):
                # Exclude common polite or instruction terms
                if not any(
                    stem in w
                    for stem in ("explain", "summar", "describ", "analy", "defin", "clarif", "compar", "provid")
                ):
                    terms.add(w)

        return terms

    def _check_technical_terms(
        self, original: str, compressed: str
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Validate preservation of important technical terms and identifiers."""
        issues: List[ValidationIssue] = []
        orig_terms = self._extract_technical_terms(original)
        comp_lower = compressed.lower()

        missing_terms: Set[str] = set()
        for term in orig_terms:
            if term not in comp_lower:
                missing_terms.add(term)
                issues.append(
                    ValidationIssue(
                        category="technical_term",
                        description=f"Technical term or identifier '{term}' from original prompt is missing in compressed prompt.",
                        severity="error",
                        original_evidence=term,
                    )
                )

        info = {
            "original_technical_terms": sorted(orig_terms),
            "missing_technical_terms": sorted(missing_terms),
        }
        return issues, info

    def _check_uncertainty(
        self, original: str, compressed: str
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Flag conservative uncertainty when semantic equivalence cannot be established locally."""
        issues: List[ValidationIssue] = []

        orig_len = len(original)
        comp_len = len(compressed)

        # 1. Severe length reduction on non-trivial prompt (> 70% reduction)
        if orig_len > 30 and comp_len > 0:
            reduction = (orig_len - comp_len) / orig_len
            if reduction > 0.70:
                issues.append(
                    ValidationIssue(
                        category="uncertainty",
                        description=f"Extreme text reduction of {reduction * 100:.1f}% exceeds local heuristic certainty boundaries.",
                        severity="warning",
                        original_evidence=f"length={orig_len}",
                        compressed_evidence=f"length={comp_len}",
                    )
                )

        # 2. Dropped qualifying adverbs or degree modifiers
        qualifying_modifiers = ["thoroughly", "comprehensively", "rigorously", "deeply", "step by step", "in detail"]
        orig_lower = original.lower()
        comp_lower = compressed.lower()
        for mod in qualifying_modifiers:
            if mod in orig_lower and mod not in comp_lower:
                issues.append(
                    ValidationIssue(
                        category="uncertainty",
                        description=f"Qualifying modifier '{mod}' was dropped in compressed prompt; semantic depth may be altered.",
                        severity="warning",
                        original_evidence=mod,
                    )
                )

        info = {
            "original_length": orig_len,
            "compressed_length": comp_len,
            "warning_count": len(issues),
        }
        return issues, info


def get_constraint_validator() -> InstructionConstraintValidator:
    """Factory function returning the default instruction constraint validator."""
    return InstructionConstraintValidator()


def validate_instruction_compression(
    original_prompt: str, compressed_prompt: str
) -> ValidationResult:
    """Convenience function to validate constraint preservation in compressed prompts."""
    return InstructionConstraintValidator().validate(original_prompt, compressed_prompt)
