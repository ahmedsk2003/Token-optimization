"""Rule-based instruction compressor for prompt token optimization.

Phase 4.2A: Rule-Based Instruction Compression
- Conservatively shortens clearly verbose instruction phrases using explicit, documented rules.
- Converts nominalized verb phrases (e.g., "please provide an explanation of" -> "explain")
  and wordy request preambles to concise direct action verbs.
- Strictly preserves task intent, meaningful modifiers, entities, numbers, examples, audience,
  tone, output formats, and explicit constraints.
- Unsafe contexts (negation, conditionals, subordinate/prepositional clauses, third-person descriptions,
  intervening modifiers, and quoted/code text) are left completely unchanged.
- Follows existing BaseCompressor conventions and is independently callable.
"""

from dataclasses import dataclass
import re
from typing import Any, List, Optional, Set, Tuple

from src.compressors.redundancy import BaseCompressor


@dataclass(frozen=True)
class InstructionRule:
    """Represents an explicit, documented rule for shortening instruction phrases.

    Attributes:
        rule_id: Unique identifier for the rule.
        description: Human-readable explanation of rule intent and safety boundaries.
        pattern: Compiled regular expression matching the verbose phrasing.
        replacement: Base replacement string (or format string with {verb}).
    """

    rule_id: str
    description: str
    pattern: re.Pattern
    replacement: str


@dataclass(frozen=True)
class InstructionCompressionResult:
    """Detailed result container for rule-based instruction compression.

    Attributes:
        original_prompt: The raw input prompt string.
        compressed_prompt: The compressed prompt string after rule-based shortening.
        rules_applied: List of rule IDs that were successfully applied.
        replacements: List of tuples (rule_id, matched_text, replacement_text).
        original_token_count: Token count of original prompt (if tokenizer provided).
        compressed_token_count: Token count of compressed prompt (if tokenizer provided).
        tokens_saved: Number of tokens saved (if tokenizer provided).
        reduction_percentage: Percentage of tokens reduced (if tokenizer provided).
    """

    original_prompt: str
    compressed_prompt: str
    rules_applied: List[str]
    replacements: List[Tuple[str, str, str]]
    original_token_count: Optional[int] = None
    compressed_token_count: Optional[int] = None
    tokens_saved: Optional[int] = None
    reduction_percentage: Optional[float] = None
    validation_result: Optional[Any] = None


# Safeguard patterns
_NUMBER_PATTERN = re.compile(
    r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"hundred|thousand|million|billion)\b",
    re.IGNORECASE,
)

_FORMAT_KEYWORDS = frozenset({
    "json", "yaml", "xml", "csv", "table", "markdown", "bullet",
    "bullets", "list", "html", "latex", "sql", "code",
})

_CONSTRAINT_DIRECTIVES = frozenset({
    "must", "only", "required", "mandatory", "limit", "maximum",
    "max", "minimum", "min", "at least", "at most", "except",
})

_NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|never|don't|do\s+not|without|cannot|can't|must\s+not|"
    r"should\s+not|won't|will\s+not|avoid)\b",
    re.IGNORECASE,
)

_CONDITIONAL_OR_SUBORDINATE_PATTERN = re.compile(
    r"\b(?:if|unless|whether|while|instead\s+of|by|in\s+order\s+to|so\s+that|"
    r"because|since|although|though|whereas)\b",
    re.IGNORECASE,
)

_UNSAFE_PRECEDING_PREPOSITIONS = re.compile(
    r"\b(?:for|of|about|against|upon|through|with|without|from|during)\s*$",
    re.IGNORECASE,
)

_THIRD_PERSON_SUBJECT = re.compile(
    r"\b(?:the|this|that|these|those|each|every|a|an|my|our|their|his|her|"
    r"its|author|paper|article|document|section|chapter|guide|manual|"
    r"he|she|it|they)\s+\w*\s*$",
    re.IGNORECASE,
)

_CODE_OR_QUOTE_PATTERN = re.compile(
    r"```[\s\S]*?```|`[^`]*?`|\"[^\"]*?\"|'[^']*?'"
)


# Documented rule catalog for Phase 4.2A
DEFAULT_INSTRUCTION_RULES: List[InstructionRule] = [
    # 1. Polite Modal + Deverbal Phrase (Most specific)
    InstructionRule(
        rule_id="MODAL_PROVIDE_EXPLANATION",
        description="Shortens 'could/can you (please) provide/give an explanation of' to 'explain'.",
        pattern=re.compile(
            r"\b(?:could|can)\s+you\s+(?:please\s+)?(?:provide|give)\s+(?:an?\s+)?explanation\s+of\b",
            re.IGNORECASE,
        ),
        replacement="explain",
    ),
    InstructionRule(
        rule_id="MODAL_PROVIDE_SUMMARY",
        description="Shortens 'could/can you (please) provide/give a summary of' to 'summarize'.",
        pattern=re.compile(
            r"\b(?:could|can)\s+you\s+(?:please\s+)?(?:provide|give)\s+(?:a\s+)?summary\s+of\b",
            re.IGNORECASE,
        ),
        replacement="summarize",
    ),
    InstructionRule(
        rule_id="MODAL_PROVIDE_DESCRIPTION",
        description="Shortens 'could/can you (please) provide/give a description of' to 'describe'.",
        pattern=re.compile(
            r"\b(?:could|can)\s+you\s+(?:please\s+)?(?:provide|give)\s+(?:a\s+)?description\s+of\b",
            re.IGNORECASE,
        ),
        replacement="describe",
    ),
    InstructionRule(
        rule_id="MODAL_PROVIDE_LIST",
        description="Shortens 'could/can you (please) provide/give a list of' to 'list'.",
        pattern=re.compile(
            r"\b(?:could|can)\s+you\s+(?:please\s+)?(?:provide|give)\s+a\s+list\s+of\b",
            re.IGNORECASE,
        ),
        replacement="list",
    ),
    InstructionRule(
        rule_id="MODAL_DIRECT_VERB",
        description="Shortens polite modal requests with 'please' ('could/can you please explain') to direct action verb.",
        pattern=re.compile(
            r"\b(?:could|can)\s+you\s+please\s+(explain|summarize|describe|analyze|define|clarify)\b",
            re.IGNORECASE,
        ),
        replacement="{verb}",
    ),
    # 2. Desire Preambles ('I would like you to explain' -> 'Explain')
    InstructionRule(
        rule_id="DESIRE_DIRECT_VERB",
        description="Shortens 'I would like/want/need you to <verb>' to direct action verb.",
        pattern=re.compile(
            r"\bi\s+(?:would\s+like|want|need)\s+you\s+to\s+(explain|summarize|describe|analyze|compare|define|clarify|list)\b",
            re.IGNORECASE,
        ),
        replacement="{verb}",
    ),
    InstructionRule(
        rule_id="DESIRE_PROVIDE_EXPLANATION",
        description="Shortens 'I would like/want/need you to provide/give an explanation of' to 'explain'.",
        pattern=re.compile(
            r"\bi\s+(?:would\s+like|want|need)\s+you\s+to\s+(?:provide|give)\s+(?:an?\s+)?explanation\s+of\b",
            re.IGNORECASE,
        ),
        replacement="explain",
    ),
    InstructionRule(
        rule_id="DESIRE_PROVIDE_SUMMARY",
        description="Shortens 'I would like/want/need you to provide/give a summary of' to 'summarize'.",
        pattern=re.compile(
            r"\bi\s+(?:would\s+like|want|need)\s+you\s+to\s+(?:provide|give)\s+(?:a\s+)?summary\s+of\b",
            re.IGNORECASE,
        ),
        replacement="summarize",
    ),
    InstructionRule(
        rule_id="DESIRE_PROVIDE_DESCRIPTION",
        description="Shortens 'I would like/want/need you to provide/give a description of' to 'describe'.",
        pattern=re.compile(
            r"\bi\s+(?:would\s+like|want|need)\s+you\s+to\s+(?:provide|give)\s+(?:a\s+)?description\s+of\b",
            re.IGNORECASE,
        ),
        replacement="describe",
    ),
    InstructionRule(
        rule_id="DESIRE_PROVIDE_LIST",
        description="Shortens 'I would like/want/need you to provide/give a list of' to 'list'.",
        pattern=re.compile(
            r"\bi\s+(?:would\s+like|want|need)\s+you\s+to\s+(?:provide|give)\s+a\s+list\s+of\b",
            re.IGNORECASE,
        ),
        replacement="list",
    ),
    # 3. Direct Verbose Deverbal Phrases
    InstructionRule(
        rule_id="EXPLAIN_PROVIDE_EXPLANATION",
        description="Shortens '(please/kindly) provide/give an explanation of' to 'explain'.",
        pattern=re.compile(
            r"\b(?:please\s+|kindly\s+)?(?:provide|give)\s+(?:an?\s+)?explanation\s+of\b",
            re.IGNORECASE,
        ),
        replacement="explain",
    ),
    InstructionRule(
        rule_id="SUMMARIZE_PROVIDE_SUMMARY",
        description="Shortens '(please/kindly) provide/give a summary of' to 'summarize'.",
        pattern=re.compile(
            r"\b(?:please\s+|kindly\s+)?(?:provide|give)\s+(?:a\s+)?summary\s+of\b",
            re.IGNORECASE,
        ),
        replacement="summarize",
    ),
    InstructionRule(
        rule_id="DESCRIBE_PROVIDE_DESCRIPTION",
        description="Shortens '(please/kindly) provide/give a description of' to 'describe'.",
        pattern=re.compile(
            r"\b(?:please\s+|kindly\s+)?(?:provide|give)\s+(?:a\s+)?description\s+of\b",
            re.IGNORECASE,
        ),
        replacement="describe",
    ),
    InstructionRule(
        rule_id="ANALYZE_CONDUCT_ANALYSIS",
        description="Shortens '(please/kindly) conduct/perform/provide an analysis of' to 'analyze'.",
        pattern=re.compile(
            r"\b(?:please\s+|kindly\s+)?(?:conduct|perform|provide)\s+(?:an?\s+)?analysis\s+of\b",
            re.IGNORECASE,
        ),
        replacement="analyze",
    ),
    InstructionRule(
        rule_id="COMPARE_MAKE_COMPARISON",
        description="Shortens '(please/kindly) make/provide a comparison of/between' to 'compare'.",
        pattern=re.compile(
            r"\b(?:please\s+|kindly\s+)?(?:make|provide)\s+(?:a\s+)?comparison\s+(?:between|of)\b",
            re.IGNORECASE,
        ),
        replacement="compare",
    ),
    InstructionRule(
        rule_id="DEFINE_PROVIDE_DEFINITION",
        description="Shortens '(please/kindly) provide/give a definition of' to 'define'.",
        pattern=re.compile(
            r"\b(?:please\s+|kindly\s+)?(?:provide|give)\s+(?:a\s+)?definition\s+of\b",
            re.IGNORECASE,
        ),
        replacement="define",
    ),
    InstructionRule(
        rule_id="CLARIFY_PROVIDE_CLARIFICATION",
        description="Shortens '(please/kindly) provide/give clarification on/regarding/about' to 'clarify'.",
        pattern=re.compile(
            r"\b(?:please\s+|kindly\s+)?(?:provide|give)\s+clarification\s+(?:on|about|regarding)\b",
            re.IGNORECASE,
        ),
        replacement="clarify",
    ),
    InstructionRule(
        rule_id="LIST_PROVIDE_LIST",
        description="Shortens '(please/kindly) provide/give a list of' to 'list' (strictly without modifying adjectives).",
        pattern=re.compile(
            r"\b(?:please\s+|kindly\s+)?(?:provide|give)\s+a\s+list\s+of\b",
            re.IGNORECASE,
        ),
        replacement="list",
    ),
]


class RuleBasedInstructionCompressor(BaseCompressor):
    """Deterministic, rule-based instruction compressor for prompt token optimization.

    Phase 4.2A:
    - Conservatively replaces verbose instruction circumlocutions with direct action verbs.
    - Preserves entities, numbers, format constraints, tone, and directives.
    - Leaves text unchanged if any safeguard indicates replacement could alter task meaning.
    """

    def __init__(
        self,
        rules: Optional[List[InstructionRule]] = None,
        validator: Optional[Any] = None,
        reject_on_validation_failure: bool = False,
    ):
        """Initialize compressor with an explicit rule catalog and optional validator.

        Args:
            rules: Optional custom catalog of InstructionRule objects.
            validator: Optional validator instance with validate(original, compressed) -> ValidationResult.
            reject_on_validation_failure: If True, candidate compression is rejected and original prompt
                is retained if validation fails (is_valid is False).
        """
        self._rules = rules if rules is not None else DEFAULT_INSTRUCTION_RULES
        self._validator = validator
        self._reject_on_validation_failure = reject_on_validation_failure

    @property
    def validator(self) -> Optional[Any]:
        """Return the configured validator, if any."""
        return self._validator

    @property
    def reject_on_validation_failure(self) -> bool:
        """Return whether candidate compression is rejected on validation failure."""
        return self._reject_on_validation_failure

    @property
    def rules(self) -> List[InstructionRule]:
        """Return the active list of instruction rules."""
        return list(self._rules)

    def _get_protected_spans(self, text: str) -> List[Tuple[int, int]]:
        """Identify spans inside quotes or code blocks where replacement is prohibited."""
        return [m.span() for m in _CODE_OR_QUOTE_PATTERN.finditer(text)]

    def _is_inside_protected_span(
        self, start: int, end: int, protected_spans: List[Tuple[int, int]]
    ) -> bool:
        """Check if a match range overlaps with any protected code or quote span."""
        for p_start, p_end in protected_spans:
            if not (end <= p_start or start >= p_end):
                return True
        return False

    def _is_safe_to_replace(
        self, prompt: str, start: int, end: int, protected_spans: List[Tuple[int, int]]
    ) -> Tuple[bool, str]:
        """Verify whether replacing the matched span is safe in context.

        Args:
            prompt: Full input prompt.
            start: Start character index of the match.
            end: End character index of the match.
            protected_spans: Spans of code blocks and quotes.

        Returns:
            Tuple of (is_safe: bool, reason: str).
        """
        # 1. Protected code/quote check
        if self._is_inside_protected_span(start, end, protected_spans):
            return False, "inside_code_or_quotes"

        # Find clause/sentence start before the match
        clause_start = 0
        for i in range(start - 1, -1, -1):
            if prompt[i] in ".!?;:\n":
                clause_start = i + 1
                break

        clause_before = prompt[clause_start:start]

        # 2. Negation check within the clause
        if _NEGATION_PATTERN.search(clause_before):
            return False, "negation_detected"

        # 3. Conditional or subordinate clause check
        if _CONDITIONAL_OR_SUBORDINATE_PATTERN.search(clause_before):
            return False, "conditional_or_subordinate_clause"

        # 4. Preceding preposition check (gerund / prepositional complement)
        if _UNSAFE_PRECEDING_PREPOSITIONS.search(clause_before.rstrip()):
            return False, "prepositional_context"

        # 5. Third-person subject check (descriptive sentence, not a user directive)
        if _THIRD_PERSON_SUBJECT.search(clause_before.rstrip()):
            return False, "third_person_subject"

        return True, ""

    def _format_replacement(
        self, raw_replacement: str, match: re.Match, full_prompt: str
    ) -> str:
        """Determine replacement text with correct capitalization."""
        if "{verb}" in raw_replacement:
            base = match.group(1).lower()
        else:
            base = raw_replacement

        matched_text = match.group(0)
        match_start = match.start()

        # Check if match was capitalized or at sentence boundary
        is_at_sentence_start = (
            match_start == 0
            or full_prompt[:match_start].strip().endswith((".", "!", "?", "\n", ":"))
        )

        if matched_text[0].isupper() or is_at_sentence_start:
            return base[0].upper() + base[1:]
        return base.lower()

    def _validate_safeguards(self, original: str, compressed: str) -> bool:
        """Post-replacement safeguard validation to ensure no critical info was lost."""
        # 1. Numbers must not be lost
        orig_nums = set(m.lower() for m in _NUMBER_PATTERN.findall(original))
        comp_nums = set(m.lower() for m in _NUMBER_PATTERN.findall(compressed))
        if not orig_nums.issubset(comp_nums):
            return False

        # 2. Format keywords must not be lost
        orig_words = set(re.findall(r"\b[a-zA-Z0-9_\-+#]+\b", original.lower()))
        comp_words = set(re.findall(r"\b[a-zA-Z0-9_\-+#]+\b", compressed.lower()))
        orig_formats = orig_words & _FORMAT_KEYWORDS
        comp_formats = comp_words & _FORMAT_KEYWORDS
        if not orig_formats.issubset(comp_formats):
            return False

        # 3. Explicit constraint directives must not be lost
        orig_constraints = orig_words & _CONSTRAINT_DIRECTIVES
        comp_constraints = comp_words & _CONSTRAINT_DIRECTIVES
        if not orig_constraints.issubset(comp_constraints):
            return False

        # 4. Negation keywords must not be lost
        orig_negs = set(m.lower() for m in _NEGATION_PATTERN.findall(original))
        comp_negs = set(m.lower() for m in _NEGATION_PATTERN.findall(compressed))
        if not orig_negs.issubset(comp_negs):
            return False

        return True

    def compress(
        self,
        prompt: str,
        validator: Optional[Any] = None,
        reject_on_validation_failure: Optional[bool] = None,
    ) -> str:
        """Compress the prompt string using rule-based instruction shortening.

        Args:
            prompt: Raw user input prompt string.
            validator: Optional validator instance overriding compressor default.
            reject_on_validation_failure: Optional boolean overriding compressor default.

        Returns:
            Compressed prompt string, or original if unsafe / unchanged / rejected.
        """
        result = self.compress_with_stats(
            prompt,
            tokenizer=None,
            validator=validator,
            reject_on_validation_failure=reject_on_validation_failure,
        )
        return result.compressed_prompt

    def compress_with_stats(
        self,
        prompt: str,
        tokenizer: Optional[Any] = None,
        validator: Optional[Any] = None,
        reject_on_validation_failure: Optional[bool] = None,
    ) -> InstructionCompressionResult:
        """Compress the prompt and calculate token reduction statistics.

        Args:
            prompt: Raw user input prompt string.
            tokenizer: Optional tokenizer instance with count_tokens(str) -> int.
            validator: Optional validator instance overriding compressor default.
            reject_on_validation_failure: Optional boolean overriding compressor default.

        Returns:
            InstructionCompressionResult containing metrics, applied rules, and validation result.
        """
        safe_prompt = prompt if prompt is not None else ""
        stripped = safe_prompt.strip()

        active_validator = validator if validator is not None else self._validator
        active_reject = (
            reject_on_validation_failure
            if reject_on_validation_failure is not None
            else self._reject_on_validation_failure
        )

        if not stripped:
            val_result = (
                active_validator.validate(safe_prompt, "")
                if active_validator is not None
                else None
            )
            return InstructionCompressionResult(
                original_prompt=safe_prompt,
                compressed_prompt="",
                rules_applied=[],
                replacements=[],
                original_token_count=0 if tokenizer is not None else None,
                compressed_token_count=0 if tokenizer is not None else None,
                tokens_saved=0 if tokenizer is not None else None,
                reduction_percentage=0.0 if tokenizer is not None else None,
                validation_result=val_result,
            )

        protected_spans = self._get_protected_spans(safe_prompt)

        # Collect candidate replacements across rules
        candidates: List[Tuple[int, int, str, str, str]] = []  # (start, end, repl, rule_id, matched_text)
        occupied_spans: List[Tuple[int, int]] = []

        for rule in self._rules:
            for match in rule.pattern.finditer(safe_prompt):
                start, end = match.span()

                # Avoid overlapping with an already accepted candidate
                overlaps = False
                for occ_start, occ_end in occupied_spans:
                    if not (end <= occ_start or start >= occ_end):
                        overlaps = True
                        break
                if overlaps:
                    continue

                is_safe, _ = self._is_safe_to_replace(
                    safe_prompt, start, end, protected_spans
                )
                if not is_safe:
                    continue

                formatted_repl = self._format_replacement(
                    rule.replacement, match, safe_prompt
                )
                candidates.append((start, end, formatted_repl, rule.rule_id, match.group(0)))
                occupied_spans.append((start, end))

        if not candidates:
            orig_tokens = tokenizer.count_tokens(safe_prompt) if tokenizer is not None else None
            val_result = (
                active_validator.validate(safe_prompt, safe_prompt)
                if active_validator is not None
                else None
            )
            return InstructionCompressionResult(
                original_prompt=safe_prompt,
                compressed_prompt=safe_prompt,
                rules_applied=[],
                replacements=[],
                original_token_count=orig_tokens,
                compressed_token_count=orig_tokens,
                tokens_saved=0 if tokenizer is not None else None,
                reduction_percentage=0.0 if tokenizer is not None else None,
                validation_result=val_result,
            )

        # Apply replacements from end of string to start to preserve character indices
        candidates.sort(key=lambda c: c[0], reverse=True)
        working_text = safe_prompt
        applied_rules: List[str] = []
        applied_replacements: List[Tuple[str, str, str]] = []

        for start, end, repl, rule_id, matched_text in candidates:
            working_text = working_text[:start] + repl + working_text[end:]
            applied_rules.append(rule_id)
            applied_replacements.append((rule_id, matched_text, repl))

        # Reverse back to forward order for output reporting
        applied_rules.reverse()
        applied_replacements.reverse()

        # Final safeguard check across whole compressed prompt
        if not self._validate_safeguards(safe_prompt, working_text):
            # Abort replacement if critical constraint was lost
            orig_tokens = tokenizer.count_tokens(safe_prompt) if tokenizer is not None else None
            val_result = (
                active_validator.validate(safe_prompt, working_text)
                if active_validator is not None
                else None
            )
            return InstructionCompressionResult(
                original_prompt=safe_prompt,
                compressed_prompt=safe_prompt,
                rules_applied=[],
                replacements=[],
                original_token_count=orig_tokens,
                compressed_token_count=orig_tokens,
                tokens_saved=0 if tokenizer is not None else None,
                reduction_percentage=0.0 if tokenizer is not None else None,
                validation_result=val_result,
            )

        # Optional preservation validation
        val_result = None
        if active_validator is not None:
            val_result = active_validator.validate(safe_prompt, working_text)
            if not val_result.is_valid and active_reject:
                # Revert candidate to original prompt if validation failed and rejection requested
                working_text = safe_prompt
                applied_rules = []
                applied_replacements = []

        orig_tokens = None
        comp_tokens = None
        tokens_saved = None
        reduction_pct = None

        if tokenizer is not None:
            orig_tokens = tokenizer.count_tokens(safe_prompt)
            comp_tokens = tokenizer.count_tokens(working_text)
            tokens_saved = max(0, orig_tokens - comp_tokens)
            if orig_tokens > 0:
                reduction_pct = (tokens_saved / orig_tokens) * 100.0
            else:
                reduction_pct = 0.0

        return InstructionCompressionResult(
            original_prompt=safe_prompt,
            compressed_prompt=working_text,
            rules_applied=applied_rules,
            replacements=applied_replacements,
            original_token_count=orig_tokens,
            compressed_token_count=comp_tokens,
            tokens_saved=tokens_saved,
            reduction_percentage=reduction_pct,
            validation_result=val_result,
        )


def compress_instruction(
    prompt: str,
    validator: Optional[Any] = None,
    reject_on_validation_failure: bool = False,
) -> str:
    """Convenience function to compress verbose instruction phrases in a prompt."""
    return RuleBasedInstructionCompressor(
        validator=validator,
        reject_on_validation_failure=reject_on_validation_failure,
    ).compress(prompt)


def get_instruction_compressor(
    validator: Optional[Any] = None,
    reject_on_validation_failure: bool = False,
) -> RuleBasedInstructionCompressor:
    """Factory function returning the default instruction compressor instance."""
    return RuleBasedInstructionCompressor(
        validator=validator,
        reject_on_validation_failure=reject_on_validation_failure,
    )
