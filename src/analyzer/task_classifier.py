"""Task classification module for prompt analysis.

Identifies the primary task characteristic/category of user prompts using
deterministic, rule-based heuristics without external dependencies or model calls.
"""

from abc import ABC, abstractmethod
import re
from typing import Optional, Set

TASK_EXPLANATION = "explanation"
TASK_SUMMARIZATION = "summarization"
TASK_COMPARISON = "comparison"
TASK_TRANSLATION = "translation"
TASK_CODE_GENERATION = "code_generation"
TASK_DEBUGGING = "debugging"
TASK_DATA_ANALYSIS = "data_analysis"
TASK_QUESTION_ANSWERING = "question_answering"
TASK_CREATIVE_WRITING = "creative_writing"
TASK_OTHER = "other"

SUPPORTED_TASK_TYPES = frozenset({
    TASK_EXPLANATION,
    TASK_SUMMARIZATION,
    TASK_COMPARISON,
    TASK_TRANSLATION,
    TASK_CODE_GENERATION,
    TASK_DEBUGGING,
    TASK_DATA_ANALYSIS,
    TASK_QUESTION_ANSWERING,
    TASK_CREATIVE_WRITING,
    TASK_OTHER,
})


class BaseTaskClassifier(ABC):
    """Abstract base class for prompt task classification."""

    @abstractmethod
    def classify(self, prompt: str) -> str:
        """Classify the primary task type of the prompt.

        Returns one of:
        - "explanation"
        - "summarization"
        - "comparison"
        - "translation"
        - "code_generation"
        - "debugging"
        - "data_analysis"
        - "question_answering"
        - "creative_writing"
        - "other"
        """
        pass


class RuleBasedTaskClassifier(BaseTaskClassifier):
    """Deterministic rule-based task classifier.

    Extracts the dominant task type using priority heuristics, vocabulary matching,
    syntactic constructs, and intent disambiguation rules to avoid false positives.
    """

    # Conversational preambles to strip before analyzing intent
    _PREAMBLE_PATTERN = re.compile(
        r"^(?:(?:please|kindly|can\s+you\s+(?:please\s+)?|could\s+you\s+(?:please\s+)?|"
        r"would\s+you\s+(?:please\s+)?|i\s+want\s+(?:you\s+)?to\s+|help\s+me\s+(?:to\s+)?|"
        r"i\s+need\s+(?:you\s+)?to\s+|i\s+would\s+like\s+(?:you\s+)?to\s+|"
        r"your\s+task\s+is\s+to\s+|you\s+are\s+tasked\s+with\s+|make\s+sure\s+to\s+|"
        r"hey(?:\s+there)?,?\s*|hi(?:\s+there)?,?\s*|hello(?:\s+there)?,?\s*)\s*)+",
        re.IGNORECASE,
    )

    # Known natural languages for translation detection
    _NATURAL_LANGUAGES: Set[str] = {
        "english", "spanish", "french", "german", "chinese", "mandarin", "cantonese",
        "japanese", "russian", "italian", "portuguese", "arabic", "hindi", "korean",
        "dutch", "latin", "greek", "turkish", "polish", "swedish", "norwegian",
        "danish", "finnish", "vietnamese", "thai", "hebrew", "persian", "farsi",
        "czech", "romanian", "hungarian", "ukrainian", "indonesian", "tagalog",
        "bengali", "urdu", "punjabi", "marathi", "tamil", "telugu", "swahili",
    }

    # Programming languages to distinguish code translation/generation from natural translation
    _PROGRAMMING_LANGUAGES: Set[str] = {
        "python", "javascript", "typescript", "c++", "cpp", "java", "c#", "csharp",
        "rust", "go", "golang", "ruby", "php", "swift", "kotlin", "scala", "sql",
        "bash", "shell", "powershell", "regex", "html", "css", "r", "perl", "code",
    }

    # 1. Debugging patterns
    _DEBUG_PATTERNS = [
        re.compile(r"\b(?:debug|troubleshoot)\b", re.IGNORECASE),
        re.compile(
            r"\b(?:fix|resolve|solve)\s+(?:the\s+|this\s+|any\s+|all\s+)?"
            r"(?:bug|bugs|error|errors|issue|issues|exception|exceptions|crash|crashes|"
            r"syntax\s+error|runtime\s+error|segmentation\s+fault|defect|failure|traceback)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:fix|repair)\s+(?:this|my|the)\s+"
            r"(?:[a-zA-Z0-9_\-\+\#]+\s+)?(?:code|function|script|snippet|program|query|method|class|implementation)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhy\s+(?:is|does)\b.*?\b(?:code|script|function|program|query|method|class|app)\b.*?"
            r"\b(?:not\s+working|failing|throwing|crashing|return(?:ing)?\s+none|return(?:ing)?\s+undefined|broken|fail|crash|throw)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhy\s+am\s+i\s+getting\s+(?:this\s+)?(?:error|exception|crash|traceback|undefined|nan)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhat(?:'s|\s+is)\s+wrong\s+with\s+(?:this|my|the)\s+"
            r"(?:[a-zA-Z0-9_\-\+\#]+\s+)?(?:code|function|script|snippet|implementation)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bfind\s+(?:the\s+)?(?:bug|bugs|error|errors|issue|issues|syntax\s+error)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:traceback\s+\(most\s+recent\s+call\s+last\)|"
            r"(?:runtimeerror|typeerror|valueerror|indexerror|keyerror|syntaxerror|"
            r"nameerror|attributeerror|zerodivisionerror|nullpointerexception))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bfix\s+(?:this|the|my)\s+(?:function|method|code|script)\s+so\s+it\b",
            re.IGNORECASE,
        ),
    ]

    # Debugging exclusion (educational explanation of bugs)
    _DEBUG_EXCLUSION_PATTERN = re.compile(
        r"^(?:explain|what\s+is)\s+(?:what\s+is\s+a\s+|a\s+)?(?:software\s+)?(?:bug|debugger|error\s+handling)\b",
        re.IGNORECASE,
    )

    # 2. Comparison patterns
    _COMPARISON_PATTERNS = [
        re.compile(
            r"\b(?:compare\s+and\s+contrast|compare)\b.*?\b(?:to|with|and|versus|vs\.?)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:comparison\s+(?:between|of))\b", re.IGNORECASE),
        re.compile(
            r"\b(?:what\s+(?:is|are)\s+the\s+differences?\s+between|differences?\s+between)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:distinguish\s+between)\b", re.IGNORECASE),
        re.compile(r"\bhow\s+do\s+[a-zA-Z0-9_\-+\s]+\s+differ\b", re.IGNORECASE),
        re.compile(
            r"\b[a-zA-Z0-9_\-\.\+]+\s+(?:vs\.?|versus)\s+[a-zA-Z0-9_\-\.\+]+\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhich\s+(?:one\s+)?(?:is|should\s+i\s+(?:use|choose))\b.*?\b(?:better|faster|preferable|or|vs|versus)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhich\s+is\s+(?:better|faster|cheaper|more\s+efficient|superior)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bpros\s+and\s+cons\s+of\b.*?\b(?:vs|versus|and|compared\s+to)\b",
            re.IGNORECASE,
        ),
    ]

    # 3. Translation patterns
    _TRANSLATION_PATTERNS = [
        re.compile(
            r"\btranslate\b.*?\b(?:into|to|in)\s+([a-zA-Z]+)(?![a-zA-Z0-9_\-\+\#])",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:give\s+(?:me\s+)?|provide\s+)?(?:the\s+)?([a-zA-Z]+)\s+translation\s+(?:of|for)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bhow\s+do\s+you\s+say\b.*?\bin\s+([a-zA-Z]+)(?![a-zA-Z0-9_\-\+\#])",
            re.IGNORECASE,
        ),
        re.compile(
            r"\btranslate\s+(?:this|the\s+following)\s*:",
            re.IGNORECASE,
        ),
    ]

    # 4. Code generation patterns
    _CODE_GEN_PATTERNS = [
        re.compile(
            r"\b(?:write|create|generate|implement|code|build|develop)\s+(?:a|an|the|some)?\s*"
            r"(?:[a-zA-Z0-9\+\#\.]+\s+)?(?:script|function|program|code|algorithm|class|method|"
            r"module|query|endpoint|api|component|application|app|service|scraper|parser|"
            r"regex|regular\s+expression|unit\s+tests?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:write|code|implement)\s+(?:a|an|in)?\s*"
            r"(?:python|javascript|typescript|c\+\+|cpp|java|c#|csharp|rust|golang|ruby|php|swift|"
            r"kotlin|scala|sql|bash|powershell|regex)\s+"
            r"(?:script|function|program|code|class|method|query|app|to|that|for)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:write|give\s+me|generate)\s+(?:a\s+|an\s+|some\s+)?"
            r"(?:sql\s+query|regex|bash\s+script|python\s+script|code\s+snippet|code)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:code\s+a|implement\s+a)\s+"
            r"(?:binary\s+search|sorting|quicksort|linked\s+list|stack|queue|tree|graph|rest\s+api|crawler|scraper|algorithm)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:translate|convert|rewrite)\b.*?\b(?:into|to)\s+"
            r"(?:python|javascript|typescript|c\+\+|cpp|java|c#|rust|go|golang|ruby|php|swift|kotlin|sql|bash|code)(?![a-zA-Z0-9_\-\+\#])",
            re.IGNORECASE,
        ),
    ]

    # 5. Summarization patterns
    _SUMMARIZATION_PATTERNS = [
        re.compile(r"\b(?:summarize|summarise|condense|sum\s+up)\b", re.IGNORECASE),
        re.compile(
            r"\b(?:give|provide|write)\s+(?:me\s+)?(?:a\s+)?"
            r"(?:brief\s+|short\s+|quick\s+|one-sentence\s+|concise\s+)?"
            r"(?:summary|synopsis|digest|abstract)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:briefly\s+summarize|executive\s+summary|in\s+a\s+nutshell)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:tl;?dr|tldr)\b", re.IGNORECASE),
        re.compile(
            r"\b(?:key\s+takeaways|main\s+points|key\s+points|core\s+points)\b",
            re.IGNORECASE,
        ),
    ]

    # 6. Creative writing patterns
    _CREATIVE_WRITING_PATTERNS = [
        re.compile(
            r"\b(?:write|compose|craft|create|draft|tell|narrate)\b.*?\b"
            r"(?:story|stories|tale|tales|fairy\s+tale|myth|myths|legend|fable|poem|poems|poetry|"
            r"rhyme|haiku|limerick|ballad|sonnet|song|song\s+lyrics|lyrics|rap|play|screenplay|"
            r"scene|dialogue|monologue|novel|fanfic|fanfiction)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:creative\s+writing|write\s+(?:a\s+)?creative\s+piece)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:write|tell)\s+(?:me\s+)?a\s+joke\b", re.IGNORECASE),
        re.compile(r"\b(?:continue|finish)\s+(?:this|the)\s+story\b", re.IGNORECASE),
        re.compile(r"\bonce\s+upon\s+a\s+time\b", re.IGNORECASE),
    ]

    # 7. Data analysis patterns
    _DATA_ANALYSIS_PATTERNS = [
        re.compile(
            r"\b(?:analyze|analyse|perform\s+(?:a\s+)?(?:data\s+|statistical\s+)?analysis|"
            r"exploratory\s+data\s+analysis|eda)\b.*?\b(?:data|dataset|csv|metrics|statistics|"
            r"figures|numbers|sales|dataframe|survey\s+responses?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:calculate|compute|find|determine)\s+(?:the\s+)?"
            r"(?:mean|median|mode|standard\s+deviation|variance|correlation|distribution|"
            r"percentile|quartile|average|statistical\s+significance|p-value)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:insights|trends|patterns|conclusions)\b.*?\b(?:from)\b.*?\b(?:data|dataset|metrics|table|csv|survey|numbers|sales)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:statistical\s+analysis|trend\s+analysis|regression\s+analysis|"
            r"correlation\s+analysis|cohort\s+analysis)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\banalyze\s+(?:the\s+)?(?:data|dataset|metrics|sales|trends|survey\s+results)\b",
            re.IGNORECASE,
        ),
    ]

    # 8. Explanation patterns
    _EXPLANATION_PATTERNS = [
        re.compile(
            r"\b(?:explain|elaborate\s+on|clarify|walk\s+me\s+through|break\s+down)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:how\s+does\b.*?\b(?:work|function|operate|protect|cause|affect|learn|develop|happen|occur))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:how\s+do\b.*?\b(?:work|function|operate|protect|cause|affect|learn|develop|happen|occur))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:give|provide)\s+(?:an|a\s+detailed)\s+explanation\s+of\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:help\s+me\s+understand|understand\s+(?:why|how|what))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:explain\s+(?:the\s+concept\s+of|how|why|what\s+is|what\s+are))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:describe\s+how\b|describe\s+the\s+(?:process|mechanism|concept|functioning)\s+of)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:why\s+does\b.*?\b(?:happen|occur|work))\b",
            re.IGNORECASE,
        ),
    ]

    # 9. Question answering patterns (factual inquiries and reading comprehension)
    _QA_PATTERNS = [
        re.compile(
            r"\bwhat\s+(?:is|are|was|were)\s+(?:the\s+)?"
            r"(?:capital|population|height|elevation|depth|distance|speed|inventor|"
            r"author|currency|name|formula)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwho\s+(?:is|was|were|invented|discovered|wrote|founded|directed|painted|created|won)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bwhen\s+(?:was|did|were|is)\b", re.IGNORECASE),
        re.compile(r"\bwhere\s+(?:is|are|was|were|can\s+one\s+find)\b", re.IGNORECASE),
        re.compile(
            r"\bwhich\s+(?:planet|country|city|element|ocean|river|state|year|animal|organ)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bhow\s+(?:many|much|tall|old|long|far|high|heavy)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:answer\s+(?:the\s+following\s+)?question|based\s+on\s+the\s+(?:passage|text|text\s+above|context|article|above),\s*"
            r"(?:what|who|when|where|which|how|why))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?:is|are|can|does|did|do|was|were)\s+[a-zA-Z0-9\s,\-]+\?\s*$",
            re.IGNORECASE,
        ),
    ]

    def _strip_preamble(self, prompt: str) -> str:
        """Strip conversational polite openers to uncover core instruction intent."""
        return self._PREAMBLE_PATTERN.sub("", prompt.strip()).strip()

    def _match_debugging(self, cleaned: str, raw: str) -> bool:
        if self._DEBUG_EXCLUSION_PATTERN.search(cleaned):
            return False
        return any(pattern.search(cleaned) or pattern.search(raw) for pattern in self._DEBUG_PATTERNS)

    def _match_comparison(self, cleaned: str) -> bool:
        return any(pattern.search(cleaned) for pattern in self._COMPARISON_PATTERNS)

    def _match_translation(self, cleaned: str) -> bool:
        for pattern in self._TRANSLATION_PATTERNS:
            match = pattern.search(cleaned)
            if match:
                groups = match.groups()
                if groups and groups[0]:
                    target = groups[0].lower()
                    # If target is a programming language, it is code generation, not translation
                    if target in self._PROGRAMMING_LANGUAGES:
                        return False
                    if target in self._NATURAL_LANGUAGES:
                        return True
                else:
                    return True
        return False

    def _match_code_generation(self, cleaned: str) -> bool:
        return any(pattern.search(cleaned) for pattern in self._CODE_GEN_PATTERNS)

    def _match_summarization(self, cleaned: str) -> bool:
        return any(pattern.search(cleaned) for pattern in self._SUMMARIZATION_PATTERNS)

    def _match_creative_writing(self, cleaned: str) -> bool:
        return any(pattern.search(cleaned) for pattern in self._CREATIVE_WRITING_PATTERNS)

    def _match_data_analysis(self, cleaned: str) -> bool:
        return any(pattern.search(cleaned) for pattern in self._DATA_ANALYSIS_PATTERNS)

    def _match_explanation(self, cleaned: str) -> bool:
        return any(pattern.search(cleaned) for pattern in self._EXPLANATION_PATTERNS)

    def _match_question_answering(self, cleaned: str) -> bool:
        return any(pattern.search(cleaned) for pattern in self._QA_PATTERNS)

    def classify(self, prompt: str) -> str:
        """Classify the primary task type of the prompt using rule-based heuristics."""
        if prompt is None:
            return TASK_OTHER

        cleaned = self._strip_preamble(prompt)
        if not cleaned:
            return TASK_OTHER

        # Detect candidate matches
        is_debug = self._match_debugging(cleaned, prompt)
        is_comp = self._match_comparison(cleaned)
        is_code = self._match_code_generation(cleaned)
        is_trans = self._match_translation(cleaned)
        is_summ = self._match_summarization(cleaned)
        is_creative = self._match_creative_writing(cleaned)
        is_data = self._match_data_analysis(cleaned)
        is_explain = self._match_explanation(cleaned)
        is_qa = self._match_question_answering(cleaned)

        # Disambiguation & Priority Rules:
        # 1. Debugging: Specific failure/traceback/bug fixing takes priority over general code gen or explanation
        if is_debug:
            return TASK_DEBUGGING

        # 2. Code generation vs translation: e.g. "translate this code to Python"
        if is_code and is_trans:
            return TASK_CODE_GENERATION

        # 3. Comparison: "compare X and Y" takes precedence over explanation ("explain difference") or QA
        if is_comp:
            return TASK_COMPARISON

        # 4. Translation taking precedence over incidental QA inside translated quotes (e.g. "How do you say 'Where is X?' in German")
        if is_trans and is_qa:
            is_qa = False

        # 5. Code generation vs creative writing: "write a python script" should never be creative writing
        if is_code and is_creative:
            is_creative = False

        # Active category candidates
        matches = [
            (TASK_TRANSLATION, is_trans),
            (TASK_CODE_GENERATION, is_code),
            (TASK_SUMMARIZATION, is_summ),
            (TASK_CREATIVE_WRITING, is_creative),
            (TASK_DATA_ANALYSIS, is_data),
            (TASK_EXPLANATION, is_explain),
            (TASK_QUESTION_ANSWERING, is_qa),
        ]
        active = [cat for cat, m in matches if m]

        # If exactly one category matched confidently
        if len(active) == 1:
            return active[0]

        # If zero categories or unresolved conflicting categories, fall back to "other"
        return TASK_OTHER


def get_task_classifier() -> BaseTaskClassifier:
    """Factory function returning the default task classifier instance."""
    return RuleBasedTaskClassifier()
