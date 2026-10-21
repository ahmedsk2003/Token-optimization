"""Core Task and Context Extraction module for prompt token optimization.

Phase 4.3A: Core Task and Context Extraction
- Independent key-information extractor operating entirely with local deterministic rules.
- Extracts structured fields:
  1. Main task or action (action verb, task type, confidence, primary directive)
  2. Key entities and named technologies (proper nouns, acronyms, languages, tools, concepts)
  3. Important context or subject matter (background context, core topic phrase, confidence)
- Reuses existing analyzer detectors (task classification, instruction detection,
  sentence segmentation, constraint detection, and format detection).
- Treats extracted information strictly as analysis output: does not rewrite, shorten,
  or delete any content from the original prompt.
- Explicitly flags uncertainty when fields cannot be determined confidently rather than
  hallucinating or inventing details.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.analyzer.constraint_detector import (
    BaseConstraintDetector,
    get_constraint_detector,
)
from src.analyzer.format_detector import (
    BaseOutputFormatDetector,
    get_output_format_detector,
)
from src.analyzer.instruction_detector import (
    BaseInstructionDetector,
    RuleBasedInstructionDetector,
    get_instruction_detector,
)
from src.analyzer.sentence_segmenter import (
    BaseSentenceSegmenter,
    get_sentence_segmenter,
)
from src.analyzer.task_classifier import (
    SUPPORTED_TASK_TYPES,
    TASK_OTHER,
    BaseTaskClassifier,
    get_task_classifier,
)
from src.compressors.redundancy import BaseCompressor


@dataclass(frozen=True)
class KeyInformationResult:
    """Documented, structured result container for core task and context extraction.

    Per Phase 4.3A specification, this output is purely analytical and does not modify,
    shorten, or delete any content from the original input prompt.

    Attributes:
        original_prompt: The raw user input prompt string (preserved exactly).
        main_task: Primary task or action directive (None if empty or not identifiable).
        action_verb: Primary action verb identified (e.g. 'explain', 'summarize', None if none).
        task_type: High-level task classification category from TaskClassifier
                   (e.g., 'explanation', 'summarization', 'code_generation', 'other', 'uncertain', 'empty').
        task_confidence: Confidence level in task identification ('high', 'medium', 'uncertain', 'empty').
        entities: List of detected named entities, proper nouns, acronyms, and quoted terms.
        technologies: List of detected programming languages, frameworks, systems, tools, and technical terms.
        key_entities_and_technologies: Combined, deduplicated list of entities and technologies in order of appearance.
        context: Contextual background sentences or clauses distinct from primary directives.
        subject_matter: Core subject matter or topic phrase being acted upon (None if missing/uncertain).
        context_confidence: Confidence level in context/subject extraction ('high', 'medium', 'uncertain', 'empty').
        is_uncertain: Boolean flag indicating if any core extraction dimension exhibits uncertainty or ambiguity.
        details: Diagnostic metadata dictionary capturing intermediate detector signals and breakdown.
    """

    original_prompt: str
    main_task: Optional[str]
    action_verb: Optional[str]
    task_type: str
    task_confidence: str
    entities: List[str]
    technologies: List[str]
    key_entities_and_technologies: List[str]
    context: List[str]
    subject_matter: Optional[str]
    context_confidence: str
    is_uncertain: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert structured result to a standard dictionary format."""
        return asdict(self)


class BaseKeyInformationExtractor(ABC):
    """Abstract base class for key-information extraction."""

    @abstractmethod
    def extract(self, prompt: str) -> KeyInformationResult:
        """Extract core task, entities/technologies, and context from prompt."""
        pass


# ---------------------------------------------------------------------------
# Lexical Knowledge Bases for Deterministic Extraction
# ---------------------------------------------------------------------------

# Programming languages (canonical lowercase)
_PROGRAMMING_LANGUAGES: Set[str] = {
    "python", "javascript", "typescript", "c++", "cpp", "java", "c#", "csharp",
    "rust", "go", "golang", "ruby", "php", "swift", "kotlin", "scala", "sql",
    "bash", "shell", "powershell", "regex", "html", "css", "r", "perl", "lua",
    "dart", "haskell", "elixir", "clojure", "matlab", "julia", "solidity",
}

# Frameworks, platforms, tools, and databases (canonical lowercase)
_FRAMEWORKS_AND_PLATFORMS: Set[str] = {
    "docker", "kubernetes", "k8s", "aws", "azure", "gcp", "react", "vue", "angular",
    "svelte", "next.js", "nextjs", "node.js", "nodejs", "express", "django", "flask",
    "fastapi", "spring", "spring boot", "redis", "postgresql", "postgres", "mysql",
    "mongodb", "sqlite", "cassandra", "dynamodb", "kafka", "rabbitmq", "spark",
    "hadoop", "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "pandas",
    "numpy", "git", "github", "gitlab", "linux", "ubuntu", "windows", "macos",
    "android", "ios", "graphql", "grpc", "rest", "api", "terraform", "ansible",
    "jenkins", "nginx", "apache", "elasticsearch", "neo4j", "snowflake", "databricks",
}

# Domain technical concepts and algorithms
_TECHNICAL_CONCEPTS: Set[str] = {
    "attention mechanism", "attention mechanisms", "transformer", "transformers",
    "quicksort", "binary search", "mergesort", "heapsort", "cryptography",
    "consensus protocol", "microservice", "microservices", "blockchain",
    "eigenvector", "eigenvectors", "eigenvalue", "eigenvalues", "backpropagation",
    "neural network", "neural networks", "deep learning", "machine learning",
    "hash table", "hash map", "time complexity", "space complexity", "worst-case",
    "best-case", "average-case", "recursion", "polymorphism", "concurrency",
    "deadlock", "mutex", "semaphore", "multithreading", "asynchronous",
    "reinforcement learning", "supervised learning", "unsupervised learning",
    "quantum computing", "quantum entanglement", "quantum annealing",
    "public key", "private key", "asymmetric encryption", "symmetric encryption",
    "fibonacci", "llm", "large language model", "large language models",
    "embedding", "embeddings", "autoscaling", "load balancing", "database indexing",
    "acid properties", "sharding", "replication", "raft", "paxos", "mapreduce",
}

# Technical identifier pattern (snake_case, camelCase, dot notations, symbols)
_TECHNICAL_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:[a-zA-Z0-9]+_[a-zA-Z0-9_]+|[a-z]+[A-Z][a-zA-Z0-9]*|[a-zA-Z_]\w*\.[a-zA-Z_]\w*)\b|"
    r"\b(?i:c\+\+|c#|\.net)\b"
)

# Quoted text or inline code spans
_QUOTED_OR_CODE_PATTERN = re.compile(r"```[\s\S]*?```|`([^`]+)`|\"([^\"]+)\"|'([^']+)'")

# Acronym pattern (2+ uppercase characters, alphanumeric)
_ACRONYM_PATTERN = re.compile(r"\b[A-Z0-9]{2,}\b")

# Conversational preambles to strip
_PREAMBLE_PATTERN = re.compile(
    r"^(?:(?:please|kindly|can\s+you\s+(?:please\s+)?|could\s+you\s+(?:please\s+)?|"
    r"would\s+you\s+(?:please\s+)?|i\s+want\s+(?:you\s+)?to\s+|help\s+me\s+(?:to\s+)?|"
    r"i\s+need\s+(?:you\s+)?to\s+|i\s+would\s+like\s+(?:you\s+)?to\s+|"
    r"your\s+task\s+is\s+to\s+|you\s+are\s+tasked\s+with\s+|make\s+sure\s+to\s+|"
    r"hey(?:\s+there)?,?\s*|hi(?:\s+there)?,?\s*|hello(?:\s+there)?,?\s*)\s*)+",
    re.IGNORECASE,
)

# Nominalized verbose verb phrases
_NOMINALIZED_TASK_PATTERNS = [
    (re.compile(r"^(?:provide|give)\s+(?:an?\s+)?explanation\s+(?:of|about|on)\s+", re.IGNORECASE), "explain"),
    (re.compile(r"^(?:provide|give)\s+(?:an?\s+)?overview\s+(?:of|about|on)\s+", re.IGNORECASE), "explain"),
    (re.compile(r"^(?:provide|give)\s+(?:a\s+)?summary\s+(?:of|about|on)\s+", re.IGNORECASE), "summarize"),
    (re.compile(r"^(?:provide|give)\s+(?:a\s+)?description\s+(?:of|about|on)\s+", re.IGNORECASE), "describe"),
    (re.compile(r"^(?:provide|give)\s+(?:an?\s+)?analysis\s+(?:of|about|on)\s+", re.IGNORECASE), "analyze"),
    (re.compile(r"^(?:provide|give)\s+(?:a\s+)?comparison\s+(?:of|between)\s+", re.IGNORECASE), "compare"),
    (re.compile(r"^(?:provide|give)\s+(?:a\s+)?definition\s+(?:of)\s+", re.IGNORECASE), "define"),
    (re.compile(r"^(?:provide|give)\s+(?:a\s+)?clarification\s+(?:of|on|about)\s+", re.IGNORECASE), "clarify"),
    (re.compile(r"^(?:provide|give)\s+(?:a\s+)?list\s+(?:of)\s+", re.IGNORECASE), "list"),
    (re.compile(r"^(?:provide|give)\s+(?:a\s+)?breakdown\s+(?:of)\s+", re.IGNORECASE), "analyze"),
]

# Common discourse stopwords and boilerplate words
_DISCOURSE_AND_BOILERPLATE: Set[str] = {
    "a", "an", "the", "this", "that", "these", "those", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "done",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our", "its", "please", "kindly", "could", "can",
    "would", "should", "will", "shall", "and", "but", "or", "so", "as", "than",
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about", "into",
    "through", "during", "before", "after", "above", "below", "also", "here", "there",
    "just", "well", "like", "want", "need", "make", "sure", "step", "way", "ways",
    "give", "gives", "giving", "provide", "provides", "providing",
}

# Trailing constraint / formatting suffixes to strip from subject matter
_TRAILING_CONSTRAINT_PATTERN = re.compile(
    r"(?:,\s*|\s+)(?:in\s+(?:simple|plain|easy|concise|brief|layman'?s?|academic|formal|informal)\s+terms|"
    r"for\s+(?:a\s+)?(?:beginner|beginners|kids|children|students|experts|novices)|"
    r"in\s+(?:markdown|json|yaml|xml|csv|a\s+table|tabular|bullet|bullets|list|single\s+line)|"
    r"as\s+(?:a\s+)?(?:markdown|json|yaml|xml|csv|table|bullet\s+list|list)|"
    r"(?:in|under|with)\s+(?:no\s+more\s+than\s+)?\d+\s+(?:words|sentences|paragraphs|bullets|items)|"
    r"concisely|briefly|step\s+by\s+step|in\s+detail)\.?$",
    re.IGNORECASE,
)

# Context clause prefixes for sentence-internal or paragraph context
_CONTEXT_INTRO_PATTERN = re.compile(
    r"^(?:given\s+(?:that\s+)?|assuming\s+(?:that\s+)?|in\s+(?:the\s+)?context\s+of\s+|"
    r"in\s+a\s+scenario\s+where\s+|for\s+a\s+system\s+(?:with|that|where)\s+|"
    r"we\s+(?:have|are|operate|run|maintain|built|experience)\s+|"
    r"our\s+(?:system|application|platform|company|team|pipeline|server)\s+|"
    r"suppose\s+(?:that\s+)?|in\s+our\s+)",
    re.IGNORECASE,
)


class KeyInformationExtractor(BaseKeyInformationExtractor, BaseCompressor):
    """Independent key-information extractor for prompt analysis.

    Extracts:
    1. Main task or action (action verb, category, directive phrase, confidence)
    2. Key entities and named technologies (languages, tools, systems, acronyms, proper nouns)
    3. Important context or subject matter (background clauses, subject phrase, confidence)

    Reuses existing analyzer detectors without modifying or shortening original prompt content.
    """

    def __init__(
        self,
        task_classifier: Optional[BaseTaskClassifier] = None,
        instruction_detector: Optional[BaseInstructionDetector] = None,
        segmenter: Optional[BaseSentenceSegmenter] = None,
        constraint_detector: Optional[BaseConstraintDetector] = None,
        format_detector: Optional[BaseOutputFormatDetector] = None,
    ):
        self.task_classifier = task_classifier if task_classifier is not None else get_task_classifier()
        self.instruction_detector = (
            instruction_detector if instruction_detector is not None else get_instruction_detector()
        )
        self.segmenter = segmenter if segmenter is not None else get_sentence_segmenter()
        self.constraint_detector = (
            constraint_detector if constraint_detector is not None else get_constraint_detector()
        )
        self.format_detector = format_detector if format_detector is not None else get_output_format_detector()

    def compress(self, prompt: str) -> str:
        """Adhere to BaseCompressor interface without altering original prompt.

        Per Phase 4.3A requirements, extracted key information is strictly analysis
        output. This method returns the original prompt text untouched.
        """
        if prompt is None:
            return ""
        return prompt

    def extract(self, prompt: str) -> KeyInformationResult:
        """Extract core task, entities/technologies, and context from input prompt.

        Args:
            prompt: Raw user input prompt string.

        Returns:
            Structured KeyInformationResult with analysis output and confidence indicators.
        """
        # Handle empty / None / whitespace-only inputs safely
        if prompt is None or not prompt.strip():
            return KeyInformationResult(
                original_prompt="" if prompt is None else prompt,
                main_task=None,
                action_verb=None,
                task_type="empty",
                task_confidence="empty",
                entities=[],
                technologies=[],
                key_entities_and_technologies=[],
                context=[],
                subject_matter=None,
                context_confidence="empty",
                is_uncertain=False,
                details={"reason": "empty_input"},
            )

        cleaned_prompt = prompt.strip()

        # Step 1: Detect tasks and action verbs
        task_info = self._extract_task_and_action(cleaned_prompt)

        # Step 2: Extract technologies and named entities
        entities, technologies, combined_entities_tech = self._extract_entities_and_technologies(cleaned_prompt)

        # Step 3: Extract context sentences and subject matter
        context_info = self._extract_context_and_subject(cleaned_prompt, task_info)

        # Step 4: Determine overall certainty
        is_uncertain = (
            task_info["task_confidence"] == "uncertain"
            or context_info["context_confidence"] == "uncertain"
        )

        return KeyInformationResult(
            original_prompt=prompt,
            main_task=task_info["main_task"],
            action_verb=task_info["action_verb"],
            task_type=task_info["task_type"],
            task_confidence=task_info["task_confidence"],
            entities=entities,
            technologies=technologies,
            key_entities_and_technologies=combined_entities_tech,
            context=context_info["context"],
            subject_matter=context_info["subject_matter"],
            context_confidence=context_info["context_confidence"],
            is_uncertain=is_uncertain,
            details={
                "task_details": task_info.get("details", {}),
                "context_details": context_info.get("details", {}),
                "detected_instructions": task_info.get("detected_instructions", []),
            },
        )

    # ---------------------------------------------------------------------------
    # Task and Action Extraction
    # ---------------------------------------------------------------------------

    def _extract_task_and_action(self, prompt: str) -> Dict[str, Any]:
        """Extract main task directive, primary action verb, task type, and confidence."""
        detected_instructions = self.instruction_detector.detect_instructions(prompt)
        classified_type = self.task_classifier.classify(prompt)

        sentences = self.segmenter.split_sentences(prompt)

        # Inspect if any sentence has an explicit action verb or question structure
        action_verb: Optional[str] = None
        main_task: Optional[str] = None
        task_confidence = "uncertain"
        task_type = classified_type

        # Look for explicit action verb in detected instructions or sentences
        for inst in detected_instructions:
            verb, stripped = self._find_action_verb(inst)
            if verb:
                action_verb = verb
                main_task = inst
                break

        # Check for nominalized verb phrases (e.g., "provide an explanation of")
        if not action_verb:
            for sent in sentences:
                sent_stripped = _PREAMBLE_PATTERN.sub("", sent.strip()).strip()
                for pat, v in _NOMINALIZED_TASK_PATTERNS:
                    if pat.search(sent_stripped):
                        action_verb = v
                        main_task = sent.strip()
                        break
                if action_verb:
                    break

        # Check for interrogative questions (e.g. "What is...", "How does...")
        is_question = False
        if not action_verb:
            for sent in sentences:
                sent_clean = sent.strip()
                if sent_clean.endswith("?") or re.match(
                    r"^(?:what|how|why|who|where|when|which|is|are|can|could)\b",
                    sent_clean,
                    re.IGNORECASE,
                ):
                    is_question = True
                    main_task = sent_clean
                    break

        if action_verb:
            task_confidence = "high"
        elif is_question:
            if classified_type == TASK_OTHER:
                task_type = "question_answering"
            task_confidence = "high"
        else:
            # Ambiguous or non-directive input (e.g., random keywords or declarative statement)
            main_task = None
            task_type = "uncertain"
            task_confidence = "uncertain"

        return {
            "main_task": main_task,
            "action_verb": action_verb,
            "task_type": task_type,
            "task_confidence": task_confidence,
            "detected_instructions": detected_instructions,
            "details": {
                "classified_task_type": classified_type,
                "is_question": is_question,
            },
        }

    def _find_action_verb(self, text: str) -> Tuple[Optional[str], str]:
        """Identify primary action verb from an instruction clause."""
        cleaned = _PREAMBLE_PATTERN.sub("", text.strip()).strip()
        tokens = re.findall(r"\b[a-zA-Z]+\b", cleaned)
        if not tokens:
            return None, cleaned

        first_word = tokens[0].lower()
        if first_word in RuleBasedInstructionDetector.ACTION_VERBS:
            return first_word, cleaned

        # Check first 3 words in case of conversational adverbs (e.g. "briefly explain")
        for word in tokens[:3]:
            w_lower = word.lower()
            if w_lower in RuleBasedInstructionDetector.ACTION_VERBS:
                return w_lower, cleaned

        return None, cleaned

    # ---------------------------------------------------------------------------
    # Entities and Technologies Extraction
    # ---------------------------------------------------------------------------

    def _extract_entities_and_technologies(
        self, prompt: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """Extract detected technologies, named entities, and an ordered combined list."""
        technologies: List[str] = []
        entities: List[str] = []
        seen_lower: Set[str] = set()

        text_lower = prompt.lower()

        # 1. Multi-word technical concepts (e.g., 'attention mechanisms', 'time complexity')
        for concept in sorted(_TECHNICAL_CONCEPTS, key=len, reverse=True):
            if concept in text_lower:
                # Find matching span in prompt for original casing
                match = re.search(r"\b" + re.escape(concept) + r"\b", prompt, re.IGNORECASE)
                val = match.group(0) if match else concept
                if concept not in seen_lower:
                    technologies.append(val)
                    seen_lower.add(concept)

        # 2. Programming languages and frameworks/tools
        for m in re.finditer(r"\b[a-zA-Z0-9_\-\+\#\.]+\b", prompt):
            word = m.group(0)
            w_lower = word.lower()
            if w_lower in _PROGRAMMING_LANGUAGES or w_lower in _FRAMEWORKS_AND_PLATFORMS:
                if w_lower not in seen_lower:
                    technologies.append(word)
                    seen_lower.add(w_lower)

        # 3. Code identifiers and symbols (e.g. `foo_bar`, `UserModel`, `os.path`)
        for m in _TECHNICAL_IDENTIFIER_PATTERN.finditer(prompt):
            ident = m.group(0)
            i_lower = ident.lower()
            if i_lower not in seen_lower and i_lower not in _DISCOURSE_AND_BOILERPLATE:
                technologies.append(ident)
                seen_lower.add(i_lower)

        # 4. Quoted terms or inline code spans
        for m in _QUOTED_OR_CODE_PATTERN.finditer(prompt):
            val = m.group(1) or m.group(2) or m.group(3) or m.group(0)
            if val and len(val.strip()) > 1:
                cleaned_val = val.strip()
                v_lower = cleaned_val.lower()
                if v_lower not in seen_lower and v_lower not in _DISCOURSE_AND_BOILERPLATE:
                    entities.append(cleaned_val)
                    seen_lower.add(v_lower)

        # 5. Acronyms (length >= 2, uppercase)
        for m in _ACRONYM_PATTERN.finditer(prompt):
            acr = m.group(0)
            a_lower = acr.lower()
            if a_lower not in seen_lower and a_lower not in _DISCOURSE_AND_BOILERPLATE:
                # If it's a known tech term, record under technologies; else entities
                if a_lower in _PROGRAMMING_LANGUAGES or a_lower in _FRAMEWORKS_AND_PLATFORMS:
                    technologies.append(acr)
                else:
                    entities.append(acr)
                seen_lower.add(a_lower)

        # 6. Proper nouns (capitalized words not at sentence starts, or multi-word proper nouns)
        sentences = re.split(r"(?<=[.!?\n])\s+", prompt)
        for sent in sentences:
            tokens = re.findall(r"\b[A-Z][a-z]+(?:'[a-z]+)?\b", sent)
            # Skip first token if capitalized solely for sentence initiation
            start_tokens = tokens[:1]
            body_tokens = tokens[1:]

            for t in body_tokens:
                t_lower = t.lower()
                if t_lower not in seen_lower and t_lower not in _DISCOURSE_AND_BOILERPLATE:
                    if t_lower in _PROGRAMMING_LANGUAGES or t_lower in _FRAMEWORKS_AND_PLATFORMS:
                        technologies.append(t)
                    else:
                        entities.append(t)
                    seen_lower.add(t_lower)

            # Check sentence starters for well-known proper names/entities
            for t in start_tokens:
                t_lower = t.lower()
                if (
                    t_lower not in seen_lower
                    and t_lower not in _DISCOURSE_AND_BOILERPLATE
                    and len(t) > 3
                    and t_lower in {
                        "australia", "fibonacci", "bayes", "newton", "euler", "leibniz",
                        "google", "microsoft", "amazon", "apple", "paris", "london",
                        "france", "germany", "japan", "china", "america", "alice", "bob",
                    }
                ):
                    entities.append(t)
                    seen_lower.add(t_lower)

        # Order combined entities and technologies by position in original prompt
        def _get_first_pos(term: str) -> int:
            idx = prompt.find(term)
            if idx == -1:
                idx = prompt.lower().find(term.lower())
            return idx if idx != -1 else 999999

        combined_entities_tech = sorted(
            list(dict.fromkeys(entities + technologies)),
            key=_get_first_pos,
        )

        return entities, technologies, combined_entities_tech

    # ---------------------------------------------------------------------------
    # Context and Subject Matter Extraction
    # ---------------------------------------------------------------------------

    def _extract_context_and_subject(
        self, prompt: str, task_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract background context sentences and core subject matter phrase."""
        sentences = self.segmenter.split_sentences(prompt)
        main_task = task_info.get("main_task")
        task_confidence = task_info.get("task_confidence")

        context: List[str] = []
        subject_matter: Optional[str] = None
        context_confidence = "uncertain"

        # 1. Identify background context sentences
        for sent in sentences:
            sent_clean = sent.strip()
            # If this sentence is the main task, skip it
            if main_task and (sent_clean in main_task or main_task in sent_clean):
                continue
            # If this sentence is a pure constraint, skip it
            constraints = self.constraint_detector.detect_constraints(sent_clean)
            has_format = self.format_detector.has_output_format(sent_clean)
            if (constraints or has_format) and len(sent_clean.split()) <= 10:
                continue

            # Check if sentence looks like context
            if _CONTEXT_INTRO_PATTERN.search(sent_clean) or len(sentences) > 1:
                context.append(sent_clean)

        # 2. Extract core subject matter
        if main_task:
            subject_matter = self._isolate_subject_matter(main_task)
            if subject_matter:
                context_confidence = "high"
            else:
                context_confidence = "uncertain"
        else:
            # In absence of a clear task directive, check if prompt is an informative topic phrase
            if task_confidence == "uncertain":
                # Check for ambiguous or nonsensical words
                words = re.findall(r"\b[a-zA-Z0-9]+\b", prompt)
                content_words = [
                    w for w in words if w.lower() not in _DISCOURSE_AND_BOILERPLATE
                ]
                if len(content_words) >= 2 and len(sentences) == 1:
                    # Informative topic phrase without an action verb
                    subject_matter = prompt.strip().rstrip(".?!")
                    context_confidence = "medium"
                else:
                    subject_matter = None
                    context_confidence = "uncertain"

        return {
            "context": context,
            "subject_matter": subject_matter,
            "context_confidence": context_confidence,
            "details": {
                "context_sentence_count": len(context),
            },
        }

    def _isolate_subject_matter(self, task_text: str) -> Optional[str]:
        """Isolate core topic phrase from a task directive."""
        cleaned = _PREAMBLE_PATTERN.sub("", task_text.strip()).strip()

        # Check nominalized verb phrases first
        for pat, _ in _NOMINALIZED_TASK_PATTERNS:
            match = pat.search(cleaned)
            if match:
                candidate = cleaned[match.end():].strip()
                return self._clean_subject_candidate(candidate)

        # Check direct action verb
        tokens = re.findall(r"\b[a-zA-Z]+\b", cleaned)
        if tokens and tokens[0].lower() in RuleBasedInstructionDetector.ACTION_VERBS:
            # Slice text after the first verb
            verb = tokens[0]
            verb_idx = cleaned.lower().find(verb.lower())
            if verb_idx != -1:
                candidate = cleaned[verb_idx + len(verb):].strip()
                return self._clean_subject_candidate(candidate)

        # Check question phrasing (e.g. "What is X", "How does X work")
        q_match = re.match(
            r"^(?:what\s+(?:is|are|was|were)\s+(?:an?\s+|the\s+)?|"
            r"how\s+(?:does|do|can|did)\s+|why\s+(?:is|are|does|do)\s+)",
            cleaned,
            re.IGNORECASE,
        )
        if q_match:
            candidate = cleaned[q_match.end():].strip()
            return self._clean_subject_candidate(candidate)

        # Fallback to cleaned text if non-empty
        return self._clean_subject_candidate(cleaned)

    def _clean_subject_candidate(self, candidate: str) -> Optional[str]:
        """Strip trailing constraints, punctuation, and extraneous whitespace."""
        if not candidate:
            return None

        # Strip trailing formatting and constraint phrases
        cleaned = _TRAILING_CONSTRAINT_PATTERN.sub("", candidate).strip()

        # Strip trailing punctuation
        cleaned = cleaned.rstrip(".!?:;,")

        # Strip any leading prepositions like 'of', 'about', 'on', 'between'
        cleaned = re.sub(r"^(?:of|about|on|between)\s+", "", cleaned, flags=re.IGNORECASE).strip()

        if not cleaned:
            return None

        return cleaned


def get_key_information_extractor(
    task_classifier: Optional[BaseTaskClassifier] = None,
    instruction_detector: Optional[BaseInstructionDetector] = None,
    segmenter: Optional[BaseSentenceSegmenter] = None,
    constraint_detector: Optional[BaseConstraintDetector] = None,
    format_detector: Optional[BaseOutputFormatDetector] = None,
) -> KeyInformationExtractor:
    """Factory function returning configured KeyInformationExtractor instance."""
    return KeyInformationExtractor(
        task_classifier=task_classifier,
        instruction_detector=instruction_detector,
        segmenter=segmenter,
        constraint_detector=constraint_detector,
        format_detector=format_detector,
    )


def extract_key_information(prompt: str) -> KeyInformationResult:
    """Convenience function to extract key information using default extractor instance."""
    return get_key_information_extractor().extract(prompt)
