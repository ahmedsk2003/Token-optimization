"""Unit tests for Phase 4.3A: Core Task and Context Extractor."""

import unittest

from src.compressors import (
    BaseCompressor,
    BaseKeyInformationExtractor,
    KeyInformationExtractor,
    KeyInformationResult,
    extract_key_information,
    get_key_information_extractor,
)


class TestKeyInformationExtractor(unittest.TestCase):
    """Test suite for KeyInformationExtractor."""

    def setUp(self):
        self.extractor = get_key_information_extractor()

    # 1. Empty and Whitespace Inputs
    def test_empty_string(self):
        res = self.extractor.extract("")
        self.assertIsInstance(res, KeyInformationResult)
        self.assertIsNone(res.main_task)
        self.assertIsNone(res.action_verb)
        self.assertEqual(res.task_type, "empty")
        self.assertEqual(res.task_confidence, "empty")
        self.assertEqual(res.entities, [])
        self.assertEqual(res.technologies, [])
        self.assertEqual(res.key_entities_and_technologies, [])
        self.assertEqual(res.context, [])
        self.assertIsNone(res.subject_matter)
        self.assertEqual(res.context_confidence, "empty")
        self.assertFalse(res.is_uncertain)
        self.assertEqual(res.original_prompt, "")

    def test_none_input(self):
        res = self.extractor.extract(None)
        self.assertIsNone(res.main_task)
        self.assertIsNone(res.action_verb)
        self.assertEqual(res.task_type, "empty")
        self.assertEqual(res.task_confidence, "empty")
        self.assertEqual(res.original_prompt, "")

    def test_whitespace_only_input(self):
        res = self.extractor.extract("   \n\t  \n  ")
        self.assertIsNone(res.main_task)
        self.assertEqual(res.task_confidence, "empty")
        self.assertEqual(res.context_confidence, "empty")

    def test_convenience_function_empty(self):
        res = extract_key_information("")
        self.assertEqual(res.task_confidence, "empty")

    # 2. Clear Tasks and Direct Action Verbs
    def test_clear_imperative_task(self):
        prompt = "Explain how transformer attention mechanisms work."
        res = self.extractor.extract(prompt)

        self.assertIsNotNone(res.main_task)
        self.assertEqual(res.action_verb, "explain")
        self.assertEqual(res.task_type, "explanation")
        self.assertEqual(res.task_confidence, "high")
        self.assertIn("transformer", [t.lower() for t in res.technologies])
        self.assertIsNotNone(res.subject_matter)
        self.assertIn("attention mechanism", res.subject_matter.lower())
        self.assertEqual(res.context_confidence, "high")
        self.assertFalse(res.is_uncertain)

    def test_clear_task_with_polite_preamble(self):
        prompt = "Could you please summarize the quarterly financial report in three bullet points?"
        res = self.extractor.extract(prompt)

        self.assertEqual(res.action_verb, "summarize")
        self.assertEqual(res.task_type, "summarization")
        self.assertEqual(res.task_confidence, "high")
        self.assertIsNotNone(res.subject_matter)
        self.assertIn("quarterly financial report", res.subject_matter.lower())

    def test_nominalized_verbose_task(self):
        prompt = "Please provide an explanation of how backpropagation updates model weights."
        res = self.extractor.extract(prompt)

        self.assertEqual(res.action_verb, "explain")
        self.assertEqual(res.task_confidence, "high")
        self.assertIn("backpropagation", [t.lower() for t in res.technologies])

    def test_clear_question_task(self):
        prompt = "What is the time complexity of quicksort in the worst-case scenario?"
        res = self.extractor.extract(prompt)

        self.assertIsNotNone(res.main_task)
        self.assertEqual(res.task_confidence, "high")
        self.assertIn("quicksort", [t.lower() for t in res.technologies])
        self.assertIn("time complexity", [t.lower() for t in res.technologies])
        self.assertIsNotNone(res.subject_matter)

    def test_coordinated_multiple_actions(self):
        prompt = "Write a Python script to parse access logs and analyze the top IP addresses."
        res = self.extractor.extract(prompt)

        self.assertIsNotNone(res.main_task)
        self.assertEqual(res.action_verb, "write")
        self.assertEqual(res.task_type, "code_generation")
        self.assertEqual(res.task_confidence, "high")
        self.assertIn("python", [t.lower() for t in res.technologies])
        self.assertGreaterEqual(len(res.details.get("detected_instructions", [])), 1)

    # 3. Multiple Entities and Proper Nouns
    def test_multiple_entities_and_acronyms(self):
        prompt = "Compare the architectural differences between AWS, Azure, and Google Cloud Platform for Docker hosting."
        res = self.extractor.extract(prompt)

        self.assertEqual(res.action_verb, "compare")
        self.assertEqual(res.task_type, "comparison")
        tech_and_ent = [item.lower() for item in res.key_entities_and_technologies]
        self.assertIn("aws", tech_and_ent)
        self.assertIn("azure", tech_and_ent)
        self.assertIn("docker", tech_and_ent)

    def test_proper_nouns_and_quoted_entities(self):
        prompt = "Summarize the historical debate between Newton and Leibniz regarding calculus in 'Principia Mathematica'."
        res = self.extractor.extract(prompt)

        self.assertEqual(res.action_verb, "summarize")
        ent_lower = [e.lower() for e in res.entities]
        self.assertTrue(any("newton" in e for e in ent_lower))
        self.assertTrue(any("leibniz" in e for e in ent_lower))
        self.assertTrue(any("principia mathematica" in e for e in ent_lower))

    # 4. Technical Terms and Code Identifiers
    def test_programming_languages_and_tools(self):
        prompt = "Write a Python script using pandas and scikit-learn to train a linear regression model."
        res = self.extractor.extract(prompt)

        self.assertEqual(res.action_verb, "write")
        self.assertEqual(res.task_type, "code_generation")
        tech_lower = [t.lower() for t in res.technologies]
        self.assertIn("python", tech_lower)
        self.assertIn("pandas", tech_lower)
        self.assertIn("scikit-learn", tech_lower)

    def test_code_identifiers_and_technical_concepts(self):
        prompt = "Debug the deadlock in process_transaction_queue when acquire_mutex() is called concurrently."
        res = self.extractor.extract(prompt)

        self.assertEqual(res.action_verb, "debug")
        self.assertEqual(res.task_type, "debugging")
        tech_lower = [t.lower() for t in res.technologies]
        self.assertIn("deadlock", tech_lower)
        self.assertTrue(any("process_transaction_queue" in t for t in tech_lower))
        self.assertTrue(any("acquire_mutex" in t for t in tech_lower))

    # 5. Missing Fields
    def test_missing_entities_and_technologies(self):
        prompt = "Explain how to bake delicious sourdough bread from scratch."
        res = self.extractor.extract(prompt)

        self.assertEqual(res.action_verb, "explain")
        self.assertEqual(res.task_type, "explanation")
        self.assertEqual(res.entities, [])
        self.assertEqual(res.technologies, [])
        self.assertEqual(res.key_entities_and_technologies, [])
        self.assertIsNotNone(res.subject_matter)
        self.assertIn("bake delicious sourdough bread", res.subject_matter)

    def test_missing_context_single_sentence(self):
        prompt = "Translate this phrase into Spanish."
        res = self.extractor.extract(prompt)

        self.assertEqual(res.action_verb, "translate")
        self.assertEqual(res.context, [])
        self.assertIsNotNone(res.subject_matter)

    # 6. Context Extraction in Multi-Sentence Prompts
    def test_multi_sentence_background_context(self):
        prompt = (
            "We operate a high-throughput microservices backend hosted on Kubernetes. "
            "Our database queries are experiencing high latency under heavy traffic. "
            "Please analyze the query bottleneck and suggest indexing strategies for PostgreSQL."
        )
        res = self.extractor.extract(prompt)

        self.assertEqual(res.action_verb, "analyze")
        self.assertGreaterEqual(len(res.context), 1)
        self.assertTrue(any("microservices" in c or "latency" in c for c in res.context))
        tech_lower = [t.lower() for t in res.technologies]
        self.assertIn("kubernetes", tech_lower)
        self.assertIn("postgresql", tech_lower)

    # 7. Ambiguous Prompts and Uncertainty Handling
    def test_ambiguous_word_list(self):
        prompt = "Blue green yellow fast slow desk chair."
        res = self.extractor.extract(prompt)

        self.assertIsNone(res.main_task)
        self.assertIsNone(res.action_verb)
        self.assertEqual(res.task_type, "uncertain")
        self.assertEqual(res.task_confidence, "uncertain")
        self.assertTrue(res.is_uncertain)

    def test_declarative_statement_without_request(self):
        prompt = "The quick brown fox jumps over the lazy dog."
        res = self.extractor.extract(prompt)

        self.assertIsNone(res.main_task)
        self.assertIsNone(res.action_verb)
        self.assertEqual(res.task_confidence, "uncertain")
        self.assertTrue(res.is_uncertain)

    def test_uncertain_prompt_does_not_invent_task(self):
        prompt = "Quantum key distribution protocols."
        res = self.extractor.extract(prompt)

        # Topic without an explicit task or action verb
        self.assertIsNone(res.main_task)
        self.assertIsNone(res.action_verb)
        self.assertEqual(res.task_confidence, "uncertain")
        self.assertTrue(res.is_uncertain)
        # Should still capture the topic subject matter
        self.assertIsNotNone(res.subject_matter)

    # 8. Analysis-Only and Compression Behavior
    def test_compress_preserves_prompt_unchanged(self):
        prompt = "Please explain the Raft consensus protocol in simple terms."
        compressed = self.extractor.compress(prompt)

        # Does not rewrite, shorten, or delete any content
        self.assertEqual(compressed, prompt)

    def test_compress_empty(self):
        self.assertEqual(self.extractor.compress(""), "")
        self.assertEqual(self.extractor.compress(None), "")

    def test_result_preserves_exact_original_prompt(self):
        prompt = "   Explain photosynthesis in plants.   "
        res = self.extractor.extract(prompt)
        self.assertEqual(res.original_prompt, prompt)

    # 9. Structure and Serialization
    def test_to_dict_keys(self):
        prompt = "Explain recursion in Python."
        res = self.extractor.extract(prompt)
        d = res.to_dict()

        expected_keys = {
            "original_prompt",
            "main_task",
            "action_verb",
            "task_type",
            "task_confidence",
            "entities",
            "technologies",
            "key_entities_and_technologies",
            "context",
            "subject_matter",
            "context_confidence",
            "is_uncertain",
            "details",
        }
        self.assertTrue(expected_keys.issubset(set(d.keys())))


if __name__ == "__main__":
    unittest.main()
