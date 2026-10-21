"""Phase 4.3A Experiment: Core Task and Context Extraction.

Evaluates KeyInformationExtractor across representative prompt scenarios:
1. Clear imperative task with technical subject matter
2. Multiple named entities, acronyms, and cloud platforms
3. Multi-sentence prompt with background context and task directive
4. Factual question with technical concepts
5. Ambiguous or unstructured input showing uncertainty representation
6. Empty / whitespace prompt handling

DO NOT RUN AUTOMATICALLY: The user runs experiment scripts manually.
"""

from typing import List, Tuple

from src.compressors.key_information import (
    KeyInformationExtractor,
    get_key_information_extractor,
)


def run_experiment() -> None:
    extractor = get_key_information_extractor()

    test_scenarios: List[Tuple[str, str, str]] = [
        (
            "Clear Task with Technical Concepts",
            "Please provide an explanation of how transformer attention mechanisms work in deep learning.",
            "Demonstrates extraction of action verb ('explain'), task type, subject matter, and technical terms.",
        ),
        (
            "Multiple Named Entities and Cloud Platforms",
            "Compare AWS, Azure, and Google Cloud Platform for hosting Docker containers and Kubernetes clusters.",
            "Demonstrates identification of multiple entities, acronyms, and technologies.",
        ),
        (
            "Multi-Sentence with Background Context",
            (
                "We operate a high-throughput microservices backend hosted on AWS. "
                "Our PostgreSQL queries are experiencing high latency under heavy traffic. "
                "Please analyze the query bottleneck and suggest indexing strategies."
            ),
            "Demonstrates separation of background context sentences from the primary directive.",
        ),
        (
            "Direct Technical Question",
            "What is the worst-case time complexity of quicksort and how does it compare to mergesort?",
            "Demonstrates question detection, algorithm concepts, and subject matter isolation.",
        ),
        (
            "Prompt without Entities or Technologies",
            "Explain how to bake delicious sourdough bread from scratch in simple terms.",
            "Demonstrates handling of missing entities without inventing details.",
        ),
        (
            "Ambiguous Prompt (Uncertainty)",
            "Blue green yellow fast slow desk chair.",
            "Demonstrates conservative uncertainty flagging for non-directive inputs.",
        ),
        (
            "Declarative Topic without Request",
            "Quantum key distribution protocols and quantum entanglement.",
            "Demonstrates identification of subject matter while flagging task uncertainty.",
        ),
        (
            "Empty Input Prompt",
            "",
            "Demonstrates safe handling of zero-token and empty prompts.",
        ),
    ]

    print("=" * 80)
    print("PHASE 4.3A EXPERIMENT: CORE TASK AND CONTEXT EXTRACTION")
    print("=" * 80)
    print(f"Total Scenarios: {len(test_scenarios)}\n")

    for idx, (title, prompt, description) in enumerate(test_scenarios, 1):
        result = extractor.extract(prompt)

        print("-" * 80)
        print(f"Scenario {idx}: {title}")
        print(f"Description: {description}")
        print(f"Original Prompt: {repr(prompt)}")
        print("Extracted Key Information:")
        print(f"  - Main Task:         {result.main_task}")
        print(f"  - Action Verb:       {result.action_verb}")
        print(f"  - Task Type:         {result.task_type} (confidence: {result.task_confidence})")
        print(f"  - Entities:          {result.entities}")
        print(f"  - Technologies:      {result.technologies}")
        print(f"  - Combined Tech/Ent: {result.key_entities_and_technologies}")
        print(f"  - Context Sentences: {result.context}")
        print(f"  - Subject Matter:    {result.subject_matter} (confidence: {result.context_confidence})")
        print(f"  - Is Uncertain:      {result.is_uncertain}")
        print(f"  - Preserved Prompt:  {result.original_prompt == prompt}")
        print()

    print("=" * 80)
    print("EXPERIMENT COMPLETE: KeyInformationExtractor demonstrated successfully.")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
