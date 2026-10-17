
from src.analyzer import PromptAnalyzer

analyzer = PromptAnalyzer()

prompts = [
    # 1. Single sentence
    "Explain machine learning.",

    # 2. Repeated information
    (
        "Machine learning allows computers to learn from data. "
        "Machine learning enables computers to learn using data."
    ),

    # 3. Unrelated information
    (
        "Machine learning helps computers learn from data. "
        "The Pacific Ocean is the largest ocean on Earth."
    ),

    # 4. Several sentences with a mixture of information
    (
        "Explain machine learning. "
        "Describe what machine learning means. "
        "Give an example of supervised learning."
    ),

    # 5. Empty prompt
    "",
]

for i, prompt in enumerate(prompts, 1):
    result = analyzer.analyze(prompt)

    print(f"\n--- Prompt {i} ---")
    print("Prompt:", repr(prompt))
    print("Semantic similarity:", result["semantic_similarity"])
    print("Redundancy score:", result["redundancy_score"])
