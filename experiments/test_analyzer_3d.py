"""Phase 3D experiment: Task classification demonstration across supported categories."""

from src.analyzer import PromptAnalyzer

analyzer = PromptAnalyzer()

prompts = [
    # 1. Explanation
    "Explain how neural networks learn via backpropagation.",

    # 2. Summarization
    "Summarize the main takeaways from the attached meeting transcript.",

    # 3. Comparison
    "Compare PostgreSQL and MongoDB. What are the key differences between them?",

    # 4. Translation
    "Translate the following user manual into Spanish: 'Press power button to start.'",

    # 5. Code Generation
    "Write a Python function to compute the Fibonacci sequence using memoization.",

    # 6. Debugging
    "Fix this Python code. Why is it throwing an IndexError: list index out of range?",

    # 7. Data Analysis
    "Analyze the following sales dataset CSV and calculate the median revenue.",

    # 8. Question Answering
    "What is the capital of Japan?",

    # 9. Creative Writing
    "Write a short story about an astronaut who discovers an alien garden on the moon.",

    # 10. Other / Non-confident match
    "Hello there! I am ready to begin today's work.",

    

    # 11. Empty prompt
    "",
    
    "Do something useful",
]

for i, prompt in enumerate(prompts, 1):
    result = analyzer.analyze(prompt)
    print(f"\n--- Prompt {i} ---")
    print("Prompt:", repr(prompt))
    print("Task Type:", result["task_type"])
    print("Instruction Count:", result["instruction_count"])
    print("Constraint Count:", result["constraint_count"])
