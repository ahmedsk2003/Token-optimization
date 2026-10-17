from src.analyzer import PromptAnalyzer

analyzer = PromptAnalyzer()

prompts = [
    "Explain machine learning.",
    '''Explain machine learning for a beginner. Include two examples.
     Use bullet points. Do not use mathematical notation.''',
    "Write a 500-word article about cybersecurity.",
    "Compare Python and JavaScript.",
    "Summarize this text in 5 bullet points and do not use technical jargon.",
]

for i, prompt in enumerate(prompts, 1):
    result = analyzer.analyze(prompt)

    print(f"\n--- Prompt {i} ---")
    print(prompt)
    print(result)