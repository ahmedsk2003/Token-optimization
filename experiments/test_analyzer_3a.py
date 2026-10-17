from src.analyzer import PromptAnalyzer

prompt = """Please explain in detail how machine learning works.
I want the explanation to be suitable for a beginner.
Please include examples and explain supervised and unsupervised learning.
Do not use mathematical notation."""

analyzer = PromptAnalyzer()
result = analyzer.analyze(prompt)

print(result)