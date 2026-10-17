from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "Explain machine learning.",
    "Describe what machine learning means.",
    "The weather is sunny today."
]

embeddings = model.encode(sentences)

print("Embedding shape:", embeddings.shape)
print("Model test successful.")