# Adaptive User-Prompt Token Compression

## Research Project

This project investigates an adaptive system for reducing tokens in user input prompts before they are sent to a target Large Language Model (LLM).

The system analyzes a user prompt, determines whether compression is appropriate, selects one or more compression techniques, generates a compressed candidate, and locally validates the candidate for token reduction, semantic preservation, and constraint preservation.

The primary comparison is between the original user prompt and the compressed user prompt.

## Research Scope

The current research focuses specifically on:

**User input prompt token optimization**

The following are outside the current scope:

* Model reasoning tokens
* Model output tokens
* KV cache optimization
* Model weight optimization
* Multimodal tokens
* Model routing

The compression system is intended to minimize additional LLM API usage wherever possible by using local tokenizers, embeddings, rules, lightweight models, and conventional algorithms for analysis and validation.

## Project Structure


prompt-token-optimizer/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
│
├── src/
│   ├── analyzer/
│   ├── compressors/
│   ├── validators/
│   ├── selector/
│   ├── evaluation/
│   └── utils/
│
├── experiments/
├── notebooks/
├── configs/
├── requirements.txt
└── README.md


## Current Phase

**Phase 1 — Project Setup**

The current phase establishes the Python environment and project structure.

No compression algorithm, prompt analyzer, compression selector, validator, or evaluation pipeline has been implemented yet.

## Environment

Python: 3.14.4

Initial dependencies include:

* NumPy
* Pandas
* Scikit-learn
* Hugging Face Transformers
* Sentence Transformers
* tiktoken
* Matplotlib
* Seaborn

The final dependencies and model-specific libraries will be determined in later phases after the target models and compression techniques are selected.

## Research Status

The project is currently in the setup stage.

Compression techniques, target LLMs, datasets, evaluation methodology, and experimental configurations will be established in subsequent research phases.
