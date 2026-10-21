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

Prompt-token-optimizer
    ├── configs
    ├── data
    │   ├── processed
    │   ├── raw
    │   └── result
    │       └── baseline
    ├── experiments
    │   ├── __init__.py
    │   ├── test_analyzer_3a.py
    │   ├── test_analyzer_3b.py
    │   ├── test_analyzer_3c.py
    │   ├── test_analyzer_3d.py
    │   ├── test_compressor_4_1a.py
    │   ├── test_compressor_4_1b.py
    │   ├── test_compressor_4_2a.py
    │   ├── test_compressor_4_2b.py
    │   └── testing_embedding_model.py
    ├── notebooks
    ├── src
    │   ├── analyzer
    │   │   ├── __init__.py
    │   │   ├── constraint_detector.py
    │   │   ├── example_detector.py
    │   │   ├── format_detector.py
    │   │   ├── instruction_detector.py
    │   │   ├── prompt_analyzer.py
    │   │   ├── semantic_scorer.py
    │   │   ├── sentence_segmenter.py
    │   │   └── task_classifier.py
    │   ├── compressors
    │   │   ├── __init__.py
    │   │   ├── instruction.py
    │   │   └── redundancy.py
    │   ├── evaluation
    │   │   └── baseline.py
    │   ├── selector
    │   ├── utils
    │   │   └── tokenizer.py
    │   └── validators
    │       ├── __init__.py
    │       └── constraint_validator.py
    ├── tests
    │   ├── test_baseline.py
    │   ├── test_compressor_instruction.py
    │   ├── test_compressor_redundancy.py
    │   ├── test_constraint_validator.py
    │   └── test_prompt_analyzer.py
    ├── .gitignore
    ├── AGENTS.md
    ├── README.MD
    └── requirements.txt


## Current Phase

**Phase 4 — Implement Compression Techniques Individually**

### Phase 3 — Prompt Analyzer: Implemented

Features include prompt statistics, instruction and constraint detection, example/output-format detection, semantic similarity, redundancy scoring, and rule-based task classification.

### Phase 4 — Compression Techniques

- **4.1A — Exact Duplicate Removal:** Implemented.
- **4.1B — Similarity-Based Redundancy Removal:** Implemented with conservative safeguards.
- **4.2A — Instruction Compression:** Implemented using local rules.
- **4.2B — Constraint Preservation and Validation:** Implemented with structural checks and uncertainty warnings. Additional safeguard test coverage should be verified.
- **4.3 — Key-Information Extraction:** Next.
- **4.4 — Query Rewriting:** Not started.
- **4.5 — Semantic Compression:** Not started.
- **4.6 — Token Pruning:** Not started.

Each technique is developed independently. Hybrid/adaptive selection and downstream LLM benchmarking have not yet been implemented.

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

The project has completed baseline measurement, Prompt Analyzer development, exact and similarity-based redundancy removal, and initial rule-based instruction compression with constraint validation.

The next milestone is Phase 4.3 — Key-Information Extraction. The objective is to identify and represent essential prompt information, including the task, entities, constraints, parameters, required output, and important context.

Compression savings and local validation results do not establish full semantic preservation. Downstream benchmarking against uncompressed prompts remains future work.