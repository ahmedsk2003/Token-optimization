# Project Agent Instructions

This is a research project focused on **adaptive user-prompt token compression for LLM applications**.

The system aims to reduce tokens from the user's input prompt before it is sent to a target LLM while preserving semantic meaning and important user constraints.

## Core Rules

* Make the smallest change necessary to complete the requested task.
* Read only files relevant to the current task.
* Do not scan the entire repository unless necessary.
* Do not modify unrelated files.
* Prefer existing code and abstractions before creating new ones.
* Do not rewrite entire files when a targeted change is sufficient.
* Preserve existing APIs and behavior unless explicitly asked to change them.
* Follow the existing project's coding style.
* Avoid unnecessary dependencies and complexity.
* Do not introduce architectural changes unless explicitly requested.

## Research Scope

The project focuses ONLY on **user input prompt token compression**.

Do not optimize or implement methods targeting:

* Model reasoning tokens
* Model output tokens
* KV cache
* Model weights
* Multimodal tokens
* Model routing

The primary research comparison is:

**Original User Prompt vs Compressed User Prompt**

Do not silently expand or change this research scope.

## Research Integrity

* Do not invent experimental results, measurements, metrics, datasets, or research findings.
* Do not present assumptions as measured results.
* Clearly distinguish implementation decisions, hypotheses, and measured experimental results.
* Do not change the research methodology or experimental criteria unless explicitly requested.
* Do not select a compression technique, model, metric, or experimental criterion merely because it is convenient to implement.
* If an implementation decision could materially affect the research methodology, explain the issue before making the change.

## Experiments

* Keep experiments isolated and reproducible.
* Preserve existing baselines.
* Do not overwrite previous experimental results unless explicitly requested.
* Store experimental outputs in the appropriate `data/results/` location.
* Keep raw data separate from processed data.
* Record important experimental configurations.
* Do not modify previous results to make them look better.
* Do not delete failed experiments; preserve them when they are relevant to reproducibility.

## Token Efficiency

* Avoid unnecessary tool calls, searches, and repository exploration.
* Do not repeatedly rediscover information already available in this file or relevant project files.
* Read only the files needed for the current task.
* Do not download large models or datasets unless explicitly requested or required by the current phase.
* Do not install additional dependencies unless they are justified by the current task.
* Avoid running expensive model inference or experiments unnecessarily.

## Coding

* Prefer simple, modular implementations.
* Keep research components separated so they can be independently tested and evaluated.
* Avoid premature optimization.
* Avoid introducing frameworks or abstractions that are not currently needed.
* Preserve reproducibility.
* Use clear and descriptive names for research components.
* Keep configuration separate from implementation where appropriate.

## Comments

* Do not add comments for obvious code.
* Add comments only when they explain:

  * non-obvious reasoning
  * important constraints
  * research-specific decisions
  * algorithmic trade-offs

## Testing

* Run the smallest relevant test after making a change.
* Do not run expensive experiments unnecessarily.
* Verify that existing functionality has not been broken.
* Report exactly what was tested.
* Do not claim a test passed unless it was actually run.

## Dependencies

* Do not add packages without a reason.
* Prefer existing project dependencies.
* Before adding a new dependency, check whether the required functionality can be implemented using an existing dependency.
* Do not upgrade or downgrade dependencies unnecessarily.
* Do not change the Python version unless explicitly requested.

## Model and Dataset Handling

* Do not download or load large Hugging Face models unless explicitly requested for the current phase.
* Do not assume a particular target LLM or tokenizer before it has been selected in the research plan.
* Do not substitute one model or tokenizer for another without documenting the change.
* Keep model-specific code isolated where possible.

## Default Behavior

When a task is ambiguous:

1. Inspect the relevant existing files.
2. Prefer the smallest reasonable interpretation.
3. Do not make unrelated improvements.
4. Ask for clarification if proceeding could significantly affect the research design.

## Final Response

After completing a task, report concisely:

1. What was changed.
2. Which files were modified or created.
3. What was tested.
4. Test results.
5. Any important assumptions, limitations, or unresolved issues.

Do not provide unnecessary explanations or repeat information already documented in the project.
