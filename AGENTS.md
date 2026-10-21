# Project Agent Instructions

## Current Handoff Status — Read First

**Research objective:** Reduce user-input prompt tokens before sending prompts to a target LLM, while preserving task intent and constraints with low-overhead local processing.

### Completed Work

- **Phase 1:** Project setup and environment.
- **Phase 2:** Baseline prompt measurement and token counting.
- **Phase 3A–3D:** Prompt Analyzer implemented, including basic statistics, instruction/constraint detection, example/output-format detection, semantic similarity, redundancy scoring, and rule-based task classification.
- **Phase 4.1A:** Exact duplicate sentence removal.
- **Phase 4.1B:** Similarity-based redundancy removal with conservative safeguards.
- **Phase 4.2A:** Rule-based instruction compression.
- **Phase 4.2B:** Constraint validator implemented, including structural constraint checks and uncertainty warnings.

### Immediate Next Task

**Phase 4.3 — Key-Information Extraction.**

Implement it as an independent module. Extract task, entities, constraints, parameters, required output, and important context. Do not combine it with other compressors or implement hybrid/adaptive selection.

### Handoff Rules

- Read this section and `README.md` first.
- Inspect only files directly relevant to Phase 4.3 and their tests. Do not scan the entire repository.
- Reuse existing interfaces and components where appropriate; inspect them before creating alternatives.
- Do not recreate completed modules or repeat previous experiments.
- Do not run real-model experiments automatically. The user runs experiment scripts and reviews their output.
- Preserve the original Phase 4 order: 4.3 Key-Information Extraction → 4.4 Query Rewriting → 4.5 Semantic Compression → 4.6 Token Pruning.
- Keep responses concise and report actual test results.

**Verification note:** The user previously reported `OK` for the targeted constraint-validator tests and full test suite. Additional tests for explicit `uncertain` confidence, dropped "thoroughly", and the 70% boundary were subsequently requested. Verify whether those additional tests were completed before claiming that coverage is confirmed.

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

## Current Project State

- Phase 3 — Prompt Analyzer: implemented.
- Phase 4.1A — Exact duplicate removal: implemented.
- Phase 4.1B — Similarity-based redundancy removal: implemented with conservative safeguards.
- Next: Phase 4.2 — Instruction compression.
- Remaining: Phase 4.3 key-information extraction, Phase 4.4 query rewriting, Phase 4.5 semantic compression, and Phase 4.6 token pruning.
- Hybrid/adaptive selection and downstream benchmark evaluation are not yet implemented.

Treat this as the current handoff status. Inspect relevant code and tests before modifying anything. Do not recreate completed components or mark a phase complete without verification.

Preserve the original Phase 4.1–4.6 order. Implement each technique independently before any hybrid/adaptive integration.

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
* Create experiment scripts when required by the current task, but do not run real-model experiments automatically unless explicitly requested.
* The user runs experiments and reviews their console output.
* Do not reproduce full experiment output in the final response unless explicitly requested.
* Report experiment execution only when it actually occurred. If the user runs the experiment separately, do not imply that the agent ran it.

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

1. Implementation summary: 3–5 bullets maximum.
2. Changed/created files: List paths, adding brief descriptions only when useful.
3. Tests: Report targeted and full-suite results accurately, including failures or tests not run.
4. Important safeguards or limitations: Include only when materially relevant, with no more than 2–3 bullets.

Do not include full real-model experiment outputs, repeated sample prompts, verbose file-by-file explanations, or a separate research-observations section unless explicitly requested.

Never claim tests or experiments passed unless they were actually executed. Do not omit a relevant failure to make the result appear successful.