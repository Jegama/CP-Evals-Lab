# Copilot Project Instructions: Multilingual Reformed Christian Q&A Toolkit

Purpose: This repo builds, generates, and evaluates doctrinally‑faithful multilingual (currently Arabic + early English) Christian Q&A datasets with a rubric‑driven evaluator.

## 1. Big Picture Architecture
- Two primary CLI entry points:
  - `cp_create_dataset.py`: Builds refined training datasets by running a multi‑step theological reasoning chain (`parrot_ai.chains.parrot_chain`) over raw Q&A sources.
  - `cp_eval_llms.py`: Unified generation + evaluation pipeline (four modes: `dataset`, `extended`, `generate-ft_evals`, `generate-api_evals`). Produces comparison CSVs + per‑run JSONL logs.
- Core library package: `parrot_ai/`
  - `core.py`: Provider abstraction (local 4‑bit HF model + API wrappers: HF Inference, OpenAI, Together, Gemini, Grok). System prompts are ONLY injected when explicitly requested (avoid silent defaults).
  - `chains.py`: Multi-step answer refinement (trusted source → two model candidates → Calvin review → final synthesis). All generations explicitly pass a system prompt when needed.
  - `llm_evaluation.py` + `evaluation_schemas.py`: Evaluation engine using OpenAI JSON schema parsing + heuristic post-processing (Arabic purity, boldness, knockouts, clamping of Overall scores).
  - `prompts/<language>.py`: Language‑scoped constants (MAIN_SYSTEM_PROMPT, CALVIN_SYS_PROMPT, reasoning & review templates, evaluation rubric prompts, heuristics lists).
- Data layout (`data/<language>/`): raw sources (`*_gotquestions.json`, `*_qa_catechism.jsonl`), canonical 100‑question eval list (`*_eval_questions.txt`), generated datasets + evaluation artifacts (training_datasets/evals, ft_evals, api_evals). Arabic extras: purity heuristics & extra rubric section.
- Flow summary:
  1. Collect raw Q&A → 2. Run chain to produce refined JSONL (messages schema) → 3. (Optionally) Fine‑tune externally → 4. Generate answers (or reuse dataset) → 5. Evaluate with rubric → 6. Aggregate means into wide CSV.

## 2. Key Conventions & Patterns
- Training / generation JSONL line structure:
  `{ "messages": [ {system?}, {user}, {assistant} ], "gen_model"?, "provider"?, "timestamp"? }`
- System prompt insertion is ALWAYS opt‑in (`--use-system-prompt` or explicit `system=` argument). Do NOT auto‑inject defaults in new code.
- Arabic evaluation mandates 100 questions (strict) in `dataset` mode; missing questions abort. Generation modes insert blank answers for missing generations (penalized deliberately).
- Column overwrite in comparison CSV requires `--overwrite`; else numeric suffix is appended. Maintain this behavior.
- Heuristic pipeline order (preserve): parse → clamp scale → (Arabic) purity penalty → clamp overalls → enforce knockouts → boldness adjustments → clamp again.
- Adding a language requires only a new `prompts/<lang>.py` plus data folder; scripts derive file names via language + prefix (`ar_`, `en_`). Keep prefix logic consistent.
- Resume logic in `cp_create_dataset.py` counts existing lines; any change to write format must not break this line-count semantic.

## 3. Developer Workflows
- Environment: `pip install -r requirements.txt`. Heavy local model deps (torch, bitsandbytes, accelerate, flash-attn) are optional & commented.
- API credentials auto-loaded from `.env` (see README). Only require vars for providers actually used.
- Dataset build (API, Arabic example):
  `python cp_create_dataset.py --language arabic --use-api --model google/gemma-3-12b-it --output data/arabic/training_datasets/ar_training_dataset_gemma.jsonl`
- Evaluation (existing dataset):
  `python cp_eval_llms.py --language arabic --mode dataset --dataset data/arabic/training_datasets/ar_training_dataset_gemma.jsonl --answers-label gemma-12b`
- Generation + evaluation (Gemini, prompted):
  `python cp_eval_llms.py --language arabic --mode generate-api_evals --provider gemini --gen-model gemini-2.5-flash --answers-label gemini-2.5-flash-prompted --use-system-prompt`
- Extended sampling uses internal random.sample; reproducibility requires seeding externally (not yet parameterized).

## 4. Extension Guidelines for AI Agents
When implementing new features:
- Reuse provider abstraction in `core.py`; do not embed raw API calls elsewhere.
- Maintain opt‑in system prompt policy (pass `system=None` to omit; never silently add MAIN_SYSTEM_PROMPT).
- For new rubric criteria: update `BASE_CSV_ROWS` / `ARABIC_EXTRA_ROWS` in `cp_eval_llms.py`, schema models in `evaluation_schemas.py`, aggregation logic, and CSV row ordering logic. Keep backward compatibility with existing CSV columns if possible.
- Preserve evaluation JSONL schema: one JSON object per line with merged meta in `append_results_jsonl`.
- Add new providers by subclassing `BaseParrotAI`, matching `.generate(prompt, system=..., model=...)` signature and environment variable gating.
- For new languages: replicate prefix pattern, supply evaluation prompt constants (EVAL_SYSTEM_PROMPT, EVAL_INSTRUCTIONS) and optional heuristics lists.

## 5. Common Pitfalls
- Forgetting `--answers-label` when dataset lacks `gen_model` → inference fails; handle gracefully or document.
- Accidentally reordering evaluation pairs: always sort using canonical question order when strict mode.
- System prompt leakage: never print or serialize internal chain template text into final answers beyond what prompts intend.
- Large local model memory spikes: ensure 4‑bit quantization config retained (do not remove BitsAndBytes usage in `LocalModelParrotAI.load_model`).

## 6. Safe Refactors
- Core improvements (logging, retries) belong in `cp_create_dataset.py` helper functions or new utilities—avoid duplicating retry logic; reuse `retry_with_backoff` if expanding.
- If modifying evaluation heuristics, keep final double clamp pattern (ensures consistency and prevents inflation).
- Adding metadata fields: prefer appending keys (non-breaking) over renaming existing ones.

## 7. Quick Reference (Files)
- CLI scripts: `cp_create_dataset.py`, `cp_eval_llms.py`
- Chains: `parrot_ai/chains.py`
- Providers: `parrot_ai/core.py`
- Evaluation engine: `parrot_ai/llm_evaluation.py`
- Evaluation schemas: `parrot_ai/evaluation_schemas.py`
- Prompts: `parrot_ai/prompts/*.py`
- Data artifacts: `data/<lang>/**`

Feedback welcome: Clarify unclear conventions or request additions before large changes.
