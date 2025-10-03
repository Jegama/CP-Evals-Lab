# Copilot Project Instructions

## Purpose
- Build doctrinally faithful multilingual Christian Q&A datasets and evaluations; current production language is Arabic with early English assets.

## Core Architecture
- `cp_create_dataset.py` orchestrates ParrotAI chains from `parrot_ai/chains.py` to rewrite raw QA into refined JSONL; resume mode counts existing lines—don't change record shape.
- `cp_eval_llms.py` wraps generation + evaluation via `parrot_ai.llm_evaluation.EvaluationEngine`; modes: dataset, extended, generate-ft_evals, generate-api_evals.
- `cp_eval_sermons.py` runs the two-step rubric in `parrot_ai/sermon_evaluation.py` (extraction then scoring) and can emit Markdown via `sermon_markdown.py`.
- `parrot_ai/core.py` centralizes providers (HF local 4-bit, OpenAI, Together, Gemini, Grok); new integrations must subclass `BaseParrotAI` and honor the `.generate(prompt, system, model)` contract.
- Prompts are language-scoped under `parrot_ai/prompts/<lang>.py`; keep system prompts opt-in only.

## Data & Conventions
- Training/generation JSONL lines follow `{ "messages": [{system?},{user},{assistant}], "gen_model"?, "provider"?, "timestamp"? }`; generation modes optionally prepend a single system-prompt line when `--use-system-prompt`.
- Data tree mirrors `data/<language>/` with canonical `<prefix>eval_questions.txt`, `training_datasets/`, `ft_evals/`, and `api_evals/`; Arabic files use the `ar_` prefix, English uses `en_`.
- Resume semantics depend on append-only writes; never reorder, reformat, or compress JSONL content.
- Evaluation requires exactly 100 canonical questions in dataset/generate modes; missing entries abort (dataset) or get empty answers (generate) to trigger penalties.

## Evaluation Pipeline Nuances
- Heuristics run in `parrot_ai/llm_evaluation.py`: parse responses → clamp scale → (Arabic) purity penalty → clamp overalls → knockout checks → boldness adjustments → final clamp; keep order intact.
- `BASE_CSV_ROWS` and `ARABIC_EXTRA_ROWS` dictate comparison CSV layout; add new rubric items in both schema (`evaluation_schemas.py`) and aggregation logic.
- Comparison CSV columns append unless `--overwrite`; labels default from `gen_model` or `--answers-label`, sanitized via `sanitize_filename` for artifact names.

## Provider & Prompt Rules
- Never auto-inject system prompts; pass `system=None` unless the CLI flag or caller explicitly requests it.
- Provider keys load from `.env`; only require variables for the providers in use.
- For generation, use `EvaluationEngine.generate_responses` rather than direct SDK calls to keep retries and logging consistent.

## Sermon Evaluation Workflow
- Step 1 extraction and Step 2 scoring outputs land in `data/sermons_evals/` with paired JSON files plus optional Markdown; aggregated CSV is appended per run.
- Audio mode requires Gemini Files API; ensure `--provider gemini`, `--mode audio`, and `--audio <path>` are passed together.

## Extending Languages
- New languages need a `parrot_ai/prompts/<lang>.py`, matching `MAIN_SYSTEM_PROMPT`, `CALVIN_SYS_PROMPT`, rubric text, purity heuristics (if any), and a `data/<lang>/` folder mirroring Arabic naming.
- Preserve prefix patterns (`ar_`, `en_`, etc.) so auto-resolved paths continue to work.
- Update README and comparison CSV defaults only after verifying 100-question coverage.
