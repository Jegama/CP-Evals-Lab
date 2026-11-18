# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Documentation Style

- Do not hard-wrap Markdown prose to a fixed maximum column width. Keep each paragraph and list item on one logical line unless Markdown structure, a code block, or readability genuinely requires a line break.
- Preserve the user's existing documentation formatting and do not reflow unrelated prose while editing nearby content.

## Project Overview

CP Evals Lab is a multilingual Reformed Christian Q&A dataset creation and evaluation toolkit for Calvinist Parrot Ministries. It provides tools for building doctrinally faithful Q&A training datasets and evaluation frameworks for assessing LLM responses and sermon homiletics.

**Current production language:** Arabic (with early English assets)

## Commands

### Local Development Environment

Use the existing virtualenvwrapper environment for Python commands and tests in this workspace:

```bash
workon cp_evals
```

`workon` is registered only in an interactive zsh session. From a non-interactive automation shell, invoke commands through interactive zsh:

```bash
zsh -ic 'workon cp_evals && python -m pytest -q'
```

Do not assume `.venv` exists or use the repository's `venv/` directory as the active environment.

### Fresh Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Unix/macOS
pip install -r requirements.txt
cp .env.example .env  # Then fill in API keys for providers you'll use
```

### Create Training Dataset
```bash
python cp_create_dataset.py --use-api --model google/gemma-3-12b-it \
  --output data/arabic/ar_training_dataset_gemma.jsonl
```
Resume after interruption with `--resume` flag.

### Generate & Evaluate LLM Responses
```bash
# Generate with system prompt (Arabic: 100 questions, default limit works)
python cp_eval_llms.py --language arabic --mode generate-api_evals \
  --provider google --gen-model gemini-2.5-flash \
  --answers-label gemini-v1_0 --use-system-prompt

# Generate with system prompt (English: 500 questions, MUST use --limit 0)
python cp_eval_llms.py --language english --mode generate-api_evals \
  --provider google --gen-model gemini-2.5-flash \
  --answers-label gemini-v1_0 --use-system-prompt --limit 0

# Evaluate existing dataset
python cp_eval_llms.py --language arabic --mode dataset \
  --dataset data/arabic/training_datasets/ar_training_dataset_gemma.jsonl \
  --answers-label gemma-12b --judge-model gpt-5-mini
```

### Evaluate Sermons
```bash
python cp_eval_sermons.py --audio data/sermons/sermon.mp3 \
  --out-dir data/sermons_evals --label my_sermon --preacher "Name" --markdown
```
Multi-run self-consistency: add `--num-scoring-runs 3`

## Architecture

### Provider System (`parrot_ai/core.py`)
All LLM providers subclass `BaseParrotAI` and implement `generate(prompt, system=None, model=None)`:
- `LocalModelParrotAI` - Local 4-bit quantized models
- `ParrotAIHF` - HuggingFace Inference API
- `ParrotAIOpenAI`, `ParrotAITogether`, `ParrotAIGemini`, `ParrotAIGrok`, `ParrotAIClaude`

Heavy dependencies (torch, transformers) are lazily imported only in `LocalModelParrotAI.load_model()`.

### Language-Scoped Prompts (`parrot_ai/prompts/<lang>.py`)
Each language module provides: `MAIN_SYSTEM_PROMPT`, `CALVIN_SYS_PROMPT`, `reasoning_prompt`, `calvin_review_prompt`, `final_answer_prompt`. System prompts are **opt-in only** (pass `system=None` unless explicitly requested).

### Multi-Step Chains (`parrot_ai/chains.py`)
- `parrot_chain()` - Full 4-step: original → reasoning → Calvin review → final synthesis
- `simple_chain()` - Single-step with system prompt
- `comparative_chain()` - Generate with multiple system prompts

### Evaluation Engine (`parrot_ai/llm_evaluation.py` + `parrot_ai/llm_evals/`)
Modes: `dataset`, `extended`, `generate-ft_evals`, `generate-api_evals`

The main orchestrator lives in `llm_evaluation.py`. Helper functions are organized in the `llm_evals/` sub-package:
- `data_loading.py` - `load_qa_pairs()`, `load_eval_questions()`
- `arabic_heuristics.py` - Arabic purity penalty, Arabic ceiling-compression calibration
- `english_heuristics.py` - English ceiling-compression calibration (Scripture citation, theological terminology, Pastoral_Acknowledgement-based cap)
- `score_processing.py` - Generic score clamping, knockouts, weighted production score

All symbols are re-exported via `llm_evals/__init__.py` for backward compatibility.

Heuristic pipeline order (do not reorder):
1. Parse LLM response → Pydantic model
2. Clamp scores to 1-5
3. Apply Arabic purity penalty (if Arabic)
4. Clamp overall scores
5. Apply ceiling-compression calibration (English or Arabic)
6. Knockout checks
7. Final clamp

### Sermon Evaluation (`parrot_ai/sermon_evaluation.py`)
Two-step process:
1. **Step 1 - Extraction**: Parse sermon into structural components
2. **Step 2 - Scoring**: Score 1-5 across homiletical rubric

Audio support via Gemini Files API; duration penalties for <35min or >50min sermons.

## Data Conventions

### JSONL Schema
```json
{
  "messages": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "<question>"},
    {"role": "assistant", "content": "<answer>"}
  ],
  "gen_model": "provider:model_id",
  "provider": "openai|google|...",
  "timestamp": "2025-02-04T...",
  "use_system_prompt": true,
  "system_prompt_label": "v1_0"
}
```
All fields after `messages` are optional.

### Directory Structure
```
data/<language>/
  <prefix>eval_questions.txt      # Canonical questions: Arabic=100, English=500
  training_datasets/              # Source & refined training sets
    evals/                        # Evaluation artifacts for dataset mode
  ft_evals/                       # Fine-tuned model artifacts
  api_evals/                      # API model artifacts
```
Prefixes: `ar_` (Arabic), `en_` (English)

**CRITICAL**: Default `--limit` is 100. English has 500 questions, so all English evaluations require `--limit 0`.

### Resume Semantics
- Long-running provider generation and evaluation must be restartable at question granularity. Persist and flush each completed answer or review before starting the next provider request; never retain an entire unfinished batch only in memory.
- Restart must reuse validated saved work and request only unfinished questions. Progress must distinguish saved work from remaining requests. Verify both graceful interruption and forced process termination near the end of a batch (including 499 of 500 questions); interruption messages must state the actual persistence guarantees.
- `--resume` counts existing lines and continues appending
- **Never reorder, reformat, or compress JSONL content** after partial generation
- Dataset shape must remain consistent

## Adding New Languages

1. Create `parrot_ai/prompts/<lang>.py` with required prompts (must define `EVAL_SYSTEM_PROMPT` and `EVAL_INSTRUCTIONS` with behavioral anchors for all sub-criteria)
2. Create `data/<lang>/` directory with:
   - `<prefix>eval_questions.txt` (exactly 100 questions)
   - Subdirectories: `training_datasets/`, `ft_evals/`, `api_evals/`
3. Add language-specific heuristics module to `parrot_ai/llm_evals/` (e.g., `<lang>_heuristics.py` with evidence detection and `calibrate_<lang>_scores()`)
4. Wire the calibration into `EvaluationEngine.evaluate()` with a language guard
5. Update `evaluation_schemas.py` for any new rubric criteria
6. Add unit tests in `tests/test_<lang>_heuristics.py`

## Environment Variables

Only configure providers you'll use in `.env`:
```
OPENAI_API_KEY=
GEMINI_API_KEY=
HF_TOKEN=
TOGETHER_API_KEY=
XAI_API_KEY=
ANTHROPIC_API_KEY=
```

Provider CLI names: `openai`, `google`, `hf`, `together`, `xai`, `anthropic`
