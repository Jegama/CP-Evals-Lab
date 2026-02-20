# LLM Evaluation Commands

## New models

```bash
python cp_eval_llms.py --language english --mode generate-api_evals --provider google --gen-model gemini-3-flash-preview --answers-label google-gemini-3-flash-v1_0 --judge-model gpt-5-mini --use-system-prompt --system-prompt-label v1_0 --limit 0 # Generate API evals for Gemini 3 Flash with GPT-5 Mini as judge - Done
python cp_eval_llms.py --language english --mode generate-api_evals --provider anthropic --gen-model claude-haiku-4-5-20251001 --answers-label claude-haiku-4-5-v1_0 --judge-model gpt-5-mini --use-system-prompt --system-prompt-label v1_0 --limit 0 # Generate API evals for Claude Haiku with GPT-5 Mini as judge - Done
```

## Anthropic as judge

```bash
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_openai_gpt-5-mini-v1_0.jsonl --judge-model claude-haiku-4-5-20251001 --answers-label openai-gpt-5-mini-v1_0-3 --limit 0 # Evaluate GPT-5 Mini with Claude as judge - Done
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_xai_grok-4-1-fast-v1_0.jsonl --judge-model claude-haiku-4-5-20251001 --answers-label xai-grok-4-1-fast-v1_0-3 --limit 0 # Evaluate Grok with Claude as judge - Done
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_anthropic_claude-haiku-4-5-v1_0.jsonl --judge-model claude-haiku-4-5-20251001 --answers-label anthropic-claude-haiku-4-5-v1_0-3 --limit 0 # Evaluate Claude Haiku with Claude as judge - Done
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_google_google-gemini-3-flash-v1_0.jsonl --judge-model claude-haiku-4-5-20251001 --answers-label google-gemini-3-flash-v1_0-3 --limit 0 # Evaluate Gemini 3 Flash with Claude as judge - Done
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_google_gemini-2.5-flash-v1_0.jsonl --judge-model claude-haiku-4-5-20251001 --answers-label google-gemini-2-5-flash-v1_0-3 --limit 0 # Evaluate Gemini 2.5 Flash with Claude as judge - Done
```

## Recommended for full coverage: 

```bash
python cp_eval_llms.py --language english --mode generate-api_evals --provider anthropic --gen-model claude-haiku-4-5-20251001 --answers-label claude-haiku-4-5-vanilla --judge-model gpt-5-mini --system-prompt-label baseline --limit 0 # Generate API evals for Claude Haiku baseline with GPT-5 Mini as judge - Done
python cp_eval_llms.py --language english --mode generate-api_evals --provider google --gen-model gemini-3-flash-preview --answers-label google-gemini-3-flash-vanilla --judge-model gpt-5-mini --system-prompt-label baseline --limit 0 # Generate API evals for Gemini 3 Flash baseline with GPT-5 Mini as judge - Done
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_google_google-gemini-3-flash-v1_0.jsonl --judge-model gemini-2.5-flash-preview-09-2025 --answers-label google-gemini-3-flash-v1_0-2 --limit 0 # Evaluate Gemini 3 Flash with Gemini 2.5 Flash as judge - Done
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_anthropic_claude-haiku-4-5-v1_0.jsonl --judge-model gemini-2.5-flash-preview-09-2025 --answers-label anthropic-claude-haiku-4-5-v1_0-2 --limit 0 # Evaluate Claude Haiku with Gemini 2.5 Flash as judge - Done
```

## New full coverage for LLM evaluation v2:

### 1. System Prompt (v1.0) with GPT-5 Mini Judge

```bash
# Google Gemini 3 Flash
python cp_eval_llms.py --language english --mode generate-api_evals --provider google --gen-model gemini-3-flash-preview --answers-label google-gemini-3-flash-v1_0 --judge-model gpt-5-mini --use-system-prompt --system-prompt-label v1_0 --limit 0

# OpenAI GPT-5 Mini
python cp_eval_llms.py --language english --mode generate-api_evals --provider openai --gen-model gpt-5-mini --answers-label openai-gpt-5-mini-v1_0 --judge-model gpt-5-mini --use-system-prompt --system-prompt-label v1_0 --limit 0

# xAI Grok 4.1 Fast
python cp_eval_llms.py --language english --mode generate-api_evals --provider xai --gen-model grok-4-1-fast-reasoning --answers-label xai-grok-4-1-fast-v1_0 --judge-model gpt-5-mini --use-system-prompt --system-prompt-label v1_0 --limit 0

# Anthropic Claude Haiku 4.5
python cp_eval_llms.py --language english --mode generate-api_evals --provider anthropic --gen-model claude-haiku-4-5-20251001 --answers-label anthropic-claude-haiku-4-5-v1_0 --judge-model gpt-5-mini --use-system-prompt --system-prompt-label v1_0 --limit 0
```

### 2. Vanilla (Baseline) with GPT-5 Mini Judge

```bash
# Google Gemini 3 Flash
python cp_eval_llms.py --language english --mode generate-api_evals --provider google --gen-model gemini-3-flash-preview --answers-label google-gemini-3-flash-vanilla --judge-model gpt-5-mini --system-prompt-label baseline --limit 0

# OpenAI GPT-5 Mini
python cp_eval_llms.py --language english --mode generate-api_evals --provider openai --gen-model gpt-5-mini --answers-label openai-gpt-5-mini-vanilla --judge-model gpt-5-mini --system-prompt-label baseline --limit 0

# xAI Grok 4.1 Fast
python cp_eval_llms.py --language english --mode generate-api_evals --provider xai --gen-model grok-4-1-fast-reasoning --answers-label xai-grok-4-1-fast-vanilla --judge-model gpt-5-mini --system-prompt-label baseline --limit 0

# Anthropic Claude Haiku 4.5
python cp_eval_llms.py --language english --mode generate-api_evals --provider anthropic --gen-model claude-haiku-4-5-20251001 --answers-label anthropic-claude-haiku-4-5-vanilla --judge-model gpt-5-mini --system-prompt-label baseline --limit 0
```

### 3. System Prompt (v1.0) with Gemini 3 Flash Judge

```bash
# Google Gemini 3 Flash (Judged by Gemini 3)
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_google_google-gemini-3-flash-v1_0.jsonl --judge-model gemini-3-flash-preview --answers-label google-gemini-3-flash-v1_0-2 --limit 0

# OpenAI GPT-5 Mini (Judged by Gemini 3)
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_openai_openai-gpt-5-mini-v1_0.jsonl --judge-model gemini-3-flash-preview --answers-label openai-gpt-5-mini-v1_0-2 --limit 0

# xAI Grok 4.1 Fast (Judged by Gemini 3)
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_xai_xai-grok-4-1-fast-v1_0.jsonl --judge-model gemini-3-flash-preview --answers-label xai-grok-4-1-fast-v1_0-2 --limit 0

# Anthropic Claude Haiku 4.5 (Judged by Gemini 3)
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_anthropic_anthropic-claude-haiku-4-5-v1_0.jsonl --judge-model gemini-3-flash-preview --answers-label anthropic-claude-haiku-4-5-v1_0-2 --limit 0
```

### 4. System Prompt (v1.1) with GPT-5 Mini Judge

```bash

# Google Gemini 3 Flash
python cp_eval_llms.py --language english --mode generate-api_evals --provider google --gen-model gemini-3-flash-preview --answers-label google-gemini-3-flash-v1_1 --judge-model gpt-5-mini --use-system-prompt --system-prompt-label v1_1 --limit 0

# OpenAI GPT-5 Mini
python cp_eval_llms.py --language english --mode generate-api_evals --provider openai --gen-model gpt-5-mini --answers-label openai-gpt-5-mini-v1_1 --judge-model gpt-5-mini --use-system-prompt --system-prompt-label v1_1 --limit 0

# xAI Grok 4.1 Fast
python cp_eval_llms.py --language english --mode generate-api_evals --provider xai --gen-model grok-4-1-fast-reasoning --answers-label xai-grok-4-1-fast-v1_1 --judge-model gpt-5-mini --use-system-prompt --system-prompt-label v1_1 --limit 0

# Anthropic Claude Haiku 4.5
python cp_eval_llms.py --language english --mode generate-api_evals --provider anthropic --gen-model claude-haiku-4-5-20251001 --answers-label anthropic-claude-haiku-4-5-v1_1 --judge-model gpt-5-mini --use-system-prompt --system-prompt-label v1_1 --limit 0
```