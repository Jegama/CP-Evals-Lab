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
python cp_eval_llms.py --language english --mode generate-api_evals --provider anthropic --gen-model claude-haiku-4-5-20251001 --answers-label claude-haiku-4-5-vanilla --judge-model gpt-5-mini --system-prompt-label baseline --limit 0 # Generate API evals for Claude Haiku baseline with GPT-5 Mini as judge - In progress
python cp_eval_llms.py --language english --mode generate-api_evals --provider google --gen-model gemini-3-flash-preview --answers-label google-gemini-3-flash-vanilla --judge-model gpt-5-mini --system-prompt-label baseline --limit 0 # Generate API evals for Gemini 3 Flash baseline with GPT-5 Mini as judge - In progress
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_google_google-gemini-3-flash-v1_0.jsonl --judge-model gemini-2.5-flash-preview-09-2025 --answers-label google-gemini-3-flash-v1_0-2 --limit 0 # Evaluate Gemini 3 Flash with Gemini 2.5 Flash as judge - In progress
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_anthropic_claude-haiku-4-5-v1_0.jsonl --judge-model gemini-2.5-flash-preview-09-2025 --answers-label anthropic-claude-haiku-4-5-v1_0-2 --limit 0 # Evaluate Claude Haiku with Gemini 2.5 Flash as judge - Done
```
