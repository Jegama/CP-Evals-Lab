### 1. System Prompt (v1.0)

```bash
# Google Gemini 3 Flash
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_google_google-gemini-3-flash-v1_0.jsonl --judge-model gpt-5-mini --answers-label google-gemini-3-flash-v1_0 --system-prompt-label v1_0 --limit 0

# OpenAI GPT-5 Mini
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_openai_openai-gpt-5-mini-v1_0.jsonl --judge-model gpt-5-mini --answers-label openai-gpt-5-mini-v1_0 --system-prompt-label v1_0 --limit 0

# xAI Grok 4.1 Fast
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_xai_xai-grok-4-1-fast-v1_0.jsonl --judge-model gpt-5-mini --answers-label xai-grok-4-1-fast-v1_0 --system-prompt-label v1_0 --limit 0

# Anthropic Claude Haiku 4.5
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_anthropic_anthropic-claude-haiku-4-5-v1_0.jsonl --judge-model gpt-5-mini --answers-label anthropic-claude-haiku-4-5-v1_0 --system-prompt-label v1_0 --limit 0
```

### 2. System Prompt (v1.1)

```bash
# Google Gemini 3 Flash
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_google_google-gemini-3-flash-v1_1.jsonl --judge-model gpt-5-mini --answers-label google-gemini-3-flash-v1_1 --system-prompt-label v1_1 --limit 0

# OpenAI GPT-5 Mini
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_openai_openai-gpt-5-mini-v1_1.jsonl --judge-model gpt-5-mini --answers-label openai-gpt-5-mini-v1_1 --system-prompt-label v1_1 --limit 0

# xAI Grok 4.1 Fast
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_xai_xai-grok-4-1-fast-v1_1.jsonl --judge-model gpt-5-mini --answers-label xai-grok-4-1-fast-v1_1 --system-prompt-label v1_1 --limit 0

# Anthropic Claude Haiku 4.5
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_anthropic_anthropic-claude-haiku-4-5-v1_1.jsonl --judge-model gpt-5-mini --answers-label anthropic-claude-haiku-4-5-v1_1 --system-prompt-label v1_1 --limit 0
```

### 3. System Prompt (v1.2)

```bash
# Google Gemini 3 Flash
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_google_google-gemini-3-flash-v1_2.jsonl --judge-model gpt-5-mini --answers-label google-gemini-3-flash-v1_2 --system-prompt-label v1_2 --limit 0

# OpenAI GPT-5 Mini
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_openai_openai-gpt-5-mini-v1_2.jsonl --judge-model gpt-5-mini --answers-label openai-gpt-5-mini-v1_2 --system-prompt-label v1_2 --limit 0

# xAI Grok 4.1 Fast
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_xai_xai-grok-4-1-fast-v1_2.jsonl --judge-model gpt-5-mini --answers-label xai-grok-4-1-fast-v1_2 --system-prompt-label v1_2 --limit 0

# Anthropic Claude Haiku 4.5
python cp_eval_llms.py --language english --mode dataset --dataset data/english/api_evals/generated_api_anthropic_anthropic-claude-haiku-4-5-v1_2.jsonl --judge-model gpt-5-mini --answers-label anthropic-claude-haiku-4-5-v1_2 --system-prompt-label v1_2 --limit 0
```