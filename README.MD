<div align="center">
	<img src="./imgs/LogoForHeader.png" alt="Calvinist Parrot" height="140" />
</div>

# Multilingual Reformed Christian Q&A Dataset & Evaluation Toolkit

* **Purpose**: Curate and refine doctrinally faithful Q&A datasets. 
* **Audience**: Ministries, researchers, model fine‑tuners. 
* **Differentiators**: Reformed theological guardrails, multi‑criteria rubric (adherence, pastoral tone, interfaith sensitivity, linguistic accuracy), multilingual roadmap.
* **Outputs**: Training JSONL, generation/eval scripts, aggregated score matrix, raw judge logs.

Building high‑quality, reformed, multilingual Q&A datasets for Calvinist Parrot Ministries.

---

## What This Repo Provides
| Area | Purpose |
|------|---------|
| Data Sources | Curated Arabic (and future English) Christian Q&A from GotQuestions + catechism material. |
| Dataset Builder | `cp_create_dataset.py` merges sources and rewrites answers via a controllable ParrotAI chain. |
| LLM Evaluation | `cp_eval_llms.py` generates or scores model answers on a fixed 100‑question evaluation set with rubric aggregation. |
| Outputs | Training JSONL (messages format), evaluation comparison CSV, per‑run JSONL judge results. |

---

## Evaluations

* **[Arabic LLM Eval](./resources/Arabic%20LLM%20Evals.md)**

---

## Directory Highlights

```
data/
	arabic/
		ar_gotquestions.json            		# Raw Arabic structured Q&A (categories + articles)
		ar_qa_catechism.jsonl           		# Additional QA pairs (messages schema)
		ar_eval_questions.txt           		# EXACTLY 100 evaluation questions (order is canonical)
		training_datasets/              		# Source & refined training sets
			ar_training_dataset_*.jsonl   		# Generated training datasets (messages schema)
			evals/                        		# Evaluation artifacts for --mode dataset
				dataset_eval_comparison.csv 	# Aggregated rubric means (dataset mode)
				eval_results_*.jsonl        	# Raw judge outputs for dataset evaluations
		ft_evals/                        		# Generation + evaluation artifacts (generate-* modes)
			generated_openai_*.jsonl      		# Auto-created datasets from OpenAI-style generation
			generated_together_*.jsonl    		# Auto-created datasets from Together generation
			ft_evals_comparison.csv       		# Aggregated rubric means (generate modes)
			eval_results_*.jsonl          		# Raw judge outputs for generate-* evaluations
		api_evals/                        		# API evaluation artifacts
			api_eval_comparison.csv        		# Aggregated rubric means (API mode)
			eval_results_*.jsonl	      		# Raw judge outputs for API evaluations
			generated_*.jsonl              		# Auto-created datasets from API generation
		english/                          		# (Future expansion: mirrors arabic/ layout)
parrot_ai/                          			# Core chain + evaluation engine components
cp_create_dataset.py            			# Build / enrich training dataset
cp_eval_llms.py                      			# Unified generation + evaluation CLI (language + mode aware)
```

---

## Data Schema (Training / Generation JSONL)
Each line: a JSON object with at least:
```jsonc
{
	"messages": [
		{"role": "system", "content": "<system prompt>"},
		{"role": "user", "content": "<question>"},
		{"role": "assistant", "content": "<answer>"}
	],
	"gen_model": "google:gemma-3-27b",          // (present when produced via cp_eval_llms.py --mode generate-*)
	"provider": "openai|together|...",          // optional
	"timestamp": "2025-08-13T12:34:56.000000"   // ISO8601
}
```

---

## Environment Setup (Windows examples)
```cmd
python -m venv .venv
".venv\Scripts\activate"
pip install --upgrade pip
pip install -r requirements.txt
```

Optional performance extras (Flash Attention) may require a compatible GPU + toolchain.
### Environment Variables (.env)

Both `cp_create_dataset.py` and `cp_eval_llms.py` can read API credentials from a `.env` file (auto‑loaded if present). See the template: `.env.example`.

Copy `.env.example` to `.env` and fill in any providers you will use:
```env
HF_TOKEN=your_huggingface_token_here
TOGETHER_API_KEY=your_together_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_google_api_key_here
XAI_API_KEY=your_xai_api_key_here
```

Only the variables needed for the chosen providers / flags must be set (e.g., `--use-api` with an HF model requires `HF_TOKEN`; using `--provider openai` requires `OPENAI_API_KEY`, etc.).

Set ad‑hoc instead of a `.env` file if preferred:

PowerShell:
```powershell
$env:HF_TOKEN="hf_xxx"
$env:OPENAI_API_KEY="sk-xxx"
```

CMD:
```cmd
set HF_TOKEN=hf_xxx
set OPENAI_API_KEY=sk-xxx
```

Unix/macOS:
```bash
export HF_TOKEN=hf_xxx
export OPENAI_API_KEY=sk-xxx
```

---

## 1. Arabic Training Dataset Creation (`cp_create_dataset.py`)
Script pipeline:
1. Load GotQuestions (`ar_gotquestions.json`).
2. Load catechism pairs (`ar_qa_catechism.jsonl`).
3. Combine, then for each pair run the ParrotAI chain to produce a refined answer using the main system prompt.
4. Emit enriched JSONL examples under `data/...`.

### Core Arguments
| Flag | Meaning |
|------|---------|
| `--model` | Base model (local or HF Inference) e.g. `google/gemma-3-12b-it`. |
| `--use-api` | Use HF Inference API instead of local loading. Requires `HF_TOKEN`. |
| `--api-provider` | Provider routing (default `nebius`). |
| `--output` | Target JSONL file. Default `data/temp_training_dataset.jsonl`. |
| `--resume` | Continue appending after existing line count. |
| `--batch-size` | Flush interval (writes every N items). |
| `--max-retries` | Retry count for transient 5xx API errors. |

### Example: API Mode (Recommended for quick start)
```cmd
python cp_create_dataset.py ^
	--use-api ^
	--model google/gemma-3-12b-it ^
	--output data/arabic/ar_training_dataset_gemma.jsonl ^
	--batch-size 10
```

### Example: Resume After Interruption
```cmd
python cp_create_dataset.py --use-api --model google/gemma-3-12b-it --output data/arabic/ar_training_dataset_gemma.jsonl --resume
```

### Local Model (GPU) Example
```cmd
python cp_create_dataset.py --model microsoft/DialoGPT-medium --output data/arabic/ar_training_dataset_local.jsonl
```

On completion, inspect a few lines:
```cmd
more data\arabic\ar_training_dataset_gemma.jsonl
```

---

## 2. Evaluation & Answer Generation (`cp_eval_llms.py`)
### Current Modes
| Mode | Purpose |
|------|---------|
| `dataset` | Evaluate a dataset against the canonical 100-question file (must contain all 100). |
| `extended` | Evaluate a random sample from an existing dataset: sample size = max(500, 10% of unique questions) capped by dataset size (no external questions file required). |
| `generate-ft_evals` | Generate answers (fine-tuned / custom models) via a provider, create a JSONL dataset, then evaluate it. |
| `generate-api_evals` | Generate answers from hosted API models (OpenAI, Gemini, Grok, Together, HF Inference) then evaluate. |

### Provider & Model Handling
Pass `--provider` plus `--gen-model` for generation modes. Supported providers (current code): `openai`, `together`, `hf`, `gemini`, `grok`.

### System Prompt
Add `--use-system-prompt` in generation modes to prepend the language's `MAIN_SYSTEM_PROMPT` (looked up in `parrot_ai.prompts.<language>`). A copy of the system prompt (if used) is also written as a single JSON object line at the top of the generated dataset for traceability.

### Rubric Structure (Aggregated Means)
Base criteria (all languages):
- Adherence (Overall)
- Kindness_and_Gentleness (Overall)
- Interfaith_Sensitivity: Respect_and_Handling_Objections, Objection_Acknowledgement, Evangelism, Gospel_Boldness

Arabic adds (when `--language arabic`):
- Arabic_Accuracy: Grammar_and_Syntax, Theological_Nuance, Contextual_Clarity, Consistency_of_Terms, Arabic_Purity

Each scored sub‑criterion appears as a row in the comparison CSV. Mean scores are rounded to 2 decimals. Missing (non‑scored) pairs are skipped; blank answers (e.g., missing generations) are penalized by the evaluation engine.

### Evaluation Question Sources
| Mode | Question Source Rules |
|------|-----------------------|
| dataset | Requires canonical `data/<lang>/<prefix>eval_questions.txt` (exactly 100 lines). Any missing question aborts with an error. |
| extended | Ignores canonical file unless you still pass `--questions-file`. Random sample drawn from the dataset itself. |
| generate-* | Loads (by default) the canonical 100-question file (limit=100). If a generated dataset lacks an answer for a question, an empty answer placeholder is inserted and will score poorly. |

Canonical file auto-inferred unless you pass `--questions-file`. Prefix: `ar_` or `en_`.

### Key Arguments (Updated)
| Flag | Applies To | Notes |
|------|------------|-------|
| `--language {arabic,english}` | all | Selects `data/<language>/` namespace (default arabic). |
| `--mode {dataset,extended,generate-ft_evals,generate-api_evals}` | all | Workflow selector. |
| `--dataset` | dataset, extended | Existing dataset JSONL (auto-resolved inside `training_datasets/` if relative). Required for `dataset`; required for `extended`. |
| `--provider` | generate-* | One of supported providers. Required for generation modes. |
| `--gen-model` | generate-* | Provider's model ID. Required for generation modes. |
| `--questions-file` | dataset, generate-* | Optional override; auto-inferred otherwise. Not required for `extended`. |
| `--answers-label` | all | Column header for scores; inferred from dataset `gen_model` (dataset mode) or defaults to `--gen-model` for generation. |
| `--judge-model` | all | Evaluation (judge) model. Default: `gpt-5-mini`. |
| `--use-system-prompt` | generate-* | Prepend & log system prompt from prompts module. |
| `--comparison-csv` | all | Override output CSV filename (stored in mode directory when relative). |
| `--results-jsonl` | all | Override raw per-run JSONL results filename. |
| `--output-dataset` | generate-* | Override generated dataset filename (stored in mode directory when relative). |
| `--overwrite` | all | Overwrite existing column if `answers-label` already present; else a numeric suffix is auto-added. |
| `--no-progress` | all | Suppress progress ticks during evaluation. |

### Default Artifact Paths
| Mode Group | Comparison CSV | Results JSONL Directory | Generated Dataset Directory |
|------------|----------------|-------------------------|----------------------------|
| dataset / extended | `data/<lang>/training_datasets/evals/dataset_eval_comparison.csv` | `data/<lang>/training_datasets/evals/` | (N/A) |
| generate-ft_evals | `data/<lang>/ft_evals/ft_evals_comparison.csv` | `data/<lang>/ft_evals/` | `data/<lang>/ft_evals/` |
| generate-api_evals | `data/<lang>/api_evals/api_evals_comparison.csv` | `data/<lang>/api_evals/` | `data/<lang>/api_evals/` |

Relative override filenames are rooted inside the appropriate directory. Absolute paths are used verbatim.

### Column Overwrite vs. Suffix Behavior
If a model/answers label already exists:
* Provide `--overwrite` to replace its column values in-place.
* Omit it to append a new column with an incremented suffix (e.g., `llama8_2`).

### Typical Arabic Workflows
Evaluate a curated training dataset (strict 100 questions):
```cmd
python cp_eval_llms.py --language arabic --mode dataset ^
	--dataset data/arabic/training_datasets/ar_training_dataset_gemma.jsonl ^
	--answers-label gemma-12b ^
	--judge-model gpt-5-mini
```

Extended evaluation (broad random sample for a large dataset):
```cmd
python cp_eval_llms.py --language arabic --mode extended ^
	--dataset data/arabic/training_datasets/ar_training_dataset_gemma.jsonl ^
	--answers-label gemma-12b-extended
```

Generate & evaluate (fine-tuned model hosted somewhere accessible via Together):
```cmd
python cp_eval_llms.py --language arabic --mode generate-ft_evals ^
	--provider together ^
	--gen-model meta-llama/Meta-Llama-3-8B-Instruct ^
	--answers-label llama3-8b-ft ^
	--use-system-prompt
```

Generate & evaluate (Gemini, vanilla prompts):
```cmd
python cp_eval_llms.py --language arabic --mode generate-api_evals ^
	--provider gemini ^
	--gen-model gemini-2.5-flash ^
	--answers-label gemini-2.5-flash-vanilla
```

Generate & evaluate (Gemini with system prompt):
```cmd
python cp_eval_llms.py --language arabic --mode generate-api_evals ^
	--provider gemini ^
	--gen-model gemini-2.5-flash ^
	--answers-label gemini-2.5-flash-prompted ^
	--use-system-prompt
```

### Viewing Aggregated Scores
```cmd
type data\arabic\training_datasets\evals\dataset_eval_comparison.csv
type data\arabic\ft_evals\ft_evals_comparison.csv
type data\arabic\api_evals\api_evals_comparison.csv
```

### Raw Results JSONL
Each evaluation appends lines to a results file named:
```
eval_results_<answers_label>__judged_by_<judge_model>.jsonl
```
Stored in the corresponding mode directory. Each line merges per-question evaluation with run metadata (dataset path, mode, provider, etc.).

### Missing Answers Handling
In generation modes, if a question from the canonical list has no generated answer, an empty string answer is inserted and penalized. This surfaces partial generations without silently dropping questions.

### Reproducibility Notes
Extended mode uses Python's `random.sample`; set `PYTHONHASHSEED` and optionally patch in a `random.seed(...)` near invocation if you require stable samples across runs.

---

## Typical End-to-End Arabic Flow
```text
1. (Optional) Build / refresh training dataset with cp_create_dataset.py
2. Fine-tune or adapt your target model externally using produced JSONL
3. Run `cp_eval_llms.py --mode dataset` (or `extended`) on the tuned model's generated dataset
4. Or let `cp_eval_llms.py` generate answers directly (`generate-*`), then score them automatically
5. Track progress across models in the comparison CSV matrices (dataset / ft / api)
```

---

## Future: English Expansion
Planned additions:
* Parallel `en_eval_questions.txt` (500 canonical English questions)
* Unified bilingual evaluation matrix
* Cross-lingual consistency scoring (e.g., doctrinal alignment across languages)

---

## Roadmap (Language Expansion Focus)
Near Term (English):
* Add `data/english/` parity: ensure `en_gotquestions.json`, `en_qa_catechism.jsonl`, and canonical `en_eval_questions.txt` (100 questions) are finalized and documented.
* Run first English baseline evaluations using existing rubric (temporary direct reuse) and note any rubric wording needing neutral language adjustments.
* Update scripts (if needed) to auto-detect `--lang en` via file path conventions.

Mid Term (Spanish):
* Introduce `data/spanish/` with `es_gotquestions.json`, `es_qa_catechism.jsonl`, `es_eval_questions.txt` (100 questions).
* Light rubric localization (tone & cultural sensitivity phrasing) while keeping scoring math identical.
* Add language tag columns in `evaluation_comparison.csv` (naming: `<label> (ar)`, `<label> (en)`, `<label> (es)`).
* Implement optional cross-language doctrinal consistency diff (start with Arabic ↔ English, then include Spanish).

Long Term / Community Help Wanted:
* Additional languages (e.g., Chinese, Hindi, French, Bengali, Portuguese, Russian, Japanese) — contributors provide: (1) legally usable Q&A sources, (2) 100-question eval set, (3) initial rubric translation notes.
* Automated doctrinal drift detection across languages.
* Optional citation grounding (Scripture + source attribution) layer.

If you'd like to help with a new language: open an issue proposing (1) language, (2) source legality/permissions, (3) draft 100 evaluation questions, (4) rubric adaptation considerations.

---

## Contributing
Internal use for now. Feel free to open issues / suggestions (naming, rubric refinement, additional criteria for English).

---

## Contact
For questions or support, please [reach out](mailto:jesusmancilla@calvinistparrotministries.org).

# Soli Deo Gloria

**"For from Him and through Him and to Him are all things. To Him be the glory forever! Amen."**
- Romans 11:36

[Copy.church](https://copy.church/explain/importance/) | [sellingJesus.org](https://sellingJesus.org/free)
---|---
![Copy.church](https://copy.church/badges/lcc_alt_pde.png) | ![sellingJesus.org/free](https://copy.church/badges/sj_standard_pd.png)

