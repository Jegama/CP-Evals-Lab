<div align="center">
    <a href="https://www.calvinistparrotministries.org">
        <img src="./imgs/LogoForHeader.png" alt="Calvinist Parrot Ministries" height="140" />
    </a>
</div>

# Multilingual Reformed Christian Q&A & Sermon Evaluation Toolkit

* **Purpose**: Build doctrinally faithful Q&A datasets, evaluate LLM responses across providers, and assess sermon homiletics.
* **Audience**: Ministries, researchers, model fine-tuners, preachers.
* **Differentiators**: Reformed theological guardrails, multi-criteria rubric with behavioral anchors (adherence, pastoral tone, interfaith sensitivity, linguistic accuracy), ceiling-compression calibration heuristics, multilingual Q&A evaluation (Arabic + English), language-agnostic sermon audio evaluation.
* **Outputs**: Training JSONL, LLM evaluation comparison CSVs, per-run judge logs, sermon scoring reports (JSON + Markdown).

Three evaluation pipelines for Calvinist Parrot Ministries: Q&A dataset scoring, API model benchmarking, and sermon homiletics assessment.

---

## What This Repo Provides
| Area | Purpose |
|------|---------|
| Data Sources | Curated Arabic and English Christian Q&A from GotQuestions + catechism material. |
| Dataset Builder | `cp_create_dataset.py` merges sources and rewrites answers via a controllable ParrotAI chain. |
| LLM Evaluation | `cp_eval_llms.py` generates or scores model answers on a fixed 100‑question evaluation set with rubric aggregation. |
| Sermon Evaluation | `cp_eval_sermons.py` performs two‑step sermon evaluation (extraction + rubric scoring) from audio files, with aggregated CSV and optional Markdown reports. |
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
	english/	                          		# English Q&A data (500 eval questions, mirrors arabic/ layout)
	sermons                      	  			# Example sermon audio inputs (mp3, wav, m4a)
	sermons_evals                 	 			# Sermon evaluation outputs (step1/step2 JSON, aggregated CSV, Markdown)
parrot_ai/                          			# Core chain + evaluation engine components
	llm_evals/                      			# Evaluation sub-package (heuristics, score processing, data loading)
		arabic_heuristics.py        			# Arabic purity penalty + ceiling-compression calibration
		english_heuristics.py       			# English ceiling-compression calibration (Scripture, theology, Pastoral_Acknowledgement cap)
		score_processing.py         			# Generic score clamping, knockouts, weighted scoring
		data_loading.py             			# QA pair and evaluation question loaders
cp_create_dataset.py            				# Build / enrich training dataset
cp_eval_llms.py                      			# Unified generation + evaluation CLI (language + mode aware)
cp_eval_sermons.py                  			# Two‑step sermon evaluation (audio) with aggregated CSV + Markdown
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
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_google_api_key_here
HF_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_api_key_here
TOGETHER_API_KEY=your_together_api_key_here
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

## 2. Sermon Evaluation (Two‑Step) — `cp_eval_sermons.py`
This CLI evaluates sermons using a language‑agnostic, homiletics‑aware rubric in two steps:

1) Step 1: Structural extraction to JSON (Scripture intro, sermon intro, proposition, body points, conclusion, FCF, comments).
2) Step 2: Analytical scoring (1–5) with strengths, growth areas, next steps, and aggregated roll‑ups.

**Audio-only:** This tool requires audio input (mp3, wav, m4a) and uses Gemini's Files API for processing and evaluation.

**Supported provider:** `gemini` (audio processing via Files API). Default model: `gemini-2.5-flash`.

### Core Features
- **Audio Verification:** Step 2 scoring receives the audio file for vocal delivery assessment (tone, pacing, emphasis).
- **Duration Penalty:** Audio evaluations extract duration using `mutagen`. Sermons <35 minutes or >50 minutes receive an automatic penalty (up to -1.0 point) applied to `Overall_Impact`.
- **Multi-Run Scoring (Self-Consistency):** Use `--num-scoring-runs N` to run N independent scoring runs with different random seeds, then harmonize via confidence-weighted averaging. A meta-evaluator LLM synthesizes feedback from all runs, capturing consensus and noting dissent. This reduces bias and improves fairness at ~4× the base cost (e.g., 3 runs + 1 harmonization call).

### Common flags
| Flag | Meaning |
|------|---------|
| `--audio <path>` | **Required.** Path to sermon audio file (mp3, wav, m4a). |
| `--model <model_id>` | Evaluator model (default `gemini-2.5-flash`). |
| `--num-scoring-runs N` | Number of parallel Step 2 runs. Default 1 (single run). Higher values (e.g., 3) improve consistency but increase cost proportionally. |
| `--out-dir <dir>` | Output directory for artifacts (default `data/sermons_evals`). |
| `--label <name>` | Short run label (used in filenames). |
| `--md-file <path>` | Optional explicit Markdown output path; defaults to `<out-dir>/<label>.md`. |
| `--preacher <name>` | Preacher identifier (stored in aggregated CSV). |
| `--markdown` | Also render a human‑friendly Markdown report. |

Environment variables: requires `GEMINI_API_KEY`.

### Examples
Single-run evaluation:
```cmd
python cp_eval_sermons.py ^
	--audio data/sermons/Ephesians_4_7-16 - Clark.mp3 ^
	--out-dir data/sermons_evals --label eph4_7-16 --preacher "Josh" --markdown
```

Multi-run self-consistency (3 parallel runs + harmonization):
```cmd
python cp_eval_sermons.py --num-scoring-runs 3 ^
	--audio data/sermons/Ephesians_4_7-16 - Clark.mp3 ^
	--out-dir data/sermons_evals --label eph4_7-16_multi --preacher "Josh" --markdown
```

Outputs per run:
- `<label>_step1_extraction.json` — single JSON line with Step 1 structure and extracted duration.
- `<label>_step2_scoring.json` — single JSON line with Step 2 scoring + calibrated aggregates (duration penalty applied).
- `sermon_aggregated_summary.csv` — appended roll‑ups grouped by preacher, including `audio_duration_minutes`, `duration_penalty`, `num_scoring_runs` columns.
- Optional `<label>.md` — readable report with Duration Analysis and Evaluation Methodology sections (if `--markdown`).

Default artifact directory: `data/sermons_evals`.

### Cost Considerations (Multi-Run)
Multi-run scoring (`--num-scoring-runs > 1`) multiplies Step 2 costs:
- **Formula:** `(N runs × Step 2 base cost) + (1 harmonization call cost)`
- **Example:** 30-minute audio sermon, single run ~$0.15 → 3 runs + harmonization ~$0.60 (4× base)
- Harmonization call uses text-only input (extracted scores + feedback) so cost is minimal compared to Step 2 runs.

Use single run (default) for routine evaluations; reserve multi-run for high-stakes assessments or model comparison studies.

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
| `generate-api_evals` | Generate answers from hosted API models (OpenAI, Gemini, Grok, Together, HF Inference, Anthropic) then evaluate. |

### Provider & Model Handling
Pass `--provider` plus `--gen-model` for generation modes. Supported providers (current code): `openai`, `together`, `hf`, `google`, `xai`, `anthropic`.

### System Prompt
Add `--use-system-prompt` in generation modes to prepend the language's `MAIN_SYSTEM_PROMPT` (looked up in `parrot_ai.prompts.<language>`). A copy of the system prompt (if used) is also written as a single JSON object line at the top of the generated dataset for traceability.

### Rubric Structure (Aggregated Means)
Base criteria (all languages):
- Adherence (Overall)
- Kindness_and_Gentleness (Overall)
- Interfaith_Sensitivity: Respect_and_Handling_Objections, Objection_Acknowledgement, Evangelism, Gospel_Boldness

Arabic adds (when `--language arabic`):
- Arabic_Accuracy: Grammar_and_Syntax, Theological_Nuance, Contextual_Clarity, Consistency_of_Terms, Arabic_Purity

Each scored sub-criterion appears as a row in the comparison CSV. Mean scores are rounded to 2 decimals. Missing (non-scored) pairs are skipped; blank answers (e.g., missing generations) are penalized by the evaluation engine.

**Ceiling-Compression Calibration:** Both English and Arabic evaluations apply post-processing heuristics that detect observable evidence in the answer text (Scripture citations, theological terminology, pastoral signals) and cap inflated scores when evidence is absent. This prevents score clustering near 5/5 on Adherence and Kindness dimensions. See `resources/English Evaluation Framework.md` and `resources/Arabic Evaluation Framework.md` for details.

### Evaluation Question Sources
| Mode | Question Source Rules |
|------|-----------------------|
| dataset | Requires canonical `data/<lang>/<prefix>eval_questions.txt`. **Arabic has 100 questions; English has 500 questions.** Any missing question aborts with an error. |
| extended | Ignores canonical file unless you still pass `--questions-file`. Random sample drawn from the dataset itself. |
| generate-* | Loads (by default) the canonical question file with `--limit 100`. **For English (500 questions), use `--limit 0` to evaluate all questions.** If a generated dataset lacks an answer for a question, an empty answer placeholder is inserted and will score poorly. |

Canonical file auto-inferred unless you pass `--questions-file`. Prefix: `ar_` or `en_`.

**CRITICAL**: The default `--limit` is 100. Since English has 500 canonical questions, all English evaluations must explicitly pass `--limit 0` to evaluate all questions and match existing comparison data.

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
| `--limit` | all | Number of questions to evaluate (default: 100). Use `--limit 0` for all questions. **English has 500 questions and requires `--limit 0`.** |
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
REM Note: Arabic has 100 questions (default --limit works); English would need --limit 0
```

Generate & evaluate (Gemini, vanilla prompts):
```cmd
python cp_eval_llms.py --language arabic --mode generate-api_evals ^
	--provider gemini ^
	--gen-model gemini-2.5-flash ^
	--answers-label gemini-2.5-flash-vanilla
REM Note: Arabic has 100 questions (default --limit works); English would need --limit 0
```

Generate & evaluate (Gemini with system prompt):
```cmd
python cp_eval_llms.py --language arabic --mode generate-api_evals ^
	--provider gemini ^
	--gen-model gemini-2.5-flash ^
	--answers-label gemini-2.5-flash-prompted ^
	--use-system-prompt
REM Note: Arabic has 100 questions (default --limit works); English would need --limit 0
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

## English Evaluation (Active)
English evaluation is fully operational with:
* `en_eval_questions.txt` (500 canonical English questions)
* Detailed behavioral anchors (5/3/1 scoring descriptions) for all Adherence, Kindness, and Interfaith sub-criteria in `EVAL_INSTRUCTIONS`
* Ceiling-compression calibration heuristics (Scripture citation detection, theological terminology checks, pastoral signal detection)
* Full parity with Arabic evaluation pipeline (minus Arabic_Accuracy dimension)

**Reminder:** English has 500 questions; always use `--limit 0` for English evaluations.

---

## Roadmap (Language Expansion Focus)
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
For questions or support, please [reach out](contact@calvinistparrotministries.org).

# Soli Deo Gloria

**"For from Him and through Him and to Him are all things. To Him be the glory forever! Amen."**
- Romans 11:36

<div align="center">
  <a href="https://copy.church/explain/importance/">
    <img src="https://copy.church/badges/lcc_alt_pde.png" alt="Copy.church" height="100" />
  </a>
  <a href="https://sellingJesus.org/free">
    <img src="https://copy.church/badges/sj_standard_pd.png" alt="sellingJesus.org/free" height="100" />
  </a>
</div>