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
		api_evals_master.csv                    # Vertical dashboard-ready API benchmark runs
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
cp_eval_benchmark.py                 			# Manifest-driven generation + cross-judge benchmark workflow
cp_build_eval_master.py              			# Retrospective master CSV builder from raw result JSONL
cp_judge_calibration.py                         # Streamlit lab for blinded human judge calibration
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
| `--skip-comparison-csv` | all | Skip the legacy transposed CSV update; used by the batch workflow. |
| `--results-jsonl` | all | Override raw per-run JSONL results filename. |
| `--output-dataset` | generate-* | Override generated dataset filename (stored in mode directory when relative). |
| `--overwrite` | all | Overwrite existing column if `answers-label` already present; else a numeric suffix is auto-added. |
| `--no-progress` | all | Suppress progress ticks during evaluation. |

### Complete English API Benchmark Round

The manifest-driven runner replaces the manual sequence of generation commands and dataset re-evaluations:

```bash
# Preview a readable plan without making API calls
python cp_eval_benchmark.py --dry-run

# Include the exact underlying cp_eval_llms.py commands when troubleshooting
python cp_eval_benchmark.py --dry-run --show-commands

# Execute the resumable plan
python cp_eval_benchmark.py
```

The default manifest is `benchmark_configs/english_api_v1_4.json`. For a future round, update its human-editable `models` list:

```json
{
  "default_judge_models": [
    "gpt-6-luna",
    "gemini-4-hydrogen",
    "claude-haiku-5-5"
  ],
  "max_parallel_operations": 4,
  "result_registry": "benchmark_configs/english_api_registry.json",
  "models": [
    {"provider": "xai", "model": "grok-4.5"},
    {
      "provider": "openai",
      "model": "gpt-5.6-luna",
      "judge_contender": true
    },
    {"provider": "google", "model": "gemini-3.6-flash"}
  ]
}
```

Every listed model generates one answer set (or resumes its saved set) and is scored independently by every model in `default_judge_models`. The English manifest uses GPT-6 Luna (OpenAI), Gemini 4 Hydrogen (Google), and Claude Haiku 5.5 (Anthropic). This replaces Gemini 3.8 Flash in the benchmark's default panel; adopting it in the dashboard requires a new scoring edition while retaining `three-family-2026-10` for historical comparisons. The runner stores separate results for each judge; panel weighting and scoring editions remain dashboard concerns. An `answers_label` is derived automatically as `{provider}-{model}-{system_prompt_label}`; specify `answers_label` on a model only when an override is needed.

`default_judge_models` must be a non-empty list of distinct model IDs. Older manifests using a scalar `default_judge_model` or `initial_judge_model` still work; a list under the singular key is also accepted. Specify only one of these fields. Other manifests may use a different panel size; the runner does not infer or automatically replace a family's judge.

Models marked `judge_contender` are also used as judges. Each contender scores the deduplicated union of every new answer set and the configured `provider_leaders`. With multiple contenders, each one runs that complete comparison independently. If a contender matches any default panel member, new-answer evaluations already assigned to the panel are not repeated. Provider leaders are saved answer sets for contender comparisons; listing a model there does not make it a judge. Contender status never promotes a model into the default panel.

`max_parallel_operations` is a concurrency ceiling, not a fixed batch size. The dependency-aware scheduler starts up to that many ready operations, gives generation work priority, and uses provider-leader evaluations to fill otherwise unused slots. Generation runs in its own subprocess with `--generate-only`, without initializing any judge client. Every panel and contender judge waits until every canonical question has either a validated answer or an explicit terminal model failure, then reads that same saved dataset in its own subprocess. A judge failure never blocks another judge. Override the manifest for one run with `--jobs`; use `--jobs 1` for fully sequential execution.

Set `"gemini_execution": "batch"` in the benchmark manifest to use the [Gemini Batch API](https://aistudio.google.com/docs/batch-api) for Google answer generation and all Gemini judges. The active English manifest enables this; older manifests default to `"sync"`. Other providers retain their existing execution path. Batch token rates are 50% of standard rates, with a target turnaround of 24 hours. The runner submits file-based JSONL batches containing only missing answers or reviews, preserves the prompts and scoring pipeline, releases local worker slots while jobs wait remotely, and polls pending jobs every 30 seconds. `--jobs` limits local submission/import workers; submitted remote batches may outnumber those workers. The dry run identifies batch execution and does not submit anything.

Batch state is stored under `gemini_batch_jobs/<artifact filename>/` next to the answer dataset or review JSONL. Keep that directory: it contains durable submission intent, remote job IDs, exact requests, and downloaded responses. Ctrl-C stops local work; it does **not** cancel submitted Google jobs or their charges. Rerunning the same command retrieves those jobs and appends only unfinished valid answers/reviews to the existing files. Matching uses hashes of the model and full request, including the question, answer, prompt, and schema, rather than response order. Imported records also retain batch IDs and provider token usage, including thinking-token usage when returned. Retryable failed or omitted responses remain incomplete and can be retried in a later run; they do not unlock judging or publish a successful operation. Generation `RECITATION` outcomes are recorded as terminal model failures under the scoring policy below.

For standalone `cp_eval_llms.py` commands, pass `--gemini-execution batch`; the CLI waits and imports the results by default. `--gemini-batch-poll-seconds` adjusts polling. `--gemini-batch-no-wait` submits/checks once and exits with code 75 while the job is pending; the benchmark uses this internally to schedule other work. Switching back to synchronous calls is blocked while that artifact has an unfinished batch, preventing duplicate paid work. Batch creation has automatic retries disabled: if the submission reply is lost, the next run searches Google's job list for the saved unique display name. If no matching job can be confirmed, it stops without creating a replacement. Only after verifying in Google AI Studio that the job was not created, rerun the affected CLI command with `--gemini-batch-retry-unconfirmed` to permit a new submission. Jobs explicitly rejected by the API, or completed with failed requests, remain retryable without that flag.

Set `"gemini_batch_sync_fallback_max": 9` to recover 1–9 failed, missing, or invalid responses from a completed batch through synchronous Gemini requests at standard pricing. The active English manifest enables this; other manifests and standalone commands default to `0` (disabled), with CLI override `--gemini-batch-sync-fallback-max 9`. Recovery works both immediately after importing a batch and on a later rerun. It reuses the exact model, prompts, schema, and scoring rules, and checkpoints each response; successful batch items are never resent. Fresh requests still use batch mode, as do retries with more failures than the configured maximum. Pending or unconfirmed remote jobs must be resolved before fallback. Each failed item gets at most one synchronous call per invocation, with the usual bounded SDK transport retries; permanent request errors or three consecutive failures stop recovery without submitting another batch in that invocation. An empty response or a blocked judge response remains incomplete if the retry also fails. Generation `RECITATION` is terminal and receives the local scoring penalty below instead of further retries. Raw fallback replies are cached beside batch state before import, and saved results identify `execution_mode: "sync_fallback"` and retain token usage.

`result_registry` is the canonical append-only list of runs included in the master CSV. It is seeded with the original 25 dashboard runs. After the selected benchmark operations finish, successful operations' validated result files are appended atomically in deterministic plan order, even when other operations fail; existing paths are deduplicated. Changing the active `models` list therefore never removes earlier master rows. `provider_leaders` remains manually maintained and is never changed automatically.

New evaluation results store in-repository `dataset` paths relative to the working repository root. When rebuilding the master CSV, legacy absolute paths containing `CP-Evals-Lab/data/` resolve to the matching file in the current checkout only if that file exists and resolves inside the repository. External or unmatched paths retain their absolute location; historical JSONL files are never rewritten by this normalization.

The current manifest defines:

- Active answer sets: Gemini 3.5 Flash-Lite, Grok 4.6, and Grok 4.7.
- Default judge panel: GPT-6 Luna, Gemini 4 Hydrogen, and Claude Haiku 5.5.
- Judge contenders: Gemini 3.5 Flash-Lite. Hydrogen remains a default panel judge and a saved provider-leader answer set.
- Provider leaders: GPT-5 Mini, GPT-5.6 Luna, GPT-6 Luna, Gemini 3 Flash, Gemini 3.5 Flash, Gemini 3.8 Flash, Gemini 4 Hydrogen, Claude Haiku 4.5, Claude Haiku 5.5, Grok 4.3, and Grok 4.5.
- English's full 500-question set (`limit: 0`).

Each operation runs in an isolated Python subprocess and writes to its own file under `data/english/api_evals/logs/`, so parallel output does not become interleaved. The parent runner shows one live tqdm bar per active worker, identifying the generator or the judge and answer set; pass `--no-live-progress` to suppress them. Failures explicitly identify `generation` or `judging`. Existing dataset and result filenames are retained, including the former first judge's results, so rerunning the same command reuses saved answers and matching valid reviews. Only unresolved answer generation blocks that answer set's judges; independent work is allowed to finish. Successful judging operations still reach the registry and master CSV, while the round exits with failure if any operation failed or was dependency-skipped. `--phase generate` still includes the full default judge panel; `--generate-only` is an option of the underlying `cp_eval_llms.py` CLI.

Benchmark operations always pass `--require-complete-answers`: unresolved missing, blank, or explicitly error-marked source answers block judging before any judge requests. Explicit terminal generation failures are accounted outcomes and receive local penalties. Standalone `cp_eval_llms.py` callers can opt into the same guard; without it, missing answers retain the existing local minimum-score penalty, which does not call a judge. Generation and judging stop immediately on SDK signature errors or permanent HTTP request failures (400, 401, 403, 404, 422), and after three consecutive other failures. The provider SDK may perform its own retries within each attempt. When generation hits this cutoff, the CLI saves successful answers and exits with failure before any further generation or judging; a later invocation resumes missing answers. Attempted evaluation results are preserved, and incomplete evaluation exits with failure. Accepting, reusing, and publishing evaluations requires at least one integer score in a recognized rubric field; empty sections and metadata alone do not count as scores, and error-marked records are excluded.

A generation response with finish reason `RECITATION` is persisted with an empty assistant answer and `generation_failure: {"reason": "RECITATION", "terminal": true}`. It is not retried on resume, including when imported from previously saved batch or synchronous fallback responses. Each judge result file receives a local `evaluation_source: "generation_failure_penalty"` record with the existing minimum score of **1/5 in every rubric field**, while only nonempty answers are sent to judges. Thus 499 answers plus one failure produce 500 scored questions, with the normal question-tag applicability rules retained. The master CSV reports `question_count: 500` and `generation_failure_count: 1`; transport/judge errors remain separate in `error_count`. Resuming reuses these penalty records, and a subsequently supplied valid answer supersedes the failure and requires an actual review. This policy applies only to answer generation: a judge returning `RECITATION`, an invalid evaluation, a network error, or an ordinary unexplained empty answer remains retryable/incomplete and never lowers the answer model's score automatically.

Synchronous Gemini generation and judging use a 120-second request timeout and up to five SDK attempts for timeouts, connection errors, and retryable HTTP responses, with exponential backoff from 5 seconds capped at 30 seconds (plus jitter within that cap). Set `CP_GEMINI_TIMEOUT_SECONDS` and `CP_GEMINI_MAX_ATTEMPTS` to positive integers to override those defaults. Permanent authentication/request errors are not retried. Batch API reads use the same bounded transport settings, while batch creation disables automatic retries to prevent duplicate paid jobs. The HTTP timeout applies to each API request, not the remote batch's execution time.

The CLI appends and flushes each generated answer and attempted review to disk before issuing the next request, so interruption preserves completed work. Resuming checks saved reviews against the current answer text, shows saved and remaining counts, and starts progress at the saved count (for example, `246/500`, with `246 reused, 254 remaining`). Fully saved stages show `500/500` with zero remaining requests. During judging, every progress label names the judge and answer model, including the first judge in a combined generation/judging operation. Work from older interrupted runs that never reached disk cannot be reconstructed from progress counters.

If interruption or a disk-full error leaves an invalid final JSONL fragment without a newline, the CLI saves that fragment beside the artifact as `<filename>.incomplete-*` before truncating it and resuming. Completed records remain byte-for-byte intact, including a valid final record missing only its newline. Recovery runs before loading datasets/reviews and before appending; a failed quarantine write leaves the source untouched. The unfinished question may need another provider request. Newline-terminated corruption is not automatically repaired.

Run one phase explicitly or override concurrency when useful:

```bash
python cp_eval_benchmark.py --jobs 4
python cp_eval_benchmark.py --jobs 1
python cp_eval_benchmark.py --phase generate
python cp_eval_benchmark.py --phase cross-judge
python cp_eval_benchmark.py --phase master
```

`--phase generate` generates/resumes the selected answer sets and scores them with the entire default panel, without contender comparisons. `--phase cross-judge` runs only contender comparisons against saved answer sets. `--phase all` does both, deduplicating answer-set/judge pairs. With the current manifest, the plan has 4 generation operations, 12 panel operations, and 28 contender operations (each contender judges the 4 active answer sets and 10 controls). These 44 operations are not 44 API calls: each judge operation covers up to 500 questions, reusing valid saved scores.

Useful workflows:

- **Evaluate a new answer model:** add it to `models`, leave `judge_contender` false or omitted, and keep the panel fixed so comparisons use the same judges.
- **Evaluate a potential judge replacement:** mark it as a contender, compare its judgments on the saved controls, and review disagreements through the calibration lab. Change the panel explicitly only after that assessment, and create a new dashboard scoring edition when judges change.
- **Compare system prompts:** use a separate manifest with a new `system_prompt_label` and distinct answer labels while keeping the same panel. Preserve old datasets to compare both prompts under identical judges.
- **Fill missing judge coverage:** select an existing answer model with its exact original `answers_label`, provider, and prompt label, and run `--phase generate`. Saved answers and completed judge results are reused; missing panel judgments are added. Never reuse a label for different answer content.

Every completed phase registers its result files and rebuilds `data/english/api_evals_master.csv` from the full registry. To rebuild the master retrospectively without calling a provider:

```bash
python cp_build_eval_master.py
```

The master CSV has one row per answer-set/judge run. `answers_label` identifies the answer set as `{provider}-{model}-{prompt}` and remains unchanged when a different judge scores it. `judge_model` and the deterministic `run_id` distinguish runs. Each rubric metric has adjacent `_mean` and `_stdev` columns; deviations are population standard deviations calculated from the same values used for the corresponding published mean. Embedded result timestamps populate `evaluated_at`.

### Blinded Human Judge Calibration

Use the Streamlit calibration lab to compare any two registered judges on their shared answer sets:

```bash
streamlit run cp_judge_calibration.py
```

The Build workspace discovers every shared numeric rubric leaf dynamically, applies the existing question-tag applicability rules, and ranks disagreement by a selectable method. The recommended composite combines normalized absolute mean delta, Pearson disagreement, and Spearman disagreement; the top two rubrics are selected automatically but remain overridable. No interfaith-specific or other rubric names are hardcoded into the selection workflow.

Each saved study mixes blinded pairwise direction trials with a smaller pointwise calibration subset. Pairwise sampling targets strict judge reversals, tie-versus-order conflicts, and random agreement controls. Pointwise sampling covers large score gaps, one-point gaps, and exact judge agreement. The Review workspace hides answer-model identities, providers, judge identities, and scores, while saving each reviewer's progress atomically. The Results workspace reports pairwise human-alignment differences with question-clustered bootstrap intervals, pointwise mean absolute error and weighted kappa, and an auditable CSV export.

By default, studies and reviewer responses are stored under `data/<language>/judge_calibration_studies/`. Study definitions record the selected evaluation context, answer sets, rubrics, sampling seed, disagreement statistics, blinded tasks, and internal judge scores so the analysis is reproducible. The answer-set panel is chosen from data the two judges have actually scored; the workflow does not require cross-judging every historical model.

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
