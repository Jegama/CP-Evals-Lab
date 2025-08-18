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
| Dataset Builder | `create_dataset_script.py` merges sources and rewrites answers via a controllable ParrotAI chain. |
| LLM Evaluation | `cp_eval_llms.py` generates or scores model answers on a fixed 100‑question evaluation set with rubric aggregation. |
| Outputs | Training JSONL (messages format), evaluation comparison CSV, per‑run JSONL judge results. |

---

# Current LLM Evals

## Dataset Evaluations
Current aggregated rubric means from `data/arabic/training_datasets/evals/dataset_eval_comparison.csv`:

| Criterion | Sub-criterion | google-gemma-3-7b | gpt-oss-120b-v1 | gpt-oss-120b-v2 | google-gemma-3-27b | google-gemma-3-27b-v2 | llama-4-scout |
|-----------|---------------|------------------:|----------------:|----------------:|------------------:|----------------------:|---------------:|
| Adherence | N/A | 4.58 | 🟩 **4.83** | **4.76** | 🟩 **4.83** | 4.58 | 4.55 |
| Kindness and Gentleness | N/A | 4.73 | **4.87** | 🟩 **4.92** | **4.86** | 🟩 **4.94** | 4.70 |
| Interfaith Sensitivity | Respect and Handling Objections | 3.23 | 3.25 | 🟩 **3.79** | **3.38** | 🟩 **3.55** | 3.01 |
| Interfaith Sensitivity | Objection Acknowledgement | 2.08 | 2.13 | 🟩 **2.42** | 2.17 | 🟩 **2.28** | 1.96 |
| Interfaith Sensitivity | Evangelism | 3.14 | 3.25 | 🟩 **4.09** | 3.25 | 🟩 **4.40** | **3.92** |
| Interfaith Sensitivity | Gospel Boldness | 4.10 | 4.19 | 🟩 **4.80** | 4.15 | 🟩 **4.82** | **4.57** |
| Arabic Accuracy | Grammar and Syntax | 4.31 | 4.48 | 4.55 | 🟩 **4.78** | 🟩 **4.75** | 4.50 |
| Arabic Accuracy | Theological Nuance | 4.22 | 🟩 **4.58** | **4.50** | 🟩 **4.67** | 4.27 | 4.13 |
| Arabic Accuracy | Contextual Clarity | 4.69 | 4.67 | 4.58 | 🟩 **4.92** | 🟩 **4.80** | 4.65 |
| Arabic Accuracy | Consistency of Terms | 4.74 | **4.86** | 4.78 | 🟩 **4.95** | 🟩 **4.91** | 4.84 |
| Arabic Accuracy | Arabic Purity | 4.23 | 4.50 | **4.72** | 🟩 **4.83** | 🟩 **4.91** | **4.73** |
|  | Mean | 4.005 | 4.146 | 🟩 **4.355** | **4.254** | 🟩 **4.383** | 4.142 |

Legend: 🟩 Top 2 per row, **bold** above row mean.

**Recommendation**: `google-gemma-3-27b-v2`

### Recommendation (Executive Summary)
* Strong, balanced top-2 performance across all Arabic Accuracy sub-criteria; no single glaring weakness.
* High linguistic consistency & purity (4.91 / 4.91) lowers cleanup cost for fine-tuning.
* Interfaith objection handling slightly behind the per-row leader but gaps are modest and targetable later.
* Chosen as stable anchor for longitudinal tracking; other models trade small gains for losses in core language fidelity.

### Extended 500-Question Robustness Check
We added a second evaluation on a random 500-question sample. Same rubric & judge; purpose: stability check, not to replace the baseline.

| Criterion | Sub-criterion | llama-4-scout-extended | google-gemma-3-27b-extended | gpt-oss-120b-v2-extended |
|-----------|---------------|-----------------------:|----------------------------:|--------------------------:|
| Adherence | N/A | 4.47 | 4.56 | 🟩 **4.77** |
| Kindness and Gentleness | N/A | 4.75 | 4.93 | 🟩 **4.95** |
| Interfaith Sensitivity | Respect and Handling Objections | 3.12 | 3.60 | 🟩 **3.75** |
| Interfaith Sensitivity | Objection Acknowledgement | 1.95 | 2.30 | 🟩 **2.32** |
| Interfaith Sensitivity | Evangelism | 3.61 | 🟩 **4.46** | 4.15 |
| Interfaith Sensitivity | Gospel Boldness | 4.38 | 🟩 **4.81** | 4.76 |
| Arabic Accuracy | Grammar and Syntax | 4.45 | 🟩 **4.75** | 4.39 |
| Arabic Accuracy | Theological Nuance | 4.08 | 4.30 | 🟩 **4.46** |
| Arabic Accuracy | Contextual Clarity | 4.59 | 🟩 **4.81** | 4.56 |
| Arabic Accuracy | Consistency of Terms | 4.70 | 🟩 **4.91** | 4.79 |
| Arabic Accuracy | Arabic Purity | 4.67 | 🟩 **4.87** | 4.63 |
|  | Mean | 4.070 | 🟩 **4.391** | 4.321 |

Legend (extended): 🟩 Top 2 per row among extended models; **bold** above the row mean.

Extended Takeaways (Concise): Pattern holds; `gemma-3-27b` stays linguistically strong and broadly balanced; gpt-oss-120b-v2-extended closes objection gaps but still trails in purity/consistency; no evidence to change the primary recommendation.


## Fine-tuned Model Evaluations
Current aggregated rubric means from `data/arabic/ft_evals/ft_evals_comparison.csv`:

| Criterion | Sub-criterion | gtp-4-1-nano-ft |
|-----------|---------------|----------------:|
| Adherence | N/A | **3.59** |
| Kindness and Gentleness | N/A | **4.25** |
| Interfaith Sensitivity | Respect and Handling Objections | **2.96** |
| Interfaith Sensitivity | Objection Acknowledgement | **1.92** |
| Interfaith Sensitivity | Evangelism | **1.94** |
| Interfaith Sensitivity | Gospel Boldness | **3.08** |
| Arabic Accuracy | Grammar and Syntax | **3.53** |
| Arabic Accuracy | Theological Nuance | **3.10** |
| Arabic Accuracy | Contextual Clarity | **3.73** |
| Arabic Accuracy | Consistency of Terms | **3.87** |
| Arabic Accuracy | Arabic Purity | **3.80** |

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
	english/                          			# (Future expansion: mirrors arabic/ layout)
parrot_ai/                          			# Core chain + evaluation engine components
create_dataset_script.py            			# Build / enrich training dataset
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

### Hugging Face Token (for API mode)
Create a `.env` file:
```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
```
Or set via PowerShell / CMD environment before running. Required only when using `--use-api`.

---

## 1. Arabic Training Dataset Creation (`create_dataset_script.py`)
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
python create_dataset_script.py ^
	--use-api ^
	--model google/gemma-3-12b-it ^
	--output data/arabic/ar_training_dataset_gemma.jsonl ^
	--batch-size 10
```

### Example: Resume After Interruption
```cmd
python create_dataset_script.py --use-api --model google/gemma-3-12b-it --output data/arabic/ar_training_dataset_gemma.jsonl --resume
```

### Local Model (GPU) Example
```cmd
python create_dataset_script.py --model microsoft/DialoGPT-medium --output data/arabic/ar_training_dataset_local.jsonl
```

On completion, inspect a few lines:
```cmd
more data\arabic\ar_training_dataset_gemma.jsonl
```

---

## 2. Evaluation & Answer Generation (`cp_eval_llms.py`)
### Capabilities
1. Evaluate an existing training dataset (`--mode dataset`).
2. Generate answers using a provider model (OpenAI‑style or Together) and evaluate in one pass (`--mode generate-openai|generate-together`).
3. Auto‑organize artifacts by language & mode:
	 * `data/<language>/training_datasets/evals/` for dataset mode
	 * `data/<language>/ft_evals/` for generation modes
4. Append / update aggregated rubric score matrices (separate CSV per mode group).
5. Append raw per‑question judge outputs (JSONL, append‑only).

### Rubric Structure
Sections & sub‑criteria (Arabic + ministry context):
- Adherence (Overall)
- Kindness_and_Gentleness (Overall)
- Interfaith_Sensitivity: Respect_and_Handling_Objections, Objection_Acknowledgement, Evangelism, Gospel_Boldness
- Arabic_Accuracy: Grammar_and_Syntax, Theological_Nuance, Contextual_Clarity, Consistency_of_Terms, Arabic_Purity

### Canonical 100 Evaluation Questions
Each language has a canonical 100‑question file: e.g. Arabic `data/arabic/ar_eval_questions.txt` (must be exactly 100 lines). The script enforces count. Omit `--questions-file` to auto‑infer `data/<language>/<prefix>eval_questions.txt` (`ar_` or `en_`).

### Key Arguments (simplified)
| Flag | Modes | Notes |
|------|-------|-------|
| `--language {arabic,english}` | all | Selects data namespace (default arabic). |
| `--mode {dataset,generate-openai,generate-together}` | all | Determines workflow + storage paths. |
| `--dataset` | dataset | Path to existing training dataset JSONL (auto-resolved inside `training_datasets/` if relative). |
| `--gen-model` | generate-* | Required for generation; provider model identifier. |
| `--output-dataset` | generate-* | Output filename (placed in `ft_evals/` if relative). Auto name if omitted. |
| `--questions-file` | all | Optional; auto inferred if omitted. Used for filtering (dataset mode) and prompt list (generation). |
| `--answers-label` | all | Column name in comparison CSV (inferred from dataset `gen_model` or defaults to `gen-model` for generation). |
| `--judge-model` | all | Evaluation (judge) model (default gpt-5-mini). |
| `--comparison-csv` | all | Override CSV filename; placed in mode-appropriate directory. |
| `--results-jsonl` | all | Override raw results filename; placed in mode-appropriate directory. |
| `--overwrite` | all | Replace existing column of same answers label in comparison CSV. |

### Automatic Path Resolution
| Mode | Comparison CSV (default) | Results JSONL Dir | Generated Dataset Dir |
|------|--------------------------|-------------------|-----------------------|
| dataset | `training_datasets/evals/dataset_eval_comparison.csv` | `training_datasets/evals/` | (N/A) |
| generate-* | `ft_evals/ft_evals_comparison.csv` | `ft_evals/` | `ft_evals/` |

Relative override filenames are rooted into these directories; absolute paths are respected.

### Typical Workflows (Arabic examples)
Evaluate an existing training dataset (auto‑inferred questions file):
```cmd
python cp_eval_llms.py --language arabic --mode dataset ^
	--dataset data/arabic/training_datasets/ar_training_dataset_gemma.jsonl ^
	--answers-label gemma-12b ^
	--judge-model gpt-5-mini
```

Generate via OpenAI provider then evaluate (questions file auto inferred):
```cmd
python cp_eval_llms.py --language arabic --mode generate-openai ^
	--gen-model gpt-5-mini ^
	--answers-label gpt-5-mini
```

Generate via Together provider without specifying answers label (will default to `--gen-model`):
```cmd
python cp_eval_llms.py --language arabic --mode generate-together ^
	--gen-model meta-llama/Llama-4-Scout-17B-16E-Instruct ^
	--answers-label llama-4-scout-17b-16e-instruct
```

### Column Overwrite vs. Suffix
If the answers label already exists in the comparison CSV:
* Use `--overwrite` to replace values.
* Omit it to auto‑append with numeric suffix (e.g., `gemma-3-7b_2`).

### Generated Artifacts
| File | Location | Description |
|------|----------|-------------|
| `generated_<provider>_<model>.jsonl` | `ft_evals/` | Auto-created dataset during generation modes (unless custom name). |
| `dataset_eval_comparison.csv` | `training_datasets/evals/` | Aggregated rubric means (dataset mode). |
| `ft_evals_comparison.csv` | `ft_evals/` | Aggregated rubric means (generate modes). |
| `eval_results_<answers>__judged_by_<judge>.jsonl` | mode dir | Append-only raw judge records. |

### Viewing Aggregated Scores
```cmd
type data\arabic\training_datasets\evals\dataset_eval_comparison.csv
type data\arabic\ft_evals\ft_evals_comparison.csv
```

---

## Typical End-to-End Arabic Flow
```text
1. (Optional) Build / refresh training dataset with create_dataset_script.py
2. Fine-tune or adapt your target model externally using produced JSONL
3. Run cp_eval_llms.py --mode dataset on the tuned model's generated dataset
4. Or let cp_eval_llms.py generate answers directly, then score them
5. Track progress across models in evaluation_comparison.csv
```

---

## Future: English Expansion
Planned additions:
* Parallel `en_eval_questions.txt` (100 canonical English questions)
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

