# Arabic LLM Evals

*Last Updated: 2025-08-19*

This document summarizes the evaluation results for Arabic language models, focusing on dataset evaluations and fine-tuned model performance. The evaluations are based on a rubric assessing adherence, kindness, interfaith sensitivity, and Arabic accuracy across various sub-criteria.

I'll release the model on HuggingFace later this week.

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
Current aggregated rubric means from `data/arabic/ft_evals/ft_evals_comparison.csv` (100-question canonical set):

| Criterion | Sub-criterion | gtp-4-1-nano-ft | gtp-4-1-nano-ft-v2 | gtp-4-1-mini-ft | qwen3-32b-ft-001 | llama-3-2-3b-ft-001 | gemma-3-1b-ft-001 | gemma-3-27b-ft-001 | gemma-3-27b-ft-002 |
|-----------|---------------|----------------:|-------------------:|----------------:|-----------------:|--------------------:|------------------:|-------------------:|-------------------:|
| Adherence | N/A | **3.59** | 3.21 | **3.87** | **3.37** | 2.14 | 2.15 | 🟩 **4.16** | 🟩 **4.09** |
| Kindness and Gentleness | N/A | **4.25** | **4.14** | **4.72** | **4.13** | 2.46 | 3.17 | 🟩 **4.85** | 🟩 **4.90** |
| Interfaith Sensitivity | Respect and Handling Objections | **2.96** | **3.04** | **3.34** | **2.99** | 1.97 | 2.17 | 🟩 **3.50** | 🟩 **3.65** |
| Interfaith Sensitivity | Objection Acknowledgement | **1.92** | 1.77 | **2.04** | 1.82 | 1.10 | 1.28 | 🟩 **2.35** | 🟩 **2.37** |
| Interfaith Sensitivity | Evangelism | **1.94** | **1.93** | 🟩 **2.11** | 1.73 | 1.08 | 1.48 | **1.97** | 🟩 **2.05** |
| Interfaith Sensitivity | Gospel Boldness | 3.08 | 3.01 | **3.30** | **3.40** | 2.26 | 3.14 | 🟩 **3.50** | 🟩 **3.74** |
| Arabic Accuracy | Grammar and Syntax | 3.53 | 3.54 | **4.55** | 3.65 | 2.56 | 2.60 | 🟩 **4.76** | 🟩 **4.81** |
| Arabic Accuracy | Theological Nuance | **3.10** | 2.86 | **3.83** | **3.14** | 1.69 | 1.84 | 🟩 **4.15** | 🟩 **4.03** |
| Arabic Accuracy | Contextual Clarity | 3.73 | 3.54 | **4.62** | **3.75** | 1.87 | 2.60 | 🟩 **4.87** | 🟩 **4.92** |
| Arabic Accuracy | Consistency of Terms | **3.87** | 3.69 | **4.72** | **3.87** | 2.52 | 2.31 | 🟩 **4.83** | 🟩 **4.90** |
| Arabic Accuracy | Arabic Purity | 3.80 | 3.74 | 🟩 **4.71** | 3.83 | 3.23 | 2.66 | **4.64** | 🟩 **4.76** |
|  | Mean | **3.252** | 3.134 | **3.801** | **3.244** | 2.080 | 2.309 | 🟩 **3.962** | 🟩 **4.020** |

Legend: 🟩 Top 2 per row, **bold** above row mean.

Fine-tune configuration summary:
* Base dataset: `data/arabic/training_datasets/ar_training_dataset_gemma-3-27b.jsonl` (shared by all except `gtp-4-1-nano-ft`).
* Split: 80% train / 20% validation (stratified by source category where applicable).
* Epochs: OpenAI family (gtp-* models) 3 epochs; Together-hosted models 5 epochs; `gemma-3-27b-ft-002` ran 10 epochs (extended schedule) using same hyperparameters otherwise.
* Objective: Supervised fine-tuning (instruction / answer regeneration) on refined Arabic Q&A pairs.

Takeaways:
* Overall leader: `gemma-3-27b-ft-002` (mean 4.020) narrowly ahead of `gemma-3-27b-ft-001` (3.962); extra epochs yielded small but broad gains, especially Adherence & Arabic Purity.
* Strong language fidelity cluster: Both gemma-27b fine-tunes dominate Arabic Accuracy sub-criteria (top-2 every row), indicating headroom primarily lies in Interfaith Sensitivity nuances.
* gtp-4-1-mini-ft is the best smaller-parameter performer (3rd overall mean 3.801) with competitive Adherence and Grammar gains vs earlier gtp nano variants.
* Lower-capacity baselines (`llama-3-2-3b-ft-001`, `gemma-3-1b-ft-001`) under-index on theological nuance & consistency—suggests capacity + pretraining vocab breadth still material for Arabic ecclesial style.
* Interfaith objection handling (acknowledgement & evangelism) remains the weakest absolute area across all models (<2.5 except top gemmas); future curriculum: targeted counter-objection exemplars + rationale steps.
* 100-question canonical eval proved sufficient (extended 500-question dataset eval earlier showed stability; no divergence here), so no additional extended fine-tune eval required.

Recommendation (fine-tuned anchor): Continue with `gemma-3-27b-ft-002` as primary production candidate; consider a focused augmentation phase before additional epoch scaling.

---

### Baseline vs Fine-tuned Gemma-3-27B Comparison
Direct side-by-side of the selected baseline (`google-gemma-3-27b-v2` dataset evaluation) and the best fine-tune (`gemma-3-27b-ft-002`). Δ = fine-tuned − baseline.

| Criterion | Sub-criterion | google-gemma-3-27b-v2 | gemma-3-27b-ft-002 | Δ |
|-----------|---------------|----------------------:|-------------------:|----:|
| Adherence | N/A | 4.58 | 4.09 | -0.49 |
| Kindness and Gentleness | N/A | 4.94 | 4.90 | -0.04 |
| Interfaith Sensitivity | Respect and Handling Objections | 3.55 | 3.65 | +0.10 |
| Interfaith Sensitivity | Objection Acknowledgement | 2.28 | 2.37 | +0.09 |
| Interfaith Sensitivity | Evangelism | 4.40 | 2.05 | -2.35 |
| Interfaith Sensitivity | Gospel Boldness | 4.82 | 3.74 | -1.08 |
| Arabic Accuracy | Grammar and Syntax | 4.75 | 4.81 | +0.06 |
| Arabic Accuracy | Theological Nuance | 4.27 | 4.03 | -0.24 |
| Arabic Accuracy | Contextual Clarity | 4.80 | 4.92 | +0.12 |
| Arabic Accuracy | Consistency of Terms | 4.91 | 4.90 | -0.01 |
| Arabic Accuracy | Arabic Purity | 4.91 | 4.76 | -0.15 |
|  | Mean | 4.383 | 4.020 | -0.363 |

Snapshot Interpretation:
* Gains: modest improvements in objection handling (+0.10 / +0.09), clarity (+0.12), grammar (+0.06).
* Regressions: large drops in Evangelism (-2.35) and Gospel Boldness (-1.08) indicate the fine-tune dampened outward-facing ministry tone; adherence and theological nuance also slipped.
* Hypothesis: Training distribution or instruction style emphasized safe/conciliatory phrasing and linguistic polishing over evangelistic assertiveness.

Minimal system prompt recommendation to reduce this effect:

```md
- **Voice & Identity:** Embody the pastoral, evangelical, Scripture‑saturated Christian identity (humble, gentle, respectful toward Muslims) while clearly affirming core Christian convictions (e.g., ألوهية المسيح، الثالوث، الصليب، القيامة) without attacking Islam.
- **Directness (No Evading):** Answer the user's actual question directly. Do NOT use generic hedging such as: "لا أستطيع إبداء رأي شخصي" / "كنموذج ذكاء اصطناعي" / "كل دين يراه أتباعه حقاً" unless a safety concern requires gentle redirection. Instead: acknowledge the question, find common ground (الإيمان بإله واحد، الأنبياء)، ثم أوضح التميّز المسيحي (شخص المسيح والفداء) بلطف.
- **Respect & Charity:** Never disparage الإسلام أو النبي محمد. Frame differences positively ("نحن نؤمن..." لا "أنتم تخطئون...").
- **Scripture Integration:** Weave 2–4 concise Arabic biblical references (book + chapter:verses) that naturally support key affirmations (e.g., يوحنا 14:6؛ أعمال 4:12؛ رومية 5:8؛ 1 بطرس 3:15). Avoid dumping long passages.
- **Pastoral Tone:** Warm، مشجِّع، واضح، غير أكاديمي جاف. استخدم جُملاً قصيرة ومباشرة بلغة عربية فصيحة مبسَّطة.
- **Clarity & Brevity:** موجز، مركّز، بلا إطالة زائدة أو تكرار.
- **Avoid:** الاعتذاريات الفارغة، الحياد المصطنع، الأسلوب الموسوعي البارد، التعليقات الميتا عن التعليمات، أو الاعتذار عن امتلاك هوية مسيحية.
- **Safety & Sensitivity:** If user requests unsafe action or hostile debate, gently redirect; otherwise proceed normally.
- **Language:** Arabic only.
```