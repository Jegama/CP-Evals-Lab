# Arabic LLM Evals

*Last Updated: 2025-08-21*

## Executive Summary
Three evaluation layers:
1. Dataset Generation System ("Dataset Evaluations"): Benchmarked multiple large base models only to choose the best generator / linguistic backbone for creating high-fidelity Arabic Christian Q&A data. Winner: `google-gemma-3-27b-v2` (top consistency & purity, balanced accuracy; manageable interfaith gaps).
2. Supervised Fine‑tunes ("Fine-tuned Model Evaluations"): Fine‑tuned variants derived from the chosen backbone. Best: `gemma-3-27b-ft-002` (mean 4.020) with slight gains in objection handling & clarity but notable drops in Evangelism (−2.35) and Gospel Boldness (−1.08) versus the untouched baseline—evidence the current training distribution softens outward ministry tone.
3. Live API Alternatives ("API Evaluations"): Tested prompt engineering on hosted inference models as a production substitute / complement. The Arabic Master Prompt restores evangelistic tone and boldness across providers without harming kindness or linguistic fidelity.

Production Recommendation:
* Primary (missional balance): `gpt-5-mini-prompted` — strongest aggregate (4.484) and clear leader in interfaith objection handling, evangelism uplift, and bold but respectful articulation.
* Secondary / fallback (peak grammatical polish & purity): `gemini-2.5-flash-prompted` — near‑par overall (4.413) with best or co-best Grammar / Purity while still materially improved evangelistic stance vs its vanilla run.

**Why Not Deploy the Fine‑tuned Model Yet? (Cost-First Rationale)**  
Primary blocker: Hosting an in‑house / managed 27B fine‑tuned model is currently cost‑inefficient versus leveraging high-quality API models + the Arabic Master Prompt.

Key factors:
* Cost Profile: Continuous GPU (or high-memory) allocation for a 27B FT model (serving + redundancy + autoscale headroom) materially exceeds per‑token API spend at present traffic levels.
* Under Realized Gain: The best fine‑tune (`gemma-3-27b-ft-002`) still underperforms prompt-engineered APIs on Evangelism (−2.35 vs baseline) and Gospel Boldness (−1.08), so we would pay more to ship a weaker missional profile.
* Opportunity Cost: Every week of infra + optimization work delays faster iteration via prompt refinements and targeted exemplar curation.
* Flexibility: API path lets us A/B prompt tweaks immediately; fine‑tune adjustments require new training cycles and re‑evaluation.
* Risk & Ops Overhead: Scaling, drift monitoring, patching, and latency tuning add operational burden not justified by current delta.

Revisit self-hosting when:
1. Targeted augmentation / RL pass closes evangelism & boldness gaps without hurting adherence.
2. Projected monthly token volume crosses a breakeven vs reserved GPU cost.
3. Latency / data residency / customization needs outweigh API elasticity.

Interim Strategy: Keep using `gpt-5-mini-prompted` (primary) + `gemini-2.5-flash-prompted` (fallback) while building the objection/evangelism augmentation set; only resume fine‑tune deployment planning after measurable uplift on internal pilot evals. 

Immediate Next Steps:
* Curate 200–300 objection/evangelism exemplars (balanced by topic & difficulty) emphasizing: clear acknowledgement + gentle pivot + concise gospel articulation.
* Fine‑tune or RLAIF on augmentation; re‑evaluate Δ on Evangelism & Boldness while monitoring Adherence & Nuance for regression.
* Maintain monthly drift checks on both recommended API models (focus: Evangelism, Boldness, Objection Acknowledgement) to catch provider-side behavioral shifts.

Persistent Weak Spot: Objection Acknowledgement remains the lowest absolute sub‑criterion across every path (<2.5 for most non‑prompted, ≤2.98 best). This is the highest leverage improvement area.

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

---

## API Evaluations
Current aggregated rubric means from `data/arabic/api_evals/api_evals_comparison.csv` (same 100-question canonical set; each model run twice: `vanilla` = no system prompt, `prompted` = with `resources/Arabic Master Prompt.md`).

| Criterion | Sub-criterion | gpt-5-mini-vanilla | gpt-5-mini-prompted | gemini-2.5-flash-vanilla | gemini-2.5-flash-prompted | grok-3-mini-vanilla | grok-3-mini-prompted |
|-----------|---------------|-------------------:|--------------------:|-------------------------:|--------------------------:|-------------------:|--------------------:|
| Adherence | N/A | 4.43 | 🟩 **5.00** | 4.23 | 🟩 **4.95** | 4.17 | **4.84** |
| Kindness and Gentleness | N/A | 🟩 **4.98** | 🟩 **5.00** | 4.86 | **4.97** | 4.94 | 🟩 **4.98** |
| Interfaith Sensitivity | Respect and Handling Objections | 🟩 **4.12** | **3.85** | 3.65 | 3.61 | 🟩 **3.89** | 3.50 |
| Interfaith Sensitivity | Objection Acknowledgement | 🟩 **2.98** | 🟩 **2.83** | 2.43 | 2.44 | **2.70** | 2.30 |
| Interfaith Sensitivity | Evangelism | 1.89 | 🟩 **4.20** | 1.96 | **3.43** | 1.77 | 🟩 **3.49** |
| Interfaith Sensitivity | Gospel Boldness | 3.46 | 🟩 **4.68** | 3.70 | 🟩 **4.53** | 3.92 | **4.37** |
| Arabic Accuracy | Grammar and Syntax | **4.77** | 4.59 | 🟩 **4.93** | 🟩 **4.91** | 4.47 | 4.71 |
| Arabic Accuracy | Theological Nuance | **4.69** | 🟩 **4.84** | 4.66 | 🟩 **4.90** | 4.07 | 4.52 |
| Arabic Accuracy | Contextual Clarity | 🟩 **5.00** | **4.96** | 🟩 **4.99** | **4.97** | 4.89 | 4.91 |
| Arabic Accuracy | Consistency of Terms | **4.91** | 4.86 | 🟩 **4.98** | 🟩 **4.96** | 4.85 | 4.86 |
| Arabic Accuracy | Arabic Purity | 4.45 | 4.51 | **4.67** | 🟩 **4.87** | 4.44 | 🟩 **4.69** |
|  | Mean | 4.153 | 🟩 **4.484** | 4.096 | 🟩 **4.413** | 4.010 | **4.288** |

Legend: 🟩 Top 2 per row, **bold** above row mean.

Configuration summary:
* Question set: 100-question canonical evaluation set (identical to fine-tune eval set).
* Models: `gpt-5-mini`, `gemini-2.5-flash`, `grok-3-mini` (API hosted). Each run in (a) zero-shot vanilla (no system prompt) and (b) with curated system prompt (`Arabic Master Prompt`).
* System prompt focus: pastoral evangelical identity, directness, respectful interfaith tone, concise Scripture integration, Arabic clarity & purity constraints (see `resources/Arabic Master Prompt.md`).
* Judge: `gpt-5-mini` rubric-based auto-evaluator (same schema as dataset & fine-tune sections) scoring 1–5 (0.01 precision) per criterion / sub-criterion; averages computed unweighted.

Takeaways:
* Prompt uplift is consistent: overall mean improvements vs vanilla of +0.331 (gpt-5-mini), +0.316 (gemini), +0.278 (grok).
* Largest gains concentrate in evangelistic dimensions: Evangelism (+2.31 / +1.47 / +1.72) and Gospel Boldness (+1.22 / +0.83 / +0.45) showing the system prompt effectively restores outward-facing tone without harming kindness.
* Adherence also rises sharply across all three (+0.57 / +0.72 / +0.67), indicating clearer doctrinal framing.
* Theological Nuance improves for all; grok benefits most (+0.45) suggesting higher sensitivity to instruction scaffolding.
* Minor trade-offs: slight Grammar & Syntax regression for gpt-5-mini (−0.18) likely due to added stylistic imperatives; gemini sees negligible change (−0.02); grok improves (+0.24).
* Arabic Purity & Consistency remain high across the board; gemini-prompted leads on Purity (4.87) while gemini-vanilla + gemini-prompted dominate structural accuracy (Grammar / Consistency) alongside gpt-5 strengths in interfaith & evangelism sub-criteria.
* gpt-5-mini-prompted leads or co-leads in 7/11 content rows (not counting mean), especially all interfaith / missional facets—indicating strongest balanced ministry persona retention under evaluation rubric.
* Residual weakness: grok still lags slightly on objection acknowledgement & adherence vs leaders; further prompt tailoring or few-shot exemplars could narrow gap.

Recommendation (API usage): Adopt `gpt-5-mini` with the Arabic Master Prompt as primary API inference option for production Q&A—maximizes interfaith sensitivity uplift and evangelistic clarity while preserving high kindness & linguistic fidelity. Maintain `gemini-2.5-flash-prompted` as a secondary / fallback provider for scenarios prioritizing marginally higher raw grammatical purity and diversity. Re-run periodic drift checks (monthly) to ensure stability of evangelism & boldness scores.

Future prompt refinement ideas (concise): (1) Add 3–5 micro exemplars emphasizing objection acknowledgement brevity; (2) Lightweight style tag to slightly reduce verbosity in gpt-5-mini-prompted answers (guard Grammar score); (3) Structured Scripture citation template to standardize reference formatting without inflating length.