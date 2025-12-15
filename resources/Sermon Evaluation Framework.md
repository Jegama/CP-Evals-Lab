# **Sermon Evaluation Framework**

This framework defines a two–step process for evaluating Christian sermons (expository / Christ‑centered) in a structured, reproducible way. It is shaped by the homiletical theology of Bryan Chappell's *Christ-Centered Preaching* and stresses that faithful proclamation both explains **what the Holy Spirit intended the text to do** for its first hearers and **what Christ now does through that same truth** for His people.  

An effective sermon does more than transfer doctrinal data; it uncovers the *purpose* (divine intent) of the biblical passage and weds that purpose to the real, shared condition of the congregation. Thus evaluation gives sustained attention to whether the preacher has:

* Identified the *subject* and *purpose* of the text (what the passage is about and what it is doing).
* Articulated a clear, text‑derived **Proposition** (subject + complement) that governs everything that follows.
* Surfaced a biblically rooted, specific **Fallen Condition Focus (FCF)**—the aspect of human fallenness, limitation, rebellion, insufficiency, disordered desire, or need that the text addresses (not always an overt sin list, but the shared condition that necessitates divine grace).
* Moved listeners from **need (FCF)** to **Christ‑centered provision**, showing how the gospel—person and work of Christ applied by the Spirit—answers the passage's burden.
* Converted exposition into **transformational, grace‑powered application** (the "so what?") that is concrete, pastorally sensitive, and derived organically from the text rather than appended moralism.

### The Centrality of the FCF
Because the FCF mediates between the ancient text and contemporary hearts, a sermon is assessed on how specifically and accurately it names the human condition the passage exposes or heals. A vague "we all struggle" is inadequate; specificity sharpens gospel clarity. Evaluation asks: Is the FCF narrow enough to drive structure, yet pastorally broad enough to connect? Is it kept God‑centered (our need before *Him*) rather than human‑centered self‑improvement? Does the sermon resolve the FCF in Christ's redemptive provision instead of pragmatic advice or behavior modification?

### Purposeful Interpretation to Practical Application
Interpretation is complete only when the Spirit's intended *purpose* for the text is carried into lived obedience, worship, repentance, hope, and mission. Therefore we scrutinize whether the sermon:
1. Traces the text's redemptive logic (not merely lexical facts).
2. Distinguishes divine mandates from pastoral wisdom suggestions (clarity of authority level).
3. Grounds every substantive application in explained textual meaning.
4. Maintains a grace motive (identity in Christ fueling obedience) rather than guilt or bare willpower.

### Evaluation Pillars (Meta‑Criteria Informing Both Steps)
1. Textual Purpose & Fidelity – Does the sermon mirror the passage's own burden and trajectory?
2. FCF Precision – Is the fallen condition concrete, text‑tethered, and determinative for structure?
3. Christ‑Centered Resolution – Does the gospel (person/work of Christ) resolve the need organically?
4. Transformational Application – Are applications specific, heart + life oriented, and grace‑driven?
5. Structural Cohesion – Do proposition, points, transitions, and conclusion all coherently serve the stated purpose?

### Why a Structured Two‑Step Process?
To reduce evaluator subjectivity and enable iterative coaching, we first extract *descriptive structure* (Step 1). We then layer *analytical judgment* (Step 2) translating raw features into rubric scores and constructive coaching. This separation disciplines the evaluator to critique what was actually preached, not what could have been preached.

In short, a sermon "scores well" here when it faithfully exposes the Spirit‑inspired purpose of the text, names and addresses the true fallen condition with gospel sufficiency, and shepherds hearers toward Christ‑formed transformation with clarity, cohesion, and pastoral wisdom.

---

## Overview of the Two-Step Process

1. **Step 1 – Descriptive Extraction (Structural Analysis)**  
	 The model parses the sermon transcript and produces a rich descriptive JSON object capturing introductions, proposition, main points (with sub‑points, illustrations, applications, comments, feedback), general comments, the Fallen Condition Focus (FCF), and a confidence score.

2. **Step 2 – Analytical Scoring (Synthesis & Coaching)**  
	 Using the structured Step 1 output, the model assigns 1–5 scores across higher‑level qualitative criteria (textual fidelity, proposition clarity, FCF identification, Christ‑centeredness, application effectiveness, structure cohesion, illustration quality, pastoral tone, overall impact) and produces strengths, growth areas, and actionable next steps.

### Advanced Evaluation Methodologies

**Self-Consistency (Multi-Run Harmonization)**  
To improve reliability and reduce variance, Step 2 can be run in "Self-Consistency" mode (default: 3 runs).
1.  **Parallel Scoring:** The Step 2 analysis is run multiple times in parallel with different random seeds.
2.  **Confidence-Weighted Averaging:** Integer scores are averaged across runs, weighted by the model's self-reported `Scoring Confidence` for each run.
3.  **Feedback Harmonization (Self-Refine):** A meta-evaluator LLM synthesizes the qualitative feedback from all runs, highlighting consensus views and noting significant minority insights (biasing toward critical feedback in subjective disagreements).

**Duration Penalty (Time Management)**  
Sermons are assessed for optimal duration (target: 35–50 minutes) to encourage discipline and respect for the congregation's capacity.
*   **Short Penalty (< 35m):** 0.1 point penalty per minute under 35m (max 1.0).
*   **Long Penalty (> 50m):** ~0.067 point penalty per minute over 50m (1 point per 15m, max 1.0).
*   The penalty is deducted strictly from the `Overall_Impact` score.

---

## Scoring Scale (Step 2)
| Score | Meaning |
| ----- | ------- |
| 5 | Exemplary: reserved for truly exceptional, publishable‑quality execution—text‑anchored, pastorally effective; no substantive improvement needed. |
| 4 | Strong: solid and text‑anchored; clearly above average; only minor refinement needed (brevity, nuance, balance). |
| 3 | Adequate: this is the expected baseline for a competent sermon—present but generic/uneven; significant sharpening would help. |
| 2 | Weak: present but unclear, forced, thin, or inconsistent; recognizable yet fails to achieve its purpose. |
| 1 | Deficient: absent, inaccurate, misleading, or counter‑productive; a fundamental element is missing or flawed. |

Calibration note: Be a tough but fair grader. The goal is growth, not mere affirmation. Treat 3 as the true midpoint (competent baseline), and do not award 4s or 5s lightly.

> **Operational Rule:** Do not use `null` for missing evidence. If a sub‑criterion is truly not evident, assign **1** (Problematic) to keep aggregates stable.

---

## Step 1 Criteria & Sub‑Criteria (Descriptive Extraction Fields — **Title Case Keys**)

### 1. Scripture Introduction

* Identifies the primary biblical text(s) accurately (reference + translation clarity).
* Provides immediate textual orientation: genre, context, setting, speaker, audience.
* Frames why this text matters today (bridging ancient context to present need).
* Avoids unrelated anecdotes before grounding in Scripture.

### 2. Sermon Introduction

* Engages interest without overshadowing the text.
* Surfaces a tension / need / question that the passage resolves.
* Naturally narrows toward the Proposition and (implicitly or explicitly) the FCF.
* Avoids moralistic clichés detached from the text.

### 3. Proposition

* A single, clear, declarative summary of the sermon's message (subject + complement).
* Text‑derived (not imposed).
* Christ‑centered orientation preferred where text warrants.  
* If implied, note deficiency with specificity.  
* **Canonical placeholder when absent:** "No explicit proposition stated".
* Evaluate precision (vagueness, over‑complexity, multiple competing propositions).

### 4. Body (Main Points Collection)

For each main point:

* **Point** – Stated as an imperative, indicative, or doctrinal truth; clearly anchored to a discrete textual segment (verse(s) indicated).
* **Summary** – Expanded explanation faithful to context (literary + redemptive).
* **Subpoints** – Coherent logical development; if absent, note consciously.
* **Illustrations** – Relevant, accurate, and in service of the point—not entertainment; **Scripture quotations are NOT illustrations**.
* **Application** – Specific, grace‑motivated, heart + life oriented (not generic moralism).
* **Comments** – Evaluate exegesis fidelity, clarity, progression toward climax, over‑proof‑texting risks, handling of original audience.
* **Feedback** – Constructive, actionable coaching (what to refine, add, trim, rephrase, or reorder).

### 5. Conclusion

* Capture the conclusion in a single paragraph that mirrors the preacher's actual landing—include the stated recap, exhortation, gospel emphasis, and tone as delivered (do not invent missing elements).
* Highlight whether the conclusion provided a compelling exhortation, climax, and pointed ending within that paragraph; if certain elements were absent, say so plainly inside the paragraph rather than fabricating them.
* **Canonical placeholder when absent:** If the preacher offered no discernible conclusion, set the field to "No explicit conclusion provided".

### 6. General Comments

* **Content Comments** – Doctrinal substance? Faithful synthesis? Christ and Gospel explicit where warranted?
* **Structure Comments** – Logical flow, unity, escalation, transitions, balance of explanation vs. application.
* **Explanation Comments** – Depth of exegesis, context (historical, literary), handling of difficult phrases, theological integration.

### 7. Fallen Condition Focus (FCF)

* **FCF** – The shared human brokenness, limitation, or need (not always explicit sin) addressed by the text. Specific and text‑rooted.
* **Comments** – Distinguish between surface problem and deeper gospel issue; confirm alignment with main points and applications; guard against purely behavioral framing; note if FCF is missing, too broad, or misaligned.

### 8. Extraction Confidence

* A floating value (0–1) reflecting internal model confidence in extraction accuracy.  
* Should consider transcript completeness, clarity, audio artifacts (if hinted), structural ambiguity, or missing proposition.

---

## Step 2 Analytical Rubric (Detailed Quantitative Scoring)

Step 2 converts the rich descriptive output (Step 1) into granular rubric‑based scores (1–5) plus qualitative feedback per category. Every score MUST be an integer 1–5 (5 = absolutely yes / exemplary; 1 = absolutely no / absent / severely deficient). After per‑criterion scoring, an aggregated summary may be produced (see Aggregated Summary section).

### A. Introduction

Sub‑Criteria:

1. **FCF Introduced** *(a specific fallen condition is derived from the preached text and previewed)*
2. **Arouses Attention** *(opens with text‑relevant tension/need rather than unrelated anecdotes)*
Feedback: Holistic, actionable coaching (affirm + improve).

### B. Proposition

Sub‑Criteria:

1. **Principle + Application Wed** *(Subject + complement form a single gospel principle with an implied or explicit response.)*
2. **Establishes Main Theme** *(Controls scope & governs all points; no competing propositions.)*
3. **Summarizes Introduction** *(Carries forward the tension/need terms raised earlier.)*
Feedback: Strengths + surgical improvements.

### C. Main Points

Sub‑Criteria:

1. **Clarity** *(Succinct, memorable phrasing—typically ≤12 words.)*
2. **Hortatory Universal Truths** *(States timeless truths that call hearers to trust/obey—**not** mere narrative recap)*
3. **Proportional & Coexistent** *(balanced coverage across points; each point meaningfully advances the single proposition; no orphan points; points logically parallel, not redundant)*
4. **Exposition Quality** *(Explains text meaning in context before application.)*
5. **Illustration Quality** *(Illustrations illuminate the stated point & remain proportionate.)*
6. **Application Quality** *(Specific, grace‑motivated, heart + life oriented.)*
Feedback: Cohesion, pacing, balance suggestions.

#### Hortatory Universal Truths – Boundary Examples

Definition: A main point that expresses a timeless, text‑derived principle/doctrinal assertion or imperative implication rather than a mere chronological or descriptive recap.

Examples:
* PASS: "God's mercy transforms our identity" (Principial, transferable.)
* PASS: "Because Christ reigns, believers resist despair" (Doctrinal + implied exhortation.)
* FAIL: "Paul moves to verse 3 where he talks about wrath" (Narrative recap only.)
* FAIL: "Verses 4–7 are about grace" (Label without hortatory force or principle.)

Scoring Heuristics:
* 5 – All points principial & action‑orienting or doctrinally robust; none are mere captions.
* 3 – Mixed: at least one point drifts into recap/caption.
* 1 – Majority are narrative descriptions with no transferable principle.


### D. Exegetical Support

Sub‑Criteria:

1. **Alignment with Text** *(Structure & emphasis mirror the passage's burden.)*
2. **Handles Difficulties** *(Engages key interpretive/translation/theological tensions honestly.)*
3. **Proof Accuracy & Clarity** *(Supports claims with sound, digestible reasoning.)*
4. **Context & Genre Considered** *(Honors literary, historical, redemptive context.)*
5. **Not Belabored** *(Stops proving once sufficient; avoids pedantic overload.)*
6. **Aids Rather Than Impresses** *(Content serves listener understanding, not scholar display.)*
Feedback: Depth vs brevity, clarity, balance.

### E. Application

Sub‑Criteria:

1. **Clear & Practical** *(Concrete next steps or heart postures identifiable.)*
2. **Redemptive Focus** *(Motivated by Christ's person/work & grace, not bare willpower.)*
3. **Mandate vs Idea Distinction** *(Explicitly marks divine commands vs pastoral wisdom suggestions.)*
4. **Passage Supported** *(Flows organically from explained meaning; no bolt‑ons.)*
Feedback: Sharpen, contextualize, motivate.

### F. Illustrations

Sub‑Criteria:

1. **Lived‑Body Detail** *(Concrete, sensory realism that builds credibility.)*
2. **Strengthens Points** *(Illumines stated truth without hijacking focus.)*
3. **Proportion** *(Length & frequency economical; avoids narrative domination.)*
Feedback: Trim / diversify / anchor to text.

### G. Conclusion

Sub‑Criteria:

1. **Summary** *(Concise recapitulation of proposition & main movements.)*
2. **Compelling Exhortation** *(Specific, gospel‑rooted call to response.)*
3. **Climax** *(Appropriate theological/pastoral crescendo, not emotional manipulation.)*
4. **Pointed End** *(Decisive landing—no meandering fade.)*
Feedback: Intensify, focus, seal.

### H. Scoring Confidence

* **Scoring Confidence** – Float 0–1 reflecting the evaluator's confidence in Step 2 scoring fidelity (quality of Step 1 data, transcript clarity, structural ambiguity, etc.).

### Scoring Guidance (Heuristic)
Be a tough but fair grader. The aim is to help the preacher improve, not only to affirm. Use the full 1–5 scale.

| Score | Descriptor | Heuristic Examples |
| ----- | ---------- | ------------------ |
| 5 | Exemplary | Reserved for truly exceptional, publishable‑quality execution; text‑anchored and pastorally effective; no substantive improvement needed. |
| 4 | Strong | Solid and clearly above average; only minor refinement needed (brevity, nuance, balance). |
| 3 | Adequate | The expected baseline for a competent sermon; present but generic/uneven; could be significantly sharpened. |
| 2 | Weak | Present but unclear, forced, thin, or inconsistent; recognizable yet fails to achieve its purpose. |
| 1 | Deficient | Absent, inaccurate, misleading, or counter‑productive; fundamental element missing or flawed. |

Tie‑breakers:
- When in doubt, default to the lower score; challenge the sermon to earn higher marks.
- Use 3 as the true midpoint (adequate), not a soft pass.

## Aggregated Summary

Compute rolled‑up composite categories for dashboards by averaging related raw scores:
* Textual_Fidelity ≈ avg(Exegetical Support.Alignment with Text, Handles Difficulties, Proof Accuracy & Clarity, Context & Genre Considered)
* Proposition_Clarity ≈ avg(Proposition.Principle + Application Wed, Establishes Main Theme, Summarizes Introduction)
* Introduction ≈ avg(Introduction.FCF Introduced, Introduction.Arouses Attention)
* Application_Effectiveness ≈ avg(Application.Clear & Practical, Redemptive Focus, Mandate vs Idea Distinction, Passage Supported, Main Points.Application Quality)
* Structure_Cohesion ≈ avg(Main Points.Proportional & Coexistent, Conclusion.Summary, Conclusion.Compelling Exhortation, Conclusion.Climax, Conclusion.Pointed End)
* Illustrations ≈ avg(Main Points.Illustration Quality, Illustrations.Lived-Body Detail, Illustrations.Strengthens Points, Illustrations.Proportion)
* Overall_Impact_Base ≈ **weighted average** of the above six aggregates using "Pillars First" weights:
  - Textual_Fidelity: **24%**
  - Application_Effectiveness: **24%**
  - Structure_Cohesion: **20%**
  - Proposition_Clarity: **12%**
  - Illustrations: **10%**
  - Introduction: **10%**
* Overall_Impact = Overall_Impact_Base - Duration_Penalty (clamped to 1.0 minimum).

**Weighting Rationale:** The "Pillars First" scheme emphasizes the core evaluation pillars (Textual Purpose & Fidelity, Transformational Application, Structural Cohesion) which together account for 68% of the overall score. Proposition clarity remains significant at 12%, while introduction and illustrations are deliberately weighted lighter (10% each) to prevent rhetorically polished but theologically thin sermons from scoring artificially high. This ensures that faithful text handling, Christ-centered application, and structural integrity drive the Overall Impact score.

**Duration Penalty Logic:**
*   If duration < 35m: `Penalty = min((35 - minutes) / 10.0, 1.0)`
*   If duration > 50m: `Penalty = min((minutes - 50) / 15.0, 1.0)`
*   Otherwise: `Penalty = 0`

---

### Rubric Table (Step 2 – Example Format)

Each top‑level category contains its sub‑criteria as keys with spaces. Feedback resides under the same category. Confidence is a top‑level scalar.

| Category | Sub‑Criterion | Sermon A | Sermon B |
| -------- | ------------- | -------- | -------- |
| Introduction | FCF Introduced | 5 | 4 |
|  | Arouses Attention | 4 | 3 |
|  | Feedback | Strong tension; trim hook | Less focus |
| Proposition | Principle + Application Wed | 5 | 4 |
|  | Establishes Main Theme | 5 | 5 |
|  | Summarizes Introduction | 5 | 3 |
|  | Feedback | Crisp & repeatable | Needs terminological alignment |
| Main Points | Clarity | 5 | 4 |
|  | Hortatory Universal Truths | 4 | 3 |
|  | Proportional & Coexistent | 4 | 3 |
|  | Exposition Quality | 5 | 4 |
|  | Illustration Quality | 3 | 2 |
|  | Application Quality | 4 | 3 |
|  | Feedback | Solid structure; tighten illus. 2 | Needs clearer imperatives |
| Exegetical Support | Alignment with Text | 5 | 4 |
|  | Handles Difficulties | 4 | 3 |
|  | Proof Accuracy & Clarity | 5 | 4 |
|  | Context & Genre Considered | 4 | 3 |
|  | Not Belabored | 5 | 5 |
|  | Aids Rather Than Impresses | 5 | 4 |
|  | Feedback | Text governs sermon | Trim ornamental detail |
| Application | Clear & Practical | 4 | 3 |
|  | Redemptive Focus | 5 | 4 |
|  | Mandate vs Idea Distinction | 4 | 2 |
|  | Passage Supported | 3 | 3 |
|  | Feedback | Gospel‑driven; label mandates | Clarify authority level |
| Illustrations | Lived‑Body Detail | 4 | 3 |
|  | Strengthens Points | 4 | 3 |
|  | Proportion | 3 | 2 |
|  | Feedback | Trim length & diversify | Reduce anecdote weight |
| Conclusion | Summary | 5 | 4 |
|  | Compelling Exhortation | 4 | 3 |
|  | Climax | 4 | 2 |
|  | Pointed End | 5 | 4 |
|  | Feedback | Strong landing; heighten climax | Sharpen summons |
| Confidence | Scoring Confidence | 0.90 | 0.78 |

---

## Common Strength Indicators
* Proposition is concise, text‑anchored, repeatable.  
* FCF is narrow enough to drive structure yet broad enough to connect pastorally.  
* Clear redemptive arc culminating in Christ's person/work.  
* Applications flow organically from exposition (not bolted on).  
* Illustrations illuminate, not entertain; minimal redundancy.  
* Balanced exposition: neither word‑study overload nor shallow gloss.

## Common Failure Modes
* Missing or nebulous proposition (multiple competing themes).  
* Moralistic applications detached from grace/redemption.  
* Overuse of disconnected cross‑references (listener fatigue).  
* FCF stated as a vague universal ("we all struggle") with no textual tether.  
* Illustrations that overshadow the point or introduce doctrinal confusion.  
* Christ absent where text trajectory clearly points to Him (e.g., redemptive-historical pivot texts).  
* Applications purely behavior‑control without gospel motivation.

---

## Example Partial Evaluation Table (Derived Aggregations)

| Aggregated Criterion | Value |
| --------------------- | ----- |
| Textual_Fidelity | 4.25 |
| Proposition_Clarity | 5.0 |
| Introduction | 4.5 |
| Application_Effectiveness | 3.6 |
| Structure_Cohesion | 4.2 |
| Illustrations | 3.7 |
| Overall_Impact | 4 |

---

## Example Outputs

### Step 1 Example JSON Output

```json
{
	"Scripture Introduction": "Today we examine Ephesians 2:1-10, where Paul contrasts spiritual death with the grace that makes us alive in Christ...",
	"Sermon Introduction": "The preacher opens with a relatable scenario about striving for approval, leading into humanity's deeper need for mercy...",
	"Proposition": "God graciously makes the spiritually dead alive in Christ so that His grace, not our works, defines our future.",
	"Body": {
		"Main Points": [
			{
				"Point": "1. Our Natural Condition: Dead in Trespasses (vv. 1-3)",
				"Summary": "Explains spiritual death—alienation from God, enslaved to world, flesh, devil.",
				"Subpoints": ["Death described (v1)", "Enslavement (v2)", "Deserving wrath (v3)"],
				"Illustrations": ["Hospital code-blue analogy"],
				"Application": "Invites honest acknowledgment of helplessness; abandon self-salvation.",
				"Comments": "Faithful to text; could nuance 'wrath' pastoral tone a bit more.",
				"Feedback": "Clarify difference between total inability vs. utter worthlessness (avoid shame framing)."
			},
			{
				"Point": "2. God's Merciful Intervention (vv. 4-7)",
				"Summary": "Unpacks 'But God' pivot—rich mercy, great love, made alive, raised, seated.",
				"Subpoints": ["But God (v4)", "Made alive (v5)", "Raised & seated (v6-7)"],
				"Illustrations": ["Judge adopting the guilty"],
				"Application": "Assures believers of secure identity; fuels humility.",
				"Comments": "Christ-union well explained; slight overuse of adoption imagery not in immediate text.",
				"Feedback": "Tie exaltation (v6) more explicitly to corporate church dimension."
			},
			{
				"Point": "3. Grace Defines Our Purpose (vv. 8-10)",
				"Summary": "Explores salvation by grace through faith leading to prepared good works.",
				"Subpoints": ["Gift nature (v8-9)", "Workmanship & mission (v10)"],
				"Illustrations": ["Restoration of a masterpiece"],
				"Application": "Encourages serving from acceptance, not for it.",
				"Comments": "Accurate; could reference faith as instrument not merit.",
				"Feedback": "Short clarifier on 'works prepared beforehand' avoiding hyper-determinism misunderstanding."
			}
		]
	},
	"General Comments": {
		"Content Comments": "Doctrinally sound; consistent Pauline theology; rich union-with-Christ emphasis.",
		"Structure Comments": "Three-point arc mirrors text structure; transitions mostly smooth.",
		"Explanation Comments": "Greek nuances paraphrased understandably; could briefly note audience (Ephesus Gentiles)."
	},
	"Fallen Condition Focus": {
		"FCF": "Human spiritual deadness & self-reliant striving that ignores need for resurrecting grace.",
		"Comments": "Well-specified; fuels urgency of divine initiative and shapes grace-centered applications."
	},
	"Extraction Confidence": 0.92
}
```

### Step 2 Example JSON Output

```json
{
	"Introduction": {
		"FCF Introduced": 5,
		"Arouses Attention": 4,
		"Feedback": "Strong tension; trim hook by ~20% for pacing."
	},
	"Proposition": {
		"Principle + Application Wed": 5,
		"Establishes Main Theme": 5,
		"Summarizes Introduction": 5,
		"Feedback": "Memorable; consistent terminology reinforces cohesion."
	},
	"Main Points": {
		"Clarity": 5,
		"Hortatory Universal Truths": 4,
		"Proportional & Coexistent": 4,
		"Exposition Quality": 5,
		"Illustration Quality": 3,
		"Application Quality": 4,
		"Feedback": "Structure text-driven; shorten 2nd illustration; specify vocational application in point 3."
	},
	"Exegetical Support": {
		"Alignment with Text": 5,
		"Handles Difficulties": 4,
		"Proof Accuracy & Clarity": 5,
		"Context & Genre Considered": 4,
		"Not Belabored": 5,
		"Aids Rather Than Impresses": 5,
		"Feedback": "Minor historical nuance (audience background) could deepen relevance."
	},
	"Application": {
		"Clear & Practical": 4,
		"Redemptive Focus": 5,
		"Mandate vs Idea Distinction": 4,
		"Passage Supported": 3,
		"Feedback": "Highlight which exhortations are wisdom vs mandate; tighten text tie in final movement."
	},
	"Illustrations": {
		"Lived-Body Detail": 4,
		"Strengthens Points": 4,
		"Proportion": 3,
		"Feedback": "One narrative risks overshadowing exposition; diversify with a shorter contrast example."
	},
	"Conclusion": {
		"Summary": 5,
		"Compelling Exhortation": 4,
		"Climax": 4,
		"Pointed End": 5,
		"Feedback": "Clear gospel landing; consider amplifying resurrection link for emotional ascent."
	},
	"Scoring Confidence": 0.90,
	"Strengths": [
		"Text tightly governs structure",
		"Crisp, repeated proposition",
		"Grace-centered, non-moralistic application"
	],
	"Growth Areas": [
		"Trim second illustration length",
		"Clarify mandate vs wisdom distinction",
		"Add a brief socio-historical audience note"
	],
	"Action Steps": [
		"Reduce illustration 2 by ~25%",
		"Insert one-sentence Ephesian context after reading",
		"Label at least one application explicitly as wisdom guidance"
	]
}
```

---

## JSON Schemas

### Step 1 JSON Schema
```json
{
	"Scripture Introduction": "string",
	"Sermon Introduction": "string",
	"Proposition": "string",  // If absent use canonical placeholder: "No explicit proposition stated"
	"Body": {
		"Main Points": [
			{
				"Point": "string",
				"Summary": "string",
				"Subpoints": ["string"],
				"Illustrations": ["string"],
				"Application": "string",
				"Comments": "string",
				"Feedback": "string"
			}
		]
	},
	"General Comments": {
		"Content Comments": "string",
		"Structure Comments": "string",
		"Explanation Comments": "string"
	},
	"Fallen Condition Focus": {
		"FCF": "string",
		"Comments": "string"
	},
	"Extraction Confidence": 0.0
}
```

### Step 2 JSON Schema
```json
{
	"Introduction": {
		"FCF Introduced": 0,
		"Arouses Attention": 0,
		"Feedback": "string"
	},
	"Proposition": {
		"Principle + Application Wed": 0,
		"Establishes Main Theme": 0,
		"Summarizes Introduction": 0,
		"Feedback": "string"
	},
	"Main Points": {
		"Clarity": 0,
		"Hortatory Universal Truths": 0,
		"Proportional & Coexistent": 0,
		"Exposition Quality": 0,
		"Illustration Quality": 0,
		"Application Quality": 0,
		"Feedback": "string"
	},
	"Exegetical Support": {
		"Alignment with Text": 0,
		"Handles Difficulties": 0,
		"Proof Accuracy & Clarity": 0,
		"Context & Genre Considered": 0,
		"Not Belabored": 0,
		"Aids Rather Than Impresses": 0,
		"Feedback": "string"
	},
	"Application": {
		"Clear & Practical": 0,
		"Redemptive Focus": 0,
		"Mandate vs Idea Distinction": 0,
		"Passage Supported": 0,
		"Feedback": "string"
	},
	"Illustrations": {
		"Lived-Body Detail": 0,
		"Strengthens Points": 0,
		"Proportion": 0,
		"Feedback": "string"
	},
	"Conclusion": {
		"Summary": 0,
		"Compelling Exhortation": 0,
		"Climax": 0,
		"Pointed End": 0,
		"Feedback": "string"
	},
	"Scoring Confidence": 0.0,
	"Strengths": ["string"],
	"Growth Areas": ["string"],
	"Action Steps": ["string"]
}
```

NOTE: Keys intentionally contain spaces for readability (valid in JSON). Downstream parsers should treat them as string literals.


### Aggregated Summary JSON Schema

```json
{
	"Aggregated": {
		"Textual_Fidelity": 0.0,
		"Proposition_Clarity": 0.0,
		"Introduction": 0.0,
		"Application_Effectiveness": 0.0,
		"Structure_Cohesion": 0.0,
		"Illustrations": 0.0,
		"Overall_Impact": 0.0
	}
}
```

---

## Implementation Notes

* Step 1 should avoid invented content—only infer from transcript context; mark clearly if data is absent (e.g., "No explicit proposition stated").  
* Use consistent verse notation (e.g., "Eph 2:1-3").  
* Avoid labeling ordinary scripture quotations as illustrations.  
* In applications, privilege grace-driven transformation over mere behavioral exhortation.  
* Confidence score rationale (internal) should consider ambiguity signals; do not expose rationale in JSON.  
* Maintain strict JSON validity (double quotes, no trailing commas).  
* Keep categories stable for downstream parsers.
* Use canonical placeholder "No explicit proposition stated" when a proposition cannot be confidently extracted.
* Aggregated metrics are rounded to 2 decimal places (round half up) for display while retaining raw precision internally.

---

## Glossary of Key Rubric Terms
* FCF Introduced – Explicit naming of the specific fallen condition the text addresses.
* Principle + Application Wed – Proposition fuses what is true with why/what response is required in Christ.
* Hortatory Universal Truths – Timeless, text‑warranted principles or implications rather than episodic narration.
* Proportional & Coexistent – Points receive balanced development and operate at the same logical altitude.
* Handles Difficulties – Engages indispensable interpretive tensions (linguistic, contextual, theological) briefly & honestly.
* Not Belabored – Stops explanatory/proof detail once sufficient for clarity & persuasion.
* Aids Rather Than Impresses – Exegetical data serves comprehension; avoids performative scholarship.
* Mandate vs Idea Distinction – Clear labeling of divine commands versus pastoral wisdom or illustrative suggestions.
* Lived-Body Detail – Concrete sensory or situational specificity that grounds illustrations in reality.
* Proportion (Illustrations) – Illustrations sized & spaced to support, not dominate, exposition.
* Pointed End – A decisive, purposeful conclusion (no drift into announcements or filler).
* Scoring Confidence – Evaluator's confidence in rubric scoring fidelity given input quality.
* Extraction Confidence – System confidence in structural parsing accuracy (Step 1).

---

## Usage Summary
1. Run Step 1 extraction on transcript.  
2. Optionally convert Step 1 JSON to Markdown for human reviewers.  
3. Run Step 2 scoring referencing extracted fields (never re-parsing raw transcript if structured data sufficient).  
4. Present combined report (markdown + both JSONs) or persist to storage as needed.

---

This framework ensures consistent, Christ-centered, text-faithful sermon evaluation and constructive coaching feedback.
