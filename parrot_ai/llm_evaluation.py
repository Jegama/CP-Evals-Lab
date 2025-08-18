"""Evaluation utilities and engine class for Arabic theological QA model assessment.

This module exposes an ``EvaluationEngine`` class encapsulating the earlier
function-based workflow so notebooks can simply do:

    from parrot_ai.evaluation import EvaluationEngine, default_engine
    engine = EvaluationEngine(model="gpt-5-mini")  # or use default_engine
    pairs = engine.load_qa_pairs("data/arabic/ar_training_dataset_small_model.jsonl")
    results = engine.batch_evaluate(pairs, limit=10)

Backward compatibility: the previous top-level functions ``load_qa_pairs``,
``evaluate_answer`` and ``batch_evaluate`` remain as thin wrappers delegating
to a module-level ``default_engine`` instance so existing notebooks continue to work.
"""

from typing import List, Tuple, Dict, Any, Optional, Iterable
from pydantic import BaseModel
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import json

load_dotenv()

# --- Data loading utility (kept standalone for reuse) ---
def load_qa_pairs(
    jsonl_path: str,
    question_list_path: Optional[str] = "data/arabic/ar_eval_questions.txt",
    limit: int = 100,
) -> List[Tuple[str, str]]:
    """Load (question, answer) pairs filtered to a curated eval question list.

    Behavior change (per user request): Instead of loading every pair in the
    dataset, we now restrict to the first ``limit`` questions present in the
    ``question_list_path`` file (one question per line) if that file exists.

    If ``question_list_path`` is None or the file is missing/empty, the
    function falls back to previous behavior (load all pairs) but still caps
    at ``limit`` if provided (>0).
    """
    question_filter: Optional[List[str]] = None
    question_set: Optional[set[str]] = None
    if question_list_path and limit != 0:
        try:
            with open(question_list_path, 'r', encoding='utf-8') as qf:
                question_filter = [ln.strip() for ln in qf if ln.strip()][:limit]
            if question_filter:
                question_set = set(question_filter)
        except FileNotFoundError:
            question_filter = None
            question_set = None
    pairs: List[Tuple[str, str]] = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if question_set and len(pairs) >= len(question_set):
                break
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = obj.get('messages') or []
            if len(messages) >= 3:
                user = messages[1].get('content') if isinstance(messages[1], dict) else None
                assistant = messages[2].get('content') if isinstance(messages[2], dict) else None
                if user and assistant:
                    if question_set is None or user in question_set:
                        pairs.append((user, assistant))
                        # preserve order as in question_filter if provided
    # Re-order according to the question list if present
    if question_filter and pairs:
        order_map = {q: i for i, q in enumerate(question_filter)}
        pairs.sort(key=lambda qa: order_map.get(qa[0], 10_000_000))
    # Apply limit fallback if no question list was used
    if question_set is None and limit and limit > 0:
        pairs = pairs[:limit]
    return pairs

def load_eval_questions(question_file: str = "data/arabic/ar_eval_questions.txt", limit: int = 100) -> List[str]:
    """Load up to ``limit`` evaluation questions from text file (one per line)."""
    try:
        with open(question_file, 'r', encoding='utf-8') as f:
            return [ln.strip() for ln in f if ln.strip()][:limit]
    except FileNotFoundError:
        return []

ARABIC_BLOCKS = [
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0x0870, 0x089F),  # Arabic Extended-B
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    (0x1EE00, 0x1EEFF) # Arabic Mathematical Alphabetic Symbols
]

def is_arabic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(start <= cp <= end for start, end in ARABIC_BLOCKS)

def basic_language_metrics(text: str) -> Dict[str, Any]:
    """Compute simple Arabic vs non-Arabic letter percentages."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return {'arabic_char_pct': 0.0, 'non_arabic_char_pct': 0.0, 'total_letters': 0}
    arabic = sum(1 for c in letters if is_arabic_char(c))
    arabic_pct = arabic / len(letters) * 100
    return {
        'arabic_char_pct': round(arabic_pct, 2),
        'non_arabic_char_pct': round(100 - arabic_pct, 2),
        'total_letters': len(letters)
    }

class AdherenceModel(BaseModel):
    Core: int
    Secondary: int
    Tertiary_Handling: int
    Biblical_Basis: int
    Consistency: int
    Overall: int

class KindnessGentlenessModel(BaseModel):
    Core_Clarity_with_Kindness: int
    Pastoral_Sensitivity: int
    Secondary_Fairness: int
    Tertiary_Neutrality: int
    Tone: int
    Overall: int

class InterfaithSensitivityModel(BaseModel):
    Respect_and_Handling_Objections: int
    Objection_Acknowledgement: int
    Evangelism: int
    Gospel_Boldness: int 
    Overall: int

class ArabicAccuracyDetailed(BaseModel):
    Grammar_and_Syntax: int
    Theological_Nuance: int
    Contextual_Clarity: int
    Consistency_of_Terms: int
    Arabic_Purity: int
    Penalty_Reason: Optional[str] = None
    Overall: int

class EvaluationResult(BaseModel):
    Adherence: AdherenceModel
    Kindness_and_Gentleness: KindnessGentlenessModel
    Interfaith_Sensitivity: InterfaithSensitivityModel
    Arabic_Accuracy: ArabicAccuracyDetailed

# Doctrine tier definitions injected so model has explicit reference (sourced from resources/Evaluation Framework.md)
DOCTRINE_TIER_DEFINITIONS = """
Doctrine Tier Definitions (for evaluator reference – do NOT output):
Core (جوهرية / غير قابلة للتمييع النسبي): الثالوث الأقدس، ألوهية وناسوت المسيح، التجسد والولادة العذراوية، الكفارة الكاملة الضرورية، موت المسيح وقيامته الجسدية، التبرير بالنعمة من خلال الإيمان وحده بالمسيح وحده، سلطان الكتاب المقدس، رجوع المسيح والدينونة الأخيرة.
Secondary (هامة لكن لا تُخرج من الإيمان): المعمودية (طريقة وموضوع)، نظام الحكم الكنسي، العشاء الرباني، المواهب الروحية (استمرارية/توقف)، أدوار المرأة، صيغ/مقاربات التقديس، العهدية/التدبيرية، ضمان الخلاص، نظريات كيفية الكفارة (عقابية بدلية، نصر المسيح، تأثير أخلاقي...).
Tertiary (اجتهادية لا تؤثر على جوهر الشركة): تفاصيل الإِسخاتُولوجيا (أمور الآخرة الزمنية)، أسلوب العبادة، نماذج المشورة، قراءات الخلق (أيام حرفية، عمر قديم، تطور موجّه)، مسائل الحرية المسيحية، تفاصيل الانضباط الكنسي، دور المنظمات المساندة، مسائل تفسيرية ثانوية.
"""

# System prompt (expanded with doctrine tier definitions + anti-relativism guardrail)
EVAL_SYSTEM_PROMPT = f"""أنت مُقَيِّم مستقل متخصص في تقييم إجابات لاهوتية مسيحية بالعربية لمخاطبة جمهور قد يكون إسلامياً.
{DOCTRINE_TIER_DEFINITIONS}
مهمة التقييم: إصدار حكم مُنضبط وفق Rubric محدد على أربعة أبعاد: الالتزام العقائدي، اللطف والرقة، الحساسية بين الأديان، ودقة العربية.
تحذير: عبارات التمييع / relativism (مثل: "كل الأديان صحيحة بنفس الدرجة"، أو رفض غير مبرر لتقديم حقائق جوهرية للمسيحية في سياق سؤال مباشر) تُعد نقصاً في Adherence/Core وGospel_Boldness و/أو Evangelism.
قيِّم النص كما هو فقط (لا تضف شرحاً ولا تُكمل الإجابة). أعد JSON موافقاً للمخطط المفروض بواسطة النظام (response_format). ممنوع أي نص خارج JSON."""

# Detailed evaluation instructions (rich rubric)
EVAL_INSTRUCTIONS = """
قواعد عامة

الدرجات الصحيحة 1–5: (1 = فشل واضح، 3 = مقبول مع نواقص، 5 = ممتاز منضبط).

“Overall” لكل بُعد ليس متوسطاً أعمى؛ اجعله ضمن ±1 من متوسط مقاييس البعد بعد تطبيق القيود أدناه.

حالات قصوى/قيود (Knockouts):
• تناقض في عقيدة جوهرية ⇒ Adherence.Core ≤ 2 وAdherence.Overall ≤ 3.
• سخرية/ازدراء ديني ⇒ Interfaith_Sensitivity.Respect_and_Handling_Objections ≤ 1 وInterfaith_Sensitivity.Overall ≤ 2.
• إذا كانت Arabic_Purity ≤ 2 فعادةً Grammar_and_Syntax ≤ 3 ما لم توجد سلامة لغوية واضحة.

إجابة فارغة ⇒ كل القيم = 1 وArabic_Accuracy.Penalty_Reason = "Empty answer".

Adherence
• Core: 1 إنكار/تحريف واضح؛ 3 عرض صحيح مع ثغرات؛ 5 عرض أمين خالٍ من تشويه مع منطق كتابي موجز عند اللزوم.
• Secondary: 1 خلط/أحكام مغلوطة؛ 3 تحديد صحيح مع نقص توازن؛ 5 عرض دقيق لموقفٍ إصلاحيٍّ مع ذكر بدائل باحترام.
• Tertiary_Handling: 1 حسم قطعي؛ 3 يذكر أنها اجتهادية بلا تفصيل؛ 5 يفرّق بوضوح مع لغة غير صدامية.
• Biblical_Basis: 1 بلا سند؛ 3 مبدأ/آية عامة؛ 5 استشهاد أو تلخيص لاهوتي مناسب موجز.
• Consistency: 1 تناقض صريح؛ 3 تماسك عام مع موضع غامض؛ 5 تماسك كامل.

Kindness_and_Gentleness
• Core_Clarity_with_Kindness: وضوح دون حدّة.
• Pastoral_Sensitivity: تعاطف وتدرّج.
• Secondary_Fairness: يعرض موقفه وبدائل بلا تبخيس.
• Tertiary_Neutrality: لغة استشارية غير فَرَضية.
• Tone: خلوّ من السخرية والتهجّم.

Interfaith_Sensitivity
• Respect_and_Handling_Objections: احترام صريح، ذكر الاعتراضات الشائعة قبل الرد.
• Objection_Acknowledgement: يذكر اعتراضات مثل الثالوث/التحريف قبل العرض.
• Evangelism: 1 غياب؛ 3 تلميح ضعيف؛ 5 دعوة واضحة للمسيح بلغة محترمة.
• Gospel_Boldness: يقيس وضوح وجرأة التأكيد على فرادة المسيح والإنجيل دون عدوان. 1 = تمييع/نسبية (كل الأديان سواء، رفض لا مبرر له للجواب)؛ 3 = تأكيد عام للمسيحية مع حذر زائد أو صياغة دفاعية مترددة؛ 5 = عرض صريح متزن لفرادة المسيح (الخلاص بموته وقيامته) مع احترام الآخرين.

Arabic_Accuracy
• Grammar_and_Syntax: صرف/نحو وسلاسة طبيعية.
• Theological_Nuance: مصطلحات دقيقة (الثالوث الأقدس، ألوهية المسيح، التبرير…).
• Contextual_Clarity: تنظيم منطقي وتجنّب الترجمة الحرفية.
• Consistency_of_Terms: ثبات المصطلحات.
• Arabic_Purity: 5 ≥ 98%؛ 4 ≥ 90%؛ 3 ≥ 75%؛ 2 ≥ 60%؛ 1 < 60% (يُراعى المحتوى لا الواجهة فقط).
• Penalty_Reason: اذكر سببًا مختصرًا عند خفض الدرجة بسبب عجمة/تعريب حرفي أو تناقض.

الإخراج
التزم تماماً بالمخطط المفروض (لا تضف حقولاً أو تعليقات).

لماذا هذا كافٍ؟ لأن Structured Outputs ستفرض المخطط (Pydantic)؛ لا حاجة لتكراره نصياً داخل الـ prompt."""

DEFAULT_MODEL = "gpt-5-mini"

# Post-processing: enforce purity & grammar caps based on heuristic percentage

def apply_purity_penalty(answer: str, result_dict: dict) -> dict:
    """Apply heuristic purity cap and related grammar adjustments."""
    lang_metrics = basic_language_metrics(answer)
    purity_pct = lang_metrics['arabic_char_pct']
    if purity_pct >= 98: cap = 5
    elif purity_pct >= 90: cap = 4
    elif purity_pct >= 75: cap = 3
    elif purity_pct >= 60: cap = 2
    else: cap = 1
    arabic_section = result_dict.get('Arabic_Accuracy', {})
    if arabic_section.get('Arabic_Purity', cap) > cap:
        arabic_section['Arabic_Purity'] = cap
        reason = arabic_section.get('Penalty_Reason') or ''
        if reason:
            reason += ' | '
        arabic_section['Penalty_Reason'] = reason + f"Capped purity (heuristic {purity_pct}%)"
    if cap <= 2:
        if arabic_section.get('Grammar_and_Syntax', 5) > 3:
            arabic_section['Grammar_and_Syntax'] = 3
            reason = arabic_section.get('Penalty_Reason') or ''
            if reason:
                reason += ' | '
            arabic_section['Penalty_Reason'] = reason + 'Grammar capped due to low purity'
        if arabic_section.get('Overall', 5) > 3:
            arabic_section['Overall'] = min(3, arabic_section['Overall'])
    arabic_section['Heuristic_Arabic_Purity_Pct'] = purity_pct
    result_dict['Arabic_Accuracy'] = arabic_section
    return result_dict

# Clamp Overall within ±1 of rounded mean of its component keys (model honesty guardrail)
def clamp_overall(section: dict, keys: list[str]) -> None:
    vals = [section[k] for k in keys if isinstance(section.get(k), int)]
    if not vals or 'Overall' not in section or not isinstance(section['Overall'], int):
        return
    target = round(sum(vals) / len(vals))
    low, high = target - 1, target + 1
    section['Overall'] = min(max(section['Overall'], low), high)

def clamp_all_overalls(result_dict: dict) -> None:
    clamp_overall(result_dict.get('Adherence', {}), [
        'Core','Secondary','Tertiary_Handling','Biblical_Basis','Consistency'
    ])
    clamp_overall(result_dict.get('Kindness_and_Gentleness', {}), [
        'Core_Clarity_with_Kindness','Pastoral_Sensitivity','Secondary_Fairness','Tertiary_Neutrality','Tone'
    ])
    clamp_overall(result_dict.get('Interfaith_Sensitivity', {}), [
        'Respect_and_Handling_Objections','Objection_Acknowledgement','Evangelism','Gospel_Boldness'
    ])
    clamp_overall(result_dict.get('Arabic_Accuracy', {}), [
        'Grammar_and_Syntax','Theological_Nuance','Contextual_Clarity','Consistency_of_Terms','Arabic_Purity'
    ])

# Clamp all scale scores to [1, 5]
def clamp_scale_scores(d: dict) -> dict:
    def clip(v):
        return max(1, min(5, int(v))) if isinstance(v, int) else 1
    for sect_key, sect in d.items():
        if not isinstance(sect, dict):
            continue
        for k, v in list(sect.items()):
            if k in ("Penalty_Reason", "Heuristic_Arabic_Purity_Pct"):
                continue
            sect[k] = clip(v)
    return d

def enforce_knockouts(answer: str, result_dict: dict) -> dict:
    """Apply rubric knockout rules and empty-answer handling."""
    if not answer.strip():
        for section_key, fields in [
            ('Adherence', ['Core','Secondary','Tertiary_Handling','Biblical_Basis','Consistency','Overall']),
            ('Kindness_and_Gentleness', ['Core_Clarity_with_Kindness','Pastoral_Sensitivity','Secondary_Fairness','Tertiary_Neutrality','Tone','Overall']),
            ('Interfaith_Sensitivity', ['Respect_and_Handling_Objections','Objection_Acknowledgement','Evangelism','Gospel_Boldness','Overall']),
            ('Arabic_Accuracy', ['Grammar_and_Syntax','Theological_Nuance','Contextual_Clarity','Consistency_of_Terms','Arabic_Purity','Overall'])
        ]:
            section = result_dict.get(section_key, {})
            for f in fields:
                section[f] = 1
            if section_key == 'Arabic_Accuracy':
                section['Penalty_Reason'] = 'Empty answer'
            result_dict[section_key] = section
        return result_dict

    adherence = result_dict.get('Adherence', {})
    if isinstance(adherence.get('Core'), int) and adherence.get('Core', 5) <= 2 and adherence.get('Overall', 5) > 3:
        adherence['Overall'] = 3
    result_dict['Adherence'] = adherence

    interfaith = result_dict.get('Interfaith_Sensitivity', {})
    if isinstance(interfaith.get('Respect_and_Handling_Objections'), int) and interfaith.get('Respect_and_Handling_Objections', 5) <= 1 and interfaith.get('Overall', 5) > 2:
        interfaith['Overall'] = 2
    result_dict['Interfaith_Sensitivity'] = interfaith

    arabic = result_dict.get('Arabic_Accuracy', {})
    if isinstance(arabic.get('Arabic_Purity'), int) and arabic.get('Arabic_Purity', 5) <= 2 and arabic.get('Grammar_and_Syntax', 5) > 3:
        arabic['Grammar_and_Syntax'] = 3
        reason = arabic.get('Penalty_Reason') or ''
        if reason:
            reason += ' | '
        arabic['Penalty_Reason'] = reason + 'Grammar capped due to low purity (knockout)'
    result_dict['Arabic_Accuracy'] = arabic
    return result_dict

# --- Boldness / anti-relativism heuristic adjustments ---
RELATIVISM_PATTERNS = [
    "كل الأديان", "جميع الأديان", "كل الديانات", "لا أستطيع تقديم رأي شخصي", "لا أستطيع إعطاء رأي شخصي",
    "كنموذج ذكاء اصطناعي", "كذكاء اصطناعي", "كنموذج لغوي", "أنا نموذج", "I cannot provide a personal opinion",
    "all religions", "true for them", "equally true"
]

BOLD_KEYWORDS = ["يسوع", "المسيح", "الصليب", "الخلاص", "القيامة", "ابن الله", "التجسد", "المخلص"]

def adjust_boldness(answer: str, result_dict: dict) -> dict:
    interfaith = result_dict.get('Interfaith_Sensitivity', {})
    # Ensure field exists
    if 'Gospel_Boldness' not in interfaith or not isinstance(interfaith.get('Gospel_Boldness'), int):
        interfaith['Gospel_Boldness'] = 3
    lower_ans = answer.lower()
    has_relativism = any(pat.lower() in lower_ans for pat in RELATIVISM_PATTERNS)
    has_bold = any(kw in answer for kw in BOLD_KEYWORDS)
    # Penalize relativism if no bold Christ-centered content
    if has_relativism and not has_bold:
        interfaith['Gospel_Boldness'] = min(interfaith.get('Gospel_Boldness', 3), 2)
        # Also cap Evangelism
        if interfaith.get('Evangelism', 5) > 3:
            interfaith['Evangelism'] = 3
    # Reward clear boldness (without overriding explicit low scores from model unless neutral)
    if has_bold and not has_relativism and interfaith.get('Gospel_Boldness', 0) < 4:
        interfaith['Gospel_Boldness'] = 4
    # If both strong bold keywords and explicit invitation words, consider 5
    if has_bold and ("توب" in answer or "تعال" in answer or "آمن" in answer) and not has_relativism:
        interfaith['Gospel_Boldness'] = max(interfaith['Gospel_Boldness'], 5)
    result_dict['Interfaith_Sensitivity'] = interfaith
    return result_dict

# Primary evaluation call

class EvaluationEngine:
    """Encapsulates evaluation logic for reuse in notebooks or scripts.

    Note: Temperature and max token parameters deliberately omitted for stability
    with gpt-5 family and Together API per user request.
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = DEFAULT_MODEL,
        system_prompt: str = EVAL_SYSTEM_PROMPT,
        instructions: str = EVAL_INSTRUCTIONS,
        seed: Optional[int] = 7,
    ) -> None:
        self.client = client or OpenAI()
        self.model = model
        self.system_prompt = system_prompt
        self.instructions = instructions
        self.seed = seed

    # -------------- Core single evaluation --------------
    def evaluate(self, question: str, answer: str) -> dict:
        """Evaluate a single (question, answer) pair returning rubric dict."""
        # Fast path for empty / whitespace-only answers: assign all 1s (lowest rubric) + penalty
        if not answer.strip():
            result_dict = {
                'Adherence': {k: 1 for k in ['Core','Secondary','Tertiary_Handling','Biblical_Basis','Consistency','Overall']},
                'Kindness_and_Gentleness': {k: 1 for k in ['Core_Clarity_with_Kindness','Pastoral_Sensitivity','Secondary_Fairness','Tertiary_Neutrality','Tone','Overall']},
                'Interfaith_Sensitivity': {k: 1 for k in ['Respect_and_Handling_Objections','Objection_Acknowledgement','Evangelism','Gospel_Boldness','Overall']},
                'Arabic_Accuracy': {**{k:1 for k in ['Grammar_and_Syntax','Theological_Nuance','Contextual_Clarity','Consistency_of_Terms','Arabic_Purity','Overall']}, 'Penalty_Reason': 'Empty answer'}
            }
            return result_dict
        user_content = f"السؤال:\n{question}\n\nالإجابة:\n{answer}\n\nقيّم وفق التعليمات السابقة."
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.instructions},
                {"role": "user", "content": user_content},
            ],
            response_format=EvaluationResult,
            seed=self.seed,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Failed to parse evaluation result from OpenAI response")
        result_dict = json.loads(parsed.model_dump_json())
        result_dict = clamp_scale_scores(result_dict)
        result_dict = apply_purity_penalty(answer, result_dict)
        clamp_all_overalls(result_dict)
        result_dict = enforce_knockouts(answer, result_dict)
        result_dict = adjust_boldness(answer, result_dict)
        clamp_all_overalls(result_dict)
        return result_dict

    # -------------- Batch utilities --------------
    def batch_evaluate(
        self,
        pairs: Iterable[Tuple[str, str]],
        limit: Optional[int] = None,
        progress: bool = True,
        stop_on_error: bool = False,
    ) -> list[dict]:
        """Evaluate multiple QA pairs.

        Args:
            pairs: iterable of (question, answer)
            limit: optional max number to process
            progress: print progress ticks
            stop_on_error: raise immediately instead of recording error dict
        """
        out: list[dict] = []
        processed = 0
        # Determine total length for progress bar
        total: Optional[int] = None
        try:
            total = len(pairs)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            total = None
        if limit is not None:
            if total is None:
                total = limit
            else:
                total = min(total, limit)

        use_bar = False
        bar = None
        if progress:
            try:
                bar = tqdm(total=total, desc="Evaluating", unit="qa")
                use_bar = True
            except Exception:  # pragma: no cover - fallback if tqdm missing
                use_bar = False

        for i, (q, a) in enumerate(pairs):
            if limit is not None and processed >= limit:
                break
            try:
                res = self.evaluate(q, a)
                out.append({"index": i, "question": q, "answer": a, "evaluation": res})
            except Exception as e:  # noqa: BLE001
                if stop_on_error:
                    if use_bar and bar is not None:
                        bar.close()
                    raise
                out.append({"index": i, "question": q, "error": str(e)})
            processed += 1
            if use_bar and bar is not None:
                bar.update(1)
        if use_bar and bar is not None:
            bar.close()
        return out

    # -------------- Dataset convenience --------------
    @staticmethod
    def load_qa_pairs(jsonl_path: str) -> List[Tuple[str, str]]:
        return load_qa_pairs(jsonl_path)

    def evaluate_dataset(
        self,
        jsonl_path: str,
        limit: Optional[int] = None,
        progress: bool = True,
    ) -> dict:
        """Load a dataset and batch evaluate.

        Returns dict with raw results list and a small summary aggregate.
        """
        pairs = self.load_qa_pairs(jsonl_path)
        results = self.batch_evaluate(pairs, limit=limit, progress=progress)
        purity_counts: Dict[int, int] = {}
        for r in results:
            eval_section = r.get('evaluation', {}).get('Arabic_Accuracy') if 'evaluation' in r else None
            if eval_section:
                p = eval_section.get('Arabic_Purity')
                if isinstance(p, int):
                    purity_counts[p] = purity_counts.get(p, 0) + 1
        summary = {
            'total_evaluated': len(results),
            'arabic_purity_distribution': purity_counts,
        }
        return {'results': results, 'summary': summary}

    # -------------- Response generation (OpenAI Responses API) --------------
    def generate_responses_openai(
        self,
        questions: List[str],
        model: Optional[str] = None,
        progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generate answers for a list of questions via OpenAI Responses API.

        Returns list of {question, answer, model, provider} dicts.
        """
        use_model = model or self.model
        out: List[Dict[str, Any]] = []
        bar = None
        if progress:
            try:
                bar = tqdm(total=len(questions), desc="Generating (OpenAI)", unit="q")
            except Exception:  # pragma: no cover
                bar = None
        for i, q in enumerate(questions):
            resp = self.client.responses.create(
                model=use_model,
                input=q,
            )
            answer = getattr(resp, 'output_text', None)
            if answer is None:
                try:
                    parts = []
                    for item in getattr(resp, 'output', []) or []:
                        text = getattr(item, 'content', None)
                        if isinstance(text, list):
                            for seg in text:
                                if isinstance(seg, dict) and seg.get('type') == 'output_text':
                                    parts.append(seg.get('text', ''))
                        elif isinstance(text, str):
                            parts.append(text)
                    answer = "".join(parts)
                except Exception:  # noqa: BLE001
                    answer = ""
            out.append({
                'index': i,
                'question': q,
                'answer': answer,
                'model': use_model,
                'provider': 'openai'
            })
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()
        return out

    def generate_responses_openai_from_file(
        self,
        question_file: str = "data/arabic/ar_eval_questions.txt",
        limit: int = 100,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Load questions from file then call ``generate_responses_openai``."""
        questions = load_eval_questions(question_file, limit=limit)
        return self.generate_responses_openai(questions, **kwargs)

    # -------------- Response generation (Together.ai) --------------
    def generate_responses_together(
        self,
        questions: List[str],
        model: str = "openai/gpt-oss-120b",
        progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generate answers using Together.ai chat completions API.

        Import performed lazily to avoid dependency unless used.
        """
        try:
            from together import Together  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "'together' package not installed. Add it to requirements and pip install."
            ) from e
        client = Together()
        out: List[Dict[str, Any]] = []
        bar = None
        if progress:
            try:
                bar = tqdm(total=len(questions), desc="Generating (Together)", unit="q")
            except Exception:  # pragma: no cover
                bar = None
        for i, q in enumerate(questions):
            messages = []
            messages.append({"role": "user", "content": q})
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            # Some client versions may return an iterator (stream) or an object with .choices
            answer = ""
            try:
                choices = getattr(resp, 'choices', None)
                if choices:
                    first = choices[0]
                    msg = getattr(first, 'message', None)
                    if msg:
                        answer = getattr(msg, 'content', '') or ''
                else:
                    # Attempt to iterate if streaming
                    collected = []
                    for chunk in resp:  # type: ignore
                        ch_choices = getattr(chunk, 'choices', None)
                        if ch_choices:
                            delta = getattr(ch_choices[0], 'delta', None)
                            if delta:
                                collected.append(getattr(delta, 'content', '') or '')
                    if collected:
                        answer = ''.join(collected)
            except Exception:  # noqa: BLE001
                answer = ""
            out.append({
                'index': i,
                'question': q,
                'answer': answer,
                'model': model,
                'provider': 'together'
            })
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()
        return out

    def generate_responses_together_from_file(
        self,
        question_file: str = "data/arabic/ar_eval_questions.txt",
        limit: int = 100,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Load questions from file then call ``generate_responses_together``."""
        questions = load_eval_questions(question_file, limit=limit)
        return self.generate_responses_together(questions, **kwargs)


# ---------------- Backward compatible top-level wrappers ---------------- #

_default_client = OpenAI()
default_engine = EvaluationEngine(client=_default_client)

def evaluate_answer(question: str, answer: str, model: str = DEFAULT_MODEL) -> dict:  # pragma: no cover - wrapper
    if model != default_engine.model:
        # temporary override via a throwaway engine
        return EvaluationEngine(client=default_engine.client, model=model).evaluate(question, answer)
    return default_engine.evaluate(question, answer)

def batch_evaluate(pairs, limit: int = 5, model: str = DEFAULT_MODEL, progress: bool = True):  # pragma: no cover - wrapper
    if model != default_engine.model:
        eng = EvaluationEngine(client=default_engine.client, model=model)
    else:
        eng = default_engine
    return eng.batch_evaluate(pairs, limit=limit, progress=progress)

__all__ = [
    'EvaluationEngine',
    'default_engine',
    'load_qa_pairs',
    'load_eval_questions',
    'evaluate_answer',
    'batch_evaluate',
    'EvaluationResult',
]