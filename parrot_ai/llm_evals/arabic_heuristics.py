"""Arabic-specific heuristic functions for evaluation score post-processing."""

import re
from typing import Dict, Any

ARABIC_BLOCKS = [
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0x0870, 0x089F),  # Arabic Extended-B
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    (0x1EE00, 0x1EEFF),  # Arabic Mathematical Alphabetic Symbols
]


def is_arabic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(start <= cp <= end for start, end in ARABIC_BLOCKS)


def basic_language_metrics(text: str) -> Dict[str, Any]:
    """Compute simple Arabic vs non-Arabic letter percentages."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return {"arabic_char_pct": 0.0, "non_arabic_char_pct": 0.0, "total_letters": 0}
    arabic = sum(1 for c in letters if is_arabic_char(c))
    arabic_pct = arabic / len(letters) * 100
    return {
        "arabic_char_pct": round(arabic_pct, 2),
        "non_arabic_char_pct": round(100 - arabic_pct, 2),
        "total_letters": len(letters),
    }


def apply_purity_penalty(answer: str, result_dict: dict) -> dict:
    """Apply heuristic purity cap and related grammar adjustments."""
    lang_metrics = basic_language_metrics(answer)
    purity_pct = lang_metrics["arabic_char_pct"]
    if purity_pct >= 98:
        cap = 5
    elif purity_pct >= 90:
        cap = 4
    elif purity_pct >= 75:
        cap = 3
    elif purity_pct >= 60:
        cap = 2
    else:
        cap = 1
    arabic_section = result_dict.get("Arabic_Accuracy", {})
    if arabic_section.get("Arabic_Purity", cap) > cap:
        arabic_section["Arabic_Purity"] = cap
        reason = arabic_section.get("Penalty_Reason") or ""
        if reason:
            reason += " | "
        arabic_section["Penalty_Reason"] = (
            reason + f"Capped purity (heuristic {purity_pct}%)"
        )
    if cap <= 2:
        if arabic_section.get("Grammar_and_Syntax", 5) > 3:
            arabic_section["Grammar_and_Syntax"] = 3
            reason = arabic_section.get("Penalty_Reason") or ""
            if reason:
                reason += " | "
            arabic_section["Penalty_Reason"] = (
                reason + "Grammar capped due to low purity"
            )
        if arabic_section.get("Overall", 5) > 3:
            arabic_section["Overall"] = min(3, arabic_section["Overall"])
    arabic_section["Heuristic_Arabic_Purity_Pct"] = purity_pct
    result_dict["Arabic_Accuracy"] = arabic_section
    return result_dict


# ---------------------------------------------------------------------------
# Arabic ceiling-compression calibration (mirrors english_heuristics.py)
# ---------------------------------------------------------------------------

# Regex for Arabic Scripture citations: "اسم_الكتاب رقم:رقم" patterns
# Supports both Western digits (3:16) and Eastern Arabic digits (٣:١٦)
_ARABIC_SCRIPTURE_RE = re.compile(
    r"(?:"
    # Old Testament
    r"تكوين|خروج|لاويين|عدد|تثنية"
    r"|يشوع|قضاة|راعوث"
    r"|صموئيل|ملوك|أخبار الأيام"
    r"|عزرا|نحميا|أستير"
    r"|أيوب|مزمور|مزامير|أمثال|جامعة|نشيد الأنشاد"
    r"|إشعياء|إرميا|مراثي إرميا|حزقيال|دانيال"
    r"|هوشع|يوئيل|عاموس|عوبديا|يونان|ميخا|ناحوم|حبقوق|صفنيا"
    r"|حجي|زكريا|ملاخي"
    # New Testament
    r"|متى|مرقس|لوقا|يوحنا|أعمال الرسل|أعمال"
    r"|رومية|كورنثوس|غلاطية|أفسس"
    r"|فيلبي|كولوسي|تسالونيكي|تيموثاوس|تيطس|فليمون|عبرانيين"
    r"|يعقوب|بطرس|يهوذا|رؤيا"
    r")"
    r"\s*[12٣]?\s*"  # optional book number
    r"[0-9٠-٩]{1,3}\s*[:：]\s*[0-9٠-٩]{1,3}"  # chapter:verse
    r"(?:\s*[-–]\s*[0-9٠-٩]{1,3})?"  # optional verse range
)

_ARABIC_THEOLOGICAL_TERMS = [
    "الكفارة البدلية",  # substitutionary atonement
    "الكفارة العقابية",  # penal atonement
    "التبرير بالإيمان",  # justification by faith
    "التبرير بالنعمة",  # justification by grace
    "الاتحاد الأقنومي",  # hypostatic union
    "التجسد",  # incarnation
    "الخطيئة الأصلية",  # original sin
    "الفساد الكلي",  # total depravity
    "الاختيار غير المشروط",  # unconditional election
    "الفداء المحدود",  # limited atonement / particular redemption
    "النعمة التي لا تُقاوَم",  # irresistible grace
    "مثابرة القديسين",  # perseverance of the saints
    "ثبات القديسين",  # perseverance of the saints (alt)
    "الاحتساب",  # imputation
    "الكفارة",  # atonement/propitiation
    "التقديس",  # sanctification
    "التمجيد",  # glorification
    "الولادة الجديدة",  # regeneration
    "لاهوت العهد",  # covenant theology
    "سيادة الله",  # sovereignty of God
    "الثالوث",  # Trinity
    "الثالوث الأقدس",  # Holy Trinity
    "ألوهية المسيح",  # deity of Christ
    "لاهوت المسيح",  # deity of Christ (alt)
    "الولادة العذراوية",  # virgin birth
    "القيامة الجسدية",  # bodily resurrection
    "المجيء الثاني",  # second coming
    "الدينونة الأخيرة",  # final judgment
    "بالنعمة وحدها",  # by grace alone (sola gratia)
    "بالإيمان وحده",  # by faith alone (sola fide)
    "بالكتاب المقدس وحده",  # by Scripture alone (sola scriptura)
    "بالمسيح وحده",  # by Christ alone (solus christus)
    "عصمة الكتاب المقدس",  # inerrancy of Scripture
]

def has_arabic_scripture_citation(answer: str) -> bool:
    """Return True if the answer contains at least one Arabic Scripture citation."""
    return bool(_ARABIC_SCRIPTURE_RE.search(answer))


def has_arabic_theological_terminology(answer: str) -> bool:
    """Return True if the answer uses at least one recognized Arabic theological term."""
    return any(term in answer for term in _ARABIC_THEOLOGICAL_TERMS)


def calibrate_arabic_scores(question: str, answer: str, result_dict: dict) -> dict:
    """Cap inflated Adherence/Kindness/Arabic_Accuracy scores when evidence is absent.

    Arabic localization of the English ceiling-compression calibration.
    """
    adherence = result_dict.get("Adherence", {})
    kindness = result_dict.get("Kindness_and_Gentleness", {})
    arabic_accuracy = result_dict.get("Arabic_Accuracy", {})

    # Biblical_Basis > 3 but no Arabic Scripture citation -> cap at 3
    if (
        isinstance(adherence.get("Biblical_Basis"), int)
        and adherence["Biblical_Basis"] > 3
        and not has_arabic_scripture_citation(answer)
    ):
        adherence["Biblical_Basis"] = 3

    # Core > 4 but no Arabic theological terminology -> cap at 4
    if (
        isinstance(adherence.get("Core"), int)
        and adherence["Core"] > 4
        and not has_arabic_theological_terminology(answer)
    ):
        adherence["Core"] = 4

    # Theological_Nuance > 3 but no theological terminology -> cap at 3
    if (
        isinstance(arabic_accuracy.get("Theological_Nuance"), int)
        and arabic_accuracy["Theological_Nuance"] > 3
        and not has_arabic_theological_terminology(answer)
    ):
        arabic_accuracy["Theological_Nuance"] = 3

    result_dict["Adherence"] = adherence
    result_dict["Kindness_and_Gentleness"] = kindness
    if arabic_accuracy:
        result_dict["Arabic_Accuracy"] = arabic_accuracy
    return result_dict
