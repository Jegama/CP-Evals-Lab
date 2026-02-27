"""Unit tests for Arabic ceiling-compression heuristic functions.

Tests cover:
- has_arabic_scripture_citation()
- has_arabic_theological_terminology()
- calibrate_arabic_scores()
"""

import pytest
from parrot_ai.llm_evaluation import (
    has_arabic_scripture_citation,
    has_arabic_theological_terminology,
    calibrate_arabic_scores,
)


# ---------- has_arabic_scripture_citation ----------

class TestHasArabicScriptureCitation:
    def test_standard_citation_john(self):
        assert has_arabic_scripture_citation("كما نقرأ في يوحنا 3:16 أن الله أحب العالم")

    def test_romans_citation(self):
        assert has_arabic_scripture_citation("رومية 5:8 تؤكد محبة الله لنا")

    def test_genesis_citation(self):
        assert has_arabic_scripture_citation("تكوين 1:1 في البدء خلق الله السماوات والأرض")

    def test_psalm_citation(self):
        assert has_arabic_scripture_citation("مزمور 23:1 الرب راعيّ فلا يعوزني شيء")

    def test_acts_citation(self):
        assert has_arabic_scripture_citation("أعمال الرسل 4:12 ليس بأحد غيره الخلاص")

    def test_verse_range(self):
        assert has_arabic_scripture_citation("رومية 8:28-30 سلسلة الخلاص الذهبية")

    def test_eastern_arabic_numerals(self):
        assert has_arabic_scripture_citation("يوحنا ٣:١٦ هكذا أحب الله العالم")

    def test_corinthians_citation(self):
        assert has_arabic_scripture_citation("كورنثوس 15:3 المسيح مات من أجل خطايانا")

    def test_no_citation(self):
        assert not has_arabic_scripture_citation("الله يحبنا وأرسل ابنه لخلاصنا")

    def test_vague_reference(self):
        assert not has_arabic_scripture_citation("يقول الكتاب المقدس أننا يجب أن نحب بعضنا")

    def test_empty_string(self):
        assert not has_arabic_scripture_citation("")

    def test_book_name_without_verse(self):
        assert not has_arabic_scripture_citation("في رسالة رومية يشرح بولس التبرير")


# ---------- has_arabic_theological_terminology ----------

class TestHasArabicTheologicalTerminology:
    def test_substitutionary_atonement(self):
        assert has_arabic_theological_terminology("الكفارة البدلية هي أساس الإنجيل")

    def test_justification_by_faith(self):
        assert has_arabic_theological_terminology("نخلص بالتبرير بالإيمان وحده")

    def test_hypostatic_union(self):
        assert has_arabic_theological_terminology("الاتحاد الأقنومي يصف طبيعتي المسيح")

    def test_trinity(self):
        assert has_arabic_theological_terminology("الثالوث هو عقيدة مسيحية أساسية")

    def test_holy_trinity(self):
        assert has_arabic_theological_terminology("الثالوث الأقدس إله واحد في ثلاثة أقانيم")

    def test_deity_of_christ(self):
        assert has_arabic_theological_terminology("ألوهية المسيح حقيقة كتابية")

    def test_total_depravity(self):
        assert has_arabic_theological_terminology("الفساد الكلي يعني أن كل جزء منا متأثر بالخطيئة")

    def test_sola_fide_arabic(self):
        assert has_arabic_theological_terminology("بالإيمان وحده ننال الخلاص")

    def test_sanctification(self):
        assert has_arabic_theological_terminology("التقديس عملية مستمرة بالروح القدس")

    def test_generic_no_terms(self):
        assert not has_arabic_theological_terminology("يسوع مات من أجلنا وقام ثانية يجب أن نثق بالله")

    def test_empty_string(self):
        assert not has_arabic_theological_terminology("")


# ---------- calibrate_arabic_scores ----------

def _base_arabic_result() -> dict:
    """Return a baseline Arabic result dict with all 5s for testing caps."""
    return {
        "Adherence": {
            "Core": 5,
            "Secondary": 5,
            "Tertiary_Handling": 5,
            "Biblical_Basis": 5,
            "Consistency": 5,
            "Overall": 5,
        },
        "Kindness_and_Gentleness": {
            "Core_Clarity_with_Kindness": 5,
            "Pastoral_Sensitivity": 5,
            "Secondary_Fairness": 5,
            "Tertiary_Neutrality": 5,
            "Tone": 5,
            "Overall": 5,
        },
        "Interfaith_Sensitivity": {
            "Respect_and_Handling_Objections": 5,
            "Objection_Acknowledgement": 5,
            "Evangelism": 5,
            "Gospel_Boldness": 5,
            "Overall": 5,
        },
        "Arabic_Accuracy": {
            "Grammar_and_Syntax": 5,
            "Theological_Nuance": 5,
            "Contextual_Clarity": 5,
            "Consistency_of_Terms": 5,
            "Arabic_Purity": 5,
            "Overall": 5,
        },
    }


class TestCalibrateArabicScores:
    def test_biblical_basis_capped_when_no_citation(self):
        result = _base_arabic_result()
        answer = "الله يحبنا ويسوع مات من أجل خطايانا. ثق به"
        result = calibrate_arabic_scores("ما هو الإنجيل؟", answer, result)
        assert result["Adherence"]["Biblical_Basis"] == 3

    def test_biblical_basis_not_capped_with_citation(self):
        result = _base_arabic_result()
        answer = "كما يقول يوحنا 3:16 هكذا أحب الله العالم. ورومية 5:8 تؤكد ذلك"
        result = calibrate_arabic_scores("ما هو الإنجيل؟", answer, result)
        assert result["Adherence"]["Biblical_Basis"] == 5

    def test_biblical_basis_not_capped_when_already_3(self):
        result = _base_arabic_result()
        result["Adherence"]["Biblical_Basis"] = 3
        answer = "الله يحبنا وأرسل ابنه"
        result = calibrate_arabic_scores("ما هو الخلاص؟", answer, result)
        assert result["Adherence"]["Biblical_Basis"] == 3

    def test_core_capped_when_no_theology(self):
        result = _base_arabic_result()
        answer = "يسوع مات من أجلنا ويجب أن نؤمن به. الله صالح"
        result = calibrate_arabic_scores("ما هي الكفارة؟", answer, result)
        assert result["Adherence"]["Core"] == 4

    def test_core_not_capped_with_theology(self):
        result = _base_arabic_result()
        answer = "عقيدة الكفارة البدلية تعلّم أن المسيح حمل عقوبتنا"
        result = calibrate_arabic_scores("ما هي الكفارة؟", answer, result)
        assert result["Adherence"]["Core"] == 5

    def test_theological_nuance_capped_when_no_terms(self):
        result = _base_arabic_result()
        answer = "يسوع هو الله وهو إنسان أيضاً. هذا ما يعلّمه الكتاب المقدس"
        result = calibrate_arabic_scores("من هو المسيح؟", answer, result)
        assert result["Arabic_Accuracy"]["Theological_Nuance"] == 3

    def test_theological_nuance_not_capped_with_terms(self):
        result = _base_arabic_result()
        answer = "الاتحاد الأقنومي يعلّم أن المسيح له طبيعتان كاملتان إلهية وبشرية"
        result = calibrate_arabic_scores("من هو المسيح؟", answer, result)
        assert result["Arabic_Accuracy"]["Theological_Nuance"] == 5

    def test_all_caps_applied_together(self):
        result = _base_arabic_result()
        answer = "الله يحب الجميع ويريدنا أن نكون صالحين ونتبعه"
        result = calibrate_arabic_scores("ماذا يجب أن أؤمن؟", answer, result)
        assert result["Adherence"]["Biblical_Basis"] == 3
        assert result["Adherence"]["Core"] == 4
        assert result["Arabic_Accuracy"]["Theological_Nuance"] == 3

    def test_excellent_answer_no_caps(self):
        result = _base_arabic_result()
        answer = (
            "أفهم سؤالك وهو سؤال مهم جداً. "
            "عقيدة التبرير بالإيمان تعلّم أننا نُعلَن أبراراً أمام الله "
            "ليس بأعمالنا بل بالإيمان بالمسيح وحده. "
            "كما يقول رومية 3:28 أن الإنسان يتبرر بالإيمان بدون أعمال الناموس. "
            "ويؤكد أفسس 2:8 أن الخلاص بالنعمة بالإيمان. "
            "الخبر السار أن هذه العطية مقدمة مجاناً لكل من يثق بالمسيح."
        )
        result = calibrate_arabic_scores("كيف نتبرر؟", answer, result)
        assert result["Adherence"]["Biblical_Basis"] == 5
        assert result["Adherence"]["Core"] == 5
        assert result["Kindness_and_Gentleness"]["Pastoral_Sensitivity"] == 5
        assert result["Arabic_Accuracy"]["Theological_Nuance"] == 5

    def test_interfaith_untouched(self):
        result = _base_arabic_result()
        answer = "الله يحبنا"
        result = calibrate_arabic_scores("ما هو الإيمان؟", answer, result)
        assert result["Interfaith_Sensitivity"]["Evangelism"] == 5
        assert result["Interfaith_Sensitivity"]["Gospel_Boldness"] == 5

    def test_missing_sections_no_error(self):
        result = {}
        answer = "إجابة ما"
        result = calibrate_arabic_scores("سؤال ما؟", answer, result)
        assert result.get("Adherence") == {}
        assert result.get("Kindness_and_Gentleness") == {}
