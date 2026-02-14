"""Unit tests for question tagging schema, flag mapping, and selective aggregation.

Tests cover:
- QuestionTag schema validation
- SUBCRITERIA_FLAG_MAP correctness
- Selective aggregation with tag-aware scoring
- Section Overall recomputation from applicable subcriteria
- Backward compatibility (question_tags=None)
- Edge cases (missing question in tags, all flags false, empty results)
"""

import pytest
from parrot_ai.evaluation_schemas import (
    DoctrineTier,
    QuestionType,
    QuestionTag,
    QuestionTagSet,
    SUBCRITERIA_FLAG_MAP,
    ALWAYS_ON_SUBCRITERIA,
)

# Import aggregate_scores from the CLI module
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cp_eval_llms import aggregate_scores


# ---------- Schema Validation ----------


class TestQuestionTagSchema:
    def test_valid_tag(self):
        tag = QuestionTag(
            question="What is the Trinity?",
            doctrine_tier=DoctrineTier.CORE,
            question_type=QuestionType.DOCTRINAL,
            applies_core_doctrine=True,
            applies_secondary_doctrine=False,
            applies_tertiary_handling=False,
            applies_pastoral=False,
            applies_interfaith=True,
            applies_evangelism=True,
            reason="Core doctrinal question about the Trinity.",
        )
        assert tag.doctrine_tier == DoctrineTier.CORE
        assert tag.applies_core_doctrine is True

    def test_factual_question_all_false(self):
        tag = QuestionTag(
            question="Who was Herod Agrippa I?",
            doctrine_tier=DoctrineTier.NOT_DIRECTLY_DOCTRINAL,
            question_type=QuestionType.FACTUAL_HISTORICAL,
            applies_core_doctrine=False,
            applies_secondary_doctrine=False,
            applies_tertiary_handling=False,
            applies_pastoral=False,
            applies_interfaith=False,
            applies_evangelism=False,
            reason="Purely factual/historical question.",
        )
        assert tag.question_type == QuestionType.FACTUAL_HISTORICAL

    def test_invalid_doctrine_tier(self):
        with pytest.raises(ValueError):
            QuestionTag(
                question="Test",
                doctrine_tier="invalid",
                question_type=QuestionType.DOCTRINAL,
                applies_core_doctrine=True,
                applies_secondary_doctrine=False,
                applies_tertiary_handling=False,
                applies_pastoral=False,
                applies_interfaith=False,
                applies_evangelism=False,
                reason="Test",
            )

    def test_invalid_question_type(self):
        with pytest.raises(ValueError):
            QuestionTag(
                question="Test",
                doctrine_tier=DoctrineTier.CORE,
                question_type="invalid_type",
                applies_core_doctrine=True,
                applies_secondary_doctrine=False,
                applies_tertiary_handling=False,
                applies_pastoral=False,
                applies_interfaith=False,
                applies_evangelism=False,
                reason="Test",
            )

    def test_tag_set_version(self):
        tag_set = QuestionTagSet(
            tags=[],
            classification_model="gpt-5-mini",
            classification_timestamp="2025-01-01T00:00:00",
        )
        assert tag_set.version == "1.0"


# ---------- Flag Mapping ----------


class TestFlagMapping:
    def test_all_flags_present(self):
        expected_flags = {
            "applies_core_doctrine",
            "applies_secondary_doctrine",
            "applies_tertiary_handling",
            "applies_pastoral",
            "applies_interfaith",
            "applies_evangelism",
        }
        assert set(SUBCRITERIA_FLAG_MAP.keys()) == expected_flags

    def test_core_doctrine_maps_to_correct_subcriteria(self):
        pairs = SUBCRITERIA_FLAG_MAP["applies_core_doctrine"]
        assert ("Adherence", "Core") in pairs
        assert ("Kindness_and_Gentleness", "Core_Clarity_with_Kindness") in pairs

    def test_secondary_doctrine_maps_correctly(self):
        pairs = SUBCRITERIA_FLAG_MAP["applies_secondary_doctrine"]
        assert ("Adherence", "Secondary") in pairs
        assert ("Kindness_and_Gentleness", "Secondary_Fairness") in pairs

    def test_tertiary_handling_maps_correctly(self):
        pairs = SUBCRITERIA_FLAG_MAP["applies_tertiary_handling"]
        assert ("Adherence", "Tertiary_Handling") in pairs
        assert ("Kindness_and_Gentleness", "Tertiary_Neutrality") in pairs

    def test_pastoral_maps_correctly(self):
        pairs = SUBCRITERIA_FLAG_MAP["applies_pastoral"]
        assert ("Kindness_and_Gentleness", "Pastoral_Sensitivity") in pairs

    def test_interfaith_maps_correctly(self):
        pairs = SUBCRITERIA_FLAG_MAP["applies_interfaith"]
        assert ("Interfaith_Sensitivity", "Respect_and_Handling_Objections") in pairs
        assert ("Interfaith_Sensitivity", "Objection_Acknowledgement") in pairs

    def test_evangelism_maps_correctly(self):
        pairs = SUBCRITERIA_FLAG_MAP["applies_evangelism"]
        assert ("Interfaith_Sensitivity", "Evangelism") in pairs
        assert ("Interfaith_Sensitivity", "Gospel_Boldness") in pairs

    def test_always_on_subcriteria(self):
        assert ("Adherence", "Biblical_Basis") in ALWAYS_ON_SUBCRITERIA
        assert ("Adherence", "Consistency") in ALWAYS_ON_SUBCRITERIA
        assert ("Kindness_and_Gentleness", "Tone") in ALWAYS_ON_SUBCRITERIA

    def test_no_overlap_between_flags_and_always_on(self):
        """No subcriteria should be both flag-controlled and always-on."""
        flag_controlled = set()
        for pairs in SUBCRITERIA_FLAG_MAP.values():
            for pair in pairs:
                flag_controlled.add(pair)
        overlap = flag_controlled & ALWAYS_ON_SUBCRITERIA
        assert overlap == set(), f"Overlap found: {overlap}"


# ---------- Helpers for Aggregation Tests ----------


def _make_eval_result(
    question: str,
    adherence: dict,
    kindness: dict,
    interfaith: dict,
) -> dict:
    """Build a single evaluation result item."""
    return {
        "question": question,
        "evaluation": {
            "Adherence": adherence,
            "Kindness_and_Gentleness": kindness,
            "Interfaith_Sensitivity": interfaith,
        },
    }


def _full_scores(core=4, secondary=4, tertiary=4, biblical=4, consistency=4,
                 adh_overall=4, core_kindness=4, pastoral=4, sec_fair=4,
                 tert_neutral=4, tone=4, kind_overall=4, respect=4,
                 objection=4, evangelism=4, boldness=4, inter_overall=4):
    """Create a complete evaluation result with specified scores."""
    return {
        "Adherence": {
            "Core": core, "Secondary": secondary, "Tertiary_Handling": tertiary,
            "Biblical_Basis": biblical, "Consistency": consistency, "Overall": adh_overall,
        },
        "Kindness_and_Gentleness": {
            "Core_Clarity_with_Kindness": core_kindness, "Pastoral_Sensitivity": pastoral,
            "Secondary_Fairness": sec_fair, "Tertiary_Neutrality": tert_neutral,
            "Tone": tone, "Overall": kind_overall,
        },
        "Interfaith_Sensitivity": {
            "Respect_and_Handling_Objections": respect, "Objection_Acknowledgement": objection,
            "Evangelism": evangelism, "Gospel_Boldness": boldness, "Overall": inter_overall,
        },
    }


def _factual_tag():
    """Tag for a factual question: all flags false."""
    return {
        "question": "Who was Herod Agrippa I?",
        "applies_core_doctrine": False,
        "applies_secondary_doctrine": False,
        "applies_tertiary_handling": False,
        "applies_pastoral": False,
        "applies_interfaith": False,
        "applies_evangelism": False,
    }


def _apologetic_tag():
    """Tag for an apologetic question: interfaith + evangelism true."""
    return {
        "question": "How should Christians respond to Islam?",
        "applies_core_doctrine": True,
        "applies_secondary_doctrine": False,
        "applies_tertiary_handling": False,
        "applies_pastoral": True,
        "applies_interfaith": True,
        "applies_evangelism": True,
    }


def _core_doctrinal_tag():
    """Tag for a core doctrinal question."""
    return {
        "question": "What is the Trinity?",
        "applies_core_doctrine": True,
        "applies_secondary_doctrine": False,
        "applies_tertiary_handling": False,
        "applies_pastoral": False,
        "applies_interfaith": False,
        "applies_evangelism": False,
    }


# ---------- Selective Aggregation ----------


class TestSelectiveAggregation:
    def test_factual_only_contributes_always_on(self):
        """A factual question (all flags false) should only contribute to
        Biblical_Basis, Consistency, and Tone."""
        results = [
            {
                "question": "Who was Herod Agrippa I?",
                "evaluation": _full_scores(
                    core=2, secondary=2, tertiary=2, biblical=5, consistency=5,
                    adh_overall=3, core_kindness=2, pastoral=2, sec_fair=2,
                    tert_neutral=2, tone=5, kind_overall=3, respect=2,
                    objection=2, evangelism=2, boldness=2, inter_overall=2,
                ),
            }
        ]
        tags = {"Who was Herod Agrippa I?": _factual_tag()}
        agg = aggregate_scores(results, include_arabic_accuracy=False, question_tags=tags)

        # Always-on scores should be present
        assert agg[("Adherence", "Biblical_Basis")] == 5.0
        assert agg[("Adherence", "Consistency")] == 5.0
        assert agg[("Kindness_and_Gentleness", "Tone")] == 5.0

        # Flag-controlled scores should NOT be present (all flags false)
        assert ("Adherence", "Core") not in agg
        assert ("Adherence", "Secondary") not in agg
        assert ("Adherence", "Tertiary_Handling") not in agg
        assert ("Kindness_and_Gentleness", "Core_Clarity_with_Kindness") not in agg
        assert ("Kindness_and_Gentleness", "Pastoral_Sensitivity") not in agg
        assert ("Interfaith_Sensitivity", "Evangelism") not in agg

    def test_core_doctrinal_includes_core_subcriteria(self):
        """A core doctrinal question contributes to Core + Core_Clarity_with_Kindness + always-on."""
        results = [
            {
                "question": "What is the Trinity?",
                "evaluation": _full_scores(core=5, core_kindness=5),
            }
        ]
        tags = {"What is the Trinity?": _core_doctrinal_tag()}
        agg = aggregate_scores(results, include_arabic_accuracy=False, question_tags=tags)

        assert agg[("Adherence", "Core")] == 5.0
        assert agg[("Kindness_and_Gentleness", "Core_Clarity_with_Kindness")] == 5.0
        assert agg[("Adherence", "Biblical_Basis")] == 4.0  # from default=4
        # Secondary should NOT be present (flag false)
        assert ("Adherence", "Secondary") not in agg

    def test_apologetic_includes_interfaith_and_evangelism(self):
        """An apologetic question contributes to interfaith + evangelism subcriteria."""
        results = [
            {
                "question": "How should Christians respond to Islam?",
                "evaluation": _full_scores(respect=5, objection=5, evangelism=5, boldness=5),
            }
        ]
        tags = {"How should Christians respond to Islam?": _apologetic_tag()}
        agg = aggregate_scores(results, include_arabic_accuracy=False, question_tags=tags)

        assert agg[("Interfaith_Sensitivity", "Respect_and_Handling_Objections")] == 5.0
        assert agg[("Interfaith_Sensitivity", "Objection_Acknowledgement")] == 5.0
        assert agg[("Interfaith_Sensitivity", "Evangelism")] == 5.0
        assert agg[("Interfaith_Sensitivity", "Gospel_Boldness")] == 5.0

    def test_mixed_questions_averages_only_applicable(self):
        """With 2 factual + 1 apologetic, Core average should only reflect the apologetic question."""
        factual_q1 = "Who was Herod Agrippa I?"
        factual_q2 = "When was the Council of Nicea?"
        apologetic_q = "How should Christians respond to Islam?"

        results = [
            {"question": factual_q1, "evaluation": _full_scores(core=2, biblical=5, consistency=5, tone=5)},
            {"question": factual_q2, "evaluation": _full_scores(core=2, biblical=4, consistency=4, tone=4)},
            {"question": apologetic_q, "evaluation": _full_scores(core=5, biblical=4, consistency=4, tone=4)},
        ]
        tags = {
            factual_q1: _factual_tag(),
            factual_q2: {**_factual_tag(), "question": factual_q2},
            apologetic_q: _apologetic_tag(),
        }
        agg = aggregate_scores(results, include_arabic_accuracy=False, question_tags=tags)

        # Core should only reflect the apologetic question (score=5)
        assert agg[("Adherence", "Core")] == 5.0
        # Biblical_Basis should reflect all 3 questions (5+4+4)/3
        assert agg[("Adherence", "Biblical_Basis")] == round((5 + 4 + 4) / 3, 2)


# ---------- Overall Recomputation ----------


class TestOverallRecomputation:
    def test_section_overall_recomputed_from_applicable(self):
        """Section Overall should be mean of applicable subcriteria means, not raw Overall."""
        results = [
            {
                "question": "Who was Herod Agrippa I?",
                "evaluation": _full_scores(
                    biblical=5, consistency=3, tone=4,
                    # These raw Overalls should be ignored when tags are present
                    adh_overall=1, kind_overall=1, inter_overall=1,
                ),
            }
        ]
        tags = {"Who was Herod Agrippa I?": _factual_tag()}
        agg = aggregate_scores(results, include_arabic_accuracy=False, question_tags=tags)

        # Adherence Overall = mean(Biblical_Basis=5, Consistency=3) = 4.0
        assert agg[("Adherence", "Overall")] == 4.0
        # Kindness Overall = mean(Tone=4) = 4.0
        assert agg[("Kindness_and_Gentleness", "Overall")] == 4.0
        # Interfaith has no applicable subcriteria -> should not be present
        assert ("Interfaith_Sensitivity", "Overall") not in agg

    def test_final_overall_from_recomputed_overalls(self):
        """Final_Overall should use the recomputed section Overalls."""
        results = [
            {
                "question": "What is the Trinity?",
                "evaluation": _full_scores(
                    core=5, biblical=4, consistency=4, core_kindness=5, tone=4,
                    adh_overall=1, kind_overall=1, inter_overall=1,
                ),
            }
        ]
        tags = {"What is the Trinity?": _core_doctrinal_tag()}
        agg = aggregate_scores(results, include_arabic_accuracy=False, question_tags=tags)

        # Adherence: mean(Core=5, Biblical_Basis=4, Consistency=4) = 4.33
        expected_adh = round((5 + 4 + 4) / 3, 2)
        assert agg[("Adherence", "Overall")] == expected_adh
        # Kindness: mean(Core_Clarity_with_Kindness=5, Tone=4) = 4.5
        expected_kind = round((5 + 4) / 2, 2)
        assert agg[("Kindness_and_Gentleness", "Overall")] == expected_kind


# ---------- Backward Compatibility ----------


class TestBackwardCompat:
    def test_no_tags_produces_same_output(self):
        """With question_tags=None, output should be identical to old behavior."""
        results = [
            {
                "question": "What is faith?",
                "evaluation": _full_scores(
                    core=4, secondary=3, tertiary=4, biblical=5, consistency=4,
                    adh_overall=4, core_kindness=4, pastoral=3, sec_fair=3,
                    tert_neutral=4, tone=4, kind_overall=4, respect=3,
                    objection=3, evangelism=4, boldness=4, inter_overall=4,
                ),
            }
        ]
        agg_no_tags = aggregate_scores(results, include_arabic_accuracy=False, question_tags=None)
        agg_empty = aggregate_scores(results, include_arabic_accuracy=False)

        assert agg_no_tags == agg_empty

        # All subcriteria should be present (no nulling)
        assert ("Adherence", "Core") in agg_no_tags
        assert ("Adherence", "Secondary") in agg_no_tags
        assert ("Adherence", "Overall") in agg_no_tags
        assert ("Interfaith_Sensitivity", "Evangelism") in agg_no_tags

    def test_all_scores_preserved_in_raw_output(self):
        """Without tags, every score from every section appears in the aggregate."""
        results = [
            {"question": "Q1", "evaluation": _full_scores()},
            {"question": "Q2", "evaluation": _full_scores()},
        ]
        agg = aggregate_scores(results, include_arabic_accuracy=False)
        # Should have all 18 subcriteria + 3 overalls + Final_Overall = 22
        # 5 adh + 5 kind + 4 inter + adh_overall + kind_overall + inter_overall + final = 18
        expected_keys = set()
        for section, subs in [
            ("Adherence", ["Core", "Secondary", "Tertiary_Handling", "Biblical_Basis", "Consistency", "Overall"]),
            ("Kindness_and_Gentleness", ["Core_Clarity_with_Kindness", "Pastoral_Sensitivity", "Secondary_Fairness", "Tertiary_Neutrality", "Tone", "Overall"]),
            ("Interfaith_Sensitivity", ["Respect_and_Handling_Objections", "Objection_Acknowledgement", "Evangelism", "Gospel_Boldness", "Overall"]),
        ]:
            for sub in subs:
                expected_keys.add((section, sub))
        expected_keys.add(("", "Final_Overall"))
        assert set(agg.keys()) == expected_keys


# ---------- Edge Cases ----------


class TestEdgeCases:
    def test_question_not_in_tags_includes_all_scores(self):
        """If a question is not found in the tags dict, all scores should be included (fail open)."""
        results = [
            {"question": "Unknown question?", "evaluation": _full_scores(core=3, evangelism=2)},
        ]
        tags = {"Some other question": _factual_tag()}
        agg = aggregate_scores(results, include_arabic_accuracy=False, question_tags=tags)

        # Should include all subcriteria since question wasn't found in tags
        assert ("Adherence", "Core") in agg
        assert ("Interfaith_Sensitivity", "Evangelism") in agg

    def test_empty_results(self):
        """Empty results list should produce empty aggregation."""
        agg = aggregate_scores([], include_arabic_accuracy=False, question_tags={})
        assert agg == {}

    def test_all_flags_false_for_all_questions(self):
        """All questions factual: only always-on subcriteria appear."""
        results = [
            {"question": "Q1", "evaluation": _full_scores(biblical=5, consistency=4, tone=4)},
            {"question": "Q2", "evaluation": _full_scores(biblical=3, consistency=5, tone=3)},
        ]
        tags = {
            "Q1": {**_factual_tag(), "question": "Q1"},
            "Q2": {**_factual_tag(), "question": "Q2"},
        }
        agg = aggregate_scores(results, include_arabic_accuracy=False, question_tags=tags)

        # Only always-on subcriteria + their section overalls
        assert ("Adherence", "Biblical_Basis") in agg
        assert ("Adherence", "Consistency") in agg
        assert ("Kindness_and_Gentleness", "Tone") in agg
        # Flag-controlled subcriteria absent
        assert ("Adherence", "Core") not in agg
        assert ("Interfaith_Sensitivity", "Evangelism") not in agg

    def test_missing_evaluation_key_skipped(self):
        """Items without 'evaluation' key should be skipped gracefully."""
        results = [
            {"question": "Q1"},
            {"question": "Q2", "evaluation": _full_scores()},
        ]
        agg = aggregate_scores(results, include_arabic_accuracy=False)
        # Should still work with the one valid result
        assert ("Adherence", "Core") in agg
