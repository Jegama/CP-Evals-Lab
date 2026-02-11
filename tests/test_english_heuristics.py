"""Unit tests for English ceiling-compression heuristic functions.

Tests cover:
- has_scripture_citation()
- has_theological_terminology()
- has_pastoral_signals()
- calibrate_english_scores()
"""

import pytest
from parrot_ai.llm_evaluation import (
    has_scripture_citation,
    has_theological_terminology,
    has_pastoral_signals,
    calibrate_english_scores,
)


# ---------- has_scripture_citation ----------

class TestHasScriptureCitation:
    def test_standard_citation(self):
        assert has_scripture_citation("As we read in John 3:16, God so loved the world.")

    def test_numbered_book(self):
        assert has_scripture_citation("Paul writes in 1 Corinthians 15:3-4 about the resurrection.")

    def test_old_testament(self):
        assert has_scripture_citation("Genesis 1:1 tells us that God created the heavens.")

    def test_psalm_singular(self):
        assert has_scripture_citation("Psalm 23:1 reminds us the Lord is our shepherd.")

    def test_psalms_plural(self):
        assert has_scripture_citation("Psalms 119:105 says thy word is a lamp.")

    def test_verse_range_with_dash(self):
        assert has_scripture_citation("See Romans 8:28-30 for the golden chain.")

    def test_verse_range_with_en_dash(self):
        assert has_scripture_citation("See Romans 8:28\u201330 for the golden chain.")

    def test_no_citation(self):
        assert not has_scripture_citation("God loves us and sent His son to save us.")

    def test_vague_bible_reference(self):
        assert not has_scripture_citation("The Bible says that we should love one another.")

    def test_empty_string(self):
        assert not has_scripture_citation("")

    def test_partial_reference_no_verse(self):
        assert not has_scripture_citation("In the book of Romans, Paul explains justification.")

    def test_multiple_citations(self):
        text = "John 14:6 and Acts 4:12 both affirm Christ's exclusivity."
        assert has_scripture_citation(text)

    def test_three_numbered_book(self):
        assert has_scripture_citation("3 John 1:4 speaks of walking in truth.")


# ---------- has_theological_terminology ----------

class TestHasTheologicalTerminology:
    def test_substitutionary_atonement(self):
        assert has_theological_terminology("Christ's substitutionary atonement saves us.")

    def test_justification_by_faith(self):
        assert has_theological_terminology("We are saved through justification by faith alone.")

    def test_sola_fide(self):
        assert has_theological_terminology("The Reformation principle of sola fide is central.")

    def test_hypostatic_union(self):
        assert has_theological_terminology("The hypostatic union describes Christ's two natures.")

    def test_total_depravity(self):
        assert has_theological_terminology("Total depravity means every part of us is affected by sin.")

    def test_case_insensitive(self):
        assert has_theological_terminology("The TRINITY is a core Christian doctrine.")

    def test_generic_language_no_terms(self):
        assert not has_theological_terminology("Jesus died for us and rose again. We should trust God.")

    def test_empty_string(self):
        assert not has_theological_terminology("")

    def test_close_but_not_matching(self):
        # "atonement" alone is not in the list (needs "substitutionary atonement" etc.)
        assert not has_theological_terminology("The atonement of Christ is important.")

    def test_sovereignty_of_god(self):
        assert has_theological_terminology("We trust in the sovereignty of God over all things.")


# ---------- has_pastoral_signals ----------

class TestHasPastoralSignals:
    def test_two_signals(self):
        text = "I understand your concern. The good news is that Christ offers hope."
        assert has_pastoral_signals(text)

    def test_single_signal_not_enough(self):
        text = "I understand your concern. Here is the doctrinal answer."
        assert not has_pastoral_signals(text)

    def test_three_signals(self):
        text = "That's a great question. I want to encourage you. There is hope in Christ."
        assert has_pastoral_signals(text)

    def test_no_signals(self):
        text = "The doctrine of election is found in Ephesians 1. God chose us before the foundation."
        assert not has_pastoral_signals(text)

    def test_empty_string(self):
        assert not has_pastoral_signals("")

    def test_case_insensitive(self):
        text = "I Understand your pain. Take Heart, for God is faithful."
        assert has_pastoral_signals(text)

    def test_pastoral_with_god_cares(self):
        text = "God cares about you deeply. Be encouraged, He is near."
        assert has_pastoral_signals(text)


# ---------- calibrate_english_scores ----------

def _base_result() -> dict:
    """Return a baseline result dict with all 5s for testing caps."""
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
    }


class TestCalibrateEnglishScores:
    def test_biblical_basis_capped_when_no_citation(self):
        result = _base_result()
        answer = "God loves us and Jesus died for our sins. Trust in Him."
        result = calibrate_english_scores("What is the Gospel?", answer, result)
        assert result["Adherence"]["Biblical_Basis"] == 3

    def test_biblical_basis_not_capped_with_citation(self):
        result = _base_result()
        answer = "As John 3:16 says, God so loved the world. Romans 5:8 confirms this."
        result = calibrate_english_scores("What is the Gospel?", answer, result)
        assert result["Adherence"]["Biblical_Basis"] == 5

    def test_biblical_basis_not_capped_when_already_3(self):
        result = _base_result()
        result["Adherence"]["Biblical_Basis"] = 3
        answer = "God loves us and sent His son."
        result = calibrate_english_scores("What is salvation?", answer, result)
        assert result["Adherence"]["Biblical_Basis"] == 3

    def test_core_capped_when_no_theology(self):
        result = _base_result()
        answer = "Jesus died for us and we should believe in Him. God is good."
        result = calibrate_english_scores("What is atonement?", answer, result)
        assert result["Adherence"]["Core"] == 4

    def test_core_not_capped_with_theology(self):
        result = _base_result()
        answer = "The doctrine of substitutionary atonement teaches that Christ bore our penalty."
        result = calibrate_english_scores("What is atonement?", answer, result)
        assert result["Adherence"]["Core"] == 5

    def test_core_not_capped_when_already_4(self):
        result = _base_result()
        result["Adherence"]["Core"] = 4
        answer = "Jesus died for us."
        result = calibrate_english_scores("What is salvation?", answer, result)
        assert result["Adherence"]["Core"] == 4

    def test_pastoral_capped_when_no_signals(self):
        result = _base_result()
        answer = "Election means God chose His people before the foundation of the world."
        result = calibrate_english_scores("What is election?", answer, result)
        assert result["Kindness_and_Gentleness"]["Pastoral_Sensitivity"] == 3

    def test_pastoral_not_capped_with_signals(self):
        result = _base_result()
        answer = "I understand this can be difficult. The good news is that God's grace covers us."
        result = calibrate_english_scores("Am I saved?", answer, result)
        assert result["Kindness_and_Gentleness"]["Pastoral_Sensitivity"] == 5

    def test_pastoral_not_capped_when_already_3(self):
        result = _base_result()
        result["Kindness_and_Gentleness"]["Pastoral_Sensitivity"] = 3
        answer = "Election is a doctrine about God's sovereignty."
        result = calibrate_english_scores("What is election?", answer, result)
        assert result["Kindness_and_Gentleness"]["Pastoral_Sensitivity"] == 3

    def test_all_caps_applied_together(self):
        """Answer with no citations, no theology, no pastoral signals gets all three caps."""
        result = _base_result()
        answer = "God loves everyone and wants us to be good people and follow Him."
        result = calibrate_english_scores("What should I believe?", answer, result)
        assert result["Adherence"]["Biblical_Basis"] == 3
        assert result["Adherence"]["Core"] == 4
        assert result["Kindness_and_Gentleness"]["Pastoral_Sensitivity"] == 3

    def test_excellent_answer_no_caps(self):
        """A genuinely excellent answer should not be capped."""
        result = _base_result()
        answer = (
            "I understand your question, and it's a great one. "
            "The doctrine of justification by faith teaches that we are declared "
            "righteous before God not by our works, but through faith in Christ alone. "
            "As Romans 3:28 says, 'a person is justified by faith apart from works of the law.' "
            "Paul also affirms in Ephesians 2:8-9 that salvation is by grace through faith. "
            "The good news is that this gift is freely offered to all who trust in Christ."
        )
        result = calibrate_english_scores("How are we justified?", answer, result)
        assert result["Adherence"]["Biblical_Basis"] == 5
        assert result["Adherence"]["Core"] == 5
        assert result["Kindness_and_Gentleness"]["Pastoral_Sensitivity"] == 5

    def test_interfaith_untouched(self):
        """Calibration should not affect Interfaith_Sensitivity scores."""
        result = _base_result()
        answer = "God loves us."
        result = calibrate_english_scores("What is faith?", answer, result)
        assert result["Interfaith_Sensitivity"]["Evangelism"] == 5
        assert result["Interfaith_Sensitivity"]["Gospel_Boldness"] == 5

    def test_missing_sections_no_error(self):
        """Should handle missing sections gracefully."""
        result = {}
        answer = "Some answer."
        result = calibrate_english_scores("Some question?", answer, result)
        # Should not raise; sections created as empty dicts
        assert result.get("Adherence") == {}
        assert result.get("Kindness_and_Gentleness") == {}
