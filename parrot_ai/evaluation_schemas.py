"""Pydantic schema models for evaluation results."""

from __future__ import annotations
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


# ---------------- Question Tagging Schemas (English selective scoring) ---------------- #


class DoctrineTier(str, Enum):
    CORE = "core"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    NOT_DIRECTLY_DOCTRINAL = "not_directly_doctrinal"


class QuestionType(str, Enum):
    FACTUAL_HISTORICAL = "factual_historical"
    DOCTRINAL = "doctrinal"
    APOLOGETIC_INTERFAITH = "apologetic_interfaith"
    PASTORAL = "pastoral"
    PRACTICAL_ETHICAL = "practical_ethical"
    COMPARATIVE_RELIGION = "comparative_religion"
    METHODOLOGICAL = "methodological"
    BIBLE_SURVEY = "bible_survey"


class QuestionTag(BaseModel):
    question: str
    doctrine_tier: DoctrineTier
    question_type: QuestionType
    applies_core_doctrine: bool
    applies_secondary_doctrine: bool
    applies_tertiary_handling: bool
    applies_pastoral: bool
    applies_interfaith: bool
    applies_evangelism: bool
    reason: str


class QuestionTagSet(BaseModel):
    tags: List[QuestionTag]
    classification_model: str
    classification_timestamp: str
    version: str = "1.0"


# Flag-to-subcriteria mapping: each flag controls which (section, subcriteria)
# pairs are applicable. Subcriteria NOT listed here are always scored.
SUBCRITERIA_FLAG_MAP: dict[str, list[tuple[str, str]]] = {
    "applies_core_doctrine": [
        ("Adherence", "Core"),
        ("Kindness_and_Gentleness", "Core_Clarity_with_Kindness"),
    ],
    "applies_secondary_doctrine": [
        ("Adherence", "Secondary"),
        ("Kindness_and_Gentleness", "Secondary_Fairness"),
    ],
    "applies_tertiary_handling": [
        ("Adherence", "Tertiary_Handling"),
        ("Kindness_and_Gentleness", "Tertiary_Neutrality"),
    ],
    "applies_pastoral": [
        ("Kindness_and_Gentleness", "Pastoral_Sensitivity"),
    ],
    "applies_interfaith": [
        ("Interfaith_Sensitivity", "Respect_and_Handling_Objections"),
        ("Interfaith_Sensitivity", "Objection_Acknowledgement"),
    ],
    "applies_evangelism": [
        ("Interfaith_Sensitivity", "Evangelism"),
        ("Interfaith_Sensitivity", "Gospel_Boldness"),
    ],
}

# Subcriteria that are always scored regardless of flags
ALWAYS_ON_SUBCRITERIA: set[tuple[str, str]] = {
    ("Adherence", "Biblical_Basis"),
    ("Adherence", "Consistency"),
    ("Kindness_and_Gentleness", "Tone"),
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
    Pastoral_Acknowledgement: Optional[str] = None  # "yes" | "partial" | "no"


class InterfaithSensitivityModel(BaseModel):
    Respect_and_Handling_Objections: int
    Objection_Acknowledgement: int
    Evangelism: int
    Gospel_Boldness: int
    Overall: int


class ArabicAccuracyDetailed(BaseModel):  # Arabic only
    Grammar_and_Syntax: int
    Theological_Nuance: int
    Contextual_Clarity: int
    Consistency_of_Terms: int
    Arabic_Purity: int
    Penalty_Reason: Optional[str] = None
    Overall: int


class EvaluationResultArabic(BaseModel):
    Adherence: AdherenceModel
    Kindness_and_Gentleness: KindnessGentlenessModel
    Interfaith_Sensitivity: InterfaithSensitivityModel
    Arabic_Accuracy: ArabicAccuracyDetailed


class EvaluationResultEnglish(BaseModel):
    Adherence: AdherenceModel
    Kindness_and_Gentleness: KindnessGentlenessModel
    Interfaith_Sensitivity: InterfaithSensitivityModel


# ---------------- Sermon Evaluation Schemas (language-agnostic) ---------------- #


class SermonPoint(BaseModel):
    Point: str
    Verses: Optional[str] = None
    Summary: str
    Subpoints: List[str] = Field(default_factory=list)
    Illustrations: List[str] = Field(default_factory=list)
    Application: List[str] = Field(default_factory=list)
    Comments: Optional[str] = None
    Feedback: Optional[str] = None


class SermonGeneralComments(BaseModel):
    Content_Comments: Optional[str] = None
    Structure_Comments: Optional[str] = None
    Explanation_Comments: Optional[str] = None


class SermonFCF(BaseModel):
    FCF: str
    Comments: Optional[str] = None


class SermonExtractionStep1(BaseModel):
    """Step 1 – Descriptive Extraction (Structural Analysis)."""

    Scripture_Introduction: str
    Sermon_Introduction: str
    Proposition: str
    Body: List[SermonPoint]
    Conclusion: str
    General_Comments: SermonGeneralComments
    Fallen_Condition_Focus: SermonFCF
    Extraction_Confidence: float
    audio_duration: Optional[float] = (
        None  # Duration in seconds (populated for audio mode)
    )


class IntroductionScores(BaseModel):
    FCF_Introduced: int
    Arouses_Attention: int
    Overall: int
    Feedback: Optional[str] = None


class PropositionScores(BaseModel):
    Principle_and_Application_Wed: int
    Establishes_Main_Theme: int
    Summarizes_Introduction: int
    Overall: int
    Feedback: Optional[str] = None


class MainPointsScores(BaseModel):
    Clarity: int
    Hortatory_Universal_Truths: int
    Proportional_and_Coexistent: int
    Exposition_Quality: int
    Illustration_Quality: int
    Application_Quality: int
    Overall: int
    Feedback: Optional[str] = None


class ExegeticalSupportScores(BaseModel):
    Alignment_with_Text: int
    Handles_Difficulties: int
    Proof_Accuracy_and_Clarity: int
    Context_and_Genre_Considered: int
    Not_Belabored: int
    Aids_Rather_Than_Impresses: int
    Overall: int
    Feedback: Optional[str] = None


class ApplicationScores(BaseModel):
    Clear_and_Practical: int
    Redemptive_Focus: int
    Mandate_vs_Idea_Distinction: int
    Passage_Supported: int
    Overall: int
    Feedback: Optional[str] = None


class IllustrationsScores(BaseModel):
    Lived_Body_Detail: int
    Strengthens_Points: int
    Proportion: int
    Overall: int
    Feedback: Optional[str] = None


class ConclusionScores(BaseModel):
    Summary: int
    Compelling_Exhortation: int
    Climax: int
    Pointed_End: int
    Overall: int
    Feedback: Optional[str] = None


class AggregatedSummary(BaseModel):
    """Computed rollups per Sermon Evaluation Framework.

    All values are 1–5 floats rounded to two decimals.
    """

    Textual_Fidelity: float
    Proposition_Clarity: float
    Introduction: float
    Application_Effectiveness: float
    Structure_Cohesion: float
    Illustrations: float
    Overall_Impact_Base: float
    Overall_Impact: float
    duration_penalty: Optional[float] = (
        None  # Penalty applied for <35min or >50min (0.0-1.0)
    )


class AggregatedSummaryFeedback(BaseModel):
    """Feedback for the aggregated summary scores."""

    Textual_Fidelity: Optional[str] = None
    Proposition_Clarity: Optional[str] = None
    Introduction: Optional[str] = None
    Application_Effectiveness: Optional[str] = None
    Structure_Cohesion: Optional[str] = None
    Illustrations: Optional[str] = None
    Overall_Impact: Optional[str] = None


class SermonScoringStep2(BaseModel):
    """Step 2 – Analytical Scoring (Synthesis & Coaching)."""

    Introduction: IntroductionScores
    Proposition: PropositionScores
    Main_Points: MainPointsScores
    Exegetical_Support: ExegeticalSupportScores
    Application: ApplicationScores
    Illustrations: IllustrationsScores
    Conclusion: ConclusionScores
    Strengths: List[str] = Field(default_factory=list)
    Growth_Areas: List[str] = Field(default_factory=list)
    Next_Steps: List[str] = Field(default_factory=list)
    Scoring_Confidence: float
    Aggregated_Summary: Optional[AggregatedSummary] = None
    Aggregated_Summary_Feedback: Optional[AggregatedSummaryFeedback] = None


class SermonScoringStep2Raw(BaseModel):
    """Model used for LLM output only (no aggregates)."""

    Introduction: IntroductionScores
    Proposition: PropositionScores
    Main_Points: MainPointsScores
    Exegetical_Support: ExegeticalSupportScores
    Application: ApplicationScores
    Illustrations: IllustrationsScores
    Conclusion: ConclusionScores
    Strengths: List[str] = Field(default_factory=list)
    Growth_Areas: List[str] = Field(default_factory=list)
    Next_Steps: List[str] = Field(default_factory=list)
    Scoring_Confidence: float


__all__ = [
    # Question tagging
    "DoctrineTier",
    "QuestionType",
    "QuestionTag",
    "QuestionTagSet",
    "SUBCRITERIA_FLAG_MAP",
    "ALWAYS_ON_SUBCRITERIA",
    # Evaluation result models
    "AdherenceModel",
    "KindnessGentlenessModel",
    "InterfaithSensitivityModel",
    "ArabicAccuracyDetailed",
    "EvaluationResultArabic",
    "EvaluationResultEnglish",
    # Sermon
    "SermonPoint",
    "SermonGeneralComments",
    "SermonFCF",
    "SermonExtractionStep1",
    "IntroductionScores",
    "PropositionScores",
    "MainPointsScores",
    "ExegeticalSupportScores",
    "ApplicationScores",
    "IllustrationsScores",
    "ConclusionScores",
    "SermonScoringStep2",
    "SermonScoringStep2Raw",
    "AggregatedSummary",
    "AggregatedSummaryFeedback",
]
