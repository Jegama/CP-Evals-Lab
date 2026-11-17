"""Shared minimum validation for accepting and reusing rubric evaluations."""

from parrot_ai.evaluation_schemas import (
    AdherenceModel,
    ArabicAccuracyDetailed,
    InterfaithSensitivityModel,
    KindnessGentlenessModel,
)
from parrot_ai.generation_failures import is_terminal_generation_failure


_RUBRIC_FIELDS = {
    section: tuple(
        name for name, field in model.model_fields.items() if field.annotation is int
    )
    for section, model in {
        "Adherence": AdherenceModel,
        "Kindness_and_Gentleness": KindnessGentlenessModel,
        "Interfaith_Sensitivity": InterfaithSensitivityModel,
        "Arabic_Accuracy": ArabicAccuracyDetailed,
    }.items()
}


def has_rubric_scores(evaluation: object) -> bool:
    """Require at least one integer score in a recognized rubric field.

    Partial historical rubrics remain usable. Empty sections, metadata, strings,
    and booleans are not scores; individual rubric scores use the integer schema
    consumed by aggregation. This is a presence check, not full schema validation.
    """
    if not isinstance(evaluation, dict):
        return False
    for section, fields in _RUBRIC_FIELDS.items():
        values = evaluation.get(section)
        if isinstance(values, dict) and any(type(values.get(key)) is int for key in fields):
            return True
    return False


def minimum_rubric_scores(language: str = "english") -> dict:
    sections = {section: {field: 1 for field in fields} for section, fields in _RUBRIC_FIELDS.items()
                if language == "arabic" or section != "Arabic_Accuracy"}
    if language == "arabic":
        sections["Arabic_Accuracy"]["Penalty_Reason"] = "Empty answer"
    return sections


def is_generation_failure_penalty(record: object) -> bool:
    if not is_terminal_generation_failure(record):
        return False
    if (record.get("evaluation_source") != "generation_failure_penalty"
            or record.get("answer") != "" or "error" in record):
        return False
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        return False
    expected = minimum_rubric_scores("arabic" if "Arabic_Accuracy" in evaluation else "english")
    return all(
        isinstance(evaluation.get(section), dict)
        and all(type(evaluation[section].get(field)) is int and evaluation[section][field] == 1
                for field, value in fields.items() if type(value) is int)
        for section, fields in expected.items()
    )


def is_scored_record(record: object) -> bool:
    """Reject error-marked records even if they contain leftover scores."""
    return (
        isinstance(record, dict)
        and "error" not in record
        and ("generation_failure" not in record or is_generation_failure_penalty(record))
        and has_rubric_scores(record.get("evaluation"))
    )


def is_missing_answer_penalty(record: object) -> bool:
    """Recognize local minimum scores without claiming a terminal model failure."""
    if (not isinstance(record, dict) or "error" in record or "generation_failure" in record
            or record.get("evaluation_source") != "missing_answer_penalty" or record.get("answer") != ""):
        return False
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        return False
    language = "arabic" if "Arabic_Accuracy" in evaluation else "english"
    return evaluation == minimum_rubric_scores(language) and all(
        type(value) is int for fields in evaluation.values() for key, value in fields.items()
        if key != "Penalty_Reason"
    )
