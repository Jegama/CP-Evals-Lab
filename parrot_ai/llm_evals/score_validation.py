"""Shared minimum validation for accepting and reusing rubric evaluations."""

from parrot_ai.evaluation_schemas import (
    AdherenceModel,
    ArabicAccuracyDetailed,
    InterfaithSensitivityModel,
    KindnessGentlenessModel,
)


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


def is_scored_record(record: object) -> bool:
    """Reject error-marked records even if they contain leftover scores."""
    return (
        isinstance(record, dict)
        and "error" not in record
        and has_rubric_scores(record.get("evaluation"))
    )
