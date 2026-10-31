"""Rubric score aggregation shared by CLI and master CSV reporting.

The tagged English benchmark uses macro-averaged section overalls: each
applicable subcriterion contributes one mean to its section, regardless of how
many questions activated that subcriterion.  This module preserves that
published behavior and records the exact value arrays used for both means and
population standard deviations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import fmean, pstdev
from typing import Dict, List, Mapping, Optional, Sequence

from parrot_ai.evaluation_schemas import (
    ALWAYS_ON_SUBCRITERIA,
    SUBCRITERIA_FLAG_MAP,
)

from .score_processing import compute_weighted_final_score
from .score_validation import is_scored_record


CORE_SECTION_ORDER = [
    "Adherence",
    "Kindness_and_Gentleness",
    "Interfaith_Sensitivity",
]

CORE_SECTION_SUBCRITERIA = {
    "Adherence": [
        "Core",
        "Secondary",
        "Tertiary_Handling",
        "Biblical_Basis",
        "Consistency",
        "Overall",
    ],
    "Kindness_and_Gentleness": [
        "Core_Clarity_with_Kindness",
        "Pastoral_Sensitivity",
        "Secondary_Fairness",
        "Tertiary_Neutrality",
        "Tone",
        "Overall",
    ],
    "Interfaith_Sensitivity": [
        "Respect_and_Handling_Objections",
        "Objection_Acknowledgement",
        "Evangelism",
        "Gospel_Boldness",
        "Overall",
    ],
}

ARABIC_ACCURACY_SUBCRITERIA = [
    "Grammar_and_Syntax",
    "Theological_Nuance",
    "Contextual_Clarity",
    "Consistency_of_Terms",
    "Arabic_Purity",
    "Overall",
]

FINAL_OVERALL_ROW = ("", "Final_Overall")
WEIGHTED_SCORE_ROW = ("", "Weighted_Production_Score")

META_ROWS = [
    ("Meta", "System_Prompt_Label"),
    ("Meta", "Judge_Model"),
]

ENGLISH_SECTION_WEIGHTS = {
    "Adherence": 0.40,
    "Interfaith_Sensitivity": 0.35,
    "Kindness_and_Gentleness": 0.25,
}


@dataclass(frozen=True)
class ScoreStatistic:
    """Aggregate statistics for one displayed benchmark metric."""

    mean: float
    stdev: float
    count: int


def build_rows_order(include_arabic: bool) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for section in CORE_SECTION_ORDER:
        for sub in CORE_SECTION_SUBCRITERIA[section]:
            rows.append((section, sub))
    if include_arabic:
        for sub in ARABIC_ACCURACY_SUBCRITERIA:
            rows.append(("Arabic_Accuracy", sub))
    rows.append(FINAL_OVERALL_ROW)
    if not include_arabic:
        rows.append(WEIGHTED_SCORE_ROW)
    rows.extend(META_ROWS)
    return rows


def score_metric_order(include_arabic: bool = False) -> list[tuple[str, str]]:
    """Return score-only metric order, excluding legacy metadata rows."""

    return [row for row in build_rows_order(include_arabic) if row[0] != "Meta"]


def _build_flag_to_subcriteria_index() -> Dict[tuple[str, str], str]:
    reverse: Dict[tuple[str, str], str] = {}
    for flag_name, pairs in SUBCRITERIA_FLAG_MAP.items():
        for pair in pairs:
            reverse[pair] = flag_name
    return reverse


_SUBCRITERIA_TO_FLAG = _build_flag_to_subcriteria_index()


def is_applicable(
    section: str, key: str, question_tag: Optional[Mapping[str, object]]
) -> bool:
    """Return whether a subcriterion contributes for a tagged question."""

    if question_tag is None:
        return True
    if key == "Overall":
        return False
    pair = (section, key)
    if pair in ALWAYS_ON_SUBCRITERIA:
        return True
    flag_name = _SUBCRITERIA_TO_FLAG.get(pair)
    if flag_name is None:
        return True
    return bool(question_tag.get(flag_name, True))


def _stat(values: Sequence[float]) -> ScoreStatistic:
    if not values:
        raise ValueError("Cannot summarize an empty score series")
    return ScoreStatistic(
        mean=round(fmean(values), 2),
        stdev=round(pstdev(values), 2),
        count=len(values),
    )


def _weighted_stdev(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values or len(values) != len(weights):
        return 0.0
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0
    weighted_mean = sum(v * w for v, w in zip(values, weights)) / total_weight
    variance = (
        sum(w * ((v - weighted_mean) ** 2) for v, w in zip(values, weights))
        / total_weight
    )
    return round(sqrt(variance), 2)


def aggregate_score_statistics(
    results: List[dict],
    include_arabic_accuracy: bool,
    question_tags: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> Dict[tuple[str, str], ScoreStatistic]:
    """Aggregate rubric means, population deviations, and contributing counts.

    When question tags are active, section ``Overall`` is summarized from the
    same array of applicable subcriterion means used by the published benchmark.
    ``Final_Overall`` uses the section-overall array.  The English weighted score
    applies the production weights to that same array and uses a weighted
    population deviation.
    """

    target_sections = list(CORE_SECTION_ORDER)
    if include_arabic_accuracy:
        target_sections.append("Arabic_Accuracy")

    use_tags = bool(question_tags)
    series: Dict[tuple[str, str], list[float]] = {}

    for item in results:
        if not is_scored_record(item):
            continue
        evaluation = item["evaluation"]

        question_text = item.get("question", "")
        question_tag: Optional[Mapping[str, object]] = None
        if use_tags and question_tags and question_text in question_tags:
            question_tag = question_tags[question_text]

        for section in target_sections:
            section_obj = evaluation.get(section, {})
            if not isinstance(section_obj, dict):
                continue
            for key, value in section_obj.items():
                if key in (
                    "Penalty_Reason",
                    "Heuristic_Arabic_Purity_Pct",
                    "Pastoral_Acknowledgement",
                ):
                    continue
                if not isinstance(value, int):
                    continue
                if use_tags and section != "Arabic_Accuracy":
                    if not is_applicable(section, key, question_tag):
                        continue
                series.setdefault((section, key), []).append(float(value))

    stats = {key: _stat(values) for key, values in series.items() if values}

    if use_tags:
        for section in target_sections:
            if section == "Arabic_Accuracy":
                continue
            component_means = [
                fmean(series[(section, key)])
                for key in CORE_SECTION_SUBCRITERIA[section]
                if key != "Overall" and (section, key) in series
            ]
            if component_means:
                stats[(section, "Overall")] = _stat(component_means)

    overall_values = [
        stats[(section, "Overall")].mean
        for section in target_sections
        if (section, "Overall") in stats
    ]
    if overall_values:
        stats[FINAL_OVERALL_ROW] = _stat(overall_values)

    if not include_arabic_accuracy:
        weighted_values: list[float] = []
        weighted_weights: list[float] = []
        means: Dict[tuple[str, str], float] = {}
        for section, weight in ENGLISH_SECTION_WEIGHTS.items():
            key = (section, "Overall")
            if key not in stats:
                continue
            value = stats[key].mean
            means[key] = value
            weighted_values.append(value)
            weighted_weights.append(weight)
        if weighted_values:
            stats[WEIGHTED_SCORE_ROW] = ScoreStatistic(
                mean=compute_weighted_final_score(means),
                stdev=_weighted_stdev(weighted_values, weighted_weights),
                count=len(weighted_values),
            )

    return stats


def aggregate_scores(
    results: List[dict],
    include_arabic_accuracy: bool,
    question_tags: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> Dict[tuple[str, str], float]:
    """Backward-compatible means-only aggregation API."""

    return {
        key: statistic.mean
        for key, statistic in aggregate_score_statistics(
            results,
            include_arabic_accuracy,
            question_tags,
        ).items()
    }


__all__ = [
    "ARABIC_ACCURACY_SUBCRITERIA",
    "CORE_SECTION_ORDER",
    "CORE_SECTION_SUBCRITERIA",
    "ENGLISH_SECTION_WEIGHTS",
    "FINAL_OVERALL_ROW",
    "META_ROWS",
    "ScoreStatistic",
    "WEIGHTED_SCORE_ROW",
    "aggregate_score_statistics",
    "aggregate_scores",
    "build_rows_order",
    "is_applicable",
    "score_metric_order",
]
