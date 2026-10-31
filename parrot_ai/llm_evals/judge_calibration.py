"""Dynamic judge-disagreement analysis and blinded human calibration studies."""

from __future__ import annotations

import csv
import hashlib
import importlib
import itertools
import json
import math
import os
import random
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from .aggregation import is_applicable
from .master_csv import (
    ResultSource,
    infer_answers_label,
    infer_dataset_metadata,
    load_jsonl,
)
from .score_validation import is_scored_record


STUDY_SCHEMA_VERSION = 1
RESPONSE_SCHEMA_VERSION = 1

RANKING_COMPOSITE = "composite"
RANKING_ABSOLUTE_DELTA = "absolute_delta"
RANKING_LOWEST_PEARSON = "lowest_pearson"
RANKING_LOWEST_SPEARMAN = "lowest_spearman"
RANKING_REVERSAL_RATE = "reversal_rate"
RANKING_METHODS = (
    RANKING_COMPOSITE,
    RANKING_ABSOLUTE_DELTA,
    RANKING_LOWEST_PEARSON,
    RANKING_LOWEST_SPEARMAN,
    RANKING_REVERSAL_RATE,
)

PAIRWISE_CHOICES = ("A", "Tie", "B", "Not applicable")
POINTWISE_CHOICES = ("1", "2", "3", "4", "5", "Not applicable")
PAIRWISE_STRATA = ("strict_reversal", "tie_conflict", "agreement")
POINTWISE_STRATA = ("large_gap", "small_gap", "agreement")


@dataclass(frozen=True, order=True)
class MetricPath:
    """A numeric leaf path in a nested evaluation result."""

    parts: tuple[str, ...]

    @property
    def key(self) -> str:
        return ".".join(self.parts)

    @property
    def label(self) -> str:
        return " › ".join(part.replace("_", " ") for part in self.parts)

    @property
    def section(self) -> str:
        return self.parts[0]

    @property
    def criterion(self) -> str:
        return self.parts[-1]

    @classmethod
    def from_key(cls, key: str) -> "MetricPath":
        return cls(tuple(part for part in key.split(".") if part))


@dataclass(frozen=True)
class EvaluationRun:
    """One registered answer-set/judge evaluation with valid records."""

    source_path: Path
    registry_index: int
    eval_version: str
    system_prompt_label: str
    language: str
    judge_model: str
    answers_label: str
    provider: str
    gen_model: str
    records_by_question: Mapping[str, dict]


@dataclass(frozen=True)
class PairedAnswerSet:
    """The same answer set scored by two judges in the same context."""

    answers_label: str
    provider: str
    gen_model: str
    primary: EvaluationRun
    comparison: EvaluationRun


@dataclass(frozen=True)
class ScoredItem:
    """One answer with both judges' scores for a single rubric metric."""

    metric: MetricPath
    question: str
    answer: str
    answers_label: str
    provider: str
    gen_model: str
    primary_score: int
    comparison_score: int


@dataclass(frozen=True)
class MetricStatistics:
    """Question-level disagreement statistics for one rubric leaf."""

    metric_key: str
    metric_label: str
    question_count: int
    item_count: int
    primary_mean: float
    comparison_mean: float
    mean_delta: float
    mean_absolute_delta: float
    exact_agreement: float
    within_one: float
    pearson: Optional[float]
    spearman: Optional[float]
    answer_pair_count: int
    strict_reversal_count: int
    tie_conflict_count: int
    same_order_count: int
    both_tie_count: int
    strict_reversal_rate: float
    tie_conflict_rate: float
    disagreement_index: float


@dataclass(frozen=True)
class PairwiseCandidate:
    metric: MetricPath
    question: str
    left: ScoredItem
    right: ScoredItem
    primary_direction: int
    comparison_direction: int
    stratum: str


def _last_value(records: Sequence[dict], key: str) -> object:
    for record in reversed(records):
        value = record.get(key)
        if value not in (None, ""):
            return value
    return ""


def _repo_path(value: object, repo_root: Path) -> Optional[Path]:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return (path if path.is_absolute() else repo_root / path).resolve()


def _deduplicate_valid_records(records: Sequence[dict]) -> dict[str, dict]:
    """Keep the latest valid evaluation for each question."""

    by_question: dict[str, dict] = {}
    for record in records:
        question = record.get("question")
        if (
            isinstance(question, str)
            and question
            and is_scored_record(record)
        ):
            by_question[question] = record
    return by_question


def load_evaluation_runs(
    sources: Iterable[ResultSource],
    *,
    repo_root: Path,
) -> list[EvaluationRun]:
    """Load registered result files into normalized logical runs."""

    runs: list[EvaluationRun] = []
    for registry_index, source in enumerate(sources):
        path = source.path
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        if not path.exists():
            continue
        all_records = load_jsonl(path)
        records_by_question = _deduplicate_valid_records(all_records)
        if not records_by_question:
            continue

        dataset_path = _repo_path(_last_value(all_records, "dataset"), repo_root)
        dataset_metadata = infer_dataset_metadata(dataset_path)
        provider = str(
            _last_value(all_records, "provider")
            or dataset_metadata.get("provider", "")
        )
        gen_model = str(
            _last_value(all_records, "gen_model")
            or dataset_metadata.get("gen_model", "")
        )
        system_prompt_label = str(
            _last_value(all_records, "system_prompt_label")
            or dataset_metadata.get("system_prompt_label", "")
        )
        answers_label = infer_answers_label(
            _last_value(all_records, "answers_label"),
            dataset_path,
            provider,
        )
        judge_model = str(_last_value(all_records, "judge_model") or "")
        if not answers_label or not judge_model:
            continue

        runs.append(
            EvaluationRun(
                source_path=path,
                registry_index=registry_index,
                eval_version=source.eval_version,
                system_prompt_label=system_prompt_label,
                language=str(_last_value(all_records, "language") or "english"),
                judge_model=judge_model,
                answers_label=answers_label,
                provider=provider,
                gen_model=gen_model,
                records_by_question=records_by_question,
            )
        )
    return runs


def available_contexts(runs: Sequence[EvaluationRun]) -> list[tuple[str, str, str]]:
    """Return available ``(language, eval_version, prompt_label)`` contexts."""

    return sorted(
        {
            (run.language, run.eval_version, run.system_prompt_label)
            for run in runs
        }
    )


def available_judges(
    runs: Sequence[EvaluationRun],
    *,
    language: str,
    eval_version: str,
    system_prompt_label: str,
) -> list[str]:
    return sorted(
        {
            run.judge_model
            for run in runs
            if run.language == language
            and run.eval_version == eval_version
            and run.system_prompt_label == system_prompt_label
        }
    )


def _latest_runs_by_answer_set(
    runs: Sequence[EvaluationRun],
    *,
    language: str,
    eval_version: str,
    system_prompt_label: str,
    judge_model: str,
) -> dict[str, EvaluationRun]:
    """Resolve duplicate logical runs by latest registry position."""

    selected: dict[str, EvaluationRun] = {}
    for run in runs:
        if (
            run.language == language
            and run.eval_version == eval_version
            and run.system_prompt_label == system_prompt_label
            and run.judge_model == judge_model
        ):
            selected[run.answers_label] = run
    return selected


def shared_answer_set_labels(
    runs: Sequence[EvaluationRun],
    *,
    language: str,
    eval_version: str,
    system_prompt_label: str,
    primary_judge: str,
    comparison_judge: str,
) -> list[str]:
    primary = _latest_runs_by_answer_set(
        runs,
        language=language,
        eval_version=eval_version,
        system_prompt_label=system_prompt_label,
        judge_model=primary_judge,
    )
    comparison = _latest_runs_by_answer_set(
        runs,
        language=language,
        eval_version=eval_version,
        system_prompt_label=system_prompt_label,
        judge_model=comparison_judge,
    )
    return sorted(primary.keys() & comparison.keys())


def pair_answer_sets(
    runs: Sequence[EvaluationRun],
    *,
    language: str,
    eval_version: str,
    system_prompt_label: str,
    primary_judge: str,
    comparison_judge: str,
    answers_labels: Optional[Iterable[str]] = None,
) -> list[PairedAnswerSet]:
    """Pair shared answer sets for two selected judges."""

    primary = _latest_runs_by_answer_set(
        runs,
        language=language,
        eval_version=eval_version,
        system_prompt_label=system_prompt_label,
        judge_model=primary_judge,
    )
    comparison = _latest_runs_by_answer_set(
        runs,
        language=language,
        eval_version=eval_version,
        system_prompt_label=system_prompt_label,
        judge_model=comparison_judge,
    )
    selected = set(answers_labels) if answers_labels is not None else None
    labels = sorted(primary.keys() & comparison.keys())
    if selected is not None:
        labels = [label for label in labels if label in selected]
    return [
        PairedAnswerSet(
            answers_label=label,
            provider=primary[label].provider or comparison[label].provider,
            gen_model=primary[label].gen_model or comparison[label].gen_model,
            primary=primary[label],
            comparison=comparison[label],
        )
        for label in labels
    ]


def _numeric_leaf_paths(
    value: object,
    prefix: tuple[str, ...] = (),
) -> dict[MetricPath, int]:
    leaves: dict[MetricPath, int] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            leaves.update(_numeric_leaf_paths(child, (*prefix, str(key))))
    elif isinstance(value, int) and not isinstance(value, bool) and prefix:
        metric = MetricPath(prefix)
        if metric.criterion != "Overall":
            leaves[metric] = value
    return leaves


def _metric_is_applicable(
    metric: MetricPath,
    question_tag: Optional[Mapping[str, object]],
) -> bool:
    if len(metric.parts) == 2:
        return is_applicable(metric.section, metric.criterion, question_tag)
    return True


def collect_scored_items(
    paired_answer_sets: Sequence[PairedAnswerSet],
    *,
    question_tags: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> dict[MetricPath, list[ScoredItem]]:
    """Collect dynamically discovered, applicable rubric leaves."""

    by_metric: dict[MetricPath, list[ScoredItem]] = defaultdict(list)
    for paired in paired_answer_sets:
        shared_questions = sorted(
            paired.primary.records_by_question.keys()
            & paired.comparison.records_by_question.keys()
        )
        for question in shared_questions:
            primary_record = paired.primary.records_by_question[question]
            comparison_record = paired.comparison.records_by_question[question]
            primary_answer = primary_record.get("answer")
            comparison_answer = comparison_record.get("answer")
            if (
                not isinstance(primary_answer, str)
                or primary_answer != comparison_answer
            ):
                continue
            primary_leaves = _numeric_leaf_paths(primary_record["evaluation"])
            comparison_leaves = _numeric_leaf_paths(comparison_record["evaluation"])
            tag = question_tags.get(question) if question_tags else None
            for metric in sorted(primary_leaves.keys() & comparison_leaves.keys()):
                if not _metric_is_applicable(metric, tag):
                    continue
                by_metric[metric].append(
                    ScoredItem(
                        metric=metric,
                        question=question,
                        answer=primary_answer,
                        answers_label=paired.answers_label,
                        provider=paired.provider,
                        gen_model=paired.gen_model,
                        primary_score=primary_leaves[metric],
                        comparison_score=comparison_leaves[metric],
                    )
                )
    return dict(by_metric)


def _pearson(values_a: Sequence[float], values_b: Sequence[float]) -> Optional[float]:
    if len(values_a) < 2 or len(values_a) != len(values_b):
        return None
    mean_a = statistics.fmean(values_a)
    mean_b = statistics.fmean(values_b)
    numerator = sum(
        (value_a - mean_a) * (value_b - mean_b)
        for value_a, value_b in zip(values_a, values_b)
    )
    denominator = math.sqrt(
        sum((value - mean_a) ** 2 for value in values_a)
        * sum((value - mean_b) ** 2 for value in values_b)
    )
    if denominator == 0:
        return None
    return numerator / denominator


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for index in range(start, end):
            ranks[ordered[index][0]] = average_rank
        start = end
    return ranks


def _spearman(values_a: Sequence[float], values_b: Sequence[float]) -> Optional[float]:
    return _pearson(_average_ranks(values_a), _average_ranks(values_b))


def _direction(left_score: int, right_score: int) -> int:
    return (left_score > right_score) - (left_score < right_score)


def build_pairwise_candidates(
    items: Sequence[ScoredItem],
) -> list[PairwiseCandidate]:
    """Build all within-question answer pairs and disagreement strata."""

    by_question: dict[str, list[ScoredItem]] = defaultdict(list)
    for item in items:
        by_question[item.question].append(item)

    candidates: list[PairwiseCandidate] = []
    for question, question_items in by_question.items():
        ordered = sorted(question_items, key=lambda item: item.answers_label)
        for left, right in itertools.combinations(ordered, 2):
            primary_direction = _direction(
                left.primary_score, right.primary_score
            )
            comparison_direction = _direction(
                left.comparison_score, right.comparison_score
            )
            if (
                primary_direction
                and comparison_direction
                and primary_direction == -comparison_direction
            ):
                stratum = "strict_reversal"
            elif (primary_direction == 0) != (comparison_direction == 0):
                stratum = "tie_conflict"
            else:
                stratum = "agreement"
            candidates.append(
                PairwiseCandidate(
                    metric=left.metric,
                    question=question,
                    left=left,
                    right=right,
                    primary_direction=primary_direction,
                    comparison_direction=comparison_direction,
                    stratum=stratum,
                )
            )
    return candidates


def calculate_metric_statistics(
    items_by_metric: Mapping[MetricPath, Sequence[ScoredItem]],
    *,
    minimum_items: int = 2,
) -> list[MetricStatistics]:
    """Calculate item-level score and within-question rank disagreement."""

    statistics_rows: list[MetricStatistics] = []
    for metric, items in sorted(items_by_metric.items()):
        if len(items) < minimum_items:
            continue
        primary_scores = [item.primary_score for item in items]
        comparison_scores = [item.comparison_score for item in items]
        deltas = [
            comparison - primary
            for primary, comparison in zip(primary_scores, comparison_scores)
        ]
        pearson = _pearson(primary_scores, comparison_scores)
        spearman = _spearman(primary_scores, comparison_scores)
        candidates = build_pairwise_candidates(items)
        stratum_counts = Counter(candidate.stratum for candidate in candidates)
        same_order = sum(
            1
            for candidate in candidates
            if candidate.primary_direction == candidate.comparison_direction
            and candidate.primary_direction != 0
        )
        both_tie = sum(
            1
            for candidate in candidates
            if candidate.primary_direction == candidate.comparison_direction == 0
        )
        pair_count = len(candidates)
        normalized_components = [
            statistics.fmean(abs(delta) for delta in deltas) / 4
        ]
        if pearson is not None:
            normalized_components.append((1 - max(-1.0, min(1.0, pearson))) / 2)
        if spearman is not None:
            normalized_components.append((1 - max(-1.0, min(1.0, spearman))) / 2)

        statistics_rows.append(
            MetricStatistics(
                metric_key=metric.key,
                metric_label=metric.label,
                question_count=len({item.question for item in items}),
                item_count=len(items),
                primary_mean=statistics.fmean(primary_scores),
                comparison_mean=statistics.fmean(comparison_scores),
                mean_delta=statistics.fmean(deltas),
                mean_absolute_delta=statistics.fmean(abs(delta) for delta in deltas),
                exact_agreement=statistics.fmean(
                    primary == comparison
                    for primary, comparison in zip(
                        primary_scores, comparison_scores
                    )
                ),
                within_one=statistics.fmean(
                    abs(primary - comparison) <= 1
                    for primary, comparison in zip(
                        primary_scores, comparison_scores
                    )
                ),
                pearson=pearson,
                spearman=spearman,
                answer_pair_count=pair_count,
                strict_reversal_count=stratum_counts["strict_reversal"],
                tie_conflict_count=stratum_counts["tie_conflict"],
                same_order_count=same_order,
                both_tie_count=both_tie,
                strict_reversal_rate=(
                    stratum_counts["strict_reversal"] / pair_count
                    if pair_count
                    else 0.0
                ),
                tie_conflict_rate=(
                    stratum_counts["tie_conflict"] / pair_count
                    if pair_count
                    else 0.0
                ),
                disagreement_index=statistics.fmean(normalized_components),
            )
        )
    return statistics_rows


def rank_metric_statistics(
    rows: Sequence[MetricStatistics],
    method: str = RANKING_COMPOSITE,
) -> list[MetricStatistics]:
    """Rank metrics without assuming any particular rubric names."""

    if method not in RANKING_METHODS:
        raise ValueError(f"Unknown disagreement ranking method: {method}")

    def ranking_value(row: MetricStatistics) -> float:
        if method == RANKING_COMPOSITE:
            return row.disagreement_index
        if method == RANKING_ABSOLUTE_DELTA:
            return abs(row.mean_delta)
        if method == RANKING_LOWEST_PEARSON:
            return -(row.pearson if row.pearson is not None else 1.0)
        if method == RANKING_LOWEST_SPEARMAN:
            return -(row.spearman if row.spearman is not None else 1.0)
        return row.strict_reversal_rate

    return sorted(
        rows,
        key=lambda row: (ranking_value(row), row.item_count, row.metric_key),
        reverse=True,
    )


def load_rubric_instructions(language: str) -> str:
    prompts = importlib.import_module(f"parrot_ai.prompts.{language}")
    return str(getattr(prompts, "EVAL_INSTRUCTIONS"))


def extract_rubric_guidance(instructions: str, metric: MetricPath) -> str:
    """Extract the criterion's anchored block from the language prompt."""

    lines = instructions.splitlines()
    general = next(
        (
            line.strip()
            for line in lines
            if line.strip().startswith("Scores 1")
        ),
        "Use the configured rubric scale and behavioral anchors.",
    )
    marker = re.compile(
        rf"^\*\s+{re.escape(metric.criterion)}(?:\s|\(|:)",
        flags=re.IGNORECASE,
    )
    start = next(
        (index for index, line in enumerate(lines) if marker.match(line.strip())),
        None,
    )
    if start is None:
        return f"{general}\n\nEvaluate only **{metric.label}**."

    block: list[str] = []
    for index in range(start, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if index > start and (
            stripped.startswith("* ")
            or stripped.endswith("Sub-guidelines:")
            or stripped.startswith("Output:")
        ):
            break
        block.append(line)
    return f"{general}\n\n" + "\n".join(block).strip()


def format_rubric_guidance_markdown(guidance: str) -> str:
    """Preserve separate display rows for the rubric's 5, 3, and 1 anchors."""

    rendered_lines: list[str] = []
    anchor = re.compile(r"^\s*(?:5|3|1):\s")
    for line in guidance.splitlines():
        if anchor.match(line) and rendered_lines and rendered_lines[-1].strip():
            rendered_lines[-1] = f"{rendered_lines[-1].rstrip()}  "
        rendered_lines.append(line)
    return "\n".join(rendered_lines)


def _largest_remainder_counts(
    total: int,
    weights: Mapping[str, float],
) -> dict[str, int]:
    raw = {key: total * weight for key, weight in weights.items()}
    counts = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(
        weights,
        key=lambda key: (raw[key] - counts[key], weights[key], key),
        reverse=True,
    )
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def _take_diverse(
    pool: list[object],
    count: int,
    *,
    rng: random.Random,
    selected_ids: set[int],
    selected_questions: set[str],
) -> list[object]:
    shuffled = list(pool)
    rng.shuffle(shuffled)
    chosen: list[object] = []
    for prefer_new_question in (True, False):
        for candidate in shuffled:
            candidate_id = id(candidate)
            question = str(getattr(candidate, "question"))
            if candidate_id in selected_ids:
                continue
            if prefer_new_question and question in selected_questions:
                continue
            selected_ids.add(candidate_id)
            selected_questions.add(question)
            chosen.append(candidate)
            if len(chosen) == count:
                return chosen
    return chosen


def _sample_pairwise_candidates(
    candidates: Sequence[PairwiseCandidate],
    total: int,
    *,
    rng: random.Random,
) -> list[PairwiseCandidate]:
    quotas = _largest_remainder_counts(
        total,
        {
            "strict_reversal": 0.60,
            "tie_conflict": 0.20,
            "agreement": 0.20,
        },
    )
    pools = {
        stratum: [
            candidate
            for candidate in candidates
            if candidate.stratum == stratum
        ]
        for stratum in PAIRWISE_STRATA
    }
    selected: list[PairwiseCandidate] = []
    selected_ids: set[int] = set()
    selected_questions: set[str] = set()
    for stratum in PAIRWISE_STRATA:
        selected.extend(
            _take_diverse(
                pools[stratum],
                quotas[stratum],
                rng=rng,
                selected_ids=selected_ids,
                selected_questions=selected_questions,
            )
        )
    if len(selected) < total:
        remaining = [
            candidate
            for stratum in PAIRWISE_STRATA
            for candidate in pools[stratum]
            if id(candidate) not in selected_ids
        ]
        selected.extend(
            _take_diverse(
                remaining,
                total - len(selected),
                rng=rng,
                selected_ids=selected_ids,
                selected_questions=selected_questions,
            )
        )
    return selected


def _sample_pointwise_items(
    items: Sequence[ScoredItem],
    total: int,
    *,
    rng: random.Random,
) -> list[ScoredItem]:
    pools = {
        "large_gap": [
            item
            for item in items
            if abs(item.primary_score - item.comparison_score) >= 2
        ],
        "small_gap": [
            item
            for item in items
            if abs(item.primary_score - item.comparison_score) == 1
        ],
        "agreement": [
            item
            for item in items
            if item.primary_score == item.comparison_score
        ],
    }
    quotas = _largest_remainder_counts(
        total,
        {"large_gap": 0.40, "small_gap": 0.40, "agreement": 0.20},
    )
    selected: list[ScoredItem] = []
    selected_ids: set[int] = set()
    selected_questions: set[str] = set()
    for stratum in POINTWISE_STRATA:
        selected.extend(
            _take_diverse(
                pools[stratum],
                quotas[stratum],
                rng=rng,
                selected_ids=selected_ids,
                selected_questions=selected_questions,
            )
        )
    if len(selected) < total:
        remaining = [
            item
            for stratum in POINTWISE_STRATA
            for item in pools[stratum]
            if id(item) not in selected_ids
        ]
        selected.extend(
            _take_diverse(
                remaining,
                total - len(selected),
                rng=rng,
                selected_ids=selected_ids,
                selected_questions=selected_questions,
            )
        )
    return selected


def _task_id(*parts: object) -> str:
    identity = "|".join(str(part) for part in parts)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _pairwise_task(
    candidate: PairwiseCandidate,
    *,
    guidance: str,
    seed: int,
    rng: random.Random,
) -> dict:
    left = candidate.left
    right = candidate.right
    primary_direction = candidate.primary_direction
    comparison_direction = candidate.comparison_direction
    if rng.random() < 0.5:
        left, right = right, left
        primary_direction *= -1
        comparison_direction *= -1
    return {
        "task_id": _task_id(
            "pairwise",
            seed,
            candidate.metric.key,
            candidate.question,
            left.answers_label,
            right.answers_label,
        ),
        "kind": "pairwise",
        "metric_key": candidate.metric.key,
        "metric_label": candidate.metric.label,
        "rubric_guidance": guidance,
        "question": candidate.question,
        "response_a": left.answer,
        "response_b": right.answer,
        "internal": {
            "stratum": candidate.stratum,
            "answer_a_label": left.answers_label,
            "answer_b_label": right.answers_label,
            "answer_a_provider": left.provider,
            "answer_b_provider": right.provider,
            "primary_score_a": left.primary_score,
            "primary_score_b": right.primary_score,
            "comparison_score_a": left.comparison_score,
            "comparison_score_b": right.comparison_score,
            "primary_direction": primary_direction,
            "comparison_direction": comparison_direction,
        },
    }


def _pointwise_task(
    item: ScoredItem,
    *,
    guidance: str,
    seed: int,
) -> dict:
    gap = abs(item.primary_score - item.comparison_score)
    stratum = "large_gap" if gap >= 2 else "small_gap" if gap == 1 else "agreement"
    return {
        "task_id": _task_id(
            "pointwise",
            seed,
            item.metric.key,
            item.question,
            item.answers_label,
        ),
        "kind": "pointwise",
        "metric_key": item.metric.key,
        "metric_label": item.metric.label,
        "rubric_guidance": guidance,
        "question": item.question,
        "response": item.answer,
        "internal": {
            "stratum": stratum,
            "answer_label": item.answers_label,
            "provider": item.provider,
            "primary_score": item.primary_score,
            "comparison_score": item.comparison_score,
        },
    }


def create_calibration_study(
    *,
    title: str,
    language: str,
    eval_version: str,
    system_prompt_label: str,
    primary_judge: str,
    comparison_judge: str,
    answer_sets: Sequence[str],
    ranking_method: str,
    selected_metrics: Sequence[MetricPath],
    all_statistics: Sequence[MetricStatistics],
    items_by_metric: Mapping[MetricPath, Sequence[ScoredItem]],
    rubric_instructions: str,
    pairwise_trials_per_metric: int = 50,
    pointwise_trials_per_metric: int = 25,
    seed: int = 20260724,
) -> dict:
    """Create a deterministic, blinded study definition."""

    if primary_judge == comparison_judge:
        raise ValueError("A calibration study requires two different judges")
    if not selected_metrics:
        raise ValueError("Select at least one rubric metric")
    if pairwise_trials_per_metric < 0 or pointwise_trials_per_metric < 0:
        raise ValueError("Trial counts cannot be negative")
    if not pairwise_trials_per_metric and not pointwise_trials_per_metric:
        raise ValueError("A calibration study needs at least one trial")

    rng = random.Random(seed)
    tasks: list[dict] = []
    for metric in selected_metrics:
        items = list(items_by_metric.get(metric, []))
        if not items:
            raise ValueError(f"No paired items are available for {metric.label}")
        guidance = extract_rubric_guidance(rubric_instructions, metric)
        if pairwise_trials_per_metric:
            candidates = build_pairwise_candidates(items)
            sampled_pairs = _sample_pairwise_candidates(
                candidates,
                pairwise_trials_per_metric,
                rng=rng,
            )
            tasks.extend(
                _pairwise_task(
                    candidate,
                    guidance=guidance,
                    seed=seed,
                    rng=rng,
                )
                for candidate in sampled_pairs
            )
        if pointwise_trials_per_metric:
            sampled_items = _sample_pointwise_items(
                items,
                pointwise_trials_per_metric,
                rng=rng,
            )
            tasks.extend(
                _pointwise_task(item, guidance=guidance, seed=seed)
                for item in sampled_items
            )
    rng.shuffle(tasks)

    created_at = datetime.now(timezone.utc).isoformat()
    identity = "|".join(
        [
            created_at,
            primary_judge,
            comparison_judge,
            system_prompt_label,
            str(seed),
            ",".join(metric.key for metric in selected_metrics),
        ]
    )
    study_id = (
        f"judge-calibration-{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
        f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:8]}"
    )
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_id": study_id,
        "title": title.strip() or "Judge calibration study",
        "created_at": created_at,
        "language": language,
        "eval_version": eval_version,
        "system_prompt_label": system_prompt_label,
        "primary_judge": primary_judge,
        "comparison_judge": comparison_judge,
        "answer_sets": list(answer_sets),
        "ranking_method": ranking_method,
        "selected_metrics": [metric.key for metric in selected_metrics],
        "sampling": {
            "seed": seed,
            "pairwise_trials_per_metric": pairwise_trials_per_metric,
            "pointwise_trials_per_metric": pointwise_trials_per_metric,
            "pairwise_target_mix": {
                "strict_reversal": 0.60,
                "tie_conflict": 0.20,
                "agreement": 0.20,
            },
            "pointwise_target_mix": {
                "large_gap": 0.40,
                "small_gap": 0.40,
                "agreement": 0.20,
            },
        },
        "metric_statistics": [asdict(row) for row in all_statistics],
        "tasks": tasks,
    }


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def save_study(study: Mapping[str, object], studies_root: Path) -> Path:
    study_id = str(study.get("study_id") or "")
    if not study_id:
        raise ValueError("Study payload has no study_id")
    study_dir = studies_root / study_id
    study_path = study_dir / "study.json"
    if study_path.exists():
        raise FileExistsError(f"Study already exists: {study_path}")
    _atomic_write_json(study_path, study)
    return study_dir


def load_study(study_dir: Path) -> dict:
    study_path = study_dir / "study.json"
    with study_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != STUDY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported study schema: {study_path}")
    return payload


def list_study_directories(studies_root: Path) -> list[Path]:
    if not studies_root.exists():
        return []
    return sorted(
        (
            path
            for path in studies_root.iterdir()
            if path.is_dir() and (path / "study.json").exists()
        ),
        key=lambda path: path.name,
        reverse=True,
    )


def _reviewer_filename(reviewer_id: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", reviewer_id.strip()).strip("-")
    normalized = normalized[:48] or "reviewer"
    digest = hashlib.sha256(reviewer_id.strip().encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{digest}.json"


def load_reviewer_responses(study_dir: Path, reviewer_id: str) -> dict:
    path = study_dir / "responses" / _reviewer_filename(reviewer_id)
    if not path.exists():
        return {
            "schema_version": RESPONSE_SCHEMA_VERSION,
            "study_id": study_dir.name,
            "reviewer_id": reviewer_id.strip(),
            "responses": {},
        }
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported response schema: {path}")
    return payload


def save_reviewer_response(
    study_dir: Path,
    reviewer_id: str,
    *,
    task_id: str,
    choice: str,
) -> Path:
    """Atomically add or replace one reviewer's answer."""

    study = load_study(study_dir)
    tasks = {
        str(task["task_id"]): task
        for task in study.get("tasks", [])
        if isinstance(task, dict)
    }
    if task_id not in tasks:
        raise ValueError(f"Unknown task ID for {study_dir.name}: {task_id}")
    task = tasks[task_id]
    allowed = PAIRWISE_CHOICES if task["kind"] == "pairwise" else POINTWISE_CHOICES
    if choice not in allowed:
        raise ValueError(f"Invalid choice for {task['kind']} task: {choice}")

    payload = load_reviewer_responses(study_dir, reviewer_id)
    payload["responses"][task_id] = {
        "choice": choice,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    path = study_dir / "responses" / _reviewer_filename(reviewer_id)
    _atomic_write_json(path, payload)
    return path


def list_reviewers(study_dir: Path) -> list[str]:
    responses_dir = study_dir / "responses"
    if not responses_dir.exists():
        return []
    reviewers: list[str] = []
    for path in sorted(responses_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        reviewer_id = payload.get("reviewer_id")
        if isinstance(reviewer_id, str) and reviewer_id:
            reviewers.append(reviewer_id)
    return reviewers


def _choice_direction(choice: str) -> Optional[int]:
    return {"A": 1, "Tie": 0, "B": -1}.get(choice)


def _bootstrap_accuracy_difference(
    rows: Sequence[tuple[str, int, int]],
    *,
    seed: int = 1701,
    iterations: int = 2000,
) -> tuple[Optional[float], Optional[float]]:
    if len(rows) < 2:
        return None, None
    by_question: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for question, primary_match, comparison_match in rows:
        by_question[question].append((primary_match, comparison_match))
    questions = list(by_question)
    if len(questions) < 2:
        return None, None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sampled = [rng.choice(questions) for _ in questions]
        matches = [
            match
            for question in sampled
            for match in by_question[question]
        ]
        estimates.append(
            statistics.fmean(comparison - primary for primary, comparison in matches)
        )
    estimates.sort()
    lower = estimates[math.floor(0.025 * (iterations - 1))]
    upper = estimates[math.ceil(0.975 * (iterations - 1))]
    return lower, upper


def _quadratic_weighted_kappa(
    human_scores: Sequence[int],
    judge_scores: Sequence[int],
    *,
    minimum: int = 1,
    maximum: int = 5,
) -> Optional[float]:
    if not human_scores or len(human_scores) != len(judge_scores):
        return None
    categories = list(range(minimum, maximum + 1))
    size = len(categories)
    observed = [[0.0 for _ in categories] for _ in categories]
    human_counts = [0.0] * size
    judge_counts = [0.0] * size
    for human, judge in zip(human_scores, judge_scores):
        human_index = human - minimum
        judge_index = judge - minimum
        observed[human_index][judge_index] += 1
        human_counts[human_index] += 1
        judge_counts[judge_index] += 1
    count = len(human_scores)
    denominator = (size - 1) ** 2
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for human_index in range(size):
        for judge_index in range(size):
            weight = ((human_index - judge_index) ** 2) / denominator
            observed_disagreement += weight * observed[human_index][judge_index]
            expected = (
                human_counts[human_index] * judge_counts[judge_index] / count
            )
            expected_disagreement += weight * expected
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return 1 - (observed_disagreement / expected_disagreement)


def analyze_study_responses(study: Mapping[str, object], response_payload: dict) -> dict:
    """Compare one reviewer's blinded judgments with both selected judges."""

    saved = response_payload.get("responses", {})
    pairwise_by_metric: dict[str, list[tuple[dict, int]]] = defaultdict(list)
    pointwise_by_metric: dict[str, list[tuple[dict, int]]] = defaultdict(list)
    not_applicable = 0
    for task in study.get("tasks", []):
        response = saved.get(task["task_id"])
        if not isinstance(response, dict):
            continue
        choice = str(response.get("choice") or "")
        if choice == "Not applicable":
            not_applicable += 1
            continue
        if task["kind"] == "pairwise":
            direction = _choice_direction(choice)
            if direction is not None:
                pairwise_by_metric[task["metric_key"]].append((task, direction))
        elif choice.isdigit():
            pointwise_by_metric[task["metric_key"]].append((task, int(choice)))

    pairwise_rows: list[dict] = []
    for metric_key, rows in sorted(pairwise_by_metric.items()):
        for slice_name, slice_rows in (
            ("all", rows),
            (
                "judge_conflicts",
                [
                    row
                    for row in rows
                    if row[0]["internal"]["stratum"]
                    in {"strict_reversal", "tie_conflict"}
                ],
            ),
        ):
            if not slice_rows:
                continue
            primary_matches = [
                int(task["internal"]["primary_direction"] == human_direction)
                for task, human_direction in slice_rows
            ]
            comparison_matches = [
                int(task["internal"]["comparison_direction"] == human_direction)
                for task, human_direction in slice_rows
            ]
            bootstrap_rows = [
                (
                    task["question"],
                    primary_match,
                    comparison_match,
                )
                for (task, _), primary_match, comparison_match in zip(
                    slice_rows, primary_matches, comparison_matches
                )
            ]
            ci_low, ci_high = _bootstrap_accuracy_difference(bootstrap_rows)
            primary_accuracy = statistics.fmean(primary_matches)
            comparison_accuracy = statistics.fmean(comparison_matches)
            pairwise_rows.append(
                {
                    "metric_key": metric_key,
                    "metric_label": slice_rows[0][0]["metric_label"],
                    "slice": slice_name,
                    "n": len(slice_rows),
                    "primary_accuracy": primary_accuracy,
                    "comparison_accuracy": comparison_accuracy,
                    "comparison_minus_primary": (
                        comparison_accuracy - primary_accuracy
                    ),
                    "difference_ci_low": ci_low,
                    "difference_ci_high": ci_high,
                }
            )

    pointwise_rows: list[dict] = []
    for metric_key, rows in sorted(pointwise_by_metric.items()):
        human = [score for _, score in rows]
        primary = [task["internal"]["primary_score"] for task, _ in rows]
        comparison = [task["internal"]["comparison_score"] for task, _ in rows]
        pointwise_rows.append(
            {
                "metric_key": metric_key,
                "metric_label": rows[0][0]["metric_label"],
                "n": len(rows),
                "primary_mae": statistics.fmean(
                    abs(judge - human_score)
                    for judge, human_score in zip(primary, human)
                ),
                "comparison_mae": statistics.fmean(
                    abs(judge - human_score)
                    for judge, human_score in zip(comparison, human)
                ),
                "primary_exact": statistics.fmean(
                    judge == human_score
                    for judge, human_score in zip(primary, human)
                ),
                "comparison_exact": statistics.fmean(
                    judge == human_score
                    for judge, human_score in zip(comparison, human)
                ),
                "primary_weighted_kappa": _quadratic_weighted_kappa(
                    human, primary
                ),
                "comparison_weighted_kappa": _quadratic_weighted_kappa(
                    human, comparison
                ),
            }
        )

    return {
        "reviewer_id": response_payload.get("reviewer_id", ""),
        "answered": len(saved),
        "total_tasks": len(study.get("tasks", [])),
        "not_applicable": not_applicable,
        "pairwise": pairwise_rows,
        "pointwise": pointwise_rows,
    }


def response_export_rows(study: Mapping[str, object], response_payload: dict) -> list[dict]:
    """Return a complete auditable task/response table for CSV export."""

    saved = response_payload.get("responses", {})
    rows: list[dict] = []
    for task in study.get("tasks", []):
        response = saved.get(task["task_id"], {})
        row = {
            "study_id": study.get("study_id", ""),
            "reviewer_id": response_payload.get("reviewer_id", ""),
            "task_id": task["task_id"],
            "kind": task["kind"],
            "metric_key": task["metric_key"],
            "question": task["question"],
            "human_choice": response.get("choice", ""),
            "saved_at": response.get("saved_at", ""),
        }
        row.update(task.get("internal", {}))
        rows.append(row)
    return rows


def rows_to_csv(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return ""
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


__all__ = [
    "EvaluationRun",
    "MetricPath",
    "MetricStatistics",
    "PAIRWISE_CHOICES",
    "POINTWISE_CHOICES",
    "PairedAnswerSet",
    "RANKING_ABSOLUTE_DELTA",
    "RANKING_COMPOSITE",
    "RANKING_LOWEST_PEARSON",
    "RANKING_LOWEST_SPEARMAN",
    "RANKING_METHODS",
    "RANKING_REVERSAL_RATE",
    "ScoredItem",
    "analyze_study_responses",
    "available_contexts",
    "available_judges",
    "build_pairwise_candidates",
    "calculate_metric_statistics",
    "collect_scored_items",
    "create_calibration_study",
    "extract_rubric_guidance",
    "format_rubric_guidance_markdown",
    "list_reviewers",
    "list_study_directories",
    "load_evaluation_runs",
    "load_reviewer_responses",
    "load_rubric_instructions",
    "load_study",
    "pair_answer_sets",
    "rank_metric_statistics",
    "response_export_rows",
    "rows_to_csv",
    "save_reviewer_response",
    "save_study",
    "shared_answer_set_labels",
]
