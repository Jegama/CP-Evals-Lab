"""Evaluation sub-package: heuristics, score processing, and data loading.

All public symbols are re-exported here so existing imports from
``parrot_ai.llm_evals`` continue to work.
"""

from .data_loading import load_qa_pairs, load_eval_questions
from .score_validation import has_rubric_scores, is_scored_record
from .arabic_heuristics import (
    ARABIC_BLOCKS,
    is_arabic_char,
    basic_language_metrics,
    apply_purity_penalty,
    has_arabic_scripture_citation,
    has_arabic_theological_terminology,
    calibrate_arabic_scores,
)
from .english_heuristics import (
    has_scripture_citation,
    has_theological_terminology,
    calibrate_english_scores,
)
from .score_processing import (
    clamp_overall,
    clamp_all_overalls,
    clamp_scale_scores,
    enforce_knockouts,
    compute_weighted_final_score,
)
from .aggregation import (
    ARABIC_ACCURACY_SUBCRITERIA,
    CORE_SECTION_ORDER,
    CORE_SECTION_SUBCRITERIA,
    ENGLISH_SECTION_WEIGHTS,
    FINAL_OVERALL_ROW,
    META_ROWS,
    ScoreStatistic,
    WEIGHTED_SCORE_ROW,
    aggregate_score_statistics,
    aggregate_scores,
    build_rows_order,
    is_applicable,
    score_metric_order,
)
from .master_csv import (
    MASTER_METADATA_COLUMNS,
    ResultSource,
    build_master_row,
    master_fieldnames,
    metric_column_prefix,
    write_master_csv,
)
from .result_registry import (
    REGISTRY_SCHEMA_VERSION,
    append_result_sources,
    load_result_registry,
    validate_result_source,
)
from .judge_calibration import (
    EvaluationRun,
    MetricPath,
    MetricStatistics,
    PairedAnswerSet,
    ScoredItem,
    analyze_study_responses,
    available_contexts,
    available_judges,
    calculate_metric_statistics,
    collect_scored_items,
    create_calibration_study,
    list_reviewers,
    list_study_directories,
    load_evaluation_runs,
    load_reviewer_responses,
    load_study,
    pair_answer_sets,
    rank_metric_statistics,
    save_reviewer_response,
    save_study,
    shared_answer_set_labels,
)

__all__ = [
    # Data loading
    "load_qa_pairs",
    "load_eval_questions",
    "has_rubric_scores",
    "is_scored_record",
    # Arabic heuristics
    "ARABIC_BLOCKS",
    "is_arabic_char",
    "basic_language_metrics",
    "apply_purity_penalty",
    "has_arabic_scripture_citation",
    "has_arabic_theological_terminology",
    "calibrate_arabic_scores",
    # English heuristics
    "has_scripture_citation",
    "has_theological_terminology",
    "calibrate_english_scores",
    # Score processing
    "clamp_overall",
    "clamp_all_overalls",
    "clamp_scale_scores",
    "enforce_knockouts",
    "compute_weighted_final_score",
    # Aggregation and reporting
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
    # Vertical master CSV
    "MASTER_METADATA_COLUMNS",
    "ResultSource",
    "build_master_row",
    "master_fieldnames",
    "metric_column_prefix",
    "write_master_csv",
    # Persistent result registry
    "REGISTRY_SCHEMA_VERSION",
    "append_result_sources",
    "load_result_registry",
    "validate_result_source",
    # Human judge calibration
    "EvaluationRun",
    "MetricPath",
    "MetricStatistics",
    "PairedAnswerSet",
    "ScoredItem",
    "analyze_study_responses",
    "available_contexts",
    "available_judges",
    "calculate_metric_statistics",
    "collect_scored_items",
    "create_calibration_study",
    "list_reviewers",
    "list_study_directories",
    "load_evaluation_runs",
    "load_reviewer_responses",
    "load_study",
    "pair_answer_sets",
    "rank_metric_statistics",
    "save_reviewer_response",
    "save_study",
    "shared_answer_set_labels",
]
