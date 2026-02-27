"""Evaluation sub-package: heuristics, score processing, and data loading.

All public symbols are re-exported here so existing imports from
``parrot_ai.llm_evals`` continue to work.
"""

from .data_loading import load_qa_pairs, load_eval_questions
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

__all__ = [
    # Data loading
    "load_qa_pairs",
    "load_eval_questions",
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
]
