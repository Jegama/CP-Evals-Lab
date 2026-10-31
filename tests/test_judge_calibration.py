"""Tests for dynamic judge disagreement and blinded study workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parrot_ai.llm_evals.judge_calibration import (
    EvaluationRun,
    MetricPath,
    PairedAnswerSet,
    ScoredItem,
    analyze_study_responses,
    calculate_metric_statistics,
    collect_scored_items,
    create_calibration_study,
    extract_rubric_guidance,
    format_rubric_guidance_markdown,
    load_evaluation_runs,
    load_reviewer_responses,
    load_study,
    rank_metric_statistics,
    save_reviewer_response,
    save_study,
    shared_answer_set_labels,
)
from parrot_ai.llm_evals.master_csv import ResultSource


def _run(
    *,
    judge: str,
    answers_label: str,
    evaluations: dict[str, dict],
    answers: dict[str, str],
) -> EvaluationRun:
    return EvaluationRun(
        source_path=Path(f"{answers_label}-{judge}.jsonl"),
        registry_index=0,
        eval_version="v2",
        system_prompt_label="v1_4",
        language="english",
        judge_model=judge,
        answers_label=answers_label,
        provider="provider",
        gen_model=answers_label,
        records_by_question={
            question: {
                "question": question,
                "answer": answers[question],
                "evaluation": evaluation,
            }
            for question, evaluation in evaluations.items()
        },
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def test_loader_preserves_scores_after_empty_retry(tmp_path: Path):
    results = tmp_path / "results.jsonl"
    valid = {
        "question": "Q1", "answer": "A1", "evaluation": {"Adherence": {"Core": 4}},
        "answers_label": "answers", "judge_model": "judge-model",
    }
    _write_jsonl(results, [
        valid,
        {"question": "Q1", "answer": "A1", "evaluation": {"Adherence": {}}},
        {"question": "Q2", "answer": "A2", "evaluation": {"Adherence": {}}},
    ])

    runs = load_evaluation_runs([ResultSource(results)], repo_root=tmp_path)

    assert len(runs) == 1
    assert runs[0].records_by_question == {"Q1": valid}


def test_loader_normalizes_legacy_labels_using_dataset_metadata(tmp_path: Path):
    dataset = (
        tmp_path
        / "data/english/api_evals"
        / "generated_api_openai_openai-gpt-5-mini-v1_4.jsonl"
    )
    _write_jsonl(
        dataset,
        [
            {
                "messages": [],
                "provider": "openai",
                "gen_model": "gpt-5-mini",
                "system_prompt_label": "v1_4",
            }
        ],
    )
    result_template = {
        "question": "Q1",
        "answer": "A1",
        "dataset": str(dataset.relative_to(tmp_path)),
        "language": "english",
        "evaluation": {"Adherence": {"Core": 4}},
    }
    primary_results = tmp_path / "primary.jsonl"
    comparison_results = tmp_path / "comparison.jsonl"
    _write_jsonl(
        primary_results,
        [
            {
                **result_template,
                "answers_label": "openai-gpt-5-mini-v1_4",
                "judge_model": "gpt-5-mini",
            }
        ],
    )
    _write_jsonl(
        comparison_results,
        [
            {
                **result_template,
                "answers_label": "openai-gpt-5-mini-v1_4-2",
                "judge_model": "gpt-5.4-mini",
            }
        ],
    )

    runs = load_evaluation_runs(
        [
            ResultSource(primary_results, "v2"),
            ResultSource(comparison_results, "v2"),
        ],
        repo_root=tmp_path,
    )

    assert [run.answers_label for run in runs] == [
        "openai-gpt-5-mini-v1_4",
        "openai-gpt-5-mini-v1_4",
    ]
    assert all(run.provider == "openai" for run in runs)
    assert all(run.gen_model == "gpt-5-mini" for run in runs)
    assert all(run.system_prompt_label == "v1_4" for run in runs)
    assert shared_answer_set_labels(
        runs,
        language="english",
        eval_version="v2",
        system_prompt_label="v1_4",
        primary_judge="gpt-5-mini",
        comparison_judge="gpt-5.4-mini",
    ) == ["openai-gpt-5-mini-v1_4"]


def test_collect_scored_items_discovers_metrics_and_uses_existing_applicability():
    answers = {"Q1": "A1", "Q2": "A2"}
    primary = _run(
        judge="primary",
        answers_label="answers",
        answers=answers,
        evaluations={
            "Q1": {
                "Adherence": {"Core": 5},
                "Kindness_and_Gentleness": {"Tone": 4},
                "Future_Section": {"Future_Metric": 2},
            },
            "Q2": {
                "Adherence": {"Core": 4},
                "Kindness_and_Gentleness": {"Tone": 3},
                "Future_Section": {"Future_Metric": 3},
            },
        },
    )
    comparison = _run(
        judge="comparison",
        answers_label="answers",
        answers=answers,
        evaluations={
            "Q1": {
                "Adherence": {"Core": 3},
                "Kindness_and_Gentleness": {"Tone": 4},
                "Future_Section": {"Future_Metric": 5},
            },
            "Q2": {
                "Adherence": {"Core": 2},
                "Kindness_and_Gentleness": {"Tone": 2},
                "Future_Section": {"Future_Metric": 4},
            },
        },
    )
    paired = PairedAnswerSet(
        answers_label="answers",
        provider="provider",
        gen_model="model",
        primary=primary,
        comparison=comparison,
    )

    items = collect_scored_items(
        [paired],
        question_tags={
            "Q1": {"applies_core_doctrine": False},
            "Q2": {"applies_core_doctrine": True},
        },
    )

    assert len(items[MetricPath(("Adherence", "Core"))]) == 1
    assert len(items[MetricPath(("Kindness_and_Gentleness", "Tone"))]) == 2
    assert len(items[MetricPath(("Future_Section", "Future_Metric"))]) == 2


def test_composite_ranking_selects_dynamic_high_disagreement_metric():
    calm = MetricPath(("Any_Section", "Calm"))
    disputed = MetricPath(("Another_Section", "Disputed"))
    items = {
        calm: [
            ScoredItem(calm, f"Q{i}", "A", "model", "p", "m", score, score)
            for i, score in enumerate((1, 2, 3, 4), start=1)
        ],
        disputed: [
            ScoredItem(
                disputed,
                f"Q{i}",
                "A",
                "model",
                "p",
                "m",
                primary,
                comparison,
            )
            for i, (primary, comparison) in enumerate(
                ((1, 5), (2, 4), (4, 2), (5, 1)),
                start=1,
            )
        ],
    }

    statistics = calculate_metric_statistics(items)
    ranked = rank_metric_statistics(statistics)

    assert ranked[0].metric_key == disputed.key
    assert ranked[0].disagreement_index > ranked[1].disagreement_index
    assert ranked[0].pearson == pytest.approx(-1.0)
    assert ranked[0].spearman == pytest.approx(-1.0)


def test_composite_disagreement_uses_per_item_gaps_when_deltas_cancel():
    collapsed_scale = MetricPath(("Any_Section", "Collapsed_Scale"))
    items = {
        collapsed_scale: [
            ScoredItem(
                collapsed_scale,
                f"Q{i}",
                "A",
                "model",
                "p",
                "m",
                primary,
                comparison,
            )
            for i, (primary, comparison) in enumerate(
                ((1, 3), (5, 3)),
                start=1,
            )
        ]
    }

    row = calculate_metric_statistics(items)[0]

    assert row.mean_delta == 0.0
    assert row.mean_absolute_delta == 2.0
    assert row.pearson is None
    assert row.spearman is None
    assert row.disagreement_index == pytest.approx(0.5)


def _study_items(metric: MetricPath) -> list[ScoredItem]:
    items: list[ScoredItem] = []
    for question_index in range(1, 7):
        question = f"Question {question_index}"
        for answers_label, primary, comparison in (
            ("model-a", 5, 1),
            ("model-b", 3, 3),
            ("model-c", 1, 5),
        ):
            items.append(
                ScoredItem(
                    metric=metric,
                    question=question,
                    answer=f"{answers_label} response to {question}",
                    answers_label=answers_label,
                    provider=answers_label,
                    gen_model=answers_label,
                    primary_score=primary,
                    comparison_score=comparison,
                )
            )
    return items


def test_study_sampling_is_blinded_resumable_and_analyzable(tmp_path: Path):
    metric = MetricPath(("Future_Section", "Dynamic_Criterion"))
    items = _study_items(metric)
    statistics = calculate_metric_statistics({metric: items})
    instructions = """General Rules
Scores 1–5 (1 = failure, 3 = acceptable, 5 = excellent).

Future Section Sub-guidelines:
* Dynamic_Criterion (future behavior):
  5: Excellent.
  3: Acceptable.
  1: Failure.
"""
    study = create_calibration_study(
        title="Dynamic study",
        language="english",
        eval_version="v2",
        system_prompt_label="v1_4",
        primary_judge="primary",
        comparison_judge="comparison",
        answer_sets=["model-a", "model-b", "model-c"],
        ranking_method="composite",
        selected_metrics=[metric],
        all_statistics=statistics,
        items_by_metric={metric: items},
        rubric_instructions=instructions,
        pairwise_trials_per_metric=6,
        pointwise_trials_per_metric=3,
        seed=42,
    )

    assert len(study["tasks"]) == 9
    for task in study["tasks"]:
        public_payload = {key: value for key, value in task.items() if key != "internal"}
        assert "answers_label" not in public_payload
        assert "provider" not in public_payload
        assert "primary_score" not in public_payload
        assert "comparison_score" not in public_payload
        assert "Dynamic Criterion" in task["metric_label"]
        assert "5: Excellent." in task["rubric_guidance"]

    study_dir = save_study(study, tmp_path / "studies")
    assert load_study(study_dir)["study_id"] == study["study_id"]

    for task in study["tasks"]:
        if task["kind"] == "pairwise":
            direction = task["internal"]["comparison_direction"]
            choice = {1: "A", 0: "Tie", -1: "B"}[direction]
        else:
            choice = str(task["internal"]["comparison_score"])
        save_reviewer_response(
            study_dir,
            "reviewer-one",
            task_id=task["task_id"],
            choice=choice,
        )

    payload = load_reviewer_responses(study_dir, "reviewer-one")
    assert len(payload["responses"]) == 9
    analysis = analyze_study_responses(study, payload)
    conflict = next(
        row
        for row in analysis["pairwise"]
        if row["slice"] == "judge_conflicts"
    )
    pointwise = analysis["pointwise"][0]

    assert conflict["comparison_accuracy"] == 1.0
    assert conflict["comparison_accuracy"] > conflict["primary_accuracy"]
    assert pointwise["comparison_mae"] == 0.0
    assert pointwise["comparison_exact"] == 1.0


def test_rubric_guidance_falls_back_for_a_new_unanchored_metric():
    guidance = extract_rubric_guidance(
        "General Rules\nScores 1–5 (1 = failure, 5 = excellent).",
        MetricPath(("New", "Unseen")),
    )

    assert "Scores 1–5" in guidance
    assert "New › Unseen" in guidance


def test_rubric_guidance_markdown_puts_score_anchors_on_separate_rows():
    guidance = """Scores 1–5.

* Consistency:
  5: Fully coherent.
  3: Mostly coherent.
  1: Incoherent."""
    hard_break = "  \n"

    assert format_rubric_guidance_markdown(guidance) == (
        "Scores 1–5.\n\n"
        f"* Consistency:{hard_break}"
        f"  5: Fully coherent.{hard_break}"
        f"  3: Mostly coherent.{hard_break}"
        "  1: Incoherent."
    )


def test_response_rejects_invalid_choice(tmp_path: Path):
    metric = MetricPath(("Section", "Criterion"))
    items = _study_items(metric)
    statistics = calculate_metric_statistics({metric: items})
    study = create_calibration_study(
        title="Invalid choice",
        language="english",
        eval_version="v2",
        system_prompt_label="v1_4",
        primary_judge="primary",
        comparison_judge="comparison",
        answer_sets=["model-a", "model-b", "model-c"],
        ranking_method="composite",
        selected_metrics=[metric],
        all_statistics=statistics,
        items_by_metric={metric: items},
        rubric_instructions="Scores 1–5.",
        pairwise_trials_per_metric=1,
        pointwise_trials_per_metric=0,
        seed=7,
    )
    study_dir = save_study(study, tmp_path)

    with pytest.raises(ValueError, match="Invalid choice"):
        save_reviewer_response(
            study_dir,
            "reviewer",
            task_id=study["tasks"][0]["task_id"],
            choice="primary judge wins",
        )
