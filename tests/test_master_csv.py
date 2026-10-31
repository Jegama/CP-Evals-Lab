"""Tests for retrospective vertical master-row construction."""

import json
from pathlib import Path

import pytest

from parrot_ai.llm_evals.master_csv import ResultSource, build_master_row, write_master_csv


def _score_payload(value: int) -> dict:
    return {
        "Adherence": {
            "Core": value,
            "Secondary": value,
            "Tertiary_Handling": value,
            "Biblical_Basis": value,
            "Consistency": value,
            "Overall": value,
        },
        "Kindness_and_Gentleness": {
            "Core_Clarity_with_Kindness": value,
            "Pastoral_Sensitivity": value,
            "Secondary_Fairness": value,
            "Tertiary_Neutrality": value,
            "Tone": value,
            "Overall": value,
        },
        "Interfaith_Sensitivity": {
            "Respect_and_Handling_Objections": value,
            "Objection_Acknowledgement": value,
            "Evangelism": value,
            "Gospel_Boldness": value,
            "Overall": value,
        },
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_build_master_row_infers_answer_identity_and_latest_timestamp(tmp_path: Path):
    dataset = (
        tmp_path
        / "data/english/api_evals/generated_api_openai_openai-model-v1_4.jsonl"
    )
    _write_jsonl(
        dataset,
        [
            {"role": "system", "content": "prompt"},
            {
                "messages": [],
                "provider": "openai",
                "gen_model": "model",
                "system_prompt_label": "v1_4",
                "use_system_prompt": True,
            },
        ],
    )
    results = tmp_path / "data/english/api_evals/results.jsonl"
    base_meta = {
        "dataset": str(dataset.relative_to(tmp_path)),
        "answers_label": "openai-model-v1_4-2",
        "judge_model": "judge-model",
        "language": "english",
    }
    _write_jsonl(
        results,
        [
            {
                **base_meta,
                "question": "Q1",
                "answer": "A1",
                "evaluation": _score_payload(1),
                "timestamp": "2026-01-01T10:00:00",
            },
            {
                **base_meta,
                "question": "Q2",
                "answer": "A2",
                "evaluation": _score_payload(5),
                "timestamp": "2026-01-02T10:00:00",
            },
        ],
    )

    row = build_master_row(
        ResultSource(results, "v2"),
        repo_root=tmp_path,
        timezone_name="America/Chicago",
    )

    assert row["answers_label"] == "openai-model-v1_4"
    assert row["provider"] == "openai"
    assert row["gen_model"] == "model"
    assert row["judge_model"] == "judge-model"
    assert row["evaluated_at"] == "2026-01-02T10:00:00-06:00"
    assert row["timestamp_source"] == "result_metadata"
    assert row["question_count"] == 2
    assert row["adherence_core_mean"] == 3.0
    assert row["adherence_core_stdev"] == 2.0
    assert str(row["run_id"]).startswith("eval_")


@pytest.mark.parametrize("later_record", [
    {"evaluation": {"Adherence": {}, "Arabic_Accuracy": {}}},
    {"evaluation": _score_payload(1), "error": "failed retry"},
])
def test_scoreless_or_failed_retry_preserves_valid_master_scores(tmp_path, later_record):
    results = tmp_path / "results.jsonl"
    valid = {"question": "Q1", "answer": "A1", "evaluation": _score_payload(5)}
    _write_jsonl(results, [valid, {"question": "Q1", "answer": "A1", **later_record}])

    row = build_master_row(ResultSource(results), repo_root=tmp_path)

    assert row["question_count"] == 1
    assert row["adherence_core_mean"] == 5.0


def test_master_rejects_scoreless_evaluations(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [{
        "question": "Q1", "answer": "A1", "evaluation": {"Adherence": {}},
    }])

    with pytest.raises(ValueError, match="No valid evaluations"):
        build_master_row(ResultSource(results), repo_root=tmp_path)


@pytest.mark.parametrize("author_dataset_exists", [True, False])
def test_master_csv_is_portable_with_legacy_absolute_dataset_paths(
    tmp_path: Path, author_dataset_exists: bool,
):
    author_root = tmp_path / "author" / "CP-Evals-Lab"
    relocated_root = tmp_path / "ci" / "renamed-checkout"
    dataset_relative = Path(
        "data/english/api_evals/generated_api_openai_openai-model-v1_4.jsonl"
    )
    results_relative = Path("data/english/api_evals/results.jsonl")
    original_dataset = author_root / dataset_relative
    dataset_records = [{
        "messages": [], "provider": "openai", "gen_model": "model",
        "system_prompt_label": "v1_4", "use_system_prompt": True,
    }]
    result_records = [{
        "dataset": str(original_dataset),
        "answers_label": "openai-model-v1_4-2", "judge_model": "judge-model",
        "timestamp": "2026-01-02T10:00:00-06:00",
        "question": "Q1", "answer": "A1", "evaluation": _score_payload(4),
    }]
    for root in (author_root, relocated_root):
        _write_jsonl(root / dataset_relative, dataset_records)
        _write_jsonl(root / results_relative, result_records)

    expected_rows = write_master_csv(
        [ResultSource(results_relative)], Path("master.csv"), repo_root=author_root,
    )
    if not author_dataset_exists:
        original_dataset.unlink()
    originals = {
        path: path.read_bytes()
        for root in (author_root, relocated_root)
        for path in root.rglob("*.jsonl")
    }

    rows = write_master_csv(
        [ResultSource(results_relative)], Path("master.csv"), repo_root=relocated_root,
    )

    assert rows == expected_rows
    assert rows[0]["source_dataset"] == dataset_relative.as_posix()
    assert rows[0]["gen_model"] == "model"
    assert (relocated_root / "master.csv").read_bytes() == (author_root / "master.csv").read_bytes()
    assert all(path.read_bytes() == data for path, data in originals.items())


@pytest.mark.parametrize("candidate_kind", ["unrelated", "missing", "directory", "symlink"])
def test_master_csv_does_not_rebase_unverified_external_dataset_paths(
    tmp_path: Path, candidate_kind: str,
):
    repo_root = tmp_path / "checkout"
    project_name = "another-project" if candidate_kind == "unrelated" else "CP-Evals-Lab"
    relative = Path("data/english/api_evals/answers.jsonl")
    external = tmp_path / "external" / project_name / relative
    _write_jsonl(external, [{"provider": "openai", "gen_model": "external-model"}])
    candidate = repo_root / relative
    candidate.parent.mkdir(parents=True)
    if candidate_kind == "unrelated":
        _write_jsonl(candidate, [{"provider": "google", "gen_model": "wrong-model"}])
    elif candidate_kind == "missing":
        # An identical basename elsewhere is not evidence of dataset identity.
        _write_jsonl(repo_root / "other" / candidate.name, [{"provider": "google"}])
    elif candidate_kind == "directory":
        candidate.mkdir()
    else:
        candidate.symlink_to(external)
    results = repo_root / "results.jsonl"
    _write_jsonl(results, [{
        "dataset": str(external), "timestamp": "2026-01-02T10:00:00-06:00",
        "question": "Q1", "answer": "A1", "evaluation": _score_payload(4),
    }])

    row = build_master_row(ResultSource(results), repo_root=repo_root)

    assert row["source_dataset"] == str(external.resolve())
    assert row["gen_model"] == "external-model"


@pytest.mark.parametrize("error_value", ["transient judge failure", "", None, False, 0])
def test_build_master_row_deduplicates_failed_attempt_before_successful_retry(
    tmp_path: Path, error_value,
):
    results = tmp_path / "results.jsonl"
    _write_jsonl(
        results,
        [
            {
                "question": "Q1",
                "answer": "A1",
                "error": error_value,
            },
            {
                "question": "Q1",
                "answer": "A1",
                "evaluation": _score_payload(5),
            },
            {
                "question": "Q2",
                "answer": "A2",
                "evaluation": _score_payload(1),
            },
        ],
    )

    row = build_master_row(ResultSource(results, "v2"), repo_root=tmp_path)

    assert row["question_count"] == 2
    assert row["error_count"] == 1
    assert row["adherence_core_mean"] == 3.0
    assert row["adherence_core_stdev"] == 2.0
