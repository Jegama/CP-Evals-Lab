"""Tests for the append-only master-CSV result registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parrot_ai.llm_evals.master_csv import ResultSource
from parrot_ai.llm_evals.result_registry import (
    append_result_sources,
    load_result_registry,
)


def _write_result(path: Path, *, valid: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "question": "Question",
        "answer": "Answer",
        "evaluation": {"Adherence": {"Core": 4}} if valid else None,
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_registry_appends_in_plan_order_and_deduplicates(tmp_path: Path):
    first = tmp_path / "data/first.jsonl"
    second = tmp_path / "data/second.jsonl"
    third = tmp_path / "data/third.jsonl"
    for path in (first, second, third):
        _write_result(path)

    registry = tmp_path / "benchmark_configs/registry.json"
    assert (
        append_result_sources(
            registry,
            [ResultSource(first), ResultSource(second)],
            repo_root=tmp_path,
        )
        == 2
    )
    assert (
        append_result_sources(
            registry,
            [ResultSource(second), ResultSource(third)],
            repo_root=tmp_path,
        )
        == 1
    )

    loaded = load_result_registry(registry, repo_root=tmp_path)
    assert [source.path for source in loaded] == [
        first.resolve(),
        second.resolve(),
        third.resolve(),
    ]
    payload = json.loads(registry.read_text(encoding="utf-8"))
    assert [item["path"] for item in payload["results"]] == [
        "data/first.jsonl",
        "data/second.jsonl",
        "data/third.jsonl",
    ]


def test_invalid_result_does_not_modify_registry(tmp_path: Path):
    valid = tmp_path / "data/valid.jsonl"
    invalid = tmp_path / "data/invalid.jsonl"
    _write_result(valid)
    _write_result(invalid, valid=False)
    registry = tmp_path / "registry.json"
    append_result_sources(
        registry,
        [ResultSource(valid)],
        repo_root=tmp_path,
    )
    before = registry.read_bytes()

    with pytest.raises(ValueError, match="no valid score records"):
        append_result_sources(
            registry,
            [ResultSource(invalid)],
            repo_root=tmp_path,
        )

    assert registry.read_bytes() == before


def test_default_registry_retains_seeded_runs_and_valid_unique_paths():
    repo_root = Path(__file__).resolve().parent.parent
    registry = repo_root / "benchmark_configs/english_api_registry.json"

    sources = load_result_registry(registry, repo_root=repo_root)

    assert len(sources) >= 25
    assert len({source.path for source in sources}) == len(sources)
    assert all(source.path.exists() for source in sources)


@pytest.mark.parametrize("evaluation", [
    {},
    {"Adherence": {}, "Interfaith_Sensitivity": {}, "Arabic_Accuracy": {}},
    {"Kindness_and_Gentleness": {"Pastoral_Acknowledgement": "yes"}},
    {"Arabic_Accuracy": {"Heuristic_Arabic_Purity_Pct": 100}},
    {"Adherence": {"Core": True}},
    {"Adherence": {"Core": "4"}},
    {"Adherence": {"unknown": 4}},
])
def test_empty_evaluation_object_cannot_be_registered(tmp_path, evaluation):
    results = tmp_path / "empty-scores.jsonl"
    results.write_text(json.dumps({"question": "Question", "answer": "Answer", "evaluation": evaluation}) + "\n")
    registry = tmp_path / "registry.json"
    with pytest.raises(ValueError, match="no valid score records"):
        append_result_sources(registry, [ResultSource(results)], repo_root=tmp_path)
    assert not registry.exists()
