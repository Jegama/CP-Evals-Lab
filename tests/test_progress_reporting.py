"""Tests for benchmark subprocess progress reporting."""

import json

import parrot_ai.llm_evaluation as evaluation_module
from parrot_ai.llm_evaluation import EvaluationEngine
from parrot_ai.llm_evals.progress_reporting import (
    PROGRESS_FILE_ENV,
    emit_progress,
)


def test_emit_progress_writes_json_lines_only_when_requested(tmp_path, monkeypatch):
    progress_path = tmp_path / "progress.log"

    emit_progress("judging", 1, 2)
    assert not progress_path.exists()

    monkeypatch.setenv(PROGRESS_FILE_ENV, str(progress_path))
    emit_progress("judging", 0, 2)
    emit_progress("judging", 1, 2)

    events = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events == [
        {"stage": "judging", "current": 0, "total": 2},
        {"stage": "judging", "current": 1, "total": 2},
    ]


def test_batch_evaluate_emits_progress_when_local_tqdm_is_disabled(monkeypatch):
    engine = object.__new__(EvaluationEngine)
    engine.evaluate = lambda question, answer: {"Adherence": {"Core": len(question + answer)}}
    events = []
    monkeypatch.setattr(
        evaluation_module,
        "emit_progress",
        lambda stage, current, total: events.append((stage, current, total)),
    )

    results = engine.batch_evaluate(
        [("q1", "a1"), ("q2", "a2")],
        progress=False,
    )

    assert len(results) == 2
    assert events == [
        ("judging", 0, 2),
        ("judging", 1, 2),
        ("judging", 2, 2),
    ]


def test_generate_responses_emits_progress_when_local_tqdm_is_disabled(
    monkeypatch,
):
    class FakeWrapper:
        def __init__(self, language):
            self.language = language

        def set_model(self, model):
            self.model = model

        def generate(self, prompt, model, system):
            return f"answer to {prompt}"

    engine = object.__new__(EvaluationEngine)
    engine.language = "english"
    events = []
    monkeypatch.setattr(evaluation_module, "ParrotAIOpenAI", FakeWrapper)
    monkeypatch.setattr(
        evaluation_module,
        "emit_progress",
        lambda stage, current, total: events.append((stage, current, total)),
    )

    responses = engine.generate_responses(
        ["q1", "q2"],
        provider="openai",
        model="test-model",
        progress=False,
    )

    assert [response["answer"] for response in responses] == [
        "answer to q1",
        "answer to q2",
    ]
    assert events == [
        ("generating", 0, 2),
        ("generating", 1, 2),
        ("generating", 2, 2),
    ]
