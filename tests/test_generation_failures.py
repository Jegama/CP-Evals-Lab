"""A model's refusal to answer counts; a judge or transport failure never does."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import cp_eval_llms as cli
import cp_eval_benchmark as benchmark
import parrot_ai.llm_evaluation as evaluation
from parrot_ai.generation_failures import TerminalGenerationFailure, load_generation_failures
from parrot_ai.llm_evals.benchmark_config import load_benchmark_config
from parrot_ai.llm_evals.aggregation import aggregate_score_statistics
from parrot_ai.llm_evals.master_csv import ResultSource, build_master_row
from parrot_ai.llm_evals.score_validation import is_scored_record, minimum_rubric_scores


FAILURE = {"reason": "RECITATION", "terminal": True}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def dataset_record(question, *, failure=None):
    record = {
        "messages": [{"role": "user", "content": question},
                     {"role": "assistant", "content": "" if failure else "Answer in Christ. Ephesians 2:8-9."}],
        "provider": "google", "gen_model": "gemini-test", "system_prompt_label": "v1_4",
    }
    if failure:
        record["generation_failure"] = failure
    return record




@pytest.mark.parametrize("judge", ["gpt-test", "claude-test", "gemini-test"])
def test_499_judge_calls_plus_one_penalty_keep_denominator_500_and_resume(tmp_path, monkeypatch, judge):
    monkeypatch.chdir(tmp_path)
    questions = [f"q{i}" for i in range(1, 501)]
    question_file = tmp_path / "questions.txt"
    question_file.write_text("\n".join(questions))
    dataset = tmp_path / "answers.jsonl"
    write_jsonl(dataset, [dataset_record(q, failure=FAILURE if q == "q253" else None) for q in questions])
    engine = evaluation.EvaluationEngine(model=judge, language="english", initialize_judge=False)
    rubric = {section: {field: 5 for field in fields} for section, fields in minimum_rubric_scores().items()}
    engine.evaluate = Mock(return_value=rubric)
    monkeypatch.setattr(cli, "EvaluationEngine", lambda **kwargs: engine)
    results = tmp_path / "reviews.jsonl"
    args = ["--mode", "dataset", "--dataset", str(dataset), "--answers-label", "answers",
            "--judge-model", judge, "--results-jsonl", str(results), "--language", "english",
            "--questions-file", str(question_file), "--limit", "0", "--require-complete-answers",
            "--skip-comparison-csv", "--no-progress"]
    assert cli.main(args) == 0
    records = [json.loads(line) for line in results.read_text().splitlines()]
    assert len(records) == 500 and all(is_scored_record(row) for row in records)
    penalty = next(row for row in records if row["question"] == "q253")
    assert penalty["evaluation"] == minimum_rubric_scores()
    assert penalty["generation_failure"] == FAILURE
    assert penalty["evaluation_source"] == "generation_failure_penalty"
    assert engine.evaluate.call_count == 499
    assert all(call.args[0] != "q253" and call.args[1] for call in engine.evaluate.call_args_list)
    tags = {q: {} for q in questions}
    stats = aggregate_score_statistics(records, False, question_tags=tags)
    assert stats[("Adherence", "Core")].count == 500
    assert stats[("Adherence", "Core")].mean == 4.99
    master = build_master_row(ResultSource(results), repo_root=tmp_path, question_tags=tags)
    assert master["question_count"] == 500
    assert master["generation_failure_count"] == 1
    assert master["error_count"] == 0
    original = results.read_bytes()
    assert cli.main(args) == 0
    assert results.read_bytes() == original
    assert engine.evaluate.call_count == 499




def test_sync_generation_records_recitation_and_continues(monkeypatch):
    wrapper = Mock()
    wrapper.generate.side_effect = ["answer", TerminalGenerationFailure("RECITATION"), "answer"]
    monkeypatch.setattr(evaluation, "ParrotAIGemini", lambda **kwargs: wrapper)
    engine = evaluation.EvaluationEngine(language="english", initialize_judge=False)
    checkpoint = Mock()
    records = engine.generate_responses(["q1", "q2", "q3"], provider="google", on_result=checkpoint, progress=False)
    assert not records.aborted
    assert records[1]["generation_failure"] == FAILURE
    assert "error" not in records[1] and records[1]["answer"] == ""
    assert wrapper.generate.call_count == checkpoint.call_count == 3


@pytest.mark.parametrize("failure", [{"reason": "NETWORK_ERROR", "terminal": True},
                                    {"reason": "RECITATION", "terminal": False},
                                    {"reason": ["RECITATION"], "terminal": True}])
def test_unknown_or_retryable_failures_cannot_unlock_judging(tmp_path, monkeypatch, failure):
    monkeypatch.chdir(tmp_path)
    questions = tmp_path / "questions.txt"
    questions.write_text("q")
    dataset = tmp_path / "answers.jsonl"
    write_jsonl(dataset, [dataset_record("q", failure=failure)])
    engine = evaluation.EvaluationEngine(language="english", initialize_judge=False)
    engine.batch_evaluate = Mock(side_effect=AssertionError("No judge calls allowed"))
    monkeypatch.setattr(cli, "EvaluationEngine", lambda **kwargs: engine)
    assert cli.main(["--mode", "dataset", "--dataset", str(dataset), "--answers-label", "answers",
                     "--language", "english", "--questions-file", str(questions), "--limit", "0",
                     "--require-complete-answers", "--no-progress"]) == 1
    engine.batch_evaluate.assert_not_called()


def test_failure_penalty_must_use_all_minimum_scores():
    record = {"question": "q", "answer": "", "generation_failure": FAILURE,
              "evaluation_source": "generation_failure_penalty", "evaluation": minimum_rubric_scores()}
    assert is_scored_record(record)
    record["evaluation"]["Adherence"]["Core"] = 5
    assert not is_scored_record(record)


def test_later_valid_answer_replaces_the_local_failure_penalty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    questions = tmp_path / "questions.txt"
    questions.write_text("q")
    dataset = tmp_path / "answers.jsonl"
    write_jsonl(dataset, [dataset_record("q", failure=FAILURE), dataset_record("q")])
    reviews = tmp_path / "reviews.jsonl"
    penalty = {"question": "q", "answer": "", "generation_failure": FAILURE,
               "evaluation_source": "generation_failure_penalty", "evaluation": minimum_rubric_scores()}
    write_jsonl(reviews, [penalty])
    original = reviews.read_bytes()
    engine = evaluation.EvaluationEngine(language="english", initialize_judge=False)
    engine.evaluate = Mock(return_value={"Adherence": {"Core": 4}})
    monkeypatch.setattr(cli, "EvaluationEngine", lambda **kwargs: engine)
    assert cli.main(["--mode", "dataset", "--dataset", str(dataset), "--answers-label", "answers",
                     "--results-jsonl", str(reviews), "--language", "english", "--questions-file", str(questions),
                     "--limit", "0", "--require-complete-answers", "--skip-comparison-csv", "--no-progress"]) == 0
    engine.evaluate.assert_called_once_with("q", "Answer in Christ. Ephesians 2:8-9.")
    assert reviews.read_bytes().startswith(original)
    master = build_master_row(ResultSource(reviews), repo_root=tmp_path)
    assert master["question_count"] == 1
    assert master["generation_failure_count"] == 0


def test_scheduler_accepts_recorded_model_failure_but_rejects_unresolved_blank(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "load_eval_questions", lambda *args, **kwargs: ["q1", "q2"])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"system_prompt_label": "v1_4", "api_evals_dir": str(tmp_path),
                                       "models": [{"provider": "google", "model": "gemini-test"}]}))
    config = load_benchmark_config(config_path, tmp_path)
    operation = benchmark.generation_operations(config)[0]
    dataset = Path(operation.command[operation.command.index("--output-dataset") + 1])
    write_jsonl(dataset, [dataset_record("q1"), dataset_record("q2", failure=FAILURE)])
    benchmark._validate_operation_output(config, operation, effective_repo_root=tmp_path)
    write_jsonl(dataset, [dataset_record("q1")])
    with pytest.raises(ValueError, match="1/2 answers missing"):
        benchmark._validate_operation_output(config, operation, effective_repo_root=tmp_path)
