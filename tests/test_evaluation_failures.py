"""Offline regression coverage for SDK failures, input validation, and resume."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import anthropic
import pytest

import cp_eval_llms as cli
import parrot_ai.llm_evaluation as evaluation_module
from parrot_ai.core import ParrotAIClaude
from parrot_ai.llm_evaluation import EvaluationEngine


def _engine():
    engine = object.__new__(EvaluationEngine)
    engine.language = "english"
    engine.model = "claude-haiku-5-5"
    engine.instructions = "Score the answer"
    engine.system_prompt = "System prompt"
    return engine


@pytest.mark.parametrize("entrypoint", ["judge", "structured_generation"])
def test_anthropic_calls_match_installed_sdk_signature(entrypoint):
    # Autospec uses the real installed SDK signature, without issuing requests.
    with anthropic.Anthropic(api_key="test-only") as client:
        parse = create_autospec(
            client.messages.parse,
            return_value=SimpleNamespace(parsed_output={"Adherence": {"Core": 4}}),
        )
        fake_client = SimpleNamespace(messages=SimpleNamespace(parse=parse))
        if entrypoint == "judge":
            engine = _engine()
            engine.client = fake_client
            result = engine.evaluate("What is grace?", "God's favor in Christ.")
        else:
            wrapper = object.__new__(ParrotAIClaude)
            wrapper._client = fake_client
            result = wrapper.generate_structured(
                [{"role": "user", "content": "Question"}], dict,
                model="claude-haiku-5-5",
            )
    assert result["Adherence"]["Core"] == 4
    assert parse.call_count == 1
    assert "temperature" not in parse.call_args.kwargs


@pytest.mark.parametrize("error", [TypeError("unsupported argument"), AttributeError("missing method")])
def test_permanent_sdk_error_stops_after_one_attempt(error, capsys):
    engine = _engine()
    engine.evaluate = Mock(side_effect=error)
    results = engine.batch_evaluate([(f"q{i}", "answer") for i in range(500)], progress=False)
    assert engine.evaluate.call_count == 1
    assert len(results) == 1 and results[0]["error"] == str(error)
    assert str(error) in capsys.readouterr().out


def test_repeated_transient_failures_stop_but_preserve_prior_success():
    engine = _engine()
    engine.evaluate = Mock(side_effect=[{"score": 4}, *[RuntimeError("unavailable")] * 3])
    results = engine.batch_evaluate([(f"q{i}", "answer") for i in range(500)], progress=False)
    assert engine.evaluate.call_count == 4
    assert results[0]["evaluation"] == {"score": 4}
    assert len(results) == 4


def test_isolated_failures_do_not_stop_subsequent_successes():
    engine = _engine()
    outcomes = [RuntimeError("temporary"), {"score": 4}] * 4
    engine.evaluate = Mock(side_effect=outcomes)
    results = engine.batch_evaluate([(f"q{i}", "answer") for i in range(8)], progress=False)
    assert len(results) == 8
    assert sum("evaluation" in r for r in results) == 4


def test_authentication_failure_stops_immediately():
    engine = _engine()
    error = RuntimeError("Unauthorized")
    error.status_code = 401
    engine.evaluate = Mock(side_effect=error)
    assert len(engine.batch_evaluate([("q", "a")] * 500, progress=False)) == 1
    assert engine.evaluate.call_count == 1


def test_blank_answer_penalty_never_calls_a_judge():
    engine = _engine()
    engine.client = Mock()
    result = engine.evaluate("Question", "  \n")
    assert result["Adherence"]["Overall"] == 1
    engine.client.messages.parse.assert_not_called()


@pytest.mark.parametrize("outcome,attempts", [(" ", 3), (TypeError("bad SDK argument"), 1)])
def test_generation_stops_on_empty_answers_or_broken_sdk(monkeypatch, outcome, attempts):
    wrapper = Mock()
    if isinstance(outcome, Exception):
        wrapper.generate.side_effect = outcome
    else:
        wrapper.generate.return_value = outcome
    monkeypatch.setattr(evaluation_module, "ParrotAIClaude", lambda **kwargs: wrapper)
    responses = _engine().generate_responses(
        [f"q{i}" for i in range(500)], provider="anthropic", progress=False,
    )
    assert wrapper.generate.call_count == attempts
    assert all("error" in r and "answer" not in r for r in responses)


def _cli_fixture(tmp_path, monkeypatch, answers):
    monkeypatch.chdir(tmp_path)
    questions = tmp_path / "questions.txt"
    questions.write_text("q1\nq2\nq3\n")
    dataset = tmp_path / "answers.jsonl"
    dataset.write_text("".join(json.dumps({"messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]}) + "\n" for question, answer in answers.items()))
    results = tmp_path / "results.jsonl"
    engine = _engine()
    monkeypatch.setattr(cli, "EvaluationEngine", lambda **kwargs: engine)
    args = [
        "--mode", "dataset", "--language", "english", "--limit", "0",
        "--dataset", str(dataset), "--questions-file", str(questions),
        "--answers-label", "answers", "--results-jsonl", str(results),
        "--require-complete-answers", "--no-progress",
    ]
    return engine, args, results


@pytest.mark.parametrize("dataset_location", ["absolute", "relative", "external"])
def test_cli_records_portable_dataset_paths(tmp_path, monkeypatch, dataset_location):
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    engine, args, results = _cli_fixture(
        repo_root, monkeypatch, {"q1": "a1", "q2": "a2", "q3": "a3"},
    )
    dataset = repo_root / "answers.jsonl"
    expected_path = "answers.jsonl"
    if dataset_location == "external":
        external = tmp_path / "external-answers.jsonl"
        dataset.rename(external)
        dataset = external
        expected_path = str(external.resolve())
        args[args.index("--dataset") + 1] = str(external)
    elif dataset_location == "relative":
        args[args.index("--dataset") + 1] = "answers.jsonl"
    original = dataset.read_bytes()
    engine.evaluate = Mock(return_value={"Adherence": {"Core": 4}})

    assert cli.main([*args, "--skip-comparison-csv"]) == 0

    saved = [json.loads(line) for line in results.read_text().splitlines()]
    assert len(saved) == 3
    assert all(record["dataset"] == expected_path for record in saved)
    assert dataset.read_bytes() == original


@pytest.mark.parametrize("answer", ["", " \n"])
def test_incomplete_dataset_is_rejected_before_judging(tmp_path, monkeypatch, answer):
    engine, args, results = _cli_fixture(tmp_path, monkeypatch, {"q1": "a1", "q2": answer})
    engine.evaluate = Mock(side_effect=AssertionError("must not call judge"))
    assert cli.main(args) == 1
    engine.evaluate.assert_not_called()
    assert not results.exists()


def test_failed_cli_preserves_successes_and_resume_only_retries_unfinished(tmp_path, monkeypatch, capsys):
    engine, args, path = _cli_fixture(tmp_path, monkeypatch, {"q1": "a1", "q2": "a2", "q3": "a3"})
    score = {"Adherence": {"Core": 4}}
    engine.evaluate = Mock(side_effect=[score, TypeError("unsupported temperature")])
    assert cli.main(args) == 1
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(saved) == 2
    assert saved[0]["evaluation"] == score
    assert saved[1]["error"] == "unsupported temperature"
    assert not (tmp_path / "data/english/training_datasets/evals/dataset_eval_comparison.csv").exists()
    assert "[done] Completed" not in capsys.readouterr().out

    before = path.read_bytes()
    engine.evaluate = Mock(return_value=score)
    assert cli.main(args) == 0
    assert [call.args[0] for call in engine.evaluate.call_args_list] == ["q2", "q3"]
    assert path.read_bytes().startswith(before)
    assert len(path.read_text().splitlines()) == 4


def test_cli_all_failed_never_reports_success(tmp_path, monkeypatch, capsys):
    engine, args, path = _cli_fixture(tmp_path, monkeypatch, {"q1": "a1", "q2": "a2", "q3": "a3"})
    engine.evaluate = Mock(side_effect=TypeError("bad SDK argument"))
    assert cli.main(args) == 1
    assert engine.evaluate.call_count == 1
    assert "error" in json.loads(path.read_text())
    output = capsys.readouterr().out
    assert "3/3 evaluations failed or were not attempted" in output
    assert "[done] Completed" not in output


def test_error_marked_dataset_record_is_not_treated_as_an_answer(tmp_path, monkeypatch):
    engine, args, path = _cli_fixture(tmp_path, monkeypatch, {"q1": "a1", "q2": "a2", "q3": "a3"})
    dataset = tmp_path / "answers.jsonl"
    records = [json.loads(line) for line in dataset.read_text().splitlines()]
    records[1]["error"] = "generation failed"
    records[1]["messages"][1]["content"] = "Provider error text, not an answer"
    dataset.write_text("".join(json.dumps(r) + "\n" for r in records))
    engine.evaluate = Mock(side_effect=AssertionError("must not call judge"))
    assert cli.main(args) == 1
    engine.evaluate.assert_not_called()
    assert not path.exists()


def test_empty_question_list_fails_without_reporting_completion(tmp_path, monkeypatch, capsys):
    engine, args, path = _cli_fixture(tmp_path, monkeypatch, {})
    (tmp_path / "questions.txt").write_text("")
    engine.evaluate = Mock(side_effect=AssertionError("must not call judge"))
    assert cli.main(args) == 1
    engine.evaluate.assert_not_called()
    assert not path.exists()
    assert "[done] Completed" not in capsys.readouterr().out


def test_stop_on_error_still_raises():
    engine = _engine()
    engine.evaluate = Mock(side_effect=RuntimeError("judge unavailable"))
    with pytest.raises(RuntimeError, match="judge unavailable"):
        engine.batch_evaluate([("q1", "a1")], progress=False, stop_on_error=True)
