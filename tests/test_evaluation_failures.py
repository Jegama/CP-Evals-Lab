"""Offline regression coverage for SDK failures, input validation, and resume."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import anthropic
import pytest
from google.genai.errors import ClientError, ServerError

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
    score = {"Adherence": {"Core": 4}}
    engine.evaluate = Mock(side_effect=[score, *[RuntimeError("unavailable")] * 3])
    results = engine.batch_evaluate([(f"q{i}", "answer") for i in range(500)], progress=False)
    assert engine.evaluate.call_count == 4
    assert results[0]["evaluation"] == score
    assert len(results) == 4


def test_isolated_failures_do_not_stop_subsequent_successes():
    engine = _engine()
    outcomes = [RuntimeError("temporary"), {"Adherence": {"Core": 4}}] * 4
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


@pytest.mark.parametrize("entrypoint", ["judge", "generation"])
@pytest.mark.parametrize("code,attempts", [
    (400, 1), (401, 1), (403, 1), (404, 1), (422, 1),
    (429, 3), (500, 3), (503, 3),
])
def test_google_api_errors_respect_batch_cutoffs(monkeypatch, entrypoint, code, attempts):
    # Use the installed SDK's actual exception shape, without issuing requests.
    error_type = ClientError if code < 500 else ServerError
    error = error_type(code, {"error": {"message": "offline regression"}})
    engine = _engine()
    engine.model = "gemini-3.8-flash"
    engine.seed = 7
    request = Mock(side_effect=error)
    if entrypoint == "judge":
        engine.client = SimpleNamespace(models=SimpleNamespace(generate_content=request))
        results = engine.batch_evaluate([("q", "a")] * 10, progress=False)
    else:
        wrapper = Mock(generate=request)
        monkeypatch.setattr(evaluation_module, "ParrotAIGemini", lambda **kwargs: wrapper)
        results = engine.generate_responses(["q"] * 10, provider="google", progress=False)
        assert results.aborted

    assert request.call_count == attempts
    assert len(results) == attempts
    assert all(result["error"] == str(error) for result in results)


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


@pytest.mark.parametrize("error_value", ["generation failed", "", None, False, 0])
@pytest.mark.parametrize("with_system_message", [False, True])
def test_error_marked_dataset_record_is_not_treated_as_an_answer(
    tmp_path, monkeypatch, error_value, with_system_message,
):
    engine, args, path = _cli_fixture(tmp_path, monkeypatch, {"q1": "a1", "q2": "a2", "q3": "a3"})
    dataset = tmp_path / "answers.jsonl"
    records = [json.loads(line) for line in dataset.read_text().splitlines()]
    records[1]["error"] = error_value
    records[1]["messages"][1]["content"] = "Provider error text, not an answer"
    if with_system_message:
        for record in records:
            record["messages"].insert(0, {"role": "system", "content": "System prompt"})
    dataset.write_text("".join(json.dumps(r) + "\n" for r in records))
    before = dataset.read_bytes()
    engine.evaluate = Mock(side_effect=AssertionError("must not call judge"))
    assert cli.main(args) == 1
    engine.evaluate.assert_not_called()
    assert not path.exists()
    assert dataset.read_bytes() == before


def test_generation_retries_error_marked_answer_with_empty_error(tmp_path, monkeypatch):
    engine, args, path = _cli_fixture(tmp_path, monkeypatch, {"q1": "a1", "q2": "poisoned", "q3": "a3"})
    dataset = tmp_path / "answers.jsonl"
    records = [json.loads(line) for line in dataset.read_text().splitlines()]
    records[1]["error"] = ""
    dataset.write_text("".join(json.dumps(record) + "\n" for record in records))
    before = dataset.read_bytes()
    engine.evaluate = Mock(return_value={"Adherence": {"Core": 4}})
    wrapper = Mock()
    wrapper.generate.return_value = "recovered a2"
    monkeypatch.setattr(evaluation_module, "ParrotAIClaude", lambda **kwargs: wrapper)
    args[args.index("--mode") + 1] = "generate-api_evals"
    args[args.index("--dataset")] = "--output-dataset"
    args.extend(["--provider", "anthropic", "--gen-model", "claude-haiku-5-5"])

    assert cli.main(args) == 0
    assert [call.kwargs["prompt"] for call in wrapper.generate.call_args_list] == ["q2"]
    assert [call.args for call in engine.evaluate.call_args_list] == [
        ("q1", "a1"), ("q2", "recovered a2"), ("q3", "a3"),
    ]
    assert dataset.read_bytes().startswith(before)
    assert len(dataset.read_text().splitlines()) == 4
    assert len(path.read_text().splitlines()) == 3


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


@pytest.mark.parametrize("language", ["english", "arabic"])
@pytest.mark.parametrize("parsed,text", [
    (None, ""), (None, "{}"), ({}, None),
    ({"Adherence": {"Core": "4"}}, None),
    ({"Adherence": {"Core": True}}, None),
    ({"Arabic_Accuracy": {"Heuristic_Arabic_Purity_Pct": 100}}, None),
])
def test_empty_gemini_output_is_a_failed_evaluation(language, parsed, text):
    engine = _engine()
    engine.language = language
    engine.model = "gemini-3.8-flash"
    engine.seed = 7
    generate = Mock(return_value=SimpleNamespace(parsed=parsed, text=text))
    engine.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))

    results = engine.batch_evaluate([("Question", "Answer")] * 10, progress=False)

    assert generate.call_count == 3
    assert len(results) == 3
    assert all("evaluation" not in result and "scores" in result["error"] for result in results)


@pytest.mark.parametrize("invalid", [
    {},
    {"Adherence": {}, "Interfaith_Sensitivity": {}, "Arabic_Accuracy": {}},
    {"Kindness_and_Gentleness": {"Pastoral_Acknowledgement": "yes"}},
    {"Adherence": {"Core": "4"}},
    {"Adherence": {"Core": True}},
    {"Adherence": {"unknown": 4}},
])
def test_cli_retries_scoreless_results_and_does_not_publish_them(tmp_path, monkeypatch, invalid):
    engine, args, path = _cli_fixture(tmp_path, monkeypatch, {"q1": "a1", "q2": "a2", "q3": "a3"})
    path.write_text("".join(json.dumps({
        "question": f"q{i}", "answer": f"a{i}", "evaluation": invalid,
    }) + "\n" for i in range(1, 4)))
    before = path.read_bytes()
    engine.evaluate = Mock(return_value=invalid)

    assert cli.main(args) == 1
    assert engine.evaluate.call_count == 3
    assert path.read_bytes().startswith(before)
    assert not (tmp_path / "data/english/training_datasets/evals/dataset_eval_comparison.csv").exists()

    engine.evaluate = Mock(return_value={"Adherence": {"Core": 4}})
    assert cli.main(args) == 0
    assert [call.args[0] for call in engine.evaluate.call_args_list] == ["q1", "q2", "q3"]


@pytest.mark.parametrize("existing_dataset", [False, True])
@pytest.mark.parametrize("require_complete", [False, True])
@pytest.mark.parametrize("permanent", [False, True])
def test_cli_generation_cutoff_preserves_answers_for_later_resume(
    tmp_path, monkeypatch, existing_dataset, require_complete, permanent,
):
    monkeypatch.chdir(tmp_path)
    questions = tmp_path / "questions.txt"
    questions.write_text("".join(f"q{i}\n" for i in range(1, 7)))
    dataset = tmp_path / "generated.jsonl"
    if existing_dataset:
        dataset.touch()
    engine = _engine()
    engine.evaluate = Mock(return_value={"Adherence": {"Core": 4}})
    monkeypatch.setattr(cli, "EvaluationEngine", lambda **kwargs: engine)
    error = TypeError("bad SDK argument") if permanent else RuntimeError("unavailable")
    wrapper = Mock()
    wrapper.generate.side_effect = ["a1", *[error] * 20]
    monkeypatch.setattr(evaluation_module, "ParrotAIClaude", lambda **kwargs: wrapper)
    results = tmp_path / "results.jsonl"
    args = [
        "--mode", "generate-api_evals", "--language", "english", "--limit", "0",
        "--provider", "anthropic", "--gen-model", "claude-haiku-5-5",
        "--questions-file", str(questions), "--output-dataset", str(dataset),
        "--results-jsonl", str(results), "--no-progress", "--skip-comparison-csv",
    ]
    if require_complete:
        args.append("--require-complete-answers")

    assert cli.main(args) == 1
    assert wrapper.generate.call_count == (2 if permanent else 4)
    engine.evaluate.assert_not_called()
    assert not results.exists()
    before = dataset.read_bytes()
    assert len(before.splitlines()) == 1
    assert json.loads(before)["messages"][1]["content"] == "a1"

    wrapper.generate.reset_mock(side_effect=True)
    wrapper.generate.side_effect = lambda prompt, **kwargs: f"answer to {prompt}"
    assert cli.main(args) == 0
    assert [call.kwargs["prompt"] for call in wrapper.generate.call_args_list] == [
        "q2", "q3", "q4", "q5", "q6",
    ]
    assert engine.evaluate.call_count == 6
    assert dataset.read_bytes().startswith(before)
    assert len(dataset.read_text().splitlines()) == 6
