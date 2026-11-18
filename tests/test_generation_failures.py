"""A model's refusal to answer counts; a judge or transport failure never does."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from google.genai import types

import cp_eval_llms as cli
import cp_eval_benchmark as benchmark
import parrot_ai.llm_evaluation as evaluation
from parrot_ai import gemini_batch as batch
from parrot_ai.generation_failures import TerminalGenerationFailure, load_generation_failures
from parrot_ai.llm_evals.benchmark_config import load_benchmark_config
from parrot_ai.llm_evals.gemini_batch_adapter import _request, judge_gemini_batch
from parrot_ai.llm_evals.aggregation import aggregate_score_statistics
from parrot_ai.llm_evals.master_csv import ResultSource, build_master_row
from parrot_ai.llm_evals.score_validation import is_scored_record, minimum_rubric_scores
from test_gemini_batch import FakeClient, response


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


def test_cached_recitation_becomes_durable_outcome_without_another_provider_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    questions = [f"q{i}" for i in range(1, 501)]
    questions_path = tmp_path / "questions.txt"
    questions_path.write_text("\n".join(questions))
    dataset = tmp_path / "answers.jsonl"
    write_jsonl(dataset, [dataset_record(q) for q in questions if q != "q253"])
    original = dataset.read_bytes()
    client = FakeClient()
    blocked = response("")
    blocked["candidates"][0]["finishReason"] = "RECITATION"
    client.rows = lambda request: {"response": blocked}
    request = _request("q253", None)
    batch.run_batch(model="gemini-test", requests={batch.request_key("gemini-test", request): request},
                    artifact=dataset, consume=lambda *args: False, client=client)
    monkeypatch.setattr(batch, "_new_client", Mock(side_effect=AssertionError("No provider calls allowed")))
    args = ["--mode", "generate-api_evals", "--generate-only", "--provider", "google",
            "--gen-model", "gemini-test", "--output-dataset", str(dataset), "--language", "english",
            "--questions-file", str(questions_path), "--limit", "0", "--require-complete-answers",
            "--gemini-execution", "batch", "--gemini-batch-sync-fallback-max", "9", "--no-progress"]
    assert cli.main(args) == 0
    assert dataset.read_bytes().startswith(original)
    assert len(cli.load_qa_pairs(str(dataset), limit=0)) == 499
    assert load_generation_failures(dataset)["q253"]["generation_failure"] == FAILURE
    completed = dataset.read_bytes()
    assert cli.main(args) == 0
    assert dataset.read_bytes() == completed
    batch._new_client.assert_not_called()


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
    client = FakeClient()
    client.rows = lambda request: {"response": response(json.dumps(rubric))}
    monkeypatch.setattr(batch, "_new_client", lambda: client)
    results = tmp_path / "reviews.jsonl"
    args = ["--mode", "dataset", "--dataset", str(dataset), "--answers-label", "answers",
            "--judge-model", judge, "--results-jsonl", str(results), "--language", "english",
            "--questions-file", str(question_file), "--limit", "0", "--require-complete-answers",
            "--gemini-execution", "batch", "--skip-comparison-csv", "--no-progress"]
    if judge.startswith("gemini"):
        client.batches.get.side_effect = lambda **kwargs: types.BatchJob(name="batches/0", state="JOB_STATE_PENDING")
        assert cli.main([*args, "--gemini-batch-no-wait"]) == 75
        saved = [json.loads(line) for line in results.read_text().splitlines()]
        assert len(saved) == 1 and saved[0]["evaluation_source"] == "generation_failure_penalty"
        client.batches.get.side_effect = client.get
    assert cli.main(args) == 0
    records = [json.loads(line) for line in results.read_text().splitlines()]
    assert len(records) == 500 and all(is_scored_record(row) for row in records)
    penalty = next(row for row in records if row["question"] == "q253")
    assert penalty["evaluation"] == minimum_rubric_scores()
    assert penalty["generation_failure"] == FAILURE
    assert penalty["evaluation_source"] == "generation_failure_penalty"
    if judge.startswith("gemini"):
        assert sum(len(rows) for rows in client.uploads.values()) == 499
        assert all("q253\n" not in row["request"]["contents"][0]["parts"][0]["text"]
                   for rows in client.uploads.values() for row in rows)
        assert client.batches.create.call_count == 1
    else:
        assert engine.evaluate.call_count == 499
        assert all(call.args[0] != "q253" and call.args[1] for call in engine.evaluate.call_args_list)
    tags = {q: {} for q in questions}
    stats = aggregate_score_statistics(records, False, question_tags=tags)
    assert stats[("Adherence", "Core")].count == 500
    # The normal Gemini postprocessing caps this fixture's Core at 4.
    assert stats[("Adherence", "Core")].mean == (3.99 if judge.startswith("gemini") else 4.99)
    master = build_master_row(ResultSource(results), repo_root=tmp_path, question_tags=tags)
    assert master["question_count"] == 500
    assert master["generation_failure_count"] == 1
    assert master["error_count"] == 0
    original = results.read_bytes()
    assert cli.main(args) == 0
    assert results.read_bytes() == original
    assert engine.evaluate.call_count == (0 if judge.startswith("gemini") else 499)
    assert client.batches.create.call_count == int(judge.startswith("gemini"))


def test_recitation_during_judging_is_an_error_not_an_answer_penalty(tmp_path, monkeypatch):
    client = FakeClient()
    blocked = response("")
    blocked["candidates"][0]["finishReason"] = "RECITATION"
    client.rows = lambda request: {"response": blocked}
    monkeypatch.setattr(batch, "_new_client", lambda: client)
    engine = evaluation.EvaluationEngine(model="gemini-test", language="english", initialize_judge=False)
    records = judge_gemini_batch(engine, [("q", "valid answer")], artifact=tmp_path / "reviews.jsonl")
    assert "RECITATION" in records[0]["error"]
    assert "generation_failure" not in records[0]
    assert not is_scored_record(records[0])


@pytest.mark.parametrize("execution", ["sync", "batch"])
@pytest.mark.parametrize("all_missing", [False, True])
@pytest.mark.parametrize("language", ["english", "arabic"])
def test_dataset_missing_answers_receive_local_scores_and_resume(tmp_path, monkeypatch, execution, all_missing, language):
    monkeypatch.chdir(tmp_path)
    questions = tmp_path / "questions.txt"
    questions.write_text("q1\nq2\nq3\n")
    dataset = tmp_path / "answers.jsonl"
    write_jsonl(dataset, [] if all_missing else [dataset_record("q1")])
    engine = evaluation.EvaluationEngine(model="gemini-test", language=language, initialize_judge=False)
    empty_scores = engine.evaluate("missing", "")
    rubric = {section: {field: 5 if type(value) is int else "" for field, value in fields.items()}
              for section, fields in minimum_rubric_scores(language).items()}
    engine.evaluate = Mock(side_effect=lambda q, a: rubric if a else empty_scores)
    monkeypatch.setattr(cli, "EvaluationEngine", lambda **kwargs: engine)
    client = FakeClient()
    client.rows = lambda request: {"response": response(json.dumps(rubric))}
    monkeypatch.setattr(batch, "_new_client", lambda: client)
    results = tmp_path / "reviews.jsonl"
    args = ["--mode", "dataset", "--dataset", str(dataset), "--answers-label", "answers",
            "--judge-model", "gemini-test", "--results-jsonl", str(results), "--language", language,
            "--questions-file", str(questions), "--limit", "0", "--gemini-execution", execution,
            "--skip-comparison-csv", "--no-progress"]

    assert cli.main([*args, "--require-complete-answers"]) == 1
    assert not results.exists()
    engine.evaluate.assert_not_called()
    assert not client.uploads
    assert cli.main(args) == 0
    records = [json.loads(line) for line in results.read_text().splitlines()]
    assert len(records) == 3 and all(is_scored_record(row) for row in records)
    for row in records:
        assert "generation_failure" not in row
        if all_missing or row["question"] != "q1":
            assert row["evaluation"] == empty_scores
    master = build_master_row(ResultSource(results), repo_root=tmp_path, question_tags={q: {} for q in ("q1", "q2", "q3")})
    assert master["question_count"] == 3
    assert master["generation_failure_count"] == master["error_count"] == 0
    assert engine.evaluate.call_count == int(execution == "sync" and not all_missing)
    assert sum(len(rows) for rows in client.uploads.values()) == int(execution == "batch" and not all_missing)
    original = results.read_bytes()
    assert cli.main(args) == 0
    assert results.read_bytes() == original
    assert engine.evaluate.call_count == int(execution == "sync" and not all_missing)
    assert sum(len(rows) for rows in client.uploads.values()) == int(execution == "batch" and not all_missing)


@pytest.mark.parametrize("replacement", ["answer", "terminal_failure"])
def test_missing_answer_penalty_is_replaced_when_source_outcome_changes(tmp_path, monkeypatch, replacement):
    monkeypatch.chdir(tmp_path)
    questions = tmp_path / "questions.txt"
    questions.write_text("q\n")
    dataset = tmp_path / "answers.jsonl"
    dataset.write_text("")
    results = tmp_path / "reviews.jsonl"
    engine = evaluation.EvaluationEngine(model="gpt-test", language="english", initialize_judge=False)
    engine.evaluate = Mock(return_value={"Adherence": {"Core": 4}})
    monkeypatch.setattr(cli, "EvaluationEngine", lambda **kwargs: engine)
    args = ["--mode", "dataset", "--dataset", str(dataset), "--answers-label", "answers",
            "--results-jsonl", str(results), "--language", "english", "--questions-file", str(questions),
            "--limit", "0", "--skip-comparison-csv", "--no-progress"]
    assert cli.main(args) == 0
    original = results.read_bytes()
    engine.evaluate.assert_not_called()

    write_jsonl(dataset, [dataset_record("q", failure=FAILURE if replacement == "terminal_failure" else None)])
    assert cli.main(args) == 0
    assert results.read_bytes().startswith(original)
    records = [json.loads(line) for line in results.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["evaluation_source"] == "missing_answer_penalty"
    if replacement == "answer":
        assert records[1]["answer"] and "evaluation_source" not in records[1]
        assert engine.evaluate.call_count == 1
    else:
        assert records[1]["evaluation_source"] == "generation_failure_penalty"
        assert records[1]["generation_failure"] == FAILURE
        engine.evaluate.assert_not_called()
    master = build_master_row(ResultSource(results), repo_root=tmp_path, question_tags={"q": {}})
    assert master["question_count"] == 1
    assert master["generation_failure_count"] == int(replacement == "terminal_failure")


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


@pytest.mark.parametrize("only_failures", [False, True])
def test_extended_sample_includes_terminal_outcomes(tmp_path, monkeypatch, only_failures):
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "answers.jsonl"
    rows = [dataset_record("blocked", failure=FAILURE)]
    if not only_failures:
        rows.extend([dataset_record("recovered", failure=FAILURE), dataset_record("recovered")])
    write_jsonl(dataset, rows)
    engine = evaluation.EvaluationEngine(language="english", initialize_judge=False)
    engine.evaluate = Mock(return_value={"Adherence": {"Core": 5}})
    monkeypatch.setattr(cli, "EvaluationEngine", lambda **kwargs: engine)
    results = tmp_path / "reviews.jsonl"
    args = ["--mode", "extended", "--language", "english", "--dataset", str(dataset),
            "--results-jsonl", str(results), "--skip-comparison-csv", "--no-progress"]
    assert cli.main(args) == 0
    master = build_master_row(ResultSource(results), repo_root=tmp_path)
    assert master["question_count"] == (1 if only_failures else 2)
    assert master["generation_failure_count"] == 1
    assert engine.evaluate.call_count == int(not only_failures)
    assert cli.main(args) == 0
    assert engine.evaluate.call_count == int(not only_failures)


@pytest.mark.parametrize("judge", ["gpt-test", "claude-test", "gemini-test"])
def test_local_and_reused_reviews_need_no_judge_client(tmp_path, monkeypatch, judge):
    monkeypatch.chdir(tmp_path)
    dataset, reviews, questions = (tmp_path / name for name in ("answers.jsonl", "reviews.jsonl", "questions.txt"))
    write_jsonl(dataset, [dataset_record("blocked", failure=FAILURE), dataset_record("answered")])
    write_jsonl(reviews, [{"question": "answered", "answer": "Answer in Christ. Ephesians 2:8-9.",
                           "evaluation": {"Adherence": {"Core": 5}}}])
    questions.write_text("blocked\nanswered\n")
    forbidden = Mock(side_effect=AssertionError("No client needed for local or reused reviews"))
    monkeypatch.setattr(evaluation, "OpenAI", forbidden)
    monkeypatch.setattr(evaluation.anthropic, "Anthropic", forbidden)
    monkeypatch.setattr(evaluation.genai, "Client", forbidden)
    args = ["--mode", "dataset", "--language", "english", "--dataset", str(dataset), "--judge-model", judge,
            "--results-jsonl", str(reviews), "--questions-file", str(questions), "--limit", "0", "--skip-comparison-csv"]
    assert cli.main(args) == 0
    saved = reviews.read_bytes()
    assert len(saved.splitlines()) == 2
    assert json.loads(saved.splitlines()[-1])["generation_failure"] == FAILURE
    assert cli.main(args) == 0 and reviews.read_bytes() == saved
    forbidden.assert_not_called()


@pytest.mark.parametrize("canonical_first", [False, True])
def test_extended_resume_keeps_legacy_reviews_and_separates_canonical_results(tmp_path, monkeypatch, canonical_first):
    monkeypatch.chdir(tmp_path)
    # The canonical list overlaps the sample at q502 and includes q0 outside it.
    monkeypatch.setattr("parrot_ai.llm_evals.extended_sample.random.sample", lambda population, k: population[-k:])
    directory = tmp_path / "data/english/training_datasets/evals"
    directory.mkdir(parents=True)
    original = directory / "eval_results_answers__judged_by_gpt-5-mini.jsonl"
    dataset = tmp_path / "answers.jsonl"
    questions = [f"q{i}" for i in range(503)]
    write_jsonl(dataset, [dataset_record(q) for q in questions])
    first = {"mode": "extended", "extended_sample_size": 500, "question": questions[-1],
             "answer": "Answer in Christ. Ephesians 2:8-9.", "evaluation": {"Adherence": {"Core": 5}}}
    write_jsonl(original, [first])
    before = original.read_bytes()
    engine = evaluation.EvaluationEngine(language="english", initialize_judge=False)
    engine.evaluate = Mock(return_value=first["evaluation"])
    monkeypatch.setattr(cli, "EvaluationEngine", lambda **kwargs: engine)
    args = ["--dataset", str(dataset), "--language", "english", "--limit", "0", "--answers-label", "answers",
            "--skip-comparison-csv", "--no-progress"]
    canonical = tmp_path / "questions.txt"
    canonical.write_text(questions[0] + "\n" + questions[-1] + "\n")
    canonical_args = ["--mode", "dataset", "--questions-file", str(canonical), *args]
    if canonical_first:
        assert cli.main(canonical_args) == 0
        assert original.read_bytes() == before
        assert engine.evaluate.call_count == 2
        with pytest.raises(ValueError, match="extended"):
            cli.main([*canonical_args, "--results-jsonl", str(original)])
    engine.evaluate.reset_mock()
    assert cli.main(["--mode", "extended", *args]) == 0
    assert original.read_bytes().startswith(before) and len(original.read_bytes().splitlines()) == 500
    assert engine.evaluate.call_count == 499
    saved = original.read_bytes()
    assert cli.main(["--mode", "extended", *args]) == 0 and original.read_bytes() == saved
    assert cli.main(canonical_args) == 0
    assert original.read_bytes() == saved
    separate = original.with_stem(original.stem + "__dataset")
    assert build_master_row(ResultSource(separate), repo_root=tmp_path)["question_count"] == 2
    assert build_master_row(ResultSource(original), repo_root=tmp_path)["question_count"] == 500
    assert engine.evaluate.call_count == (499 if canonical_first else 501)


def test_fresh_extended_run_preserves_other_results_before_sample_publication(tmp_path, monkeypatch):
    from parrot_ai.llm_evals import extended_sample

    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "data/english/training_datasets/evals"
    directory.mkdir(parents=True)
    canonical = directory / "eval_results_answers__judged_by_gpt-5-mini.jsonl"
    prior = b'{"mode":"dataset"}\n{"question":"\xd8'
    canonical.write_bytes(prior)
    dataset = tmp_path / "answers.jsonl"
    write_jsonl(dataset, [dataset_record("q")])
    args = ["--mode", "extended", "--language", "english", "--dataset", str(dataset), "--answers-label", "answers",
            "--limit", "0", "--skip-comparison-csv", "--no-progress"]
    engine = evaluation.EvaluationEngine(language="english", initialize_judge=False)
    engine.evaluate = Mock(return_value={"Adherence": {"Core": 5}})
    monkeypatch.setattr(cli, "EvaluationEngine", lambda **kwargs: engine)
    with monkeypatch.context() as patch:
        patch.setattr(extended_sample.os, "link", Mock(side_effect=OSError("disk full")))
        with pytest.raises(OSError, match="disk full"):
            cli.main(args)
    engine.evaluate.assert_not_called()
    assert list(directory.iterdir()) == [canonical]
    assert cli.main(args) == 0
    assert canonical.read_bytes() == prior
    assert canonical.with_stem(canonical.stem + "__extended").is_file()


def test_master_export_follows_restored_answers_even_when_replacement_judging_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "answers.jsonl"
    results = tmp_path / "reviews.jsonl"
    questions = tmp_path / "questions.txt"
    questions.write_text("restored\nreplaced\n")
    restored, replaced = dataset_record("restored"), dataset_record("replaced")
    write_jsonl(dataset, [restored, replaced])
    answer = restored["messages"][-1]["content"]
    base = {"dataset": str(dataset), "question": "restored", "answer": answer}
    write_jsonl(results, [
        {**base, "evaluation": {"Adherence": {"Core": 5}}},
        {**base, "answer": "superseded", "evaluation": {"Adherence": {"Core": 2}}},
        {**base, "question": "replaced", "answer": "", "evaluation": minimum_rubric_scores(),
         "evaluation_source": "generation_failure_penalty", "generation_failure": FAILURE},
    ])
    engine = evaluation.EvaluationEngine(language="english", initialize_judge=False)
    engine.evaluate = Mock(side_effect=RuntimeError("judge unavailable"))
    monkeypatch.setattr(cli, "EvaluationEngine", lambda **kwargs: engine)
    assert cli.main(["--mode", "dataset", "--language", "english", "--limit", "0",
                     "--dataset", str(dataset), "--questions-file", str(questions),
                     "--results-jsonl", str(results), "--skip-comparison-csv", "--no-progress"]) == 1
    engine.evaluate.assert_called_once_with("replaced", answer)
    master = build_master_row(ResultSource(results), repo_root=tmp_path)
    assert master["question_count"] == 1
    assert master["adherence_core_mean"] == 5
    assert master["generation_failure_count"] == 0
    assert master["error_count"] == 1


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
