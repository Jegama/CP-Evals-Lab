"""Interrupt real benchmark processes and resume with offline provider stubs."""

import json
import os
from collections import Counter
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_first_judge_failure_does_not_block_others_and_rerun_reuses_saved_work(tmp_path, monkeypatch, capsys):
    import socket
    from types import SimpleNamespace
    import cp_eval_benchmark as benchmark
    import cp_eval_llms as cli
    import parrot_ai.llm_evaluation as evaluation_module
    from parrot_ai.llm_evaluation import EvaluationEngine

    monkeypatch.chdir(tmp_path)
    def deny_network(*args, **kwargs):
        raise AssertionError("This test must not contact a provider")
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    data = tmp_path / "data/english"
    data.mkdir(parents=True)
    (data / "en_eval_questions.txt").write_text("q1\nq2\nq3\nq4\nq5\nq6\n")
    manifest = tmp_path / "config.json"
    manifest.write_text(json.dumps({
        "language": "english", "limit": 0, "system_prompt_label": "offline",
        "default_judge_models": ["judge-a", "judge-b", "judge-c"],
        "models": [{"provider": "openai", "model": "offline-model",
                    "judge_contender": True, "judge_model": "judge-d"}],
    }))
    config = benchmark.load_benchmark_config(manifest, tmp_path)
    # Mimic artifacts from the former combined generation/first-judge worker.
    target = config.generation_targets[0]
    dataset = benchmark.generated_dataset_path(config, target)
    dataset.parent.mkdir(parents=True)
    dataset.write_text(json.dumps({"messages": [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "Answer to q1"},
    ]}) + "\n")
    first_review = benchmark.results_path(config, target.answers_label, "judge-a")
    first_review.write_text(json.dumps({
        "question": "q1", "answer": "Answer to q1",
        "evaluation": {"Adherence": {"Core": 4}}, "mode": "generate-api_evals",
    }) + "\n")
    original_answer = dataset.read_bytes()
    original_review = first_review.read_bytes()
    calls = []
    failing = True
    class OfflineWrapper:
        def __init__(self, **kwargs):
            pass
        def set_model(self, model):
            pass
        def generate(self, prompt, **kwargs):
            calls.append(("generating", prompt))
            return "Answer to " + prompt

    def make_engine(**kwargs):
        engine = object.__new__(EvaluationEngine)
        engine.language = kwargs["language"]
        engine.model = kwargs["model"]
        engine.prompts = SimpleNamespace(MAIN_SYSTEM_PROMPT="Offline system prompt")
        def evaluate(question, answer):
            assert kwargs["initialize_judge"]
            calls.append((engine.model, question))
            if failing and engine.model == "judge-a" and question not in ("q1", "q2"):
                raise ConnectionError("offline simulated network outage")
            return {"Adherence": {"Core": 4}}
        engine.evaluate = evaluate
        return engine

    monkeypatch.setattr(cli, "EvaluationEngine", make_engine)
    monkeypatch.setattr(evaluation_module, "ParrotAIOpenAI", OfflineWrapper)
    # Real scheduler and CLI with sequential workers for deterministic call tracing.
    def run():
        return benchmark.run_benchmark(
            config, phase="all", dry_run=False, jobs=1,
            repo_root=tmp_path, run_evaluation=cli.main,
        )
    with pytest.raises(benchmark.BenchmarkRunError, match="1 failed and 0 dependency-skipped") as exc:
        run()
    assert len(exc.value.completed_sources) == 3
    output = capsys.readouterr().out
    assert "[failed] judging: judge-a" in output
    assert "[failed] generation:" not in output
    assert ("generating", "q1") not in calls
    assert ("judge-a", "q1") not in calls
    assert dataset.read_bytes().startswith(original_answer)
    assert first_review.read_bytes().startswith(original_review)
    for judge in ("judge-b", "judge-c", "judge-d"):
        assert sum(model == judge for model, _ in calls) == 6

    result_dir = data / "api_evals"
    saved = {path: path.read_bytes() for path in result_dir.glob("*.jsonl")}
    assert len(saved) == 5  # One answer dataset and four independent judges.
    calls.clear()
    failing = False
    sources = run()
    assert len(sources) == 4
    assert calls == [("judge-a", q) for q in ("q3", "q4", "q5", "q6")]
    for path, original in saved.items():
        assert path.read_bytes().startswith(original)
        if "__judged_by_judge-a" not in path.name:
            assert path.read_bytes() == original
    output = capsys.readouterr().out
    assert "Answers: 6/6 saved; 0 remaining" in output
    assert "Reviews: 2/6 saved; 4 remaining" in output


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX process-group signals")
@pytest.mark.parametrize("interrupt_signal", ["SIGINT", "SIGKILL"])
@pytest.mark.parametrize("stage", ["generating", "judging"])
def test_real_interruption_and_restart_preserves_checkpoints(tmp_path, stage, interrupt_signal):
    # The runner and CLI are real. Only provider calls are replaced, in the
    # worker subprocess itself; sockets are disabled to prevent live API calls.
    worker = tmp_path / "cp_eval_llms.py"
    worker.write_text(
        f"import sys\nsys.path.insert(0, {str(REPO_ROOT)!r})\n" + '''
import json
import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace

import cp_eval_llms as cli
import parrot_ai.llm_evaluation as evaluation_module
from parrot_ai.llm_evaluation import EvaluationEngine

def deny_network(*args, **kwargs):
    raise AssertionError("This test must not contact a provider")
socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network

def request(stage, question, model):
    with Path("calls.jsonl").open("a") as stream:
        stream.write(json.dumps({"stage": stage, "question": question, "model": model}) + "\\n")
        stream.flush()
    if os.environ.get("PAUSE_STAGE") == stage and question == os.environ.get("PAUSE_QUESTION"):
        Path("ready").write_text("waiting on the next request")
        while True:
            time.sleep(0.05)

class OfflineWrapper:
    def __init__(self, **kwargs):
        pass
    def set_model(self, model):
        pass
    def generate(self, prompt, model, system):
        request("generating", prompt, model)
        return "Answer to " + prompt

def make_engine(**kwargs):
    engine = object.__new__(EvaluationEngine)
    engine.language = kwargs["language"]
    engine.model = kwargs["model"]
    engine.prompts = SimpleNamespace(MAIN_SYSTEM_PROMPT="Offline system prompt")
    def evaluate(question, answer):
        request("judging", question, engine.model)
        return {"Adherence": {"Core": 4}}
    engine.evaluate = evaluate
    return engine

evaluation_module.ParrotAIOpenAI = OfflineWrapper
cli.EvaluationEngine = make_engine
raise SystemExit(cli.main(sys.argv[1:]))
'''
    )
    driver = tmp_path / "run_benchmark.py"
    driver.write_text(
        f"import sys\nsys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "import cp_eval_benchmark\nraise SystemExit(cp_eval_benchmark.main())\n"
    )
    data = tmp_path / "data/english"
    data.mkdir(parents=True)
    question_count = 500
    questions = [f"q{i}" for i in range(1, question_count + 1)]
    (data / "en_eval_questions.txt").write_text("\n".join(questions) + "\n")
    (data / "en_question_tags.json").write_text('{"tags": []}')
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "language": "english",
        "system_prompt_label": "offline",
        "use_system_prompt": True,
        "default_judge_models": ["judge-a", "judge-b", "judge-c"],
        "models": [{"provider": "openai", "model": "offline-model"}],
        "result_registry": str(tmp_path / "registry.json"),
    }))
    command = [sys.executable, str(driver), "--config", str(config), "--jobs", "1", "--no-live-progress"]
    env = {**os.environ, "PAUSE_STAGE": stage, "PAUSE_QUESTION": questions[-1]}
    env.pop("CP_EVAL_PROGRESS_FILE", None)
    process = subprocess.Popen(
        command, cwd=tmp_path, env=env, start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.monotonic() + 45
        while not (tmp_path / "ready").exists():
            if process.poll() is not None:
                pytest.fail(process.communicate()[0])
            if time.monotonic() >= deadline:
                pytest.fail("Offline worker did not reach its interruption point")
            time.sleep(0.02)
        os.killpg(process.pid, getattr(signal, interrupt_signal))
        output, _ = process.communicate(timeout=10)
    finally:
        # Clean up the still-running group if an assertion or wait failed.
        # Do not signal a group again after its parent has been reaped.
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    if interrupt_signal == "SIGINT":
        assert process.returncode == 130, output
        assert "answers and reviews checkpointed to disk" in output
        assert "in-flight request" in output
    else:
        assert process.returncode == -signal.SIGKILL

    dataset = data / "api_evals/generated_api_openai_openai-offline-model-offline.jsonl"
    result_dir = data / "api_evals"
    first_results = result_dir / "eval_results_openai-offline-model-offline__judged_by_judge-a.jsonl"
    answers = [record for record in _jsonl(dataset) if "messages" in record]
    assert len(answers) == (question_count - 1 if stage == "generating" else question_count)
    assert len(_jsonl(first_results)) == (0 if stage == "generating" else question_count - 1)
    saved_answers = dataset.read_bytes()
    saved_reviews = first_results.read_bytes() if first_results.exists() else b""

    # Restart in a fresh process, with the same manifest and saved files.
    env.pop("PAUSE_STAGE")
    restarted = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=45)
    assert restarted.returncode == 0, restarted.stdout + restarted.stderr
    assert dataset.read_bytes().startswith(saved_answers)
    assert first_results.read_bytes().startswith(saved_reviews)
    answers = [record for record in _jsonl(dataset) if "messages" in record]
    assert len(answers) == question_count
    assert {record["messages"][0]["content"] for record in answers} == set(questions)
    for judge in ("judge-a", "judge-b", "judge-c"):
        results = _jsonl(result_dir / f"eval_results_openai-offline-model-offline__judged_by_{judge}.jsonl")
        assert len(results) == question_count
        assert {record["question"] for record in results} == set(questions)
    calls = _jsonl(tmp_path / "calls.jsonl")
    counts = Counter((call["stage"], call["question"], call["model"]) for call in calls)
    # Only the interrupted request is repeated. Saved work is never requested again.
    for model, call_stage in [("offline-model", "generating"), *[(j, "judging") for j in ("judge-a", "judge-b", "judge-c")]]:
        for question in questions:
            count = counts[(call_stage, question, model)]
            repeated = call_stage == stage and question == questions[-1] and model in ("offline-model", "judge-a")
            assert count == (2 if repeated else 1), (model, question, count)
    log_name = (
        "generate_openai-offline-model-offline.log" if stage == "generating"
        else "judge_judge-a_openai-offline-model-offline.log"
    )
    resumed_log = result_dir / "logs" / log_name
    assert "499/500 saved; 1 remaining" in resumed_log.read_text()
    assert len(json.loads((tmp_path / "registry.json").read_text())["results"]) == 3
