"""Tests for manifest-driven benchmark command construction."""

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

import cp_eval_benchmark as benchmark
from cp_eval_benchmark import cross_judge_commands, generation_commands, run_benchmark
from parrot_ai.llm_evals.benchmark_config import (
    expected_dynamic_sources,
    load_benchmark_config,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "benchmark_configs/english_api_v1_4.json"


@pytest.fixture(autouse=True)
def isolated_manifest(tmp_path, monkeypatch):
    """Exercise a fixed plan without depending on the user's active benchmark."""
    path = tmp_path / "config.json"
    payload = {
        "system_prompt_label": "v1_4",
        "api_evals_dir": str(tmp_path / "results"),
        "result_registry": str(tmp_path / "english_api_registry.json"),
        "master_csv": str(tmp_path / "master.csv"),
        "question_tags": str(tmp_path / "tags.json"),
        "models": [
            {"provider": "xai", "model": "grok-4.5"},
            {"provider": "openai", "model": "gpt-5.6-luna", "judge_contender": True},
        ],
        "provider_leaders": [
            {
                "provider": provider,
                "model": model,
                "answers_label": f"{provider}-{model}-v1_4",
                "dataset": str(tmp_path / f"{model}.jsonl"),
            }
            for provider, model in [
                ("openai", "gpt-5-mini"),
                ("google", "gemini-3-flash"),
                ("anthropic", "claude-haiku-4-5"),
                ("xai", "grok-4.5"),
            ]
        ],
    }
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(__name__ + ".CONFIG_PATH", path)
    return path


def _write_operation_result(command, *, error=False):
    path = Path(command[command.index("--results-jsonl") + 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"question": "Question", "answer": "Answer"}
    if error:
        record["error"] = "SDK failure"
    else:
        record["evaluation"] = {"Adherence": {"Core": 4}}
    path.write_text(json.dumps(record) + "\n")


def _unique_answer_set_counts(config) -> tuple[int, int]:
    generation_labels = {
        target.answers_label for target in config.generation_targets
    }
    all_labels = generation_labels | {
        target.answers_label for target in config.control_targets
    }
    return len(all_labels), len(all_labels - generation_labels)


def test_manifest_builds_generation_and_deduplicated_cross_judge_commands():
    config = load_benchmark_config(CONFIG_PATH, REPO_ROOT)

    generation = generation_commands(config)
    cross_judge = cross_judge_commands(config)
    answer_set_count, _ = _unique_answer_set_counts(config)

    assert len(generation) == len(config.generation_targets)
    assert len(cross_judge) == answer_set_count
    assert config.default_judge_model == "gpt-5-mini"
    assert config.max_parallel_operations == 4
    assert [target.gen_model for target in config.judge_contenders] == [
        "gpt-5.6-luna"
    ]
    assert all("--skip-comparison-csv" in command for command in generation)
    assert all("--skip-comparison-csv" in command for command in cross_judge)
    assert all("--require-complete-answers" in command for command in generation + cross_judge)
    assert all("gpt-5.6-luna" in command for command in cross_judge)
    assert len(expected_dynamic_sources(config)) == len(generation) + len(
        cross_judge
    )

    luna_command = next(
        command
        for command in generation
        if "openai-gpt-5.6-luna-v1_4" in command
    )
    assert "gpt-5.6-luna" in luna_command
    assert "gpt-5-mini" in luna_command


def test_dry_run_is_human_readable_and_never_calls_provider_evaluation(capsys):
    config = load_benchmark_config(CONFIG_PATH, REPO_ROOT)
    expected_operations = len(generation_commands(config)) + len(
        cross_judge_commands(config)
    )

    def fail_if_called(_: list[str]) -> int:
        raise AssertionError("dry run called cp_eval_llms")

    run_benchmark(
        config,
        phase="all",
        dry_run=True,
        run_evaluation=fail_if_called,
    )

    output = capsys.readouterr().out
    assert "Benchmark plan" in output
    assert "Default judge: gpt-5-mini" in output
    assert "openai:gpt-5.6-luna [judge contender]" in output
    assert f"total provider evaluation operations: {expected_operations}" in output
    assert "english_api_registry.json" in output
    assert "python cp_eval_llms.py" not in output


def test_show_commands_adds_exact_shell_commands(capsys):
    config = load_benchmark_config(CONFIG_PATH, REPO_ROOT)

    run_benchmark(
        config,
        phase="generate",
        dry_run=True,
        show_commands=True,
        run_evaluation=lambda _: 0,
    )

    assert "$ python cp_eval_llms.py" in capsys.readouterr().out


def test_each_contender_judges_new_models_and_provider_leaders():
    config = load_benchmark_config(CONFIG_PATH, REPO_ROOT)
    targets = list(config.generation_targets)
    targets[0] = replace(targets[0], judge_contender=True)
    two_contenders = replace(config, generation_targets=tuple(targets))

    commands = cross_judge_commands(two_contenders)
    answer_set_count, _ = _unique_answer_set_counts(two_contenders)

    assert len(commands) == answer_set_count * 2
    judges = [command[command.index("--judge-model") + 1] for command in commands]
    assert judges.count("grok-4.5") == answer_set_count
    assert judges.count("gpt-5.6-luna") == answer_set_count


def test_contender_matching_default_judge_does_not_repeat_primary_scoring():
    config = load_benchmark_config(CONFIG_PATH, REPO_ROOT)
    targets = list(config.generation_targets)
    targets[0] = replace(
        targets[0],
        judge_contender=True,
        judge_model=config.default_judge_model,
    )
    default_contender = replace(config, generation_targets=tuple(targets))

    commands = cross_judge_commands(default_contender)
    answer_set_count, control_only_count = _unique_answer_set_counts(
        default_contender
    )
    judges = [command[command.index("--judge-model") + 1] for command in commands]

    assert len(commands) == control_only_count + answer_set_count
    assert judges.count("gpt-5-mini") == control_only_count
    assert judges.count("gpt-5.6-luna") == answer_set_count


def test_no_contenders_means_no_additional_judging():
    config = load_benchmark_config(CONFIG_PATH, REPO_ROOT)
    no_contenders = replace(
        config,
        generation_targets=tuple(
            replace(target, judge_contender=False)
            for target in config.generation_targets
        ),
    )

    assert cross_judge_commands(no_contenders) == []


def test_single_contender_uses_available_worker_slots_and_waits_for_generation():
    config = load_benchmark_config(CONFIG_PATH, REPO_ROOT)
    contender = next(
        target for target in config.generation_targets if target.judge_contender
    )
    single_contender = replace(config, generation_targets=(contender,))

    lock = threading.Lock()
    first_wave = threading.Barrier(4)
    generation_finished = threading.Event()
    active = 0
    max_active = 0
    call_count = 0
    self_judging_started_after_generation: list[bool] = []

    def run_operation(command: list[str]) -> int:
        nonlocal active, max_active, call_count
        mode = command[command.index("--mode") + 1]
        answers_label = command[command.index("--answers-label") + 1]
        with lock:
            active += 1
            max_active = max(max_active, active)
            call_count += 1
            ordinal = call_count
        try:
            if ordinal <= 4:
                first_wave.wait(timeout=2)
            if mode == "generate-api_evals":
                generation_finished.set()
            elif answers_label == contender.answers_label:
                self_judging_started_after_generation.append(
                    generation_finished.is_set()
                )
            _write_operation_result(command)
            return 0
        finally:
            with lock:
                active -= 1

    completed_sources = run_benchmark(
        single_contender,
        phase="all",
        dry_run=False,
        jobs=4,
        repo_root=REPO_ROOT,
        run_evaluation=run_operation,
    )

    assert call_count == 6
    assert max_active == 4
    assert self_judging_started_after_generation == [True]
    assert len(completed_sources) == 6
    assert completed_sources[0].path.name.startswith(
        "eval_results_openai-gpt-5.6-luna-v1_4"
    )


def test_single_contender_dry_run_fills_slots_without_inventing_a_fixed_batch(capsys):
    config = load_benchmark_config(CONFIG_PATH, REPO_ROOT)
    contender = next(
        target for target in config.generation_targets if target.judge_contender
    )
    single_contender = replace(config, generation_targets=(contender,))

    run_benchmark(
        single_contender,
        phase="all",
        dry_run=True,
        jobs=4,
        repo_root=REPO_ROOT,
    )

    output = capsys.readouterr().out
    assert "Initial worker slots (4/4 used)" in output
    assert "Ready queue after initial slots (1)" in output
    assert "Dependency-gated operations (1)" in output


def test_failed_generation_skips_only_its_dependent_contender_evaluation():
    config = load_benchmark_config(CONFIG_PATH, REPO_ROOT)
    contender = next(
        target for target in config.generation_targets if target.judge_contender
    )
    single_contender = replace(config, generation_targets=(contender,))
    called_labels: list[str] = []

    def fail_generation(command: list[str]) -> int:
        mode = command[command.index("--mode") + 1]
        answers_label = command[command.index("--answers-label") + 1]
        called_labels.append(answers_label)
        if mode != "generate-api_evals":
            _write_operation_result(command)
        return 1 if mode == "generate-api_evals" else 0

    with pytest.raises(RuntimeError, match="1 failed and 1 dependency-skipped"):
        run_benchmark(
            single_contender,
            phase="all",
            dry_run=False,
            jobs=4,
            repo_root=REPO_ROOT,
            run_evaluation=fail_generation,
        )

    assert called_labels.count(contender.answers_label) == 1
    assert len(called_labels) == 5


def test_zero_exit_with_error_only_file_is_failed_and_blocks_dependents(capsys):
    config = load_benchmark_config(CONFIG_PATH, REPO_ROOT)
    contender = config.judge_contenders[0]
    config = replace(config, generation_targets=(contender,), control_targets=())
    calls = []

    def false_success(command):
        calls.append(command)
        _write_operation_result(command, error=True)
        return 0

    with pytest.raises(benchmark.BenchmarkRunError, match="1 failed and 1 dependency-skipped") as exc:
        run_benchmark(config, phase="all", dry_run=False, run_evaluation=false_success)

    assert len(calls) == 1
    assert exc.value.completed_sources == []
    assert "no valid score records" in capsys.readouterr().out


def test_failed_round_registers_successful_operations_and_returns_failure(monkeypatch, tmp_path):
    config = load_benchmark_config(CONFIG_PATH, REPO_ROOT)
    good, bad = generation_commands(config)
    _write_operation_result(good)
    _write_operation_result(bad, error=True)

    def run_round(*args, **kwargs):
        return run_benchmark(
            config, phase="generate", dry_run=False,
            run_evaluation=lambda command: 1 if command == bad else 0,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark, "run_benchmark", run_round)
    # Keep the actual runner available inside the replacement above.
    status = benchmark.main(["--config", str(CONFIG_PATH), "--phase", "generate"])

    assert status == 1
    registry = json.loads(config.result_registry.read_text())
    assert len(registry["results"]) == 1
    assert registry["results"][0]["path"].endswith(Path(good[good.index("--results-jsonl") + 1]).name)
    assert config.master_csv.exists()
