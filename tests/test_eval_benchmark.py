"""Tests for manifest-driven benchmark command construction."""

import json
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

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
    monkeypatch.setattr(benchmark, "load_eval_questions", lambda *args, **kwargs: ["Question"])
    return path


def _write_operation_result(command, *, error=False):
    generation = "--generate-only" in command
    path_flag = "--output-dataset" if generation else "--results-jsonl"
    path = Path(command[command.index(path_flag) + 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    record = (
        {"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}
        if generation else {"question": "Question", "answer": "Answer"}
    )
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

    assert len(generation) == len(config.generation_targets) * 2
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
    assert len(expected_dynamic_sources(config)) == len(generation) + len(cross_judge) - len(config.generation_targets)

    luna_command = next(
        command
        for command in generation
        if "openai-gpt-5.6-luna-v1_4" in command
    )
    assert "gpt-5.6-luna" in luna_command
    assert "--generate-only" in luna_command
    assert "--judge-model" not in luna_command


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
    assert "Default judge panel (1):" in output
    assert "Judges wait for every question to have a valid answer" in output
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

    assert call_count == 7
    assert max_active == 4
    assert self_judging_started_after_generation == [True, True]
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
    assert "Dependency-gated operations (2)" in output


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

    with pytest.raises(RuntimeError, match="1 failed and 2 dependency-skipped"):
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

    with pytest.raises(benchmark.BenchmarkRunError, match="1 failed and 2 dependency-skipped") as exc:
        run_benchmark(config, phase="all", dry_run=False, run_evaluation=false_success)

    assert len(calls) == 1
    assert exc.value.completed_sources == []
    output = capsys.readouterr().out
    assert "[failed] generation:" in output
    assert "1/1 answers missing" in output
    assert "answer generation failed:" in output


def test_failed_round_registers_successful_operations_and_returns_failure(monkeypatch, tmp_path):
    config = load_benchmark_config(CONFIG_PATH, REPO_ROOT)
    commands = generation_commands(config)
    good, bad = [command for command in commands if "--generate-only" not in command]

    def run_operation(command):
        _write_operation_result(command, error=command == bad)
        return 1 if command == bad else 0

    def run_round(*args, **kwargs):
        return run_benchmark(
            config, phase="generate", dry_run=False,
            run_evaluation=run_operation,
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


PANEL = ("gpt-6-luna", "gemini-3.8-flash", "claude-haiku-5-5")


def _load_with_judges(**fields):
    payload = json.loads(CONFIG_PATH.read_text())
    payload.update(fields)
    CONFIG_PATH.write_text(json.dumps(payload))
    return load_benchmark_config(CONFIG_PATH, REPO_ROOT)


@pytest.mark.parametrize("field", ["default_judge_models", "default_judge_model"])
def test_load_judge_panel(field):
    config = _load_with_judges(**{field: list(PANEL)})

    assert config.default_judge_models == PANEL
    assert config.default_judge_model == PANEL[0]
    assert config.initial_judge_model == PANEL[0]


@pytest.mark.parametrize("field", ["default_judge_model", "initial_judge_model"])
def test_load_legacy_single_judge(field):
    config = _load_with_judges(**{field: "gemini-3.8-flash"})

    assert config.default_judge_models == ("gemini-3.8-flash",)


@pytest.mark.parametrize("value", [None, [], "gpt-6-luna", {}, 3, [None], [3], [" "]])
def test_reject_invalid_judge_panel(value):
    with pytest.raises(ValueError, match="default_judge_models"):
        _load_with_judges(default_judge_models=value)


def test_reject_duplicate_judges_after_trimming():
    with pytest.raises(ValueError, match="must be unique"):
        _load_with_judges(default_judge_models=["gpt-6-luna", " gpt-6-luna "])


def test_reject_ambiguous_judge_selection_fields():
    with pytest.raises(ValueError, match="only one judge selection field"):
        _load_with_judges(default_judge_models=list(PANEL), default_judge_model="gpt-5-mini")


def test_panel_generates_once_and_judges_identical_dataset_with_each_member():
    config = _load_with_judges(default_judge_models=list(PANEL))
    operations = benchmark.selected_operations(config, "generate")
    generation = [op for op in operations if op.phase == "generate"]
    panel = [op for op in operations if op.phase == "panel"]

    assert len(generation) == len(config.generation_targets)
    assert len(operations) == len(config.generation_targets) * (len(PANEL) + 1)
    for initial in generation:
        command = initial.command
        label = command[command.index("--answers-label") + 1]
        dataset = command[command.index("--output-dataset") + 1]
        following = [op for op in panel if label in op.command]
        assert len(following) == len(PANEL)
        assert all(op.dependencies == (initial.operation_id,) for op in following)
        assert all(op.command[op.command.index("--dataset") + 1] == dataset for op in following)
        assert "--generate-only" in command
        assert "--judge-model" not in command
        judges = [op.command[op.command.index("--judge-model") + 1] for op in following]
        assert judges == list(PANEL)
    assert all("--require-complete-answers" in op.command for op in operations)
    assert all(op.command[op.command.index("--limit") + 1] == "0" for op in operations)
    assert len({op.result_path for op in panel}) == len(panel)
    all_operations = benchmark.selected_operations(config, "all")
    assert [source.path.resolve() for source in expected_dynamic_sources(config)] == [
        op.result_path for op in all_operations if op.result_path is not None
    ]


@pytest.mark.parametrize("judge_model", PANEL)
def test_contender_matching_any_panel_member_only_adds_control_evaluations(judge_model):
    config = _load_with_judges(default_judge_models=list(PANEL))
    targets = tuple(
        replace(target, judge_model=judge_model) if target.judge_contender else target
        for target in config.generation_targets
    )
    config = replace(config, generation_targets=targets)
    _, control_count = _unique_answer_set_counts(config)
    contender_ops = benchmark.selected_operations(config, "cross-judge")
    all_ops = benchmark.selected_operations(config, "all")

    assert len(contender_ops) == control_count
    assert all(op.target_kind == "provider leader" for op in contender_ops)
    judge_ops = [op for op in all_ops if op.result_path is not None]
    assert len({op.result_path for op in judge_ops}) == len(judge_ops)
    assert len(expected_dynamic_sources(config)) == len(judge_ops)


def test_panel_scores_models_without_contender_flag_and_runs_after_generation():
    config = _load_with_judges(default_judge_models=list(PANEL))
    target = replace(config.generation_targets[0], judge_contender=False)
    config = replace(config, generation_targets=(target,), control_targets=())
    generation_done = threading.Event()
    panel_started = threading.Barrier(3)
    calls = []

    def run_operation(command):
        calls.append(command)
        if command[command.index("--mode") + 1] == "generate-api_evals":
            generation_done.set()
        else:
            assert generation_done.is_set()
            panel_started.wait(timeout=2)
        _write_operation_result(command)
        return 0

    sources = run_benchmark(config, phase="all", dry_run=False, jobs=3, run_evaluation=run_operation)

    assert len(calls) == 4
    assert len(sources) == 3
    assert cross_judge_commands(config) == []


@pytest.mark.parametrize("failed_judge", PANEL)
def test_failed_panel_member_preserves_other_judges(failed_judge, capsys):
    config = _load_with_judges(default_judge_models=list(PANEL))
    target = replace(config.generation_targets[0], judge_contender=False)
    config = replace(config, generation_targets=(target,), control_targets=())
    calls = []

    def run_operation(command):
        if "--generate-only" in command:
            _write_operation_result(command)
            return 0
        judge = command[command.index("--judge-model") + 1]
        calls.append(judge)
        _write_operation_result(command, error=judge == failed_judge)
        return 0

    with pytest.raises(benchmark.BenchmarkRunError, match="1 failed and 0 dependency-skipped") as exc:
        run_benchmark(config, phase="generate", dry_run=False, run_evaluation=run_operation)

    assert set(calls) == set(PANEL)
    assert len(exc.value.completed_sources) == 2
    assert all(failed_judge not in source.path.name for source in exc.value.completed_sources)
    assert f"[failed] judging: {failed_judge}" in capsys.readouterr().out


def test_failed_initial_operation_blocks_additional_panel_judging():
    config = _load_with_judges(default_judge_models=list(PANEL))
    config = replace(config, generation_targets=config.generation_targets[:1], control_targets=())
    calls = []

    def fail(command):
        calls.append(command)
        return 1

    with pytest.raises(benchmark.BenchmarkRunError, match="1 failed and 3 dependency-skipped"):
        run_benchmark(config, phase="generate", dry_run=False, run_evaluation=fail)

    assert len(calls) == 1


def test_panel_dry_run_reports_panel_and_contender_work_without_side_effects(capsys):
    config = _load_with_judges(default_judge_models=list(PANEL))

    def fail_if_called(_):
        raise AssertionError("dry run called provider")

    run_benchmark(config, phase="generate", dry_run=True, run_evaluation=fail_if_called)

    output = capsys.readouterr().out
    assert "Default judge panel (3)" in output
    assert "answer generation: 2" in output
    assert "panel judging: 6" in output
    assert "contender judging: 0" in output
    assert "operation counts are not fresh API calls" in output
    assert all(judge in output for judge in PANEL)
    assert not config.api_evals_dir.exists()
    assert not config.result_registry.exists()


def test_completed_panel_registers_every_judge_without_changing_manifest(monkeypatch):
    config = _load_with_judges(default_judge_models=list(PANEL))
    original_manifest = CONFIG_PATH.read_text()

    def fake_subprocess(operation, **kwargs):
        _write_operation_result(operation.command)
        return 0

    monkeypatch.setattr(benchmark, "_run_subprocess", fake_subprocess)
    status = benchmark.main(["--config", str(CONFIG_PATH), "--phase", "generate"])

    assert status == 0
    registry = json.loads(config.result_registry.read_text())
    registered = {Path(source["path"]).name for source in registry["results"]}
    assert registered == {
        op.result_path.name for op in benchmark.generation_operations(config) if op.result_path is not None
    }
    assert len(registered) == len(config.generation_targets) * len(PANEL)
    assert config.master_csv.exists()
    assert CONFIG_PATH.read_text() == original_manifest


def test_live_progress_identifies_judge_and_retains_resumed_counts(tmp_path, monkeypatch):
    config = _load_with_judges(default_judge_models=list(PANEL))
    operation = benchmark.generation_operations(config)[1]
    progress_path = tmp_path / "progress.log"
    bar = Mock()
    bar.total = None
    bar.reset.side_effect = lambda total: setattr(bar, "total", total)
    monkeypatch.setattr(benchmark, "tqdm", lambda **kwargs: bar)
    display = benchmark._LiveProgressDisplay(enabled=True)
    display.start(operation, progress_path, position=0)

    def update(event):
        with progress_path.open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        display.refresh()

    update({"stage": "judging", "current": 246, "total": 500, "reused": 246})
    label = bar.set_description_str.call_args.args[0]
    assert label == f"{operation.progress_label} | judging (246 reused, 254 remaining)"
    assert label.startswith(PANEL[0] + " -> ")
    assert bar.n == 246 and bar.total == 500
    # Subsequent events need not repeat the reuse metadata.
    update({"stage": "judging", "current": 247, "total": 500})
    assert "246 reused, 253 remaining" in bar.set_description_str.call_args.args[0]
    assert bar.n == 247
    display.close()


def test_additional_panel_labels_identify_their_own_judge():
    config = _load_with_judges(default_judge_models=list(PANEL))
    operations = benchmark.generation_operations(config)
    for operation in operations:
        if operation.phase == "generate":
            assert "--judge-model" not in operation.command
            continue
        judge = operation.command[operation.command.index("--judge-model") + 1]
        assert operation.stage_label("judging").startswith(judge + " -> ")
