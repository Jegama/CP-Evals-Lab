"""Run a complete, resumable API benchmark round from one JSON manifest."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from tqdm import tqdm

from cp_build_eval_master import DEFAULT_CONFIG, build_from_config
from parrot_ai.llm_evals.benchmark_config import (
    BenchmarkConfig,
    generated_dataset_path,
    load_benchmark_config,
    results_path,
    sanitize_filename,
)
from parrot_ai.llm_evals.master_csv import ResultSource
from parrot_ai.llm_evals.progress_reporting import PROGRESS_FILE_ENV
from parrot_ai.llm_evals.result_registry import append_result_sources, validate_result_source


RunEvaluation = Callable[[list[str]], int]


class BenchmarkRunError(RuntimeError):
    """A failed round, retaining successful sources for registry publication."""

    def __init__(self, message: str, completed_sources: list[ResultSource]):
        super().__init__(message)
        self.completed_sources = completed_sources


@dataclass(frozen=True)
class BenchmarkOperation:
    operation_id: str
    phase: str
    target_kind: str
    priority: int
    description: str
    progress_label: str
    command: tuple[str, ...]
    result_path: Path
    dependencies: tuple[str, ...] = ()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate answer sets, score them with the default judge, run any "
            "contender judging through a dependency-aware worker queue, and "
            "rebuild the master CSV."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase",
        choices=["all", "generate", "cross-judge", "master"],
        default="all",
        help="Run the full workflow or one resumable phase (default: all)",
    )
    parser.add_argument(
        "--jobs",
        type=_positive_int,
        help=(
            "Maximum ready operations to run concurrently; defaults to the "
            "manifest's max_parallel_operations"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a human-readable scheduling plan without calling providers",
    )
    parser.add_argument(
        "--show-commands",
        action="store_true",
        help="Also show the full cp_eval_llms.py command for each operation",
    )
    parser.add_argument(
        "--no-live-progress",
        action="store_true",
        help="Suppress the parent runner's live multi-operation progress bars",
    )
    return parser.parse_args(argv)


def _common_args(config: BenchmarkConfig) -> list[str]:
    return [
        "--language",
        config.language,
        "--system-prompt-label",
        config.system_prompt_label,
        "--limit",
        str(config.limit),
        "--question-tags",
        str(config.question_tags.resolve()),
        "--skip-comparison-csv",
        "--require-complete-answers",
    ]


def _generation_operation_id(answers_label: str) -> str:
    return f"generate:{answers_label}"


def generation_operations(config: BenchmarkConfig) -> list[BenchmarkOperation]:
    operations: list[BenchmarkOperation] = []
    for target in config.generation_targets:
        result_path = results_path(
            config,
            target.answers_label,
            config.default_judge_model,
        ).resolve()
        command = [
            "--mode",
            "generate-api_evals",
            "--provider",
            target.provider,
            "--gen-model",
            target.gen_model,
            "--answers-label",
            target.answers_label,
            "--judge-model",
            config.default_judge_model,
            "--output-dataset",
            str(generated_dataset_path(config, target).resolve()),
            "--results-jsonl",
            str(result_path),
            *_common_args(config),
        ]
        if config.use_system_prompt:
            command.append("--use-system-prompt")
        marker = " [judge contender]" if target.judge_contender else ""
        operations.append(
            BenchmarkOperation(
                operation_id=_generation_operation_id(target.answers_label),
                phase="generate",
                target_kind="new model",
                priority=0,
                description=(
                    f"{target.provider}:{target.gen_model}{marker} -> "
                    f"generate answers and judge with {config.default_judge_model}"
                ),
                progress_label=f"{target.provider}:{target.gen_model}",
                command=tuple(command),
                result_path=result_path,
            )
        )
    return operations


def generation_commands(config: BenchmarkConfig) -> list[list[str]]:
    """Backward-compatible command-only view used by tests and integrations."""

    return [list(operation.command) for operation in generation_operations(config)]


def cross_judge_operations(config: BenchmarkConfig) -> list[BenchmarkOperation]:
    answer_sets = [
        (
            target.answers_label,
            generated_dataset_path(config, target),
            "new model",
        )
        for target in config.generation_targets
    ]
    seen_labels = {answers_label for answers_label, _, _ in answer_sets}
    answer_sets.extend(
        (target.answers_label, target.dataset, "provider leader")
        for target in config.control_targets
        if target.answers_label not in seen_labels
    )

    completed_pairs = {
        (target.answers_label, config.default_judge_model)
        for target in config.generation_targets
    }
    operations: list[BenchmarkOperation] = []
    for contender in config.judge_contenders:
        judge_model = contender.contender_judge_model
        for answers_label, dataset, target_kind in answer_sets:
            pair = (answers_label, judge_model)
            if pair in completed_pairs:
                continue
            completed_pairs.add(pair)
            dependencies = (
                (_generation_operation_id(answers_label),)
                if target_kind == "new model"
                else ()
            )
            result_path = results_path(
                config,
                answers_label,
                judge_model,
            ).resolve()
            command = [
                "--mode",
                "dataset",
                "--dataset",
                str(dataset.resolve()),
                "--answers-label",
                answers_label,
                "--judge-model",
                judge_model,
                "--results-jsonl",
                str(result_path),
                *_common_args(config),
            ]
            operations.append(
                BenchmarkOperation(
                    operation_id=f"judge:{judge_model}:{answers_label}",
                    phase="contender",
                    target_kind=target_kind,
                    priority=1 if target_kind == "provider leader" else 2,
                    description=(
                        f"{judge_model} judges {answers_label} ({target_kind})"
                    ),
                    progress_label=f"{judge_model} -> {answers_label}",
                    command=tuple(command),
                    result_path=result_path,
                    dependencies=dependencies,
                )
            )
    return operations


def cross_judge_commands(config: BenchmarkConfig) -> list[list[str]]:
    """Backward-compatible command-only view used by tests and integrations."""

    return [list(operation.command) for operation in cross_judge_operations(config)]


def selected_operations(
    config: BenchmarkConfig,
    phase: str,
) -> list[BenchmarkOperation]:
    if phase == "master":
        return []
    operations: list[BenchmarkOperation] = []
    if phase in ("all", "generate"):
        operations.extend(generation_operations(config))
    if phase in ("all", "cross-judge"):
        operations.extend(cross_judge_operations(config))

    selected_ids = {operation.operation_id for operation in operations}
    return [
        replace(
            operation,
            dependencies=tuple(
                dependency
                for dependency in operation.dependencies
                if dependency in selected_ids
            ),
        )
        for operation in operations
    ]


def _sorted_operations(
    operations: Sequence[BenchmarkOperation],
) -> list[BenchmarkOperation]:
    positions = {
        operation.operation_id: index for index, operation in enumerate(operations)
    }
    return sorted(
        operations,
        key=lambda operation: (
            operation.priority,
            positions[operation.operation_id],
        ),
    )


def _render_command(operation: BenchmarkOperation) -> str:
    return shlex.join(["python", "cp_eval_llms.py", *operation.command])


def _print_operation_queue(
    operations: Sequence[BenchmarkOperation],
    *,
    jobs: int,
    show_commands: bool,
) -> None:
    ready = _sorted_operations(
        [operation for operation in operations if not operation.dependencies]
    )
    waiting = _sorted_operations(
        [operation for operation in operations if operation.dependencies]
    )
    initial = ready[:jobs]
    queued = ready[jobs:]

    print(f"  Initial worker slots ({len(initial)}/{jobs} used):")
    for index, operation in enumerate(initial, start=1):
        print(f"    {index}. {operation.description}")
        if show_commands:
            print(f"       $ {_render_command(operation)}")

    if queued:
        print(f"  Ready queue after initial slots ({len(queued)}):")
        for operation in queued:
            print(f"    - {operation.description}")
            if show_commands:
                print(f"      $ {_render_command(operation)}")

    if waiting:
        print(f"  Dependency-gated operations ({len(waiting)}):")
        for operation in waiting:
            dependencies = ", ".join(operation.dependencies)
            print(f"    - {operation.description}")
            print(f"      waits for: {dependencies}")
            if show_commands:
                print(f"      $ {_render_command(operation)}")


def print_benchmark_plan(
    config: BenchmarkConfig,
    phase: str,
    *,
    jobs: int,
    show_commands: bool,
) -> None:
    generation = generation_operations(config)
    contender = cross_judge_operations(config)
    operations = selected_operations(config, phase)

    print("Benchmark plan")
    print(f"  Maximum concurrent operations: {jobs}")
    print("  Scheduling: dependency-aware; jobs is a ceiling, not a batch size")
    print(f"  Default judge: {config.default_judge_model}")
    print(f"  System prompt: {config.system_prompt_label}")
    print(f"  New models ({len(config.generation_targets)}):")
    for index, target in enumerate(config.generation_targets, start=1):
        marker = " [judge contender]" if target.judge_contender else ""
        print(f"    {index}. {target.provider}:{target.gen_model}{marker}")

    print(f"  Provider leaders ({len(config.control_targets)}):")
    for index, target in enumerate(config.control_targets, start=1):
        identity = (
            f"{target.provider}:{target.gen_model}"
            if target.provider and target.gen_model
            else target.answers_label
        )
        print(f"    {index}. {identity}")

    if config.judge_contenders:
        print(f"  Judge contenders ({len(config.judge_contenders)}):")
        for target in config.judge_contenders:
            operation_count = sum(
                target.contender_judge_model in operation.command
                for operation in contender
            )
            print(
                f"    - {target.contender_judge_model}: "
                f"{operation_count} additional evaluations"
            )
    else:
        print("  Judge contenders: none")

    selected_generation = [
        operation for operation in operations if operation.phase == "generate"
    ]
    selected_contender = [
        operation for operation in operations if operation.phase == "contender"
    ]
    print("  Selected operations:")
    print(f"    - generation/default judging: {len(selected_generation)}")
    print(f"    - contender judging: {len(selected_contender)}")
    print(f"    - total provider evaluation operations: {len(operations)}")
    print(f"    - result registry: {config.result_registry}")
    print(f"    - master CSV: {config.master_csv}")

    if operations:
        print("\nScheduler preview")
        _print_operation_queue(
            operations,
            jobs=jobs,
            show_commands=show_commands,
        )
    elif phase == "master":
        print("\n  No provider operations; only the master CSV will be rebuilt.")


def _operation_log_path(
    config: BenchmarkConfig,
    operation: BenchmarkOperation,
) -> Path:
    filename = f"{sanitize_filename(operation.operation_id)}.log"
    return config.api_evals_dir / "logs" / filename


def _operation_progress_path(
    config: BenchmarkConfig,
    operation: BenchmarkOperation,
) -> Path:
    filename = f"{sanitize_filename(operation.operation_id)}.progress.log"
    return config.api_evals_dir / "logs" / filename


@dataclass
class _ProgressTail:
    path: Path
    offset: int = 0
    remainder: str = ""

    def read_events(self) -> list[dict[str, Any]]:
        try:
            if self.path.stat().st_size < self.offset:
                self.offset = 0
                self.remainder = ""
            with self.path.open("r", encoding="utf-8") as stream:
                stream.seek(self.offset)
                chunk = stream.read()
                self.offset = stream.tell()
        except OSError:
            return []

        text = self.remainder + chunk
        if not text:
            return []
        lines = text.splitlines(keepends=True)
        self.remainder = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self.remainder = lines.pop()

        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events


@dataclass
class _ProgressBarState:
    operation: BenchmarkOperation
    tail: _ProgressTail
    bar: Any
    stage: str | None = None


class _LiveProgressDisplay:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._states: dict[str, _ProgressBarState] = {}

    def write(self, message: str) -> None:
        if self.enabled:
            tqdm.write(message, file=sys.stdout)
        else:
            print(message)

    def start(
        self,
        operation: BenchmarkOperation,
        progress_path: Path,
        position: int,
    ) -> None:
        if not self.enabled:
            return
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("", encoding="utf-8")
        bar = tqdm(
            total=None,
            desc=f"{operation.progress_label} | starting",
            unit="q",
            position=position,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        self._states[operation.operation_id] = _ProgressBarState(
            operation=operation,
            tail=_ProgressTail(progress_path),
            bar=bar,
        )

    def refresh(self) -> None:
        if not self.enabled:
            return
        for state in self._states.values():
            events = state.tail.read_events()
            if not events:
                continue
            changed = False
            for event in events:
                stage = event.get("stage")
                current = event.get("current")
                total = event.get("total")
                if not isinstance(stage, str) or not isinstance(current, int):
                    continue
                if total is not None and not isinstance(total, int):
                    continue
                if stage != state.stage:
                    state.stage = stage
                    state.bar.reset(total=total)
                    state.bar.set_description_str(
                        f"{state.operation.progress_label} | {stage}"
                    )
                elif state.bar.total != total:
                    state.bar.total = total
                state.bar.n = max(0, current)
                changed = True
            if changed:
                state.bar.refresh()

    def finish(self, operation: BenchmarkOperation) -> None:
        if not self.enabled:
            return
        self.refresh()
        state = self._states.pop(operation.operation_id, None)
        if state is not None:
            state.bar.close()

    def close(self) -> None:
        for state in self._states.values():
            state.bar.close()
        self._states.clear()


def _run_subprocess(
    operation: BenchmarkOperation,
    *,
    repo_root: Path,
    log_path: Path,
    progress_path: Path | None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    operation_args = list(operation.command)
    environment = None
    if progress_path is not None:
        if "--no-progress" not in operation_args:
            operation_args.append("--no-progress")
        environment = os.environ.copy()
        environment[PROGRESS_FILE_ENV] = str(progress_path.resolve())
    command = [
        sys.executable,
        "-u",
        str(repo_root / "cp_eval_llms.py"),
        *operation_args,
    ]
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(command)}\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=environment,
        )
    return completed.returncode


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _execute_operations(
    config: BenchmarkConfig,
    operations: Sequence[BenchmarkOperation],
    *,
    jobs: int,
    show_commands: bool,
    repo_root: Path,
    run_evaluation: RunEvaluation | None,
    live_progress: bool,
) -> None:
    pending = {operation.operation_id: operation for operation in operations}
    completed_ids: set[str] = set()
    failed_ids: set[str] = set()
    skipped_ids: set[str] = set()
    running: dict[Future[int], tuple[BenchmarkOperation, int]] = {}
    positions = {
        operation.operation_id: index for index, operation in enumerate(operations)
    }
    started = 0
    finished = 0
    available_positions = list(range(jobs))
    display = _LiveProgressDisplay(
        live_progress
        and run_evaluation is None
        and sys.stdout.isatty()
    )

    def execute(operation: BenchmarkOperation) -> int:
        if run_evaluation is not None:
            return run_evaluation(list(operation.command))
        return _run_subprocess(
            operation,
            repo_root=repo_root,
            log_path=_operation_log_path(config, operation),
            progress_path=(
                _operation_progress_path(config, operation)
                if display.enabled
                else None
            ),
        )

    try:
        with ThreadPoolExecutor(
            max_workers=jobs,
            thread_name_prefix="cp-eval",
        ) as executor:
            while pending or running:
                blocked_ids = failed_ids | skipped_ids
                newly_skipped = [
                    operation
                    for operation in pending.values()
                    if any(
                        dependency in blocked_ids
                        for dependency in operation.dependencies
                    )
                ]
                for operation in newly_skipped:
                    pending.pop(operation.operation_id)
                    skipped_ids.add(operation.operation_id)
                    dependency = next(
                        item
                        for item in operation.dependencies
                        if item in blocked_ids
                    )
                    display.write(
                        f"[skipped] {operation.description} "
                        f"(dependency failed: {dependency})"
                    )

                available_slots = len(available_positions)
                ready = [
                    operation
                    for operation in pending.values()
                    if all(
                        dependency in completed_ids
                        for dependency in operation.dependencies
                    )
                ]
                ready.sort(
                    key=lambda operation: (
                        operation.priority,
                        positions[operation.operation_id],
                    )
                )
                for operation in ready[:available_slots]:
                    pending.pop(operation.operation_id)
                    started += 1
                    position = available_positions.pop(0)
                    display.write(
                        f"[start {started}/{len(operations)}] "
                        f"{operation.description}"
                    )
                    if show_commands:
                        display.write(f"  $ {_render_command(operation)}")
                    progress_path = _operation_progress_path(config, operation)
                    if run_evaluation is None:
                        log_path = _operation_log_path(config, operation)
                        display.write(
                            f"  log: {_display_path(log_path, repo_root)}"
                        )
                    display.start(operation, progress_path, position)
                    future = executor.submit(execute, operation)
                    running[future] = (operation, position)

                if not running:
                    if pending:
                        unresolved = ", ".join(sorted(pending))
                        raise RuntimeError(
                            "Benchmark dependency graph cannot make progress: "
                            f"{unresolved}"
                        )
                    break

                done, _ = wait(
                    running,
                    timeout=0.25 if display.enabled else None,
                    return_when=FIRST_COMPLETED,
                )
                display.refresh()
                for future in done:
                    operation, position = running.pop(future)
                    available_positions.append(position)
                    available_positions.sort()
                    display.finish(operation)
                    try:
                        exit_code = future.result()
                        if exit_code == 0:
                            validate_result_source(ResultSource(operation.result_path))
                    except Exception as exc:
                        exit_code = 1
                        display.write(
                            f"[failed] {operation.description}: {exc}"
                        )
                    if exit_code == 0:
                        completed_ids.add(operation.operation_id)
                        finished += 1
                        display.write(
                            f"[complete {finished}/{len(operations)}] "
                            f"{operation.description}"
                        )
                    else:
                        failed_ids.add(operation.operation_id)
                        display.write(
                            f"[failed] {operation.description} "
                            f"(exit code {exit_code})"
                        )
    finally:
        display.close()

    if failed_ids or skipped_ids:
        raise BenchmarkRunError(
            "Benchmark finished with "
            f"{len(failed_ids)} failed and "
            f"{len(skipped_ids)} dependency-skipped operations",
            [
                ResultSource(operation.result_path, config.eval_version)
                for operation in operations
                if operation.operation_id in completed_ids
            ],
        )


def run_benchmark(
    config: BenchmarkConfig,
    *,
    phase: str,
    dry_run: bool,
    jobs: int | None = None,
    show_commands: bool = False,
    live_progress: bool = True,
    repo_root: Path | None = None,
    run_evaluation: RunEvaluation | None = None,
) -> list[ResultSource]:
    effective_jobs = jobs or config.max_parallel_operations
    operations = selected_operations(config, phase)
    effective_repo_root = repo_root or Path.cwd()

    if dry_run:
        print_benchmark_plan(
            config,
            phase,
            jobs=effective_jobs,
            show_commands=show_commands,
        )
        return []

    if operations:
        _execute_operations(
            config,
            operations,
            jobs=effective_jobs,
            show_commands=show_commands,
            repo_root=effective_repo_root,
            run_evaluation=run_evaluation,
            live_progress=live_progress,
        )
    return [
        ResultSource(
            path=operation.result_path,
            eval_version=config.eval_version,
        )
        for operation in operations
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path.cwd()
    config = load_benchmark_config(args.config, repo_root)
    jobs = args.jobs or config.max_parallel_operations
    exit_code = 0
    try:
        completed_sources = run_benchmark(
            config,
            phase=args.phase,
            dry_run=args.dry_run,
            jobs=jobs,
            show_commands=args.show_commands,
            live_progress=not args.no_live_progress,
            repo_root=repo_root,
        )
    except BenchmarkRunError as exc:
        print(f"[failed] {exc}")
        completed_sources = exc.completed_sources
        exit_code = 1
    except KeyboardInterrupt:
        print(
            "\n[interrupted] Benchmark stopped. Rerun the same command to reuse "
            "previously saved datasets and evaluations."
        )
        return 130
    if not args.dry_run and args.phase in (
        "all",
        "generate",
        "cross-judge",
        "master",
    ):
        if completed_sources:
            added = append_result_sources(
                config.result_registry,
                completed_sources,
                repo_root=repo_root,
            )
            print(
                f"[registry] Added {added} new result files -> "
                f"{config.result_registry}"
            )
        if exit_code == 0 or completed_sources:
            build_from_config(args.config, repo_root)
    elif args.dry_run:
        print("\nDry run complete: no provider calls were made.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
