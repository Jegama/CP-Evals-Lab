"""Configuration helpers for repeatable API benchmark rounds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .master_csv import ResultSource


@dataclass(frozen=True)
class GenerationTarget:
    provider: str
    gen_model: str
    answers_label: str
    judge_contender: bool = False
    judge_model: str = ""

    @property
    def contender_judge_model(self) -> str:
        return self.judge_model or self.gen_model


@dataclass(frozen=True)
class ControlTarget:
    provider: str
    gen_model: str
    answers_label: str
    dataset: Path


@dataclass(frozen=True)
class BenchmarkConfig:
    path: Path
    language: str
    eval_version: str
    system_prompt_label: str
    use_system_prompt: bool
    default_judge_models: tuple[str, ...]
    max_parallel_operations: int
    limit: int
    timezone: str
    question_tags: Path
    master_csv: Path
    api_evals_dir: Path
    result_registry: Path
    generation_targets: tuple[GenerationTarget, ...]
    control_targets: tuple[ControlTarget, ...]
    gemini_execution: str = "sync"
    gemini_batch_sync_fallback_max: int = 0

    @property
    def judge_contenders(self) -> tuple[GenerationTarget, ...]:
        return tuple(
            target for target in self.generation_targets if target.judge_contender
        )

    @property
    def default_judge_model(self) -> str:
        """First panel judge, retained for callers selecting one comparison judge."""

        return self.default_judge_models[0]

    @property
    def initial_judge_model(self) -> str:
        """Backward-compatible name for the default judge."""

        return self.default_judge_model


def _repo_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _default_judge_models(payload: dict) -> tuple[str, ...]:
    fields = ("default_judge_models", "default_judge_model", "initial_judge_model")
    present = [field for field in fields if field in payload]
    if len(present) > 1:
        raise ValueError(f"Specify only one judge selection field: {', '.join(present)}")
    field = present[0] if present else "default_judge_model"
    value = payload.get(field, "gpt-5-mini")
    # Legacy manifests may still use the singular key (scalar or list).
    if isinstance(value, str) and field != "default_judge_models":
        value = [value]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list of model IDs")
    if any(not isinstance(model, str) or not model.strip() for model in value):
        raise ValueError(f"{field} must contain non-empty model ID strings")
    return tuple(model.strip() for model in value)


def load_benchmark_config(path: Path, repo_root: Path) -> BenchmarkConfig:
    if not path.is_absolute():
        path = repo_root / path
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    eval_version = str(payload.get("eval_version", "v2"))
    system_prompt_label = str(payload["system_prompt_label"])
    model_items = payload.get("models")
    if model_items is None:
        model_items = payload.get("generation_targets", [])
    legacy_cross_judge = str(payload.get("cross_judge_model", ""))
    generation_targets_list: list[GenerationTarget] = []
    for item in model_items:
        provider = str(item["provider"])
        gen_model = str(item.get("model", item.get("gen_model", "")))
        answers_label = str(
            item.get("answers_label")
            or f"{provider}-{gen_model}-{system_prompt_label}"
        )
        judge_contender = bool(item.get("judge_contender", False))
        if legacy_cross_judge and gen_model == legacy_cross_judge:
            judge_contender = True
        generation_targets_list.append(
            GenerationTarget(
                provider=provider,
                gen_model=gen_model,
                answers_label=answers_label,
                judge_contender=judge_contender,
                judge_model=str(item.get("judge_model", "")),
            )
        )
    generation_targets = tuple(generation_targets_list)

    leader_items = payload.get("provider_leaders")
    if leader_items is None:
        leader_items = payload.get("control_targets", [])
    control_targets = tuple(
        ControlTarget(
            provider=str(item.get("provider", "")),
            gen_model=str(item.get("model", item.get("gen_model", ""))),
            answers_label=str(item["answers_label"]),
            dataset=_repo_path(item["dataset"], repo_root),
        )
        for item in leader_items
    )
    config = BenchmarkConfig(
        path=path,
        language=str(payload.get("language", "english")),
        eval_version=eval_version,
        system_prompt_label=system_prompt_label,
        use_system_prompt=bool(payload.get("use_system_prompt", True)),
        default_judge_models=_default_judge_models(payload),
        max_parallel_operations=int(payload.get("max_parallel_operations", 4)),
        limit=int(payload.get("limit", 0)),
        timezone=str(payload.get("timezone", "America/Chicago")),
        question_tags=_repo_path(
            str(payload.get("question_tags", "data/english/en_question_tags.json")),
            repo_root,
        ),
        master_csv=_repo_path(
            str(payload.get("master_csv", "data/english/api_evals_master.csv")),
            repo_root,
        ),
        api_evals_dir=_repo_path(
            str(payload.get("api_evals_dir", "data/english/api_evals")),
            repo_root,
        ),
        result_registry=_repo_path(
            str(
                payload.get(
                    "result_registry",
                    "benchmark_configs/english_api_registry.json",
                )
            ),
            repo_root,
        ),
        generation_targets=generation_targets,
        control_targets=control_targets,
        gemini_execution=str(payload.get("gemini_execution", "sync")),
        gemini_batch_sync_fallback_max=payload.get("gemini_batch_sync_fallback_max", 0),
    )
    validate_benchmark_config(config)
    return config


def validate_benchmark_config(config: BenchmarkConfig) -> None:
    if type(config.gemini_batch_sync_fallback_max) is not int or config.gemini_batch_sync_fallback_max < 0:
        raise ValueError("gemini_batch_sync_fallback_max must be a non-negative integer")
    if config.gemini_execution not in ("sync", "batch"):
        raise ValueError("gemini_execution must be sync or batch")
    if not config.default_judge_models or any(
        not isinstance(model, str) or not model.strip()
        for model in config.default_judge_models
    ):
        raise ValueError("default_judge_models must contain non-empty model IDs")
    if len(config.default_judge_models) != len(set(config.default_judge_models)):
        raise ValueError("Default judge model IDs must be unique")
    if config.max_parallel_operations < 1:
        raise ValueError("max_parallel_operations must be at least 1")
    if not config.generation_targets:
        raise ValueError("Benchmark models list cannot be empty")
    for target in config.generation_targets:
        if not target.provider or not target.gen_model or not target.answers_label:
            raise ValueError("Every benchmark model needs provider, model, and label")
    labels = [target.answers_label for target in config.generation_targets]
    if len(labels) != len(set(labels)):
        raise ValueError("Generation target answers_label values must be unique")
    control_labels = [target.answers_label for target in config.control_targets]
    if len(control_labels) != len(set(control_labels)):
        raise ValueError("Control target answers_label values must be unique")
    contender_judges = [
        target.contender_judge_model for target in config.judge_contenders
    ]
    if len(contender_judges) != len(set(contender_judges)):
        raise ValueError("Judge contender model IDs must be unique")
    if config.limit < 0:
        raise ValueError("Benchmark limit cannot be negative")


def sanitize_filename(name: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def generated_dataset_path(config: BenchmarkConfig, target: GenerationTarget) -> Path:
    return config.api_evals_dir / (
        f"generated_api_{target.provider}_{sanitize_filename(target.answers_label)}.jsonl"
    )


def results_path(config: BenchmarkConfig, answers_label: str, judge_model: str) -> Path:
    return config.api_evals_dir / (
        f"eval_results_{sanitize_filename(answers_label)}"
        f"__judged_by_{sanitize_filename(judge_model)}.jsonl"
    )


def expected_dynamic_sources(config: BenchmarkConfig) -> list[ResultSource]:
    """Return panel and contender result files in deterministic order."""

    paths: list[Path] = []
    seen_pairs: set[tuple[str, str]] = set()
    for target in config.generation_targets:
        for judge_model in config.default_judge_models:
            pair = (target.answers_label, judge_model)
            seen_pairs.add(pair)
            paths.append(results_path(config, *pair))

    contender_answer_sets = [
        (target.answers_label, generated_dataset_path(config, target))
        for target in config.generation_targets
    ]
    seen_labels = {label for label, _ in contender_answer_sets}
    contender_answer_sets.extend(
        (target.answers_label, target.dataset)
        for target in config.control_targets
        if target.answers_label not in seen_labels
    )
    for contender in config.judge_contenders:
        judge_model = contender.contender_judge_model
        for answers_label, _ in contender_answer_sets:
            pair = (answers_label, judge_model)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            paths.append(results_path(config, *pair))
    return [ResultSource(path, config.eval_version) for path in paths]


__all__ = [
    "BenchmarkConfig",
    "ControlTarget",
    "GenerationTarget",
    "expected_dynamic_sources",
    "generated_dataset_path",
    "load_benchmark_config",
    "results_path",
    "sanitize_filename",
    "validate_benchmark_config",
]
