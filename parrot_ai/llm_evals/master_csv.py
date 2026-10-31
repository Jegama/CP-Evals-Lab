"""Build the vertical master CSV consumed by the public evaluation dashboard."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from .aggregation import aggregate_score_statistics, score_metric_order


MASTER_METADATA_COLUMNS = [
    "run_id",
    "answers_label",
    "provider",
    "gen_model",
    "system_prompt_label",
    "judge_model",
    "eval_version",
    "evaluated_at",
    "timestamp_source",
    "question_count",
    "error_count",
    "source_dataset",
    "source_results",
]


@dataclass(frozen=True)
class ResultSource:
    """One raw evaluation result file selected for the master CSV."""

    path: Path
    eval_version: str = "v2"


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if isinstance(item, dict):
                records.append(item)
    return records


def load_question_tags(path: Optional[Path]) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        item["question"]: item
        for item in payload.get("tags", [])
        if isinstance(item, dict) and isinstance(item.get("question"), str)
    }


def metric_column_prefix(section: str, subcriterion: str) -> str:
    parts = [part for part in (section, subcriterion) if part]
    return "_".join(parts).lower()


def master_fieldnames(include_arabic: bool = False) -> list[str]:
    fields = list(MASTER_METADATA_COLUMNS)
    for section, subcriterion in score_metric_order(include_arabic):
        prefix = metric_column_prefix(section, subcriterion)
        fields.extend([f"{prefix}_mean", f"{prefix}_stdev"])
    return fields


def _last_metadata_value(records: Sequence[dict], key: str) -> object:
    for record in reversed(records):
        value = record.get(key)
        if value not in (None, ""):
            return value
    return ""


def _resolve_repo_path(value: object, repo_root: Path) -> Optional[Path]:
    """Resolve dataset metadata, including paths saved in another checkout."""
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    repo_root = repo_root.resolve()
    if not path.is_absolute():
        return (repo_root / path).resolve()

    resolved = path.resolve()
    if resolved.is_relative_to(repo_root):
        return resolved

    # Historical benchmark records used .../CP-Evals-Lab/data/... paths. Prefer
    # the current checkout, even if the old checkout still exists. Require the
    # complete data-relative path and reject missing files or escaping symlinks.
    for index in range(len(path.parts) - 1):
        if path.parts[index:index + 2] == ("CP-Evals-Lab", "data"):
            candidate = repo_root.joinpath(*path.parts[index + 1:]).resolve()
            if candidate.is_relative_to(repo_root) and candidate.is_file():
                return candidate
    return resolved


def infer_dataset_metadata(dataset_path: Optional[Path]) -> dict[str, object]:
    if dataset_path is None or not dataset_path.exists():
        return {}
    metadata: dict[str, object] = {}
    for record in load_jsonl(dataset_path):
        for key in (
            "gen_model",
            "provider",
            "system_prompt_label",
            "use_system_prompt",
        ):
            value = record.get(key)
            if key not in metadata and value not in (None, ""):
                metadata[key] = value
        if "gen_model" in metadata and "provider" in metadata:
            break
    return metadata


def infer_answers_label(
    stored_label: object,
    dataset_path: Optional[Path],
    provider: object,
) -> str:
    """Identify the answer set without legacy judge-specific numeric suffixes."""

    if dataset_path is not None and isinstance(provider, str) and provider:
        stem = dataset_path.stem
        for kind in ("api", "ft"):
            prefix = f"generated_{kind}_{provider}_"
            if stem.startswith(prefix):
                return stem[len(prefix) :]
    return str(stored_label or "")


def _relative_display_path(path: Optional[Path], repo_root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _parse_timestamp(value: object, timezone_name: str) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
    return parsed


def _git_added_timestamp(
    results_path: Path, repo_root: Path, timezone_name: str
) -> Optional[datetime]:
    try:
        relative_path = results_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None
    completed = subprocess.run(
        [
            "git",
            "log",
            "--follow",
            "--diff-filter=A",
            "--format=%aI",
            "--",
            relative_path.as_posix(),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    timestamps = [
        parsed
        for line in completed.stdout.splitlines()
        if (parsed := _parse_timestamp(line.strip(), timezone_name)) is not None
    ]
    return min(timestamps) if timestamps else None


def infer_evaluated_at(
    records: Sequence[dict],
    results_path: Path,
    repo_root: Path,
    timezone_name: str,
) -> tuple[str, str]:
    embedded = [
        parsed
        for record in records
        if (parsed := _parse_timestamp(record.get("timestamp"), timezone_name))
        is not None
    ]
    if embedded:
        return max(embedded).isoformat(), "result_metadata"

    committed = _git_added_timestamp(results_path, repo_root, timezone_name)
    if committed is not None:
        return committed.isoformat(), "git_commit"

    modified = datetime.fromtimestamp(
        results_path.stat().st_mtime,
        tz=ZoneInfo(timezone_name),
    )
    return modified.isoformat(), "filesystem_mtime"


def _deduplicate_evaluations(records: Sequence[dict]) -> list[dict]:
    """Keep the latest valid evaluation for each question, preserving order."""

    by_question: dict[str, dict] = {}
    order: list[str] = []
    seen_questions: set[str] = set()
    for record in records:
        question = record.get("question")
        if not isinstance(question, str) or not question:
            continue
        if question not in seen_questions:
            seen_questions.add(question)
            order.append(question)
        if isinstance(record.get("evaluation"), dict):
            by_question[question] = record
    return [by_question[question] for question in order if question in by_question]


def build_master_row(
    source: ResultSource,
    *,
    repo_root: Path,
    question_tags: Optional[Mapping[str, Mapping[str, object]]] = None,
    timezone_name: str = "America/Chicago",
    language: str = "english",
) -> dict[str, object]:
    results_path = source.path
    if not results_path.is_absolute():
        results_path = repo_root / results_path
    results_path = results_path.resolve()
    if not results_path.exists():
        raise FileNotFoundError(f"Evaluation results not found: {results_path}")

    all_records = load_jsonl(results_path)
    results = _deduplicate_evaluations(all_records)
    if not results:
        raise ValueError(f"No valid evaluations found in {results_path}")

    dataset_path = _resolve_repo_path(
        _last_metadata_value(all_records, "dataset"), repo_root
    )
    dataset_metadata = infer_dataset_metadata(dataset_path)

    provider = _last_metadata_value(all_records, "provider") or dataset_metadata.get(
        "provider", ""
    )
    gen_model = _last_metadata_value(all_records, "gen_model") or dataset_metadata.get(
        "gen_model", ""
    )
    system_prompt_label = _last_metadata_value(
        all_records, "system_prompt_label"
    ) or dataset_metadata.get("system_prompt_label", "")
    judge_model = str(_last_metadata_value(all_records, "judge_model") or "")
    answers_label = infer_answers_label(
        _last_metadata_value(all_records, "answers_label"),
        dataset_path,
        provider,
    )
    evaluated_at, timestamp_source = infer_evaluated_at(
        all_records,
        results_path,
        repo_root,
        timezone_name,
    )

    statistics = aggregate_score_statistics(
        results,
        include_arabic_accuracy=language == "arabic",
        question_tags=question_tags,
    )
    error_count = sum(1 for record in all_records if record.get("error"))
    run_identity = "|".join(
        [
            answers_label,
            judge_model,
            source.eval_version,
            evaluated_at,
            _relative_display_path(results_path, repo_root),
        ]
    )
    run_id = f"eval_{hashlib.sha256(run_identity.encode('utf-8')).hexdigest()[:16]}"

    row: dict[str, object] = {
        "run_id": run_id,
        "answers_label": answers_label,
        "provider": provider,
        "gen_model": gen_model,
        "system_prompt_label": system_prompt_label,
        "judge_model": judge_model,
        "eval_version": source.eval_version,
        "evaluated_at": evaluated_at,
        "timestamp_source": timestamp_source,
        "question_count": len(results),
        "error_count": error_count,
        "source_dataset": _relative_display_path(dataset_path, repo_root),
        "source_results": _relative_display_path(results_path, repo_root),
    }

    for section, subcriterion in score_metric_order(language == "arabic"):
        statistic = statistics.get((section, subcriterion))
        prefix = metric_column_prefix(section, subcriterion)
        row[f"{prefix}_mean"] = "" if statistic is None else statistic.mean
        row[f"{prefix}_stdev"] = "" if statistic is None else statistic.stdev
    return row


def write_master_csv(
    sources: Iterable[ResultSource],
    output_path: Path,
    *,
    repo_root: Path,
    question_tags_path: Optional[Path] = None,
    timezone_name: str = "America/Chicago",
    language: str = "english",
) -> list[dict[str, object]]:
    question_tags = load_question_tags(question_tags_path)
    rows = [
        build_master_row(
            source,
            repo_root=repo_root,
            question_tags=question_tags,
            timezone_name=timezone_name,
            language=language,
        )
        for source in sources
    ]
    run_ids = [str(row["run_id"]) for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Master CSV sources produced duplicate run IDs")

    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=master_fieldnames(language == "arabic"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


__all__ = [
    "MASTER_METADATA_COLUMNS",
    "ResultSource",
    "build_master_row",
    "infer_answers_label",
    "infer_dataset_metadata",
    "infer_evaluated_at",
    "load_jsonl",
    "load_question_tags",
    "master_fieldnames",
    "metric_column_prefix",
    "write_master_csv",
]
