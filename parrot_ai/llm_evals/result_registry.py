"""Persistent registry of evaluation result files included in the master CSV."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from .master_csv import ResultSource, load_jsonl
from .score_validation import is_scored_record


REGISTRY_SCHEMA_VERSION = 1


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else repo_root / path).resolve()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_result_registry(
    registry_path: Path,
    *,
    repo_root: Path,
) -> list[ResultSource]:
    if not registry_path.is_absolute():
        registry_path = repo_root / registry_path
    if not registry_path.exists():
        return []
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported result registry schema in {registry_path}: "
            f"{payload.get('schema_version')!r}"
        )
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Result registry has no results array: {registry_path}")

    sources: list[ResultSource] = []
    seen_paths: set[Path] = set()
    for index, item in enumerate(results):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(
                f"Invalid result registry entry {index} in {registry_path}"
            )
        path = _resolve_path(item["path"], repo_root)
        if path in seen_paths:
            raise ValueError(f"Duplicate result registry path: {path}")
        seen_paths.add(path)
        sources.append(
            ResultSource(
                path=path,
                eval_version=str(item.get("eval_version", "v2")),
            )
        )
    return sources


def validate_result_source(source: ResultSource) -> None:
    if not source.path.exists():
        raise FileNotFoundError(f"Completed evaluation result not found: {source.path}")
    records = load_jsonl(source.path)
    if not any(is_scored_record(record) for record in records):
        raise ValueError(
            f"Completed evaluation result has no valid score records: {source.path}"
        )


def append_result_sources(
    registry_path: Path,
    sources: Iterable[ResultSource],
    *,
    repo_root: Path,
) -> int:
    """Validate and append new result paths atomically in the supplied order."""

    if not registry_path.is_absolute():
        registry_path = repo_root / registry_path
    registry_path = registry_path.resolve()
    existing = load_result_registry(registry_path, repo_root=repo_root)
    existing_paths = {source.path.resolve() for source in existing}

    additions: list[ResultSource] = []
    for source in sources:
        resolved = ResultSource(
            path=source.path.resolve(),
            eval_version=source.eval_version,
        )
        if resolved.path in existing_paths:
            continue
        validate_result_source(resolved)
        existing_paths.add(resolved.path)
        additions.append(resolved)

    if not additions:
        return 0

    combined = [*existing, *additions]
    payload = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "results": [
            {
                "path": _display_path(source.path, repo_root),
                "eval_version": source.eval_version,
            }
            for source in combined
        ],
    }

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{registry_path.name}.",
        suffix=".tmp",
        dir=registry_path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, registry_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return len(additions)


__all__ = [
    "REGISTRY_SCHEMA_VERSION",
    "append_result_sources",
    "load_result_registry",
    "validate_result_source",
]
