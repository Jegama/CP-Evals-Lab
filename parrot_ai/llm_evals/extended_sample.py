"""Persist the question population selected for one extended evaluation."""

import json
import math
import os
import random
import tempfile
from collections.abc import Iterator
from pathlib import Path

from ..jsonl_checkpoints import recover_jsonl_tail
from .data_loading import load_jsonl


def extended_sample_path(results: Path) -> Path:
    return results.with_name(results.name + ".sample.json")


def default_extended_results(standard: Path) -> Path:
    """Separate new samples from canonical runs while resuming older artifacts."""
    from ..gemini_batch import state_directory

    extended = standard.with_stem(standard.stem + "__extended")
    if extended.exists() or extended_sample_path(extended).exists():
        return extended
    if extended_sample_path(standard).exists():
        return standard
    if standard.exists():
        # The shared legacy name may belong to a running canonical evaluation.
        # Inspect complete rows without repairing or modifying its checkpoint.
        with standard.open("rb") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(row, dict) and row.get("mode") == "extended":
                    return standard
    else:
        # Older no-wait runs used the shared filename before their first review.
        for path in state_directory(standard).glob("*.job.json"):
            if json.loads(path.read_text(encoding="utf-8")).get("version", 1) == 1:
                return standard
    return extended


def load_extended_sample(results: Path) -> list[str] | None:
    path = extended_sample_path(results)
    if not path.exists():
        return None
    questions = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(questions, list) or not questions
            or not all(isinstance(q, str) and q.strip() for q in questions)
            or len(questions) != len(set(questions))):
        raise ValueError(f"Invalid extended question sample: {path}")
    return questions


def _legacy_batch_questions(results: Path, answers: dict[str, str], engine) -> Iterator[str]:
    """Recover the selection of a pre-checkpoint Gemini job, even before its first poll."""
    from ..gemini_batch import state_directory

    by_content = None
    for path in sorted(state_directory(results).glob("*.job.json")):
        job = json.loads(path.read_text(encoding="utf-8"))
        if job["model"] != engine.model:
            continue
        if by_content is None:
            by_content = {engine.evaluation_content(q, a): q for q, a in answers.items() if a}
        for row in load_jsonl(path.with_suffix(".requests.jsonl")):
            text = row["request"]["contents"][0]["parts"][0]["text"]
            question = by_content.get(text.removeprefix(engine.instructions + "\n\n"))
            if question is not None:
                yield question


def _initial_sample(results: Path, answers: dict[str, str], engine) -> list[str]:
    size = min(len(answers), max(500, math.ceil(len(answers) / 10)))
    selected: dict[str, None] = {}
    if results.exists():
        recover_jsonl_tail(results)
        for row in load_jsonl(results):
            question = row.get("question")
            if row.get("mode") != "extended" or not isinstance(question, str) or question not in answers:
                continue
            old_size = row.get("extended_sample_size")
            if not selected and type(old_size) is int and old_size > 0:
                size = min(old_size, len(answers))
            selected[question] = None
            if len(selected) == size:
                return list(selected)
    for question in _legacy_batch_questions(results, answers, engine):
        selected[question] = None
        if len(selected) == size:
            return list(selected)
    population = [q for q in answers if q not in selected]
    return [*selected, *random.sample(population, size - len(selected))]


def select_extended_questions(results: Path, answers: dict[str, str], engine) -> list[str]:
    questions = load_extended_sample(results)
    if questions is None:
        questions = _initial_sample(results, answers, engine)
        # Publish a complete sample before any request. Concurrent starters
        # read the winner's selection and a torn temporary file is never read.
        path = extended_sample_path(results)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(questions, stream, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(name, path)
            except FileExistsError:
                questions = load_extended_sample(results)
                if questions is None:
                    raise RuntimeError(f"Extended sample disappeared during selection: {path}")
        finally:
            Path(name).unlink(missing_ok=True)
    missing = set(questions).difference(answers)
    if missing:
        raise ValueError(f"{len(missing)} saved extended-sample questions are missing from the dataset for {results}")
    return questions
